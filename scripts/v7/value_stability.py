"""Paired value-learning intervention. Old experiment sources remain immutable."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import time

from resched_ppo import encode_sample, decode_sample, sources as old_sources
from six_step_learning import gz_save, gz_read, sha
from value_stability_inputs import (OUT, PREREG, FIT, OLD_BANK, MIX, MINUTES, BASE, INFO,
    SEEDS, ARMS, SCALE, RUNTIME, read, save, demand, trajectories)


def folder(arm, seed):
    return OUT/f'{arm}-{seed}'


def prepare():
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
    assert not (OUT/'campaign.json').exists()
    assert read(OUT/'input-preflight.json')['passed']
    assert read(OUT/'smoke.json')['passed']
    from value_stability_eval import cases
    sources = old_sources()
    for p in [PREREG, FIT, OLD_BANK, MIX, MINUTES, BASE, INFO, OUT/'bank.json',
              Path('src/yard_rl/v7/crane/value_normalization.py'),
              *[Path('scripts/v7')/n for n in ('value_stability.py', 'value_stability_inputs.py',
                                              'value_stability_eval.py')]]:
        sources[p.as_posix()] = sha(p)
    cfg = dict(arms=ARMS, model_seeds=SEEDS, trajectories=trajectories(), evaluation_cases=cases(),
        updates=6, days_per_update=2, max_samples=2048, epochs=4, batch_size=64,
        lr=.0003, clip=.2, value_coefficient=.5, entropy_coefficient=.01, gradient_clip=1.,
        return_scale=SCALE, discount=1., observation_s=86400, engine_s=93600, runtime=RUNTIME,
        architecture=dict(kind='transformer', width=32, heads=4, layers=2),
        initialization='same fresh initialization and fresh Adam in each pair; not continuation',
        normalization='cumulative population moments of selected training returns, once/update; global',
        optimizer_contract='retain all Adam state unchanged when preserving critic head outputs',
        checkpoint_selection='final update 5 only; validation update 2 is diagnostic, no selection',
        evaluation='fixed argmax; separate stochastic validation for value prediction',
        authorization='explicit user request for bounded synthetic diagnostic; original gate rejection retained')
    save(OUT/'campaign.json', dict(configuration=cfg, sources=sources,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))


def check():
    obj = read(OUT/'campaign.json')
    for path, value in obj['sources'].items():
        assert sha(path) == value, ('Frozen source changed', path)
    return obj['configuration']


def checkpoint(path, net, optimizer, *, arm, seed, iteration):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_config=net.config, state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(), arm=arm, seed=seed, iteration=iteration,
        return_scale=SCALE, resume_boundary='between complete outer updates'), path)


def restore(path):
    import torch
    from yard_rl.v7.crane.value_normalization import NormalizedValueNet
    obj = torch.load(path, map_location='cpu', weights_only=False)
    net = NormalizedValueNet(**obj['model_config']).eval()
    net.load_state_dict(obj['state_dict'])
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
    optimizer.load_state_dict(obj['optimizer'])
    torch.set_rng_state(obj['torch_rng'])
    return net, optimizer, obj


def value_diagnostic(net, samples, metadata):
    import torch
    from yard_rl.v7.crane.actor_critic import pack_states
    from yard_rl.v7.crane.on_policy import action_statistics
    values, logps = [], []
    with torch.no_grad():
        for begin in range(0, len(samples), 64):
            batch = samples[begin:begin+64]
            ll, v = net(pack_states([s.state for s in batch]))
            lp, _ = action_statistics(ll, batch)
            values.extend(v.tolist()); logps.extend(lp.tolist())
    target = torch.tensor([s.target for s in samples], dtype=torch.float64)
    residual = target - torch.tensor(values, dtype=torch.float64)
    centered = (residual - residual.mean()) / residual.std(unbiased=False).clamp_min(1e-8)
    return dict(rmse=float(residual.square().mean().sqrt()), mean_residual=float(residual.mean()),
        raw_positive=int((residual > 0).sum()), centered_positive=int((centered > 0).sum()),
        values=values, action_logps=logps,
        by_load_time=[dict(load=load, hour_bin=hour, count=len(ix),
            rmse=float(residual[ix].square().mean().sqrt()), mean_residual=float(residual[ix].mean()),
            raw_positive=int((residual[ix]>0).sum()), centered_positive=int((centered[ix]>0).sum()))
            for load, hour in sorted({(m['load'], int(s.state.time_fraction*24)//6)
                                      for s, m in zip(samples, metadata)})
            if (ix := [i for i, (s, m) in enumerate(zip(samples, metadata))
                       if m['load']==load and int(s.state.time_fraction*24)//6==hour])])


def train(task, replay=False):
    import torch
    from yard_rl.v7.crane.value_normalization import NormalizedValueNet, fit
    from yard_rl.v7.crane.on_policy import OnPolicyResolver
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1)
    cfg = check(); arm, seed = task; root = folder(arm, seed)
    torch.manual_seed(seed)
    net = NormalizedValueNet(**cfg['architecture']).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    if not replay:
        assert not (root/'initial.pt').exists()
        checkpoint(root/'initial.pt', net, optimizer, arm=arm, seed=seed, iteration=-1)
    history = []; started = time.perf_counter()
    for it in range(cfg['updates']):
        begin = time.perf_counter(); rows = []
        if replay:
            # Restore the entire preceding checkpoint, including Adam state.
            net, optimizer, obj = restore(root/('initial.pt' if it==0 else f'update-{it-1}.pt'))
            assert obj['iteration'] == it-1
            corpus = gz_read(root/f'samples-{it}.json.gz')
            selected = [decode_sample(s) for s in corpus['samples']]
            metadata = corpus['metadata']
        else:
            samples = []; all_meta = []
            for case in cfg['trajectories'][2*it:2*it+2]:
                resolver = OnPolicyResolver(net, stochastic=True, collect=True, seed=seed+1000+case['index'])
                ep = run_episode(load=case['load'], seed=case['input_seed'], dispatcher='RL_CRANE',
                    crane_net=net, joint_resolver=resolver, information_profile=read(INFO),
                    demand_profile=demand(case['split'], case['index']), **RUNTIME)
                assert ep.admitted==case['load'] and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
                stats = resolver.finish(ep.phi_krw, SCALE)
                samples.extend(resolver.samples); all_meta.extend([case]*len(resolver.samples))
                rows.append(dict(case=case, result=ep.as_dict(), accounting=stats, cost_epochs=resolver.costs))
                print(f'train {arm} {seed} update={it} day={case["index"]} decisions={stats["samples"]}', flush=True)
            rng = torch.Generator().manual_seed(seed+2000+it)
            indices = torch.randperm(len(samples), generator=rng)[:cfg['max_samples']].tolist()
            selected = [samples[i] for i in indices]; metadata = [all_meta[i] for i in indices]
            gz_save(root/f'samples-{it}.json.gz', dict(source_decisions=len(samples), indices=indices,
                samples=[encode_sample(s) for s in selected], metadata=metadata))
        before = value_diagnostic(net, selected, metadata)
        optimization = fit(net, optimizer, selected, arm=arm, seed=seed+3000+it,
                           max_samples=cfg['max_samples'], epochs=cfg['epochs'], batch_size=cfg['batch_size'])
        after = value_diagnostic(net, selected, metadata)
        if replay:
            expected, expected_opt, _ = restore(root/f'update-{it}.pt')
            assert all(torch.equal(v, expected.state_dict()[k]) for k, v in net.state_dict().items())
            assert_nested_equal(optimizer.state_dict(), expected_opt.state_dict())
        else:
            checkpoint(root/f'update-{it}.pt', net, optimizer, arm=arm, seed=seed, iteration=it)
            gz_save(root/f'fit-diagnostic-{it}.json.gz', dict(before=before, after=after))
        history.append(dict(iteration=it, days=rows, optimization=optimization,
            samples_sha256=sha(root/f'samples-{it}.json.gz'), checkpoint_sha256=sha(root/f'update-{it}.pt'),
            seconds=time.perf_counter()-begin))
        if not replay:
            save(root/'training.json', dict(arm=arm, seed=seed, history=history, seconds=time.perf_counter()-started))
        print(f'fit {arm} {seed} update={it} seconds={history[-1]["seconds"]:.1f} rmse={before["rmse"]:.3f}->{after["rmse"]:.3f}', flush=True)
    if replay:
        save(root/'replay.json', dict(passed=True, updates=6, optimizer_exact=True,
            seconds=time.perf_counter()-started, scope='saved-experience full-checkpoint optimizer replay'))
    else:
        checkpoint(root/'final.pt', net, optimizer, arm=arm, seed=seed, iteration=5)


def assert_nested_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys()==b.keys()
        for k in a: assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a)==len(b)
        for x, y in zip(a, b): assert_nested_equal(x, y)
    else: assert a==b, (a, b)


def run(command, workers):
    from value_stability_eval import cases, evaluate
    tasks = [(arm, seed) for arm in ARMS for seed in SEEDS] if command in ('train', 'replay') else list(range(len(cases())))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(train, t, command=='replay') if command in ('train','replay') else pool.submit(evaluate,t) for t in tasks]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result(); print(f'{command}: {n}/{len(tasks)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','train','replay','evaluate','smoke','repeat'])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='smoke':
        from value_stability_eval import smoke
        smoke()
    elif args.command=='repeat':
        from value_stability_eval import cases, evaluate
        for arm in ARMS:
            i = next(i for i,c in enumerate(cases()) if c['arm']==arm and c['stage']=='final' and c['split']=='holdout')
            evaluate(i, repeat=True)
    else: run(args.command,args.workers)
