"""Controlled K=1,2,3,6 hour teacher training and fresh per-flow evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from benchmark_default_eval import save, memory
from six_step_learning import checkpoint, gz_read, gz_save, labelset, load_net, sha

OUT = Path('outputs/reports/yr323_horizon_sweep')
SOURCE = Path('outputs/reports/yr332_reachable_teacher_corrected')
PREREG = Path('docs/research/v7-discharge/horizon-sweep-prereg.md')
SEEDS = (9914701, 9914702, 9914703)
LOADS = (3500, 7500, 15000)
KS = (1, 2, 3, 6)
RUNTIME = dict(candidate_cap=None, reuse_rehandle_slots=True,
               decision_probe_shortcircuit=True, preserve_external_demand=True)
CASES = [(arm, seed, load, day, 0) for arm in ('initial', 'k1', 'k2', 'k3', 'k6')
         for seed in SEEDS for day in range(3) for load in LOADS]
CASES += [('sf_spt', 0, load, day, 0) for day in range(3) for load in LOADS]
CASES += [(f'k{k}', SEEDS[0], 15000, 0, 1) for k in (1, 2, 6)]


def name(case):
    return '-'.join(map(str, case))


def folder(k, seed):
    return OUT/f'k-{k}'/f'model-{seed}'


def model_path(arm, seed):
    if arm == 'sf_spt':
        return None
    if arm in ('initial', 'k3'):
        return SOURCE/f'model-{seed}'/('initial.pt' if arm == 'initial' else 'reachable-5.pt')
    return folder(int(arm[1:]), seed)/'final-5.pt'


def check_sources():
    m = json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    for path, expected in {**m['sources'], **m['reused_artifacts']}.items():
        assert sha(path) == expected, ('source changed during campaign', path)


def common_choices(data):
    """The teacher horizon may change costs, never behaviour or selected questions."""
    cs = data['result']['crane']
    result = dict(trajectory={k: data['result'][k] for k in
        ('phi_krw', 'admitted', 'n_censored', 'c_wait', 'c_vessel', 'c_move', 'c_rehandle')},
        choices=cs['main_choices_sha256'], searches=cs['witness_search'],
        samples=[{k: s[k] for k in ('decision_key', 'action', 'crane', 'job', 'row', 'flow', 'is_store')}
                 for s in data['samples']])
    return json.loads(json.dumps(result))


def train(k, seed):
    import torch
    from yard_rl.v7.crane.policy import CraneNet, add_store_input
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    assert k in (1, 2, 6)
    torch.set_num_threads(1)
    check_sources()
    torch.manual_seed(seed)
    behaviour = add_store_input(CraneNet(feature_mode='feasible'))
    initial = SOURCE/f'model-{seed}'/'initial.pt'
    assert all(torch.equal(v, load_net(initial).state_dict()[key])
               for key, v in behaviour.state_dict().items())
    net = copy.deepcopy(behaviour)
    trainer = CraneTrainer(net, loss_kind='pairwise')
    rows = []
    for it, load in enumerate(LOADS*2):
        start = time.perf_counter()
        input_seed = 9914800 + 100*SEEDS.index(seed) + 10*it
        ep = run_episode(load=load, seed=input_seed, dispatcher='RL_CRANE', crane_net=behaviour,
            crane_budget=RolloutBudget(max_labels=24, identity_checks=0), horizon_s=3600.*k,
            workers=3, crane_sampling='coverage', crane_alternative_contract='reachable', **RUNTIME)
        data = dict(k=k, model_seed=seed, iteration=it, load=load, input_seed=input_seed,
            contract='reachable', runtime=RUNTIME, behaviour_sha256=sha(initial),
            seconds=time.perf_counter()-start, memory=memory(),
            result=ep.as_dict(), samples=[asdict(s) for s in ep.crane_labels])
        path = folder(k, seed)/f'labels-{it}.json.gz'
        gz_save(path, data)  # Preserve failures before checking.
        cs = ep.crane_stats
        assert ep.policy_exceptions == cs['hook_errors'] == cs['factual_mismatch'] == cs['force_failed'] == 0
        assert cs['sampled'] == cs['no_witness']+cs['labeled']
        assert ep.admitted == load and len(ep.crane_labels) >= 4
        reference = gz_read(SOURCE/f'model-{seed}'/f'labels-reachable-{it}.json.gz')
        assert common_choices(data) == common_choices(reference), 'K changed behaviour/sample/questions'
        for c in cs['comparisons']:
            assert c['requested_end_s'] == c['t']+3600.*k
            assert c['cost_end_s'] == min(c['requested_end_s'], 93600.)
            assert c['effective_horizon_s'] == c['cost_end_s']-c['t'] > 0
        fit = trainer.fit(labelset(data), seed=input_seed)
        assert fit.steps == 50
        cp = folder(k, seed)/f'final-{it}.pt'
        checkpoint(cp, net, trainer, k=k, model_seed=seed, iteration=it, corpus_sha256=sha(path))
        rows.append(dict(iteration=it, load=load, input_seed=input_seed, corpus=path.as_posix(),
            corpus_sha256=sha(path), checkpoint=cp.as_posix(), checkpoint_sha256=sha(cp),
            fit=asdict(fit), stats=cs, seconds=time.perf_counter()-start))
        save(folder(k, seed)/'training.json', dict(k=k, model_seed=seed, rows=rows,
            behaviour='frozen initial model', torch=torch.__version__, python=sys.version,
            model_config=net.model_config(), runtime=RUNTIME))
        print(f'K={k} model={seed} update={it+1}/6: pairs={cs["labeled"]}, '
              f'{rows[-1]["seconds"]:.1f}s', flush=True)


def evaluate(index, smoke=False):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe
    from evaluate_arrival_order import HASHES
    torch.set_num_threads(1)
    if not smoke:
        check_sources()
    case = CASES[index]
    arm, seed, load, day, repeat = case
    input_seed = 9915900+100*day+10*LOADS.index(load)
    cp = model_path(arm, seed)
    if smoke:
        load, input_seed = 80, 9916299
    net = load_net(cp) if cp else None
    original, ann = ledger.install_announcer()
    probe = FlowProbe(ann)
    start = time.perf_counter()
    try:
        ep = episode.run_episode(load=load, seed=input_seed,
            dispatcher='SF_SPT' if arm == 'sf_spt' else 'RL_CRANE',
            crane_net=net, evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    data = dict(case=case, input_seed=input_seed, runtime=RUNTIME, result=ep.as_dict(),
        probe=probe.result, seconds=time.perf_counter()-start,
        checkpoint=cp.as_posix() if cp else None, checkpoint_sha256=sha(cp) if cp else None)
    dest = 'smoke' if smoke else name(case)
    gz_save(OUT/'evaluation'/f'{dest}.json.gz', data)
    gz_save(OUT/'orders'/f'{dest}.json.gz', probe.rows)
    gz_save(OUT/'flow-orders'/f'{dest}.json.gz', probe.flow_rows)
    assert ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert ep.admitted == load and probe.result['ledger_audit']['skipped'] == 0
    if arm in ('initial', 'k3', 'sf_spt') and not smoke:
        old_arm = 'reachable-5' if arm == 'k3' else arm
        old = gz_read(SOURCE/'evaluation'/f'{old_arm}-{seed}-{load}-{day}-0.json.gz')
        assert data['result'] == old['result']
        for key in HASHES:
            assert data['probe'][key] == old['probe'][key], (case, key)
        assert probe.rows == gz_read(SOURCE/'orders'/f'{old_arm}-{seed}-{load}-{day}-0.json.gz')
    print(f'eval {index} {dest}: truck={ep.breakdown["mean_turn_time_s"]/60:.3f}min, '
          f'{data["seconds"]:.1f}s', flush=True)


def smoke():
    import torch
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1)
    net = load_net(SOURCE/f'model-{SEEDS[0]}'/'initial.pt')
    previous = None
    for k in KS:
        ep = run_episode(load=300, seed=9916299, dispatcher='RL_CRANE', crane_net=net,
            crane_budget=RolloutBudget(max_labels=4, identity_checks=0), horizon_s=3600.*k,
            workers=1, crane_sampling='coverage', crane_alternative_contract='reachable', **RUNTIME)
        data = dict(k=k, result=ep.as_dict(), samples=[asdict(s) for s in ep.crane_labels])
        gz_save(OUT/'smoke'/f'k{k}.json.gz', data)
        c = ep.crane_stats
        assert ep.policy_exceptions == c['force_failed'] == c['factual_mismatch'] == c['hook_errors'] == 0
        assert c['labeled'] > 0
        for pair in c['comparisons']:
            assert pair['cost_end_s'] == min(pair['t']+3600.*k, 93600.)
        if previous is not None:
            assert common_choices(data) == common_choices(previous)
        previous = data
        print(f'smoke K={k}: {c["labeled"]} common pairs', flush=True)
    evaluate(0, smoke=True)
    save(OUT/'smoke'/'verification.json', dict(status='PASS', ks=KS,
        common_questions=True, chronology=True, truck_ledger_exact=True, horizon_bounds=True))


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    assert not git('status', '--short', '--untracked-files=no'), 'commit code first'
    previous = json.loads((SOURCE/'campaign.json').read_text(encoding='utf-8'))
    for p, h in previous['sources'].items():
        assert sha(p) == h, ('original source changed', p)
    sources = {**previous['sources'], **{p.as_posix(): sha(p) for p in
        (Path(__file__), Path('scripts/v7/horizon_metrics.py'), PREREG,
         Path('tests/v7/test_horizon_flow_metrics.py'))}}
    reused = {p.as_posix(): sha(p) for p in [SOURCE/'campaign.json', SOURCE/'verification.json']}
    for seed in SEEDS:
        for f in ('initial.pt', 'training.json',
                  *[f'{prefix}-{i}.{ext}' for i in range(6)
                    for prefix, ext in (('reachable', 'pt'), ('labels-reachable', 'json.gz'))]):
            p = SOURCE/f'model-{seed}'/f
            reused[p.as_posix()] = sha(p)
    config = dict(ks=KS, horizon_seconds=[3600*k for k in KS], model_seeds=SEEDS, loads=LOADS,
        updates=6, optimizer_steps=50, pairs_per_update=24, behaviour='frozen initial network',
        train_workers=3, branch_workers=3, eval_workers=4, torch_threads=1, runtime=RUNTIME,
        alternative_contract='reachable', final_checkpoint_iteration=5, evaluation=CASES,
        reuse_k3=True, claim_scope='explicit user-authorized synthetic horizon diagnostic',
        scenario_calibration='UNCONFIRMED', primary_observation_seconds=86400,
        secondary_observation_seconds=93600)
    manifest = dict(configuration=config, sources=sources, reused_artifacts=reused,
        repro=repro_stamp(experiment='YR-323', seeds=dict(model=list(SEEDS),
            training=[9914800+100*s+10*i for s in range(3) for i in range(6)],
            evaluation=[9915900+100*d+10*l for d in range(3) for l in range(3)]),
            params=config, prereg=PREREG.as_posix()))
    if (OUT/'campaign.json').exists():
        old = json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
        assert old['sources'] == sources and old['reused_artifacts'] == reused
    else:
        save(OUT/'campaign.json', manifest)


def campaign():
    prepare()
    def one(operation, index, k=None):
        label = f'{operation}-{index}' + (f'-k{k}' if k is not None else '')
        marker = OUT/'completed'/f'{label}.json'
        if marker.exists():
            return
        log = OUT/'logs'/f'{label}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, '--operation', operation,
               '--seed' if operation == 'train' else '--index', str(index)]
        if k is not None:
            cmd += ['--k', str(k)]
        with log.open('w', encoding='utf-8') as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if p.returncode:
            raise RuntimeError(f'{label}: '+log.read_text(encoding='utf-8')[-6000:])
        save(marker, dict(operation=operation, index=index, k=k, success=True))
    submitted, evaluations = set(), []
    def submit_evals(pool, indices):
        for i in indices:
            assert i not in submitted
            submitted.add(i)
            evaluations.append(pool.submit(one, 'evaluate', i))
    with ThreadPoolExecutor(max_workers=3) as train_pool, ThreadPoolExecutor(max_workers=4) as eval_pool:
        ready = [i for i, c in enumerate(CASES) if c[0] in ('initial', 'k3', 'sf_spt')]
        random.Random(9916298).shuffle(ready)
        submit_evals(eval_pool, ready)
        jobs = [(k, s) for k in (6, 1, 2) for s in SEEDS]
        running = {train_pool.submit(one, 'train', s, k): (k, s) for k, s in jobs}
        done = []
        for future in as_completed(running):
            future.result()
            k, s = running[future]
            done.append((k, s))
            submit_evals(eval_pool, [i for i, c in enumerate(CASES) if c[0] == f'k{k}' and c[1] == s])
            save(OUT/'progress.json', dict(training_done=done, evaluation_submitted=len(submitted)))
            print(f'training complete {len(done)}/9: K={k}, model={s}', flush=True)
        for n, future in enumerate(as_completed(evaluations), 1):
            future.result()
            save(OUT/'progress.json', dict(training_done=done, evaluation_done=n, total=len(CASES)))
    assert len(submitted) == len(CASES)
    print(f'Campaign complete: 54 new updates, 18 reused updates, {len(CASES)} fresh evaluations.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('campaign', 'train', 'evaluate', 'smoke'), default='campaign')
    p.add_argument('--k', type=int, choices=KS, default=1)
    p.add_argument('--seed', type=int, default=SEEDS[0])
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    if args.operation == 'campaign':
        campaign()
    elif args.operation == 'train':
        train(args.k, args.seed)
    elif args.operation == 'evaluate':
        evaluate(args.index)
    else:
        smoke()
