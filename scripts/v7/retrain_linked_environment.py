"""YR-332-aq: retrain the crane model in the calibrated linked-visit environment (user request).

The learning method is the ag method unchanged: three initial seeds, six updates over loads
3,500/7,500/15,000 twice, up to 24 counterfactual pairs per update, 50 optimizer steps,
3-hour cost horizon, frozen initial behavior policy, reachable alternatives, pairwise loss.
Only the environment changes: aj demand marginals + ak observed visit minutes + al visit links
(+ an alias fix). Prereg: docs/research/v7-discharge/linked-retraining-prereg.md
"""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from six_step_learning import checkpoint, gz_read, gz_save, labelset, load_net, sha
from benchmark_default_eval import save
from completion_learning import LOADS, RULES, RUNTIME, PROFILE, SEEDS
from completion_learning import folder as previous_folder

OUT = Path('outputs/reports/yr332_linked_retraining')
PREREG = Path('docs/research/v7-discharge/linked-retraining-prereg.md')
DEMAND = Path('outputs/reports/yr332_visit_links/demand-profiles.json')
VARIANT = 'observed_visit_links'
ARMS = ('initial', 'trained', 'previous')
CASES = [(rule, 0, load, day, 0) for rule in RULES for load in LOADS for day in range(3)]
CASES += [(arm, seed, load, day, 0) for arm in ARMS for seed in SEEDS for load in LOADS for day in range(3)]
CASES += [('SF_SPT', 0, 7500, 0, 1), ('trained', SEEDS[0], 7500, 0, 1)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def profile():
    return read(PROFILE)


def demand():
    return read(DEMAND)['profiles'][VARIANT]


def folder(seed):
    return OUT / f'model-{seed}'


def train_seed(seed, it):
    return 9923200 + 100 * SEEDS.index(seed) + 10 * it


def eval_seed(load, day):
    return 9923600 + 100 * day + 10 * LOADS.index(load)


def model_path(arm, seed):
    if not seed:
        return None
    if arm == 'previous':
        return previous_folder(seed) / 'final-5.pt'
    if arm == 'initial':
        # Same weights as this campaign's initial.pt (train() asserts equality); the saved ag
        # file exists before training starts, so fixed-policy runs can overlap with training.
        return previous_folder(seed) / 'initial.pt'
    return folder(seed) / 'final-5.pt'


def name(case):
    return '-'.join(map(str, case))


def sources():
    paths = [Path(__file__), PREREG, PROFILE, DEMAND, Path('outputs/reports/yr332_visit_links/link-definition.json'),
             Path('outputs/reports/yr332_visit_arrivals/visit-arrival-templates.json'), *Path('src/yard_rl/v7').rglob('*.py')]
    paths += [Path('scripts/v7') / p for p in ('completion_learning.py', 'six_step_learning.py', 'horizon_metrics.py',
        'audit_admission_ledger.py', 'verify_job_completion.py', 'benchmark_default_eval.py', 'evaluate_observed_demand.py')]
    paths += [previous_folder(s) / f for s in SEEDS for f in ('initial.pt', 'final-5.pt')]
    return {p.as_posix(): sha(p) for p in paths}


def check():
    for p, h in read(OUT / 'campaign.json')['sources'].items():
        assert sha(p) == h, ('campaign source changed', p)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no']).strip()
    assert not (OUT / 'campaign.json').exists()
    config = dict(model_seeds=SEEDS, loads=LOADS, updates=6, pairs_per_update=24, optimizer_steps_per_update=50,
        horizon_s=10800, behavior='frozen initial policy', loss='pairwise', sampling='coverage',
        alternative_contract='reachable', rule_dispatchers=RULES, arms=ARMS, cases=CASES, runtime=RUNTIME,
        demand_variant=VARIANT, demand_sha256=sha(DEMAND), profile_sha256=sha(PROFILE),
        training_seed_formula='9923200 + 100*model_index + 10*iteration',
        evaluation_seed_formula='9923600 + 100*day + 10*load_index', primary_seconds=86400, engine_seconds=93600,
        previous_models='ag final-5.pt (trained without links on uncalibrated inputs), evaluated unchanged',
        claim_scope='user-requested diagnostic retraining in the calibrated synthetic environment; no adoption',
        baseline='SF_SPT', decision='deterministic argmax through unchanged feasible resolver', execution_workers=14)
    save(OUT / 'campaign.json', dict(configuration=config, sources=sources(),
        repro=repro_stamp(experiment='YR-332-aq', seeds=dict(models=list(SEEDS),
            training=[train_seed(s, i) for s in SEEDS for i in range(6)],
            evaluation=[eval_seed(l, d) for d in range(3) for l in LOADS]), params=config, prereg=PREREG.as_posix())))


def train(seed):
    import torch
    from yard_rl.v7.crane.policy import CraneNet, add_store_input
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1);check();torch.manual_seed(seed)
    behavior = add_store_input(CraneNet(feature_mode='feasible'))
    old = load_net(previous_folder(seed) / 'initial.pt').state_dict()
    assert all(torch.equal(v, old[k]) for k, v in behavior.state_dict().items()), 'initial weights differ from ag'
    net = copy.deepcopy(behavior);trainer = CraneTrainer(net, loss_kind='pairwise')
    initial = folder(seed) / 'initial.pt'
    assert not initial.exists()
    checkpoint(initial, behavior, information_profile_sha256=sha(PROFILE), model_seed=seed, demand_sha256=sha(DEMAND))
    rows = []
    for it, load in enumerate(LOADS * 2):
        start = time.perf_counter();input_seed = train_seed(seed, it)
        ep = run_episode(load=load, seed=input_seed, dispatcher='RL_CRANE', crane_net=behavior,
            information_profile=profile(), demand_profile=demand(),
            crane_budget=RolloutBudget(max_labels=24, identity_checks=0), horizon_s=10800., workers=2,
            crane_sampling='coverage', crane_alternative_contract='reachable', **RUNTIME)
        data = dict(model_seed=seed, iteration=it, input_seed=input_seed, load=load, result=ep.as_dict(),
                    samples=[asdict(s) for s in ep.crane_labels], seconds=time.perf_counter() - start)
        path = folder(seed) / f'labels-{it}.json.gz';gz_save(path, data)
        c = ep.crane_stats
        assert ep.policy_exceptions == c['hook_errors'] == c['factual_mismatch'] == c['force_failed'] == 0
        assert ep.admitted == load and len(ep.crane_labels) >= 4
        assert c['sampled'] == c['no_witness'] + c['labeled']
        fit = trainer.fit(labelset(data), seed=input_seed);assert fit.steps == 50
        cp = folder(seed) / f'final-{it}.pt'
        checkpoint(cp, net, trainer, model_seed=seed, iteration=it, corpus_sha256=sha(path),
                   information_profile_sha256=sha(PROFILE), demand_sha256=sha(DEMAND))
        rows.append(dict(iteration=it, load=load, input_seed=input_seed, corpus=path.as_posix(), corpus_sha256=sha(path),
            checkpoint=cp.as_posix(), checkpoint_sha256=sha(cp), fit=asdict(fit), stats=c,
            seconds=time.perf_counter() - start))
        save(folder(seed) / 'training.json', dict(rows=rows, model_config=net.model_config(), torch=torch.__version__))
        print(f'model={seed} update={it + 1}/6 pairs={c["labeled"]} seconds={rows[-1]["seconds"]:.1f}', flush=True)


def evaluate(index):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    torch.set_num_threads(1)
    check()
    arm, seed, load, day, repeat = CASES[index]
    input_seed = eval_seed(load, day)
    cp = model_path(arm, seed)
    original, ann = ledger.install_announcer()

    class Probe(FlowProbe):
        def finish(self, mbt, records, episode_):
            self.visit_state = dict(stats=dict(mbt.visit_stats), plans={vid: dict(stops=s['stops'], travel=s['travel'],
                exit=s['exit'], gate_in=s['gate_in'], gate_out=s['gate_out']) for vid, s in mbt.visits.items()})
            return super().finish(mbt, records, episode_)
    probe = Probe(ann)
    start = time.perf_counter()
    try:
        ep = episode.run_episode(load=load, seed=input_seed, dispatcher='RL_CRANE' if seed else arm,
            crane_net=load_net(cp) if cp else None, information_profile=profile(), demand_profile=demand(),
            evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = ep.as_dict();metrics = raw_metrics(probe.rows);assert_metrics(metrics, result)
    flow_rows = copy.deepcopy(probe.flow_rows)
    for r in flow_rows:
        if r['flow'].startswith('GATE_'):
            r['finish_s'] = r['yard_done_s']
    completion = {f'at{h}': aggregate(flow_rows, h * 3600.) for h in (24, 26)}
    data = dict(case=CASES[index], input_seed=input_seed, demand_sha256=sha(DEMAND), result=result, probe=probe.result,
        metrics_by_flow=metrics, completion_by_flow=completion, orders_sha256=digest(probe.rows),
        schedule_sha256=digest(probe.schedule), visits=probe.visit_state,
        checkpoint_sha256=sha(cp) if cp else None, seconds=time.perf_counter() - start)
    assert ep.admitted == load and ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert probe.result['ledger_audit']['skipped'] == 0
    assert all(r['review_count'] == 1 for r in probe.rows)
    assert all(abs(r['attempt']['review_s'] - r['notice_s']) <= 1e-6 for r in probe.rows)
    assert probe.visit_state['stats']['late_releases'] == probe.visit_state['stats']['late_exits'] == 0
    for sub, value in [('evaluation', data), ('orders', probe.rows), ('flow-orders', flow_rows)]:
        path = OUT / sub / (name(CASES[index]) + '.json.gz');assert not path.exists();gz_save(path, value)
    print(f'eval {index} {name(CASES[index])}: work={result["mean_gate_to_job_done_s"] / 60:.3f}min {data["seconds"]:.1f}s',
          flush=True)


def run_many(mode, workers):
    check()
    if mode == 'train':
        values = SEEDS
    else:
        need_trained = mode == 'evaluate-trained'
        values = [i for i, c in enumerate(CASES) if (c[0] == 'trained') == need_trained]
        values.sort(key=lambda i: (-CASES[i][2], i))

    def run(value):
        cmd = 'train' if mode == 'train' else 'evaluate'
        if cmd == 'evaluate':
            existing = [(OUT / sub / (name(CASES[value]) + '.json.gz')).exists() for sub in ('evaluation', 'orders', 'flow-orders')]
            if all(existing):
                return f'{value} already complete'
            if any(existing):
                raise RuntimeError(f'Partial artifacts for {value}; inspect, do not overwrite')
        log = OUT / 'logs' / f'{cmd}-{value}.log';log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with log.open('w', encoding='utf-8') as handle:
            p = subprocess.run([sys.executable, __file__, cmd, str(value)], stdout=handle, stderr=subprocess.STDOUT, env=env)
        if p.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8'))
        return f'{cmd} {value} done'
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job in as_completed([pool.submit(run, v) for v in values]):
            done += 1
            print(f'{mode} {done}/{len(values)} {job.result()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'train', 'evaluate', 'train-all', 'evaluate-fixed', 'evaluate-trained'))
    parser.add_argument('value', nargs='?', type=int)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train(args.value)
    elif args.command == 'evaluate':
        evaluate(args.value)
    elif args.command == 'train-all':
        run_many('train', 3)
    elif args.command == 'evaluate-fixed':
        run_many('evaluate-fixed', args.workers)
    else:
        run_many('evaluate-trained', args.workers)
