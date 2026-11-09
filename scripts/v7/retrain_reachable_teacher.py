"""Paired teacher-contract correction, common behaviour and held-out evaluation."""
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

OUT = Path('outputs/reports/yr332_reachable_teacher_corrected')
PREREG = Path('docs/research/v7-discharge/reachable-teacher-horizon-amendment.md')
SEEDS = (9914701, 9914702, 9914703)
LOADS = (3500, 7500, 15000)
CONTRACTS = ('legacy_force', 'reachable')
RULES = dict(fcfs_arrival='FCFS_ARRIVAL', spt='SPT', nearest='NEAREST',
    vessel_fcfs='VESSEL_FCFS', vessel_nearest='VESSEL_NEAREST',
    unidirectional='UNIDIRECTIONAL', distance_wait='DISTANCE_WAIT', sf_spt='SF_SPT')
ARMS = ('initial', 'legacy_force-2', 'legacy_force-5', 'reachable-2', 'reachable-5')
CASES = [(arm, seed, load, day, 0) for day in range(3) for load in LOADS
         for seed in SEEDS for arm in ARMS]
CASES += [(arm, 0, load, day, 0) for day in range(3) for load in LOADS for arm in RULES]
CASES += [('reachable-5', SEEDS[0], 15000, 0, 1)]
RUNTIME = dict(candidate_cap=None, reuse_rehandle_slots=True,
               decision_probe_shortcircuit=True, preserve_external_demand=True)


def name(case):
    return '-'.join(map(str, case))


def check_sources():
    manifest = json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    for path, expected in manifest['sources'].items():
        assert sha(path) == expected, ('source changed during run', path)


def train(model_seed, smoke=False):
    import torch
    from yard_rl.v7.crane.policy import CraneNet, add_store_input
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1)
    if not smoke:
        check_sources()
    torch.manual_seed(model_seed)
    behaviour = add_store_input(CraneNet(feature_mode='feasible'))
    folder = OUT/('smoke' if smoke else f'model-{model_seed}')
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint(folder/'initial.pt', behaviour, model_seed=model_seed)
    nets = {arm: copy.deepcopy(behaviour) for arm in CONTRACTS}
    trainers = {arm: CraneTrainer(net, loss_kind='pairwise') for arm, net in nets.items()}
    rows = []
    loads = (300,) if smoke else LOADS*2
    for it, load in enumerate(loads):
        seed = 9914699 if smoke else 9914800+100*SEEDS.index(model_seed)+10*it
        shared = None
        for contract in CONTRACTS:
            started = time.perf_counter()
            ep = run_episode(load=load, seed=seed, dispatcher='RL_CRANE', crane_net=behaviour,
                crane_budget=RolloutBudget(max_labels=4 if smoke else 24, identity_checks=0),
                horizon_s=1800. if smoke else 10800., workers=1 if smoke else 3,
                crane_sampling='coverage', crane_alternative_contract=contract, **RUNTIME)
            data = dict(model_seed=model_seed, iteration=it, load=load, input_seed=seed,
                contract=contract, runtime=RUNTIME, behaviour_sha256=sha(folder/'initial.pt'),
                seconds=time.perf_counter()-started, memory=memory(),
                result=ep.as_dict(), samples=[asdict(s) for s in ep.crane_labels])
            path = folder/f'labels-{contract}-{it}.json.gz'
            gz_save(path, data)  # Preserve failures before asserting.
            cs = ep.crane_stats
            assert ep.policy_exceptions == cs['hook_errors'] == cs['factual_mismatch'] == 0, cs
            if contract == 'reachable':
                assert cs['force_failed'] == 0, cs
                assert cs['sampled'] == cs['no_witness'] + cs['labeled'], cs
            assert ep.admitted == load and len(ep.crane_labels) >= 4, cs
            trajectory = {k: data['result'][k] for k in
                ('phi_krw','admitted','c_wait','c_vessel','c_move','c_rehandle','n_censored')}
            trajectory['choices'] = cs['main_choices_sha256']
            if shared is None:
                shared = trajectory
            else:
                assert shared == trajectory, 'teacher search changed the main behaviour'
            previous = Path('outputs/reports/yr332_reachable_teacher')/f'model-{model_seed}'/path.name
            if previous.exists() and not smoke:
                old = gz_read(previous)['result']
                assert old['crane']['main_choices_sha256'] == cs['main_choices_sha256']
                for key in ('phi_krw','admitted','c_wait','c_vessel','c_move','c_rehandle','n_censored'):
                    assert old[key] == data['result'][key], ('horizon correction changed main run', key)
            fit = trainers[contract].fit(labelset(data), seed=seed)
            assert fit.steps == 50, fit
            checkpoint(folder/f'{contract}-{it}.pt', nets[contract], trainers[contract],
                model_seed=model_seed, iteration=it, corpus_sha256=sha(path))
            rows.append(dict(iteration=it, contract=contract, input_seed=seed, load=load,
                corpus=path.as_posix(), corpus_sha256=sha(path), fit=asdict(fit),
                seconds=time.perf_counter()-started, stats=cs))
            save(folder/'training.json', dict(rows=rows, behaviour='frozen initial model',
                model_seed=model_seed, model_config=behaviour.model_config(), runtime=RUNTIME,
                torch=torch.__version__, python=sys.version, device='cpu'))
            print(f'model {model_seed} update {it+1}/{len(loads)} {contract}: '
                  f'{cs["labeled"]}/{cs["sampled"]} pairs, {rows[-1]["seconds"]:.1f}s', flush=True)
    if smoke:
        # Same corrected teacher through a real process boundary.
        ep2 = run_episode(load=300, seed=9914699, dispatcher='RL_CRANE', crane_net=behaviour,
            crane_budget=RolloutBudget(max_labels=4, identity_checks=0), horizon_s=1800.,
            workers=2, crane_sampling='coverage', crane_alternative_contract='reachable', **RUNTIME)
        reference = gz_read(folder/'labels-reachable-0.json.gz')
        assert json.loads(json.dumps(ep2.as_dict())) == reference['result']
        assert json.loads(json.dumps([asdict(s) for s in ep2.crane_labels])) == reference['samples']
        save(folder/'serial-parallel.json', dict(exact=True, labeled=ep2.crane_stats['labeled']))


def evaluate(index, smoke=False):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    torch.set_num_threads(1)
    if not smoke:
        check_sources()
    case = CASES[index]
    arm, model_seed, load, day, repeat = case
    seed = 9915900 + 100*day + 10*LOADS.index(load)
    cp = None if arm in RULES else OUT/f'model-{model_seed}'/(arm+'.pt')
    if smoke:
        load, seed, cp = 80, 9914699, OUT/'smoke/reachable-0.pt'
    net = load_net(cp) if cp else None
    original, ann = ledger.install_announcer()
    probe = ledger.LedgerProbe(ann)
    started = time.perf_counter()
    try:
        ep = episode.run_episode(load=load, seed=seed, dispatcher=RULES.get(arm, 'RL_CRANE'),
            crane_net=net, evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = dict(case=case, input_seed=seed, runtime=RUNTIME, result=ep.as_dict(),
        probe=probe.result, seconds=time.perf_counter()-started,
        checkpoint=cp.as_posix() if cp else None, checkpoint_sha256=sha(cp) if cp else None)
    filename = ('smoke' if smoke else name(case))+'.json.gz'
    gz_save(OUT/'evaluation'/filename, result)
    gz_save(OUT/'orders'/filename, probe.rows)
    assert ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert ep.admitted == load and probe.result['ledger_audit']['skipped'] == 0
    print(f'evaluation {index}: {arm}/{model_seed}/{load}/{day}, '
          f'turn={ep.breakdown["mean_turn_time_s"]/60:.3f}min', flush=True)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    git = lambda *a: subprocess.check_output(['git', *a], encoding='utf-8').strip()
    assert not git('status', '--short', '--untracked-files=no'), 'commit code first'
    sources = [Path(__file__), PREREG,
        Path('docs/research/v7-discharge/reachable-teacher-prereg.md'), *Path('src/yard_rl/v7').rglob('*.py'),
        *[Path('scripts/v7')/f for f in ('six_step_learning.py','audit_admission_ledger.py',
            'evaluate_classical_rules.py','evaluate_arrival_order.py','benchmark_default_eval.py')],
        Path('src/yard_rl/integrated/repro.py')]
    config = dict(model_seeds=SEEDS, loads=LOADS, updates=6, pairs_per_update=24,
        horizon_s=10800., optimizer_steps=50, behaviour='frozen initial model',
        train_concurrency=2, branch_workers=3, eval_workers=6, torch_threads=1,
        runtime=RUNTIME, contracts=CONTRACTS, evaluation=CASES, rules=RULES,
        final_checkpoint_iteration=5, claim_scope='synthetic diagnostic; calibration unresolved')
    data = dict(configuration=config, sources={p.as_posix():sha(p) for p in sources},
        repro=repro_stamp(experiment='YR-332-ab-c',
            seeds=dict(model=list(SEEDS),
                training=[9914800+100*s+10*i for s in range(3) for i in range(6)],
                evaluation=[9915900+100*d+10*l for d in range(3) for l in range(3)]),
            params=config, prereg=PREREG.as_posix()))
    path = OUT/'campaign.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        assert previous['sources'] == data['sources']
    else:
        save(path, data)


def campaign():
    prepare()
    def one(operation, index):
        marker = OUT/'completed'/f'{operation}-{index}.json'
        if marker.exists():
            return
        log = OUT/'logs'/f'{operation}-{index}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, '--operation', operation,
               '--model-seed' if operation == 'train' else '--index', str(index)]
        with log.open('w', encoding='utf-8') as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if p.returncode:
            raise RuntimeError(f'{operation}/{index}: '+log.read_text(encoding='utf-8')[-6000:])
        save(marker, dict(operation=operation, index=index, success=True))
    for operation, jobs, workers in (('train', list(SEEDS), 2),
                                    ('evaluate', list(range(len(CASES))), 6)):
        random.Random(9914698).shuffle(jobs)
        completed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(one, operation, i) for i in jobs]):
                future.result()
                completed.append(1)
                save(OUT/'progress.json', dict(phase=operation, done=len(completed), total=len(jobs)))
                print(f'{operation}: {len(completed)}/{len(jobs)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('campaign','train','evaluate','smoke'), default='campaign')
    p.add_argument('--model-seed', type=int, default=SEEDS[0])
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    if args.operation == 'campaign':
        campaign()
    elif args.operation == 'train':
        train(args.model_seed)
    elif args.operation == 'evaluate':
        evaluate(args.index)
    else:
        train(9914699, smoke=True)
        evaluate(0, smoke=True)
