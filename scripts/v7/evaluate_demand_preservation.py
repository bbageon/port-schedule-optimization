"""Frozen-policy, paired admission correction; no training or price changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pathlib import Path
import random
import subprocess
import sys
import time

import audit_admission_ledger as prior
from benchmark_default_eval import save
from six_step_learning import gz_read, gz_save, load_net, sha
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_demand_preservation')
PREREG = Path('docs/research/v7-discharge/demand-preservation-prereg.md')
BASE = [c for c in prior.CASES if c[0] != 'linear' and c[3] == 0]
CASES = [(mode, *case) for case in BASE for mode in ('legacy', 'preserved')]
CASES.append(('preserved', 'flow', 9913501, 15000, 1))
WORKERS = 6


def name(case):
    return case[0] + '-' + prior.name(case[1:])


def evaluate(index, smoke=False):
    import torch
    import yard_rl.v7.stage.episode as episode
    from yard_rl.v7.crane.policy import CraneNet
    torch.set_num_threads(1)
    case = CASES[index]
    mode, arm, model_seed, load, repeat = case
    old_case = tuple(case[1:])
    cp = prior.checkpoint(old_case)
    net = load_net(cp) if cp else None
    default = arm.startswith('default_')
    if default:
        torch.manual_seed(model_seed + 2)
        net = CraneNet().eval()
    seed = model_seed if default else 9913900 + 10 * prior.LOADS.index(load)
    if smoke:
        seed, load = 9914509, 80
    original, ann = prior.install_announcer()
    probe = prior.LedgerProbe(ann)
    started = time.perf_counter()
    try:
        result = episode.run_episode(load=load, seed=seed,
            dispatcher=prior.classical.RULES.get(arm, 'RL_CRANE'), crane_net=net,
            evaluation_probe=probe, candidate_cap=12 if default else None,
            evaluation_mode=arm[8:] if default else 'optimized',
            decision_probe_shortcircuit=not default,
            preserve_external_demand=mode == 'preserved')
    finally:
        episode.V3Announcer = original
    assert result.policy_exceptions == result.rollout_worlds == result.traded_edges == 0
    if mode == 'preserved':
        assert result.admitted == load
        assert probe.result['ledger_audit']['skipped'] == 0
        for row in probe.rows:
            assert row['times']['gate_in_s'] == row['scheduled_gate_s']
            assert row['times']['block_in_s'] == row['scheduled_gate_s'] + row['travel_s']
    out = dict(case=case, input_seed=seed, wall_seconds=time.perf_counter() - started,
        result=result.as_dict(), probe=probe.result,
        checkpoint=cp.as_posix() if cp else None, checkpoint_sha256=sha(cp) if cp else None)
    if not smoke:
        reference = prior.path_for(old_case if not repeat else (*old_case[:3], 0))
        old = gz_read(reference)
        assert out['probe']['metadata'] == old['probe']['metadata']
        if mode == 'legacy':
            normalized = copy.deepcopy(out['result'])
            assert normalized['environment'].pop('preserve_external_demand') is False
            assert normalized == old['result']
            assert digest(normalized) == old['probe']['result_sha256']
            for key in prior.HASHES[:-1]:
                assert out['probe'][key] == old['probe'][key], (case, key)
            assert probe.rows == gz_read(prior.OUT / 'orders' / reference.name)
        out.update(prior_reference=reference.as_posix(), prior_sha256=sha(reference),
                   legacy_exact_replay=mode == 'legacy')
    filename = ('smoke-' + mode if smoke else name(case)) + '.json.gz'
    gz_save(OUT / 'evaluation' / filename, out)
    gz_save(OUT / 'orders' / filename, probe.rows)
    print(f'{filename}: admitted={result.admitted}, cost={result.phi_krw:.3f}', flush=True)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not prior.classical.git('status', '--short', '--untracked-files=no')
    sources = [Path(__file__), Path('scripts/v7/report_demand_preservation.py'), PREREG,
        *[Path(p) for p in prior.source_hashes()],
        Path('src/yard_rl/integrated/repro.py')]
    inputs = {prior.path_for(c) for c in BASE} | {
        prior.OUT / 'orders' / prior.path_for(c).name for c in BASE} | {
        prior.checkpoint(c) for c in BASE if prior.checkpoint(c)}
    config = dict(cases=CASES, workers=WORKERS, torch_threads=1, training=False,
        official_end_s=86400, engine_end_s=93600, policy_changed=False,
        price_changed=False, physical_capacity_changed=False,
        single_change='fixed arrivals survive reservation-based admission rejection')
    save(OUT / 'campaign.json', dict(task='YR-332-aa', configuration=config,
        sources={p.as_posix():sha(p) for p in sources},
        input_sha256={p.as_posix():sha(p) for p in sorted(inputs)},
        repro=repro_stamp(experiment='YR-332-aa',
            seeds=dict(input=[9913900,9913910,9913920,9913320], model=list(prior.SEEDS)),
            params=config, prereg=PREREG.as_posix())))


def campaign():
    prepare()
    def one(index):
        log = OUT / 'logs' / f'{index}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', encoding='utf-8') as f:
            proc = subprocess.run([sys.executable, __file__, '--operation', 'one', '--index', str(index)],
                stdout=f, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if proc.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8')[-4000:])
        return index
    order = list(range(len(CASES)))
    random.Random(9914510).shuffle(order)
    done = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for future in as_completed([pool.submit(one, i) for i in order]):
            done.append(future.result())
            save(OUT / 'progress.json', dict(completed=done, total=len(CASES)))
            print(f'paired demand: {len(done)}/{len(CASES)}', flush=True)
    from report_demand_preservation import report
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('campaign','one','smoke'), default='campaign')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.operation == 'campaign':
        campaign()
    else:
        evaluate(args.index, smoke=args.operation == 'smoke')
