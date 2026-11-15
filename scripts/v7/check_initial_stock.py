"""YR-332-at input campaign and ao frozen-rule sensitivity; see preregistration."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from audit_day_shapes import BASE, INFO, check_stage, read
from completion_learning import LOADS, RUNTIME
from six_step_learning import gz_save, sha
from yard_rl.integrated.repro import repro_stamp, write_result
from yard_rl.v7.day_shape import demand_for_shape
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_initial_stock')
SHAPES = Path('outputs/reports/yr332_day_shapes')
PREREG = Path('docs/research/v7-discharge/initial-stock-prereg.md')
BANK = SHAPES / 'bank-periodic_linear_24.json'
PREFLIGHT = SHAPES / 'scenario-preflight.json'


def sources():
    paths = [Path(__file__), PREREG, BANK, PREFLIGHT, BASE, INFO,
             *Path('src/yard_rl/v7').rglob('*.py'),
             *[Path('scripts/v7') / p for p in ('audit_day_shapes.py', 'completion_learning.py',
                'horizon_metrics.py', 'audit_admission_ledger.py', 'verify_job_completion.py',
                'six_step_learning.py', 'benchmark_default_eval.py')]]
    return {p.as_posix(): sha(p) for p in paths}


def cases():
    rows = [dict(kind='stock', seed=9934102, load=15000, shape='fixed', index=0,
                 rule=rule, mode='preserve') for rule in ('SF_SPT', 'FIFO')]
    rows += [dict(kind='stock', seed=9934100, load=3500, shape='fixed', index=0,
                  rule='SF_SPT', mode=mode) for mode in ('legacy', 'preserve')]
    rows += [dict(kind='shape', seed=9934100 + i, load=load, shape=shape, index=i,
                  rule=rule, mode='preserve') for i in range(3) for load in LOADS
             for shape in ('fixed', 'variable') for rule in ('SF_SPT', 'FIFO', 'LWKR')]
    return rows


def prepare():
    assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no']).strip()
    seeds = sorted({a['seed'] for r in read(PREFLIGHT)['records'] for a in r['attempts']})
    config = dict(seeds=seeds, new_seeds=list(range(9935000, 9935020)), loads=LOADS,
                  cases=cases(), runtime=RUNTIME, stock_modes=['legacy', 'preserve'],
                  scope='input consistency and frozen-rule sensitivity; no training or operational claim')
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'campaign.json').exists()
    write_result(OUT / 'campaign.json', dict(configuration=config, sources=sources(),
        repro=repro_stamp(experiment='YR-332-at/ao', seeds=dict(inputs=seeds,
            new_inputs=config['new_seeds'], evaluation=[9934100, 9934101, 9934102]),
            params={'configuration': config}, prereg=PREREG.as_posix())))


def check():
    campaign = read(OUT / 'campaign.json')
    for path, expected in campaign['sources'].items():
        assert sha(Path(path)) == expected, ('source changed during campaign', path)
    return campaign


def demand(kind='fixed', index=0):
    bank = read(BANK)
    masses = bank['fixed_hour_masses'] if kind == 'fixed' else bank['splits']['train'][index]['hour_masses']
    return demand_for_shape(read(BASE)['profiles']['observed_visit_links'], masses,
                            source={'experiment': 'YR-332-at/ao', 'shape': kind, 'index': index})


def build(seed, load, mode):
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    return build_stage(seed=seed, load=load, profile=build_h21_profile(),
        information_profile=read(INFO), demand_profile=demand(), initial_stock_mode=mode)


def _input_case(case):
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    seed, load = case
    geometry = build_h21_profile().block
    old, new = build(seed, load, 'legacy'), build(seed, load, 'preserve')
    stats = check_stage(new, load)
    assert old['vessel_schedule'] == new['vessel_schedule'] and old['fleet'] == new['fleet']
    before = Counter(c.size for s in old['scenarios'].values() for c in s.containers.values())
    after = Counter(c.size for s in new['scenarios'].values() for c in s.containers.values())
    assert before == after
    for b, scn in new['scenarios'].items():
        prior = old['scenarios'][b]
        assert scn.jobs == prior.jobs and scn.vessels == prior.vessels
        targets = [e['target'] for e in new['schedule'] if e['block'] == b and e['target']]
        targets += [j.target_container for j in scn.jobs if j.target_container]
        assert len(targets) == len(set(targets)) and set(targets) <= set(scn.containers)
        protected = {e['target'] for e in old['schedule'] if e['block'] == b and e['target']}
        protected |= {j.target_container for j in prior.jobs if j.target_container}
        assert all(prior.containers[c] == scn.containers[c] for c in protected)
        piles = {}
        for c in scn.containers.values():
            assert 1 <= c.bay <= geometry.bay_count and 1 <= c.row <= geometry.row_count
            assert 1 <= c.tier <= geometry.tier_max
            piles.setdefault((c.bay, c.row), []).append(c)
        for pile in piles.values():
            assert sorted(c.tier for c in pile) == list(range(1, len(pile) + 1))
            assert len({c.size for c in pile}) == 1
    old_requests = {e['job_id']: (e['requested_flow'], e['block']) for e in old['schedule']}
    assert old_requests == {e['job_id']: (e['requested_flow'], e['block']) for e in new['schedule']}
    same = old['schedule'] == new['schedule'] and old['scenarios'] == new['scenarios']
    if not old['flow_fallbacks_total']:
        assert same
    row = dict(seed=seed, load=load, before_conversions=old['flow_fallbacks_total'],
               after_conversions=new['flow_fallbacks_total'], unchanged=same, stats=stats,
               initial_stock=new['initial_stock'], stock_count=sum(after.values()),
               stock_sizes={str(k.value): v for k, v in after.items()},
               before_sha256=digest((old['scenarios'], old['schedule'])),
               after_sha256=digest((new['scenarios'], new['schedule'])))
    return row


def input_case(case):
    try:
        return _input_case(case)
    except Exception as error:
        return dict(seed=case[0], load=case[1], error=repr(error), report=getattr(error, 'report', None))


def inputs(workers):
    campaign = check(); cfg = campaign['configuration']
    jobs = [(s, n) for s in cfg['seeds'] + cfg['new_seeds'] for n in cfg['loads']]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(input_case, c): c for c in jobs}
        for future in as_completed(pending):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append(dict(seed=pending[future][0], load=pending[future][1], error=repr(error),
                                 report=getattr(error, 'report', None)))
            write_result(OUT / 'input-progress.json', dict(records=rows))
            if len(rows) % 12 == 0 or len(rows) == len(jobs):
                print(f'inputs {len(rows)}/{len(jobs)}', flush=True)
    rows.sort(key=lambda r: (r['seed'], r['load']))
    repeat = input_case((9934102, 15000))
    assert repeat == next(r for r in rows if (r['seed'], r['load']) == (9934102, 15000))
    failures = [r for r in rows if 'error' in r]
    write_result(OUT / 'input-audit.json', dict(records=rows, failures=failures, repeat_equal=True,
        campaign_sha256=sha(OUT / 'campaign.json'), repro=campaign['repro']))
    assert not failures, failures


def evaluate(index):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    torch.set_num_threads(1)
    campaign = check(); case = campaign['configuration']['cases'][index]
    destination = OUT / 'evaluation' / f'{index:03d}.json.gz'
    assert not destination.exists()
    original, announcer = ledger.install_announcer()

    class Probe(FlowProbe):
        def start(self, built, mbt):
            self.stock = built['initial_stock']
            super().start(built, mbt)

        def finish(self, mbt, records, ep):
            for sim in mbt.blocks.values():
                sim.check_invariants()
            self.visit_stats = dict(mbt.visit_stats)
            return super().finish(mbt, records, ep)

    probe = Probe(announcer)
    started = time.perf_counter()
    try:
        ep = episode.run_episode(load=case['load'], seed=case['seed'], dispatcher=case['rule'],
            information_profile=read(INFO), demand_profile=demand(case['shape'], case['index']),
            initial_stock_mode=case['mode'], evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = ep.as_dict(); metrics = raw_metrics(probe.rows); assert_metrics(metrics, result)
    flow_rows = copy.deepcopy(probe.flow_rows)
    for r in flow_rows:
        if r['flow'].startswith('GATE_'):
            r['finish_s'] = r['yard_done_s']
    data = dict(index=index, case=case, result=result, probe=probe.result, metrics_by_flow=metrics,
                completion_by_flow={f'at{h}': aggregate(flow_rows, h * 3600.) for h in (24, 26)},
                stock=probe.stock, visit_stats=probe.visit_stats, schedule_sha256=digest(probe.schedule),
                orders_sha256=digest(probe.rows), seconds=time.perf_counter() - started,
                campaign_sha256=sha(OUT / 'campaign.json'), repro=campaign['repro'])
    # Persist evidence before rejecting a failed execution.
    for sub, value in [('evaluation', data), ('orders', probe.rows), ('flow-orders', flow_rows)]:
        path = OUT / sub / f'{index:03d}.json.gz'; path.parent.mkdir(parents=True, exist_ok=True)
        assert not path.exists(); gz_save(path, value)
    assert ep.admitted == case['load'] and ep.policy_exceptions == ep.rollout_worlds == ep.traded_edges == 0
    assert probe.result['ledger_audit']['skipped'] == 0
    assert all(r['review_count'] == 1 for r in probe.rows)
    assert all(abs(r['attempt']['review_s'] - r['notice_s']) <= 1e-6 for r in probe.rows)
    assert probe.visit_stats['late_releases'] == probe.visit_stats['late_exits'] == 0
    print(f'eval {index} {case}: work={result["mean_gate_to_job_done_s"] / 60:.3f}min {data["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'inputs', 'evaluate'])
    parser.add_argument('index', nargs='?', type=int)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'inputs': inputs(args.workers)
    else: evaluate(args.index)
