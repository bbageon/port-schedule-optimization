"""YR-332-an: prove the realized-time alias fix leaves results unchanged.

1. readers: every `gate_in_s` access in v7 source and v7 scripts (static list).
2. inputs: rebuild every saved calibrated input (ak 3 variants x 3 loads; aj = ak variant 1)
   with the legacy alias (must equal the saved digest) and with the fix (only the alias moves).
   ag inputs without calibration must be byte-identical either way.
3. runs: rerun saved ak cases with the fix, and once with a poisoned alias; results,
   order rows, candidates, choices, events, jobs and records must equal the saved files.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_default_eval import save
from six_step_learning import gz_read, load_net, sha
from completion_learning import LOADS, RUNTIME, PROFILE
from evaluate_observed_demand import checkpoint, read
from evaluate_visit_arrivals import (CASES, HASHES, INPUT_SEEDS, VARIANTS, demand_profile,
                                     name, profile_sha)

OUT = Path('outputs/reports/yr332_bnct_time_field')
AK = Path('outputs/reports/yr332_visit_arrivals')
AJ = Path('outputs/reports/yr332_order_representativeness')
AG_SEEDS = {load: [9917600 + 100 * day + 10 * i for day in range(3)] for i, load in enumerate(LOADS)}
SENTINEL = -12345.0
RUNS = [('observed_hours_visit_minutes', 'SF_SPT', 0, 3500, 0, 'fixed'),
        ('observed_hours_visit_minutes', 'trained', 9917101, 3500, 0, 'fixed'),
        ('observed_hours_visit_minutes', 'SF_SPT', 0, 7500, 0, 'fixed'),
        ('observed_mix_hours', 'SF_SPT', 0, 3500, 0, 'fixed'),
        ('observed_hours_visit_minutes', 'SF_SPT', 0, 3500, 0, 'poisoned')]


def readers():
    rows = []
    for path in sorted([*Path('src/yard_rl/v7').rglob('*.py'), *Path('scripts/v7').glob('*.py')]):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        lines = path.read_text(encoding='utf-8').splitlines()
        for node in ast.walk(tree):
            kind = None
            if isinstance(node, ast.Attribute) and node.attr == 'gate_in_s':
                kind = 'attribute'
            elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                  and node.slice.value == 'gate_in_s'):
                kind = 'subscript'
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == 'get' and node.args and isinstance(node.args[0], ast.Constant)
                  and node.args[0].value == 'gate_in_s'):
                kind = 'get'
            if kind and isinstance(getattr(node, 'ctx', ast.Load()), ast.Load):
                rows.append(dict(file=path.as_posix(), line=node.lineno, kind=kind,
                                 code=lines[node.lineno - 1].strip()[:160]))
    return sorted(rows, key=lambda r: (r['file'], r['line']))


def inputs():
    from yard_rl.v7.evaluation_probe import digest
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    saved = {(r['variant'], r['load']): r['schedule_sha256']
             for r in read(AK / 'input-preflight.json')['rows']}
    info, out = read(PROFILE), []
    for variant in VARIANTS:
        for load in LOADS:
            args = dict(load=load, seed=INPUT_SEEDS[load], profile=build_h21_profile(),
                        information_profile=info, demand_profile=demand_profile(variant))
            old, new = build_stage(**args, bnct_time_sync=False), build_stage(**args)
            other = all({k: v for k, v in a.items() if k != 'gate_in_s'} ==
                        {k: v for k, v in b.items() if k != 'gate_in_s'}
                        for a, b in zip(old['schedule'], new['schedule']))
            changed = sum(a['gate_in_s'] != b['gate_in_s'] for a, b in zip(old['schedule'], new['schedule']))
            out.append(dict(source='ak' if variant != 'observed_mix_hours' else 'aj=ak variant 1',
                variant=variant, load=load, seed=INPUT_SEEDS[load], orders=len(new['schedule']),
                legacy_digest_equals_saved=digest(old['schedule']) == saved[(variant, load)],
                saved_schedule_sha256=saved[(variant, load)], new_schedule_sha256=digest(new['schedule']),
                alias_changed=changed, resynced=new['bnct_gate_in_resynced'], other_fields_identical=other,
                alias_equals_arrival=all(e['gate_in_s'] == e['arrival_s'] for e in new['schedule'])))
    for load, seeds in AG_SEEDS.items():
        for seed in seeds:
            args = dict(load=load, seed=seed, profile=build_h21_profile(), information_profile=info)
            old, new = build_stage(**args, bnct_time_sync=False), build_stage(**args)
            out.append(dict(source='ag (no calibration)', load=load, seed=seed, orders=len(new['schedule']),
                identical=old['schedule'] == new['schedule'], resynced=new['bnct_gate_in_resynced'],
                schedule_sha256=digest(new['schedule'])))
    return out


def run(index):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    import yard_rl.v7.stage.orders as orders
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    torch.set_num_threads(1)
    variant, arm, seed, load, repeat, mode = RUNS[index]
    case = (variant, arm, seed, load, repeat)
    if mode == 'poisoned':
        def poison(e):
            if 'gate_in_s' not in e:
                return False
            e['gate_in_s'] = SENTINEL
            return True
        orders.sync_realized_gate_in = poison
    cp = checkpoint(arm, seed)
    original, ann = ledger.install_announcer();probe = FlowProbe(ann)
    started = time.perf_counter()
    try:
        ep = episode.run_episode(load=load, seed=INPUT_SEEDS[load], dispatcher='RL_CRANE' if seed else arm,
            crane_net=load_net(cp) if cp else None, information_profile=read(PROFILE),
            demand_profile=demand_profile(variant), evaluation_probe=probe, **RUNTIME)
    finally:
        episode.V3Announcer = original
    result = ep.as_dict();metrics = raw_metrics(probe.rows);assert_metrics(metrics, result)
    flows = copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'):
            r['finish_s'] = r['yard_done_s']
    completion = {f'at{h}': aggregate(flows, h * 3600.) for h in (24, 26)}
    old = gz_read(AK / 'evaluation' / (name(case) + '.json.gz'))
    alias = [e['gate_in_s'] for e in probe.schedule]
    row = dict(case=list(case), mode=mode, seconds=time.perf_counter() - started,
        result_equal=result == old['result'], orders_equal=digest(probe.rows) == old['orders_sha256'],
        hashes_equal={k: probe.result[k] == old['probe'][k] for k in HASHES},
        completion_equal=completion == old['completion_by_flow'],
        schedule_sha256_saved=old['schedule_sha256'], schedule_sha256_now=digest(probe.schedule),
        input_sha256_saved=old['probe']['metadata']['input_sha256'],
        input_sha256_now=probe.result['metadata']['input_sha256'],
        alias_state=('sentinel' if all(a == SENTINEL for a in alias) else
                     'equals_arrival' if all(e['gate_in_s'] == e['arrival_s'] for e in probe.schedule) else 'mixed'),
        truck_mean_min=metrics['ALL']['mean_gate_to_job_done_s'] / 60)
    path = OUT / 'runs' / f'{index}.json';path.parent.mkdir(parents=True, exist_ok=True)
    save(path, row)
    print(json.dumps(row), flush=True)


def run_all():
    def job(index):
        log = OUT / 'logs' / f'run-{index}.log';log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with log.open('w', encoding='utf-8') as handle:
            p = subprocess.run([sys.executable, __file__, 'run', str(index)], stdout=handle,
                               stderr=subprocess.STDOUT, env=env)
        if p.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8'))
        return index
    with ThreadPoolExecutor(max_workers=len(RUNS)) as pool:
        for future in as_completed([pool.submit(job, i) for i in range(len(RUNS))]):
            print('done', future.result(), flush=True)


def report():
    reader_rows = readers()
    input_rows = inputs()
    runs = [read(OUT / 'runs' / f'{i}.json') for i in range(len(RUNS))]
    calibrated = [r for r in input_rows if r['source'] != 'ag (no calibration)']
    ag = [r for r in input_rows if r['source'] == 'ag (no calibration)']
    checks = dict(
        legacy_reproduces_saved=all(r['legacy_digest_equals_saved'] for r in calibrated),
        fix_moves_only_alias=all(r['other_fields_identical'] and r['alias_equals_arrival'] for r in calibrated),
        uncalibrated_inputs_identical=all(r['identical'] and r['resynced'] == 0 for r in ag),
        runs_identical=all(r['result_equal'] and r['orders_equal'] and all(r['hashes_equal'].values())
                           and r['completion_equal'] for r in runs),
        poisoned_alias_not_read=any(r['mode'] == 'poisoned' and r['alias_state'] == 'sentinel' and r['result_equal']
                                    for r in runs),
        schedule_digest_changed_for_calibrated=all(r['schedule_sha256_now'] != r['schedule_sha256_saved'] for r in runs))
    data = dict(task='YR-332-an', status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        readers=reader_rows, inputs=input_rows, runs=runs,
        alias_changed_total=sum(r['alias_changed'] for r in calibrated),
        orders_total=sum(r['orders'] for r in calibrated),
        reproduction='build_stage(..., bnct_time_sync=False) rebuilds the saved aj/ak schedule digests; '
                     'the default now aligns gate_in_s with arrival_s after every calibration.',
        script_sha256=sha(Path(__file__)))
    save(OUT / 'verification.json', data)
    print(json.dumps(dict(status=data['status'], checks=checks, alias_changed_total=data['alias_changed_total'],
                          orders_total=data['orders_total'], readers=len(reader_rows)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'run-all', 'report'))
    parser.add_argument('value', nargs='?', type=int)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.value)
    elif args.command == 'run-all':
        run_all()
    else:
        report()
