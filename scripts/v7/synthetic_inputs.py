"""Frozen, separately paired mix and parametric-arrival diagnostics (ao/ar)."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import statistics as st
import subprocess
import time

import numpy as np

from audit_day_shapes import BASE, INFO, check_stage, read
from completion_learning import LOADS, RUNTIME
from evaluate_visit_links import input_stats, rule_checks
from six_step_learning import gz_save, sha
from yard_rl.integrated.repro import repro_stamp, write_result
from yard_rl.v7.day_shape import demand_for_shape
from yard_rl.v7.day_mix import add_daily_mix
from yard_rl.v7.evaluation_probe import digest

OUT = Path('outputs/reports/yr332_synthetic_inputs')
PREREG = Path('docs/research/v7-discharge/synthetic-four-step-prereg.md')
BANK = Path('outputs/reports/yr332_initial_stock/bank-stock-preserved.json')
MEASUREMENT = Path('outputs/reports/yr332_day_flow_volume/measurement.json')
TEMPLATES = Path('outputs/reports/yr332_visit_arrivals/visit-arrival-templates.json')


def save(name, value):
    write_result(OUT / name, value)


def minute_metrics(matrix):
    x = np.asarray(matrix, dtype=float)
    x = x[x.sum(axis=1) >= 2]
    result = {}
    for width in (1, 10):
        b = x.reshape(len(x), 60 // width, width).sum(axis=2)
        result[f'index_{width}m'] = float(np.mean(b.var(axis=1) / b.mean(axis=1)))
    centred = x - x.mean(axis=1, keepdims=True)
    denominator = np.sum(centred[:, :-1] ** 2, axis=1)
    ok = denominator > 0
    result['lag1'] = float(np.mean(np.sum(centred[ok, :-1] * centred[ok, 1:], axis=1) / denominator[ok]))
    return result


def calibrate_minutes():
    templates = read(TEMPLATES)
    matrix = np.array([x for days in templates['minute_counts'].values() for x in days if sum(x) >= 2])
    counts = matrix.sum(axis=1)
    coincidence = float(np.mean((matrix * (matrix - 1)).sum(axis=1) / (counts * (counts - 1))))
    alpha = (1 - coincidence) / (60 * coincidence - 1) if coincidence > 1 / 60 else None
    observed = minute_metrics(matrix)
    rows = []
    for method in ('poisson', 'dirichlet_minutes'):
        if method == 'dirichlet_minutes' and alpha is None:
            continue
        repeats = []
        for i in range(100):
            rng = np.random.default_rng(9936100 + i)
            probs = (np.full((len(counts), 60), 1 / 60) if method == 'poisson'
                     else rng.dirichlet(np.full(60, alpha), size=len(counts)))
            sample = np.array([rng.multinomial(int(n), p) for n, p in zip(counts, probs)])
            repeats.append(minute_metrics(sample))
        mean = {k: st.mean(r[k] for r in repeats) for k in observed}
        errors = {k: abs(mean[k] - observed[k]) for k in observed}
        passed = all(errors[k] <= limit for k, limit in dict(index_1m=.10, index_10m=.15, lag1=.05).items())
        rows.append(dict(method=method, mean=mean, errors=errors, passed=passed,
                         range={k: [min(r[k] for r in repeats), max(r[k] for r in repeats)] for k in observed}))
    selected = next((r['method'] for r in rows if r['passed']), None)
    result = dict(observed=observed, alpha=alpha, coincidence=coincidence, candidates=rows, selected=selected,
                  input_hour_count=len(counts), source_sha256=sha(TEMPLATES),
                  scope='development fit; conditional counts; high-volume extrapolation unvalidated')
    save('minute-calibration.json', result)
    return result


def prepare():
    assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no']).strip()
    assert not (OUT / 'campaign.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    daily = read(MEASUREMENT)['day_variation']['per_date']
    bank = add_daily_mix(read(BANK), daily)
    save('bank.json', bank)
    minutes = calibrate_minutes()
    observed_shares = [d['gate_out_share'] for d in daily]
    shares = [r['gate_out_share'] for r in bank['splits']['train']]
    save('mix-calibration.json', dict(observed_mean=st.mean(observed_shares), observed_sd=st.pstdev(observed_shares),
        observed_range=[min(observed_shares), max(observed_shares)], train_mean=st.mean(shares), train_sd=st.pstdev(shares),
        train_range=[min(shares), max(shares)], legacy_fixed=read(BASE)['profiles']['observed_visit_links']['gate_out_share'],
        source_sha256=sha(MEASUREMENT), limitation='interpolation reduces variance; same development family across splits'))
    cases = [dict(axis='mix', arm=arm, split='train', index=i, load=load, seed=9937100+i,
                  rule=rule) for i in range(3) for load in LOADS for arm in ('fixed', 'variable')
             for rule in ('SF_SPT', 'FIFO', 'LWKR')]
    if minutes['selected']:
        cases += [dict(axis='minutes', arm=arm, split='train', index=i, load=load, seed=9937100+i,
                       rule=rule) for i in range(3) for load in LOADS for arm in ('observed', 'parametric')
                  for rule in ('SF_SPT', 'FIFO')]
    paths = [Path(__file__), PREREG, BASE, INFO, BANK, MEASUREMENT, TEMPLATES,
             *Path('src/yard_rl/v7').rglob('*.py')]
    config = dict(cases=cases, runtime=RUNTIME, loads=LOADS, input_splits={k: len(v) for k, v in bank['splits'].items()},
                  selected_within_hour=minutes['selected'], alpha=minutes['alpha'],
                  claim_scope='input calibration and fixed-policy sensitivity only')
    save('campaign.json', dict(configuration=config, sources={p.as_posix(): sha(p) for p in paths},
        repro=repro_stamp(experiment='YR-332-ao/ar', seeds=dict(input=[9937100, 9937101, 9937102],
            calibration=list(range(9936100,9936200))), params=config, prereg=PREREG.as_posix())))
    print(json.dumps(dict(mix=read(OUT/'mix-calibration.json'), minutes=minutes), ensure_ascii=True), flush=True)


def profile(case):
    bank = read(OUT / 'bank.json'); row = bank['splits'][case['split']][case['index']]
    p = demand_for_shape(read(BASE)['profiles']['observed_visit_links'], row['hour_masses'],
                         source=dict(axis=case['axis'], split=case['split'], index=case['index']))
    p['gate_out_share'] = (bank['fixed_gate_out_share'] if case['axis'] == 'mix' and case['arm'] == 'fixed'
                           else row['gate_out_share'])
    if case['axis'] != 'mix' and case['arm'] == 'parametric':
        cal = read(OUT / 'minute-calibration.json')
        if not cal['selected']:
            raise ValueError('no within-hour model passed the frozen development criteria')
        p['within_hour'] = dict(method=cal['selected'], source=dict(calibration_sha256=sha(OUT/'minute-calibration.json')))
        if cal['selected'] == 'dirichlet_minutes':
            p['within_hour']['alpha'] = cal['alpha']
    return p


def input_case(case):
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    built = build_stage(load=case['load'], seed=case['seed'], profile=build_h21_profile(),
                        information_profile=read(INFO), demand_profile=profile(case), initial_stock_mode='preserve')
    checks = rule_checks(built)
    assert not any(checks.values())
    return dict(case=case, audit=check_stage(built, case['load']), statistics=input_stats(built), rules=checks,
                demand=profile(case), stock=built['initial_stock'],
                scenarios_sha256=digest(built['scenarios']), vessels_sha256=digest(built['vessel_schedule']))


def inputs(workers):
    bank = read(OUT / 'bank.json')
    cases = [dict(axis='mix', arm=arm, split=split, index=i, load=load,
                  seed=9937100+100*k+i) for k,(split, rows) in enumerate(bank['splits'].items())
             for i in range(len(rows)) for load in LOADS for arm in ('fixed','variable')]
    if read(OUT/'minute-calibration.json')['selected']:
        cases += [dict(axis='minutes', arm='parametric', split=split, index=i, load=load,
                       seed=9937100+100*k+i) for k,(split, rows) in enumerate(bank['splits'].items())
                  for i in range(len(rows)) for load in LOADS]
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(input_case,c):c for c in cases}
        for f in as_completed(tasks):
            try: records.append(f.result())
            except Exception as e: records.append(dict(case=tasks[f],error=repr(e),details=getattr(e,'report',None)))
            if len(records)%24==0: print(f'inputs {len(records)}/{len(cases)}',flush=True)
    records.sort(key=lambda r: json.dumps(r['case'],sort_keys=True))
    repeat = input_case(cases[0])
    assert repeat == next(r for r in records if r['case']==cases[0])
    failures=[r for r in records if 'error' in r]
    save('input-audit.json',dict(records=records,failures=failures,repeat_equal=True))
    assert not failures, failures


def evaluate(index):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    torch.set_num_threads(1)
    campaign=read(OUT/'campaign.json'); case=campaign['configuration']['cases'][index]
    for path, expected in campaign['sources'].items():
        assert sha(Path(path)) == expected, ('source changed',path)
    original,announcer=ledger.install_announcer()
    class Probe(FlowProbe):
        def finish(self,mbt,records,ep):
            for sim in mbt.blocks.values(): sim.check_invariants()
            self.visit_stats=dict(mbt.visit_stats)
            return super().finish(mbt,records,ep)
    probe=Probe(announcer);started=time.perf_counter()
    try:
        ep=episode.run_episode(load=case['load'],seed=case['seed'],dispatcher=case['rule'],
            information_profile=read(INFO),demand_profile=profile(case),evaluation_probe=probe,**RUNTIME)
    finally: episode.V3Announcer=original
    result=ep.as_dict();metrics=raw_metrics(probe.rows);assert_metrics(metrics,result)
    flows=copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'): r['finish_s']=r['yard_done_s']
    data=dict(case=case,result=result,metrics=metrics,probe=probe.result,visits=probe.visit_stats,
        completion={f'at{h}':aggregate(flows,h*3600.) for h in (24,26)},
        orders_sha256=digest(probe.rows),schedule_sha256=digest(probe.schedule),seconds=time.perf_counter()-started)
    for sub,value in [('evaluation',data),('orders',probe.rows),('flow-orders',flows)]:
        path=OUT/sub/f'{index:03d}.json.gz';assert not path.exists();gz_save(path,value)
    assert ep.admitted==case['load'] and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
    assert probe.result['ledger_audit']['skipped']==0
    assert all(r['review_count']==1 for r in probe.rows)
    assert probe.visit_stats['late_releases']==probe.visit_stats['late_exits']==0
    print(f'{index}: {case} {result["mean_gate_to_job_done_s"]/60:.3f} min {data["seconds"]:.1f}s',flush=True)


def run(axis,workers):
    cases=read(OUT/'campaign.json')['configuration']['cases']
    selected=[i for i,c in enumerate(cases) if c['axis']==axis]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(evaluate,i):i for i in selected if not (OUT/'evaluation'/f'{i:03d}.json.gz').exists()}
        for n,f in enumerate(as_completed(tasks),1):
            f.result();print(f'{axis} {n}/{len(tasks)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','inputs','mix','minutes'])
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='inputs':inputs(a.workers)
    else:run(a.command,a.workers)
