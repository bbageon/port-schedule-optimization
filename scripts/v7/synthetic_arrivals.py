"""Apply the validated correlated minute model, preserving hourly visit contracts."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import subprocess
import time

from synthetic_inputs import OUT as INPUTS, profile as mix_profile
from audit_day_shapes import BASE,INFO,check_stage,read
from completion_learning import LOADS,RUNTIME
from evaluate_visit_links import input_stats,rule_checks
from six_step_learning import gz_save,gz_read,sha
from yard_rl.integrated.repro import write_result,repro_stamp
from yard_rl.v7.evaluation_probe import digest

OUT=Path('outputs/reports/yr332_parametric_arrivals')
PREREG=Path('docs/research/v7-discharge/minute-arrival-revision-2.md')
CAL=INPUTS/'multiscale-minute-calibration.json'


def profile(case):
    old=dict(case,axis='mix',arm='variable')
    p=mix_profile(old)
    if case['arm']=='parametric':
        cal=read(CAL);assert cal['passed']
        p['within_hour']=dict(method='correlated_minutes',parameters=cal['parameters'],
                              source=dict(calibration_sha256=sha(CAL),unit='physical visits'))
    return p


def cases():
    return [dict(axis='minutes',arm=arm,split='train',index=i,load=load,seed=9937100+i,rule=rule)
            for i in range(3) for load in LOADS for arm in ('observed','parametric') for rule in ('SF_SPT','FIFO')]


def prepare():
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    assert read(INPUTS/'summary.json')['cases']==54
    assert read(INPUTS/'minute-implementation-check.json')['passed']
    assert not (OUT/'campaign.json').exists()
    paths=[Path(__file__),Path('scripts/v7/synthetic_inputs.py'),PREREG,
        Path('docs/research/v7-discharge/minute-arrival-revision.md'),CAL,BASE,INFO,INPUTS/'bank.json',
        *Path('src/yard_rl/v7').rglob('*.py')]
    cfg=dict(cases=cases(),runtime=RUNTIME,calibration=read(CAL),scope='conditional synthetic minute-process sensitivity')
    write_result(OUT/'campaign.json',dict(configuration=cfg,sources={p.as_posix():sha(p) for p in paths},
        repro=repro_stamp(experiment='YR-332-ar',seeds=dict(input=[9937100,9937101,9937102]),params=cfg,prereg=PREREG.as_posix())))


def input_case(case):
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    built=build_stage(load=case['load'],seed=case['seed'],profile=build_h21_profile(),
        information_profile=read(INFO),demand_profile=profile(case))
    checks=rule_checks(built);assert not any(checks.values())
    return dict(case=case,audit=check_stage(built,case['load']),statistics=input_stats(built),rules=checks,
                scenarios_sha256=digest(built['scenarios']),vessels_sha256=digest(built['vessel_schedule']),
                visits_sha256=digest([(r['job_id'],r['flow'],r['block'],r.get('visit')) for r in built['schedule']]))


def inputs(workers):
    bank=read(INPUTS/'bank.json')
    tasks=[dict(axis='minutes',arm='parametric',split=split,index=i,load=load,seed=9937100+100*k+i)
           for k,(split,rows) in enumerate(bank['splits'].items()) for i in range(len(rows)) for load in LOADS]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(input_case,c):c for c in tasks}
        for f in as_completed(fs):rows.append(f.result())
    old=read(INPUTS/'input-audit.json')['records']
    for row in rows:
        c=row['case'];prior=next(r for r in old if all(r['case'][k]==c[k] for k in ('split','index','load','seed')) and r['case']['arm']=='variable')
        assert prior['scenarios_sha256']==row['scenarios_sha256']
        assert prior['vessels_sha256']==row['vessels_sha256']
        assert prior['audit']['flow_counts']==row['audit']['flow_counts']
        assert prior['audit']['hour_counts']==row['audit']['hour_counts']
        assert prior['statistics']['visits']==row['statistics']['visits']
    rows.sort(key=lambda r:(r['case']['split'],r['case']['index'],r['case']['load']))
    assert input_case(tasks[0])==next(r for r in rows if r['case']==tasks[0])
    write_result(OUT/'input-audit.json',dict(records=rows,cases=len(rows),repeat_equal=True,
        same_stock_vessels_hourly_flows_visits=True))
    print('parametric input contracts',len(rows),'passed',flush=True)


def evaluate(i):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe,aggregate
    from verify_job_completion import raw_metrics,assert_metrics
    torch.set_num_threads(1);campaign=read(OUT/'campaign.json');case=cases()[i]
    for p,h in campaign['sources'].items():assert sha(Path(p))==h,('source changed',p)
    original,announcer=ledger.install_announcer()
    class Probe(FlowProbe):
        def finish(self,mbt,records,ep):
            for sim in mbt.blocks.values():sim.check_invariants()
            self.visits=dict(mbt.visit_stats)
            return super().finish(mbt,records,ep)
    probe=Probe(announcer);started=time.perf_counter()
    try:
        ep=episode.run_episode(load=case['load'],seed=case['seed'],dispatcher=case['rule'],
            information_profile=read(INFO),demand_profile=profile(case),evaluation_probe=probe,**RUNTIME)
    finally:episode.V3Announcer=original
    result=ep.as_dict();metrics=raw_metrics(probe.rows);assert_metrics(metrics,result)
    flows=copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'):r['finish_s']=r['yard_done_s']
    data=dict(case=case,result=result,metrics=metrics,probe=probe.result,visits=probe.visits,
        completion={f'at{h}':aggregate(flows,h*3600.) for h in (24,26)},orders_sha256=digest(probe.rows),
        schedule_sha256=digest(probe.schedule),seconds=time.perf_counter()-started)
    if case['arm']=='observed':
        oldcases=read(INPUTS/'campaign.json')['configuration']['cases']
        oi=next(j for j,c in enumerate(oldcases) if c['arm']=='variable' and all(c[k]==case[k] for k in ('index','load','seed','rule')))
        old=gz_read(INPUTS/'evaluation'/f'{oi:03d}.json.gz')
        assert old['orders_sha256']==data['orders_sha256']
        assert all(old['probe'][k]==data['probe'][k] for k in
                   ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'))
        data['prior_run_exact']=True
    for sub,value in [('evaluation',data),('orders',probe.rows),('flow-orders',flows)]:
        path=OUT/sub/f'{i:03d}.json.gz';assert not path.exists();gz_save(path,value)
    assert ep.admitted==case['load'] and ep.policy_exceptions==ep.traded_edges==ep.rollout_worlds==0
    assert probe.result['ledger_audit']['skipped']==0 and all(r['review_count']==1 for r in probe.rows)
    assert probe.visits['late_releases']==probe.visits['late_exits']==0
    print(f'minutes {i} {case} {result["mean_gate_to_job_done_s"]/60:.3f}min {data["seconds"]:.1f}s',flush=True)


def run(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(evaluate,i) for i in range(len(cases())) if not (OUT/'evaluation'/f'{i:03d}.json.gz').exists()]
        for n,f in enumerate(as_completed(fs),1):f.result();print(f'minutes {n}/{len(fs)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','inputs','run']);p.add_argument('--workers',type=int,default=10)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='inputs':inputs(a.workers)
    else:run(a.workers)
