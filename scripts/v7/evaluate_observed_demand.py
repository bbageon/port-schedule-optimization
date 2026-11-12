"""Frozen-policy sensitivity to observed flow mix, then clock-hour arrivals."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_default_eval import save
from six_step_learning import gz_read, gz_save, load_net, sha
from completion_learning import LOADS, RULES, SEEDS, RUNTIME, PROFILE, folder

OUT = Path('outputs/reports/yr332_order_representativeness')
PREREG = Path('docs/research/v7-discharge/observed-demand-prereg.md')
VARIANTS = ('original','observed_mix','observed_mix_hours')
POLICIES = [(rule,0) for rule in RULES]+[(arm,seed) for arm in ('initial','trained') for seed in SEEDS]
CASES = [(variant,arm,seed,load,0) for variant in VARIANTS for load in LOADS for arm,seed in POLICIES]
CASES += [('original','SF_SPT',0,7500,1),('observed_mix','trained',SEEDS[0],7500,1),
          ('observed_mix_hours','SF_SPT',0,7500,1)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint(arm, seed):
    return folder(seed)/('initial.pt' if arm=='initial' else 'final-5.pt') if seed else None


def demand_profile(variant):
    return read(OUT/'demand-profiles.json')['profiles'][variant]


def sources():
    paths=[Path(__file__),PREREG,PROFILE,OUT/'demand-profiles.json',OUT/'input-audit.json',
           *Path('src/yard_rl/v7').rglob('*.py')]
    paths += [Path('scripts/v7')/p for p in ('completion_learning.py','six_step_learning.py',
        'horizon_metrics.py','audit_admission_ledger.py','verify_job_completion.py','benchmark_default_eval.py',
        'evaluate_classical_rules.py','evaluate_arrival_order.py','verify_reservation_contract.py')]
    paths += [checkpoint(arm,seed) for arm,seed in POLICIES if seed]
    return {p.as_posix():sha(p) for p in paths}


def check():
    campaign=read(OUT/'campaign.json')
    for p,h in campaign['sources'].items():
        assert sha(p)==h,('Frozen experiment input changed',p)


def prepare():
    from yard_rl.integrated.repro import repro_stamp
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    assert not (OUT/'campaign.json').exists()
    config=dict(variants=VARIANTS,policies=POLICIES,cases=CASES,loads=LOADS,
        evaluation_seeds=[9921100+10*l for l in range(3)],runtime=RUNTIME,
        time_contract='actual gate entry to job completion, censored at 24h; retain unfinished work',
        primary_seconds=86400,engine_seconds=93600,training=False,teacher_worlds=0,
        selected_policy='all three saved initial and final models; no result-based selection',
        comparisons=['observed_mix minus original: flow probability only',
                     'observed_mix_hours minus observed_mix: clock-hour CDF only'],
        scope='One fresh input per load; conditional environmental sensitivity, no significance or external validity claim',
        execution_workers=12,information_profile_sha256=sha(PROFILE))
    save(OUT/'campaign.json',dict(configuration=config,sources=sources(),
        repro=repro_stamp(experiment='YR-332-aj-fixed-policy',seeds=dict(input=config['evaluation_seeds'],models=list(SEEDS)),
                          params=config,prereg=PREREG.as_posix())))


def evaluate(index, smoke=False, compatibility=False):
    import torch
    import audit_admission_ledger as ledger
    import yard_rl.v7.stage.episode as episode
    from horizon_metrics import FlowProbe, aggregate
    from verify_job_completion import raw_metrics, assert_metrics
    from yard_rl.v7.evaluation_probe import digest
    torch.set_num_threads(1)
    if not smoke:
        check()
    variant,arm,seed,load,repeat=CASES[index]
    input_seed=9921100+10*LOADS.index(load)
    if compatibility:
        assert variant=='original' and load==7500 and repeat==0
        input_seed=9917610
    if smoke:
        load,input_seed=80,9921999
    cp=checkpoint(arm,seed)
    original,ann=ledger.install_announcer();probe=FlowProbe(ann)
    started=time.perf_counter()
    try:
        ep=episode.run_episode(load=load,seed=input_seed,
            dispatcher='RL_CRANE' if seed else arm,crane_net=load_net(cp) if cp else None,
            information_profile=read(PROFILE),demand_profile=demand_profile(variant),
            evaluation_probe=probe,**RUNTIME)
    finally:
        episode.V3Announcer=original
    result=ep.as_dict();metrics=raw_metrics(probe.rows);assert_metrics(metrics,result)
    flows=copy.deepcopy(probe.flow_rows)
    for r in flows:
        if r['flow'].startswith('GATE_'):
            r['finish_s']=r['yard_done_s']
    completion={f'at{h}':aggregate(flows,h*3600.) for h in (24,26)}
    name=('smoke-' if smoke else 'compatibility-' if compatibility else '')+'-'.join(map(str,CASES[index]))
    data=dict(case=CASES[index],input_seed=input_seed,profile=demand_profile(variant),result=result,
        probe=probe.result,metrics_by_flow=metrics,completion_by_flow=completion,
        orders_sha256=digest(probe.rows),schedule_sha256=digest(probe.schedule),
        checkpoint_sha256=sha(cp) if cp else None,seconds=time.perf_counter()-started)
    assert ep.admitted==load and ep.policy_exceptions==ep.rollout_worlds==ep.traded_edges==0
    assert probe.result['ledger_audit']['skipped']==0
    assert all(r['review_count']==1 for r in probe.rows)
    assert all(abs(r['attempt']['review_s']-r['notice_s'])<=1e-6 for r in probe.rows)
    if compatibility:
        oldname=f'{arm}-{seed}-{load}-0-0.json.gz'
        old=gz_read(Path('outputs/reports/yr332_completion_learning/evaluation')/oldname)
        assert result==old['result']
        assert data['orders_sha256']==old['orders_sha256']
        for key in ('event_sha256','candidate_sha256','choice_sha256'):
            if key in probe.result:
                assert probe.result[key]==old['probe'][key]
        assert completion==old['completion_by_flow']
        data['exact_old_default']=True
    for sub,value in [('evaluation',data),('orders',probe.rows),('flow-orders',flows)]:
        path=OUT/sub/(name+'.json.gz');assert not path.exists();gz_save(path,value)
    print(f'{index}: {name}, truck={metrics["ALL"]["mean_gate_to_job_done_s"]/60:.3f}min, {data["seconds"]:.1f}s',flush=True)


def run_all():
    check()
    def job(index):
        name='-'.join(map(str,CASES[index]))
        existing=[(OUT/sub/(name+'.json.gz')).exists() for sub in ('evaluation','orders','flow-orders')]
        if all(existing):
            return dict(index=index,state='already complete')
        if any(existing):
            raise RuntimeError(f'Partial artifacts for {index}; inspect, do not overwrite')
        log=OUT/'logs'/f'evaluate-{index}.log';log.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with log.open('w',encoding='utf-8') as handle:
            p=subprocess.run([sys.executable,__file__,'evaluate',str(index)],stdout=handle,stderr=subprocess.STDOUT,env=env)
        if p.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8'))
        return dict(index=index,state='complete')
    # Spread the long inputs over the same bounded pool; no performance selection.
    indices=sorted(range(len(CASES)),key=lambda i:(-CASES[i][3],i))
    done=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(job,i) for i in indices]):
            row=future.result();done.append(row)
            save(OUT/'progress.json',dict(completed=len(done),total=len(CASES),rows=done))
            print(f'{len(done)}/{len(CASES)} {row}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','evaluate','smoke','compatibility','run-all'));p.add_argument('index',type=int,nargs='?')
    args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='run-all':run_all()
    else:evaluate(args.index,smoke=args.command=='smoke',compatibility=args.command=='compatibility')
