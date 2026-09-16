"""Run a frozen connection test then a paired 30-day supply-contract diagnosis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from run_request_audit import ROOT, run_one, now, sha
from run_admission_replay import save, source_commit

VARIANTS={'original_baseline':('NO_REALLOC','ORIGINAL'),
          'balanced_baseline':('NO_REALLOC','COUNT_BALANCED'),
          'balanced_full':('RL','COUNT_BALANCED')}


def child(args):
    os.sched_setaffinity(0,{args.cpu})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    from yard_rl.v3.stage.month import plan_days, plan_month
    seed=9_900_700
    days=plan_days(seed,(300,)) if args.phase=='smoke' else plan_month(seed)
    arm,args.supply_mode=VARIANTS[args.variant]
    args.admission_mode='PRESERVE'
    args.diagnose_admissions=args.isolated_progress=True
    try:
        result=run_one(args.variant,arm,seed,days,args,sha(args.checkpoint))
        save(args.out/args.variant/'progress.json',{'state':'completed','at':now(),
             'pid':os.getpid(),'cpu':args.cpu,'phase':args.phase,
             'supply_plan_audit':result['supply_plan_audit'],
             'container_flow_summary':result['container_flow_summary']})
    except BaseException:
        folder=args.out/args.variant
        folder.mkdir(parents=True,exist_ok=True)
        save(folder/'failure.json',{'at':now(),'traceback':traceback.format_exc()})
        raise


def phase(args,name):
    folder=args.out/name
    folder.mkdir(exist_ok=False)
    workers={}
    try:
        for cpu,variant in enumerate(VARIANTS):
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(folder),
                 '--checkpoint',args.checkpoint,'--prereg',args.prereg,
                 '--phase',name,'--variant',variant,'--cpu',str(cpu)]
            with (folder/f'{variant}.log').open('xb') as log:
                workers[variant]=subprocess.Popen(cmd,cwd=ROOT,stdout=log,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        save(folder/'launch.json',{'at':now(),'source_commit':source_commit(),
            'workers':{v:{'pid':p.pid,'cpu':i} for i,(v,p) in enumerate(workers.items())}})
        while True:
            codes={v:p.poll() for v,p in workers.items()}
            save(args.out/'progress.json',{'at':now(),'state':'running','phase':name,'exit_codes':codes})
            if any(c not in (None,0) for c in codes.values()):
                raise RuntimeError(f'Validation worker failed: {codes}')
            if all(c==0 for c in codes.values()):
                break
            time.sleep(5)
    finally:
        for p in workers.values():
            if p.poll() is None:
                p.terminate()
        for p in workers.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
    results={v:json.loads((folder/v/'result.json').read_text()) for v in VARIANTS}
    checks={'same_truck_requests':len({r['requested_identity_sha256'] for r in results.values()})==1,
            'recording':all(all(r['recording_checks'].values()) for r in results.values()),
            'physical_container_chains':all(r['container_flow_summary']['passed'] for r in results.values()),
            'same_balanced_vessels':results['balanced_baseline']['supply_plan_audit']==results['balanced_full']['supply_plan_audit'],
            'same_vessel_volume':len({r['vessel_work_summary']['requested_moves'] for r in results.values()})==1}
    if name=='smoke':
        # No change is expected in this already feasible small scenario. This
        # checks that plan mode + the observer do not themselves change operation.
        a,b=results['original_baseline'],results['balanced_baseline']
        checks['no_change_when_original_feasible']=a['days']==b['days']
        checks['all_smoke_work_completed']=all(r['request_summary']['states']=={'COMPLETED':300}
            and r['vessel_work_summary']['all_requested_work_completed'] for r in results.values())
    summary={'phase':name,'checks':checks,'passed':all(checks.values()),
             'arms':{v:{'requests':r['request_summary'],'vessels':r['vessel_work_summary'],
                    'container_flow':r['container_flow_summary'],'supply':r['supply_plan_audit'],
                    'cost_krw':sum(d['phi_krw'] for d in r['days'])} for v,r in results.items()},
             'new_training_runs':0,'independent_policy_evaluations':0,'claim_eligible':False}
    save(folder/'summary.json',summary)
    if not summary['passed']:
        raise RuntimeError('Connection/recording checks failed; do not advance')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--prereg',required=True)
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--cpu',type=int)
    p.add_argument('--phase',choices=('smoke','full'))
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    if args.variant:
        return child(args)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Use a clean frozen checkout')
    if args.launch:
        import re
        available=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text()).group(1))
        if 3*3*1024**2 > .8*available:
            raise RuntimeError('Three workers exceed memory budget')
        checkpoint_sha,prereg_sha=sha(args.checkpoint),sha(args.prereg)
        args.out.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--out',str(args.out),
             '--checkpoint',args.checkpoint,'--prereg',args.prereg]
        with (args.out/'supervisor.log').open('xb') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True)
        receipt={'at':now(),'pid':process.pid,'source_commit':source_commit(),
                 'checkpoint_sha256':checkpoint_sha,'prereg_sha256':prereg_sha,
                 'cpus':[0,1,2],'user_cpu_limit':20,'new_training_runs':0}
        save(args.out/'launch.json',receipt)
        print(json.dumps(receipt))
        return
    os.sched_setaffinity(0,set(range(20)))
    try:
        phase(args,'smoke')
        phase(args,'full')
        save(args.out/'progress.json',{'state':'completed','at':now(),'claim_eligible':False,
            'next':'Review quantity, physical container links, remaining work; independent runs not auto-started'})
    except BaseException:
        save(args.out/'failure.json',{'at':now(),'traceback':traceback.format_exc()})
        save(args.out/'progress.json',{'state':'failed','at':now()})
        raise


if __name__=='__main__':
    main()
