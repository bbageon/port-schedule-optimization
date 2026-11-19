"""Run the fixed post-training sequence; any execution or integrity error stops it."""
import subprocess
import sys
import time
import argparse

from global_policy_inputs import OUT,SEEDS,ARMS,save
from global_policy_plan import check,model_path


def main(plot_python):
    check()
    assert all((model_path(a,s)/'final.pt').exists() for a in ARMS for s in SEEDS)
    stages=[('optimizer replay',['global_policy_train.py','replay','--workers','6']),
            ('fixed evaluation',['global_policy_eval.py','evaluate','--workers','6']),
            ('environment replay',['global_policy_eval.py','repeat','--workers','6']),
            ('full ledger audit',['audit_global_policy.py']),
            ('fixed aggregation',['analyze_global_policy.py']),
            ('report',['report_global_policy.py']),('figure',['plot_global_policy.py'])]
    record=[]
    for name,args in stages:
        print('STAGE:',name,flush=True);start=time.perf_counter()
        executable=plot_python if name=='figure' else sys.executable
        result=subprocess.run([executable,'-u','scripts/v7/'+args[0],*args[1:]])
        record.append(dict(stage=name,arguments=args,executable=executable,returncode=result.returncode,seconds=time.perf_counter()-start))
        save(OUT/'post-training-execution.json',dict(passed=len(record)==len(stages) and result.returncode==0,stages=record))
        result.check_returncode()
    print('Fixed post-training sequence complete. Inspect the figure, then run verify_global_policy.py.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plot-python',default=sys.executable)
    main(p.parse_args().plot_python)
