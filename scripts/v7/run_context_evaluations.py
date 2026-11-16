"""Schedule the frozen 72 cases without changing inputs, models, or budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

from context_learning import OUT,check,evaluate,evaluation_cases
from six_step_learning import gz_read,sha
from yard_rl.integrated.repro import write_result


def main(kind,workers):
    check();cases=evaluation_cases();done=[]
    for i,c in enumerate(cases):
        p=OUT/'evaluation'/f'{i:03d}.json.gz'
        if not p.exists():continue
        d=gz_read(p);assert d['case']==c
        assert d['result']['admitted']==c['load']
        assert d['result']['policy_exceptions']==d['result']['traded_edges']==d['result']['rollout_worlds']==0
        assert d['probe']['ledger_audit']['skipped']==0
        assert d['visits']['late_releases']==d['visits']['late_exits']==0
        assert all((OUT/sub/p.name).exists() for sub in ('orders','flow-orders'))
        done.append(i)
    pending=[i for i,c in enumerate(cases) if i not in done and (kind!='rules' or c['model_seed']==0)]
    write_result(OUT/f'evaluation-schedule-{kind}.json',dict(completed=done,pending=pending,
        workers=workers,script_sha256=sha(Path(__file__)),
        scope='Only execution order changes: rules can run during label collection; frozen 72 cases preserved'))
    if workers==1:
        for n,i in enumerate(pending,1):evaluate(i);print(f'{kind}: {n}/{len(pending)}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fs=[pool.submit(evaluate,i) for i in pending]
            for n,f in enumerate(as_completed(fs),1):f.result();print(f'{kind}: {n}/{len(pending)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('rules','remaining'))
    p.add_argument('--workers',type=int,default=1);a=p.parse_args();main(a.kind,a.workers)
