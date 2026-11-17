"""Evaluate frozen checkpoints as available; no adaptive stopping or selection."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from value_stability import check, folder
from value_stability_eval import cases, evaluate
from value_stability_inputs import ARMS, SEEDS, OUT


def stable(path):
    return path.exists() and time.time()-path.stat().st_mtime > 5


def main():
    check(); plan=cases(); pending=set(range(len(plan))); jobs={}; finished=0
    with ProcessPoolExecutor(max_workers=10) as pool:
        while pending or jobs:
            trained=all(stable(folder(a,s)/'final.pt') for a in ARMS for s in SEEDS)
            capacity=10 if trained else 4
            for i in sorted(pending):
                if len(jobs)>=capacity:break
                c=plan[i]
                if c['split']=='holdout' and not trained:continue
                if c['arm'] in ARMS and not stable(folder(c['arm'],c['seed'])/f'{c["stage"]}.pt'):continue
                jobs[pool.submit(evaluate,i)]=i;pending.remove(i)
            for job in list(jobs):
                if job.done():
                    job.result();del jobs[job];finished+=1
                    print(f'fixed evaluation progress {finished}/126',flush=True)
            if pending or jobs:time.sleep(5)
    print('All 126 frozen evaluations finished.',flush=True)


if __name__=='__main__': main()
