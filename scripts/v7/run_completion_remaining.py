"""Run independent remaining cases; wait for frozen final checkpoints to be ready."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from completion_learning import OUT, CASES


def run(index):
    arm,seed,*_=CASES[index]
    if arm=='trained':
        for _ in range(7200):
            try:
                ready=json.loads((OUT/f'model-{seed}'/'training.json').read_text())
                if len(ready['rows'])==6:break
            except (FileNotFoundError,ValueError):pass
            time.sleep(2)
        else:raise TimeoutError(f'model {seed} did not finish')
    log=OUT/'logs'/f'evaluate-{index}.log'
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'scripts/v7/completion_learning.py','evaluate',str(index)],
                         stdout=f,stderr=subprocess.STDOUT)
    print(f'evaluate {index}: exit={p.returncode}',flush=True)
    if p.returncode:raise RuntimeError(log.read_text(encoding='utf-8'))


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=8) as pool:
        for job in as_completed([pool.submit(run,i) for i in range(63,len(CASES))]):job.result()
