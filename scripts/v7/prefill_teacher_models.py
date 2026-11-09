"""Evaluate already frozen checkpoints, only during the early training phase."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark_default_eval import save
from retrain_reachable_teacher import OUT, CASES, RULES, SEEDS, check_sources, name
from six_step_learning import gz_read, sha


def ready_indices(last_model_early=False):
    finished = [s for s in SEEDS if (OUT/'completed'/f'train-{s}.json').exists()]
    if len(finished) >= 2:
        if not last_model_early or len(finished) == 3:
            return []
        # Only use the first half of the last model's training. Leave all of
        # updates 4, 5 and 6 before the normal evaluation phase can start.
        pending = next(s for s in SEEDS if s not in finished)
        report = OUT/f'model-{pending}'/'training.json'
        try:
            if len(json.loads(report.read_text(encoding='utf-8'))['rows']) >= 6:
                return []
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    progress=OUT/'progress.json'
    if progress.exists() and json.loads(progress.read_text(encoding='utf-8'))['phase']!='train':
        return []
    rules=[i for i,c in enumerate(CASES) if c[0] in RULES]
    if not all((OUT/'completed'/f'evaluate-{i}.json').exists() for i in rules):
        return []  # No overlap with the six-worker rule evaluator.
    ready=[]
    for index,(arm,seed,load,day,repeat) in enumerate(CASES):
        if arm in RULES or repeat or (OUT/'completed'/f'evaluate-{index}.json').exists():
            continue
        report=OUT/f'model-{seed}'/'training.json'
        if not report.exists():
            continue
        try:
            rows=json.loads(report.read_text(encoding='utf-8'))['rows']
        except json.JSONDecodeError:
            continue  # A training process is publishing its next completed step.
        if arm=='initial' or any(f'{r["contract"]}-{r["iteration"]}'==arm for r in rows):
            ready.append(index)
    return ready[:6]


def main(last_model_early=False):
    check_sources()
    wave=1+max((int(p.stem.split('-')[-1]) for p in (OUT/'prefill').glob('model-*.json')),default=-1)
    while indices:=ready_indices(last_model_early):
        started=time.perf_counter()
        def one(index):
            log=OUT/'logs'/f'evaluate-{index}.log'
            with log.open('w',encoding='utf-8') as f:
                p=subprocess.run([sys.executable,str(Path(__file__).with_name('retrain_reachable_teacher.py')),
                    '--operation','evaluate','--index',str(index)],stdout=f,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if p.returncode:
                raise RuntimeError(log.read_text(encoding='utf-8')[-4000:])
            d=gz_read(OUT/'evaluation'/(name(CASES[index])+'.json.gz'))
            assert d['case']==list(CASES[index]) and sha(d['checkpoint'])==d['checkpoint_sha256']
            save(OUT/'completed'/f'evaluate-{index}.json',dict(operation='evaluate',index=index,
                success=True,prefill='immutable checkpoint'))
            return index
        with ThreadPoolExecutor(max_workers=6) as pool:
            for f in as_completed([pool.submit(one,i) for i in indices]):
                print(f'frozen model prefill {wave}: case {f.result()} complete',flush=True)
        save(OUT/'prefill'/f'model-{wave}.json',dict(indices=indices,workers=6,
            source=str(Path(__file__)),source_sha256=sha(__file__),seconds=time.perf_counter()-started,
            checkpoint_selection='preregistered initial/intermediate/final only; immutable files'))
        wave+=1
    print('frozen model prefill stopped; remaining cases belong to the normal campaign',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--last-model-early',action='store_true')
    main(parser.parse_args().last_model_early)
