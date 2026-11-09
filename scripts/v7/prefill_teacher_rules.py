"""Compute bounded waves of independent rule controls while models train.

No learning/evaluation parameters change. The main campaign recognizes the
same completion markers, so each rule case is evaluated once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark_default_eval import save
from retrain_reachable_teacher import OUT, CASES, RULES, check_sources, name
from six_step_learning import gz_read, sha


def run(wave):
    check_sources()
    progress = OUT/'progress.json'
    if progress.exists():
        assert json.loads(progress.read_text(encoding='utf-8'))['phase'] == 'train'
    # At most one wave of six; do not run after the last training model finishes.
    assert len(list((OUT/'completed').glob('train-*.json'))) < 3
    indices = [i for i,c in enumerate(CASES) if c[0] in RULES][6*wave:6*wave+6]
    assert len(indices) == 6 and 0 <= wave < 12
    started = time.perf_counter()
    def one(index):
        marker = OUT/'completed'/f'evaluate-{index}.json'
        if marker.exists():
            return index
        log = OUT/'logs'/f'evaluate-{index}.log'
        with log.open('w',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,str(Path(__file__).with_name('retrain_reachable_teacher.py')),
                '--operation','evaluate','--index',str(index)],stdout=f,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if p.returncode:
            raise RuntimeError(log.read_text(encoding='utf-8')[-4000:])
        d=gz_read(OUT/'evaluation'/(name(CASES[index])+'.json.gz'))
        assert d['case']==list(CASES[index]) and d['checkpoint'] is None
        save(marker,dict(operation='evaluate',index=index,success=True,prefill_wave=wave))
        return index
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(one,i) for i in indices]):
            print(f'rule prefill wave {wave}: case {f.result()} complete',flush=True)
    save(OUT/'prefill'/f'{wave}.json',dict(indices=indices,workers=6,
        source=str(Path(__file__)),source_sha256=sha(__file__),seconds=time.perf_counter()-started,
        runtime_sources_unchanged=True,overlap='independent fixed rule controls during training'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave',type=int,required=True)
    run(p.parse_args().wave)
