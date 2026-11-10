"""Resume complete evaluations and schedule only fully trained checkpoints."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import time

from completion_learning import OUT, CASES
from run_completion_remaining import run
from six_step_learning import gz_read


def complete(index):
    name='-'.join(map(str,CASES[index]))+'.json.gz'
    paths=[OUT/s/name for s in ('evaluation','orders','flow-orders')]
    if not any(p.exists() for p in paths):return False
    assert all(p.exists() for p in paths), ('partial artifacts',index)
    data=gz_read(paths[0]);assert data['case']==list(CASES[index])
    for p in paths[1:]:gz_read(p)
    lines=(OUT/'logs'/f'evaluate-{index}.log').read_text(encoding='utf-8').splitlines()
    assert lines[-1].startswith(f'eval {index} '), ('no successful completion record',index)
    return True


def ready(index):
    arm,seed,*_=CASES[index]
    if arm!='trained':return True
    try:return len(json.loads((OUT/f'model-{seed}'/'training.json').read_text())['rows'])==6
    except (FileNotFoundError,ValueError):return False


if __name__=='__main__':
    pending=[i for i in range(63,len(CASES)) if not complete(i)]
    print('remaining cases',pending,flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        active={}
        while pending or active:
            for i in list(pending):
                if len(active)==8:break
                if ready(i):active[pool.submit(run,i)]=i;pending.remove(i)
            if not active:time.sleep(2);continue
            done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
            for job in done:job.result();del active[job]
