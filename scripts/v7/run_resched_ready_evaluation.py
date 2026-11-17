"""Execute the already frozen 54 cases as checkpoints become available."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

from resched_ppo import OUT, cases, check, evaluate, folder


def ready(case):
    return 'kind' not in case or (folder(case['kind'],case['model_seed'])/f'{case["stage"]}.pt').exists()


def main():
    check();pending=list(range(len(cases())));finished=0
    # Output existence is not sufficient to declare a case complete.
    for i in list(pending):
        files=[OUT/sub/f'{i:03d}.json.gz' for sub in ('evaluation','orders','flow-orders')]
        if any(p.exists() for p in files):
            assert all(p.exists() for p in files),('incomplete prior output',i)
            pending.remove(i);finished+=1
    with ProcessPoolExecutor(max_workers=6) as pool:
        active={}
        while pending or active:
            for i in list(pending):
                if len(active)>=6:break
                if ready(cases()[i]):
                    active[pool.submit(evaluate,i)]=i
                    pending.remove(i)
            if active:
                done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:
                    f.result();active.pop(f);finished+=1
                    print(f'evaluation complete {finished}/54',flush=True)
            else:
                log=(OUT/'train.log').read_text('utf-8')
                if 'Traceback (most recent call last)' in log:
                    raise RuntimeError('Training failed; remaining checkpoints unavailable')
                time.sleep(5)


if __name__=='__main__':main()
