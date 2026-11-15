"""Run frozen, preregistered cases in separate single-threaded processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys

from check_initial_stock import OUT, check, read, sha, write_result


def main(phase, workers):
    manifest=check()
    execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--short','--untracked-files=no']).strip())
    assert not dirty, 'Commit tracked edits before starting the campaign'
    indices=list(range(4)) if phase=='stock' else list(range(4,58))
    jobs=sorted(indices,key=lambda i:-manifest['configuration']['cases'][i]['load'])
    results=[]
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    settings=dict(phase=phase,workers=workers,indices=indices,runner_sha256=sha(__file__),
        campaign_sha256=sha(OUT/'campaign.json'),process_threads=1,
        execution_head=execution_head,tracked_dirty=dirty,
        code_pin=manifest['repro']['code']['git_head'])
    write_result(OUT/f'execution-{phase}.json',settings)

    def run(i):
        paths=[OUT/sub/f'{i:03d}.json.gz' for sub in ('evaluation','orders','flow-orders')]
        exists=[p.exists() for p in paths]
        if all(exists):
            from six_step_learning import gz_read
            assert gz_read(paths[0])['campaign_sha256']==settings['campaign_sha256']
            return dict(index=i,returncode=0,reused=True)
        if any(exists):
            return dict(index=i,returncode=1,error='partial artifacts; inspect without overwrite')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=OUT/'logs'/f'eval-{i:03d}.log'
        with log.open('w',encoding='utf-8') as handle:
            p=subprocess.run([sys.executable,'scripts/v7/check_initial_stock.py','evaluate',str(i)],
                stdout=handle,stderr=subprocess.STDOUT,env=env)
        return dict(index=i,returncode=p.returncode,reused=False,log=log.as_posix())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run,i) for i in jobs]):
            result=future.result();results.append(result)
            write_result(OUT/f'execution-{phase}-progress.json',dict(settings=settings,results=results))
            print(f'{phase} {len(results)}/{len(jobs)}: {result}',flush=True)
    write_result(OUT/f'execution-{phase}-result.json',dict(settings=settings,results=results))
    assert all(r['returncode']==0 for r in results),results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['stock','shapes'])
    parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    main(args.phase,args.workers)
