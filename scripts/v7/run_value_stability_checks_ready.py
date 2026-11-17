"""Finish prespecified replays/audits after their dependencies have completed."""
from pathlib import Path
import subprocess
import time

from value_stability import folder, run
from value_stability_eval import cases, evaluate
from value_stability_inputs import OUT, ARMS, SEEDS, save


def await_files(paths, label):
    paths=list(paths);last=0.
    while True:
        completed=sum(p.exists() and time.time()-p.stat().st_mtime>5 for p in paths)
        if completed==len(paths):return
        if time.monotonic()-last>60:
            print(f'waiting for {label}: {completed}/{len(paths)}',flush=True);last=time.monotonic()
        time.sleep(5)


def main():
    await_files([folder(a,s)/'final.pt' for a in ARMS for s in SEEDS],'final models')
    run('replay',6)
    plan=cases()
    value_ids=[i for i,c in enumerate(plan) if c['mode']=='stochastic']
    await_files([OUT/'validation-samples'/f'{i:03d}.json.gz' for i in value_ids],'stochastic validation')
    import torch
    import value_stability_signals as signals
    torch.set_num_threads(1);signals.gradients();signals.choices()
    await_files([OUT/'flow-orders'/f'{i:03d}.json.gz' for i in range(len(plan))],'all fixed evaluations')
    for arm in ARMS:
        i=next(i for i,c in enumerate(plan) if c['arm']==arm and c['stage']=='final' and c['split']=='holdout')
        evaluate(i,repeat=True)
    import value_stability_analysis as analysis
    import report_value_stability as report
    import verify_value_stability as verify
    analysis.main()
    subprocess.run([r'C:\Users\geonu\anaconda3\python.exe','scripts/v7/plot_value_stability.py'],check=True)
    report.main();verify.main()
    save(OUT/'pipeline-complete.json',dict(completed=True,manual_figure_review_pending=True,
        dashboard_publication_pending=True))
    print('Computed results and numerical verification complete; manual review/publication pending.',flush=True)


if __name__=='__main__':main()
