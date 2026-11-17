"""Wait for the frozen campaign, then run serial validation and report calculations."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('outputs/reports/yr332_resched_ppo')


def main():
    while True:
        training=(ROOT/'train.log').read_text('utf-8')
        evaluation=(ROOT/'evaluate.log').read_text('utf-8')
        if 'Traceback (most recent call last)' in training+evaluation:
            raise RuntimeError('Campaign error: automatic finalization stopped')
        if 'train: 6/6' in training and 'evaluation complete 54/54' in evaluation:break
        time.sleep(10)
    commands=[
        [sys.executable,'scripts/v7/resched_ppo.py','replay'],
        [sys.executable,'scripts/v7/resched_ppo.py','repeat'],
        [sys.executable,'scripts/v7/report_resched_ppo.py','benchmark'],
        [sys.executable,'scripts/v7/diagnose_resched_gradients.py'],
        [sys.executable,'scripts/v7/report_resched_ppo.py','summarize'],
        [str(Path(sys.base_prefix)/'python.exe'),'scripts/v7/plot_resched_schedule.py'],
        [sys.executable,'scripts/v7/report_resched_ppo.py','document']]
    for i,command in enumerate(commands):
        print(f'validation step {i+1}/{len(commands)}: {command[1:]}',flush=True)
        subprocess.run(command,check=True)
    print('MEASUREMENTS AND REPORT COMPLETE; visual review, gates and publication remain',flush=True)


if __name__=='__main__':main()
