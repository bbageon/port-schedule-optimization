"""Does the feasible-first fix clear the seed 21,000,000 / Y17 stall? (YR-317-k)

Diagnostic, not registered evidence: it runs from the working tree so the fix under
test is the one in src/. Same frozen inputs and the same NO_REALLOC policy as the
80-run evaluation, replayed only as far as --until-day, watching one block for the
stall signature (no crane in service, no committed crane time, released work waiting).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/v3'))
os.chdir(ROOT)


class Reached(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=21_000_000)
    parser.add_argument('--block', default='Y17')
    parser.add_argument('--until-day', type=float, default=8.0)
    parser.add_argument('--pruning', choices=('legacy', 'feasible_first'), required=True)
    parser.add_argument('--cpu', type=int, default=17)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)

    import torch
    torch.set_num_threads(1)
    from run_independent_evaluation import expected_month, read
    from yard_rl.v3.stage import daily_observation
    from yard_rl.v3.stage.month import plan_month
    from yard_rl.v3.stage.month_run import run_month

    cfg = read(ROOT / 'outputs/reports/yr317_v3_independent_eval/config.json')
    expected = expected_month(SimpleNamespace(workspace=ROOT), cfg, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    holder, stalls, streak = {}, [], Counter()
    original_observe = daily_observation.DailyObserver.observe

    def observe_hook(self, terminal, t, bridge):
        holder['terminal'] = terminal
        return original_observe(self, terminal, t, bridge)

    daily_observation.DailyObserver.observe = observe_hook
    started = time.monotonic()
    rows = []

    def on_state(row):
        terminal = holder.get('terminal')
        if terminal is None:
            return
        block = terminal.blocks[args.block]
        idle = (block.fleet is not None
                and all(c.state.assigned_job is None for c in block.fleet.all()))
        waiting = sum(1 for j in block.jobs.values() if j.status.name in ('WAITING', 'RELEASED'))
        stalled = idle and waiting >= 20
        streak['n'] = streak['n'] + 1 if stalled else 0
        if streak['n'] * 300 >= 3600:                       # one hour of the signature
            stalls.append(dict(at_day=row['at_s'] / 86400, waiting=waiting,
                               consecutive_samples=streak['n']))
        if row['at_s'] >= args.until_day * 86400:
            raise Reached()

    def on_day(day):
        rows.append(dict(index=day.index, load=day.load, phi_krw=day.phi_krw,
                         n_censored=day.n_censored, elapsed_s=time.monotonic() - started))
        print(json.dumps(rows[-1]), flush=True)

    state = 'completed'
    try:
        with torch.inference_mode():
            run_month(arm='NO_REALLOC', seed=args.seed, days=plan_month(args.seed), workers=1,
                      explore=0, admission_mode='PRESERVE', supply_mode='COUNT_BALANCED',
                      capture_requests=True, diagnose_admissions=True, capture_daily=True,
                      daily_sample_s=300.0, expected_input=expected,
                      candidate_pruning=args.pruning, measure_latency=True,
                      on_observation=on_state, on_day=on_day)
    except Reached:
        state = 'reached_target_day'
    finally:
        daily_observation.DailyObserver.observe = original_observe

    summary = dict(state=state, seed=args.seed, block=args.block, pruning=args.pruning,
                   until_day=args.until_day, elapsed_s=time.monotonic() - started,
                   stall_windows=len(stalls), first_stall=stalls[0] if stalls else None,
                   last_stall=stalls[-1] if stalls else None, days=rows,
                   scope='Working-tree diagnostic replay; not registered evidence.')
    (args.out / f'validate-{args.pruning}.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('state', 'pruning', 'stall_windows', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()
