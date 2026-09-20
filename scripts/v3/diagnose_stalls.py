"""Read-only stall scan over saved operating-state records (YR-317-j).

A block is *stalled* when, for at least ``--min-hours`` consecutive samples, no
crane is in service, no crane time is committed, and at least ``--min-released``
released (ready) yard jobs are waiting. Cranes idle in front of ready work is not
an operational outcome; it is a dispatch defect signature. The script never runs
a simulation and never modifies evidence.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def scan_run(path, *, min_released=20, min_hours=1.0, sample_s=300.0):
    need = int(round(min_hours * 3600 / sample_s))
    streak: dict[str, int] = {}
    onset: dict[str, float] = {}
    peak: dict[str, int] = {}
    stalls: dict[str, dict] = {}
    last_at = 0.0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            s = json.loads(line)
            last_at = s['at_s']
            for block, v in s['blocks'].items():
                idle = (v['yard_jobs_in_service'] == 0 and v['crane_committed_remaining_s'] == 0
                        and v['yard_jobs_released_unfinished'] >= min_released)
                if idle:
                    streak[block] = streak.get(block, 0) + 1
                    if streak[block] == 1:
                        onset[block] = s['at_s']
                    peak[block] = max(peak.get(block, 0), v['yard_jobs_released_unfinished'])
                    if streak[block] >= need:
                        rec = stalls.setdefault(block, dict(onset_day=onset[block] / 86400, episodes=0, longest_h=0.0))
                        rec['last_seen_day'] = s['at_s'] / 86400
                        rec['peak_released'] = peak[block]
                        rec['queue_at_last'] = v['truck_queue']
                        rec['vessel_unfinished_at_last'] = v['vessel_yard_unfinished']
                        rec['longest_h'] = max(rec['longest_h'], streak[block] * sample_s / 3600)
                else:
                    if block in stalls and streak.get(block, 0) >= need:
                        stalls[block]['episodes'] += 1
                    streak[block] = 0
                    peak[block] = 0
    for block, rec in stalls.items():
        if streak.get(block, 0) >= need:
            rec['episodes'] += 1
            rec['unresolved_at_end'] = True
        else:
            rec['unresolved_at_end'] = False
    return dict(end_day=last_at / 86400, stalled_blocks=stalls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='run folders containing operating-state.jsonl.gz')
    parser.add_argument('--out', required=True)
    parser.add_argument('--min-released', type=int, default=20)
    parser.add_argument('--min-hours', type=float, default=1.0)
    args = parser.parse_args()
    report = {}
    for folder in args.runs:
        folder = Path(folder)
        state = folder / 'operating-state.jsonl.gz'
        if not state.is_file():
            report[str(folder)] = dict(error='no operating-state.jsonl.gz')
            continue
        result = scan_run(state, min_released=args.min_released, min_hours=args.min_hours)
        completion = folder / 'completion.json'
        if completion.is_file():
            c = json.loads(completion.read_text(encoding='utf-8'))
            m = c.get('month', {})
            result['month'] = dict(seed=m.get('seed'), arm=m.get('arm'), cost_28d_krw=m.get('cost_28d_krw'),
                                   censored=m.get('censored'), unbound=m.get('unbound_at_cutoff'))
        report[str(folder)] = result
        print(json.dumps(dict(run=str(folder), stalled=list(result['stalled_blocks'])), ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(dict(criteria=dict(min_released=args.min_released, min_hours=args.min_hours),
                                              runs=report), ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
