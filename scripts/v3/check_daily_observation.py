"""Small frozen-source on/off replay; this is not independent performance evidence."""
import argparse
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

from run_request_audit import run_one, sha


def main():
    import torch
    from yard_rl.v3.stage.month import plan_days
    from yard_rl.v3.eval.contracts import write_json
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prereg', required=True)
    args = p.parse_args()
    args.out = str(Path(args.out).resolve())
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.prereg = str(Path(args.prereg).resolve())
    Path(args.out).mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {4})
    seed = 9_900_720
    days = plan_days(seed, (60, 60))
    checks, times = {}, {}
    for arm in ('NO_REALLOC', 'RL', 'RL_TIME'):
        results = []
        for enabled in (False, True):
            cfg = SimpleNamespace(**vars(args), admission_mode='PRESERVE',
                supply_mode='ORIGINAL', diagnose_admissions=True, isolated_progress=True,
                capture_daily=enabled, daily_sample_s=300.0, experiment='YR-317-d2')
            label = ('on_' if enabled else 'off_') + arm
            results.append(run_one(label, arm, seed, days, cfg, sha(args.checkpoint)))
        off, on = results
        def raw_file(label, name):
            with gzip.open(Path(args.out)/label/name, 'rt', encoding='utf-8') as stream:
                return stream.read()
        old_days = lambda r: [{k:v for k,v in d.items() if k != 'operational'} for d in r['days']]
        checks[arm] = dict(days_unchanged=old_days(off) == old_days(on),
            request_inputs_unchanged=off['requested_identity_sha256'] == on['requested_identity_sha256'],
            request_outcomes_unchanged=off['request_summary'] == on['request_summary'],
            request_events_unchanged=raw_file('off_'+arm, 'requests.jsonl.gz') == raw_file('on_'+arm, 'requests.jsonl.gz'),
            container_events_unchanged=raw_file('off_'+arm, 'container-links.jsonl.gz') == raw_file('on_'+arm, 'container-links.jsonl.gz'),
            container_chain_unchanged=off['container_flow_summary'] == on['container_flow_summary'],
            all_recording_checks=all(off['recording_checks'].values()) and all(on['recording_checks'].values()),
            daily_file_present=(Path(args.out)/('on_'+arm)/'daily-final.jsonl').is_file(),
            samples_complete=on['daily_observation']['samples'] == 577,
            same_boundary=on['days'][0]['operational']['end'] == on['days'][1]['operational']['start'])
        times[arm] = dict(off_s=off['elapsed_s'], on_s=on['elapsed_s'])
        write_json(Path(args.out)/'progress.json', dict(last_arm=arm, checks=checks, times=times))
    passed = all(all(c.values()) for c in checks.values())
    write_json(Path(args.out)/'summary.json', dict(passed=passed, checks=checks, timings=times,
        seed=seed, days=2, trucks_per_day=60, sample_s=300,
        purpose='observer equivalence and file wiring only', independent_performance_runs=0))
    if not passed:
        raise RuntimeError('Observer replay changed results or lost measurements')


if __name__ == '__main__':
    main()
