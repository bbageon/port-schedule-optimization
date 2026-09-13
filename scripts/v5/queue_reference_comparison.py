"""Queue one-CPU reference comparison after other terminal workloads finish."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def busy_workloads():
    result = subprocess.run(['ps', '-eo', 'pid,args'], text=True, check=True,
                            capture_output=True).stdout.splitlines()[1:]
    patterns = ('yard_rl.v4.crane', 'scripts/v4/eval_crane_split.py',
                'scripts/v4/probe_crane_trajectory.py', 'yard_rl.v5.ppo.continuous',
                'replay_residual_diagnosis.py')
    return [line.strip() for line in result if any(p in line for p in patterns)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allow-concurrent-one-core', action='store_true')
    args = ap.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError('Queue and workers must be restricted to one CPU')
    state = dict(state='queued', started_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), cpu_affinity=affinity, stage='waiting_for_other_workloads',
                 concurrent_exception=args.allow_concurrent_one_core, finished=[])

    def record(**updates):
        state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
        write(output / 'queue-status.json', state)

    def await_lane():
        if args.allow_concurrent_one_core:
            return
        while True:
            busy = busy_workloads()
            if not busy:
                return
            record(state='queued', blocking_workloads=busy)
            time.sleep(30)

    worker = Path(__file__).with_name('compare_frozen_policies.py')

    def run_worker(arm, name, original, seed, destination):
        await_lane()
        record(state='running', stage=name, blocking_workloads=[])
        command = [sys.executable, '-u', str(worker), '--arm', arm,
                   '--original-run', str(original), '--seed-bundle', str(seed),
                   '--output', str(destination)]
        with (output / (name + '.stdout.log')).open('x', encoding='utf-8') as stdout, \
             (output / (name + '.stderr.log')).open('x', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            record(child_pid=child.pid)
            code = child.wait()
        state['finished'].append(dict(stage=name, exit_code=code))
        record(child_pid=None)
        return code

    try:
        record()
        await_lane()
        smoke = output / 'smoke'
        smoke.mkdir()
        smoke_original = root / 'outputs/v5/yr306-wait-smoke-4549ae9'
        smoke_seed = root / 'outputs/v5/yr306-cargo-smoke-seed-83de18d/seed-data.json.gz'
        for arm in ('final', 'initial', 'rule'):
            if run_worker(arm, 'smoke-' + arm, smoke_original, smoke_seed, smoke / arm):
                raise RuntimeError('Short wiring check failed; long comparison was not started')
        if run_worker('initial', 'smoke-initial-repeat', smoke_original, smoke_seed,
                      smoke / 'initial-repeat'):
            raise RuntimeError('Short repeat failed')
        a, b = read(smoke / 'initial/report.json'), read(smoke / 'initial-repeat/report.json')
        normalize = lambda value: {k: v for k, v in value.items() if k not in ('code', 'wall_seconds')}
        if normalize(a) != normalize(b) or read(smoke / 'initial/days.json') != read(smoke / 'initial-repeat/days.json'):
            raise RuntimeError('Short repeat is not deterministic')
        if read(smoke / 'rule/report.json')['traded_edges'] != 0:
            raise RuntimeError('KEEP baseline unexpectedly reallocated')
        write(output / 'smoke-verification.json', dict(state='passed', frozen=True, updates=0,
              all_three_arms_passed=True, initial_repeat_identical=True,
              artifact_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in smoke.glob('*/report.json')}))
        original = root / 'outputs/v5/yr306-wait-30d-4549ae9'
        seed = root / 'outputs/reports/yr306_seed_regeneration/seed-9900306/seed-data.json.gz'
        failures = []
        for arm in ('final', 'initial', 'rule'):
            if run_worker(arm, 'full-' + arm, original, seed, output / arm):
                failures.append(arm)
        if failures:
            raise RuntimeError('Full arms failed; no whole-window savings claim: ' + ','.join(failures))
        record(stage='summarizing')
        subprocess.run([sys.executable, str(worker), '--summarize', '--output', str(output)], check=True)
        record(state='completed', stage='reference_comparison_complete')
    except BaseException as error:
        record(state='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
