"""Queue the fixed 20-month / three-policy evaluation after the supply audit.

Use a clean frozen checkout; one new process/engine for every seed-policy pair.
No training, result-based resampling, silent cache reuse, or policy changes.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from run_request_audit import ROOT, now, run_one
from independent_eval_checks import (ARMS, SEEDS, audit_run, month_row,
    paired_summary, read, save, sha, smoke_summary, supply_preflight, write_report)


def verified_config(args):
    cfg = read(args.config)
    if tuple(cfg['seeds']) != SEEDS or tuple(cfg['arms']) != ARMS:
        raise ValueError('Expected the preregistered 20 x 3 design')
    if cfg['workers_max'] > 16 or cfg['cpu_limit'] != 20:
        raise ValueError('CPU/memory budget changed')
    for name, expected in cfg['files'].items():
        if sha(args.workspace / name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    if sha(ROOT / 'src/yard_rl/v3/stage/supply_plan.py') != cfg['solver_sha256']:
        raise ValueError('Supply correction implementation changed')
    return cfg


def expected_month(args, cfg, seed):
    from yard_rl.v3.eval.seed_bank import load_bundle, digest
    bank = args.workspace / cfg['bank']
    original = next(r for r in read(bank / 'summary.json')['rows'] if r['seed'] == seed)
    payload = load_bundle(bank / original['bundle'], expected_sha256=original['bundle_sha256'])
    revised = read(args.workspace / cfg['supply_inputs'] / f'seed-{seed}.json')
    if (payload['seed'] != seed or revised['seed'] != seed
            or revised['source_bundle_sha256'] != original['bundle_sha256']
            or digest(payload['schedule']) != original['schedule_sha256']
            or digest(payload['initial_scenarios']) != original['initial_scenarios_sha256']
            or digest(revised['vessels']) != revised['revised_vessels_sha256']):
        raise ValueError('Frozen input hashes disagree')
    return dict(schedule_sha256=original['schedule_sha256'],
                initial_scenarios_sha256=original['initial_scenarios_sha256'],
                vessels_sha256=revised['revised_vessels_sha256'])


def child(args, cfg):
    import torch
    from yard_rl.v3.stage.month import plan_days, plan_month
    from yard_rl.v3.eval.seed_bank import create_month_payload, digest
    from yard_rl.v3.stage.supply_plan import balance_vessel_supply
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = args.out / ('smoke' if args.smoke else 'months') / str(args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        days = plan_days(args.seed, (60, 60))
        data = create_month_payload(args.seed, days=days)
        initial = {b: len(s['containers']) for b, s in data['initial_scenarios'].items()}
        vessels, _ = balance_vessel_supply(data['vessels'], data['schedule'], initial,
                                          {b: 1440 for b in initial})
        expected = dict(schedule_sha256=digest(data['schedule']),
                        initial_scenarios_sha256=digest(data['initial_scenarios']),
                        vessels_sha256=digest(vessels))
    else:
        days = plan_month(args.seed)
        expected = expected_month(args, cfg, args.seed)
    job = SimpleNamespace(out=str(folder), checkpoint=str(args.workspace / cfg['checkpoint']),
        prereg=str(args.workspace / cfg['prereg']), admission_mode='PRESERVE',
        supply_mode='COUNT_BALANCED', diagnose_admissions=True, isolated_progress=True,
        capture_daily=True, daily_sample_s=300.0, experiment='YR-317-d',
        purpose='frozen-policy independent monthly evaluation' if not args.smoke else 'diagnostic wiring check',
        expected_input=expected)
    result = run_one(args.arm, args.arm, args.seed, days, job, cfg['checkpoint_sha256'])
    audit = audit_run(folder / args.arm)
    if not audit['passed']:
        raise RuntimeError('Saved run audit failed; evidence preserved')
    save(folder / args.arm / 'completion.json', dict(at=now(), state='completed',
        pid=os.getpid(), cpu=args.cpu, input_hashes=expected,
        result_sha256=sha(folder / args.arm / 'result.json'), audit=audit,
        month=month_row(result)))


def command(args, **flags):
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(args.config),
           '--workspace', str(args.workspace), '--out', str(args.out)]
    for key, value in flags.items():
        cmd += ['--' + key.replace('_', '-')]
        if value is not True:
            cmd += [str(value)]
    return cmd


def resources(cfg):
    raw = Path('/proc/meminfo').read_text()
    available = int(re.search(r'MemAvailable:\s+(\d+)', raw).group(1)) * 1024
    memory_workers = int(.8 * available // (cfg['worker_budget_gib'] * 1024**3))
    return min(cfg['workers_max'], memory_workers), available


def run_jobs(args, cfg, jobs, *, smoke=False):
    limit, available = resources(cfg)
    if limit < 1:
        raise RuntimeError('Insufficient available memory for one worker')
    limit = min(limit, 3) if smoke else limit
    # Diagnostic probes can run while the supply run owns CPU 2.
    free = deque(range(3, 3 + limit) if smoke else range(limit))
    pending, active, completed = deque(jobs), {}, []
    phase = 'smoke' if smoke else 'months'
    save(args.out / f'{phase}-resources.json', dict(at=now(), workers=limit,
        mem_available_bytes=available, worker_budget_gib=cfg['worker_budget_gib'], cpus=list(free)))
    try:
        while pending or active:
            while pending and free:
                seed, arm = pending.popleft()
                cpu = free.popleft()
                folder = args.out / phase / str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / f'{arm}.log').open('xb') as log:
                    flags = dict(seed=seed, arm=arm, cpu=cpu)
                    if smoke:
                        flags['smoke'] = True
                    proc = subprocess.Popen(command(args, **flags), cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                active[(seed, arm)] = (proc, cpu)
            for key, (proc, cpu) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                seed, arm = key
                folder = args.out / phase / str(seed) / arm
                if code != 0 or not (folder / 'completion.json').exists():
                    raise RuntimeError(f'{phase} worker {key} failed with exit {code}; no replacement seed')
                completed.append(read(folder / 'completion.json'))
                del active[key]
                free.append(cpu)
            save(args.out / 'progress.json', dict(at=now(), state='running', phase=phase,
                completed=len(completed), planned=len(jobs), pending=len(pending),
                active=[dict(seed=s, arm=a, pid=p.pid, cpu=c) for (s, a), (p, c) in active.items()]))
            if active:
                time.sleep(10)
    finally:
        for proc, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in active.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return completed


def supervise(args, cfg):
    smoke = run_jobs(args, cfg, [(9_900_722, arm) for arm in ARMS], smoke=True)
    validation = smoke_summary(smoke)
    save(args.out / 'smoke-summary.json', validation)
    if not validation['passed']:
        raise RuntimeError('Diagnostic wiring/recording check failed')
    supply = args.workspace / cfg['supply_run']
    while True:
        status = read(supply / 'progress.json')
        if status['state'] == 'failed' or (supply / 'failure.json').exists():
            raise RuntimeError('Preceding supply run failed; independent evaluation not started')
        if status['state'] == 'completed':
            break
        if time.time() - (supply / 'progress.json').stat().st_mtime > 300:
            raise RuntimeError('Supply supervisor heartbeat stale; check before advancing')
        save(args.out / 'progress.json', dict(at=now(), state='waiting_for_supply',
            independent_runs=0, planned=60, supply_status=status))
        time.sleep(30)
    verdict = supply_preflight(supply, cfg['checkpoint_sha256'])
    save(args.out / 'supply-preflight.json', verdict)
    if not verdict['passed']:
        raise RuntimeError('Supply/input/physical record checks failed; independent evaluation not started')
    verified_config(args)
    # Round-robin by month, preserving all policies and all seeds including losses.
    completed = run_jobs(args, cfg, [(s, a) for s in SEEDS for a in ARMS])
    rows = [c['month'] for c in completed]
    save(args.out / 'month-results.json', sorted(rows, key=lambda r: (r['seed'], r['arm'])))
    summary = paired_summary(rows)
    save(args.out / 'summary.json', summary)
    write_report(args.out / 'results.md', rows, summary)
    save(args.out / 'progress.json', dict(at=now(), state='completed', independent_runs=60,
        new_training_runs=0, claim_eligible=False, summary='summary.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--cpu', type=int)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    for name in ('workspace', 'config', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    os.chdir(ROOT)
    cfg = verified_config(args)
    if args.arm:
        if not 0 <= args.cpu < cfg['cpu_limit']:
            raise ValueError('CPU outside approved range')
        try:
            return child(args, cfg)
        except BaseException:
            folder = args.out / ('smoke' if args.smoke else 'months') / str(args.seed) / args.arm
            save(folder / 'failure.json', dict(at=now(), traceback=traceback.format_exc()))
            raise
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Use a clean frozen source checkout')
    if args.launch:
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out / 'supervisor.log').open('xb') as log:
            proc = subprocess.Popen(command(args), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        receipt = dict(at=now(), pid=proc.pid, source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(), config_sha256=sha(args.config),
            checkpoint_sha256=cfg['checkpoint_sha256'], planned_independent_runs=60,
            source_checkout=str(ROOT), state='queued_after_supply', new_training_runs=0)
        save(args.out / 'launch.json', receipt)
        print(json.dumps(receipt))
        return
    os.sched_setaffinity(0, set(range(20)))
    try:
        supervise(args, cfg)
    except BaseException:
        save(args.out / 'failure.json', dict(at=now(), traceback=traceback.format_exc()))
        save(args.out / 'progress.json', dict(at=now(), state='failed', claim_eligible=False))
        raise


if __name__ == '__main__':
    main()
