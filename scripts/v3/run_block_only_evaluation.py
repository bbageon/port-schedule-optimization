"""Add the fixed twenty RL_SPACE months without changing the running three-arm batch."""
import argparse
from collections import deque
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
from run_request_audit import ROOT, now
from independent_eval_checks import SEEDS, read, save, sha
import run_independent_evaluation as base

GIB = 1024**3


def process_rss(pid):
    try:
        raw = Path(f'/proc/{pid}/status').read_text()
        return int(re.search(r'VmRSS:\s+(\d+)', raw).group(1)) * 1024
    except (FileNotFoundError, ProcessLookupError, AttributeError):
        return 0


def quota(*, total, available, primary_cpus, primary_rss, extra_rss,
          external_rss, occupied, budget=3*GIB):
    """Reserve primary capacity even between workers; count other simulations too."""
    extra_live = len(extra_rss)
    future_primary = sum(max(budget, r) for r in primary_rss)
    future_primary += max(0, len(primary_cpus)-len(primary_rss))*budget
    external = sum(max(budget,r) for r in external_rss)
    extra_reserved = sum(max(budget, r) for r in extra_rss)
    reserve_headroom = sum(max(0, budget-r) for r in primary_rss + extra_rss + external_rss)
    reserve_headroom += max(0, len(primary_cpus)-len(primary_rss))*budget
    # Two GiB for supervisors/runtime, plus 20% physical-memory headroom.
    slots_total = int((.8*total - external - 2*GIB - future_primary - extra_reserved)//budget)
    slots_available = int((.8*available - reserve_headroom - GIB)//budget)
    slots_count = 16 - len(primary_cpus) - len(external_rss) - extra_live
    slots = max(0, min(slots_total, slots_available, slots_count))
    cpus = [c for c in range(20) if c not in set(primary_cpus) | set(occupied)]
    return cpus[:slots]


def reserved_primary_cpus(status, configured):
    """Keep capacity for queued jobs; release idle cores only after the queue drains."""
    if status['state'] in ('completed', 'failed'):
        return []
    if (status.get('phase') != 'months' or type(status.get('pending')) is not int
            or status['pending'] != 0):
        return list(configured)
    active = status['active']
    cpus = [item['cpu'] for item in active]
    if len(cpus) != len(set(cpus)) or not set(cpus) <= set(configured):
        raise ValueError('Primary active CPUs disagree with its reserved capacity')
    return sorted(cpus)


def verified_config(args):
    cfg = read(args.config)
    if cfg['arm'] != 'RL_SPACE' or tuple(cfg['seeds']) != SEEDS:
        raise ValueError('Expected exactly twenty fixed RL_SPACE months')
    for name, expected in cfg['files'].items():
        if sha(args.workspace / name) != expected:
            raise ValueError(f'Frozen add-on artifact changed: {name}')
    frozen_base = ROOT / cfg['base_config']
    if read(frozen_base) != read(args.workspace / cfg['base_config']):
        raise ValueError('Workspace and frozen base configuration contents differ')
    old = base.verified_config(SimpleNamespace(config=frozen_base,
                                              workspace=args.workspace))
    if read(args.workspace / cfg['primary_run'] / 'launch.json')['config_sha256'] != sha(frozen_base):
        raise ValueError('Primary run did not use the frozen base config')
    old = dict(old, prereg=cfg['prereg'])
    return cfg, old


def resources(args, cfg, active):
    primary = args.workspace / cfg['primary_run']
    status = read(primary / 'progress.json')
    terminal = status['state'] in ('completed', 'failed')
    primary_active = status.get('active', [])
    configured = read(primary / 'months-resources.json')['cpus'] if not terminal else []
    reserved = reserved_primary_cpus(status, configured)
    # If a supervisor has failed, no new add-on work is launched; active work drains.
    blocked = status['state'] in ('failed', 'draining_after_failure') or (primary / 'failure.json').exists()
    stale = not terminal and time.time() - (primary / 'progress.json').stat().st_mtime > 300
    mem = Path('/proc/meminfo').read_text()
    get = lambda k: int(re.search(k+r':\s+(\d+)', mem).group(1))*1024
    primary_rss = [process_rss(v['pid']) for v in primary_active]
    extra_rss = [process_rss(p.pid) for p, _ in active.values()]
    external = [(pid, r) for pid in cfg['external_pids'] if (r := process_rss(pid))]
    external_rss = [r for _, r in external]
    external_cpus = set()
    for pid, _ in external:
        try:
            external_cpus.update(os.sched_getaffinity(pid))
        except ProcessLookupError:
            pass
    cpus = quota(total=get('MemTotal'), available=get('MemAvailable'), primary_cpus=reserved,
                 primary_rss=primary_rss, extra_rss=extra_rss, external_rss=external_rss,
                 occupied=[c for _, c in active.values()] + list(external_cpus))
    info = dict(primary_state=status['state'], primary_reserved_cpus=reserved,
        primary_active=len(primary_active), primary_pending=status.get('pending'),
        addon_active=len(active), external_active=len(external_rss), external_cpus=sorted(external_cpus),
        mem_total_bytes=get('MemTotal'), mem_available_bytes=get('MemAvailable'),
        budget_per_worker_gib=3, total_worker_limit=16, cpu_limit=20,
        primary_failed=blocked, primary_stale=stale)
    return ([] if blocked or stale else cpus), info


def command(args, **flags):
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--workspace', str(args.workspace),
           '--config', str(args.config), '--out', str(args.out)]
    if getattr(args, 'recover_from', None) is not None and 'seed' not in flags:
        cmd += ['--recover-from', str(args.recover_from)]
    for key, value in flags.items():
        cmd += ['--'+key.replace('_', '-')]
        if value is not True:
            cmd += [str(value)]
    return cmd


def validate_block_only(folder, *, smoke=False):
    result = read(folder / 'result.json')
    completion = read(folder / 'completion.json')
    checks = dict(arm=result['arm'] == 'RL_SPACE', no_temporal_actions=result['time'] == 0,
        no_daily_temporal_actions=all(d['n_time'] == 0 for d in result['days']),
        records=completion['audit']['passed'], result_hash=completion['result_sha256'] == sha(folder/'result.json'))
    if smoke:
        checks['spatial_path_exercised'] = result['space'] > 0
    verdict = dict(passed=all(checks.values()), checks=checks, spatial=result['space'],
                   temporal=result['time'], remaining_work=completion['audit'])
    save(folder / 'block-only-audit.json', verdict)
    if not verdict['passed']:
        raise ValueError(f'Block-only path check failed: {checks}')
    return verdict


def run_jobs(args, cfg, jobs, *, smoke=False, retained=()):
    pending, active, completed, failed = deque(jobs), {}, list(retained), []
    phase = 'smoke' if smoke else 'months'
    try:
        while pending or active:
            cpus, info = resources(args, cfg, active)
            if info['primary_failed'] and not failed:
                failed.append(dict(reason='Primary batch failed; preserve all active evidence'))
            if not failed:
                for cpu in cpus:
                    if not pending:
                        break
                    seed = pending.popleft()
                    folder = args.out / phase / str(seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    with (folder/'RL_SPACE.log').open('xb') as log:
                        flags = dict(seed=seed, cpu=cpu)
                        if smoke:
                            flags['smoke'] = True
                        proc = subprocess.Popen(command(args, **flags), cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    active[seed] = (proc, cpu)
            for seed, (proc, cpu) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                folder = args.out / phase / str(seed) / 'RL_SPACE'
                if code != 0 or not (folder/'completion.json').exists():
                    failed.append(dict(seed=seed, exit_code=code))
                else:
                    completed.append(read(folder/'completion.json'))
                del active[seed]
            save(args.out/'progress.json', dict(at=now(),
                state='draining_after_failure' if failed else ('running' if active else 'waiting_for_resources'),
                phase=phase, planned=len(jobs)+len(retained), completed=len(completed),
                retained=len(retained), pending=len(pending), failed=failed,
                active=[dict(seed=s, arm='RL_SPACE', pid=p.pid, cpu=c) for s,(p,c) in active.items()],
                resources=info))
            if failed and not active:
                break
            if pending or active:
                time.sleep(10)
        save(args.out/f'{phase}-outcomes.json', dict(completed=completed, failed=failed, pending=list(pending)))
        if failed:
            raise RuntimeError(f'Add-on failed; other active runs finished and were saved: {failed}')
        return completed
    finally:
        # Unexpected interrupt: only this supervisor's own children are owned here.
        for proc, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in active.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def supervise(args, cfg, old):
    import torch
    from yard_rl.v3.eval.contracts import runtime_identity
    torch.set_num_threads(1)
    reference = read(args.workspace / cfg['reference_manifest'])
    current = runtime_identity()
    if current != reference['contract']['runtime']:
        raise ValueError('Simulation source/config/runtime changed from the primary batch')
    save(args.out/'source-compatibility.json', dict(passed=True, runtime=current,
        reference_manifest=cfg['reference_manifest'], changes='Only policy arm: RL_SPACE; no simulation changes'))
    retained = []
    if args.recover_from is not None:
        from recover_block_only_evaluation import recover_completed
        retained = recover_completed(args, cfg, old)
    else:
        smoke = run_jobs(args, cfg, [9_900_722], smoke=True)
        folder = args.out/'smoke/9900722/RL_SPACE'
        verdict = validate_block_only(folder, smoke=True)
        result = read(folder/'result.json')
        primary_smoke = read(args.workspace/cfg['primary_run']/'smoke/9900722/RL/result.json')
        if result['requested_identity_sha256'] != primary_smoke['requested_identity_sha256']:
            raise ValueError('Block-only smoke differs from primary request inputs')
        save(args.out/'smoke-summary.json', dict(passed=True, validation=verdict, completion=smoke,
            same_requests_as_primary=True, independent_runs=0))
    kept = {c['month']['seed'] for c in retained}
    completed = run_jobs(args, cfg, [s for s in SEEDS if s not in kept], retained=retained)
    save(args.out/'month-results.json', [c['month'] for c in completed])
    primary = args.workspace/cfg['primary_run']
    while read(primary/'progress.json')['state'] != 'completed':
        if (primary/'failure.json').exists():
            raise RuntimeError('All block runs preserved; primary batch failed before merged analysis')
        save(args.out/'progress.json', dict(at=now(), state='waiting_for_primary', phase='merge',
            completed=20, planned=20, new_training_runs=0))
        time.sleep(30)
    from block_only_comparison import merge_runs
    merged = args.out.with_name(args.out.name+'-combined')
    merge_runs(primary, args.out, merged)
    save(args.out/'progress.json', dict(at=now(), state='completed', phase='merged',
        completed=20, planned=20, total_policy_runs=80, new_training_runs=0,
        combined_output=str(merged), claim_eligible=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'config', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--recover-from', type=Path,
                        help='Copy verified finished runs into a new output; run only unstarted seeds')
    args = parser.parse_args()
    for name in ('workspace', 'config', 'out'):
        setattr(args, name, getattr(args,name).resolve())
    if args.recover_from is not None:
        args.recover_from = args.recover_from.resolve()
    os.chdir(ROOT)
    cfg, old = verified_config(args)
    if args.seed is not None:
        if args.seed not in ((9_900_722,) if args.smoke else SEEDS) or not 0 <= args.cpu < 20:
            raise ValueError('Unregistered seed or CPU')
        args.arm = 'RL_SPACE'
        try:
            base.child(args, old)
            validate_block_only(args.out/('smoke' if args.smoke else 'months')/str(args.seed)/args.arm,
                                smoke=args.smoke)
        except BaseException:
            folder = args.out/('smoke' if args.smoke else 'months')/str(args.seed)/args.arm
            save(folder/'failure.json', dict(at=now(), traceback=traceback.format_exc()))
            raise
        return
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Use a clean frozen checkout')
    if args.launch:
        if args.recover_from is not None:
            from recover_block_only_evaluation import recovery_plan
            recovery_plan(args, cfg, old, require_stopped=True)
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out/'supervisor.log').open('xb') as log:
            proc = subprocess.Popen(command(args),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        receipt = dict(at=now(),pid=proc.pid,source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],text=True).strip(), source_checkout=str(ROOT),
            config_sha256=sha(args.config),checkpoint_sha256=old['checkpoint_sha256'],
            planned_independent_runs=20,primary_run=cfg['primary_run'],new_training_runs=0,
            recover_from=str(args.recover_from) if args.recover_from else None)
        save(args.out/'launch.json',receipt)
        print(receipt)
        return
    os.sched_setaffinity(0,set(range(20)))
    try:
        supervise(args,cfg,old)
    except BaseException:
        save(args.out/'failure.json',dict(at=now(),traceback=traceback.format_exc()))
        save(args.out/'progress.json',dict(at=now(),state='failed',claim_eligible=False))
        raise


if __name__ == '__main__':
    main()
