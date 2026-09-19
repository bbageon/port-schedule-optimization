"""Launch a frozen vertical pair; smoke and main require separate explicit calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('NO_REALLOC', 'RL_TIME')
CPUS = (18, 19)
GIB = 1024**3
FROZEN_KEYS = ('source_commit', 'config_sha256', 'base_config_sha256', 'prereg_sha256',
    'checkpoint_sha256', 'environment_spec_sha256', 'runtime_source_sha256', 'network_identities')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def processes():
    rows = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            argv = [s.decode('utf8', errors='replace') for s in (folder/'cmdline').read_bytes().split(b'\0') if s]
            status = dict(line.split(':', 1) for line in (folder/'status').read_text().splitlines() if ':' in line)
            rows.append(dict(pid=int(folder.name), ppid=int(status['PPid']), argv=argv,
                rss_bytes=int(status.get('VmRSS', '0 kB').split()[0])*1024,
                cpus=sorted(os.sched_getaffinity(int(folder.name)))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rows


def option(argv, flag):
    return argv[argv.index(flag)+1] if flag in argv and argv.index(flag)+1 < len(argv) else None


def same_output(row, path):
    value = option(row['argv'], '--out')
    return value is not None and Path(value).resolve() == path.resolve()


def resources():
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Launch inside WSL/Linux')
    mem_raw = Path('/proc/meminfo').read_text()
    mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in mem_raw.splitlines()}
    rows = processes()
    names = {'run_independent_evaluation.py', 'run_block_only_evaluation.py', 'run_vertical_comparison.py'}
    def worker(row):
        return bool(names.intersection(Path(arg).name for arg in row['argv'])) and (
            '--seed' in row['argv'] or '--arm' in row['argv'])
    v3 = [row for row in rows if worker(row)]
    conflicts = [row for row in rows if set(row['cpus']).intersection(CPUS)
        and len(row['cpus']) <= 2 and (row['rss_bytes'] >= GIB//8 or worker(row))]
    known = [row for row in rows if any('/scripts/v3/' in arg or '/scripts/v5/' in arg
        or arg.startswith('scripts/v3/') or arg.startswith('scripts/v5/') for arg in row['argv'])]
    checks = dict(memory_available_12gib=mem['MemAvailable'] >= 12*GIB,
        vertical_cores_free=not conflicts, v3_workers_with_pair_within20=len(v3)+2 <= 20,
        cpus_available=set(CPUS) <= os.sched_getaffinity(0))
    result = dict(at=now(), checks=checks, meminfo=mem, meminfo_raw=mem_raw,
        reserved_cpus=list(CPUS), reserve_gib=dict(vertical_pair=6, future_existing_worker=3, margin=3),
        v3_workers=len(v3), known_processes=known, conflicting_processes=conflicts,
        scope='Separate pair reservation; existing supervisors and their own budgets are unchanged')
    if not all(checks.values()):
        raise RuntimeError('Resource preflight failed: '+json.dumps(result, ensure_ascii=False))
    return result


def validate_pair(folder):
    from vertical_comparison_io import pair_summary
    completions, results = [], []
    for arm in ARMS:
        base = folder/arm
        receipt = read(base/'completion.json')
        if (receipt['state'] != 'completed' or receipt['audit']['passed'] is not True
                or receipt['result_sha256'] != sha(base/'result.json')
                or receipt['manifest_sha256'] != sha(base/'manifest.json')
                or (base/'failure.json').exists()):
            raise ValueError('Completion audit/hash failed: '+arm)
        completions.append(receipt)
        results.append(read(base/'result.json'))
    if completions[0]['identity'] != completions[1]['identity']:
        raise ValueError('Paired completion identities differ')
    return dict(completions=completions, summary=pair_summary(results))


def smoke_gate(args):
    smoke = args.out/'smoke'
    launch = read(smoke/'launch.json')
    progress = read(smoke/'progress.json')
    if progress['state'] != 'completed':
        raise ValueError('Both smoke arms must finish successfully before the explicit main launch')
    if any(same_output(row, smoke) for row in processes()):
        raise ValueError('A smoke process remains alive')
    for key, path in (('config_sha256', args.config),
            ('runner_sha256', ROOT/'scripts/v3/run_vertical_comparison.py'),
            ('launcher_sha256', Path(__file__).resolve())):
        if launch[key] != sha(path):
            raise ValueError('Frozen smoke-to-main source/config differs: '+key)
    pair = validate_pair(smoke)
    return {k: pair['completions'][0]['identity'][k] for k in FROZEN_KEYS}


def child_command(args, arm, cpu):
    cmd = [sys.executable, '-u', str(ROOT/'scripts/v3/run_vertical_comparison.py'),
        '--workspace', str(args.workspace), '--config', str(args.config),
        '--out', str(args.phase), '--arm', arm, '--cpu', str(cpu)]
    return cmd+(['--smoke'] if args.smoke else [])


def supervise(args):
    children, failed, completed = {}, {}, {}
    try:
        save(args.phase/'preflight.json', resources())
        for arm, cpu in zip(ARMS, CPUS):
            with (args.phase/(arm+'.log')).open('xb') as log:
                proc = subprocess.Popen(child_command(args, arm, cpu), cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children[arm] = (proc, cpu)
            save(args.phase/'children.json', {name: dict(pid=child.pid, cpu=core)
                for name, (child, core) in children.items()})
        while children:
            rows = {row['pid']: row for row in processes()}
            active = []
            for arm, (proc, cpu) in list(children.items()):
                code = proc.poll()
                progress_path = args.phase/arm/'progress.json'
                progress = read(progress_path) if progress_path.exists() else None
                if code is None:
                    active.append(dict(arm=arm, pid=proc.pid, cpu=cpu,
                        rss_bytes=rows.get(proc.pid, {}).get('rss_bytes'), progress=progress))
                    continue
                if code != 0 or not (args.phase/arm/'completion.json').exists():
                    failed[arm] = dict(exit_code=code, progress=progress)
                else:
                    completed[arm] = dict(exit_code=code, completion_path=str(args.phase/arm/'completion.json'))
                del children[arm]
            save(args.phase/'progress.json', dict(at=now(), pid=os.getpid(),
                state='draining_after_failure' if failed and active else 'running',
                phase=args.phase.name, active=active, completed=completed, failed=failed))
            if children:
                time.sleep(10)
        if failed:
            raise RuntimeError('Failed arm evidence retained; its peer was allowed to finish')
        pair = validate_pair(args.phase)
        if args.main:
            smoke_identity = smoke_gate(args)
            if any({k: row['identity'][k] for k in FROZEN_KEYS} != smoke_identity for row in pair['completions']):
                raise ValueError('Main frozen identities differ from successful smoke')
        save(args.phase/'pair-summary.json', pair['summary'])
        save(args.phase/'progress.json', dict(at=now(), pid=os.getpid(), state='completed',
            phase=args.phase.name, active=[], completed=completed, failed={}))
    except BaseException:
        # Never terminate an independent peer. Preserve its PID for manual monitoring.
        save(args.phase/'failure.json', dict(at=now(), traceback=traceback.format_exc()))
        save(args.phase/'progress.json', dict(at=now(), pid=os.getpid(),
            state='draining_after_supervisor_failure', phase=args.phase.name,
            active=[dict(arm=arm, pid=proc.pid, cpu=cpu) for arm, (proc, cpu) in children.items()],
            completed=completed, failed=failed))
        for arm, (proc, cpu) in children.items():
            proc.wait()
            if proc.returncode != 0:
                failed[arm] = dict(exit_code=proc.returncode)
            elif (args.phase/arm/'completion.json').exists():
                completed[arm] = dict(exit_code=0, completion_path=str(args.phase/arm/'completion.json'))
        save(args.phase/'progress.json', dict(at=now(), pid=os.getpid(), state='failed',
            phase=args.phase.name, active=[], completed=completed, failed=failed,
            reason='See failure.json; no child was terminated by this launcher'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--smoke', action='store_true')
    phase.add_argument('--main', action='store_true')
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--check', action='store_true', help='Read-only launch/resource checks; do not create outputs')
    args = parser.parse_args()
    args.workspace, args.config, args.out = (p.resolve() for p in (args.workspace, args.config, args.out))
    args.phase = args.out/('smoke' if args.smoke else 'main')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONPATH'] = str(ROOT/'src')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.supervise:
        supervise(args)
        return
    preflight = resources()
    smoke_identity = smoke_gate(args) if args.main else None
    if args.check:
        if args.phase.exists() or (args.smoke and args.out.exists()):
            raise ValueError('The requested new output already exists')
        print(json.dumps(dict(resources=preflight, successful_smoke_identity=smoke_identity)))
        return
    runner = ROOT/'scripts/v3/run_vertical_comparison.py'
    launch = dict(at=now(), phase=args.phase.name, launcher_pid=os.getpid(),
        config=str(args.config), config_sha256=sha(args.config), runner_sha256=sha(runner),
        launcher_sha256=sha(Path(__file__).resolve()), workspace=str(args.workspace),
        frozen_checkout=str(ROOT), initial_resources=preflight, successful_smoke_identity=smoke_identity,
        commands={arm: child_command(args, arm, cpu) for arm, cpu in zip(ARMS, CPUS)})
    if args.smoke:
        args.out.mkdir(parents=True, exist_ok=False)
    args.phase.mkdir(exist_ok=False)
    save(args.phase/'launch.json', launch)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--workspace', str(args.workspace),
        '--config', str(args.config), '--out', str(args.out),
        '--smoke' if args.smoke else '--main', '--supervise']
    with (args.phase/'supervisor.log').open('xb') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch['supervisor_pid'] = proc.pid
    save(args.phase/'launch.json', launch)
    print(json.dumps(dict(supervisor_pid=proc.pid, phase=str(args.phase),
        progress=str(args.phase/'progress.json'), automatically_starts_main=False)))


if __name__ == '__main__':
    main()
