"""Launch layout-specific training then same-seed evaluation (YR-317-h2).

Phases are explicit and separate: `--smoke` runs a tiny train+eval in both environments;
`--main` runs the registered 7-day train+eval and requires a completed, hash-identical smoke.
Inside a phase the supervisor runs the two trainings in parallel, then the four evaluations.
It never kills a peer on failure and never starts another phase by itself.
"""
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
ENVIRONMENTS = ('legacy', 'vertical')
ARMS = ('NO_REALLOC', 'RL_TIME')
GIB = 1024**3
FROZEN_KEYS = ('source_commit', 'config_sha256', 'prereg_sha256', 'runtime_source_sha256')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def processes():
    rows = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            argv = [s.decode('utf8', errors='replace') for s in (folder / 'cmdline').read_bytes().split(b'\0') if s]
            status = dict(line.split(':', 1) for line in (folder / 'status').read_text().splitlines() if ':' in line)
            rows.append(dict(pid=int(folder.name), ppid=int(status['PPid']), argv=argv,
                rss_bytes=int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
                cpus=sorted(os.sched_getaffinity(int(folder.name)))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rows


def option(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def same_output(row, path):
    value = option(row['argv'], '--out')
    return value is not None and Path(value).resolve() == path.resolve()


def cpu_plan(cfg):
    res = cfg['resources']
    train = {env: sorted(int(c) for c in res['train_cpus'][env]) for env in ENVIRONMENTS}
    evaluation = {(env, arm): int(res['eval_cpus'][f'{env}/{arm}']) for env in ENVIRONMENTS for arm in ARMS}
    return train, evaluation


def resources(cfg, needed_cpus):
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Launch inside WSL/Linux')
    mem_raw = Path('/proc/meminfo').read_text()
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in mem_raw.splitlines()}
    rows = processes()
    names = {'run_independent_evaluation.py', 'run_block_only_evaluation.py', 'run_vertical_comparison.py',
             'run_layout_training.py', 'run_layout_evaluation.py'}
    def worker(row):
        return bool(names.intersection(Path(arg).name for arg in row['argv'])) and (
            '--seed' in row['argv'] or '--arm' in row['argv'] or '--environment' in row['argv'])
    v3 = [row for row in rows if worker(row)]
    busy = set()
    for row in rows:
        if len(row['cpus']) <= 16 and (row['rss_bytes'] >= GIB // 8 or worker(row)):
            busy |= set(row['cpus'])
    conflicts = sorted(busy & set(needed_cpus))
    checks = dict(memory_available=mem['MemAvailable'] >= cfg['resources']['memory_reserve_gib'] * GIB,
        requested_cores_free=not conflicts,
        total_within_limit=len(busy | set(needed_cpus)) <= cfg['resources']['cpu_limit'],
        cpus_available=set(needed_cpus) <= os.sched_getaffinity(0))
    result = dict(at=now(), checks=checks, meminfo=mem, needed_cpus=sorted(needed_cpus),
        busy_cpus=sorted(busy), conflicting_cpus=conflicts, v3_workers=len(v3),
        known_processes=[row for row in rows if any('/scripts/v3/' in a or '/scripts/v5/' in a for a in row['argv'])])
    if not all(checks.values()):
        raise RuntimeError('Resource preflight failed: ' + json.dumps(result, ensure_ascii=False))
    return result


def wait_children(children, phase_dir, label):
    failed, completed = {}, {}
    while children:
        rows = {row['pid']: row for row in processes()}
        active = []
        for key, (proc, cpus, done_path) in list(children.items()):
            code = proc.poll()
            progress_path = done_path.parent / 'progress.json'
            progress = read(progress_path) if progress_path.exists() else None
            if code is None:
                active.append(dict(job=key, pid=proc.pid, cpus=cpus,
                    rss_bytes=rows.get(proc.pid, {}).get('rss_bytes'), progress=progress))
                continue
            if code != 0 or not done_path.exists():
                failed[key] = dict(exit_code=code, progress=progress)
            else:
                completed[key] = dict(exit_code=code, completion_path=str(done_path))
            del children[key]
        save(phase_dir / 'progress.json', dict(at=now(), pid=os.getpid(), stage=label,
            state='draining_after_failure' if failed and active else 'running',
            active=active, completed=completed, failed=failed))
        if children:
            time.sleep(10)
    return completed, failed


def spawn(cmd, log_path, cwd):
    with log_path.open('xb') as log:
        return subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def train_command(args, env, cpus):
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/v3/run_layout_training.py'), '--workspace', str(args.workspace),
           '--config', str(args.config), '--out', str(args.phase / 'train'), '--environment', env,
           '--cpus', ','.join(map(str, cpus))]
    return cmd + (['--smoke'] if args.smoke else [])


def eval_command(args, env, arm, cpu):
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/v3/run_layout_evaluation.py'), '--workspace', str(args.workspace),
           '--config', str(args.config), '--train', str(args.phase / 'train'), '--out', str(args.phase / 'eval'),
           '--environment', env, '--arm', arm, '--cpu', str(cpu)]
    return cmd + (['--smoke'] if args.smoke else [])


def validate_phase(args, cfg):
    sys.path.insert(0, str(ROOT / 'scripts/v3'))
    from layout_experiment_io import pair_summary
    n_days = len(cfg['smoke']['loads']) if args.smoke else cfg['main']['n_days']
    summary, trainings = {}, {}
    for env in ENVIRONMENTS:
        t = read(args.phase / 'train' / env / 'completion.json')
        if t['state'] != 'completed' or not t['checks']['passed'] or (args.phase / 'train' / env / 'failure.json').exists():
            raise ValueError('Training completion check failed: ' + env)
        trainings[env] = t
        results, completions = [], []
        for arm in ARMS:
            base = args.phase / 'eval' / env / arm
            receipt = read(base / 'completion.json')
            if (receipt['state'] != 'completed' or receipt['audit']['passed'] is not True
                    or receipt['result_sha256'] != sha(base / 'result.json')
                    or receipt['manifest_sha256'] != sha(base / 'manifest.json')
                    or (base / 'failure.json').exists()):
                raise ValueError(f'Completion audit/hash failed: {env}/{arm}')
            if receipt['identity']['checkpoint_sha256'] != t['checks']['final_checkpoint_sha256']:
                raise ValueError(f'Evaluation used a checkpoint other than the final training day: {env}/{arm}')
            completions.append(receipt)
            results.append(read(base / 'result.json'))
        if completions[0]['identity'] != completions[1]['identity']:
            raise ValueError('Paired completion identities differ: ' + env)
        summary[env] = pair_summary(env, results, n_days=n_days)
    return dict(passed=True, phase=args.phase.name, n_days=n_days, environments=summary,
        training={env: dict(elapsed_s=t['elapsed_s'], total_labels=t['checks']['total_labels'],
                            optimizer_steps=t['checks']['optimizer_steps'], fit_days=t['checks']['fit_days'],
                            final_checkpoint_sha256=t['checks']['final_checkpoint_sha256'])
                  for env, t in trainings.items()},
        claim_eligible=False, statistical_inference=False,
        purpose='Each environment: its own trained time-only weights vs NO_REALLOC on the shared evaluation seed')


def smoke_gate(args):
    smoke = args.out / 'smoke'
    launch, progress = read(smoke / 'launch.json'), read(smoke / 'progress.json')
    if progress['state'] != 'completed':
        raise ValueError('The smoke phase must finish successfully before the explicit main launch')
    if any(same_output(row, smoke / 'train') or same_output(row, smoke / 'eval') for row in processes()):
        raise ValueError('A smoke process remains alive')
    for key, path in (('config_sha256', args.config),
                      ('trainer_sha256', ROOT / 'scripts/v3/run_layout_training.py'),
                      ('evaluator_sha256', ROOT / 'scripts/v3/run_layout_evaluation.py'),
                      ('launcher_sha256', Path(__file__).resolve())):
        if launch[key] != sha(path):
            raise ValueError('Frozen smoke-to-main source/config differs: ' + key)
    summary = read(smoke / 'summary.json')
    if not summary['passed']:
        raise ValueError('Smoke summary did not pass')
    receipt = read(smoke / 'eval' / 'legacy' / 'NO_REALLOC' / 'completion.json')
    return {k: receipt['identity'][k] for k in FROZEN_KEYS}


def supervise(args, cfg):
    train_cpus, eval_cpus = cpu_plan(cfg)
    children = {}
    try:
        save(args.phase / 'preflight.json', resources(cfg, {c for cs in train_cpus.values() for c in cs}))
        (args.phase / 'train').mkdir(exist_ok=False)
        for env in ENVIRONMENTS:
            proc = spawn(train_command(args, env, train_cpus[env]), args.phase / 'train' / (env + '.log'), ROOT)
            children[f'train/{env}'] = (proc, train_cpus[env], args.phase / 'train' / env / 'completion.json')
        save(args.phase / 'children.json', {k: dict(pid=p.pid, cpus=c) for k, (p, c, _) in children.items()})
        completed, failed = wait_children(children, args.phase, 'train')
        if failed:
            raise RuntimeError('Training failed; evidence retained: ' + json.dumps(failed))
        save(args.phase / 'preflight-eval.json', resources(cfg, set(eval_cpus.values())))
        (args.phase / 'eval').mkdir(exist_ok=False)
        for (env, arm), cpu in eval_cpus.items():
            (args.phase / 'eval' / env).mkdir(exist_ok=True)
            proc = spawn(eval_command(args, env, arm, cpu), args.phase / 'eval' / env / (arm + '.log'), ROOT)
            children[f'eval/{env}/{arm}'] = (proc, [cpu], args.phase / 'eval' / env / arm / 'completion.json')
        save(args.phase / 'children.json', {k: dict(pid=p.pid, cpus=c) for k, (p, c, _) in children.items()})
        completed_eval, failed = wait_children(children, args.phase, 'eval')
        if failed:
            raise RuntimeError('Evaluation failed; peer evidence retained: ' + json.dumps(failed))
        summary = validate_phase(args, cfg)
        if args.main:
            smoke_identity = smoke_gate(args)
            for env in ENVIRONMENTS:
                receipt = read(args.phase / 'eval' / env / 'NO_REALLOC' / 'completion.json')
                if {k: receipt['identity'][k] for k in FROZEN_KEYS} != smoke_identity:
                    raise ValueError('Main frozen identities differ from the successful smoke')
        save(args.phase / 'summary.json', summary)
        save(args.phase / 'progress.json', dict(at=now(), pid=os.getpid(), stage='done', state='completed',
            completed={**completed, **completed_eval}, failed={}))
    except BaseException:
        save(args.phase / 'failure.json', dict(at=now(), traceback=traceback.format_exc()))
        save(args.phase / 'progress.json', dict(at=now(), pid=os.getpid(), state='draining_after_supervisor_failure',
            active=[dict(job=k, pid=p.pid, cpus=c) for k, (p, c, _) in children.items()]))
        for key, (proc, cpus, done_path) in children.items():
            proc.wait()
        save(args.phase / 'progress.json', dict(at=now(), pid=os.getpid(), state='failed',
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
    parser.add_argument('--check', action='store_true', help='Read-only launch/resource checks; create nothing')
    args = parser.parse_args()
    args.workspace, args.config, args.out = (p.resolve() for p in (args.workspace, args.config, args.out))
    args.phase = args.out / ('smoke' if args.smoke else 'main')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONPATH'] = str(ROOT / 'src')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cfg = read(args.config)
    if args.supervise:
        supervise(args, cfg)
        return
    train_cpus, _ = cpu_plan(cfg)
    preflight = resources(cfg, {c for cs in train_cpus.values() for c in cs})
    smoke_identity = smoke_gate(args) if args.main else None
    if args.check:
        if args.phase.exists():
            raise ValueError('The requested phase output already exists')
        print(json.dumps(dict(resources=preflight, successful_smoke_identity=smoke_identity)))
        return
    launch = dict(at=now(), phase=args.phase.name, launcher_pid=os.getpid(), config=str(args.config),
        config_sha256=sha(args.config), trainer_sha256=sha(ROOT / 'scripts/v3/run_layout_training.py'),
        evaluator_sha256=sha(ROOT / 'scripts/v3/run_layout_evaluation.py'),
        launcher_sha256=sha(Path(__file__).resolve()), workspace=str(args.workspace), frozen_checkout=str(ROOT),
        initial_resources=preflight, successful_smoke_identity=smoke_identity,
        train_commands={env: train_command(args, env, cpus) for env, cpus in train_cpus.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    args.phase.mkdir(exist_ok=False)
    save(args.phase / 'launch.json', launch)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--workspace', str(args.workspace),
           '--config', str(args.config), '--out', str(args.out), '--smoke' if args.smoke else '--main', '--supervise']
    with (args.phase / 'supervisor.log').open('xb') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    launch['supervisor_pid'] = proc.pid
    save(args.phase / 'launch.json', launch)
    print(json.dumps(dict(supervisor_pid=proc.pid, phase=str(args.phase),
                          progress=str(args.phase / 'progress.json'), automatically_starts_main=False)))


if __name__ == '__main__':
    main()
