"""Train one layout-specific time-only model (YR-317-h2), one process per environment.

The companion launcher supplies the CPU set and pairs the two environments. This
child never evaluates, never selects a checkpoint and never resumes an output dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

from layout_experiment_io import (ENVIRONMENTS, TRAIN_ARM, plan_for, read, save, sha, spec_digest,
                                  training_checks, validate_config)


def now():
    return datetime.now(timezone.utc).isoformat()


def workspace_file(workspace, name):
    path = (workspace / Path(name)).resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        raise ValueError(f'Missing or outside-workspace frozen artifact: {name}')
    return path


def verify_source(expected):
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    if head != expected or dirty:
        raise RuntimeError('A clean checkout of the exact pinned source commit is required')
    return head


def load_frozen(args):
    from yard_rl.v3.layouts import synthetic_vertical_spec
    from yard_rl.v3.world.integrated.profiles import build_h21_profile
    cfg = read(args.config)
    validate_config(cfg, synthetic_vertical_spec(build_h21_profile(), n_blocks=21))
    prereg = workspace_file(args.workspace, cfg['prereg'])
    if sha(prereg) != cfg['prereg_sha256']:
        raise ValueError('The fixed experiment contract changed')
    verify_source(cfg['source_commit'])
    if args.out.is_relative_to(ROOT):
        raise ValueError('Output must be outside the frozen source checkout')
    return cfg, sha(args.config), prereg


def execute(args, cfg, config_hash, prereg, folder):
    import torch
    from yard_rl.integrated.repro import repro_stamp
    from yard_rl.v3.eval.contracts import runtime_identity
    from yard_rl.v3.train.month_loop import run_month_training

    phase = 'smoke' if args.smoke else 'main'
    p = cfg[phase]
    seed, days = plan_for(cfg, phase, 'train')
    spec = cfg['environments'][args.environment]['environment_spec']
    spec_sha = spec_digest(spec)
    workers = max(1, len(args.cpus) - 1)
    started = time.monotonic()
    progress = dict(at=now(), state='preparing', environment=args.environment, phase=phase,
                    seed=seed, pid=os.getpid(), cpus=args.cpus, workers=workers,
                    days_planned=len(days), day_index_completed=-1, elapsed_s=0.0)
    save(folder / 'progress.json', progress)
    runtime = runtime_identity()
    stamp = repro_stamp(experiment='YR-317-h2-layout-training',
        seeds={'train': [seed], 'init': [p['init_seed']], 'days': [d.seed for d in days]},
        params=dict(environment=args.environment, arm=TRAIN_ARM, admission_mode='PRESERVE',
                    supply_mode='COUNT_BALANCED', labels_per_day=p['labels_per_day'],
                    workers=workers, n_days=len(days), loads=[d.load for d in days]),
        prereg=str(prereg),
        extra=dict(source_commit=cfg['source_commit'], config_sha256=config_hash,
                   prereg_sha256=cfg['prereg_sha256'], environment_spec_sha256=spec_sha,
                   runtime_source_sha256=runtime['source_and_config_sha256'],
                   runner_sha256=sha(__file__),
                   helper_sha256=sha(Path(__file__).with_name('layout_experiment_io.py'))))
    if stamp['code']['git_head'] != cfg['source_commit'] or stamp['code']['git_dirty'] is not False:
        raise RuntimeError('Reproduction stamp differs from clean pinned source')
    save(folder / 'manifest.json', dict(started_at=now(), environment=args.environment, phase=phase,
        seed=seed, init_seed=p['init_seed'], labels_per_day=p['labels_per_day'], workers=workers,
        plan=[dict(index=d.index, load=d.load, seed=d.seed, train=d.is_train) for d in days],
        environment_spec=spec, environment_spec_sha256=spec_sha, repro=stamp, runtime=runtime,
        configuration=cfg, cpus=args.cpus, pid=os.getpid(), claim_eligible=False,
        purpose='Layout-specific time-only training; no evaluation or checkpoint selection here'))
    save(folder / 'environment-spec.json', spec)
    log_path = folder / 'train.log'
    train_dir = folder / 'train'

    def log(message):
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(f'{now()} {message}\n')
        done = sorted(train_dir.glob('ckpt_[0-9]*.pt'))
        progress.update(at=now(), state='running', elapsed_s=time.monotonic() - started,
                        day_index_completed=len(done) - 1, last_message=message[:200])
        save(folder / 'progress.json', progress)
        print(message, flush=True)

    progress['state'] = 'running'
    save(folder / 'progress.json', progress)
    torch.set_num_threads(1)
    run_month_training(seed=seed, labels_per_day=p['labels_per_day'], out_dir=train_dir,
                       workers=workers, days=days, log=log, init_seed=p['init_seed'],
                       admission_mode='PRESERVE', arm=TRAIN_ARM, supply_mode='COUNT_BALANCED',
                       environment_spec=spec)
    progress.update(at=now(), state='checking', elapsed_s=time.monotonic() - started)
    save(folder / 'progress.json', progress)
    checks = training_checks(train_dir, env_spec_sha256=spec_sha, n_days=len(days),
                             labels_per_day=p['labels_per_day'], init_seed=p['init_seed'], seed=seed)
    verify_source(cfg['source_commit'])
    completion = dict(at=now(), state='completed', environment=args.environment, phase=phase, seed=seed,
        pid=os.getpid(), cpus=args.cpus, elapsed_s=time.monotonic() - started, checks=checks,
        manifest_sha256=sha(folder / 'manifest.json'), environment_spec_sha256=spec_sha,
        identity=dict(source_commit=cfg['source_commit'], config_sha256=config_hash,
                      prereg_sha256=cfg['prereg_sha256'],
                      runtime_source_sha256=runtime['source_and_config_sha256'],
                      runtime_unchanged=runtime == runtime_identity()))
    save(folder / 'completion.json', completion)
    if not checks['passed'] or not completion['identity']['runtime_unchanged']:
        raise RuntimeError('Training evidence check failed; raw evidence retained')
    progress.update(at=now(), state='completed', elapsed_s=completion['elapsed_s'],
                    day_index_completed=len(days) - 1)
    save(folder / 'progress.json', progress)
    print(json.dumps(completion, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='phase folder; <out>/<environment> is created')
    parser.add_argument('--environment', choices=ENVIRONMENTS, required=True)
    parser.add_argument('--cpus', required=True, help='comma-separated Linux CPUs for main process + branch workers')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for key in ('workspace', 'config', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    args.cpus = sorted({int(x) for x in args.cpus.split(',') if x})
    os.chdir(ROOT)
    cfg, config_hash, prereg = load_frozen(args)
    if (not hasattr(os, 'sched_setaffinity') or not args.cpus
            or not all(0 <= c < 20 for c in args.cpus) or not set(args.cpus) <= os.sched_getaffinity(0)):
        raise RuntimeError('Available Linux CPUs inside the authorized 0..19 range are required')
    os.sched_setaffinity(0, set(args.cpus))
    os.nice(10)
    folder = args.out / args.environment
    folder.mkdir(parents=True, exist_ok=False)
    try:
        execute(args, cfg, config_hash, prereg, folder)
    except BaseException as exc:
        failure = dict(at=now(), state='failed', pid=os.getpid(), environment=args.environment,
                       error=repr(exc), traceback=traceback.format_exc())
        save(folder / 'failure.json', failure)
        progress = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
        save(folder / 'progress.json', {**progress, **failure})
        raise


if __name__ == '__main__':
    main()
