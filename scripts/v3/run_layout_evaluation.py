"""Evaluate one frozen layout-specific model (or NO_REALLOC) in its own environment (YR-317-h2).

One process per environment x policy. The checkpoint is the final-day checkpoint of
that environment's completed training, verified against the training completion record.
This child never trains, never selects a checkpoint and never resumes an output dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
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

from independent_eval_checks import audit_run, month_row
from layout_experiment_io import (ARMS, ENVIRONMENTS, build_eval_input, digest, operational_outcomes,
                                  plan_for, read, recording_checks, save, save_result, sha, spec_digest,
                                  validate_config)


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


def trained_checkpoint(args, cfg):
    """Final-day checkpoint of this environment's completed training, hash-verified."""
    completion = read(args.train / args.environment / 'completion.json')
    if completion['state'] != 'completed' or not completion['checks']['passed']:
        raise ValueError('Training for this environment did not complete its checks')
    if (completion['identity']['source_commit'] != cfg['source_commit']
            or completion['identity']['prereg_sha256'] != cfg['prereg_sha256']):
        raise ValueError('Training evidence belongs to another pinned source or contract')
    path = Path(completion['checks']['final_checkpoint'])
    if not path.is_absolute():
        path = (args.train / args.environment / path).resolve()
    expected = completion['checks']['final_checkpoint_sha256']
    if sha(path) != expected:
        raise ValueError('Final checkpoint changed after training completion')
    return path, expected, completion


def execute(args, cfg, config_hash, prereg, folder):
    import torch
    from yard_rl.integrated.repro import repro_stamp
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import arm_contract, network_identity, runtime_identity
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
    from yard_rl.v3.stage.month_run import run_month

    phase = 'smoke' if args.smoke else 'main'
    started = time.monotonic()
    seed, days = plan_for(cfg, phase, 'eval')
    progress = dict(at=now(), state='preparing', environment=args.environment, arm=args.arm, phase=phase,
                    seed=seed, pid=os.getpid(), cpu=args.cpu, day_index_completed=-1,
                    days_planned=len(days), elapsed_s=0.0)
    save(folder / 'progress.json', progress)
    expected = build_eval_input(seed, days, folder)
    checkpoint, checkpoint_sha, training = trained_checkpoint(args, cfg)
    seller, buyer, model_description = _load_nets(str(checkpoint))
    weights = [network_identity(n) for n in (seller, buyer)]
    runtime = runtime_identity()
    spec = cfg['environments'][args.environment]['environment_spec']
    spec_sha = spec_digest(spec)
    job = dict(_label=f'{args.environment}/{args.arm}', arm=args.arm, seed=seed, days=days,
        seller_net=seller, buyer_net=buyer, workers=1, explore=0,
        admission_mode='PRESERVE', supply_mode='COUNT_BALANCED', capture_requests=True,
        diagnose_admissions=True, capture_daily=True, daily_sample_s=300.0,
        expected_input=expected, environment_spec=spec)
    contract = arm_contract(job, runtime, run_month)
    pair_contract = {**contract, 'label': f'PAIRED_{args.environment.upper()}',
                     'settings': {**contract['settings'], 'arm': f'PAIRED_{args.environment.upper()}'}}
    identity = dict(source_commit=cfg['source_commit'], config_sha256=config_hash,
        prereg_sha256=cfg['prereg_sha256'], checkpoint_sha256=checkpoint_sha,
        training_completion_sha256=sha(args.train / args.environment / 'completion.json'),
        environment_spec_sha256=spec_sha, pair_contract_sha256=digest(pair_contract),
        runtime_source_sha256=runtime['source_and_config_sha256'], network_identities=weights)
    stamp = repro_stamp(experiment='YR-317-h2-layout-evaluation', seeds={'base': [seed],
        'days': [d.seed for d in days]}, params={'run': contract['settings']}, prereg=str(prereg),
        extra={**identity, 'runner_sha256': sha(__file__),
               'helper_sha256': sha(Path(__file__).with_name('layout_experiment_io.py'))})
    if stamp['code']['git_head'] != cfg['source_commit'] or stamp['code']['git_dirty'] is not False:
        raise RuntimeError('Reproduction stamp differs from clean pinned source')
    save(folder / 'manifest.json', dict(started_at=now(), contract=contract, repro=stamp, **identity,
        checkpoint=str(checkpoint), checkpoint_description=model_description,
        training_completion=training, configuration=cfg, expected_input=expected,
        environment_spec=spec, cpu=args.cpu, pid=os.getpid(), smoke=args.smoke, claim_eligible=False,
        purpose='Layout-specific frozen model in its own environment; descriptive single-seed case'))
    save(folder / 'environment-spec.json', spec)
    save(folder / 'canonical-input-hashes.json', expected)
    progress['state'] = 'running'
    last_progress = [0.0]

    def heartbeat(state='running', **extra):
        progress.update(at=now(), state=state, elapsed_s=time.monotonic() - started, **extra)
        save(folder / 'progress.json', progress)
        last_progress[0] = time.monotonic()

    def on_day(day):
        row = dict(at=now(), elapsed_s=time.monotonic() - started, **day.as_dict())
        with (folder / 'days.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        heartbeat(day_index_completed=day.index, truck_skipped=day.truck_skipped,
                  skip_reasons=day.truck_skip_reasons)
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    heartbeat()
    reset_rollout_calls()
    with gzip.open(folder / 'operating-state.jsonl.gz', 'wt', encoding='utf-8') as stream:
        def on_state(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            if row['boundary'] or time.monotonic() - last_progress[0] >= 30:
                stream.flush()
                heartbeat(simulation_time_s=row['at_s'])
        with torch.inference_mode():
            res = run_month(**{k: v for k, v in job.items() if k != '_label'},
                            on_day=on_day, on_observation=on_state)
    heartbeat('saving')
    result = save_result(folder, res, environment=args.environment, arm=args.arm, seed=seed, days=days,
        elapsed_s=time.monotonic() - started, identity=identity, runtime=runtime,
        expected_input=expected, stamp=stamp, rollout_count=rollout_calls())
    result['finished_at'] = now()
    result['recording_checks'] = recording_checks(result, environment=args.environment,
        expected_input=expected, environment_spec_sha256=spec_sha,
        weights_unchanged=weights == [network_identity(n) for n in (seller, buyer)],
        checkpoint_unchanged=sha(checkpoint) == checkpoint_sha,
        runtime_unchanged=runtime == runtime_identity(), rollout_count=rollout_calls())
    result['recording_checks']['frozen_configuration'] = (
        sha(args.config) == config_hash and sha(prereg) == cfg['prereg_sha256'])
    result['operational_outcomes'] = operational_outcomes(result)
    save(folder / 'result.json', result)
    if not all(result['recording_checks'].values()):
        raise RuntimeError('Input/runtime/recording guard failed; raw evidence retained')
    verify_source(cfg['source_commit'])
    heartbeat('auditing')
    audit = audit_run(folder)
    if not audit['passed']:
        raise RuntimeError('Saved evidence audit failed; raw evidence retained')
    completion = dict(at=now(), state='completed', pid=os.getpid(), cpu=args.cpu,
        environment=args.environment, arm=args.arm, seed=seed, input_hashes=expected,
        result_sha256=sha(folder / 'result.json'), manifest_sha256=sha(folder / 'manifest.json'),
        audit=audit, month=month_row(result), operational_outcomes=result['operational_outcomes'],
        identity={**identity, 'requested_identity_sha256': result['requested_identity_sha256'],
                  'runtime_schedule_sha256': (res.environment_manifest or {}).get('runtime_schedule_sha256')})
    save(folder / 'completion.json', completion)
    heartbeat('completed', audit_passed=True, operational_outcomes=result['operational_outcomes'])
    print(json.dumps(completion, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True, help='completed training phase folder')
    parser.add_argument('--out', type=Path, required=True, help='eval phase folder; <out>/<env>/<arm> is created')
    parser.add_argument('--environment', choices=ENVIRONMENTS, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for key in ('workspace', 'config', 'train', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    os.chdir(ROOT)
    cfg, config_hash, prereg = load_frozen(args)
    if (not hasattr(os, 'sched_setaffinity') or not 0 <= args.cpu < 20
            or args.cpu not in os.sched_getaffinity(0)):
        raise RuntimeError('An available Linux CPU inside the authorized 0..19 range is required')
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = args.out / args.environment / args.arm
    folder.mkdir(parents=True, exist_ok=False)
    try:
        execute(args, cfg, config_hash, prereg, folder)
    except BaseException as exc:
        failure = dict(at=now(), state='failed', pid=os.getpid(), cpu=args.cpu,
            environment=args.environment, arm=args.arm, error=repr(exc), traceback=traceback.format_exc())
        save(folder / 'failure.json', failure)
        progress = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
        save(folder / 'progress.json', {**progress, **failure})
        raise


if __name__ == '__main__':
    main()
