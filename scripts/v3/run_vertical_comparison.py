"""One frozen vertical policy per process, using the existing monthly contract.

The companion launcher supplies free CPUs and pairs processes. This child never
starts another simulation, trains, selects a seed, or resumes an existing output.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

from independent_eval_checks import audit_run, month_row, read, save, sha
from vertical_comparison_io import (ARMS, MAIN_SEED, SMOKE_SEED, claim_output,
    digest, operational_outcomes, recording_checks, save_result, validate_config)


def now():
    return datetime.now(timezone.utc).isoformat()


def workspace_file(workspace, name):
    path = (workspace/Path(name)).resolve()
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
    from run_independent_evaluation import verified_config
    from yard_rl.v3.layouts import synthetic_vertical_spec
    from yard_rl.v3.world.integrated.profiles import build_h21_profile
    config_hash = sha(args.config)
    cfg = read(args.config)
    base_path = workspace_file(args.workspace, cfg['base_config'])
    if sha(base_path) != cfg['base_config_sha256']:
        raise ValueError('Original independent configuration changed')
    base = verified_config(SimpleNamespace(config=base_path, workspace=args.workspace))
    validate_config(cfg, base, synthetic_vertical_spec(build_h21_profile(), n_blocks=21))
    prereg = workspace_file(args.workspace, cfg['prereg'])
    if sha(prereg) != cfg['prereg_sha256']:
        raise ValueError('The new fixed comparison contract changed')
    verify_source(cfg['source_commit'])
    if args.out.is_relative_to(ROOT):
        raise ValueError('Output must be outside the frozen source checkout')
    return cfg, base, config_hash


def prepare_inputs(args, cfg, base, folder):
    from run_independent_evaluation import expected_month
    from yard_rl.v3.eval.seed_bank import create_month_payload, validate_payload, write_bundle
    from yard_rl.v3.stage.month import plan_days, plan_month
    from yard_rl.v3.stage.supply_plan import balance_vessel_supply
    if not args.smoke:
        expected = expected_month(SimpleNamespace(workspace=args.workspace), base, MAIN_SEED)
        return MAIN_SEED, plan_month(MAIN_SEED), expected
    days = plan_days(SMOKE_SEED, (60, 60))
    payload = create_month_payload(SMOKE_SEED, days=days)
    validation = validate_payload(payload, require_month=False)
    initial = {b: len(s['containers']) for b, s in payload['initial_scenarios'].items()}
    vessels, supply = balance_vessel_supply(payload['vessels'], payload['schedule'],
                                           initial, {b: 1440 for b in initial})
    expected = dict(schedule_sha256=digest(payload['schedule']),
        initial_scenarios_sha256=digest(payload['initial_scenarios']), vessels_sha256=digest(vessels))
    save(folder/'diagnostic-input.json', dict(validation=validation, supply_plan_audit=supply,
        canonical_bundle_sha256=write_bundle(folder/'canonical-input.json.gz', payload),
        balanced_vessels_sha256=write_bundle(folder/'balanced-vessels.json.gz', vessels)))
    return SMOKE_SEED, days, expected


def execute(args, cfg, base, config_hash, folder):
    import torch
    from yard_rl.integrated.repro import repro_stamp
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import arm_contract, network_identity, runtime_identity
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
    from yard_rl.v3.stage.month_run import run_month

    started = time.monotonic()
    progress = dict(at=now(), state='preparing', arm=args.arm,
        seed=SMOKE_SEED if args.smoke else MAIN_SEED, pid=os.getpid(), cpu=args.cpu,
        day_index_completed=-1, days_planned=2 if args.smoke else 30, elapsed_s=0.0)
    save(folder/'progress.json', progress)
    seed, days, expected = prepare_inputs(args, cfg, base, folder)
    checkpoint = workspace_file(args.workspace, base['checkpoint'])
    if sha(checkpoint) != base['checkpoint_sha256']:
        raise ValueError('Checkpoint changed before loading')
    seller, buyer, model_description = _load_nets(str(checkpoint))
    weights = [network_identity(n) for n in (seller, buyer)]
    runtime = runtime_identity()
    spec = cfg['environment_spec']
    job = dict(_label=args.arm, arm=args.arm, seed=seed, days=days,
        seller_net=seller, buyer_net=buyer, workers=1, explore=0,
        admission_mode='PRESERVE', supply_mode='COUNT_BALANCED', capture_requests=True,
        diagnose_admissions=True, capture_daily=True, daily_sample_s=300.0,
        expected_input=expected, environment_spec=spec)
    contract = arm_contract(job, runtime, run_month)
    pair_contract = {**contract, 'label': 'PAIRED_VERTICAL',
                     'settings': {**contract['settings'], 'arm': 'PAIRED_VERTICAL'}}
    identity = dict(source_commit=cfg['source_commit'], config_sha256=config_hash,
        base_config_sha256=cfg['base_config_sha256'], prereg_sha256=cfg['prereg_sha256'],
        checkpoint_sha256=base['checkpoint_sha256'], environment_spec_sha256=digest(spec),
        pair_contract_sha256=digest(pair_contract),
        runtime_source_sha256=runtime['source_and_config_sha256'], network_identities=weights)
    prereg = workspace_file(args.workspace, cfg['prereg'])
    stamp = repro_stamp(experiment='YR-317-h-vertical-transfer', seeds={'base': [seed],
        'days': [d.seed for d in days]}, params={'run': contract['settings']}, prereg=str(prereg),
        extra={**identity, 'runner_sha256': sha(__file__),
               'helper_sha256': sha(Path(__file__).with_name('vertical_comparison_io.py'))})
    if stamp['code']['git_head'] != cfg['source_commit'] or stamp['code']['git_dirty'] is not False:
        raise RuntimeError('Reproduction stamp differs from clean pinned source')
    manifest = dict(started_at=now(), contract=contract, repro=stamp, **identity,
        checkpoint=str(checkpoint), checkpoint_description=model_description,
        configuration=cfg, base_configuration_path=str(args.workspace/cfg['base_config']),
        expected_input=expected, environment_spec=spec, cpu=args.cpu, pid=os.getpid(),
        smoke=args.smoke, claim_eligible=False, purpose='Predefined single-case vertical transfer diagnostic')
    save(folder/'manifest.json', manifest)
    save(folder/'environment-spec.json', spec)
    save(folder/'canonical-input-hashes.json', expected)
    progress['state'] = 'running'
    last_progress = [0.0]

    def heartbeat(state='running', **extra):
        progress.update(at=now(), state=state, elapsed_s=time.monotonic()-started, **extra)
        save(folder/'progress.json', progress)
        last_progress[0] = time.monotonic()

    def on_day(day):
        row = dict(at=now(), elapsed_s=time.monotonic()-started, **day.as_dict())
        with (folder/'days.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        heartbeat(day_index_completed=day.index, truck_skipped=day.truck_skipped,
                  skip_reasons=day.truck_skip_reasons)
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    heartbeat()
    reset_rollout_calls()
    with gzip.open(folder/'operating-state.jsonl.gz', 'wt', encoding='utf-8') as stream:
        def on_state(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            if row['boundary'] or time.monotonic()-last_progress[0] >= 30:
                stream.flush()
                heartbeat(simulation_time_s=row['at_s'])
        with torch.inference_mode():
            res = run_month(**{k: v for k, v in job.items() if k != '_label'},
                            on_day=on_day, on_observation=on_state)
    heartbeat('saving')
    result = save_result(folder, res, arm=args.arm, seed=seed, days=days,
        elapsed_s=time.monotonic()-started, identity=identity, runtime=runtime,
        expected_input=expected, stamp=stamp, rollout_count=rollout_calls())
    result['finished_at'] = now()
    result['recording_checks'] = recording_checks(result, expected_input=expected,
        environment_spec_sha256=digest(spec),
        weights_unchanged=weights == [network_identity(n) for n in (seller, buyer)],
        checkpoint_unchanged=sha(checkpoint) == base['checkpoint_sha256'],
        runtime_unchanged=runtime == runtime_identity(), rollout_count=rollout_calls())
    result['recording_checks']['frozen_configuration'] = (
        sha(args.config) == config_hash and sha(prereg) == cfg['prereg_sha256']
        and sha(args.workspace/cfg['base_config']) == cfg['base_config_sha256'])
    result['operational_outcomes'] = operational_outcomes(result)
    save(folder/'result.json', result)
    if not all(result['recording_checks'].values()):
        raise RuntimeError('Input/runtime/recording guard failed; raw evidence retained')
    verify_source(cfg['source_commit'])
    heartbeat('auditing')
    audit = audit_run(folder)
    if not audit['passed']:
        raise RuntimeError('Saved evidence audit failed; raw evidence retained')
    completion = dict(at=now(), state='completed', pid=os.getpid(), cpu=args.cpu,
        input_hashes=expected, result_sha256=sha(folder/'result.json'),
        manifest_sha256=sha(folder/'manifest.json'), audit=audit, month=month_row(result),
        operational_outcomes=result['operational_outcomes'], identity={**identity,
            'requested_identity_sha256': result['requested_identity_sha256'],
            'runtime_schedule_sha256': res.environment_manifest['runtime_schedule_sha256']})
    save(folder/'completion.json', completion)
    heartbeat('completed', audit_passed=True, operational_outcomes=result['operational_outcomes'])
    print(json.dumps(completion, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for key in ('workspace', 'config', 'out'):
        setattr(args, key, getattr(args, key).resolve())
    os.chdir(ROOT)
    cfg, base, config_hash = load_frozen(args)
    if (not hasattr(os, 'sched_setaffinity') or not 0 <= args.cpu < 20
            or args.cpu not in os.sched_getaffinity(0)):
        raise RuntimeError('An available Linux CPU inside the authorized 0..19 range is required')
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = claim_output(args.out, args.arm)
    try:
        execute(args, cfg, base, config_hash, folder)
    except BaseException as exc:
        failure = dict(at=now(), state='failed', pid=os.getpid(), cpu=args.cpu,
            arm=args.arm, error=repr(exc), traceback=traceback.format_exc())
        save(folder/'failure.json', failure)
        progress = read(folder/'progress.json') if (folder/'progress.json').exists() else {}
        save(folder/'progress.json', {**progress, **failure})
        raise


if __name__ == '__main__':
    main()
