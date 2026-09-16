"""Recover completed raw evidence without changing or selecting the fixed sample."""
from dataclasses import asdict
from pathlib import Path
import shutil

from independent_eval_checks import ARMS, SEEDS, audit_run, month_row, read, save, sha

RAW_FILES = ('manifest.json', 'result.json', 'days.jsonl', 'daily-final.jsonl',
             'requests.jsonl.gz', 'container-links.jsonl.gz', 'operating-state.jsonl.gz')


def require_compatible(manifest, result, contract, checkpoint_hash):
    if manifest['contract'] != contract:
        raise ValueError('Prior execution settings/model/engine differ; do not reuse its result')
    if (manifest['repro'] != result['repro']
            or manifest['repro']['checkpoint_sha256'] != checkpoint_hash
            or manifest['repro']['code']['git_dirty'] is not False):
        raise ValueError('Prior result provenance differs or is not clean')
    if (result['seed'] != contract['settings']['seed']
            or result['arm'] != contract['settings']['arm']
            or result['plan'] != contract['settings']['days']):
        raise ValueError('Prior result does not match its declared seed/policy/days')


def recover_completed(args, cfg):
    import torch
    from run_request_audit import now
    from run_independent_evaluation import expected_month
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import arm_contract, runtime_identity
    from yard_rl.v3.stage.month import plan_month
    from yard_rl.v3.stage.month_run import run_month
    torch.set_num_threads(1)
    source = args.recover_from
    if source == args.out or not source.is_dir():
        raise ValueError('Recovery must read an existing distinct run directory')
    launch = read(source / 'launch.json')
    if launch['config_sha256'] != sha(args.config):
        raise ValueError('Recovery config differs from the original frozen config')
    seller, buyer, _ = _load_nets(str(args.workspace / cfg['checkpoint']))
    runtime = runtime_identity()
    completed, attempts = [], []
    for seed in SEEDS:
        for arm in ARMS:
            old = source / 'months' / str(seed) / arm
            if not (old / 'result.json').exists():
                attempts.append(dict(seed=seed, arm=arm, action='restart' if old.exists() else 'not_started'))
                continue
            result, manifest = read(old / 'result.json'), read(old / 'manifest.json')
            expected = expected_month(args, cfg, seed)
            job = dict(_label=arm, arm=arm, seed=seed, days=plan_month(seed),
                seller_net=seller, buyer_net=buyer, capture_requests=True, capture_daily=True,
                daily_sample_s=300.0, admission_mode='PRESERVE', supply_mode='COUNT_BALANCED',
                diagnose_admissions=True, expected_input=expected)
            contract = arm_contract(job, runtime, run_month)
            require_compatible(manifest, result, contract, cfg['checkpoint_sha256'])
            folder = args.out / 'months' / str(seed) / arm
            folder.mkdir(parents=True, exist_ok=False)
            hashes = {name: sha(old / name) for name in RAW_FILES}
            for name in RAW_FILES:
                shutil.copy2(old / name, folder / name)
                if sha(folder / name) != hashes[name]:
                    raise ValueError('Recovery copy changed a raw artifact')
            audit = audit_run(folder)
            if not audit['passed']:
                raise ValueError('Re-audit of completed run failed; do not silently discard or replace it')
            value = dict(at=now(), state='completed', reused=True,
                original_folder=str(old), original_source_commit=result['repro']['code']['git_head'],
                raw_artifact_hashes=hashes, input_hashes=expected,
                result_sha256=sha(folder / 'result.json'), audit=audit, month=month_row(result))
            save(folder / 'completion.json', value)
            completed.append(value)
            attempts.append(dict(seed=seed, arm=arm, action='reused_after_full_reaudit'))
    save(args.out / 'recovery.json', dict(at=now(), previous_run=str(source),
        retained=len(completed), attempts=attempts, original_artifacts_modified=False,
        no_training_or_seed_replacement=True))
    return completed
