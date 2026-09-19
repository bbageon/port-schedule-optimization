"""Resume a quiescent Block-only queue without rerunning or selecting outcomes.

--preflight performs read-only validation while the old supervisor may still live.
Actual recovery requires that supervisor to have stopped and uses a new output.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from independent_eval_checks import SEEDS, read, save, sha


def process_alive(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid previous supervisor PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def source_child_pids(source, *, proc_root=Path('/proc')):
    """Find orphan workers too; a supervisor's last progress can predate a fork."""
    if not proc_root.is_dir():
        raise RuntimeError('Recovery takeover requires Linux process inspection')
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            argv = os.fsdecode((entry/'cmdline').read_bytes()).split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if ('--seed' not in argv or '--out' not in argv
                or not any(Path(value).name == 'run_block_only_evaluation.py' for value in argv)):
            continue
        output_index = argv.index('--out') + 1
        if output_index >= len(argv) or not argv[output_index]:
            raise ValueError('Cannot establish an existing Block-only worker output')
        if Path(argv[output_index]).resolve() == source.resolve():
            found.append(int(entry.name))
    return sorted(found)


def source_state(source, out, config_hash, *, require_stopped):
    if (source == out or source.is_relative_to(out) or out.is_relative_to(source)
            or not source.is_dir()):
        raise ValueError('Recovery requires separate source and output directories')
    if (source / 'failure.json').exists():
        raise ValueError('Previous run has a failure marker; no automatic recovery')
    launch, status = read(source / 'launch.json'), read(source / 'progress.json')
    if launch['config_sha256'] != config_hash:
        raise ValueError('Recovery config differs from the original frozen config')
    if (status.get('state') != 'waiting_for_resources' or status.get('phase') != 'months'
            or status.get('active') != [] or status.get('failed') != []
            or status.get('planned') != len(SEEDS)):
        raise ValueError('Only a healthy, idle monthly queue can be recovered')
    if require_stopped and process_alive(launch['pid']):
        raise ValueError('Previous supervisor must stop before queue takeover')
    if require_stopped and (children := source_child_pids(source)):
        raise ValueError(f'Previous output still has live workers: {children}')
    return launch, status


def tree_hashes(folder):
    paths = sorted(folder.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Recovery source must contain real files, not links')
    if any(p.name == 'failure.json' or p.suffix == '.tmp' for p in paths):
        raise ValueError('Recovery source contains failure or incomplete write evidence')
    return {p.relative_to(folder).as_posix(): sha(p) for p in paths if p.is_file()}


def classify_months(source, status):
    """Reject any attempted-but-incomplete month; never silently restart one."""
    months = source / 'months'
    if not months.is_dir():
        raise ValueError('Previous monthly evidence directory is missing')
    found = {p.name for p in months.iterdir()}
    if not found <= {str(s) for s in SEEDS}:
        raise ValueError('Unexpected monthly run outside the fixed seed list')
    completed, pending = [], []
    for seed in SEEDS:
        folder = months / str(seed)
        if not folder.exists():
            pending.append(seed)
            continue
        if (not folder.is_dir() or not (folder / 'RL_SPACE.log').is_file()
                or {p.name for p in folder.iterdir()} != {'RL_SPACE', 'RL_SPACE.log'}
                or not (folder / 'RL_SPACE/completion.json').is_file()):
            raise ValueError(f'Partial or unexpected month {seed}; preserve and inspect it')
        completed.append(seed)
    if (status.get('completed') != len(completed) or status.get('pending') != len(pending)
            or not completed or not pending):
        raise ValueError('Previous progress does not match completed and unstarted seeds')
    return completed, pending


def recovery_plan(args, cfg, old, *, require_stopped):
    """Read-only validation of all finished evidence against current frozen inputs."""
    import torch
    from block_only_comparison import _validated_run
    from recover_independent_evaluation import require_compatible
    from run_independent_evaluation import expected_month
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import arm_contract, runtime_identity
    from yard_rl.v3.stage.month import plan_month
    from yard_rl.v3.stage.month_run import run_month
    torch.set_num_threads(1)
    source, out = args.recover_from, args.out
    launch, status = source_state(source, out, sha(args.config), require_stopped=require_stopped)
    if (launch['checkpoint_sha256'] != old['checkpoint_sha256']
            or launch['primary_run'] != cfg['primary_run']
            or launch['planned_independent_runs'] != len(SEEDS)
            or sha(args.workspace / old['checkpoint']) != old['checkpoint_sha256']):
        raise ValueError('Previous launch or current checkpoint differs from the fixed design')
    if (out / 'months').exists() or (out / 'recovery.json').exists():
        raise ValueError('Recovery output already contains run evidence')
    completed, pending = classify_months(source, status)
    runtime = runtime_identity()
    reference = read(args.workspace / cfg['reference_manifest'])
    compatible = read(source / 'source-compatibility.json')
    smoke = read(source / 'smoke-summary.json')
    if (runtime != reference['contract']['runtime'] or compatible.get('passed') is not True
            or compatible['runtime'] != runtime or smoke.get('passed') is not True
            or smoke.get('same_requests_as_primary') is not True
            or smoke.get('validation', {}).get('passed') is not True):
        raise ValueError('Previous smoke or current runtime is incompatible')
    smoke_folder = source / 'smoke/9900722/RL_SPACE'
    smoke_result, smoke_done = read(smoke_folder/'result.json'), read(smoke_folder/'completion.json')
    if (smoke_done['state'] != 'completed' or smoke_done['audit']['passed'] is not True
            or smoke_done['result_sha256'] != sha(smoke_folder/'result.json')
            or smoke_result['time'] != 0 or smoke_result['space'] <= 0):
        raise ValueError('Previous smoke result is not complete and valid')
    seller, buyer, _ = _load_nets(str(args.workspace / old['checkpoint']))
    retained, evidence = [], []
    for seed in completed:
        folder = source / 'months' / str(seed) / 'RL_SPACE'
        _, result, manifest, validated = _validated_run(folder, seed, 'RL_SPACE')
        expected = expected_month(args, old, seed)
        contract = arm_contract(dict(_label='RL_SPACE', arm='RL_SPACE', seed=seed,
            days=plan_month(seed), seller_net=seller, buyer_net=buyer, capture_requests=True,
            capture_daily=True, daily_sample_s=300.0, admission_mode='PRESERVE',
            supply_mode='COUNT_BALANCED', diagnose_admissions=True, expected_input=expected),
            runtime, run_month)
        require_compatible(manifest, result, contract, old['checkpoint_sha256'])
        completion = read(folder/'completion.json')
        if completion['input_hashes'] != expected:
            raise ValueError('Completed month differs from current frozen input bundles')
        retained.append(completion)
        evidence.append(dict(seed=seed, validation=validated,
                             files=tree_hashes(folder.parent)))
    # Preserve original logs/receipts separately; never replace the new launch record.
    receipt_files = {p.name: sha(p) for p in source.iterdir() if p.is_file()}
    return dict(previous_run=str(source), previous_pid=launch['pid'],
        completed=completed, pending=pending, retained=retained, evidence=evidence,
        receipt_files=receipt_files, smoke_files=tree_hashes(source/'smoke'),
        source_launch_sha256=sha(source/'launch.json'), original_artifacts_modified=False,
        no_training_or_seed_replacement=True)


def copy_checked(source, target, hashes):
    if target.exists():
        raise ValueError('Recovery copy must not overwrite existing evidence')
    target.mkdir(parents=True, exist_ok=False)
    for name, expected in hashes.items():
        old, new = source / name, target / name
        if sha(old) != expected:
            raise ValueError('Previous evidence changed after recovery validation')
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
        if sha(new) != expected:
            raise ValueError('Recovery copy changed an artifact')


def recover_completed(args, cfg, old):
    from run_request_audit import now
    plan = recovery_plan(args, cfg, old, require_stopped=True)
    source = args.recover_from
    copy_checked(source, args.out/'previous-run-evidence', plan['receipt_files'])
    copy_checked(source/'smoke', args.out/'smoke', plan['smoke_files'])
    shutil.copy2(source/'smoke-summary.json', args.out/'smoke-summary.json')
    for item in plan['evidence']:
        relative = Path('months') / str(item['seed'])
        copy_checked(source/relative, args.out/relative, item['files'])
    # Keep original completion/audit bytes; recovery lineage lives in this receipt.
    save(args.out/'recovery.json', dict(at=now(), **plan))
    return plan['retained']


def main():
    from run_request_audit import ROOT
    from run_block_only_evaluation import verified_config
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'config', 'out', 'recover-from'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--preflight', action='store_true', required=True)
    args = parser.parse_args()
    for name in ('workspace', 'config', 'out', 'recover_from'):
        setattr(args, name, getattr(args, name).resolve())
    os.chdir(ROOT)
    cfg, old = verified_config(args)
    plan = recovery_plan(args, cfg, old, require_stopped=False)
    print(json.dumps(dict(passed=True, scope='read-only; old supervisor may still be alive',
        completed=plan['completed'], pending=plan['pending'], source_launch_sha256=plan['source_launch_sha256'],
        verified_completed_months=len(plan['retained']), files=sum(len(e['files']) for e in plan['evidence']))))


if __name__ == '__main__':
    main()
