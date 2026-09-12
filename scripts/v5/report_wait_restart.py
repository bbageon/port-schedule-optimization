"""Verify and archive a live YR-306 admission-wait restart, not a final verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from yard_rl.experiments.gate_harness import attach_common_gates, report_from_dict

SOURCE = '4549ae97db4aad834cad02698fad8c8b8c880f99'
SEED_SHA = 'efdb45e2ec8feafed6987cf48dfe022ca2959dad8c4f1d56bee680456081af7a'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('smoke', 'run', 'source', 'report'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--launcher-pid', type=int, required=True)
    args = ap.parse_args()
    manifest, smoke = read(args.run/'manifest.json'), read(args.smoke/'report.json')
    smoke_manifest = read(args.smoke/'manifest.json')
    smoke_adm = read(args.smoke/'admissions.json')
    old = read(Path('outputs/v5/yr306-cargo-30d-83de18d/manifest.json'))
    for code in (manifest['code'], smoke_manifest['code']):
        require(code['git_head'] == SOURCE and code['git_dirty'] is False, 'Unpinned source')
        require(code['torch_threads'] == 1, 'Unexpected thread count')
    require(manifest['code']['source_sha256'] == smoke['code']['source_sha256'], 'Source mismatch')
    for key in ('seed', 'days', 'ppo', 'learning_window_s', 'initial_occupancy',
                'action_mode', 'counterfactual_worlds_allowed', 'fixed_seed'):
        require(manifest[key] == old[key], 'Changed original setting: ' + key)
    require(manifest['fixed_seed']['sha256'] == SEED_SHA, 'Unexpected input hash')
    require(sha(Path('outputs/reports/yr306_seed_regeneration/seed-9900306/seed-data.json.gz'))
            == SEED_SHA, 'Input bytes changed')
    require(smoke['state'] == 'completed' and smoke['counterfactual_worlds'] == 0,
            'Smoke did not complete without counterfactuals')
    require(smoke['admitted'] == 180 and smoke['skipped'] == 0 and
            len(smoke['updates']) == 24 and smoke['learning_intervals'] == 1440,
            'Unexpected smoke counts')
    require(len(smoke_adm['vessels']) == 24 and smoke_adm['vessel_failed'] == 0,
            'Smoke vessel admission failure')
    require(smoke['cargo']['physical_inventory'] == smoke['cargo']['expected_inventory'],
            'Smoke inventory conservation failure')
    for key in ('initial_checkpoint', 'final_checkpoint'):
        record = smoke[key]
        require(sha(args.smoke/record['path']) == record['sha256'], 'Smoke checkpoint changed')
    require(sha(args.run/'initial.pt') == sha(Path('outputs/v5/yr306-cargo-30d-83de18d/initial.pt')),
            'Original initial policy differs')
    for folder in (args.run, args.smoke):
        require(read(folder/'container_contract.json')['passed'], 'Invalid container contract')
    diff = subprocess.check_output(['git', '-C', str(args.source), 'diff',
                                    '415e3e7', '--', 'src/yard_rl/v5', 'tests/v5'])
    require(not diff, 'Source differs from tested admission-wait correction')
    require(not subprocess.check_output(['git', '-C', str(args.source), 'status',
                                        '--porcelain']), 'Source worktree changed')
    # Read coherent hourly progress without pausing or changing the running simulation.
    live_names = ('status.json', 'admissions.json', 'cargo-status.json', 'days.json')
    for _ in range(10):
        chunks = {n: (args.run/n).read_bytes() for n in live_names}
        status, cargo = json.loads(chunks['status.json']), json.loads(chunks['cargo-status.json'])
        if chunks['status.json'] == (args.run/'status.json').read_bytes() and cargo['time_s'] == status['time_s']:
            break
    else:
        raise RuntimeError('Progress changed during snapshot; rerun after hourly write')
    require(status['state'] == 'running' and status['updates'] > 0, 'Learning not yet verified')
    require(status['skipped'] == 0 and status['vessel_failed'] == 0, 'Live admission failure')
    admitted = json.loads(chunks['admissions.json'])
    require(admitted['admitted'] == status['admitted'] and admitted['skipped'] == 0
            and admitted['vessel_failed'] == 0, 'Admission snapshot mismatch')
    require(cargo['physical_inventory'] == cargo['expected_inventory'], 'Live inventory mismatch')
    days = json.loads(chunks['days.json'])
    require(all(d['physical_invariants_checked'] for d in days), 'Missing daily physical check')
    pid = manifest['code']['pid']
    affinity = subprocess.check_output(['wsl', '--exec', 'taskset', '-pc', str(pid)], text=True).strip()
    require(affinity.rsplit(':', 1)[-1].strip() == '23', 'CPU affinity differs')
    command = subprocess.check_output(['wsl', '--exec', 'cat', f'/proc/{pid}/cmdline'])
    command = command.decode('utf-8').replace('\0', ' ').strip()
    require('yard_rl.v5.ppo.continuous' in command and args.run.name in command, 'PID reused')
    root = args.report
    require(not (root/'verification.json').exists(), 'Do not overwrite a prior snapshot')
    for label, source, names in (
        ('smoke', args.smoke, ('manifest.json', 'status.json', 'report.json', 'admissions.json',
                             'cargo-status.json', 'days.json', 'container_contract.json', 'events.jsonl')),
        ('startup', args.run, ('manifest.json', 'container_contract.json'))):
        folder = root/label
        folder.mkdir(parents=True, exist_ok=False)
        for name in names:
            shutil.copyfile(source/name, folder/name)
    for name, data in chunks.items():
        (root/'startup'/name).write_bytes(data)
    for label, source in (('smoke', args.smoke), ('startup', args.run)):
        shutil.copyfile(source.with_suffix('.launch.sh'), root/label/'launch.sh')
    value = dict(task='YR-306',scope='short CLI verification and live long-run startup only',
        captured_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,
        preregistration='.claude/docs/dashboard-task-specs/YR-306-wait-restart.md',
        prior_fix_commit='415e3e7',source_matches_tested_fix=True,
        preregistration_sha256_at_source=sha(args.source/'.claude/docs/dashboard-task-specs/YR-306-wait-restart.md'),
        code=manifest['code'],absolute_seed=manifest['seed'],config=manifest['ppo'],
        same_long_input_and_settings=True,seed_sha256=SEED_SHA,
        long_run=str(args.run),smoke_run=str(args.smoke),old_world_resumed=False,
        initial_checkpoint_sha256=sha(args.run/'initial.pt'),
        initial_policy_matches_original=True,
        smoke_initial_checkpoint_sha256=sha(args.smoke/'initial.pt'),
        prior_initial_checkpoint_sha256=sha(Path('outputs/v5/yr306-cargo-30d-83de18d/initial.pt')),
        launcher_pid=args.launcher_pid,linux_pid=pid,cpu_affinity=affinity,command=command,
        smoke=dict(state=smoke['state'],updates=len(smoke['updates']),admitted=smoke['admitted'],
                   skipped=smoke['skipped'],vessels=len(smoke_adm['vessels']),
                   counterfactual_worlds=smoke['counterfactual_worlds'],cargo=smoke['cargo']),
        startup=status,cargo=cargo,long_run_restarted=True,full_30_days_completed=False,
        artifacts_sha256={p.relative_to(root).as_posix():sha(p)
                          for p in root.rglob('*') if p.is_file()})
    value = attach_common_gates(value, report_from_dict(read(root/'gate-at-start.json')))
    (root/'verification.json').write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(startup=status,smoke=value['smoke'],cpu_affinity=affinity),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
