"""Read actual child manifests/processes and preserve a one-time start receipt."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(__file__).resolve().parent
run = root/'run-8c24c80'
main, smoke = run/'main', run/'smoke'
launch, children = read(main/'launch.json'), read(main/'children.json')
assert read(smoke/'progress.json')['state'] == 'completed'
assert not (main/'failure.json').exists()
rows = {}
for arm, cpu in (('NO_REALLOC', 18), ('RL_TIME', 19)):
    folder = main/arm
    manifest, progress = read(folder/'manifest.json'), read(folder/'progress.json')
    pid = children[arm]['pid']
    proc = Path('/proc')/str(pid)
    argv = [part.decode('utf-8') for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
    status = dict(line.split(':', 1) for line in (proc/'status').read_text().splitlines() if ':' in line)
    assert pid == manifest['pid'] == progress['pid'] and os.sched_getaffinity(pid) == {cpu}
    assert 'run_vertical_comparison.py' in argv[2] and '--smoke' not in argv
    assert manifest['contract']['settings']['seed'] == 20_000_000
    assert manifest['contract']['settings']['n_days'] == 30
    assert progress['state'] == 'running' and not (folder/'failure.json').exists()
    for key, value in launch['successful_smoke_identity'].items():
        assert manifest[key] == value, (arm, key)
    rows[arm] = dict(pid=pid, cpu=cpu, argv=argv, progress=progress,
        memory_bytes=int(status['VmRSS'].split()[0])*1024,
        manifest_sha256=sha(folder/'manifest.json'), expected_input=manifest['expected_input'],
        pair_contract_sha256=manifest['pair_contract_sha256'])
assert rows['NO_REALLOC']['expected_input'] == rows['RL_TIME']['expected_input']
assert rows['NO_REALLOC']['pair_contract_sha256'] == rows['RL_TIME']['pair_contract_sha256']
base = read(root/'config.json')
workspace = root.parents[2]
assert sha(workspace/base['base_config']) == base['base_config_sha256']
preserved = {}
for pid in (23325, 36166, 4583):
    proc = Path('/proc')/str(pid)
    assert proc.exists(), pid
    preserved[str(pid)] = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8').strip()
receipt = dict(at=datetime.now(timezone.utc).isoformat(), state='running',
    seed=20_000_000, planned_runs=2, source_commit=base['source_commit'],
    supervisor_pid=launch['supervisor_pid'], config_sha256=sha(root/'config.json'),
    smoke_summary_sha256=sha(smoke/'pair-summary.json'),
    same_frozen_contract_as_smoke=True, new_training_runs=0, arms=rows,
    original_processes_still_alive=preserved, claim_eligible=False)
with (root/'started.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(state='running', seed=20_000_000,
    arms={arm: dict(pid=row['pid'], cpu=row['cpu']) for arm, row in rows.items()})))
