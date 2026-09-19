from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/v3'))
import recover_block_only_evaluation as recovery
from independent_eval_checks import SEEDS, read, save, sha


def idle_source(tmp_path):
    source = tmp_path/'old'
    save(source/'launch.json', dict(pid=123, config_sha256='frozen'))
    save(source/'progress.json', dict(state='waiting_for_resources', phase='months',
         active=[], failed=[], planned=20, completed=1, pending=19))
    folder = source/'months'/str(SEEDS[0])
    save(folder/'RL_SPACE/completion.json', dict(state='completed', loss_retained=True))
    (folder/'RL_SPACE.log').write_bytes(b'original log\n')
    return source


def test_preflight_allows_idle_live_supervisor_but_takeover_refuses_it(tmp_path, monkeypatch):
    source = idle_source(tmp_path)
    monkeypatch.setattr(recovery, 'process_alive', lambda _: True)
    _, status = recovery.source_state(source, tmp_path/'new', 'frozen', require_stopped=False)
    assert status['active'] == []
    with pytest.raises(ValueError, match='supervisor must stop'):
        recovery.source_state(source, tmp_path/'new', 'frozen', require_stopped=True)
    monkeypatch.setattr(recovery, 'process_alive', lambda _: False)
    monkeypatch.setattr(recovery, 'source_child_pids', lambda _: [])
    recovery.source_state(source, tmp_path/'new', 'frozen', require_stopped=True)


def test_dead_supervisor_with_orphan_worker_cannot_be_taken_over(tmp_path, monkeypatch):
    source = idle_source(tmp_path)
    monkeypatch.setattr(recovery, 'process_alive', lambda _: False)
    monkeypatch.setattr(recovery, 'source_child_pids', lambda _: [456])
    with pytest.raises(ValueError, match='still has live workers'):
        recovery.source_state(source, tmp_path/'new', 'frozen', require_stopped=True)


def test_worker_scan_matches_output_and_seed_but_not_other_runs_or_supervisor(tmp_path):
    proc = tmp_path/'proc'
    out = tmp_path/'old'
    for pid, script, output, tail in (
            (1, 'run_block_only_evaluation.py', out, ['--seed', '20000000']),
            (2, 'run_block_only_evaluation.py', out, []),
            (3, 'run_block_only_evaluation.py', tmp_path/'other', ['--seed', '20000000']),
            (4, 'run_independent_evaluation.py', out, ['--seed', '20000000'])):
        folder = proc/str(pid)
        folder.mkdir(parents=True)
        argv = ['python', '/frozen/'+script, '--out', str(output), *tail]
        (folder/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
    assert recovery.source_child_pids(out, proc_root=proc) == [1]


@pytest.mark.parametrize('problem', ['active', 'failed', 'failure_file', 'config', 'same_output'])
def test_recovery_rejects_active_failed_or_changed_source(tmp_path, problem):
    source = idle_source(tmp_path)
    status = read(source/'progress.json')
    if problem == 'active': status['active'] = [dict(pid=99)]
    if problem == 'failed': status['failed'] = [dict(seed=SEEDS[0])]
    if problem == 'failure_file': save(source/'failure.json', dict(reason='preserve'))
    save(source/'progress.json', status)
    with pytest.raises(ValueError):
        recovery.source_state(source, source if problem == 'same_output' else tmp_path/'new',
            'changed' if problem == 'config' else 'frozen', require_stopped=False)


def test_only_absent_seed_folders_are_unstarted_not_partial_attempts(tmp_path):
    source = idle_source(tmp_path)
    status = read(source/'progress.json')
    completed, pending = recovery.classify_months(source, status)
    assert completed == [SEEDS[0]] and pending == list(SEEDS[1:])
    partial = source/'months'/str(SEEDS[1])
    partial.mkdir()
    with pytest.raises(ValueError, match='Partial or unexpected'):
        recovery.classify_months(source, status)


def test_classification_rejects_progress_disagreement_and_nonregistered_seed(tmp_path):
    source = idle_source(tmp_path)
    status = read(source/'progress.json')
    status['completed'] = 2
    with pytest.raises(ValueError, match='progress does not match'):
        recovery.classify_months(source, status)
    (source/'months/999').mkdir()
    with pytest.raises(ValueError, match='fixed seed list'):
        recovery.classify_months(source, status)


def test_checked_copy_preserves_bytes_and_rejects_overwrite_or_changed_source(tmp_path):
    source = idle_source(tmp_path)/'months'/str(SEEDS[0])
    hashes = recovery.tree_hashes(source)
    recovery.copy_checked(source, tmp_path/'copy', hashes)
    assert recovery.tree_hashes(tmp_path/'copy') == hashes == recovery.tree_hashes(source)
    with pytest.raises(ValueError, match='not overwrite'):
        recovery.copy_checked(source, tmp_path/'copy', hashes)
    (source/'RL_SPACE.log').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed after'):
        recovery.copy_checked(source, tmp_path/'another', hashes)


@pytest.mark.parametrize('name', ['failure.json', 'result.json.tmp'])
def test_tree_hashes_reject_failure_and_incomplete_writes(tmp_path, name):
    (tmp_path/name).write_bytes(b'preserve')
    with pytest.raises(ValueError, match='failure or incomplete'):
        recovery.tree_hashes(tmp_path)


def test_recovery_preserves_original_completion_logs_and_launch_receipt(tmp_path, monkeypatch):
    source = idle_source(tmp_path)
    save(source/'smoke-summary.json', dict(passed=True))
    save(source/'smoke/9900722/RL_SPACE/result.json', dict(smoke=True))
    (source/'supervisor.log').write_bytes(b'original supervisor log')
    month = source/'months'/str(SEEDS[0])
    retained = [read(month/'RL_SPACE/completion.json')]
    plan = dict(previous_run=str(source), retained=retained,
        receipt_files={p.name:sha(p) for p in source.iterdir() if p.is_file()},
        smoke_files=recovery.tree_hashes(source/'smoke'),
        evidence=[dict(seed=SEEDS[0], files=recovery.tree_hashes(month))])
    before = recovery.tree_hashes(source)
    def check_plan(args, cfg, old, *, require_stopped):
        assert require_stopped is True
        return plan
    monkeypatch.setattr(recovery, 'recovery_plan', check_plan)
    out = tmp_path/'new'
    save(out/'launch.json', dict(pid=999))
    result = recovery.recover_completed(NS(recover_from=source, out=out), {}, {})
    assert result == retained and recovery.tree_hashes(source) == before
    assert read(out/'launch.json') == dict(pid=999)
    assert read(out/'previous-run-evidence/launch.json')['pid'] == 123
    assert recovery.tree_hashes(out/'months'/str(SEEDS[0])) == recovery.tree_hashes(month)
    assert (out/'previous-run-evidence/supervisor.log').read_bytes() == b'original supervisor log'
