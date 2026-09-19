"""Launcher guard regression checks; no simulation or real child is started."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher(monkeypatch):
    script = Path(__file__).resolve().parents[2]/'scripts/v3/launch_vertical_comparison.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location('vertical_pair_launcher_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def valid_smoke(tmp_path, monkeypatch, launcher):
    m = launcher
    runner = tmp_path/'scripts/v3/run_vertical_comparison.py'
    runner.parent.mkdir(parents=True)
    runner.write_text('frozen child', encoding='utf8')
    cfg = tmp_path/'config.json'
    cfg.write_text('{}', encoding='utf8')
    out = tmp_path/'pair'
    smoke = out/'smoke'
    smoke.mkdir(parents=True)
    m.save(smoke/'launch.json', dict(config_sha256=m.sha(cfg), runner_sha256=m.sha(runner),
        launcher_sha256=m.sha(Path(m.__file__))))
    m.save(smoke/'progress.json', {'state': 'completed'})
    identity = {key: key for key in m.FROZEN_KEYS}
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'processes', lambda: [])
    monkeypatch.setattr(m, 'validate_pair', lambda _: {'completions': [{'identity': identity}]*2})
    return SimpleNamespace(out=out, config=cfg), identity


def test_main_accepts_successful_frozen_smoke(launcher, valid_smoke):
    args, identity = valid_smoke
    assert launcher.smoke_gate(args) == identity


@pytest.mark.parametrize('invalid', ('failed_smoke', 'changed_config', 'live_smoke'))
def test_main_rejects_invalid_smoke(launcher, valid_smoke, monkeypatch, invalid):
    args, _ = valid_smoke
    smoke = args.out/'smoke'
    if invalid == 'failed_smoke':
        launcher.save(smoke/'progress.json', {'state': 'failed'})
    elif invalid == 'changed_config':
        args.config.write_text('{"changed":true}', encoding='utf8')
    else:
        monkeypatch.setattr(launcher, 'processes',
            lambda: [{'argv': ['child', '--out', str(smoke)]}])
    with pytest.raises(ValueError):
        launcher.smoke_gate(args)


def test_failed_arm_does_not_terminate_peer_and_preserves_evidence(tmp_path, monkeypatch, launcher):
    m = launcher
    phase = tmp_path/'smoke'
    phase.mkdir()
    args = SimpleNamespace(phase=phase, smoke=True, main=False, workspace=tmp_path,
                           config=tmp_path/'config.json', out=tmp_path)
    started = []

    class Child:
        def __init__(self, arm):
            self.arm, self.pid, self.returncode, self.polls = arm, 700+len(started), None, 0
            started.append(self)
            folder = phase/arm
            folder.mkdir()
            m.save(folder/'progress.json', {'state': 'running', 'day_index_completed': 0})
            if arm == 'RL_TIME':
                m.save(folder/'completion.json', {'preserved': True})

        def poll(self):
            self.polls += 1
            if self.arm == 'RL_TIME' and self.polls == 1:
                return None
            self.returncode = 1 if self.arm == 'NO_REALLOC' else 0
            return self.returncode

        def wait(self):
            self.returncode = 1 if self.arm == 'NO_REALLOC' else 0
            return self.returncode

        def terminate(self):
            pytest.fail('Independent peer must never be terminated')

        def kill(self):
            pytest.fail('Independent peer must never be killed')

    monkeypatch.setattr(m, 'resources', lambda: {'checks': {'test': True}})
    monkeypatch.setattr(m, 'processes', lambda: [])
    monkeypatch.setattr(m.subprocess, 'Popen', lambda cmd, **_: Child(cmd[cmd.index('--arm')+1]))
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='peer was allowed to finish'):
        m.supervise(args)
    progress = m.read(phase/'progress.json')
    assert progress['state'] == 'failed'
    assert progress['failed']['NO_REALLOC']['exit_code'] == 1
    assert progress['completed']['RL_TIME']['exit_code'] == 0
    assert started[1].polls >= 2
    assert m.read(phase/'RL_TIME'/'completion.json')['preserved']
    assert len(m.read(phase/'children.json')) == 2
    assert (phase/'failure.json').exists()


def test_pair_gate_rejects_changed_result_even_when_audit_says_pass(tmp_path, monkeypatch, launcher):
    m = launcher
    monkeypatch.setitem(sys.modules, 'vertical_comparison_io',
                        SimpleNamespace(pair_summary=lambda _: {'passed': True}))
    for arm in m.ARMS:
        folder = tmp_path/arm
        folder.mkdir()
        m.save(folder/'result.json', {'arm': arm})
        m.save(folder/'manifest.json', {'frozen': True})
        m.save(folder/'completion.json', dict(state='completed', audit={'passed': True},
            result_sha256=m.sha(folder/'result.json'), manifest_sha256=m.sha(folder/'manifest.json'),
            identity={'shared': True}))
    assert m.validate_pair(tmp_path)['summary']['passed']
    m.save(tmp_path/'RL_TIME'/'result.json', {'changed': True})
    with pytest.raises(ValueError, match='audit/hash failed'):
        m.validate_pair(tmp_path)
