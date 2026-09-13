"""Reference runner contracts; do not alter the production simulator or models."""
from collections import Counter
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import pytest
import torch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                           'scripts/v5/compare_frozen_policies.py'))


def test_savings_sign_and_zero_denominator():
    savings = MODULE['savings']
    assert savings(80, 100) == {'saving_krw': 20, 'saving_percent': 20}
    assert savings(120, 100) == {'saving_krw': -20, 'saving_percent': -20}
    assert savings(2, 0) == {'saving_krw': -2, 'saving_percent': None}


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_invalid_cost_rejected(bad):
    with pytest.raises(RuntimeError):
        MODULE['savings'](bad, 10)


def test_rule_selects_actual_keep_row_only():
    runtime = object.__new__(MODULE['RuleRuntime'])
    runtime.role_counts = Counter()
    rows = torch.zeros(3, 21)
    rows[0, 12] = 1
    assert runtime.select('seller', 'Y01', 0, rows) == 0
    with pytest.raises(RuntimeError, match='buyer'):
        runtime.select('buyer', 'Y01', 0, rows)
    rows[0, 12] = 0
    with pytest.raises(RuntimeError, match='encoding'):
        runtime.select('seller', 'Y01', 0, rows)


def test_rule_calls_existing_executor_and_counts_assignments(monkeypatch):
    from yard_rl.v5.ppo.model import BlockPolicy
    from yard_rl.v5.world.contract.schema import CandidateKind
    calls = []
    errors = {'n': 0}
    monkeypatch.setitem(MODULE['RuleRuntime'].__init__.__globals__, '_rule_policy',
                        lambda _: (lambda sim, dp: calls.append(dp), errors))
    runtime = MODULE['RuleRuntime'](BlockPolicy(), training=False)
    sim = NS(now=60, last_assignments=lambda: {'a': NS(action=CandidateKind.SERVE)})
    runtime.block_of = {id(sim): 'Y01'}
    runtime.mbt = object()
    runtime.bridge = NS(_sync=lambda *args, **kw: None)
    runtime.execute(sim, 'decision')
    assert calls == ['decision'] and runtime.crane_actions['SERVE'] == 1
    errors['n'] = 1
    with pytest.raises(RuntimeError, match='fallback'):
        runtime.execute(sim, 'decision')


def test_digest_changes_only_with_model_parameters():
    from yard_rl.v5.ppo.model import BlockPolicy
    policy = BlockPolicy()
    before = MODULE['policy_digest'](policy)
    policy.eval()
    assert MODULE['policy_digest'](policy) == before
    with torch.no_grad():
        next(policy.parameters()).add_(1)
    assert MODULE['policy_digest'](policy) != before


def comparison_fixture(root):
    for arm, cost in (('final', 80), ('initial', 120), ('rule', 100)):
        target = root / arm
        target.mkdir()
        MODULE['write_json'](target / 'manifest.json', dict(seed=1, days=[1, 2, 3],
            fixed_seed={'sha256': 'same'}, measurement_window_s=[86400, 172800],
            code={'source_sha256': 'same'}))
        MODULE['write_json'](target / 'days.json', [])
        MODULE['write_json'](target / 'report.json', dict(state='completed', checks={'valid': True},
            cost_krw=cost * 2, measurement_cost_krw=cost, cost_breakdown={},
            cargo={'active_unfinished_jobs': {'VESSEL_LOAD': 10}}, residual={}))


def test_comparison_uses_paired_window_and_does_not_claim_formal_pass(tmp_path):
    comparison_fixture(tmp_path)
    result = MODULE['summarize'](tmp_path)
    assert result['comparisons']['final_vs_rule']['measurement_window']['saving_percent'] == 20
    assert result['comparisons']['initial_vs_rule']['whole_window']['saving_percent'] == -20
    assert not result['formal_performance_pass']


@pytest.mark.parametrize('mismatch', ['window', 'source', 'failed'])
def test_unpaired_or_failed_run_has_no_comparison(tmp_path, mismatch):
    comparison_fixture(tmp_path)
    name = 'report.json' if mismatch == 'failed' else 'manifest.json'
    target = tmp_path / 'rule' / name
    value = MODULE['read'](target)
    if mismatch == 'window':
        value['measurement_window_s'][0] = 0
    elif mismatch == 'source':
        value['code']['source_sha256'] = 'changed'
    else:
        value['state'] = 'failed'
    MODULE['write_json'](target, value)
    with pytest.raises(RuntimeError):
        MODULE['summarize'](tmp_path)


def test_queue_recognizes_training_and_its_followup_evaluation(monkeypatch):
    queue = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                              'scripts/v5/queue_reference_comparison.py'))
    monkeypatch.setattr(queue['subprocess'], 'run', lambda *a, **kw: NS(stdout=(
        'PID COMMAND\n12 python -m yard_rl.v4.crane --workers 10\n'
        '13 python scripts/v4/eval_crane_split.py\n14 python idle.py\n')))
    assert len(queue['busy_workloads']()) == 2
