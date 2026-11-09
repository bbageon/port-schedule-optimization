"""The teacher uses real selection constraints and a shared float32 score."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from yard_rl.v7.crane.action_contract import (
    assignment_signature, input_key, reachable_alternative)
from yard_rl.v7.crane.policy import CraneNet, CranePolicy, float32_key
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.integrated.baselines import ResolverPolicy
from test_candidate_recall import state, generate


@pytest.fixture
def decision(monkeypatch):
    # Distinct inputs isolate action reachability from representation collision.
    def features(sim, cid, gc, **kw):
        return [float(gc.candidate_id)] + [0.] * 8
    monkeypatch.setattr('yard_rl.v7.crane.policy.crane_features', features)
    monkeypatch.setattr('yard_rl.v7.crane.action_contract.crane_features', features)
    sim, dp = state()
    gb = generate(sim, dp, None)
    net = CraneNet(feature_mode='feasible')
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    pref = CranePolicy(net)
    picked = ResolverPolicy(pref).decide(sim, dp, gb)['YC-A']
    return sim, dp, gb, pref, picked


def test_full_candidate_alternative_is_selected_without_mutating_live_policy(decision):
    sim, dp, gb, pref, picked = decision
    cands = [g for g in gb['YC-A'].items if g.job_ref and g is not picked]
    target = cands[-1]
    before = assignment_signature(ResolverPolicy(pref).decide(sim, dp, gb))
    alt, witness, attempts = reachable_alternative(pref, sim, dp, gb,
        [('YC-A', picked, cands)], ('YC-A', picked, target))
    assert alt is target and target.candidate_id >= 12
    assert attempts[-1]['reason'] == 'accepted'
    assert len(gb['YC-A'].items) == 31
    assert not pref.score_once and not pref.force_once
    assert before == assignment_signature(ResolverPolicy(pref).decide(sim, dp, gb))
    pref.score_once[tuple(witness['input_key'])] = witness['score']
    assert witness['assignment'] == assignment_signature(ResolverPolicy(pref).decide(sim, dp, gb))
    pairs = [('YC-A', g) for g in gb['YC-A'].items]
    assert pref.rank_many(sim, pairs) == [pref.rank(sim, *p) for p in pairs]


@pytest.mark.parametrize('constraint', ['serve_first', 'mandatory'])
def test_score_intervention_cannot_bypass_priority(decision, constraint):
    sim, dp, gb, pref, picked = decision
    target = gb['YC-A'].items[1]
    items = list(gb['YC-A'].items)
    if constraint == 'serve_first':
        target = replace(target, kind=CandidateKind.PRE_REHANDLE)
        items[target.candidate_id] = target
    else:
        items[picked.candidate_id] = replace(picked, mandatory=True)
        picked = items[picked.candidate_id]
    gb['YC-A'] = replace(gb['YC-A'], items=tuple(items))
    alt, witness, attempts = reachable_alternative(pref, sim, dp, gb,
        [('YC-A', picked, [target])], ('YC-A', picked, target))
    assert alt is witness is None
    assert attempts[0]['reason'] == 'not_selected_by_resolver'
    assert not pref.score_once


def test_same_float32_inputs_cannot_receive_different_teacher_scores(decision, monkeypatch):
    sim, dp, gb, pref, picked = decision
    target = gb['YC-A'].items[1]
    def features(sim, cid, gc, **kw):
        return [1.0 + gc.candidate_id * 1e-10] + [0.] * 8
    monkeypatch.setattr('yard_rl.v7.crane.policy.crane_features', features)
    monkeypatch.setattr('yard_rl.v7.crane.action_contract.crane_features', features)
    assert float32_key([1.0]) == float32_key([1.0 + 1e-10])
    alt, witness, attempts = reachable_alternative(pref, sim, dp, gb,
        [('YC-A', picked, [target])], ('YC-A', picked, target))
    assert alt is witness is None
    assert attempts[0]['reason'] == 'same_input_as_picked'
    pref.score_once[input_key(pref, sim, 'YC-A', target)] = -3.
    assert pref.rank(sim, 'YC-A', picked)[1] == pref.rank(sim, 'YC-A', target)[1] == -3.


def test_rejected_decision_spends_budget_without_submitting_worlds(decision, monkeypatch):
    from yard_rl.v7.crane.collect import CraneCollector
    from yard_rl.v7.stage.rollout import RolloutBudget
    sim, dp, gb, pref, picked = decision
    monkeypatch.setattr('yard_rl.v7.crane.action_contract.reachable_alternative',
                        lambda *a: (None, None, [dict(reason='not_selected_by_resolver')]))
    budget = RolloutBudget(max_labels=1, stride=1)
    collector = CraneCollector(SimpleNamespace(), budget=budget, horizon_s=1800)
    collector.attach(pref)
    submitted = []
    collector.pool.submit = submitted.append
    hook = collector.hook(SimpleNamespace(blocks={'Y01':sim}), SimpleNamespace(decided=set()), {}, {})
    hook(sim, dp, gb, {'YC-A':picked})
    hook(sim, dp, gb, {'YC-A':picked})
    assert collector.stats['sampled'] == collector.stats['no_witness'] == budget.taken == 1
    assert not submitted and not collector.rows
    collector.close()


def test_search_restores_preference_after_resolver_error(decision, monkeypatch):
    sim, dp, gb, pref, picked = decision
    target = gb['YC-A'].items[1]
    def broken(*a):
        raise RuntimeError('dry-run failure')
    monkeypatch.setattr(ResolverPolicy, 'decide', broken)
    with pytest.raises(RuntimeError, match='dry-run failure'):
        reachable_alternative(pref, sim, dp, gb,
            [('YC-A', picked, [target])], ('YC-A', picked, target))
    assert not pref.score_once and not pref.force_once


def test_joint_replay_rejects_matching_target_with_wrong_other_crane(monkeypatch):
    from yard_rl.v7.crane.rollout import CraneRollout, run_crane_job_with
    expected = (('A', 'SERVE', 'one', None), ('B', 'SERVE', 'two', None))
    changed = (('A', 'SERVE', 'alt', None), ('B', 'WAIT', None, None))
    job = SimpleNamespace(key=(0., 'Y01', 'A'), t=0., crane='A', picked_job='one',
        alt_job='alt', alternative_contract='reachable', factual_assignment=expected,
        witness=dict(assignment=(('A', 'SERVE', 'alt', None), ('B', 'SERVE', 'two', None))))
    def branch(self, job, *, force_job):
        return dict(phi=1., picked='one' if force_job is None else 'alt',
                    assignment=expected if force_job is None else changed)
    monkeypatch.setattr(CraneRollout, 'branch', branch)
    result = run_crane_job_with(None, 1800, job)
    assert result['factual_ok'] and not result['forced_ok']
