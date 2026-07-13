"""YR-027 외부트럭 전용 baseline 계약."""
from dataclasses import dataclass

import pytest

from yard_rl.policies.direct_baselines import (
    DirectJobRulePolicy,
    DirectRule,
    direct_baseline_policies,
)


@dataclass(frozen=True)
class Candidate:
    job_id: str
    block_entry_s: float
    wait_s: float
    reach_s: float
    estimated_service_s: float
    blocker_count: int


def test_fifo_and_longest_wait_are_block_entry_aliases_not_gate_order():
    early = Candidate("B", 100.0, 900.0, 30.0, 60.0, 2)
    late = Candidate("A", 200.0, 800.0, 10.0, 20.0, 0)
    for rule in (DirectRule.FIFO, DirectRule.LONGEST_WAIT):
        assert DirectJobRulePolicy(rule).act(None, [late, early]) is early


def test_nearest_blocker_and_shortest_service_use_only_named_primary():
    near = Candidate("near", 200.0, 1.0, 5.0, 100.0, 3)
    clean = Candidate("clean", 300.0, 999.0, 20.0, 80.0, 0)
    quick = Candidate("quick", 400.0, 1.0, 30.0, 10.0, 2)
    pool = [quick, clean, near]
    assert DirectJobRulePolicy(DirectRule.NEAREST_JOB).act(None, pool) is near
    assert DirectJobRulePolicy(DirectRule.MIN_BLOCKER).act(None, pool) is clean
    assert DirectJobRulePolicy(DirectRule.SHORTEST_ESTIMATED_SERVICE_TIME).act(None, pool) is quick


def test_immediate_cost_and_shortest_service_are_exact_aliases():
    pool = [
        Candidate("B", 10.0, 20.0, 1.0, 30.0, 0),
        Candidate("A", 20.0, 10.0, 2.0, 30.0, 1),
    ]
    shortest = DirectJobRulePolicy(DirectRule.SHORTEST_ESTIMATED_SERVICE_TIME)
    greedy = DirectJobRulePolicy(DirectRule.IMMEDIATE_COST_GREEDY)
    assert shortest.act(None, pool).job_id == greedy.act(None, pool).job_id == "B"


def test_policy_list_and_empty_guard():
    assert [policy.name for policy in direct_baseline_policies()] == [rule.value for rule in DirectRule]
    with pytest.raises(ValueError, match="feasible candidate"):
        DirectJobRulePolicy(DirectRule.FIFO).act(None, [])
