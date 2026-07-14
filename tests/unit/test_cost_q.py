"""Focused tests for the YR-027 direct-job Cost-Q policy."""
from dataclasses import dataclass

import pytest

from yard_rl.policies.cost_q import CostQAgent, CostQConfig, CostQTable


@dataclass(frozen=True)
class State:
    period: int
    zone: int


@dataclass(frozen=True)
class Candidate:
    job_id: str
    feature: tuple
    wait_s: float
    estimated_service_s: float
    block_entry_s: float


def candidate(
    job_id: str,
    feature: str,
    *,
    wait: float = 10.0,
    service: float = 10.0,
    entry: float = 0.0,
) -> Candidate:
    return Candidate(job_id, (feature,), wait, service, entry)


def set_visited(agent: CostQAgent, state: State, item: Candidate, q: float) -> None:
    key = agent.key(state, item)
    agent.table.q[key] = q
    agent.table.n[key] = 1


def test_q0_standard_min_backup_and_visit_power_learning_rate():
    cfg = CostQConfig(learning_rate_power=0.6)
    agent = CostQAgent(cfg, seed=1)
    state, next_state = State(0, 0), State(1, 0)
    chosen = candidate("chosen", "current")
    learned_next = candidate("learned", "learned-next")
    unseen_next = candidate("unseen", "unseen-next")
    set_visited(agent, next_state, learned_next, 7.0)

    # The unseen feasible key participates at Q0=0, so target = cost + 0.
    assert agent.update(
        state, chosen, 2.0, next_state, [learned_next, unseen_next], done=False
    ) == pytest.approx(2.0)
    assert agent.table.visits(agent.key(state, chosen)) == 1

    set_visited(agent, next_state, unseen_next, 3.0)
    updated = agent.update(
        state, chosen, 2.0, next_state, [learned_next, unseen_next], done=False
    )
    alpha = 2 ** -0.6
    assert updated == pytest.approx(2.0 + alpha * (5.0 - 2.0))
    assert agent.table.visits(agent.key(state, chosen)) == 2


def test_terminal_target_is_cost_and_gamma_contract():
    state = State(0, 0)
    chosen = candidate("chosen", "current")
    agent = CostQAgent(CostQConfig(learning_rate_power=1.0))
    assert agent.update(state, chosen, 4.5, None, [], done=True) == 4.5
    # YR-030-b: γ∈(0,1] 은 실험 축으로 개방 — YR-027 프로토콜은 호출부에서 1.0 고정
    assert CostQConfig(gamma=0.99).gamma == 0.99
    assert CostQConfig().gamma == 1.0
    with pytest.raises(ValueError):
        CostQConfig(gamma=0.0)
    with pytest.raises(ValueError):
        CostQConfig(gamma=1.5)


def test_training_prioritizes_unvisited_unique_keys_before_epsilon():
    state = State(0, 0)
    first = candidate("x-first", "x", wait=20)
    duplicate = candidate("x-duplicate", "x", wait=5)
    second = candidate("y", "y")

    # A duplicate Job with the same signature must not give that key more weight.
    with_duplicate = [
        CostQAgent(seed=seed).act_train(state, [first, duplicate, second], 0.0).feature
        for seed in range(20)
    ]
    without_duplicate = [
        CostQAgent(seed=seed).act_train(state, [first, second], 0.0).feature
        for seed in range(20)
    ]
    assert with_duplicate == without_duplicate

    agent = CostQAgent(seed=3)
    set_visited(agent, state, first, -100.0)
    # The only unseen signature wins even when epsilon exploration is requested.
    assert agent.act_train(state, [first, second], epsilon=1.0) is second


def test_seeded_epsilon_exploration_is_reproducible():
    state = State(0, 0)
    choices = [candidate(str(index), str(index)) for index in range(4)]
    agents = [CostQAgent(seed=17), CostQAgent(seed=17)]
    for agent in agents:
        for item in choices:
            set_visited(agent, state, item, 1.0)
    sequences = [
        [agent.act_train(state, choices, epsilon=1.0).job_id for _ in range(12)]
        for agent in agents
    ]
    assert sequences[0] == sequences[1]


def test_argmin_tie_break_is_wait_service_entry_then_job_id():
    state = State(0, 0)
    choices = [
        candidate("short-wait", "0", wait=9, service=1, entry=0),
        candidate("long-service", "1", wait=10, service=5, entry=0),
        candidate("late-entry", "2", wait=10, service=4, entry=2),
        candidate("B", "3", wait=10, service=4, entry=1),
        candidate("A", "4", wait=10, service=4, entry=1),
    ]
    agent = CostQAgent(seed=0)
    for item in choices:
        set_visited(agent, state, item, 3.0)
    assert agent.act_train(state, choices, epsilon=0.0).job_id == "A"


def test_eval_uses_whole_decision_fallback_and_records_coverage():
    state = State(0, 0)
    quick = candidate("quick", "quick", service=2)
    slow_unseen = candidate("slow", "slow", service=9)
    agent = CostQAgent(seed=0)
    set_visited(agent, state, quick, 5.0)

    # Mixed known/unseen candidates never enter a partly learned argmin.
    assert agent.act(state, [quick, slow_unseen]) is quick
    assert agent.fallback_count == 1
    assert agent.fallback_rate == 1.0
    assert agent.coverage_rate == 0.5

    set_visited(agent, state, slow_unseen, 1.0)
    assert agent.act(state, [quick, slow_unseen]) is slow_unseen
    assert agent.diagnostics.fully_covered_decisions == 1
    assert agent.fallback_rate == 0.5
    assert agent.coverage_rate == 0.75


def test_agent_save_load_round_trip_preserves_table_stats_and_rng(tmp_path):
    state = State(2, 3)
    choices = [candidate(str(index), str(index)) for index in range(3)]
    agent = CostQAgent(CostQConfig(learning_rate_power=0.8), seed=23)
    for index, item in enumerate(choices):
        set_visited(agent, state, item, float(index))
    agent.act(state, choices)
    agent.act_train(state, choices, epsilon=1.0)  # advance RNG before save

    path = tmp_path / "cost-q.json"
    agent.save(path)
    restored = CostQAgent.load(path)

    assert restored.cfg == agent.cfg
    assert restored.table.q == agent.table.q
    assert restored.table.n == agent.table.n
    assert restored.diagnostics.as_dict() == agent.diagnostics.as_dict()
    expected = [agent.act_train(state, choices, 1.0).job_id for _ in range(10)]
    actual = [restored.act_train(state, choices, 1.0).job_id for _ in range(10)]
    assert actual == expected


def test_table_round_trip_and_empty_candidate_guards(tmp_path):
    table = CostQTable()
    key = ((1, 2), ("direction", 3))
    table.q[key], table.n[key] = 1.25, 4
    path = tmp_path / "table.json"
    table.save(path)
    restored = CostQTable.load(path)
    assert restored.q == table.q and restored.n == table.n

    agent = CostQAgent()
    with pytest.raises(ValueError, match="feasible candidate"):
        agent.act(State(0, 0), [])
    with pytest.raises(ValueError, match="feasible next key"):
        agent.update(State(0, 0), candidate("x", "x"), 1.0, State(1, 0), [], False)
