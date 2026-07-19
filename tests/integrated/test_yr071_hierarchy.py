"""YR-071 — 계층목적(사전식 4-tier)·JR objective 훅 계약.

핵심: ① 상위 tier(트럭 대기)는 하위 tier 가 아무리 좋아도 못 뒤집는다 ② objective
미지정 기본 경로는 기존 scalar argmin 과 동치 ③ JR_NEW 는 완주·결정론.
"""
import pytest

from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import JointRolloutGreedy, run_joint_episode
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.objectives import hierarchy_key
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario

RC = RewardCalculator.assumed_default()


def test_hierarchy_tier_dominance():
    worse_wait_better_ops = {"truck_wait": 0.2, "interference": 0.0, "lane_cong": 0.0}
    better_wait_worse_ops = {"truck_wait": 0.1, "interference": 99.0, "lane_cong": 99.0}
    assert hierarchy_key(better_wait_worse_ops) < hierarchy_key(worse_wait_better_ops)


def test_hierarchy_tie_falls_through_tiers():
    base = {"truck_wait": 1.0, "long_wait": 0.5, "vessel_delay": 0.2, "rehandle": 3.0}
    assert hierarchy_key(dict(base, long_wait=0.4)) < hierarchy_key(base)      # tier-B
    assert hierarchy_key(dict(base, vessel_delay=0.1)) < hierarchy_key(base)   # tier-C
    assert hierarchy_key(dict(base, rehandle=2.0)) < hierarchy_key(base)       # tier-D
    assert hierarchy_key(dict(base, depart_delay=0.01)) > hierarchy_key(base)  # tier-C 합산


def test_unknown_term_goes_to_tier_d():
    k = hierarchy_key({"truck_wait": 1.0, "brand_new_term": 5.0})
    assert k[0] == pytest.approx(1.0) and k[3] == pytest.approx(5.0)


def _small_sim(seed):
    prof = build_calibrated_profile()
    params = TerminalGenParams(n_external=8, n_vessels=1, gaussian=False)
    return TerminalSimulator(prof, generate_terminal_scenario(prof, seed, params),
                             check_invariants=True)


def test_default_path_equals_sum_objective():
    """contributions 합 = total_normalized 항등 → 합산 objective 는 scalar 경로와
    같은 결정을 내려야 한다 (term_sink 누적·훅 배선의 등가 검증)."""
    gen = CandidateGenerator()
    row_a = run_joint_episode(_small_sim(730000), JointRolloutGreedy(RC, generator=gen),
                              RC, generator=gen)
    row_b = run_joint_episode(
        _small_sim(730000),
        JointRolloutGreedy(RC, generator=gen,
                           objective=lambda t: round(sum(t.values()), 9)),
        RC, generator=gen)
    assert row_a["total_cost"] == pytest.approx(row_b["total_cost"])
    assert row_a["action_mix"]["counts"] == row_b["action_mix"]["counts"]
    assert row_a["n_decisions"] == row_b["n_decisions"]


def test_jr_new_completes_and_deterministic():
    gen = CandidateGenerator()
    rows = [run_joint_episode(_small_sim(730001),
                              JointRolloutGreedy(RC, generator=gen, objective=hierarchy_key),
                              RC, generator=gen) for _ in range(2)]
    assert rows[0]["completion_rate"] == 1.0 and rows[0]["backlog"] == 0
    assert rows[0]["total_cost"] == pytest.approx(rows[1]["total_cost"])
    assert rows[0]["action_mix"] == rows[1]["action_mix"]
