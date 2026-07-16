"""YR-044 총비용용 baseline + 행동분포 건전성 계약.

핵심: 비교 기준(baseline)이 퇴화하면 어떤 승리 주장도 성립하지 않는다 (YR-039 무효 사유 2).
"""
from types import SimpleNamespace

import pytest

from yard_rl.contract.schema import CandidateKind
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (BaselinePreference, TerminalSimulator, build_integrated_profile)
from yard_rl.integrated.baselines import (ActionMix, ActionMixError, BeamLookahead,
                                         FIFOPreference, JointImmediateCostGreedy,
                                         JointRolloutGreedy, ResolverPolicy,
                                         ServiceFirstSPTPreference, assert_healthy_action_mix,
                                         run_joint_episode)
from yard_rl.integrated.cost_config import RewardCalculator, neutral_lambda_config
from yard_rl.integrated.scenario_gen import generate_terminal_scenario

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator(neutral_lambda_config())
SEED = 310000


def _sim():
    return TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED), info_level=LEVEL)


class _DegenerateSPT(BaselinePreference):
    """YR-039 퇴화 baseline 재현 — 전 kind 에 소요시간 적용 (짧은 REPOSITION 이 SERVE 를 이김)."""

    def rank(self, sim, cid, gc):
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        return (dur,) + super().rank(sim, cid, gc)


# ------------------------------------------------------- 건전성 계약 (핵심)
def test_health_contract_catches_yr039_degenerate_baseline():
    """YR-039 의 퇴화 SPT 를 계약이 잡아낸다 (무효 사유 2 재발 방지)."""
    r = run_joint_episode(_sim(), ResolverPolicy(_DegenerateSPT(), "SPT"), RC, level=LEVEL)
    assert r["_mix"].serve_when_available() < 0.25
    with pytest.raises(ActionMixError, match="퇴화"):
        assert_healthy_action_mix(r["_mix"], label="SPT")


def test_health_contract_passes_service_first_and_fifo():
    for pref, name in ((ServiceFirstSPTPreference(), "SF_SPT"), (FIFOPreference(), "FIFO")):
        r = run_joint_episode(_sim(), ResolverPolicy(pref, name), RC, level=LEVEL)
        assert_healthy_action_mix(r["_mix"], label=name)      # 예외 없이 통과
        assert r["completion_rate"] == 1.0


def test_health_contract_unit_thresholds():
    mix = ActionMix(counts={"SERVE": 5, "REPOSITION": 95}, serve_available=100, serve_taken=5)
    with pytest.raises(ActionMixError):
        assert_healthy_action_mix(mix)
    ok = ActionMix(counts={"SERVE": 60, "WAIT": 40}, serve_available=100, serve_taken=60)
    assert_healthy_action_mix(ok)


# ------------------------------------------------- 즉시비용 편향 (문서화 고정)
def test_immediate_cost_greedy_is_degenerate_by_construction():
    """무효판정 §6.2 문자 그대로의 '즉시비용(다음 결정까지) argmin' 은 짧은 행동을 우대해 퇴화.

    YR-044 실측: 완료율 41%·평균대기 119분. 진단 전용이며 baseline 으로 쓰지 않는다는 근거.
    """
    r = run_joint_episode(_sim(), JointImmediateCostGreedy(RC), RC, level=LEVEL)
    with pytest.raises(ActionMixError):
        assert_healthy_action_mix(r["_mix"], label="IMMEDIATE")


# ------------------------------------------------------------ 1차 baseline
def test_joint_rollout_greedy_is_healthy_and_competitive():
    """고정 시간창 rollout — 퇴화 없이 base 정책(ServiceFirstSPT) 이하 총비용."""
    base = run_joint_episode(_sim(), ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                             RC, level=LEVEL)
    policy = JointRolloutGreedy(RC, horizon_s=600.0)
    roll = run_joint_episode(_sim(), policy, RC, level=LEVEL)
    assert_healthy_action_mix(roll["_mix"], label="ROLLOUT")
    assert roll["completion_rate"] == 1.0
    assert roll["total_cost"] <= base["total_cost"]        # 1-step 정책개선
    assert roll["combo_truncations"] == policy.n_truncated  # 조용한 후보 축소 금지


def test_joint_rollout_counts_combo_truncation():
    """후보 조합을 max_combos로 줄이면 반드시 계수되어 평가 결과에 노출된다."""
    policy = JointRolloutGreedy(RC, max_combos=64)
    dp = SimpleNamespace(crane_ids=("YC-1", "YC-2"))
    gen_by = {cid: SimpleNamespace(items=tuple(SimpleNamespace(feasible=True)
                                                for _ in range(9)))
              for cid in dp.crane_ids}
    assert len(list(policy._combos(dp, gen_by))) == 64
    assert policy.n_truncated == 1


def test_joint_rollout_deterministic():
    a = run_joint_episode(_sim(), JointRolloutGreedy(RC, horizon_s=300.0), RC, level=LEVEL)
    b = run_joint_episode(_sim(), JointRolloutGreedy(RC, horizon_s=300.0), RC, level=LEVEL)
    assert a["total_cost"] == b["total_cost"]
    assert a["action_mix"] == b["action_mix"]


def test_beam_lookahead_runs_and_is_healthy():
    r = run_joint_episode(_sim(), BeamLookahead(RC, horizon_s=300.0, width=2), RC, level=LEVEL)
    assert_healthy_action_mix(r["_mix"], label="BEAM")
    assert r["completion_rate"] == 1.0


# ------------------------------------------------------------ 공정 비교 계약
def test_all_policies_share_constraints_and_complete():
    """전 정책 동일 정보·후보·제약·비용 config — 위반 0·완주."""
    for p in (ResolverPolicy(BaselinePreference(), "VW"),
              ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
              JointRolloutGreedy(RC, horizon_s=300.0)):
        r = run_joint_episode(_sim(), p, RC, level=LEVEL)
        assert r["backlog"] == 0 and r["completion_rate"] == 1.0
