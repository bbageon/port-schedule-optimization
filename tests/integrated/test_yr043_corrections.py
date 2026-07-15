"""YR-043 목적함수·행동공간 정정 회귀 가드 — YR-039 무효 사유 재발 방지.

근거: [무효 판정 §6](.claude/docs/strategy-history/2026-07-15-YR-039-무효판정-imbalance-지배.md)
"""
import pytest

from yard_rl.contract import CandidateKind
from yard_rl.contract.schema import COST_TERMS
from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import (BaselinePreference, CandidateGenerator, CentralResolver,
                               ReferenceDispatcher, TerminalSimulator, build_integrated_profile,
                               build_minimal_terminal_scenario, record_episode)
from yard_rl.integrated.cost_config import LambdaMode, neutral_lambda_config
from yard_rl.integrated.encoding import encode_observation
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from yard_rl.experiments.terminal_cost import (CostDominanceError, assert_no_dominance,
                                              contribution_shares)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
GEN = CandidateGenerator()


def _run(sc=None):
    sim = TerminalSimulator(PROF, sc or build_minimal_terminal_scenario(), info_level=LEVEL)
    recs = record_episode(sim, ReferenceDispatcher(), info_level=LEVEL, episode_id="E")
    return sim, recs


# ---------------------------------------------------------------- imbalance
def test_imbalance_is_load_based_and_bounded():
    """I(t) = (max−min)/ΣLoad ∈ [0,1] — 누적 완료건수 pstdev 폐기 (YR-039 무효 사유 1)."""
    sim, _ = _run()
    assert 0.0 <= sim.load_imbalance() <= 1.0
    # 전 크레인 idle → Load 합 0 → I=0
    assert sim.load_imbalance() == 0.0 or sim.terminal


def test_imbalance_no_longer_dominates():
    """imbalance 가 총비용을 지배하지 않는다 (YR-039: 97.6~99.9% → 정정)."""
    _, recs = _run(generate_terminal_scenario(PROF, 310000))
    shares = contribution_shares([r.cost for r in recs])
    assert shares["imbalance"] < 0.70, f"imbalance {shares['imbalance']:.1%} 지배 재발"


def test_imbalance_bounded_per_episode():
    """rate=I/T_shift → 에피소드 적산이 O(1) (누적·미정규화 폭주 방지)."""
    sim, _ = _run(generate_terminal_scenario(PROF, 310001))
    assert sim.cost.episode_raw()["imbalance"] < 10.0


# ------------------------------------------------------------ dominance guard
def test_dominance_guard_fires_on_yr039_pattern():
    with pytest.raises(CostDominanceError, match="지배"):
        assert_no_dominance({"imbalance": 0.976, "truck_wait": 0.024})


def test_dominance_guard_passes_balanced():
    assert_no_dominance({t: 1.0 / len(COST_TERMS) for t in COST_TERMS})


# ------------------------------------------------------------------ λ 중립
def test_neutral_lambda_is_static_one():
    """YR-043 트랙은 λ_vessel=1.0 고정 (동적 위험 비활성 — 본선 축 분리)."""
    cfg = neutral_lambda_config()
    assert cfg.lambda_vessel.mode == LambdaMode.STATIC
    assert all(cfg.lambda_vessel.lam(r) == 1.0 for r in (0.0, 0.3, 0.6, 0.9, 1.0))


# ------------------------------------------------------------------- WAIT
def test_wait_is_selectable_action():
    """WAIT 가 resolver pair 에 포함돼 정책이 실제로 선택 가능 (YR-043 복구)."""
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(), info_level=LEVEL)
    dp = sim.run_until_decision()
    gb = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}

    class WaitLover(BaselinePreference):
        def rank(self, sim, cid, gc):
            return (0, 0.0, "") if gc.job_ref is None else (9, 0.0, "z")

    res = CentralResolver(WaitLover()).resolve(sim, dp, gb)
    assert all(r.action == CandidateKind.WAIT for r in res.resolutions)
    assert all(r.chosen_candidate_id is None for r in res.resolutions)   # 계약 None⟺WAIT


def test_baseline_still_prefers_work_over_wait():
    """WAIT 복구가 baseline 거동을 바꾸지 않는다 (WAIT 최하위 tier)."""
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(), info_level=LEVEL)
    dp = sim.run_until_decision()
    gb = {c: GEN.generate(sim, c, LEVEL) for c in dp.crane_ids}
    res = CentralResolver(BaselinePreference()).resolve(sim, dp, gb)
    assert any(r.action == CandidateKind.SERVE for r in res.resolutions)


def test_wait_is_learnable_in_encoding():
    """encoding actionable 이 WAIT 를 포함하고 wait_pos 를 노출 (Q채점·replay 대상)."""
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(), info_level=LEVEL)
    dp = sim.run_until_decision()
    from yard_rl.integrated.adapter import capture
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "e", 0)
    enc = encode_observation(state, obs[0])
    assert enc.wait_pos is not None
    assert enc.actionable[enc.wait_pos] is True     # WAIT 도 학습 행동 (YR-043)
    assert enc.actionable == enc.selectable


# ------------------------------------------------------------------- mask
def test_pre_rehandle_window_not_masked():
    """§8.4 '도착 전 완료 가능'(pre_window)은 mask 아님 — 정보(ETA 가시)만 게이트."""
    import inspect
    src = inspect.getsource(GEN._pre_rehandle)
    assert "self.pre_window" not in src, "pre_window 가 다시 mask 게이트로 사용됨"


def test_info_gate_still_masks_eta():
    """정보 제약은 유지 — BLOCK_ARRIVAL 레벨에서 PRE_REHANDLE 미발행 (누출 0)."""
    sim = TerminalSimulator(PROF, build_minimal_terminal_scenario(),
                            info_level=InformationLevel.BLOCK_ARRIVAL)
    sim.run_until_decision()
    cid = sim.fleet.ids()[0]
    assert GEN._pre_rehandle(sim, cid, sim.now, InformationLevel.BLOCK_ARRIVAL) == []


# --------------------------------------------------------------- 시나리오
def test_gaussian_mean_condition_varies_and_is_deterministic():
    """평균조건 가우시안 — seed 별 μ 주변 변주, 동일 seed 결정론."""
    ns = [sum(1 for j in generate_terminal_scenario(PROF, s).jobs if j.is_external_truck)
          for s in (310000, 310001, 310002, 310003)]
    assert len(set(ns)) > 1                       # 변주 발생
    assert all(28 <= n <= 52 for n in ns)         # μ=40 ±2σ(12%) 근방
    a = generate_terminal_scenario(PROF, 310000)
    b = generate_terminal_scenario(PROF, 310000)
    assert [j.actual_block_arrival for j in a.jobs] == [j.actual_block_arrival for j in b.jobs]


def test_gaussian_off_gives_mu():
    sc = generate_terminal_scenario(PROF, 310000, TerminalGenParams(gaussian=False))
    assert sum(1 for j in sc.jobs if j.is_external_truck) == 40
