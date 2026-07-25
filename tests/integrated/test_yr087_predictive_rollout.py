"""YR-087 현실형 예측 rollout — 예측 교체 정확성 + 결정론 회귀 가드.

기존 rollout(오라클, 진짜 미래 사용) 대비, 미래 트럭 도착을 provided_eta±Uniform(eta_error) 예측으로
교체함을 고정. 원본 불변(deepcopy 격리)·트럭집합 보존·ETA 밴드 준수·동일 seed 결정론.
"""
import random

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator, neutral_lambda_config
from yard_rl.integrated.predictive_rollout import (DEFAULT_ETA_ERROR_S, PredictiveRollout,
                                                   make_predicted_scratch)
from yard_rl.integrated.scenario_gen import generate_terminal_scenario

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator(neutral_lambda_config())
SEED = 310000


def _sim_and_dp():
    s = TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED), info_level=LEVEL)
    dp = s.run_until_decision()
    return s, dp


def test_predicted_scratch_retimes_future_arrivals_within_eta_band():
    s, _ = _sim_and_dp()
    now = s.now
    future = {jid: j for jid, j in s.jobs.items()
              if getattr(j, "is_external_truck", False) and j.actual_block_arrival is not None
              and j.actual_block_arrival > now}
    assert future, "시나리오에 미도착 외부트럭이 있어야 함"
    orig = {jid: j.actual_block_arrival for jid, j in future.items()}
    eta = {jid: j.provided_eta for jid, j in future.items()}

    p = make_predicted_scratch(s, random.Random(0))

    # 원본 불변 (deepcopy 격리)
    assert all(s.jobs[jid].actual_block_arrival == orig[jid] for jid in future)
    # 트럭 집합 보존 (오는 트럭·목적지는 제공정보 — 도착시각만 예측)
    assert set(p.jobs) == set(s.jobs)
    changed = 0
    for jid in future:
        pt = p.jobs[jid].actual_block_arrival
        lo = max(now + 1.0, eta[jid] - DEFAULT_ETA_ERROR_S)
        hi = eta[jid] + DEFAULT_ETA_ERROR_S
        assert lo - 1e-6 <= pt <= hi + 1e-6, f"{jid} 예측 {pt} ∉ ETA밴드[{lo},{hi}]"
        if abs(pt - orig[jid]) > 1e-9:
            changed += 1
    # 예측은 truth 아님 — 대다수 도착시각이 진짜와 달라짐
    assert changed >= 0.8 * len(future)


def test_predicted_scratch_queue_intact_and_runs():
    """큐 재구성 후에도 엔진이 정상 진행·완주 (heapify 파손 없음)."""
    from yard_rl.integrated.baselines import _apply, _wait_of
    s, dp = _sim_and_dp()                # dp = 첫 결정 (p 는 이 pending 을 공유)
    p = make_predicted_scratch(s, random.Random(1))
    gen = CandidateGenerator()
    steps = 0
    while dp is not None and steps < 5000:
        gb = {c: gen.generate(p, c, LEVEL) for c in dp.crane_ids}
        _apply(p, {c: _wait_of(gb[c]) for c in dp.crane_ids})   # 결정 닫고 최소 진행 (WAIT)
        dp = p.run_until_decision()
        steps += 1
    assert dp is None, "예측 시나리오가 종결에 도달해야 함 (큐 정상)"


def test_predictive_rollout_decide_is_deterministic():
    s, dp = _sim_and_dp()
    gen = CandidateGenerator()
    gen_by = {c: gen.generate(s, c, LEVEL) for c in dp.crane_ids}
    a1 = PredictiveRollout(RC, horizon_s=600.0, k=2, seed=5).decide(s, dp, gen_by)
    a2 = PredictiveRollout(RC, horizon_s=600.0, k=2, seed=5).decide(s, dp, gen_by)
    assert set(a1) == set(dp.crane_ids)
    assert ({c: a1[c].candidate_id for c in a1} == {c: a2[c].candidate_id for c in a2})
