"""정책용 최소 본선 신호 (스케줄중심) 회귀 테스트 — YR-088 파생.

계약: 정책 상태는 선박별 (schedule_slack, flow_margin) 요약 둘뿐(STS/YT raw 미노출).
schedule_slack 은 마감 다가올수록 줄고, flow_margin 은 STS 막힘과 정합(≤0)해야 한다.
"""
from __future__ import annotations

from statistics import mean

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (calibrated_load_params,
                                             generate_terminal_scenario)
from yard_rl.integrated.vessel_signals import (flow_margin_s, minimal_vessel_state,
                                               most_urgent, schedule_slack_s)

PROF = build_calibrated_profile()


def _episode(seed):
    """에피소드를 SF-SPT 로 돌며 결정마다 (vid, blocked, flow_margin, schedule_slack) 기록."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(
        PROF, seed, calibrated_load_params("high")), check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    gen, pol = CandidateGenerator(), ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    recs = []
    dp = sim.run_until_decision()
    while dp is not None:
        for vid, s in minimal_vessel_state(sim, sim.now).items():
            recs.append((vid, sim.vessels[vid].sts_blocked,
                         s["flow_margin_s"], s["schedule_slack_s"]))
        gen_by = {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        _apply(sim, pol.decide(sim, dp, gen_by))
        dp = sim.run_until_decision()
    return recs


def test_flow_margin_le_zero_when_sts_blocked():
    """STS 가 막힌 모든 결정 순간 flow_margin ≤ 0 (동행 지표 정확)."""
    for seed in range(820000, 820003):
        blocked = [fm for _, blk, fm, _ in _episode(seed) if blk and fm is not None]
        assert blocked, f"seed {seed}: STS 막힘 순간이 없음 — 시나리오 확인"
        assert all(fm <= 0 for fm in blocked), f"seed {seed}: 막혔는데 flow_margin>0"


def test_flow_margin_discriminates_block():
    """막힘 순간 flow_margin 평균 < 건강 순간 평균 (신호가 위험을 분별)."""
    recs = _episode(820001)
    blocked = [fm for _, blk, fm, _ in recs if blk and fm is not None]
    healthy = [fm for _, blk, fm, _ in recs if not blk and fm is not None]
    assert mean(blocked) < mean(healthy)


def test_schedule_slack_decreases_over_episode():
    """개시된 선박은 마감이 다가와 schedule_slack 이 시작보다 끝에서 작다."""
    recs = _episode(820001)
    first, last = {}, {}
    for vid, _, _, ss in recs:
        if ss is not None:
            first.setdefault(vid, ss)
            last[vid] = ss
    assert first, "schedule_slack 관측 없음"
    for vid in first:
        assert last[vid] < first[vid], f"{vid}: slack 이 안 줄음"


def test_minimal_state_only_two_summaries():
    """정책 상태는 요약 2개만 — STS/YT raw 누출 0 (최소화 계약)."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(
        PROF, 820000, calibrated_load_params("high")), check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    sim.run_until_decision()
    ms = minimal_vessel_state(sim, sim.now)
    assert ms, "본선 없음"
    for s in ms.values():
        assert set(s) == {"schedule_slack_s", "flow_margin_s"}


def test_most_urgent_picks_min_slack():
    """가장 급한 선박 = schedule_slack 최소 (평균 아닌 최악 보호)."""
    sim = TerminalSimulator(PROF, generate_terminal_scenario(
        PROF, 820001, calibrated_load_params("high")), check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    for _ in range(30):        # 몇 결정 진행해 선박 개시
        dp = sim.run_until_decision()
        if dp is None:
            break
        gen = CandidateGenerator()
        _apply(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF").decide(
            sim, dp, {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}))
    mu = most_urgent(sim, sim.now)
    if mu is not None:
        vid, s = mu
        allslack = [x["schedule_slack_s"] for x in minimal_vessel_state(sim, sim.now).values()
                    if x["schedule_slack_s"] is not None]
        assert s["schedule_slack_s"] == min(allslack)
