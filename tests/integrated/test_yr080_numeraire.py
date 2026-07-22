"""YR-080 단계 4 — 기준재 목적·정의 통일 계약.

① 기준재 항등식: 트럭대기 3600s → 기여 1.0, 선석초과 3600s → ρ_vessel, weight-0 항 → 0.
② λ 이중곱 금지: numeraire 에서 λ_vessel ≡ 1.0 (ρ_vessel 이 유일 배율).
③ 학습비용=보고 KPI 등식: episode_raw['vessel_delay'] == kpis.berth_overrun_s.
④ 결정 비율 스케일 불변: 보상 전체 상수배는 후보 순위를 바꾸지 않는다 (Ng 1999).
"""
from __future__ import annotations

from yard_rl.contract.schema import COST_TERMS
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import (NUMERAIRE_WEIGHT, RewardCalculator,
                                            numeraire_v1_config)
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario


def test_numeraire_identity():
    rc = RewardCalculator.numeraire_v1()
    raw = {t: 0.0 for t in COST_TERMS}
    raw["truck_wait"] = 3600.0        # 트럭대기 1h = 기준재 1.0
    raw["vessel_delay"] = 3600.0      # 선석초과 1h = ρ_vessel
    raw["interference"] = 12345.0     # weight 0 — 기여 0 이어야 함
    cb = rc.cost_for(interval_start_s=0.0, interval_end_s=1.0, raw=raw, risk_max=0.9)
    contrib = cb.contributions()
    assert abs(contrib["truck_wait"] - 1.0) < 1e-9
    assert abs(contrib["vessel_delay"] - NUMERAIRE_WEIGHT["vessel_delay"]) < 1e-9
    assert contrib["interference"] == 0.0
    zero_terms = [t for t, w in NUMERAIRE_WEIGHT.items() if w == 0.0]
    assert all(contrib[t] == 0.0 for t in zero_terms)
    assert abs(cb.total_normalized - (1.0 + NUMERAIRE_WEIGHT["vessel_delay"])) < 1e-6


def test_lambda_no_double_multiplication():
    """ρ_vessel 이 유일한 본선 배율 — λ 가 또 곱해지면 이중곱."""
    pol = numeraire_v1_config().lambda_vessel
    for risk in (0.0, 0.3, 0.6, 0.9, 1.0):
        assert pol.lam(risk) == 1.0


def test_berth_overrun_equals_cost_raw():
    """학습이 최적화하는 양(비용 raw) == 보고하는 양(KPI) — 정의 통일 계약.

    타이트 마감(vessel_deadline_mult 축소)으로 실제 초과를 유발해 0==0 자명 통과 방지.
    """
    profile = build_integrated_profile()
    params = TerminalGenParams(n_external=8, n_vessels=2, vessel_moves=6,
                               vessel_deadline_mult=0.35)
    sim = TerminalSimulator(profile, generate_terminal_scenario(profile, 890001, params),
                            check_invariants=True)
    run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                      RewardCalculator.numeraire_v1(), generator=CandidateGenerator())
    raw_vessel = sim.cost.episode_raw()["vessel_delay"]
    assert raw_vessel > 0.0, "초과 미발생 — 시나리오 긴장도 확인 (자명 통과 방지)"
    assert abs(raw_vessel - sim.kpis.berth_overrun_s) < 1e-6, (
        f"학습비용({raw_vessel}) != 보고 KPI({sim.kpis.berth_overrun_s}) — 정의 분열")


def test_decision_ratio_scale_invariant():
    """전체 상수배(학습 스케일)는 후보 간 순위를 보존 — 결정 비율과 학습 스케일 분리."""
    rc = RewardCalculator.numeraire_v1()
    a = {t: 0.0 for t in COST_TERMS}
    b = dict(a)
    a["truck_wait"], a["vessel_delay"] = 7200.0, 0.0      # 후보 A: 트럭 2h
    b["truck_wait"], b["vessel_delay"] = 0.0, 3600.0      # 후보 B: 선석초과 1h
    ta = rc.cost_for(interval_start_s=0, interval_end_s=1, raw=a, risk_max=0).total_normalized
    tb = rc.cost_for(interval_start_s=0, interval_end_s=1, raw=b, risk_max=0).total_normalized
    assert ta < tb                       # ρ_vessel=33 > 2 — B 가 더 비쌈
    for k in (0.1, 1.0, 10.0):
        assert (ta * k < tb * k) == (ta < tb)
