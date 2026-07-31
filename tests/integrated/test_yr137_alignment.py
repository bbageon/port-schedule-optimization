"""YR-137 — v2 라벨·평가 정렬 계약 고정 (10차 피드백 회귀 테스트).

①미래 actual_gate_in 을 바꿔도 곡선 불변(공개 예약만 사용) ②실현 완료 = hard 초과비용
③양하(DISCHARGE)는 본선비용 0 ④관측 규칙(미진입 제외 = 평가 검열 일치).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from yard_rl.integrated.cost_curve_v2 import (KappaFit, delay_cost_curve_v2,
                                              j_truck_realized, j_vessel_realized,
                                              observed_gate_in)
from yard_rl.integrated.vessel import VesselWorkType


@pytest.fixture(scope="module")
def sim():
    from yard_rl.experiments.yr090_dense_vessel import BASE, _sim
    return _sim("high-tight", BASE["high-tight"])


def test_future_gate_in_mutation_invariant(sim):
    """미래 gate-in(사전 기록)을 변조해도 곡선이 변하면 안 된다 — 공개 예약이 1차."""
    kf = KappaFit.load()
    jid = next(jid for jid in sorted(sim.jobs)
               if getattr(sim.jobs[jid], "is_external_truck", False)
               and sim.jobs[jid].status.name == "PLANNED"
               and sim.jobs[jid].actual_gate_in is not None
               and sim.jobs[jid].actual_gate_in > sim.now)
    j = sim.jobs[jid]
    before = delay_cost_curve_v2(sim, jid, kf)
    saved = j.actual_gate_in
    try:
        j.actual_gate_in = saved + 1234.5          # 미래 실현 변조
        after = delay_cost_curve_v2(sim, jid, kf)
    finally:
        j.actual_gate_in = saved
    assert before == after


def test_observed_gate_in_rule(sim):
    j_future = next(j for j in sim.jobs.values()
                    if getattr(j, "is_external_truck", False)
                    and j.actual_gate_in is not None and j.actual_gate_in > sim.now)
    a = observed_gate_in(sim, j_future)
    assert a == pytest.approx(j_future.appointment_gate_time)   # 미래 → 공개 예약


def _stub_sim(now, jobs, vessels):
    prof = SimpleNamespace(long_wait_sla_s=1800.0, cranes=[])
    return SimpleNamespace(now=now, jobs=jobs, vessels=vessels, profile=prof,
                           fleet=None, end=1e9)


def test_label_uses_hard_for_realized_and_discharge_zero():
    from yard_rl.experiments.yr135_advantage_q import _j_total_v2
    kf = KappaFit(kappa_t_s=400.0, bias_t_s=0.0, kappa_v_s=1000.0, bias_v_s=0.0)
    st = SimpleNamespace(name="DONE")
    truck = SimpleNamespace(is_external_truck=True, status=st,
                            actual_gate_in=0.0, actual_gate_out=5000.0)
    load = SimpleNamespace(work_type=VesselWorkType.LOAD,
                           plan=SimpleNamespace(planned_completion_s=1000.0),
                           truth=SimpleNamespace(actual_completion_s=4600.0))
    disc = SimpleNamespace(work_type=VesselWorkType.DISCHARGE,
                           plan=SimpleNamespace(planned_completion_s=1000.0),
                           truth=SimpleNamespace(actual_completion_s=9000.0))
    s = _stub_sim(10000.0, {"T1": truck}, {"V-L": load, "V-D": disc})
    got = _j_total_v2(s, kf)
    # 기대 = hard 실현: 트럭(체류+초과) + 적하 hard — 양하는 0 이어야 함
    d_t = 300.0 + 1800.0 + 180.0 + 300.0
    want = j_truck_realized(5000.0, 0.0, 0.0 + d_t) + j_vessel_realized(4600.0, 1000.0)
    assert got == pytest.approx(want)
    # 양하 제거 전후 동일 → 양하 과금 0 확인
    s2 = _stub_sim(10000.0, {"T1": truck}, {"V-L": load})
    assert _j_total_v2(s2, kf) == pytest.approx(got)


def test_unentered_truck_excluded_censoring_match():
    from yard_rl.experiments.yr135_advantage_q import _j_total_v2
    kf = KappaFit(kappa_t_s=400.0, bias_t_s=0.0, kappa_v_s=1000.0, bias_v_s=0.0)
    st = SimpleNamespace(name="PLANNED")
    future = SimpleNamespace(is_external_truck=True, status=st,
                             actual_gate_in=99000.0, actual_gate_out=None)
    s = _stub_sim(10.0, {"T-F": future}, {})
    assert _j_total_v2(s, kf) == 0.0               # 미진입 = 라벨 미계상 (평가 검열 일치)


def test_realized_hard_formulas():
    assert j_truck_realized(4000.0, 0.0, 3000.0) == pytest.approx(4000 / 3600 + 1000 / 3600)
    assert j_truck_realized(2000.0, 0.0, 3000.0) == pytest.approx(2000 / 3600)   # 초과 0
    assert j_vessel_realized(1300.0, 1000.0) == pytest.approx(10 * 300 / 3600)
    assert j_vessel_realized(700.0, 1000.0) == 0.0
