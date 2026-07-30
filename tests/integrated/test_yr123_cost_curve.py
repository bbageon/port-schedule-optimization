"""YR-123 — 공통 지연 한계비용 곡선 API 계약 고정 (단조·기울기·단위·결측·결정론)."""
from __future__ import annotations

import pytest

from yard_rl.experiments.yr090_dense_vessel import BASE, _sim
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.cost_curve import (DTS_DEFAULT, _truck_point, delay_cost_curve,
                                           marginal_delay_cost)

W, L, SLA, CAP = 1.0, 1.0, 1800.0, 1e9


def test_truck_pure_monotone_and_knee():
    waited = 600.0                                  # 남은 여유 1200s
    c1 = _truck_point(waited, SLA, 600.0, CAP, W, L)
    c2 = _truck_point(waited, SLA, 1200.0, CAP, W, L)
    c3 = _truck_point(waited, SLA, 1800.0, CAP, W, L)
    assert 0.0 < c1 < c2 < c3
    slope_pre = c1 / 600.0                          # 여유 안: w/h
    slope_post = (c3 - c2) / 600.0                  # 여유 밖: (w+l)/h
    assert slope_pre == pytest.approx(W / 3600.0)
    assert slope_post == pytest.approx((W + L) / 3600.0)


def test_truck_already_over_sla_double_rate():
    c = _truck_point(SLA + 100.0, SLA, 600.0, CAP, W, L)
    assert c == pytest.approx((W + L) * 600.0 / 3600.0)


def test_truck_window_censoring():
    assert _truck_point(0.0, SLA, 600.0, 300.0, W, L) == \
        pytest.approx(_truck_point(0.0, SLA, 300.0, 300.0, W, L))


def test_truck_unit_scaling_linear():
    base = _truck_point(1200.0, SLA, 900.0, CAP, W, L)
    twice = _truck_point(1200.0, SLA, 900.0, CAP, 2 * W, 2 * L)
    assert twice == pytest.approx(2.0 * base)


@pytest.fixture(scope="module")
def sim():
    from yard_rl.experiments.yr088_joint_rl import LEVEL
    from yard_rl.integrated.baselines import _apply, _wait_of
    from yard_rl.integrated.candidates import CandidateGenerator
    s = _sim("high-tight", BASE["high-tight"])
    gen = CandidateGenerator()
    for _ in range(60):             # 외부트럭이 실제 도착할 때까지 WAIT 로 전진
        if any(getattr(j, "is_external_truck", False) and j.status.name != "PLANNED"
               for j in s.jobs.values()):
            break
        dp = s.run_until_decision()
        if dp is None:
            break
        _apply(s, {c: _wait_of(gen.generate(s, c, LEVEL)) for c in dp.crane_ids})
    return s


def _first(sim, pred):
    for jid in sorted(sim.jobs):
        if pred(sim.jobs[jid]):
            return jid
    return None


def test_determinism_and_kinds(sim):
    rc = RewardCalculator.numeraire_v1()
    for jid in list(sorted(sim.jobs))[:12]:
        a = delay_cost_curve(sim, jid, rc=rc)
        b = delay_cost_curve(sim, jid, rc=rc)
        assert a == b                               # 결정론 (frozen dataclass 동등성)
        assert a.kind in ("truck", "truck_eta", "truck_no_eta",
                          "vessel_load", "vessel_discharge", "other")
        costs = [p.cost for p in a.points]
        assert costs == sorted(costs)               # Δt 단조
        for p in a.points:
            assert p.lo <= p.cost <= p.hi or (p.cost == 0.0 and p.hi >= 0.0)


def test_truck_curve_escalation_is_remaining_slack(sim):
    jid = _first(sim, lambda j: getattr(j, "is_external_truck", False)
                 and j.status.name != "PLANNED")
    if jid is None:
        pytest.skip("도착 트럭 없음 (t0)")
    cur = delay_cost_curve(sim, jid)
    sla = float(sim.profile.long_wait_sla_s)
    waited = max(0.0, sim.cum_wait(jid))
    if sla - waited <= max(DTS_DEFAULT):
        assert cur.escalation_start_s == pytest.approx(max(0.0, sla - waited))


def test_planned_truck_eta_has_zero_lower_band(sim):
    jid = _first(sim, lambda j: getattr(j, "is_external_truck", False)
                 and j.status.name == "PLANNED"
                 and getattr(j, "provided_eta", None) is not None)
    if jid is None:
        pytest.skip("ETA 있는 미도착 트럭 없음")
    cur = delay_cost_curve(sim, jid)
    assert cur.kind == "truck_eta"
    assert all(p.lo == 0.0 and p.hi >= p.cost for p in cur.points)


def test_planned_truck_no_eta_fail_closed(sim):
    jid = _first(sim, lambda j: getattr(j, "is_external_truck", False)
                 and j.status.name == "PLANNED")
    if jid is None:
        pytest.skip("미도착 트럭 없음")
    j = sim.jobs[jid]
    saved = j.provided_eta
    try:
        j.provided_eta = None
        cur = delay_cost_curve(sim, jid)
        assert cur.kind == "truck_no_eta"
        assert all(p.cost == 0.0 and p.hi > 0.0 for p in cur.points)   # 청구 0·상한만
    finally:
        j.provided_eta = saved


def test_vessel_load_curve_nonneg_monotone(sim):
    jid = _first(sim, lambda j: getattr(j, "vessel_id", None) is not None
                 and getattr(j, "flow", None) is not None
                 and j.flow.name == "VESSEL_LOAD")
    if jid is None:
        pytest.skip("본선 LOAD 작업 없음 (t0)")
    cur = delay_cost_curve(sim, jid)
    assert cur.kind == "vessel_load"
    costs = [p.cost for p in cur.points]
    assert all(c >= 0.0 for c in costs) and costs == sorted(costs)


def test_vessel_tight_slack_delay_costs_positive():
    """여유가 빡빡한 본선: 공급을 10분 늦추면 완료비용이 유의미하게 증가해야 한다."""
    from yard_rl.integrated.vessel_cost import VesselSupplyState, completion_cost
    base = dict(now=0.0, planned_completion_s=300.0, planned_start_s=0.0,
                cadence_s=120.0, remaining_moves=1, buffer_level=0,
                steady_onset_s=1e9, horizon_s=100000.0)   # 공급은 후보 박스뿐
    early = VesselSupplyState(supply_etas=(60.0,), **base)
    late = VesselSupplyState(supply_etas=(660.0,), **base)
    d = completion_cost(late) - completion_cost(early)
    assert d > 0.5                                       # 지연 → 선석초과 비용 실증


def test_single_point_matches_curve(sim):
    jid = sorted(sim.jobs)[0]
    p = marginal_delay_cost(sim, jid, 300.0)
    cur = delay_cost_curve(sim, jid, (300.0,))
    assert p == cur.points[0]
