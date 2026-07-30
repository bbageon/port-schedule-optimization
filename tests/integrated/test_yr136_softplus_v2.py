"""YR-136 v2.1 — softplus 계약·골든 (수용 기준 고정: 앵커·단조·가법성·한계·동결 κ)."""
from __future__ import annotations

import pytest

from yard_rl.integrated.cost_curve_v2 import (KappaFit, delay_cost_curve_v2,
                                              delta_cost_truck, delta_cost_vessel,
                                              j_truck, r_truck, r_vessel,
                                              truck_target_s)

K = 400.0


def test_anchor_rates_exact():
    assert r_truck(1000.0, 1000.0, K) == pytest.approx(1.5)      # D_T 에서 초과확률 50%
    assert r_vessel(500.0, 500.0, K) == pytest.approx(5.0)


def test_rate_monotone_and_bounds():
    rs = [r_truck(o, 0.0, K) for o in (-5000.0, -400.0, 0.0, 400.0, 5000.0)]
    assert rs == sorted(rs) and all(1.0 < r < 2.0 for r in rs[1:-1])
    assert rs[0] > 1.0 and rs[-1] < 2.0
    vs = [r_vessel(f, 0.0, K) for f in (-5000.0, 0.0, 5000.0)]
    assert vs == sorted(vs) and 0.0 < vs[0] and vs[-1] < 10.0


def test_delta_interval_additivity():
    for x in (-1200.0, -100.0, 300.0):
        a = delta_cost_truck(x, -3600.0, 0.0, 900.0, K)
        b = delta_cost_truck(x, -3600.0, 0.0, 300.0, K) \
            + delta_cost_truck(x + 300.0, -3600.0, 0.0, 600.0, K)
        assert a == pytest.approx(b, rel=1e-9)
        av = delta_cost_vessel(x, 0.0, 900.0, K)
        bv = delta_cost_vessel(x, 0.0, 300.0, K) + delta_cost_vessel(x + 300.0, 0.0, 600.0, K)
        assert av == pytest.approx(bv, rel=1e-9)


def test_delta_bounds_and_asymptotes():
    dt = 600.0
    early = delta_cost_truck(-50 * K, -3600.0, 0.0, dt, K)       # 마감 훨씬 전
    late = delta_cost_truck(50 * K, -3600.0, 0.0, dt, K)         # 마감 훨씬 후
    assert early == pytest.approx(dt / 3600.0, rel=1e-3)         # 기본 1/h
    assert late == pytest.approx(2.0 * dt / 3600.0, rel=1e-3)    # 상한 2/h
    assert delta_cost_vessel(-50 * K, 0.0, dt, K) == pytest.approx(0.0, abs=1e-6)
    assert delta_cost_vessel(50 * K, 0.0, dt, K) == pytest.approx(10.0 * dt / 3600.0,
                                                                  rel=1e-3)
    mid = delta_cost_truck(0.0, -3600.0, 0.0, dt, K)
    assert early < mid < late                                    # 지연 비용 비감소·단조


def test_j_truck_contains_base_dwell():
    # sp 항이 0 에 수렴하는 영역에서 J 는 순수 체류비용 (Ô−A)/3600
    assert j_truck(0.0, -7200.0, 1e9, K) == pytest.approx(2.0, rel=1e-6)


def test_kappa_fit_frozen_load():
    f1, f2 = KappaFit.load(), KappaFit.load()
    assert f1 == f2                                               # 결정론
    assert f1.kappa_t_s > 0 and f1.kappa_v_s > 0
    assert f1.n_truck >= 100 and f1.n_vessel >= 8                 # 적합 표본 실재


@pytest.fixture(scope="module")
def sim():
    from yard_rl.experiments.yr088_joint_rl import LEVEL
    from yard_rl.experiments.yr090_dense_vessel import BASE, _sim
    from yard_rl.integrated.baselines import _apply, _wait_of
    from yard_rl.integrated.candidates import CandidateGenerator
    s = _sim("high-tight", BASE["high-tight"])
    gen = CandidateGenerator()
    for _ in range(60):
        if any(getattr(j, "is_external_truck", False) and j.status.name == "WAITING"
               for j in s.jobs.values()):
            break
        dp = s.run_until_decision()
        if dp is None:
            break
        _apply(s, {c: _wait_of(gen.generate(s, c, LEVEL)) for c in dp.crane_ids})
    return s


def test_curve_v2_truck_contract(sim):
    kf = KappaFit.load()
    jid = next(jid for jid in sorted(sim.jobs)
               if getattr(sim.jobs[jid], "is_external_truck", False)
               and sim.jobs[jid].status.name == "WAITING")
    a = delay_cost_curve_v2(sim, jid, kf)
    b = delay_cost_curve_v2(sim, jid, kf)
    assert a == b and a.kind == "truck"
    costs = [p.cost for p in a.points]
    assert costs == sorted(costs) and all(p.lo == p.cost == p.hi for p in a.points)
    assert costs[0] >= 60.0 / 3600.0 - 1e-9                       # 기본 체류율 하한
    assert a.escalation_start_s is not None and a.escalation_start_s >= 0.0
    # escalation_start = 초과확률 50% 도달까지의 예측 여유 (D_T − 보정 Ô)
    j = sim.jobs[jid]
    from yard_rl.integrated.cost_curve_v2 import predict_gate_out
    d_t = truck_target_s(sim, j.actual_gate_in)
    o_c = predict_gate_out(sim, jid) + kf.bias_t_s
    assert a.escalation_start_s == pytest.approx(max(0.0, d_t - o_c))


def test_curve_v2_no_eta_fail_closed(sim):
    kf = KappaFit.load()
    jid = next((jid for jid in sorted(sim.jobs)
                if getattr(sim.jobs[jid], "is_external_truck", False)
                and sim.jobs[jid].status.name == "PLANNED"), None)
    if jid is None:
        pytest.skip("미도착 트럭 없음")
    j = sim.jobs[jid]
    saved = j.provided_eta
    try:
        j.provided_eta = None
        cur = delay_cost_curve_v2(sim, jid, kf)
        assert cur.kind == "truck_no_eta"
        assert all(p.cost == 0.0 and p.hi > 0.0 for p in cur.points)
    finally:
        j.provided_eta = saved


def test_curve_v2_vessel_load(sim):
    kf = KappaFit.load()
    jid = next((jid for jid in sorted(sim.jobs)
                if getattr(sim.jobs[jid], "vessel_id", None) is not None
                and getattr(sim.jobs[jid], "flow", None) is not None
                and sim.jobs[jid].flow.name == "VESSEL_LOAD"), None)
    if jid is None:
        pytest.skip("본선 LOAD 없음")
    cur = delay_cost_curve_v2(sim, jid, kf)
    assert cur.kind == "vessel_load"
    costs = [p.cost for p in cur.points]
    assert all(c >= 0.0 for c in costs) and costs == sorted(costs)
