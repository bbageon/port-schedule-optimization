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


def test_kappa_explicit_wiring_fail_fast():
    """11차: 계약 물리 적합만 통과 — 구 물리 파일이면 즉시 실패."""
    from yard_rl.integrated.cost_curve_v2 import KAPPA_PATH, KAPPA_V2P_PATH
    kf = KappaFit.load(KAPPA_V2P_PATH, require_contract_physics=True)
    assert kf.kappa_t_s > 0 and kf.src_sha and "v2p" in kf.src_path
    with pytest.raises(ValueError):
        KappaFit.load(KAPPA_PATH, require_contract_physics=True)


def test_exact_window_all_candidates_same_end():
    """11차: 후보별 창 길이 불일치 제거 — scratch 가 정확히 t0+600 에 동결."""
    from yard_rl.experiments.yr088_joint_rl import LEVEL
    from yard_rl.experiments.yr135_advantage_q import (_exact_window_rollout,
                                                       _sim_contract)
    from yard_rl.integrated.baselines import (ResolverPolicy,
                                              ServiceFirstSPTPreference, _wait_of)
    from yard_rl.integrated.candidates import CandidateGenerator
    from yard_rl.experiments.yr090_dense_vessel import BASE
    s = _sim_contract("high-tight", BASE["high-tight"])
    dp = s.run_until_decision()
    gen = CandidateGenerator()
    base = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    t_end = s.now + 600.0
    for c in dp.crane_ids:                       # 서로 다른 후보(WAIT/첫 SERVE)로 확인
        items = [g for g in gen.generate(s, c, LEVEL).items if g.feasible]
        for g in items[:2]:
            assign = {cid: (_wait_of(gen.generate(s, cid, LEVEL)) if cid != c else g)
                      for cid in dp.crane_ids}
            sc = _exact_window_rollout(s, assign, gen, base, t_end)
            assert sc.now == pytest.approx(t_end)
    assert s.now < t_end                          # 원본 오염 없음 (deepcopy 격리)


def test_vessel_paired_no_effect_is_zero_and_joint_once():
    """11차: 무영향 후보 ΔV=0 · 같은 선박 2크레인 = joint 1회 계산."""
    from yard_rl.experiments.yr090_dense_vessel import BASE
    from yard_rl.experiments.yr135_advantage_q import _sim_contract
    from yard_rl.integrated.vessel_cost import (candidate_vessel_delta,
                                                vessel_serves_delta)
    s = _sim_contract("high-tight", BASE["high-tight"])
    s.run_until_decision()
    # 본선 서빙이 없는 조합(빈 assign) → 정확히 0
    assert candidate_vessel_delta(s, {}, kappa_s=900.0, rho=10.0) == 0.0
    v = next((x for x in s.vessels.values()
              if x.work_type.name == "LOAD" and not x.done), None)
    if v is None:
        pytest.skip("LOAD 본선 없음")
    d1 = vessel_serves_delta(s, v, (180.0, 240.0), kappa_s=900.0, rho=10.0)
    d2 = vessel_serves_delta(s, v, (180.0,), kappa_s=900.0, rho=10.0) \
        + vessel_serves_delta(s, v, (240.0,), kappa_s=900.0, rho=10.0)
    assert d1 <= 0.0                              # 공급 추가는 비용 증가 불가
    assert d1 >= d2 - 1e-9                        # joint 1회 ≥ 개별합 (이중계상 방지)


def test_label_truck_only_excludes_vessels():
    """11차: v3 라벨의 트럭 항에 본선 비용이 섞이면 이중계상 — include_vessels=False 확인."""
    from yard_rl.experiments.yr135_advantage_q import _j_total_v2
    kf = KappaFit(kappa_t_s=400.0, bias_t_s=0.0, kappa_v_s=1000.0, bias_v_s=0.0)
    load = SimpleNamespace(work_type=VesselWorkType.LOAD,
                           plan=SimpleNamespace(planned_completion_s=100.0),
                           truth=SimpleNamespace(actual_completion_s=5000.0))
    s = _stub_sim(10000.0, {}, {"V-L": load})
    assert _j_total_v2(s, kf, include_vessels=True) > 0.0
    assert _j_total_v2(s, kf, include_vessels=False) == 0.0


def test_future_gate_out_not_read_uses_public_anchor():
    """적대검증 minor 정정: 출문이 창 밖(미래 실현)이면 actual_gate_out 미열람 —
    공개 평균 앵커(service_end+300s)로 대체, 미래값 변조에 불변."""
    from yard_rl.experiments.yr135_advantage_q import _j_total_v2
    kf = KappaFit(kappa_t_s=400.0, bias_t_s=0.0, kappa_v_s=1000.0, bias_v_s=0.0)
    st = SimpleNamespace(name="DONE")
    t = SimpleNamespace(is_external_truck=True, status=st, actual_gate_in=0.0,
                        actual_gate_out=99999.0, service_end=9500.0)
    s = _stub_sim(10000.0, {"T": t}, {})
    d_t = 300.0 + 1800.0 + 180.0 + 300.0
    want = j_truck_realized(9500.0 + 300.0, 0.0, d_t)
    assert _j_total_v2(s, kf) == pytest.approx(want)
    t.actual_gate_out = 88888.0
    assert _j_total_v2(s, kf) == pytest.approx(want)


def test_realized_hard_formulas():
    assert j_truck_realized(4000.0, 0.0, 3000.0) == pytest.approx(4000 / 3600 + 1000 / 3600)
    assert j_truck_realized(2000.0, 0.0, 3000.0) == pytest.approx(2000 / 3600)   # 초과 0
    assert j_vessel_realized(1300.0, 1000.0) == pytest.approx(10 * 300 / 3600)
    assert j_vessel_realized(700.0, 1000.0) == 0.0
