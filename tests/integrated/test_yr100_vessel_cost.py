"""YR-100 — 본선 비용 계산식 4계약 + 적대검증 정정 테스트 (spec: YR-100 게이트).

① 기준시간·검열캡: surrogate κ→0 극한 = 평가비용, horizon 캡 = CLEAROUT 정합, ETD 필드 부재.
② 공급 fold: 엔진 점화식 t_k=max(t_{k-1}+c, a_k) — 굶은 STS 도착 즉시 완료.
③ 반사실: 공급외 상태 전부 동일 assert. ④ truth 미열람.
pairing 정정: delta≤0 보장·duration 단조·joint 1회(2배 금지)·이송중 공급 주입.
"""
import math
from types import SimpleNamespace

import pytest

from yard_rl.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType
from yard_rl.integrated.vessel_cost import (
    VesselSupplyState, candidate_vessel_delta, compare_completion_cost, completion_cost,
    inflight_supply_etas, load_supply_state, overrun_cost, projected_completion_s,
    vessel_serves_delta)

RHO, SC = 33.0, 3600.0


def _st(**kw):
    base = dict(now=0.0, planned_completion_s=1000.0, planned_start_s=0.0,
                cadence_s=100.0, remaining_moves=3, buffer_level=3,
                supply_etas=(), steady_onset_s=0.0, steady_pace_s=0.0, horizon_s=None)
    base.update(kw)
    return VesselSupplyState(**base)


# ---------------------------------------------------------------- 계약 ① 기준시간·surrogate·캡
def test_gate1_no_etd_field():
    assert "etd_s" not in VesselSupplyState.__dataclass_fields__      # 구조적 강제


def test_gate1_hinge_limit_matches_eval_cost():
    for f, pc in ((1500.0, 1000.0), (900.0, 1000.0), (1000.0, 1000.0)):
        exact = RHO * max(0.0, f - pc) / SC
        assert overrun_cost(f, pc, kappa_s=0.0, margin_s=0.0) == pytest.approx(exact)
        assert 0.0 <= overrun_cost(f, pc, kappa_s=300.0) - exact <= RHO * 300.0 * math.log(2) / SC + 1e-9


def test_gate1_censoring_cap_matches_clearout():
    # 엔진 CLEAROUT = end−pc 상한. F 가 end 를 넘으면 캡 — 검열영역 두 세계 delta = 0
    a = _st(planned_completion_s=500.0, buffer_level=0, remaining_moves=5,
            steady_onset_s=2000.0, steady_pace_s=300.0, horizon_s=1200.0)
    b = _st(planned_completion_s=500.0, buffer_level=0, remaining_moves=5,
            steady_onset_s=1500.0, steady_pace_s=300.0, horizon_s=1200.0)
    assert projected_completion_s(a) > 1200.0 and projected_completion_s(b) > 1200.0
    assert compare_completion_cost(a, b) == pytest.approx(0.0)         # 유령 이득 차단


def test_gate1_rho_scale_injectable():
    assert overrun_cost(1500.0, 1000.0, kappa_s=0.0, rho=11.0) == pytest.approx(11.0 * 500.0 / SC)


def test_gate1_margin_anticipatory():
    c0 = overrun_cost(900.0, 1000.0, kappa_s=100.0, margin_s=0.0)
    c1 = overrun_cost(900.0, 1000.0, kappa_s=100.0, margin_s=200.0)
    assert c1 > c0 > 0.0


def test_monotone_in_slack():
    costs = [overrun_cost(1000.0, pc) for pc in (1400.0, 1200.0, 1000.0, 800.0)]
    assert costs == sorted(costs)


# ---------------------------------------------------------------- 계약 ② 엔진 점화식 fold
def test_gate2_full_buffer_is_pure_cadence():
    assert projected_completion_s(_st(buffer_level=3, remaining_moves=3)) == pytest.approx(300.0)


def test_gate2_starving_sts_completes_on_arrival():
    # b=0·eta 250·steady 1000: 엔진 = move@250(도착 즉시)·@1000·@1100 → F=1100 (1200 아님)
    st = _st(buffer_level=0, remaining_moves=3, supply_etas=(250.0,), steady_onset_s=1000.0)
    assert projected_completion_s(st) == pytest.approx(1100.0)


def test_gate2_more_supply_never_later():
    poor = _st(buffer_level=0, remaining_moves=4, supply_etas=(400.0,), steady_onset_s=900.0)
    rich = _st(buffer_level=2, remaining_moves=4, supply_etas=(400.0,), steady_onset_s=900.0)
    assert projected_completion_s(rich) <= projected_completion_s(poor)


def test_gate2_not_started_first_move_at_start_plus_cadence():
    st = _st(planned_start_s=500.0, buffer_level=3, remaining_moves=2)
    assert projected_completion_s(st) == pytest.approx(700.0)          # 엔진: start+cadence 첫 move


# ---------------------------------------------------------------- 계약 ③ 반사실
def test_gate3_supply_only_difference_enforced():
    with pytest.raises(ValueError):
        compare_completion_cost(_st(now=0.0), _st(now=10.0))
    with pytest.raises(ValueError):                                    # 공급외 상태(버퍼) 불일치 거부
        compare_completion_cost(_st(buffer_level=3), _st(buffer_level=1))


def test_gate3_relieved_supply_wins():
    keep = _st(planned_completion_s=350.0, buffer_level=0, remaining_moves=3,
               steady_onset_s=300.0, steady_pace_s=150.0)
    xfer = _st(planned_completion_s=350.0, buffer_level=0, remaining_moves=3,
               steady_onset_s=150.0, steady_pace_s=100.0)
    assert compare_completion_cost(keep, xfer) < 0.0
    assert compare_completion_cost(keep, keep) == pytest.approx(0.0)


# ---------------------------------------------------------------- 계약 ④ + 빌더
class _FakeTransfer:
    n_units, move_time_s = 3, 180.0
    def waiting_count(self):
        return 0


class _FakeSim:
    def __init__(self, now, vessels, heap=()):
        self.now = now
        self.vessels = vessels
        self.transfer = _FakeTransfer()
        self.jobs = {}
        self.queue = SimpleNamespace(_heap=list(heap))
        self.end = 100_000.0


def _vessel(pc=2000.0, moves=5, started=True, buffer_level=1, work=VesselWorkType.LOAD):
    v = VesselProcess("V", work, VesselPlan(
        planned_start_s=0.0, planned_completion_s=pc, completion_basis=None,
        etd_s=pc + 999.0, total_moves=moves, sts_move_interval_s=100.0))
    if started:
        v.started, v.remaining_moves, v.buffer_level = True, moves, buffer_level
    return v


def _ev(t, vid="V", kind="TRANSFER_ARRIVE"):
    return SimpleNamespace(time=t, payload=vid, kind_name=kind)


def test_gate4_truth_never_read():
    v = _vessel()
    sim = _FakeSim(100.0, {"V": v})
    a = load_supply_state(sim, v)
    v.truth.actual_completion_s = 123456.0
    assert load_supply_state(sim, v) == a


def test_gate4_load_symptom_pc_visible():
    v = _vessel()
    st = load_supply_state(_FakeSim(0.0, {"V": v}), v)
    assert st is not None and st.planned_completion_s == 2000.0


def test_inflight_supply_injected():
    v = _vessel(buffer_level=0)
    sim = _FakeSim(100.0, {"V": v}, heap=[_ev(400.0), _ev(50.0), _ev(600.0, vid="W"),
                                          _ev(500.0, kind="STS_MOVE")])
    assert inflight_supply_etas(sim, "V") == (400.0,)                  # 과거·타선박·타종류 제외
    st = load_supply_state(sim, v)
    assert 400.0 in st.supply_etas                                     # base 세계에 실공급 반영


def test_discharge_excluded():
    v = _vessel(work=VesselWorkType.DISCHARGE)
    assert load_supply_state(_FakeSim(0.0, {"V": v}), v) is None


# ---------------------------------------------------------------- pairing 정정 (delta 게이팅)
def test_delta_negative_when_urgent_starving():
    v = _vessel(pc=600.0, moves=4, buffer_level=0)
    d = vessel_serves_delta(_FakeSim(100.0, {"V": v}), v, (120.0,))
    assert d < 0.0


def test_delta_never_positive():
    for pc in (600.0, 1500.0, 3000.0):
        for b in (0, 2):
            v = _vessel(pc=pc, moves=4, buffer_level=b)
            assert vessel_serves_delta(_FakeSim(100.0, {"V": v}), v, (120.0,)) <= 1e-12


def test_delta_vanishes_with_huge_slack():
    v = _vessel(pc=50_000.0, moves=4, buffer_level=0)
    assert abs(vessel_serves_delta(_FakeSim(100.0, {"V": v}), v, (120.0,))) < 1e-6


def test_delta_grows_as_deadline_tightens():
    ds = []
    for pc in (3000.0, 1500.0, 900.0):
        v = _vessel(pc=pc, moves=4, buffer_level=0)
        ds.append(vessel_serves_delta(_FakeSim(100.0, {"V": v}), v, (120.0,)))
    assert ds[0] >= ds[1] >= ds[2]


def test_delta_duration_monotone_slower_less_credit():
    v = _vessel(pc=900.0, moves=4, buffer_level=0)
    sim = _FakeSim(100.0, {"V": v})
    d_fast = vessel_serves_delta(sim, v, (120.0,))
    d_slow = vessel_serves_delta(sim, v, (300.0,))
    assert d_fast <= d_slow <= 0.0                                     # knee 역전 제거


def test_delta_saturates_with_rich_buffer():
    v = _vessel(pc=900.0, moves=4, buffer_level=4)                     # 공급 과잉
    assert vessel_serves_delta(_FakeSim(100.0, {"V": v}), v, (120.0,)) == pytest.approx(0.0)


def test_joint_serving_not_double_counted():
    v = _vessel(pc=900.0, moves=6, buffer_level=0)
    sim = _FakeSim(100.0, {"V": v})
    single = vessel_serves_delta(sim, v, (120.0,))
    joint = vessel_serves_delta(sim, v, (120.0, 120.0))
    assert joint <= single < 0.0                                       # joint 가 더 큰 이득
    assert abs(joint) < 2.0 * abs(single) - 1e-9                       # 합산 2배 금지


def test_candidate_delta_empty_assign_zero():
    v = _vessel()
    assert candidate_vessel_delta(_FakeSim(0.0, {"V": v}), {}) == 0.0
