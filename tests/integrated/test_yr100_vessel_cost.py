"""YR-100 — 본선 비용 계산식 4계약 게이트 테스트 (spec: YR-100-vessel-cost-formula.md).

① 기준시간: surrogate 는 κ→0·margin=0 극한서 평가비용 33·max(0,F−pc)/3600 과 일치,
   상태에 ETD 없음(구조 강제). ② 공급경로: naive m·cadence 금지 — 굶주림 정지가 F 에.
③ 반사실: KEEP/TRANSFER 는 같은 now·pc 강제(assert), 공급측 차이만 비교.
④ 정보: truth(actual_*) 미열람 — truth 를 바꿔도 출력 불변.
"""
import math

import pytest

from yard_rl.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType
from yard_rl.integrated.vessel_cost import (
    VesselSupplyState, candidate_vessel_delta, compare_completion_cost, completion_cost,
    load_supply_state, overrun_cost, projected_completion_s, serve_supply_delta)

RHO, SC = 33.0, 3600.0


def _st(**kw):
    base = dict(now=0.0, planned_completion_s=1000.0, planned_start_s=0.0,
                cadence_s=100.0, remaining_moves=3, buffer_level=3,
                supply_etas=(), steady_onset_s=0.0, steady_pace_s=0.0)
    base.update(kw)
    return VesselSupplyState(**base)


# ---------------------------------------------------------------- 계약 ① 기준시간·surrogate
def test_gate1_no_etd_field():
    assert "etd_s" not in VesselSupplyState.__dataclass_fields__      # 구조적 강제


def test_gate1_hinge_limit_matches_eval_cost():
    for f, pc in ((1500.0, 1000.0), (900.0, 1000.0), (1000.0, 1000.0)):
        exact = RHO * max(0.0, f - pc) / SC
        assert overrun_cost(f, pc, kappa_s=0.0, margin_s=0.0) == pytest.approx(exact)
        # softplus 는 hinge 위쪽 κ·ln2 이내 (평가비용과의 괴리 상계)
        assert 0.0 <= overrun_cost(f, pc, kappa_s=300.0) - exact <= RHO * 300.0 * math.log(2) / SC + 1e-9


def test_gate1_margin_anticipatory():
    f, pc = 900.0, 1000.0                       # 아직 초과 전
    c0 = overrun_cost(f, pc, kappa_s=100.0, margin_s=0.0)
    c1 = overrun_cost(f, pc, kappa_s=100.0, margin_s=200.0)
    assert c1 > c0 > 0.0                        # margin = 선행 보수화(초과 전에도 신호)


def test_monotone_in_slack():
    f = 1000.0
    costs = [overrun_cost(f, pc) for pc in (1400.0, 1200.0, 1000.0, 800.0)]
    assert costs == sorted(costs)               # 마감이 당겨질수록(여유↓) 비용 단조↑


# ---------------------------------------------------------------- 계약 ② 공급경로 fold
def test_gate2_full_buffer_is_pure_cadence():
    st = _st(buffer_level=3, remaining_moves=3)
    assert projected_completion_s(st) == pytest.approx(300.0)          # 정지 0


def test_gate2_starvation_stall_included():
    # 버퍼 0·확정공급 1개(250s)·steady 없음(늦은 onset) → naive 300 보다 늦어야 함
    st = _st(buffer_level=0, remaining_moves=3, supply_etas=(250.0,),
             steady_onset_s=1000.0)
    f = projected_completion_s(st)
    assert f > 300.0
    # 1번째 move: 250 도착 후 시작→350, 2·3번째: steady 1000·1100 가용 → 1100+100
    assert f == pytest.approx(1200.0)


def test_gate2_more_supply_never_later():
    poor = _st(buffer_level=0, remaining_moves=4, supply_etas=(400.0,), steady_onset_s=900.0)
    rich = _st(buffer_level=2, remaining_moves=4, supply_etas=(400.0,), steady_onset_s=900.0)
    assert projected_completion_s(rich) <= projected_completion_s(poor)


def test_gate2_not_started_uses_planned_start():
    st = _st(planned_start_s=500.0, buffer_level=3, remaining_moves=2)
    assert projected_completion_s(st) == pytest.approx(700.0)


# ---------------------------------------------------------------- 계약 ③ 반사실
def test_gate3_same_now_pc_enforced():
    a = _st(now=0.0)
    b = _st(now=10.0)
    with pytest.raises(ValueError):
        compare_completion_cost(a, b)


def test_gate3_relieved_supply_wins():
    keep = _st(planned_completion_s=350.0, buffer_level=0, remaining_moves=3,
               steady_onset_s=300.0, steady_pace_s=150.0)      # 공급 느림 (크레인 눌림)
    xfer = _st(planned_completion_s=350.0, buffer_level=0, remaining_moves=3,
               steady_onset_s=150.0, steady_pace_s=100.0)      # 트럭 이전 → 공급 빨라짐
    d = compare_completion_cost(keep, xfer)
    assert d < 0.0                                             # TRANSFER 이득 = 음수
    assert compare_completion_cost(keep, keep) == pytest.approx(0.0)


# ---------------------------------------------------------------- 계약 ④ truth 미열람
class _FakeTransfer:
    n_units, move_time_s = 3, 180.0
    def waiting_count(self):
        return 0


class _FakeSim:
    def __init__(self, now, vessels):
        self.now = now
        self.vessels = vessels
        self.transfer = _FakeTransfer()
        self.jobs = {}


def _vessel(pc=2000.0, moves=5, started=True, buffer_level=1, work=VesselWorkType.LOAD):
    v = VesselProcess("V", work, VesselPlan(
        planned_start_s=0.0, planned_completion_s=pc, completion_basis=None,
        etd_s=pc + 999.0, total_moves=moves, sts_move_interval_s=100.0))
    if started:
        v.started, v.remaining_moves, v.buffer_level = True, moves, buffer_level
    return v


def test_gate4_truth_never_read():
    v = _vessel()
    sim = _FakeSim(100.0, {"V": v})
    a = load_supply_state(sim, v)
    v.truth.actual_completion_s = 123456.0                     # truth 오염
    b = load_supply_state(sim, v)
    assert a == b                                              # 출력 불변 = 미열람


def test_gate4_load_symptom_pc_visible():
    # LOAD 는 completion_basis=None(SYMPTOM 관측축)이어도 pc 는 비용 청구 기준이므로 가시
    v = _vessel()
    st = load_supply_state(_FakeSim(0.0, {"V": v}), v)
    assert st is not None and st.planned_completion_s == 2000.0


def test_discharge_excluded():
    v = _vessel(work=VesselWorkType.DISCHARGE)
    assert load_supply_state(_FakeSim(0.0, {"V": v}), v) is None


# ---------------------------------------------------------------- ExecutionQ delta 게이팅
def test_serve_delta_negative_when_urgent_starving():
    v = _vessel(pc=600.0, moves=4, buffer_level=0)             # 임박 + 굶주림
    d = serve_supply_delta(_FakeSim(100.0, {"V": v}), v, serve_duration_s=120.0)
    assert d < 0.0                                             # 본선 서빙 = 비용 감소


def test_serve_delta_vanishes_with_huge_slack():
    v = _vessel(pc=50_000.0, moves=4, buffer_level=0)          # 여유 막대
    d = serve_supply_delta(_FakeSim(100.0, {"V": v}), v, serve_duration_s=120.0)
    assert abs(d) < 1e-6                                       # "여유 있으면 트럭처럼"


def test_serve_delta_grows_as_deadline_tightens():
    ds = []
    for pc in (3000.0, 1500.0, 900.0):
        v = _vessel(pc=pc, moves=4, buffer_level=0)
        ds.append(serve_supply_delta(_FakeSim(100.0, {"V": v}), v, serve_duration_s=120.0))
    assert ds[0] >= ds[1] >= ds[2]                             # 급할수록 |이득| 커짐(더 음수)


def test_candidate_delta_empty_assign_zero():
    v = _vessel()
    assert candidate_vessel_delta(_FakeSim(0.0, {"V": v}), {}) == 0.0
