"""YR-091 비통과 크레인 물리 — idle/down 상시 장벽·초기 분산·순서/간격 불변식 (외부감사 결함2).

감사 실측: idle 크레인 위치가 장애물로 예약되지 않아 비통과 RMG 가 관통(mid/high SF
실행당 1~7회) + shared 크레인 동일 시작 bay. 본 테스트가 그 재발을 구조로 차단한다.
"""
import pytest

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          run_joint_episode)
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.cost_config import RewardCalculator, neutral_lambda_config
from yard_rl.integrated.reservation import Corridor, Reservation, ReservationTable
from yard_rl.integrated.scenario_gen import generate_terminal_scenario
from yard_rl.sim.constraints import ConstraintViolation

PROF = build_integrated_profile()
RC = RewardCalculator(neutral_lambda_config())
SEED = 310000


def _sim():
    return TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED),
                             info_level=InformationLevel.PRE_ADVICE, check_invariants=True)


# ------------------------------------------------- 단위: idle 장벽
def test_idle_position_blocks_crossing_corridor():
    """감사 재현 시나리오: A 가 bay 20 에 idle, B 가 1→40 통과 시도 → 이제 거부."""
    t = ReservationTable(safety_gap_bay=2.0)
    t.set_idle_position("YC-A", 20.0)
    r = Reservation(crane_id="YC-B", job_token="J1", corridor=Corridor(1.0, 40.0),
                    slots=frozenset(), lane_id=None, release_at=100.0)
    assert t.reject_reason(r) == "CRANE_INTERFERENCE"
    with pytest.raises(ConstraintViolation):
        t.reserve(r)


def test_idle_barrier_respects_gap_and_own_position():
    t = ReservationTable(safety_gap_bay=2.0)
    t.set_idle_position("YC-A", 20.0)
    t.set_idle_position("YC-B", 5.0)
    # B 자신의 idle 위치는 자기 예약을 막지 않는다
    ok = Reservation("YC-B", "J1", Corridor(4.0, 10.0), frozenset(), None, 100.0)
    assert t.reject_reason(ok) is None
    # gap(2.0) 밖이면 허용: [1, 17.9] vs idle 20 → 17.9+2 < 20 ✗ (경계) → 17.5 로 확인
    near = Reservation("YC-B", "J2", Corridor(4.0, 17.5), frozenset(), None, 100.0)
    assert t.reject_reason(near) is None
    # gap 안(18.5+2 > 20)이면 거부
    tooclose = Reservation("YC-B", "J3", Corridor(4.0, 18.5), frozenset(), None, 100.0)
    assert t.reject_reason(tooclose) == "CRANE_INTERFERENCE"


def test_reserved_crane_uses_corridor_not_stale_idle_pos():
    """예약 중 크레인은 corridor 가 장벽 — 낡은 idle 위치는 무시된다."""
    t = ReservationTable(safety_gap_bay=2.0)
    t.set_idle_position("YC-A", 20.0)
    t.reserve(Reservation("YC-A", "JA", Corridor(30.0, 35.0), frozenset(), None, 100.0))
    # A 는 [30,35] 로 이동 중 — bay 20 부근은 이제 통과 가능해야 한다 (idle 항목 마스킹)
    b = Reservation("YC-B", "JB", Corridor(15.0, 25.0), frozenset(), None, 100.0)
    assert t.reject_reason(b) is None


# ------------------------------------------------- 엔진: 초기 분산·순서
def test_shared_cranes_start_spread_not_same_bay():
    sim = _sim()
    pos = [sim.fleet.get(c).state.position_bay for c in sim.fleet.ids()]
    assert len(set(pos)) == len(pos), "shared 크레인 동일 시작 bay 금지 (감사 결함)"
    gap = sim.reservations.safety_gap_bay
    ordered = sorted(pos)
    assert all(b - a >= gap for a, b in zip(ordered, ordered[1:]))
    # 상시 장벽이 초기부터 등록
    assert set(sim.reservations.idle_positions()) == set(sim.fleet.ids())


def test_full_episode_no_order_swap_and_invariants():
    """SF 전체 에피소드 — 매 이벤트 불변식(check_invariants=True)에 순서·간격 포함, 완주 유지."""
    sim = _sim()
    r = run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"), RC,
                          generator=CandidateGenerator())
    assert r["completion_rate"] == 1.0
    order = sim._rail_order
    pos = {c: sim.fleet.get(c).state.position_bay for c in order}
    assert all(pos[a] <= pos[b] for a, b in zip(order, order[1:]))


def test_order_swap_raises():
    sim = _sim()
    a, b = sim._rail_order[0], sim._rail_order[-1]
    pa = sim.fleet.get(a).state.position_bay
    sim.fleet.get(a).state.position_bay = sim.fleet.get(b).state.position_bay + 5.0
    with pytest.raises(ConstraintViolation):
        sim.check_invariants()
    sim.fleet.get(a).state.position_bay = pa      # 복원


def test_min_gap_violation_raises():
    sim = _sim()
    a, b = sim._rail_order[0], sim._rail_order[1]
    pa = sim.fleet.get(a).state.position_bay
    sim.fleet.get(a).state.position_bay = sim.fleet.get(b).state.position_bay - 0.5
    with pytest.raises(ConstraintViolation):
        sim.check_invariants()
    sim.fleet.get(a).state.position_bay = pa
