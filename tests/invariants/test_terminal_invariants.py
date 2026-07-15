"""통합 시뮬레이터 불변조건 — 결정론·자원충돌·deferred commit·비용항등·누출 (YR-036).

check_invariants=True 로 실행 — 매 이벤트 후 위반 시 즉시 예외.
"""
import copy

import pytest

from yard_rl.contract import CandidateKind, dumps, loads
from yard_rl.contract.validate import validate_joint as _vj
from yard_rl.domain.enums import InformationLevel, JobStatus
from yard_rl.integrated import (CraneAssignment, ReferenceDispatcher, TerminalSimulator,
                               build_integrated_profile, build_minimal_terminal_scenario,
                               record_episode)
from yard_rl.integrated.reservation import Corridor, Reservation, ReservationTable
from yard_rl.sim.constraints import ConstraintViolation

PROF = build_integrated_profile()


def _fresh():
    return TerminalSimulator(PROF, build_minimal_terminal_scenario(), check_invariants=True)


def _run(sim):
    ReferenceDispatcher().run(sim)
    return sim


def test_full_run_completes_all_jobs():
    sim = _run(_fresh())
    assert sim.terminal
    assert sim.unfinished_backlog() == 0
    assert all(j.status == JobStatus.DONE for j in sim.jobs.values())
    assert sim.kpis.completed_external == 3
    assert sim.kpis.completed_vessel == 2
    assert all(v.done for v in sim.vessels.values())
    assert sim.reservations.orphan_count() == 0   # 완료 후 예약 해제


def test_determinism_two_constructions():
    s1, s2 = _run(_fresh()), _run(_fresh())
    assert s1.event_log == s2.event_log
    assert s1.event_stream_hash() == s2.event_stream_hash()
    assert s1.cost.episode_raw() == s2.cost.episode_raw()


def test_determinism_deepcopy_midrun():
    """중도 deepcopy 분기 후 양쪽 동일 (YR-031 beam 요건 — itertools.count 금지)."""
    sim = _fresh()
    disp = ReferenceDispatcher()
    for _ in range(2):
        dp = sim.run_until_decision()
        assert dp is not None
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=disp.select(sim, cid, cands))
                       if cands else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    branch = copy.deepcopy(sim)
    _run(sim)
    _run(branch)
    assert sim.event_stream_hash() == branch.event_stream_hash()
    assert sim.cost.episode_raw() == branch.cost.episode_raw()


def test_cost_interval_identity():
    """Σ cut()[k] == episode_raw()[k] — 중복·누락 0 (∀13항)."""
    sim = _fresh()
    disp = ReferenceDispatcher()
    total: dict[str, float] = {}
    while True:
        dp = sim.run_until_decision()
        for k, v in sim.cost.cut().items():
            total[k] = total.get(k, 0.0) + v
        if dp is None:
            break
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=disp.select(sim, cid, cands))
                       if cands else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    ep = sim.cost.episode_raw()
    for k in ep:
        assert abs(total.get(k, 0.0) - ep[k]) < 1e-6, k


def test_every_record_validates_and_roundtrips():
    sim = _fresh()
    recs = record_episode(sim, ReferenceDispatcher(),
                          info_level=InformationLevel.PRE_ADVICE, episode_id="EP")
    assert len(recs) >= 1
    for r in recs:
        _vj(r)                          # joint 제약 (dup token·lane·자격)
        assert loads(dumps(r)) == r     # round-trip
    assert recs[-1].terminal is True
    assert recs[-1].next_state is None
    assert recs[-1].next_observations == ()
    # 비-terminal record 는 next_state 보유
    if len(recs) >= 2:
        assert recs[0].next_state is not None


def test_deferred_commit_no_early_visibility():
    """작업 중 크레인의 미완료 이동은 observable_stacks 에 미반영 (누출 0)."""
    sim = _fresh()
    dp = sim.run_until_decision()
    # J-OUT-A (target C-A1, blocker C-A2@ (5,1,2)) 를 서비스하는 후보를 잡는다
    cid = dp.crane_ids[0]
    ref = next(r for r in sim.candidates_for(cid) if r.job_id == "J-OUT-A")
    others = [c for c in dp.crane_ids if c != cid]
    sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=ref))
    for oc in others:                    # 나머지 크레인은 WAIT (단일 관심작업 격리)
        sim.assign(oc, CraneAssignment(oc, CandidateKind.WAIT))
    sim.close_decision()
    # 완료 전: 스택은 그대로 (C-A1 최상단 아직 C-A2, C-A1 미반출)
    stk = sim.observable_stacks()
    assert "C-A1" in stk.containers and "C-A2" in stk.containers
    assert stk.containers["C-A2"].tier == 2    # blocker 아직 원위치
    # 완료까지 진행 → 물리 실현
    _run(sim)
    assert "C-A1" not in sim.stacks.containers  # 반출됨
    assert sim.stacks.containers["C-A2"].tier == 1  # blocker 재배치(내려앉음)


def test_equipment_down_defers_to_completion():
    """작업 중 EquipmentDown → 진행작업 무중단, 완료 후 DOWN (비선점)."""
    sim = _fresh()
    disp = ReferenceDispatcher()
    saw_down_pending = False
    while True:
        dp = sim.run_until_decision()
        yb = sim.fleet.get("YC-B")
        if yb.down_pending:
            saw_down_pending = True
        if dp is None:
            break
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=disp.select(sim, cid, cands))
                       if cands else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    assert sim.terminal   # 장애·복구 주입에도 완주


def test_plan_change_updates_vessel_completion():
    sim = _fresh()
    assert sim.vessels["V-DISCH"].plan.planned_completion_s == 7200.0
    # PLAN_CHANGE(t=1500) 이후로 진행
    while sim.now < 1600 and sim.run_until_decision() is not None:
        for cid in sim._pending:
            sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))
        if sim._pending:
            sim.close_decision()
    assert sim.vessels["V-DISCH"].plan.planned_completion_s == 6800.0


def test_reservation_table_rejects_conflicts():
    rt = ReservationTable(safety_gap_bay=2.0)
    rt.reserve(Reservation("YC-A", "T1", Corridor(4, 8), frozenset({(5, 1)}), "L1", 100.0))
    # 동일 token → DUP_JOB
    with pytest.raises(ConstraintViolation, match="DUP_JOB"):
        rt.reserve(Reservation("YC-B", "T1", Corridor(30, 32), frozenset({(30, 1)}), "L2", 100.0))
    # 동일 lane → LANE_CONFLICT
    with pytest.raises(ConstraintViolation, match="LANE_CONFLICT"):
        rt.reserve(Reservation("YC-B", "T2", Corridor(30, 32), frozenset({(30, 1)}), "L1", 100.0))
    # corridor 간섭 (gap 미만) → CRANE_INTERFERENCE
    with pytest.raises(ConstraintViolation, match="CRANE_INTERFERENCE"):
        rt.reserve(Reservation("YC-B", "T3", Corridor(9, 11), frozenset({(11, 1)}), "L2", 100.0))
    # 슬롯 충돌 → SLOT_CONFLICT
    with pytest.raises(ConstraintViolation, match="SLOT_CONFLICT"):
        rt.reserve(Reservation("YC-B", "T4", Corridor(30, 32), frozenset({(5, 1)}), "L2", 100.0))
    # 충분히 떨어진 corridor·다른 자원 → 성공
    rt.reserve(Reservation("YC-B", "T5", Corridor(30, 34), frozenset({(31, 1)}), "L2", 100.0))
    assert rt.orphan_count() == 2


def test_vessel_symptom_never_risk_in_records():
    """SYMPTOM 본선(V-LOAD)은 전 record 에서 risk known=0·symptom known=1."""
    sim = _fresh()
    recs = record_episode(sim, ReferenceDispatcher(),
                          info_level=InformationLevel.PRE_ADVICE, episode_id="EP")
    seen = 0
    for r in recs:
        for v in r.state.vessels:
            if v.vessel_id == "V-LOAD":
                seen += 1
                assert v.mode.value == "SYMPTOM"
                assert v.features.known_of("risk") is False
                assert v.features.known_of("delay_symptom_score") is True
    assert seen >= 1


def test_ablation_zeroes_group():
    """ablation_off={LANE} → 레인 그룹 필드 전부 known=0·value=0."""
    sim = _fresh()
    recs = record_episode(sim, ReferenceDispatcher(), info_level=InformationLevel.PRE_ADVICE,
                          episode_id="EP", ablation_off={"LANE"})
    r = recs[0]
    assert r.state.features.known_of("lane_congestion_mean") is False
    assert r.state.features.value_of("lane_congestion_mean") == 0.0
    for o in r.observations:
        assert o.candidates.queue_summary.known_of("lane_cong_mean") is False


def test_enum_ablation_normalized():
    """AblationGroup enum 을 넘겨도 audit.ablation_off 는 .value 로 정규화 (문자열 API 와 동치)."""
    from yard_rl.contract import AblationGroup
    recs = record_episode(_fresh(), ReferenceDispatcher(), info_level=InformationLevel.PRE_ADVICE,
                          episode_id="E", ablation_off=(AblationGroup.LANE,))
    assert recs[0].audit.ablation_off == ("LANE",)   # str(enum) 이 아니라 .value


def test_vessel_risk_ablation_runs():
    """VESSEL_RISK ablation 이 RISK 본선(V-DISCH)에서 크래시하지 않는다 (validate_vessel 면제)."""
    from yard_rl.contract import AblationGroup
    recs = record_episode(_fresh(), ReferenceDispatcher(), info_level=InformationLevel.PRE_ADVICE,
                          episode_id="E", ablation_off=(AblationGroup.VESSEL_RISK,))
    assert len(recs) >= 1
    v = recs[0].state.vessels[0]        # V-DISCH (RISK)
    assert v.features.known_of("risk") is False   # ablation 으로 known=0


def test_equipment_up_cancels_pending_down():
    """작업 중 EquipmentDown 후 완료 전 EquipmentUp → 지연 DOWN 취소 (영구 DOWN 방지)."""
    from yard_rl.integrated.scenario import InjectedEvent
    sc = build_minimal_terminal_scenario()
    sc.injected_events = [InjectedEvent(2000.0, "EQUIPMENT_DOWN", "YC-B"),
                          InjectedEvent(2050.0, "EQUIPMENT_UP", "YC-B")]
    sim = TerminalSimulator(PROF, sc, check_invariants=True)
    _run(sim)
    assert sim.terminal and sim.unfinished_backlog() == 0
    assert sim.fleet.get("YC-B").down is False
    assert sim.fleet.get("YC-B").down_pending is False


def test_truck_wait_excludes_service_time():
    """truck_wait 적분이 서비스시간을 포함하지 않음 — queue_area == Σ 대기표본 (dispatch 시점 종료)."""
    sim = _run(_fresh())
    assert sim.kpis.waiting_count() == 0
    er = sim.cost.episode_raw()
    assert abs(er["truck_wait"] - sum(sim.kpis.wait_samples_s)) < 1e-6
