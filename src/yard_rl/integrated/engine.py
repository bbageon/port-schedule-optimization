"""통합 터미널 이벤트 시뮬레이터 — 다중 YC·본선·이송·레인 (YR-036).

단일 YC sim/engine.py 를 동결하고 순수 프리미티브만 재사용해 조립한 신규 엔진.
핵심: deferred commit(dispatch=예약·완료=스택 커밋 → 진행 중 이동 비관측·누출 0),
4종 자원 lock(ReservationTable), rate/delta 구간 비용 적분, clear-out drain, 결정론.
YR-050: 결정 시점은 SERVE 뿐 아니라 ETA 주도 선제 기회(PRE_ADVICE 한정)로도 열린다 —
wake 스케줄은 provided_eta·결정 지평만 사용(도착 진실 미열람), 술어는 candidates 와 공유.
"""
from __future__ import annotations

import copy
import hashlib

from ..domain.enums import CraneStatus, JobFlow, JobStatus, ServiceMode
from ..domain.models import Container
from ..sim.constraints import ConstraintViolation
from ..sim.kpis import KpiTracker
from ..sim.stack import YardStacks
from ..sim.travel_time import gantry_m, move_container, trolley_m
from .cost import CostAccumulator
from .cranes import CraneFleet
from .ledger import CostCause
from .events import EventKind, EventQueue
from .jobplan import JobPlan, JobRef, Move
from .lane import LaneNetwork
from .reservation import Corridor, Reservation, ReservationTable
from .transfer import TransferFleet
from .vessel import VesselWorkType

from dataclasses import dataclass
from ..contract.schema import CandidateKind
from ..domain.enums import InformationLevel
from .candidates import eta_opportunity

_EPS = 1e-9


@dataclass(frozen=True)
class TerminalDecision:
    time: float
    crane_ids: tuple[str, ...]


@dataclass(frozen=True)
class CraneAssignment:
    crane_id: str
    action: CandidateKind
    job_ref: JobRef | None = None
    # WAIT 사유 (resolver 판별: NO_FEASIBLE | LOST_CONTENTION) — 경합 이력 집계용 (YR-056)
    yield_reason: str | None = None


@dataclass(frozen=True)
class CommitProjection:
    """dry_run_commit 결과 — resolver 의 joint feasibility 오라클 (commit 과 동일 경로)."""

    plans: dict            # crane_id -> JobPlan (수용된 배정)
    reasons: dict          # crane_id -> reject 코드 (미수용)


class TerminalSimulator:
    def __init__(self, profile, scenario, *, check_invariants: bool = True,
                 info_level: InformationLevel = InformationLevel.BLOCK_ARRIVAL,
                 enable_cost_ledger: bool = False):
        self.profile = profile
        self.scenario = scenario
        self._check = check_invariants
        self.info_level = info_level     # PRE_REHANDLE/REPOSITION 정보시점 게이팅 (YR-037)
        self._enable_ledger = enable_cost_ledger   # 비용 인과 ledger (기본 off → golden 불변, YR-038)
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self._validate()
        geom = self.profile.block
        self.stacks = YardStacks(geom, copy.deepcopy(self.scenario.containers))
        self.jobs = {j.job_id: copy.deepcopy(j) for j in self.scenario.jobs}
        self.fleet = CraneFleet()
        for spec in self.profile.cranes:
            self.fleet.add(spec, geom.transfer_row)
        self.reservations = ReservationTable(self.profile.safety_gap_bay)
        from .ledger import CostLedger
        self.cost = CostAccumulator(ledger=CostLedger() if self._enable_ledger else None)
        self.kpis = KpiTracker(sla_s=self.profile.long_wait_sla_s)
        self.transfer = TransferFleet(
            self.profile.transfer.fleet_id, self.profile.transfer.kind,
            self.profile.transfer.n_units, self.profile.transfer.move_time_s)
        self.lanes = LaneNetwork(self.profile.lane_graph)
        self.vessels = {v.vessel_id: copy.deepcopy(v) for v in self.scenario.vessels}
        self.queue = EventQueue()
        self.clock = 0.0
        self.end = self.scenario.end_time
        self._terminal = False
        self._pending: tuple[str, ...] = ()
        # YR-075-a: 재조작 목적지 선택 훅 (opt-in). None(기본)이면 stk.find_slot greedy
        # 그대로 = 골든 바이트 동일. 설정 시 selector(sim, stk, blocker, spec, exclude)
        # → (bay,row)|None. 평가 오라클·후보 확장이 재사용 (rollout deepcopy 보존).
        self.slot_selector = getattr(self, "slot_selector", None)
        self._assigned: dict[str, CraneAssignment] = {}
        self._active_plans: dict[str, JobPlan] = {}
        self.event_log: list[tuple[float, str, str]] = []
        self.resolution_log: list = []       # JointResolution (YR-037, 계약 밖 side-channel)
        # YR-050 ETA wake 스케줄 — 외부 반출트럭 provided_eta 가 결정 지평에 들어오는 시각.
        # actual_* 은 절대 읽지 않는다(누출 0). queue 이벤트가 아니라 데이터로 두고
        # run_until_decision 이 PRE_ADVICE 일 때만 소비 → 낮은 정보수준의 이벤트 스트림·
        # golden 은 불변이고, info_level 을 reset 후에 바꿔도(record_episode) 정상 동작한다.
        # GATE_OUT 한정: 선제 재조작 기회(eta_opportunity)는 반출 대상에서만 생기므로
        # GATE_IN wake 는 항상 no-op — 시드하지 않는다.
        horizon = self.profile.decision_horizon_s
        self._eta_wakes: list[tuple[float, str]] = sorted(
            (max(0.0, j.provided_eta - horizon), j.job_id)
            for j in self.scenario.jobs
            if j.flow == JobFlow.GATE_OUT and j.target_container is not None
            and j.provided_eta is not None
            and max(0.0, j.provided_eta - horizon) < self.end)
        self._wake_idx = 0
        # wake 1회당 크레인별 1회 질문(armed) — 기회가 잔존하는 동안 모든 이벤트가 결정을
        # 재개방하면 WAIT-최하위 baseline 이 REPOSITION 을 반복 선택하는 되먹임(결정 464건·
        # REPO 88% 실측)이 생긴다. 거절한 크레인은 다음 wake 까지 선제 전용 결정을 안 연다.
        # SERVE 주도 결정에서는 PRE 후보가 기존(YR-048)대로 계속 발행된다.
        self._eta_armed: set[str] = set()
        self._seed_events()
        self._refresh_rates()

    def _validate(self):
        geom = self.profile.block
        occupied: set[tuple[int, int, int]] = set()
        for c in self.scenario.containers.values():
            if not (1 <= c.bay <= geom.bay_count and 1 <= c.row <= geom.row_count
                    and 1 <= c.tier <= geom.tier_max):
                raise ConstraintViolation("INVALID_SLOT", f"{c.container_id} 슬롯 범위 밖")
            slot = (c.bay, c.row, c.tier)
            if slot in occupied:
                raise ConstraintViolation("DUPLICATE_EVENT", f"슬롯 중복 {slot}")
            occupied.add(slot)
        for (b, r, t) in occupied:
            if t > 1 and (b, r, t - 1) not in occupied:
                raise ConstraintViolation("FLOATING_CONTAINER", f"({b},{r},{t}) 아래 빔")
        for j in self.jobs_input():
            if j.target_container is not None and j.target_container not in self.scenario.containers:
                raise ConstraintViolation("UNMATCHED_JOB", f"{j.job_id} 대상 부재")

    def jobs_input(self):
        return self.scenario.jobs

    def _seed_events(self):
        for j in sorted(self.scenario.jobs, key=lambda x: x.job_id):
            if j.is_external_truck:
                self.queue.push(j.actual_block_arrival, EventKind.BLOCK_ARRIVAL, j.job_id)
            else:
                # YR-080 단계2: 양하(STORE) job 은 시간 해제가 아니라 박스의 물리 도착
                # (STS→이송→TRANSFER_ARRIVE→VESSEL_RELEASED)으로 해제 — 인과 사슬 양하 절반.
                if j.is_vessel_linked and j.service_mode == ServiceMode.STORE:
                    continue
                self.queue.push(j.release_time, EventKind.JOB_RELEASED, j.job_id)
        for v in sorted(self.scenario.vessels, key=lambda x: x.vessel_id):
            self.queue.push(v.plan.planned_start_s, EventKind.VESSEL_START, v.vessel_id)
        for ie in sorted(self.scenario.injected_events, key=lambda x: (x.time, x.kind, x.target)):
            kind = {"EQUIPMENT_DOWN": EventKind.EQUIPMENT_DOWN,
                    "EQUIPMENT_UP": EventKind.EQUIPMENT_UP,
                    "PLAN_CHANGE": EventKind.PLAN_CHANGE}[ie.kind]
            self.queue.push(ie.time, kind, ie.target, data=ie.data)
        self.queue.push(self.scenario.horizon_s, EventKind.HORIZON, "HORIZON")

    # ------------------------------------------------------------- queries
    @property
    def now(self) -> float:
        return self.clock

    @property
    def terminal(self) -> bool:
        return self._terminal

    def observable_stacks(self) -> YardStacks:
        """정책 관측용 — 커밋된 스택만 (진행 중 이동 미반영)."""
        return self.stacks

    def cum_wait(self, job_id: str) -> float:
        """트럭 진실 누적대기 (realized_at 게이팅·비용 전용 — feature 값 아님)."""
        j = self.jobs.get(job_id)
        if j is None or not j.is_external_truck or j.actual_block_arrival is None:
            return 0.0
        if j.actual_block_arrival > self.clock:
            return 0.0
        return self.clock - j.actual_block_arrival

    def eligible_cranes(self, bay: int) -> tuple[str, ...]:
        return tuple(cid for cid in self.fleet.ids()
                     if self.fleet.spec(cid).service_bay_min <= bay <= self.fleet.spec(cid).service_bay_max)

    def _lane_for(self, bay: int) -> str | None:
        ids = self.profile.lane_graph.lane_ids
        return ids[(bay - 1) % len(ids)] if ids else None

    # ------------------------------------------------------------- main loop
    def run_until_decision(self) -> TerminalDecision | None:
        if self._pending:
            raise RuntimeError("직전 결정 미해소 — close_decision 먼저")
        while True:
            if self._terminal:
                return None
            nt = self.queue.peek_time()
            if nt is not None and nt <= self.clock + _EPS:
                self._process_next_event()
                continue
            if self._consume_due_wakes():   # YR-050: 동시각 이벤트 처리 후·결정 판정 전
                continue
            idle = self._decision_cranes()
            if idle and self.clock < self.end - _EPS:
                self._pending = idle
                self._assigned = {}
                self._eta_armed -= set(idle)   # 결정에 포함 = 이번 wake 의 질문 소진
                return TerminalDecision(self.clock, idle)
            wt = self._next_wake_time()
            if nt is None and wt is None:
                if any(c.state.assigned_job for c in self.fleet.all()):
                    raise RuntimeError("작업 중인데 완료 이벤트 없음 — 엔진 버그")
                self._finalize()
                return None
            if wt is not None and (nt is None or wt < nt - _EPS):
                self._advance(wt)           # 이벤트 없는 구간의 시계 전진 — 다음 순회에서 소비
                continue
            self._process_next_event()

    # ------------------------------------------------------ ETA wake (YR-050)
    def _next_wake_time(self) -> float | None:
        """미소비 wake 의 최소 시각 — PRE_ADVICE 외 정보수준에서는 항상 None(완전 비활성)."""
        if self.info_level != InformationLevel.PRE_ADVICE:
            return None
        if self._wake_idx >= len(self._eta_wakes):
            return None
        return self._eta_wakes[self._wake_idx][0]

    def _consume_due_wakes(self) -> bool:
        """현재 시각 도래 wake 를 전부 소비 — 전 크레인 arm + yield 해제로 결정 평가를 연다.

        wake 는 1회성(같은 시각 무한 WAIT 재결정 없음). arm 은 크레인이 결정에 한 번
        포함되면 소진 — 그 사이 바쁜 크레인은 armed 를 유지해 놓친 wake 를 유휴화 시점에
        받는다. 소비는 event_log 에 남아 PRE_ADVICE 의 event_stream_hash 에 가시화된다.
        """
        if self.info_level != InformationLevel.PRE_ADVICE:
            return False
        fired = False
        while (self._wake_idx < len(self._eta_wakes)
               and self._eta_wakes[self._wake_idx][0] <= self.clock + _EPS):
            _, jid = self._eta_wakes[self._wake_idx]
            self._wake_idx += 1
            self.event_log.append((self.clock, "ETA_WAKE", jid))
            fired = True
        if fired:
            self._eta_armed = set(self.fleet.ids())
            self._clear_yields()
        return fired

    def _decision_cranes(self) -> tuple[str, ...]:
        out = []
        for yc in self.fleet.all():
            if not yc.idle or yc.yielded:
                continue
            # SERVE 실행가능(전 정보수준). ETA 주도 선제 기회(PRE_ADVICE 한정, YR-050)는
            # wake 로 armed 된 크레인만 연다 — 술어는 후보 생성기와 공유(어긋남 차단).
            if (self.candidates_for(yc.crane_id)
                    or (yc.crane_id in self._eta_armed
                        and eta_opportunity(self, yc.crane_id, self.info_level))):
                out.append(yc.crane_id)
        return tuple(out)

    # ------------------------------------------------------------- candidates
    def candidates_for(self, crane_id: str) -> list[JobRef]:
        """이 크레인이 지금 실행 가능하고 예약 성공하는 SERVE 후보 (결정론 정렬).

        YR-037 이 mandatory 보존·pruning·pre-rehandle 로 교체할 seam. 여기서는 최소 열거.
        """
        yc = self.fleet.get(crane_id)
        if not yc.idle or yc.yielded:
            return []
        spec = self.fleet.spec(crane_id)
        out: list[JobRef] = []
        for jid in sorted(self.jobs):
            j = self.jobs[jid]
            if not self._dispatchable(j, crane_id):
                continue
            if self.reservations.job_taken(j.job_id) is not None:
                continue
            ref = self._jobref(j, spec, yc)
            if ref is None:
                continue
            plan = self._plan(crane_id, ref)
            if plan is None:
                continue
            if not self.reservations.can_reserve(self._reservation(plan)):
                continue
            out.append(ref)
        return out

    def _dispatchable(self, j, crane_id: str) -> bool:
        if j.status not in (JobStatus.WAITING, JobStatus.RELEASED):
            return False
        if j.assigned_crane is not None:
            return False
        spec = self.fleet.spec(crane_id)
        if j.target_container is not None:
            c = self.stacks.containers.get(j.target_container)
            if c is None or not c.work_available:
                return False
            if not (spec.service_bay_min <= c.bay <= spec.service_bay_max):
                return False
            if not self.stacks.rehandle_capacity_ok(j.target_container, spec):
                return False
        if j.service_mode == ServiceMode.STORE:      # 신규 반입 — 적재 슬롯 필요 (YR-080 §1)
            yc = self.fleet.get(crane_id)
            if self.stacks.find_slot(j.inbound_size, spec, yc.state.position_bay,
                                     yc.state.trolley_row) is None:
                return False
        return True

    def _jobref(self, j, spec, yc) -> JobRef | None:
        if j.target_container is not None:
            bay = self.stacks.containers[j.target_container].bay
        elif j.service_mode == ServiceMode.STORE:
            slot = self.stacks.find_slot(j.inbound_size, spec, yc.state.position_bay,
                                         yc.state.trolley_row)
            if slot is None:
                return None
            bay = slot[0]
        else:
            return None
        return JobRef(job_id=j.job_id, token=j.job_id, kind=CandidateKind.SERVE,
                      target_container=j.target_container, lane_id=self._lane_for(bay),
                      eligible_crane_ids=self.eligible_cranes(bay),
                      is_vessel=j.is_vessel_linked, is_external=j.is_external_truck)

    # ------------------------------------------------------------- planning
    def _plan(self, crane_id: str, ref: JobRef, *, extra_exclude=frozenset()) -> JobPlan | None:
        """스택 미변형으로 JobPlan 계산 (deferred). 실패 시 None.

        kind 분기: SERVE(GATE_IN 장치 / GATE_OUT·본선 반출) · PRE_REHANDLE(blocker 만 relocate,
        target 잔존) · REPOSITION(컨테이너 미조작, 빈 주행만). extra_exclude 는 dry_run 순차 예약.
        """
        yc = self.fleet.get(crane_id)
        spec = self.fleet.spec(crane_id)
        geom = self.profile.block
        cur_bay, cur_row = yc.state.position_bay, yc.state.trolley_row

        if ref.kind == CandidateKind.REPOSITION:
            tb = ref.reposition_target_bay
            if tb is None:
                return None
            tb = float(min(max(tb, spec.service_bay_min), spec.service_bay_max))
            dist = gantry_m(geom, cur_bay, tb)
            t_dist = trolley_m(geom, cur_row, geom.transfer_row)
            dur = dist / spec.gantry_speed_mps + t_dist / spec.trolley_speed_mps
            return JobPlan(crane_id=crane_id, job_id=ref.job_id, token=None,
                           kind=CandidateKind.REPOSITION, moves=(),
                           corridor=(min(cur_bay, tb), max(cur_bay, tb)), slots=frozenset(),
                           lane_id=None, start_s=self.clock, duration_s=dur, end_bay=tb,
                           end_row=float(geom.transfer_row), rehandles=0,
                           loaded_gantry_m=0.0, empty_gantry_m=dist)

        # YR-047: 가상 진행의 격리는 exclude 집합이 운반한다 — 각 blocker 의 목적지는 배치 즉시
        # exclude 에 들어가고, 원천 스택(대상과 같은 pile)은 호출마다 제외되므로, find_slot/
        # top_tier 는 가상으로 변형된 스택을 다시 읽지 않는다. 따라서 스택 전체 deepcopy 없이
        # 원본을 읽기 전용으로 써도 SERVE/REPOSITION 은 결과 동일 (등가성: test_plan_no_deepcopy.py).
        # 단 PRE_REHANDLE 의 slots·corridor 는 **의도적으로 구버전과 다르다**: 구버전은 place() 가
        # b.bay/b.row 를 목적지로 덮어쓴 '뒤' 장부를 기록하는 별칭(aliasing) 버그로 원천 pile 을
        # 예약에서 누락했다(과소예약 — SERVE 는 target 반출이 같은 pile 을 재추가해 우연히 은폐).
        # 신버전은 원천 pile 도 slots·corridor 에 포함한다 — 크레인이 실제로 주행·조작하는 범위.
        # stk 를 변형하는 호출(remove/place)을 추가하면 격리 전제가 깨진다 — 금지.
        stk = self.stacks
        exclude = set(self.reservations.reserved_slots()) | set(extra_exclude)
        moves: list[Move] = []
        touched_bays = {cur_bay}
        slots: set[tuple[int, int]] = set()
        total_s = loaded_m = empty_m = 0.0
        rehandles = 0
        j = self.jobs[ref.job_id]

        if j.service_mode == ServiceMode.STORE and ref.kind == CandidateKind.SERVE:
            dest = stk.find_slot(j.inbound_size, spec, cur_bay, cur_row, exclude=frozenset(exclude))
            if dest is None:
                return None
            db, dr = dest
            dtier = stk.top_tier(db, dr) + 1
            src = (db, geom.transfer_row, 1)
            mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
            inbound = Container(container_id=f"IN_{j.job_id}", size=j.inbound_size,
                                load_status=j.inbound_load, block=geom.block_id,
                                bay=db, row=dr, tier=dtier)
            moves.append(Move(inbound.container_id, src, (db, dr, dtier),
                              mv.loaded_gantry_m, mv.empty_gantry_m, mv.duration_s, inbound=inbound))
            total_s += mv.duration_s
            if j.is_external_truck:      # 본선 양하는 트럭 위치잡기 없음 (인계 0초 — 단계0 결정)
                total_s += spec.truck_positioning_time_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row
            touched_bays |= {db}
            slots.add((db, dr))
        else:
            target_id = j.target_container
            for blocker_id in stk.blockers_above(target_id):
                b = stk.containers[blocker_id]
                src = (b.bay, b.row, b.tier)
                excl = frozenset(exclude | {(b.bay, b.row)})
                if self.slot_selector is not None:      # YR-075-a opt-in (오라클/후보)
                    dest = self.slot_selector(self, stk, b, spec, excl)
                else:
                    dest = stk.find_slot(b.size, spec, float(b.bay), float(b.row),
                                         exclude=excl)
                if dest is None:
                    return None
                db, dr = dest
                dtier = stk.top_tier(db, dr) + 1
                # 구 place() 검증 승계 + exclude 계약 가드 — 위의 격리 전제가 기대는 후조건이므로
                # find_slot 이 변경되면 여기서 즉시 발화해야 한다 (조용한 물리 불일치 계획 방지).
                if (dtier > geom.tier_max or not stk.stack_size_ok(db, dr, b.size)
                        or dest in exclude or dest == (b.bay, b.row)):
                    raise RuntimeError(f"({db},{dr}) find_slot 후조건 위반")
                mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
                moves.append(Move(blocker_id, src, (db, dr, dtier),
                                  mv.loaded_gantry_m, mv.empty_gantry_m, mv.duration_s))
                total_s += mv.duration_s
                loaded_m += mv.loaded_gantry_m
                empty_m += mv.empty_gantry_m
                cur_bay, cur_row = mv.end_bay, mv.end_row
                rehandles += 1
                exclude.add((db, dr))
                touched_bays |= {b.bay, db}
                slots |= {(b.bay, b.row), (db, dr)}
            if ref.kind == CandidateKind.SERVE:
                # 대상 반출 (PRE_REHANDLE 은 blocker 만 치우고 target 잔존)
                target = stk.containers[target_id]
                src = (target.bay, target.row, target.tier)
                dst = (target.bay, geom.transfer_row, 1)
                mv = move_container(spec, geom, cur_bay, cur_row, src, dst)
                moves.append(Move(target_id, src, dst, mv.loaded_gantry_m, mv.empty_gantry_m,
                                  mv.duration_s, depart=True))
                total_s += mv.duration_s
                if j.is_external_truck:
                    total_s += spec.truck_positioning_time_s
                loaded_m += mv.loaded_gantry_m
                empty_m += mv.empty_gantry_m
                cur_bay, cur_row = mv.end_bay, mv.end_row
                touched_bays |= {target.bay}
                slots.add((target.bay, target.row))

        lo, hi = min(touched_bays), max(touched_bays)
        return JobPlan(crane_id=crane_id, job_id=j.job_id, token=ref.token, kind=ref.kind,
                       moves=tuple(moves), corridor=(lo, hi), slots=frozenset(slots),
                       lane_id=ref.lane_id, start_s=self.clock, duration_s=total_s,
                       end_bay=cur_bay, end_row=cur_row, rehandles=rehandles,
                       loaded_gantry_m=loaded_m, empty_gantry_m=empty_m)

    def _reservation(self, plan: JobPlan) -> Reservation:
        return Reservation(crane_id=plan.crane_id, job_token=plan.token,
                           corridor=Corridor(plan.corridor[0], plan.corridor[1]),
                           slots=plan.slots, lane_id=plan.lane_id,
                           release_at=plan.start_s + plan.duration_s)

    # ------------------------------------------------------------- decision commit
    def assign(self, crane_id: str, assignment: CraneAssignment) -> None:
        if crane_id not in self._pending:
            raise ConstraintViolation("DECISION_COVERAGE", f"{crane_id} 결정 대상 아님")
        if crane_id in self._assigned:
            raise ConstraintViolation("DECISION_COVERAGE", f"{crane_id} 이미 배정")
        yc = self.fleet.get(crane_id)
        if assignment.action == CandidateKind.WAIT:
            yc.yielded = True
            if assignment.yield_reason == "LOST_CONTENTION":
                yc.recent_yield_count += 1     # 경합 이력 (COORD feature, YR-056)
            self._assigned[crane_id] = assignment
            return
        ref = assignment.job_ref
        if ref is None:
            raise ConstraintViolation("BAD_ASSIGN", f"{crane_id} {assignment.action.value} 인데 job 없음")
        plan = self._plan(crane_id, ref)
        if plan is None:
            raise ConstraintViolation("NOT_DISPATCHABLE", f"{crane_id}:{ref.job_id} 계획 불가")
        self.reservations.reserve(self._reservation(plan))   # 2차 방어선
        self._active_plans[crane_id] = plan
        yc.state.assigned_job = plan.job_id
        yc.state.status = CraneStatus.HANDLING
        yc.state.available_at = plan.start_s + plan.duration_s
        yc.is_loaded = assignment.action != CandidateKind.REPOSITION
        # SERVE 만 job 을 RUNNING 으로 (PRE_REHANDLE 은 대상 job 을 서비스하지 않음 — PLANNED 잔존,
        # 동시 SERVE 는 token 예약이 차단). REPOSITION 은 job 없음.
        if assignment.action == CandidateKind.SERVE:
            j = self.jobs[plan.job_id]
            j.status = JobStatus.RUNNING
            j.assigned_crane = crane_id
            j.service_start = self.clock
            if j.is_external_truck:   # 서비스 시작 = dispatch 시점 (대기 적분 종료, engine.py:185)
                self.kpis.service_started(j.job_id, self.clock)
        self.cost.accrue("crane_travel", plan.loaded_gantry_m,
                         cause=CostCause.DISPATCH, subject=crane_id)
        self.cost.accrue("empty_travel", plan.empty_gantry_m,
                         cause=CostCause.DISPATCH, subject=crane_id)
        self.cost.accrue("rehandle", float(plan.rehandles),
                         cause=CostCause.DISPATCH, subject=crane_id)
        self.queue.push(plan.start_s + plan.duration_s, EventKind.JOB_COMPLETED, crane_id)
        self.event_log.append((self.clock, "DISPATCH", f"{crane_id}:{plan.job_id}"))
        self._assigned[crane_id] = assignment

    def close_decision(self) -> None:
        if set(self._assigned) != set(self._pending):
            raise ConstraintViolation("DECISION_COVERAGE",
                                      f"{set(self._assigned)} != {set(self._pending)}")
        self._pending = ()
        self._refresh_rates()
        if self._check:
            self.check_invariants()

    def commit_decisions(self, assignments: list[CraneAssignment]) -> None:
        for a in sorted(assignments, key=lambda x: x.crane_id):
            self.assign(a.crane_id, a)
        self.close_decision()

    def dry_run_commit(self, choices: dict) -> CommitProjection:
        """resolver joint-feasibility 오라클 — choices{crane_id→JobRef|None} 를 실제 커밋과
        동일 경로(_plan + reject_reason, crane_id 순)로 시뮬한다. 라이브 상태 미변형.

        불변식(D-ORACLE): dry_run.plans == commit 실제 배정 plans → resolver 가 수용한 joint 는
        assign→reserve() 에서 예외 없이 통과한다.
        """
        scratch = copy.deepcopy(self.reservations)
        plans: dict = {}
        reasons: dict = {}
        for cid in sorted(choices):
            ref = choices[cid]
            if ref is None:
                continue
            plan = self._plan(cid, ref, extra_exclude=scratch.reserved_slots())
            if plan is None:
                reasons[cid] = "NO_PLAN"
                continue
            r = self._reservation(plan)
            reason = scratch.reject_reason(r)
            if reason is not None:
                reasons[cid] = reason
                continue
            scratch.reserve(r)
            plans[cid] = plan
        return CommitProjection(plans, reasons)

    def last_assignments(self) -> dict[str, CraneAssignment]:
        return dict(self._assigned)

    def active_plan(self, crane_id: str):
        """실행 중 JobPlan (idle 이면 None) — 관측 계층의 상대 의도 산출용 (YR-056).

        이미 commit 된 사실(예약·이동 중 계획)만 노출 — 미래 예측·진실값 아님.
        """
        return self._active_plans.get(crane_id)

    # ------------------------------------------------------------- internals
    def _process_next_event(self):
        ev = self.queue.pop()
        self._advance(ev.time)
        self._handle(ev)
        self._refresh_rates()
        if self._check:
            self.check_invariants()

    def _advance(self, t: float):
        if t < self.clock - _EPS:
            raise RuntimeError(f"시간 역행 {self.clock}->{t}")
        if t > self.clock:
            lo, hi = min(self.clock, self.end), min(t, self.end)
            if hi > lo:
                q0, tl0 = self.kpis.queue_area_s, self.kpis.tail_area_s
                self.kpis.integrate(lo, hi)
                self.cost.accrue("truck_wait", self.kpis.queue_area_s - q0,
                                 cause=CostCause.WAIT_INTEGRAL)
                self.cost.accrue("long_wait", self.kpis.tail_area_s - tl0,
                                 cause=CostCause.WAIT_INTEGRAL)
                self.transfer.integrate(lo, hi)
                occ = frozenset(r.lane_id for r in self.reservations.active() if r.lane_id)
                self.lanes.integrate(lo, hi, occ)
                for v in self.vessels.values():
                    if v.sts_blocked and not v.done:
                        v.sts_wait_accum_s += (hi - lo)
                self.cost.advance(lo, hi)
            self.clock = t

    def _refresh_rates(self):
        self.cost.set_rate("sts_wait", sum(1 for v in self.vessels.values()
                                           if v.sts_blocked and not v.done))
        self.cost.set_rate("transfer_wait", self.transfer.waiting_count())
        occ = frozenset(r.lane_id for r in self.reservations.active() if r.lane_id)
        self.cost.set_rate("lane_cong", self.lanes.occupancy(occ)[0])
        self.cost.set_rate("interference", sum(1 for c in self.fleet.all() if c.yielded))
        self.cost.set_rate("imbalance", self.load_imbalance() / self.profile.shift_len_s)

    def load_imbalance(self) -> float:
        """§10.2 크레인별 **작업부하** 불균형 I(t) ∈ [0,1] (YR-043 재정의).

        Load_i(t) = 현재 작업 잔여시간 + 할당·예약 작업 예상 서비스시간 (idle=0).
        I(t) = (max−min)/Σ Load, Σ=0 이면 0.

        누적 완료건수 pstdev 는 폐기 — 처리건수 균등화는 사용자 목적이 아니었고, 누적·미정규화라
        총비용을 지배했다 (YR-039 무효 판정). rate=I/T_shift → ∫ 이 에피소드당 O(1).
        """
        loads = [max(0.0, c.state.available_at - self.clock) if c.state.assigned_job else 0.0
                 for c in self.fleet.all()]
        total = sum(loads)
        if total <= 0.0 or len(loads) < 2:
            return 0.0
        return (max(loads) - min(loads)) / total

    def _clear_yields(self):
        for c in self.fleet.all():
            c.yielded = False

    def _handle(self, ev):
        self.event_log.append((ev.time, ev.kind_name, ev.payload))
        k = ev.kind_name
        if k == "BLOCK_ARRIVAL":
            j = self.jobs[ev.payload]
            j.status = JobStatus.WAITING
            self.kpis.truck_arrived(j.job_id, ev.time)
            self._clear_yields()
        elif k == "JOB_RELEASED":
            self.jobs[ev.payload].status = JobStatus.RELEASED
            self._clear_yields()
        elif k == "JOB_COMPLETED":
            self._complete(ev.payload)
        elif k == "VESSEL_START":
            self._vessel_start(ev.payload)
        elif k == "STS_MOVE":
            self._sts_move(ev.payload)
        elif k == "TRANSFER_ARRIVE":
            self._transfer_arrive(ev.payload)
        elif k == "EQUIPMENT_DOWN":
            if ev.payload in self.fleet._cranes:
                self._equipment_down(ev.payload)
        elif k == "EQUIPMENT_UP":
            if ev.payload in self.fleet._cranes:
                yc = self.fleet.get(ev.payload)
                # down_pending 중(작업 진행 중) UP 이면 지연 DOWN 을 취소 — 정전이 실현 전 해소.
                yc.down = False
                yc.down_pending = False
                self._clear_yields()
        elif k == "PLAN_CHANGE":
            self._plan_change(ev.payload, ev.data)
        elif k == "VESSEL_RELEASED":
            # YR-080 단계2: 양하 박스 야드 인계 완료 → 해당 job 선택 가능.
            # 같은 시각 TRANSFER_ARRIVE(priority 2) 뒤에 처리(priority 3) — tie-break 고정.
            self.jobs[ev.payload].status = JobStatus.RELEASED
            self._clear_yields()
        elif k in ("HORIZON", "ETA_UPDATED"):
            pass
        else:
            raise RuntimeError(f"미지원 이벤트 {k}")

    def _complete(self, crane_id: str):
        plan = self._active_plans.pop(crane_id, None)
        yc = self.fleet.get(crane_id)
        if plan is None:
            raise RuntimeError(f"{crane_id} 완료 이벤트인데 활성 계획 없음")
        for mv in plan.moves:                      # 물리 실현 (dispatch 아님 — 여기서만)
            if mv.inbound is not None:
                self.stacks.place(mv.inbound, mv.dst[0], mv.dst[1])
            elif mv.depart:
                self.stacks.remove(mv.container_id)
            else:                                  # blocker 재배치 — 같은 객체를 재배치
                cont = self.stacks.containers[mv.container_id]
                self.stacks.remove(mv.container_id)
                self.stacks.place(cont, mv.dst[0], mv.dst[1])
        self.reservations.release(crane_id)
        yc.state.position_bay, yc.state.trolley_row = plan.end_bay, plan.end_row
        yc.state.assigned_job = None
        yc.state.status = CraneStatus.IDLE
        yc.state.available_at = self.clock
        yc.is_loaded = False
        yc.recent_completions += 1
        yc.state.loaded_travel_m += plan.loaded_gantry_m
        yc.state.empty_travel_m += plan.empty_gantry_m
        self.kpis.add_travel(plan.loaded_gantry_m, plan.empty_gantry_m)
        self.kpis.add_rehandles(plan.rehandles)
        # SERVE 만 대상 job 을 완료·served 계상 (PRE_REHANDLE 은 target 잔존·job 미완료,
        # REPOSITION 은 job 없음 — kind 가드로 jobs.get(sentinel/PLANNED) 오설정 차단).
        if plan.kind == CandidateKind.SERVE:
            yc.served_count += 1
            j = self.jobs.get(plan.job_id)
            if j is not None:
                j.status = JobStatus.DONE
                j.service_end = self.clock
                j.rehandle_count = plan.rehandles
                self.kpis.job_completed(external=j.is_external_truck, deadline=j.deadline,
                                        end=self.clock)
                # YR-080 단계3: 적하 반출 완료 → 박스를 안벽으로 이송 — 인과 사슬의
                # 적하 절반 (YC 반출 → YT → 안벽버퍼 → STS 처리 가능).
                if j.flow == JobFlow.VESSEL_LOAD and j.vessel_id is not None:
                    self._transfer_request(j.vessel_id)
        if yc.down_pending:
            yc.down, yc.down_pending = True, False
        self._clear_yields()

    # ------------------------------------------------------------- vessel/transfer
    def _vessel_start(self, vid: str):
        v = self.vessels[vid]
        if v.started:
            return
        v.started = True
        v.remaining_moves = v.plan.total_moves
        # YR-080 단계3: 적하 유령 pre-fill 삭제 — 안벽 버퍼의 박스는 야드 반출 완료가
        # 만든다(_complete → _transfer_request). 야드작업과 무관하게 버퍼가 차던
        # 인과 단절(가중치를 올려도 행동 불변, YR-080d)의 적하 절반 교정.
        self.queue.push(self.clock + v.plan.sts_move_interval_s, EventKind.STS_MOVE, vid)

    def _can_sts_process(self, v) -> bool:
        if v.work_type == VesselWorkType.DISCHARGE:
            return v.buffer_level < v.plan.quay_buffer_cap
        return v.buffer_level > 0

    def _sts_move(self, vid: str):
        v = self.vessels[vid]
        if v.done or not v.started:
            return
        if not self._can_sts_process(v):
            if v.sts_blocked_since_s is None:
                v.sts_blocked_since_s = self.clock
            return
        if v.sts_blocked_since_s is not None:
            v.sts_blocked_since_s = None
        if v.work_type == VesselWorkType.DISCHARGE:
            v.buffer_level += 1
            self._transfer_request(vid)      # 양하 박스 → 야드 이송 (도착 시 job 해제)
        else:
            v.buffer_level -= 1              # 적하 소비 — 보충은 야드 반출 완료가 (단계3)
        v.remaining_moves -= 1
        if v.remaining_moves <= 0:
            self._vessel_finish(vid)
        else:
            self.queue.push(self.clock + v.plan.sts_move_interval_s, EventKind.STS_MOVE, vid)

    def _transfer_request(self, vid: str):
        arrive = self.transfer.request(self.clock, vid)
        if arrive is not None:
            self.queue.push(arrive, EventKind.TRANSFER_ARRIVE, vid)

    def _release_next_discharge(self, vid: str):
        """양하 박스 1개 야드 인계 → 그 선박의 다음 PLANNED 양하 job 1건 해제 (FIFO).

        YR-080 단계2 — job_id 정렬 = 생성 순 FIFO (결정론). 이벤트로 우회해
        event_log 가시성·동시각 tie-break(TRANSFER_ARRIVE 2 → VESSEL_RELEASED 3) 고정.
        """
        for jid in sorted(self.jobs):
            j = self.jobs[jid]
            if (j.vessel_id == vid and j.status == JobStatus.PLANNED
                    and j.service_mode == ServiceMode.STORE):
                self.queue.push(self.clock, EventKind.VESSEL_RELEASED, jid)
                return

    def _transfer_arrive(self, vid: str):
        v = self.vessels[vid]
        if v.work_type == VesselWorkType.DISCHARGE:
            v.buffer_level = max(0, v.buffer_level - 1)   # box 야드 인계
            self._release_next_discharge(vid)             # YR-080: 도착 박스 = job 해제
        else:
            v.buffer_level += 1                           # box 안벽 staged
        nxt = self.transfer.dispatch_pending(self.clock)
        if nxt is not None:
            arrive, nvid = nxt
            self.queue.push(arrive, EventKind.TRANSFER_ARRIVE, nvid)
        if not v.done and v.sts_blocked_since_s is not None and self._can_sts_process(v):
            v.sts_blocked_since_s = None
            self.queue.push(self.clock, EventKind.STS_MOVE, vid)
        self._clear_yields()      # 버퍼 변화로 결정 지형이 바뀜 — yielded 크레인 재개방 (단계3)

    def _vessel_finish(self, vid: str):
        v = self.vessels[vid]
        v.done = True
        v.remaining_moves = 0
        v.sts_blocked_since_s = None
        v.truth.actual_completion_s = self.clock
        pc = v.plan.planned_completion_s
        if pc is not None and self.clock > pc:
            # YR-080 단계4: 비용(vessel_delay=선석 초과)과 보고 KPI(berth_overrun_s)를
            # **같은 호출부·같은 식**으로 동시 적립 — 학습이 최적화하는 양 == 보고하는 양.
            self.cost.accrue("vessel_delay", self.clock - pc,
                             cause=CostCause.VESSEL_FINISH, subject=vid)
            self.kpis.add_berth_overrun(self.clock - pc)
        if v.plan.etd_s is not None and self.clock > v.plan.etd_s:
            self.cost.accrue("depart_delay", self.clock - v.plan.etd_s,
                             cause=CostCause.VESSEL_FINISH, subject=vid)

    def _equipment_down(self, crane_id: str):
        yc = self.fleet.get(crane_id)
        if yc.state.assigned_job is not None:
            yc.down_pending = True         # 진행작업 무중단 (비선점) → 완료 후 DOWN
        else:
            yc.down = True

    def _plan_change(self, vid: str, data):
        v = self.vessels.get(vid)
        if v is None:
            return
        upd = dict(data or ())
        p = v.plan
        from .vessel import VesselPlan
        v.plan = VesselPlan(
            planned_start_s=p.planned_start_s,
            planned_completion_s=upd.get("planned_completion_s", p.planned_completion_s),
            completion_basis=upd.get("completion_basis", p.completion_basis),
            etd_s=upd.get("etd_s", p.etd_s), total_moves=p.total_moves,
            sts_move_interval_s=p.sts_move_interval_s, quay_buffer_cap=p.quay_buffer_cap)
        for jid, dl in upd.get("job_deadlines", ()):
            if jid in self.jobs:
                self.jobs[jid].deadline = dl

    # ------------------------------------------------------------- finalize
    def _finalize(self):
        if self._terminal:
            return
        self._advance(max(self.clock, self.end))
        for v in self.vessels.values():
            # YR-080 단계4: is_symptom 제외 → **선석 종료시각 존재** 기준으로 교정 —
            # 결정3(적하도 계획시각 부여, 관측 SYMPTOM 은 별개)에 따라 미완 적하도
            # 선석 초과를 정산한다 (이전엔 SYMPTOM 이라 미완 clear-out 에서 누락).
            pc = v.plan.planned_completion_s
            if not v.done and pc is not None and self.end > pc:
                self.cost.accrue("vessel_delay", self.end - pc,
                                 cause=CostCause.CLEAROUT, subject=v.vessel_id)
                self.kpis.add_berth_overrun(self.end - pc)
        self.kpis.close_censored(self.end)
        self._terminal = True

    # ------------------------------------------------------------- invariants
    def check_invariants(self):
        geom = self.profile.block
        seen: set[str] = set()
        for (bay, row), pile in self.stacks._stacks.items():
            if len(pile) > geom.tier_max:
                raise ConstraintViolation("TIER_OVERFLOW", f"({bay},{row})")
            for tier, cid in enumerate(pile, start=1):
                if cid in seen:
                    raise ConstraintViolation("DUP_CONTAINER", cid)
                seen.add(cid)
                c = self.stacks.containers[cid]
                if (c.bay, c.row, c.tier) != (bay, row, tier):
                    raise ConstraintViolation("POSITION_DESYNC", cid)
        if set(self.stacks.containers) != seen:
            raise ConstraintViolation("CONTAINER_LOST", "stack 불일치")
        busy = {c.crane_id for c in self.fleet.all() if c.state.assigned_job is not None}
        if set(self._active_plans) != busy:
            raise ConstraintViolation("PLAN_DESYNC", f"{set(self._active_plans)} != {busy}")
        if self.reservations.orphan_count() != len(self._active_plans):
            raise ConstraintViolation("RESERVE_DESYNC", "예약·계획 불일치")
        for c in self.fleet.all():
            spec = self.fleet.spec(c.crane_id)
            if not (spec.service_bay_min <= c.state.position_bay <= spec.service_bay_max):
                raise ConstraintViolation("OUT_OF_RANGE", f"{c.crane_id}")
        self._assert_pairwise_resources()

    def _assert_pairwise_resources(self):
        """상시 4-lock 무결성 — 삽입시점뿐 아니라 매 이벤트 후 active 예약 쌍별 검사 (YR-037)."""
        active = self.reservations.active()
        gap = self.reservations.safety_gap_bay
        for i in range(len(active)):
            for jx in range(i + 1, len(active)):
                a, b = active[i], active[jx]
                if a.job_token is not None and a.job_token == b.job_token:
                    raise ConstraintViolation("TOKEN_DOUBLE", f"{a.crane_id}·{b.crane_id} {a.job_token}")
                if a.lane_id is not None and a.lane_id == b.lane_id:
                    raise ConstraintViolation("LANE_DOUBLE", f"{a.crane_id}·{b.crane_id} {a.lane_id}")
                if a.corridor.overlaps(b.corridor, gap):
                    raise ConstraintViolation("CORRIDOR_OVERLAP", f"{a.crane_id}·{b.crane_id}")
                if a.slots & b.slots:
                    raise ConstraintViolation("SLOT_DOUBLE", f"{a.crane_id}·{b.crane_id}")

    def unfinished_backlog(self) -> int:
        return sum(1 for j in self.jobs.values()
                   if j.status in (JobStatus.PLANNED, JobStatus.WAITING, JobStatus.RELEASED))

    def event_stream_hash(self) -> str:
        blob = "|".join(f"{round(t, 6)}:{k}:{p}" for (t, k, p) in self.event_log)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _pstdev(xs: list[int]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
