"""통합 터미널 이벤트 시뮬레이터 — 다중 YC·본선·이송·레인 (YR-036).

단일 YC sim/engine.py 를 동결하고 순수 프리미티브만 재사용해 조립한 신규 엔진.
핵심: deferred commit(dispatch=예약·완료=스택 커밋 → 진행 중 이동 비관측·누출 0),
4종 자원 lock(ReservationTable), rate/delta 구간 비용 적분, clear-out drain, 결정론.
"""
from __future__ import annotations

import copy
import hashlib

from ..domain.enums import CraneStatus, JobFlow, JobStatus
from ..domain.models import Container
from ..sim.constraints import ConstraintViolation
from ..sim.kpis import KpiTracker
from ..sim.stack import YardStacks
from ..sim.travel_time import gantry_m, move_container, trolley_m
from .cost import CostAccumulator
from .cranes import CraneFleet
from .events import EventKind, EventQueue
from .jobplan import JobPlan, JobRef, Move
from .lane import LaneNetwork
from .reservation import Corridor, Reservation, ReservationTable
from .transfer import TransferFleet
from .vessel import VesselWorkType

from dataclasses import dataclass
from ..contract.schema import CandidateKind

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


class TerminalSimulator:
    def __init__(self, profile, scenario, *, check_invariants: bool = True):
        self.profile = profile
        self.scenario = scenario
        self._check = check_invariants
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
        self.cost = CostAccumulator()
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
        self._assigned: dict[str, CraneAssignment] = {}
        self._active_plans: dict[str, JobPlan] = {}
        self.event_log: list[tuple[float, str, str]] = []
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
            idle = self._decision_cranes()
            if idle and self.clock < self.end - _EPS:
                self._pending = idle
                self._assigned = {}
                return TerminalDecision(self.clock, idle)
            if nt is None:
                if any(c.state.assigned_job for c in self.fleet.all()):
                    raise RuntimeError("작업 중인데 완료 이벤트 없음 — 엔진 버그")
                self._finalize()
                return None
            self._process_next_event()

    def _decision_cranes(self) -> tuple[str, ...]:
        out = []
        for yc in self.fleet.all():
            if not yc.idle or yc.yielded:
                continue
            if self.candidates_for(yc.crane_id):
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
        if j.flow == JobFlow.GATE_IN:
            yc = self.fleet.get(crane_id)
            if self.stacks.find_slot(j.inbound_size, spec, yc.state.position_bay,
                                     yc.state.trolley_row) is None:
                return False
        return True

    def _jobref(self, j, spec, yc) -> JobRef | None:
        if j.target_container is not None:
            bay = self.stacks.containers[j.target_container].bay
        elif j.flow == JobFlow.GATE_IN:
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
    def _plan(self, crane_id: str, ref: JobRef) -> JobPlan | None:
        """스택 미변형으로 JobPlan 계산 (deferred). 실패 시 None."""
        yc = self.fleet.get(crane_id)
        spec = self.fleet.spec(crane_id)
        geom = self.profile.block
        work = copy.deepcopy(self.stacks)
        exclude = set(self.reservations.reserved_slots())
        cur_bay, cur_row = yc.state.position_bay, yc.state.trolley_row
        moves: list[Move] = []
        touched_bays = {cur_bay}
        slots: set[tuple[int, int]] = set()
        total_s = loaded_m = empty_m = 0.0
        rehandles = 0
        j = self.jobs[ref.job_id]

        if j.flow == JobFlow.GATE_IN:
            dest = work.find_slot(j.inbound_size, spec, cur_bay, cur_row, exclude=frozenset(exclude))
            if dest is None:
                return None
            db, dr = dest
            dtier = work.top_tier(db, dr) + 1
            src = (db, geom.transfer_row, 1)
            mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
            inbound = Container(container_id=f"IN_{j.job_id}", size=j.inbound_size,
                                load_status=j.inbound_load, block=geom.block_id,
                                bay=db, row=dr, tier=dtier)
            moves.append(Move(inbound.container_id, src, (db, dr, dtier),
                              mv.loaded_gantry_m, mv.empty_gantry_m, mv.duration_s, inbound=inbound))
            total_s += mv.duration_s + spec.truck_positioning_time_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row
            touched_bays |= {db}
            slots.add((db, dr))
        else:
            target_id = j.target_container
            for blocker_id in work.blockers_above(target_id):
                b = work.containers[blocker_id]
                src = (b.bay, b.row, b.tier)
                dest = work.find_slot(b.size, spec, float(b.bay), float(b.row),
                                      exclude=frozenset(exclude | {(b.bay, b.row)}))
                if dest is None:
                    return None
                db, dr = dest
                dtier = work.top_tier(db, dr) + 1
                work.remove(blocker_id)
                mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
                work.place(b, db, dr)
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
            target = work.containers[target_id]
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
            self._assigned[crane_id] = assignment
            return
        ref = assignment.job_ref
        if ref is None:
            raise ConstraintViolation("BAD_ASSIGN", f"{crane_id} SERVE 인데 job 없음")
        plan = self._plan(crane_id, ref)
        if plan is None:
            raise ConstraintViolation("NOT_DISPATCHABLE", f"{crane_id}:{ref.job_id} 계획 불가")
        self.reservations.reserve(self._reservation(plan))   # 2차 방어선
        self._active_plans[crane_id] = plan
        j = self.jobs[plan.job_id]
        j.status = JobStatus.RUNNING
        j.assigned_crane = crane_id
        j.service_start = self.clock
        yc.state.assigned_job = plan.job_id
        yc.state.status = CraneStatus.HANDLING
        yc.state.available_at = plan.start_s + plan.duration_s
        yc.is_loaded = True
        # 서비스 시작 = dispatch 시점 (단일 YC engine.py:185 계승). 대기 적분이 서비스시간을
        # 포함하지 않도록 여기서 _waiting 에서 제거 — 완료 시점이 아니라 dispatch 시점.
        if j.is_external_truck:
            self.kpis.service_started(j.job_id, self.clock)
        self.cost.accrue("crane_travel", plan.loaded_gantry_m)
        self.cost.accrue("empty_travel", plan.empty_gantry_m)
        self.cost.accrue("rehandle", float(plan.rehandles))
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

    def last_assignments(self) -> dict[str, CraneAssignment]:
        return dict(self._assigned)

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
                self.cost.accrue("truck_wait", self.kpis.queue_area_s - q0)
                self.cost.accrue("long_wait", self.kpis.tail_area_s - tl0)
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
        counts = [c.served_count for c in self.fleet.all()]
        self.cost.set_rate("imbalance", _pstdev(counts))

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
            self._equipment_down(ev.payload)
        elif k == "EQUIPMENT_UP":
            yc = self.fleet.get(ev.payload)
            # down_pending 중(작업 진행 중) UP 이면 지연 DOWN 을 취소 — 정전이 실현 전 해소.
            yc.down = False
            yc.down_pending = False
            self._clear_yields()
        elif k == "PLAN_CHANGE":
            self._plan_change(ev.payload, ev.data)
        elif k in ("HORIZON", "ETA_UPDATED", "VESSEL_RELEASED"):
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
        yc.served_count += 1
        yc.state.loaded_travel_m += plan.loaded_gantry_m
        yc.state.empty_travel_m += plan.empty_gantry_m
        self.kpis.add_travel(plan.loaded_gantry_m, plan.empty_gantry_m)
        self.kpis.add_rehandles(plan.rehandles)
        j = self.jobs.get(plan.job_id)
        if j is not None:
            j.status = JobStatus.DONE
            j.service_end = self.clock
            j.rehandle_count = plan.rehandles
            self.kpis.job_completed(external=j.is_external_truck, deadline=j.deadline,
                                    end=self.clock)
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
        if v.work_type == VesselWorkType.LOAD:
            for _ in range(min(v.plan.quay_buffer_cap, v.plan.total_moves)):
                self._transfer_request(vid)
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
        else:
            v.buffer_level -= 1
        self._transfer_request(vid)
        v.remaining_moves -= 1
        if v.remaining_moves <= 0:
            self._vessel_finish(vid)
        else:
            self.queue.push(self.clock + v.plan.sts_move_interval_s, EventKind.STS_MOVE, vid)

    def _transfer_request(self, vid: str):
        arrive = self.transfer.request(self.clock, vid)
        if arrive is not None:
            self.queue.push(arrive, EventKind.TRANSFER_ARRIVE, vid)

    def _transfer_arrive(self, vid: str):
        v = self.vessels[vid]
        if v.work_type == VesselWorkType.DISCHARGE:
            v.buffer_level = max(0, v.buffer_level - 1)   # box 야드 인계
        else:
            v.buffer_level += 1                           # box 안벽 staged
        nxt = self.transfer.dispatch_pending(self.clock)
        if nxt is not None:
            arrive, nvid = nxt
            self.queue.push(arrive, EventKind.TRANSFER_ARRIVE, nvid)
        if not v.done and v.sts_blocked_since_s is not None and self._can_sts_process(v):
            v.sts_blocked_since_s = None
            self.queue.push(self.clock, EventKind.STS_MOVE, vid)

    def _vessel_finish(self, vid: str):
        v = self.vessels[vid]
        v.done = True
        v.remaining_moves = 0
        v.sts_blocked_since_s = None
        v.truth.actual_completion_s = self.clock
        pc = v.plan.planned_completion_s
        if pc is not None and self.clock > pc:
            self.cost.accrue("vessel_delay", self.clock - pc)
        if v.plan.etd_s is not None and self.clock > v.plan.etd_s:
            self.cost.accrue("depart_delay", self.clock - v.plan.etd_s)

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
            if not v.done and not v.is_symptom() and self.end > v.plan.planned_completion_s:
                self.cost.accrue("vessel_delay", self.end - v.plan.planned_completion_s)
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
