"""단일 YC 이벤트 기반 시뮬레이터 — 구현계획 02 §1.

- 연속시간 이산사건, non-preemptive: 작업 시작 후 중단 없음.
- 정책 호출은 의사결정 시점(크레인 idle + dispatch 가능 작업 존재)에서만.
- clear-out: horizon 이후 신규 도착 없음, drain 구간 동안 잔여 작업 처리 허용,
  종료시점 미처리 대기도 queue-area 에 적분 (03 §2.2).
- 스택 변형은 dispatch 시점에 일괄 적용 (단일 YC + non-preemptive 라 관측 동등).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from ..domain.enums import CraneStatus, JobFlow, JobStatus
from ..domain.models import Container, CraneState, Job, TerminalProfile
from ..domain.scenario import Scenario
from ..domain.validators import validate_scenario
from .constraints import ConstraintEngine, ConstraintViolation
from .events import EventKind, EventQueue
from .kpis import KpiTracker
from .stack import YardStacks
from .travel_time import move_container

_EPS = 1e-9


@dataclass(frozen=True)
class DecisionPoint:
    time: float
    crane_id: str


@dataclass(frozen=True)
class ExecutionRecord:
    job_id: str
    start: float
    duration_s: float
    rehandles: int
    loaded_gantry_m: float
    empty_gantry_m: float


class YardSimulator:
    def __init__(self, profile: TerminalProfile, scenario: Scenario, *,
                 check_invariants: bool = True):
        self.profile = profile
        self.scenario = scenario
        self.constraints = ConstraintEngine(profile)
        self._check = check_invariants
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        validate_scenario(self.scenario.jobs, self.scenario.containers, self.profile)
        geom, spec = self.profile.block, self.profile.crane
        self.stacks = YardStacks(geom, copy.deepcopy(self.scenario.containers))
        self.jobs: dict[str, Job] = {j.job_id: copy.deepcopy(j) for j in self.scenario.jobs}
        self.crane = CraneState(
            crane_id=spec.crane_id,
            position_bay=float(spec.service_bay_min),
            trolley_row=float(geom.transfer_row),
            service_bay_min=spec.service_bay_min,
            service_bay_max=spec.service_bay_max,
        )
        self.kpis = KpiTracker(sla_s=self.profile.long_wait_sla_s)
        self.queue = EventQueue()
        self.clock = 0.0
        self._terminal = False
        self.event_log: list[tuple[float, str, str]] = []
        for j in sorted(self.jobs.values(), key=lambda x: x.job_id):
            if j.is_external_truck:
                self.queue.push(j.actual_block_arrival, EventKind.BLOCK_ARRIVAL, j.job_id)
            else:
                self.queue.push(j.release_time, EventKind.JOB_RELEASED, j.job_id)
        self.queue.push(self.scenario.horizon_s, EventKind.HORIZON, "HORIZON")

    # ------------------------------------------------------------- queries
    @property
    def now(self) -> float:
        return self.clock

    @property
    def terminal(self) -> bool:
        return self._terminal

    def crane_idle(self) -> bool:
        return self.crane.assigned_job is None

    def dispatchable_jobs(self) -> list[Job]:
        """진실 수준의 dispatch 가능 작업 (정보 필터는 env 책임). 결정론적 정렬."""
        out = [j for j in self.jobs.values()
               if self.constraints.is_dispatchable(j, self.crane, self.stacks)]
        out.sort(key=lambda j: j.job_id)
        return out

    # ------------------------------------------------------------- main loop
    def run_until_decision(self) -> DecisionPoint | None:
        """다음 의사결정 시점까지 진행. 종료 시 None (KPI 는 종료시각까지 적분됨)."""
        end = self.scenario.end_time
        while True:
            if self._terminal:
                return None
            if self.crane_idle() and self.clock < end - _EPS and self.dispatchable_jobs():
                return DecisionPoint(self.clock, self.crane.crane_id)
            nt = self.queue.peek_time()
            if nt is None:
                if not self.crane_idle():
                    raise RuntimeError("크레인 작업 중인데 완료 이벤트가 없음 — 엔진 버그")
                # 잔여 대기(미처리 backlog)의 대기시간을 종료시각까지 적분
                self._advance(max(self.clock, end))
                self._terminal = True
                return None
            ev = self.queue.pop()
            self._advance(ev.time)
            self._handle(ev)
            if self._check:
                self.constraints.check_invariants(self.stacks, self.jobs, self.crane, self.clock)

    # ------------------------------------------------------------- dispatch
    def execute_job(self, job_id: str) -> ExecutionRecord:
        """의사결정: 선택된 작업을 예약·실행 (재조작 포함). 완료 이벤트를 스케줄."""
        job = self.jobs[job_id]
        self.constraints.validate_assignment(job, self.crane, self.stacks)
        geom, spec = self.profile.block, self.profile.crane
        t0 = self.clock
        cur_bay, cur_row = self.crane.position_bay, self.crane.trolley_row
        total_s, loaded_m, empty_m, rehandles = 0.0, 0.0, 0.0, 0

        if job.flow == JobFlow.GATE_IN:
            dest = self.stacks.find_slot(job.inbound_size, spec, cur_bay, cur_row)
            if dest is None:  # is_dispatchable 이 걸렀어야 함 (2중 차단)
                raise ConstraintViolation("NO_SAFE_SLOT", f"{job_id} 장치슬롯 없음")
            db, dr = dest
            dtier = self.stacks.top_tier(db, dr) + 1
            src = (db, geom.transfer_row, 1)  # 트럭은 목적 bay 차선에 정차
            mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
            inbound = Container(container_id=f"IN_{job.job_id}", size=job.inbound_size,
                                load_status=job.inbound_load, block=geom.block_id,
                                bay=db, row=dr, tier=dtier)
            # place 는 tier 를 재계산하므로 생성 좌표는 참고값
            self.stacks.place(inbound, db, dr)
            total_s += mv.duration_s + spec.truck_positioning_time_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row
        else:
            # GATE_OUT / VESSEL_* : 야드 컨테이너 반출 (재조작 선처리 후 본 작업)
            target_id = job.target_container
            target = self.stacks.containers[target_id]
            for blocker_id in self.stacks.blockers_above(target_id):
                b = self.stacks.containers[blocker_id]
                src = (b.bay, b.row, b.tier)
                dest = self.stacks.find_slot(b.size, spec, float(b.bay), float(b.row),
                                             exclude={(b.bay, b.row)})
                if dest is None:
                    raise ConstraintViolation("NO_SAFE_SLOT", f"{job_id} 재조작 슬롯 없음")
                db, dr = dest
                dtier = self.stacks.top_tier(db, dr) + 1
                self.stacks.remove(blocker_id)
                mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
                self.stacks.place(b, db, dr)
                total_s += mv.duration_s
                loaded_m += mv.loaded_gantry_m
                empty_m += mv.empty_gantry_m
                cur_bay, cur_row = mv.end_bay, mv.end_row
                rehandles += 1
            src = (target.bay, target.row, target.tier)
            dst = (target.bay, geom.transfer_row, 1)  # 인계지점(차선)으로 상차
            self.stacks.remove(target_id)
            mv = move_container(spec, geom, cur_bay, cur_row, src, dst)
            total_s += mv.duration_s
            if job.is_external_truck:
                total_s += spec.truck_positioning_time_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row

        # 상태 갱신 (dispatch 시점 일괄)
        job.status = JobStatus.RUNNING
        job.assigned_crane = self.crane.crane_id
        job.service_start = t0
        job.rehandle_count = rehandles
        self.crane.assigned_job = job.job_id
        self.crane.status = CraneStatus.HANDLING
        self.crane.position_bay, self.crane.trolley_row = cur_bay, cur_row
        self.crane.available_at = t0 + total_s
        self.crane.loaded_travel_m += loaded_m
        self.crane.empty_travel_m += empty_m
        if job.is_external_truck:
            self.kpis.service_started(job.job_id, t0)
        self.kpis.add_travel(loaded_m, empty_m)
        self.kpis.add_rehandles(rehandles)
        self.queue.push(t0 + total_s, EventKind.JOB_COMPLETED, job.job_id)
        self.event_log.append((t0, "DISPATCH", job.job_id))
        return ExecutionRecord(job.job_id, t0, total_s, rehandles, loaded_m, empty_m)

    def skip_to_next_event(self):
        """모든 행동이 mask 된 경우: 다음 외부 이벤트까지 자동 진행 (02 §6)."""
        nt = self.queue.peek_time()
        if nt is None:
            self._advance(max(self.clock, self.scenario.end_time))
            self._terminal = True
            return
        ev = self.queue.pop()
        self._advance(ev.time)
        self._handle(ev)

    # ------------------------------------------------------------- internals
    def _advance(self, t: float):
        if t < self.clock - _EPS:
            raise RuntimeError(f"시간 역행: {self.clock} -> {t}")
        if t > self.clock:
            self.kpis.integrate(self.clock, t)
            self.clock = t

    def _handle(self, ev):
        self.event_log.append((ev.time, ev.kind_name, ev.payload))
        if ev.kind_name == "BLOCK_ARRIVAL":
            job = self.jobs[ev.payload]
            job.status = JobStatus.WAITING
            self.kpis.truck_arrived(job.job_id, ev.time)
        elif ev.kind_name == "JOB_RELEASED":
            self.jobs[ev.payload].status = JobStatus.RELEASED
        elif ev.kind_name == "JOB_COMPLETED":
            job = self.jobs[ev.payload]
            job.status = JobStatus.DONE
            job.service_end = ev.time
            self.crane.assigned_job = None
            self.crane.status = CraneStatus.IDLE
            self.kpis.job_completed(external=job.is_external_truck,
                                    deadline=job.deadline, end=ev.time)
        elif ev.kind_name == "HORIZON":
            pass
        else:
            raise RuntimeError(f"미지원 이벤트 {ev.kind_name}")

    # ------------------------------------------------------------- summary
    def unfinished_backlog(self) -> int:
        return sum(1 for j in self.jobs.values()
                   if j.status in (JobStatus.PLANNED, JobStatus.WAITING, JobStatus.RELEASED))
