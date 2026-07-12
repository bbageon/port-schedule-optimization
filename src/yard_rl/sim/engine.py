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
from .constraints import CRANE_TASK_SENTINEL, ConstraintEngine, ConstraintViolation
from .events import EventKind, EventQueue
from .kpis import KpiTracker
from .stack import YardStacks
from .travel_time import gantry_m, move_container, trolley_m

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
        # True 면 dispatch 후보가 없어도 idle 시점에 의사결정을 낸다
        # (env 의 사전행동 — 포지셔닝·선재조작 — 용. env 가 scope 에 따라 설정)
        self.yield_idle_decisions = False
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
        """다음 의사결정 시점까지 진행. 종료 시 None.

        02 §1.2: 같은 시각의 이벤트를 전부 소진한 뒤에만 의사결정을 반환한다
        (동시 도착 작업이 후보에서 누락되지 않도록). 평가 윈도우는 [0, end_time]
        — KPI 적분은 end 에서 절단되며, end 이후는 이미 시작된 작업의 완료
        이벤트 처리만 일어난다 (non-preemptive drain).
        """
        end = self.scenario.end_time
        while True:
            if self._terminal:
                return None
            nt = self.queue.peek_time()
            # 동시각 이벤트 소진이 의사결정보다 먼저 (02 §1.2 처리순서)
            if nt is not None and nt <= self.clock + _EPS:
                self._process_next_event()
                continue
            if self.crane_idle() and self.clock < end - _EPS and (
                    self.yield_idle_decisions or self.dispatchable_jobs()):
                return DecisionPoint(self.clock, self.crane.crane_id)
            if nt is None:
                if not self.crane_idle():
                    raise RuntimeError("크레인 작업 중인데 완료 이벤트가 없음 — 엔진 버그")
                self._finalize()
                return None
            self._process_next_event()

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
            (dt, lm, em, rehandles,
             cur_bay, cur_row) = self._relocate_blockers(target_id, cur_bay, cur_row)
            total_s += dt
            loaded_m += lm
            empty_m += em
            target = self.stacks.containers[target_id]
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

    def _relocate_blockers(self, target_id: str, cur_bay: float, cur_row: float
                           ) -> tuple[float, float, float, int, float, float]:
        """대상 위 blocker 를 전부 최근접 합법슬롯으로 이동 (체이닝).

        반환: (소요시간, 적재 gantry m, 빈 gantry m, 재조작 수, 크레인 최종 bay/row)
        """
        geom, spec = self.profile.block, self.profile.crane
        total_s, loaded_m, empty_m, count = 0.0, 0.0, 0.0, 0
        for blocker_id in self.stacks.blockers_above(target_id):
            b = self.stacks.containers[blocker_id]
            src = (b.bay, b.row, b.tier)
            dest = self.stacks.find_slot(b.size, spec, float(b.bay), float(b.row),
                                         exclude={(b.bay, b.row)})
            if dest is None:  # rehandle_capacity_ok 가 걸렀어야 함 (2중 차단)
                raise ConstraintViolation("NO_SAFE_SLOT", f"{target_id} 재조작 슬롯 없음")
            db, dr = dest
            dtier = self.stacks.top_tier(db, dr) + 1
            self.stacks.remove(blocker_id)
            mv = move_container(spec, geom, cur_bay, cur_row, src, (db, dr, dtier))
            self.stacks.place(b, db, dr)
            total_s += mv.duration_s
            loaded_m += mv.loaded_gantry_m
            empty_m += mv.empty_gantry_m
            cur_bay, cur_row = mv.end_bay, mv.end_row
            count += 1
        return total_s, loaded_m, empty_m, count, cur_bay, cur_row

    # ---------------------------------------- 사전 행동 (control_scope 확장)
    def execute_positioning(self, target_bay: float) -> float:
        """빈 크레인 사전 포지셔닝 (plus_positioning, 실험설계안 §8.4).

        컨테이너를 잡지 않고 목표 bay 차선측으로 이동만 한다. 이동거리는
        빈 주행으로 계상 — 비용(w_M)과 편익(미래 작업 근접)을 정책이 학습.
        """
        if not self.crane_idle():
            raise ConstraintViolation("DOUBLE_ASSIGN", "포지셔닝: 크레인 사용 중")
        geom, spec = self.profile.block, self.profile.crane
        tb = float(min(max(target_bay, spec.service_bay_min), spec.service_bay_max))
        dist = gantry_m(geom, self.crane.position_bay, tb)
        t_dist = trolley_m(geom, self.crane.trolley_row, geom.transfer_row)
        dur = dist / spec.gantry_speed_mps + t_dist / spec.trolley_speed_mps
        self.crane.assigned_job = CRANE_TASK_SENTINEL
        self.crane.status = CraneStatus.MOVING
        self.crane.position_bay, self.crane.trolley_row = tb, float(geom.transfer_row)
        self.crane.empty_travel_m += dist
        self.kpis.add_travel(0.0, dist)
        self.kpis.positioning_count += 1
        self.queue.push(self.clock + dur, EventKind.JOB_COMPLETED, CRANE_TASK_SENTINEL)
        self.event_log.append((self.clock, "POSITIONING", f"bay={tb:g}"))
        return dur

    def execute_pre_rehandle(self, job_id: str) -> float:
        """도착 전 재조작 선처리 (plus_pre_rehandle, 02 §6.1).

        미도착 GATE_OUT 대상의 blocker 만 미리 치운다. 재조작 비용(w_R)은
        동일하게 계상 — 편익은 도착 후 서비스시간 단축으로 실현된다.
        """
        job = self.jobs[job_id]
        if not self.crane_idle():
            raise ConstraintViolation("DOUBLE_ASSIGN", "선재조작: 크레인 사용 중")
        target_id = job.target_container
        c = self.stacks.containers.get(target_id)
        if c is None or not c.work_available:
            raise ConstraintViolation("NOT_DISPATCHABLE", f"{job_id} 선재조작 대상 없음")
        if not (self.crane.service_bay_min <= c.bay <= self.crane.service_bay_max):
            raise ConstraintViolation("OUT_OF_RANGE", f"{job_id} 대상 bay {c.bay}")
        blockers = self.stacks.blockers_above(target_id)
        if not blockers:
            raise ConstraintViolation("NOT_DISPATCHABLE", f"{job_id} blocker 없음")
        if not self.stacks.rehandle_capacity_ok(target_id, self.profile.crane):
            raise ConstraintViolation("NO_SAFE_SLOT", f"{job_id} 선재조작 슬롯 없음")
        t0 = self.clock
        (dur, lm, em, cnt,
         cb, cr) = self._relocate_blockers(target_id, self.crane.position_bay,
                                           self.crane.trolley_row)
        self.crane.assigned_job = CRANE_TASK_SENTINEL
        self.crane.status = CraneStatus.HANDLING
        self.crane.position_bay, self.crane.trolley_row = cb, cr
        self.crane.loaded_travel_m += lm
        self.crane.empty_travel_m += em
        self.kpis.add_travel(lm, em)
        self.kpis.add_rehandles(cnt)
        self.kpis.pre_rehandle_count += cnt
        self.queue.push(t0 + dur, EventKind.JOB_COMPLETED, CRANE_TASK_SENTINEL)
        self.event_log.append((t0, "PRE_REHANDLE", job_id))
        return dur

    def skip_to_next_event(self):
        """모든 행동이 mask 된 경우: 다음 외부 이벤트까지 자동 진행 (02 §6)."""
        if self.queue.peek_time() is None:
            self._finalize()
            return
        self._process_next_event()

    # ------------------------------------------------------------- internals
    def _process_next_event(self):
        ev = self.queue.pop()
        self._advance(ev.time)
        self._handle(ev)
        if self._check:
            self.constraints.check_invariants(self.stacks, self.jobs, self.crane, self.clock)

    def _advance(self, t: float):
        """시계 전진. KPI 적분은 평가 윈도우 [0, end_time] 에서 절단 —
        마지막 작업 길이에 따라 정책별 측정 창이 달라지는 것을 방지."""
        if t < self.clock - _EPS:
            raise RuntimeError(f"시간 역행: {self.clock} -> {t}")
        if t > self.clock:
            cap = self.scenario.end_time
            lo, hi = min(self.clock, cap), min(t, cap)
            if hi > lo:
                self.kpis.integrate(lo, hi)
            self.clock = t

    def _finalize(self):
        """종료 처리 (1회): 잔여 대기를 종료시각까지 적분하고, 미완료 작업의
        종료비용을 계상한다 (03 §2.2 — 미루는 정책이 유리해지지 않도록).

        - 미완료 본선작업: max(0, end - deadline) 을 vessel_delay 에 가산
        - 미서비스 외부트럭: (end - 도착) 검열 대기 표본으로 포함
        """
        if self._terminal:
            return
        end = self.scenario.end_time
        self._advance(max(self.clock, end))
        for j in self.jobs.values():
            if (j.is_vessel_linked and j.deadline is not None and end > j.deadline
                    and j.status in (JobStatus.PLANNED, JobStatus.RELEASED, JobStatus.WAITING)):
                self.kpis.vessel_delay_s += end - j.deadline
        self.kpis.close_censored(end)
        self._terminal = True

    def _handle(self, ev):
        self.event_log.append((ev.time, ev.kind_name, ev.payload))
        if ev.kind_name == "BLOCK_ARRIVAL":
            job = self.jobs[ev.payload]
            job.status = JobStatus.WAITING
            self.kpis.truck_arrived(job.job_id, ev.time)
        elif ev.kind_name == "JOB_RELEASED":
            self.jobs[ev.payload].status = JobStatus.RELEASED
        elif ev.kind_name == "JOB_COMPLETED":
            if ev.payload == CRANE_TASK_SENTINEL:   # 포지셔닝·선재조작 종료
                self.crane.assigned_job = None
                self.crane.status = CraneStatus.IDLE
                return
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
