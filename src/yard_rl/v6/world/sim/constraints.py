"""SafetyConstraintEngine — Hard Constraint 중앙관리 (구현계획 02 §8).

안전·도달영역·슬롯 적합성은 학습 대상이 아니라 항상 지키는 규칙.
후보 생성 단계와 실행 직전, 두 번 차단한다. soft penalty 로 대체하지 않는다.
비통과·안전거리 등 다중 YC 제약은 Exp-4(YR-013) 에서 추가한다.
"""
from __future__ import annotations

from ..domain.enums import JobFlow, JobStatus
from ..domain.models import CraneState, Job, TerminalProfile
from .stack import YardStacks


class ConstraintViolation(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


# 포지셔닝·선재조작 중 크레인 점유 표시 (Job 이 아닌 크레인 단독작업)
CRANE_TASK_SENTINEL = "__CRANE_TASK__"


class ConstraintEngine:
    def __init__(self, profile: TerminalProfile):
        self.profile = profile

    # --- 후보/실행 판정 (두 단계 공용) ---
    def job_bay(self, job: Job, stacks: YardStacks) -> int | None:
        """작업이 수행될 bay. GATE_IN 은 슬롯 선정 전이면 None."""
        if job.target_container is not None and job.target_container in stacks.containers:
            return stacks.containers[job.target_container].bay
        return None

    def is_dispatchable(self, job: Job, crane: CraneState, stacks: YardStacks) -> bool:
        if job.status not in (JobStatus.WAITING, JobStatus.RELEASED):
            return False
        if job.assigned_crane is not None:
            return False
        if job.target_container is not None:
            c = stacks.containers.get(job.target_container)
            if c is None or not c.work_available:
                return False
            if not (crane.service_bay_min <= c.bay <= crane.service_bay_max):
                return False
            # 재조작 슬롯 고갈이면 후보 제외 (실행 중 NO_SAFE_SLOT 크래시 방지)
            if not stacks.rehandle_capacity_ok(job.target_container, self.profile.crane):
                return False
        if job.flow == JobFlow.GATE_IN:
            # 합법 장치슬롯이 하나라도 있어야 함
            if stacks.find_slot(job.inbound_size, self.profile.crane,
                                crane.position_bay, crane.trolley_row) is None:
                return False
        return True

    def validate_assignment(self, job: Job, crane: CraneState, stacks: YardStacks) -> None:
        """실행 직전 최종 차단 (2차)."""
        if crane.assigned_job is not None:
            raise ConstraintViolation("DOUBLE_ASSIGN", f"{crane.crane_id} 이미 {crane.assigned_job} 수행 중")
        if not self.is_dispatchable(job, crane, stacks):
            raise ConstraintViolation("NOT_DISPATCHABLE", f"{job.job_id} 실행 불가 상태")

    # --- 상태 불변조건 (매 이벤트 후) — 02 §1.3 ---
    def check_invariants(self, stacks: YardStacks, jobs: dict[str, Job],
                         crane: CraneState, now: float) -> None:
        geom = self.profile.block
        seen: set[str] = set()
        for (bay, row), pile in stacks._stacks.items():
            if len(pile) > geom.tier_max:
                raise ConstraintViolation("TIER_OVERFLOW", f"({bay},{row}) {len(pile)}단")
            if not (1 <= bay <= geom.bay_count and 1 <= row <= geom.row_count):
                raise ConstraintViolation("OUT_OF_BLOCK", f"({bay},{row})")
            for tier, cid in enumerate(pile, start=1):
                if cid in seen:
                    raise ConstraintViolation("DUP_CONTAINER", cid)
                seen.add(cid)
                c = stacks.containers[cid]
                if (c.bay, c.row, c.tier) != (bay, row, tier):
                    raise ConstraintViolation("POSITION_DESYNC", f"{cid}: {(c.bay, c.row, c.tier)} != {(bay, row, tier)}")
        if set(stacks.containers) != seen:
            raise ConstraintViolation("CONTAINER_LOST", "stack 목록과 컨테이너 dict 불일치")
        # 작업·크레인 일관성
        running = [j for j in jobs.values() if j.status == JobStatus.RUNNING]
        if len(running) > 1:
            raise ConstraintViolation("MULTI_RUN", f"{[j.job_id for j in running]}")
        if running and crane.assigned_job != running[0].job_id:
            raise ConstraintViolation("ASSIGN_DESYNC", f"crane={crane.assigned_job} run={running[0].job_id}")
        if (not running and crane.assigned_job is not None
                and crane.assigned_job != CRANE_TASK_SENTINEL):
            raise ConstraintViolation("ASSIGN_DESYNC", f"실행작업 없이 crane={crane.assigned_job}")
        if not (crane.service_bay_min <= crane.position_bay <= crane.service_bay_max):
            raise ConstraintViolation("OUT_OF_RANGE", f"crane bay={crane.position_bay}")
        for j in jobs.values():
            if j.status == JobStatus.DONE:
                if j.service_end is None or j.service_end > now + 1e-9:
                    raise ConstraintViolation("TIME_TRAVEL", j.job_id)
                if j.service_start is not None and j.service_end < j.service_start:
                    raise ConstraintViolation("NEGATIVE_DURATION", j.job_id)
