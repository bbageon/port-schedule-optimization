"""Action Masking — 실행 불가능 rule 차단 (구현계획 02 §6 mask 조건표)."""
from __future__ import annotations

from ..domain.enums import ControlScope, InformationLevel, PriorityRule
from ..domain.models import CraneState, Job, TerminalProfile
from ..sim.stack import YardStacks

N_ACTIONS = len(PriorityRule)


def build_mask(candidates: list[Job], *, level: InformationLevel, scope: ControlScope,
               crane: CraneState, stacks: YardStacks, profile: TerminalProfile) -> list[bool]:
    mask = [False] * N_ACTIONS
    if not candidates:
        return mask
    has_any = bool(candidates)
    has_target = any(j.target_container is not None for j in candidates)
    has_vessel = any(j.is_vessel_linked for j in candidates)
    has_eta = any(j.provided_eta is not None for j in candidates)

    def bay_of(j: Job) -> float | None:
        if j.target_container is not None and j.target_container in stacks.containers:
            return float(stacks.containers[j.target_container].bay)
        return None

    has_same_bay = any(b is not None and abs(b - crane.position_bay) <= 1.0
                       for b in (bay_of(j) for j in candidates))

    mask[PriorityRule.FIFO] = has_any
    mask[PriorityRule.LONGEST_WAIT] = has_any
    mask[PriorityRule.NEAREST_JOB] = has_any
    mask[PriorityRule.MIN_REHANDLE] = has_target
    mask[PriorityRule.VESSEL_PRIORITY] = has_vessel
    mask[PriorityRule.SAME_BAY_BATCH] = has_same_bay
    mask[PriorityRule.EARLIEST_PROVIDED_ARRIVAL] = (
        level == InformationLevel.PRE_ADVICE and has_eta)
    mask[PriorityRule.PRE_REHANDLE] = False   # Exp-3C (YR-011-c) 에서 구현
    mask[PriorityRule.WAIT_YIELD] = False     # 대기 허용 운영조건 미정의 (Exp-4 간섭용)
    return mask
