"""Action Masking — 실행 불가능 rule 차단 (구현계획 02 §6 조건표).

Exp-2/3 확장: EPA(도착예상 순) 는 정보수준별 예측자(eta_of)가 값을 주는
후보가 있거나, control_scope 가 포지셔닝을 허용하고 이동할 가치가 있는
(|Δbay|>1) 임박 미래작업이 있을 때 열린다. PRE_REHANDLE 은 plus_pre_rehandle
전용 — 조건은 02 §6.1 (대상 존재·blocker·시간창·합법 슬롯).
"""
from __future__ import annotations

from ..domain.enums import ControlScope, InformationLevel, JobFlow, PriorityRule
from ..domain.models import CraneState, Job, TerminalProfile
from ..sim.stack import YardStacks

N_ACTIONS = len(PriorityRule)
PRE_REHANDLE_MIN_WINDOW_S = 600.0  # 도착 전 최소 시간창 (assumed)


def scope_allows_positioning(scope: ControlScope) -> bool:
    return scope in (ControlScope.PLUS_POSITIONING, ControlScope.PLUS_PRE_REHANDLE)


def future_job_bay(j: Job, stacks: YardStacks, profile: TerminalProfile,
                   crane: CraneState) -> float | None:
    """미래작업의 예상 작업 bay (포지셔닝 목표)."""
    if j.target_container is not None and j.target_container in stacks.containers:
        return float(stacks.containers[j.target_container].bay)
    if j.flow == JobFlow.GATE_IN and j.inbound_size is not None:
        slot = stacks.find_slot(j.inbound_size, profile.crane,
                                crane.position_bay, crane.trolley_row)
        return float(slot[0]) if slot else None
    return None


def positioning_targets(future: list[Job], *, now: float, crane: CraneState,
                        stacks: YardStacks, profile: TerminalProfile,
                        eta_of) -> list[Job]:
    """포지셔닝 가치가 있는 임박 미래작업 (horizon 내, |Δbay|>1)."""
    out = []
    for j in future:
        eta = eta_of(j)
        if eta is None or eta - now > profile.decision_horizon_s:
            continue
        bay = future_job_bay(j, stacks, profile, crane)
        if bay is None or abs(bay - crane.position_bay) <= 1.0:
            continue  # 이미 근접 → 이동가치 없음 (0-시간 루프 방지 겸)
        out.append(j)
    return out


def pre_rehandle_targets(future: list[Job], *, now: float, crane: CraneState,
                         stacks: YardStacks, profile: TerminalProfile,
                         eta_of) -> list[Job]:
    """선재조작 조건(02 §6.1)을 만족하는 미래 반출작업."""
    out = []
    for j in future:
        if j.flow != JobFlow.GATE_OUT or j.target_container is None:
            continue
        c = stacks.containers.get(j.target_container)
        if c is None or not c.work_available:
            continue
        if not (crane.service_bay_min <= c.bay <= crane.service_bay_max):
            continue
        if not stacks.blockers_above(j.target_container):
            continue
        eta = eta_of(j)
        if eta is None or eta - now < PRE_REHANDLE_MIN_WINDOW_S:
            continue  # 도착 전 시간창 부족
        if not stacks.rehandle_capacity_ok(j.target_container, profile.crane):
            continue
        out.append(j)
    return out


def build_mask(candidates: list[Job], *, level: InformationLevel, scope: ControlScope,
               crane: CraneState, stacks: YardStacks, profile: TerminalProfile,
               future: list[Job] | None = None, eta_of=None,
               now: float = 0.0) -> list[bool]:
    mask = [False] * N_ACTIONS
    future = future or []
    has_any = bool(candidates)
    has_target = any(j.target_container is not None for j in candidates)
    has_vessel = any(j.is_vessel_linked for j in candidates)

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
    if eta_of is not None:
        has_eta_now = any(eta_of(j) is not None for j in candidates)
        can_position = (scope_allows_positioning(scope) and bool(
            positioning_targets(future, now=now, crane=crane, stacks=stacks,
                                profile=profile, eta_of=eta_of)))
        mask[PriorityRule.EARLIEST_PROVIDED_ARRIVAL] = has_eta_now or can_position
        if scope == ControlScope.PLUS_PRE_REHANDLE:
            mask[PriorityRule.PRE_REHANDLE] = bool(
                pre_rehandle_targets(future, now=now, crane=crane, stacks=stacks,
                                     profile=profile, eta_of=eta_of))
    mask[PriorityRule.WAIT_YIELD] = False  # 대기 허용 운영조건 미정의 (Exp-4 간섭용)
    return mask
