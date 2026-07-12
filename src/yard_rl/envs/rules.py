"""PriorityRuleExecutor — rule 이 실제 Job 을 결정 (구현계획 02 §6).

같은 State·후보에서 항상 같은 Job 을 선택하도록 결정론적으로 구현.
동점 해소: SLA 초과 → deadline slack → 누적대기 → 예상 재조작 → job_id.

주: FIFO(블록 도착순)와 LONGEST_WAIT(누적대기 최대)는 외부트럭만 있을 때
순서가 거의 일치한다. PoC 에서는 FIFO 를 게이트 진입순, LONGEST_WAIT 를
블록 대기순으로 구분해 rule 간 차이를 유지한다 (assumed 해석).
"""
from __future__ import annotations

from ..domain.enums import PriorityRule
from ..domain.models import CraneState, Job, TerminalProfile
from ..sim.stack import YardStacks
from ..sim.travel_time import estimate_reach_s

_INF = float("inf")


def effective_arrival(job: Job) -> float:
    if job.is_external_truck:
        return job.actual_block_arrival
    return job.release_time


def gate_order_key(job: Job) -> float:
    if job.is_external_truck:
        return job.actual_gate_in
    return job.release_time


def blockers_of(job: Job, stacks: YardStacks) -> int:
    if job.target_container is None or job.target_container not in stacks.containers:
        return 0
    return len(stacks.blockers_above(job.target_container))


def reach_s(job: Job, crane: CraneState, stacks: YardStacks, profile: TerminalProfile) -> float:
    """크레인→작업 픽업지점 예상 이동시간."""
    geom, spec = profile.block, profile.crane
    if job.target_container is not None and job.target_container in stacks.containers:
        c = stacks.containers[job.target_container]
        return estimate_reach_s(spec, geom, crane.position_bay, crane.trolley_row,
                                float(c.bay), float(c.row))
    # GATE_IN: 예상 장치슬롯의 bay 차선에서 픽업
    slot = stacks.find_slot(job.inbound_size, spec, crane.position_bay, crane.trolley_row)
    if slot is None:
        return _INF
    return estimate_reach_s(spec, geom, crane.position_bay, crane.trolley_row,
                            float(slot[0]), float(geom.transfer_row))


def _tie_break_key(job: Job, now: float, sla_s: float, stacks: YardStacks):
    wait = max(0.0, now - effective_arrival(job)) if job.is_external_truck else 0.0
    sla_exceeded = 0 if (job.is_external_truck and wait > sla_s) else 1  # 초과 우선
    slack = (job.deadline - now) if job.deadline is not None else _INF
    return (sla_exceeded, slack, -wait, blockers_of(job, stacks), job.job_id)


class PriorityRuleExecutor:
    def __init__(self, profile: TerminalProfile):
        self.profile = profile

    def select(self, rule: PriorityRule, candidates: list[Job], *,
               crane: CraneState, stacks: YardStacks, now: float) -> Job:
        if not candidates:
            raise ValueError("후보 없음 — mask 로 걸렀어야 함")
        sla = self.profile.long_wait_sla_s

        def key(primary):
            return lambda j: (primary(j), *_tie_break_key(j, now, sla, stacks))

        if rule == PriorityRule.FIFO:
            return min(candidates, key=key(gate_order_key))
        if rule == PriorityRule.LONGEST_WAIT:
            return min(candidates, key=key(effective_arrival))
        if rule == PriorityRule.NEAREST_JOB:
            return min(candidates, key=key(
                lambda j: reach_s(j, crane, stacks, self.profile)))
        if rule == PriorityRule.MIN_REHANDLE:
            pool = [j for j in candidates if j.target_container is not None]
            return min(pool, key=key(lambda j: blockers_of(j, stacks)))
        if rule == PriorityRule.VESSEL_PRIORITY:
            pool = [j for j in candidates if j.is_vessel_linked]
            return min(pool, key=key(lambda j: (j.deadline - now) if j.deadline else _INF))
        if rule == PriorityRule.SAME_BAY_BATCH:
            pool = [j for j in candidates
                    if self._job_bay(j, stacks) is not None
                    and abs(self._job_bay(j, stacks) - crane.position_bay) <= 1.0]
            return min(pool, key=key(lambda j: abs(self._job_bay(j, stacks) - crane.position_bay)))
        if rule == PriorityRule.EARLIEST_PROVIDED_ARRIVAL:
            pool = [j for j in candidates if j.provided_eta is not None]
            return min(pool, key=key(lambda j: j.provided_eta))
        raise ValueError(f"PoC 미지원 rule {rule.name}")  # PRE_REHANDLE·WAIT_YIELD 는 Exp-3C/4

    def _job_bay(self, job: Job, stacks: YardStacks) -> float | None:
        if job.target_container is not None and job.target_container in stacks.containers:
            return float(stacks.containers[job.target_container].bay)
        return None  # GATE_IN 은 슬롯 미정 — SAME_BAY_BATCH 대상에서 제외
