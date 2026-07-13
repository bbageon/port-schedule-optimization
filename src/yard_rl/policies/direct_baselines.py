"""YR-027 외부트럭 Job 직접선택 baseline.

기존 PriorityRuleExecutor 는 FIFO 에 gate 시각을 쓰고 공통 SLA/deadline
tie-break 를 적용하므로 재사용하지 않는다. SLA 제약은 env 후보집합에만 적용된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol, TypeVar


class DirectCandidateProtocol(Protocol):
    job_id: str
    block_entry_s: float
    wait_s: float
    reach_s: float
    estimated_service_s: float
    blocker_count: int


CandidateT = TypeVar("CandidateT", bound=DirectCandidateProtocol)


class DirectRule(str, Enum):
    FIFO = "FIFO"
    LONGEST_WAIT = "LONGEST_WAIT"
    NEAREST_JOB = "NEAREST_JOB"
    MIN_BLOCKER = "MIN_BLOCKER"
    SHORTEST_ESTIMATED_SERVICE_TIME = "SHORTEST_ESTIMATED_SERVICE_TIME"
    IMMEDIATE_COST_GREEDY = "IMMEDIATE_COST_GREEDY"


@dataclass(frozen=True)
class DirectJobRulePolicy:
    rule: DirectRule

    @property
    def name(self) -> str:
        return self.rule.value

    def act(self, _state: object, candidates: Iterable[CandidateT]) -> CandidateT:
        feasible = list(candidates)
        if not feasible:
            raise ValueError("direct-job baseline requires a feasible candidate")

        def entry_job(item: DirectCandidateProtocol) -> tuple[float, str]:
            return (float(item.block_entry_s), str(item.job_id))

        if self.rule in (DirectRule.FIFO, DirectRule.LONGEST_WAIT):
            # 같은 now 에서는 max wait == earliest BLOCK_ENTRY.
            key = entry_job
        elif self.rule == DirectRule.NEAREST_JOB:
            key = lambda item: (float(item.reach_s), *entry_job(item))
        elif self.rule == DirectRule.MIN_BLOCKER:
            key = lambda item: (int(item.blocker_count), *entry_job(item))
        elif self.rule in (
            DirectRule.SHORTEST_ESTIMATED_SERVICE_TIME,
            DirectRule.IMMEDIATE_COST_GREEDY,
        ):
            # (q_t-1)/(60*N_config) 은 후보 공통이므로 immediate cost와 동일.
            key = lambda item: (float(item.estimated_service_s), *entry_job(item))
        else:  # pragma: no cover - Enum 확장 시 방어
            raise ValueError(f"unsupported direct rule: {self.rule}")
        return min(feasible, key=key)


def direct_baseline_policies() -> list[DirectJobRulePolicy]:
    return [DirectJobRulePolicy(rule) for rule in DirectRule]
