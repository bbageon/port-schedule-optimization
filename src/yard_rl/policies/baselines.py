"""Baseline 정책 — 구현계획 03 §1.1.

CURRENT_RULE(현행 작업규칙)은 미확보 상태 — 확보 전까지 어떤 결과도
'실제 운영 대비 개선율'로 표현하지 않는다. 비교 기준은 휴리스틱 4종.
"""
from __future__ import annotations

from ..domain.enums import PriorityRule

# 후보가 있으면 항상 하나는 열려 있는 순서 (FIFO·LONGEST_WAIT·NEAREST 는 has_any)
_FALLBACK_ORDER = [PriorityRule.LONGEST_WAIT, PriorityRule.FIFO, PriorityRule.NEAREST_JOB,
                   PriorityRule.MIN_REHANDLE, PriorityRule.VESSEL_PRIORITY,
                   PriorityRule.SAME_BAY_BATCH]


class FixedRulePolicy:
    """단일 rule 고정 정책. rule 이 mask 되면 결정론적 fallback 순서로 대체."""

    def __init__(self, rule: PriorityRule):
        self.rule = rule
        self.name = rule.name

    def act(self, state, mask: list[bool]) -> int:
        if mask[self.rule]:
            return int(self.rule)
        for r in _FALLBACK_ORDER:
            if mask[r]:
                return int(r)
        for a, ok in enumerate(mask):  # 방어적: 사전행동만 열린 상태
            if ok:
                return a
        raise RuntimeError("가능한 rule 없음 — mask 전부 False 인데 step 호출됨")


def baseline_policies() -> list[FixedRulePolicy]:
    return [FixedRulePolicy(PriorityRule.FIFO),
            FixedRulePolicy(PriorityRule.LONGEST_WAIT),
            FixedRulePolicy(PriorityRule.NEAREST_JOB),
            FixedRulePolicy(PriorityRule.MIN_REHANDLE)]
