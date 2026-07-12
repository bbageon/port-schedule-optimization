"""이벤트 정의와 동시 이벤트 우선순위 — 구현계획 02 §1.1~1.2.

같은 시각: 완료(0) → 장비(1) → 도착·release(2) → ETA 갱신(3) → 의사결정(4).
동순위는 발행 순서(seq)로 결정론적 처리.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import IntEnum


class EventKind(IntEnum):
    JOB_COMPLETED = 0
    EQUIPMENT = 1        # 고장·복구 (PoC 미사용, 자리만)
    BLOCK_ARRIVAL = 2    # 외부트럭 블록 도착
    JOB_RELEASED = 2     # 내부·본선 작업 선택 가능 (도착과 동순위)
    ETA_UPDATED = 3      # Exp-3 전용 (PoC 미사용)
    HORIZON = 4          # 신규 도착 마감 marker


@dataclass(order=True)
class _Entry:
    time: float
    kind: int
    seq: int
    payload: str = field(compare=False)  # job_id 또는 marker
    kind_name: str = field(compare=False)


class EventQueue:
    def __init__(self):
        self._heap: list[_Entry] = []
        self._seq = itertools.count()

    def push(self, time: float, kind: EventKind, payload: str = ""):
        heapq.heappush(self._heap, _Entry(time, int(kind), next(self._seq), payload, kind.name))

    def pop(self) -> _Entry:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float | None:
        return self._heap[0].time if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
