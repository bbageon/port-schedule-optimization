"""이벤트 정의와 동시 이벤트 우선순위 — 구현계획 02 §1.1~1.2.

같은 시각: 완료 → 장비 → 도착·release(동순위) → ETA 갱신 → marker.
주의: IntEnum 은 동일 값이면 alias 가 되므로(도착·release 를 같은 값으로 두면
이벤트 종류가 합쳐지는 버그) 값은 전부 다르게 두고, 동시각 처리순위는
별도 매핑(_PRIORITY)으로 정의한다. 동순위는 발행 순서(seq)로 결정론적.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum


class EventKind(IntEnum):
    JOB_COMPLETED = 0
    EQUIPMENT = 1        # 고장·복구 (PoC 미사용, 자리만)
    BLOCK_ARRIVAL = 2    # 외부트럭 블록 도착
    JOB_RELEASED = 3     # 내부·본선 작업 선택 가능
    ETA_UPDATED = 4      # Exp-3 전용 (PoC 미사용)
    HORIZON = 5          # 신규 도착 마감 marker


_PRIORITY: dict[EventKind, int] = {
    EventKind.JOB_COMPLETED: 0,
    EventKind.EQUIPMENT: 1,
    EventKind.BLOCK_ARRIVAL: 2,
    EventKind.JOB_RELEASED: 2,   # 도착과 동순위 (02 §1.2)
    EventKind.ETA_UPDATED: 3,
    EventKind.HORIZON: 4,
}


@dataclass(order=True)
class _Entry:
    time: float
    priority: int
    seq: int
    payload: str = field(compare=False)  # job_id 또는 marker
    kind_name: str = field(compare=False)


class EventQueue:
    def __init__(self):
        self._heap: list[_Entry] = []
        # 일반 int 카운터 — itertools.count 는 deepcopy 불가 (YR-031 beam 분기
        # 가 시뮬레이터 상태 복제를 요구). 동작은 동일 (발행 순서 결정론).
        self._seq = 0

    def push(self, time: float, kind: EventKind, payload: str = ""):
        self._seq += 1
        heapq.heappush(self._heap,
                       _Entry(time, _PRIORITY[kind], self._seq, payload, kind.name))

    def pop(self) -> _Entry:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float | None:
        return self._heap[0].time if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
