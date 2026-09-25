"""통합 시뮬레이터 이벤트 — 단일 YC sim/events.py 의 superset (YR-036).

sim/events.py 를 편집하지 않는다 (단일 YC golden n_events 회귀 원천 차단). 여기 전용
EventKind·_PRIORITY·EventQueue 를 둔다. 동시각 처리순위는 값이 아니라 _PRIORITY 로 정의하고
동순위는 발행순서(seq)로 결정론. seq 는 plain int — itertools.count 금지(deepcopy 요건, YR-031).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum


class EventKind(IntEnum):
    JOB_COMPLETED = 0
    EQUIPMENT_DOWN = 1
    EQUIPMENT_UP = 2
    TRANSFER_ARRIVE = 3   # 이송 유닛 도착 → 유닛 free
    STS_MOVE = 4          # 본선 STS 1 move 처리시점
    BLOCK_ARRIVAL = 5     # 외부트럭 블록 도착
    JOB_RELEASED = 6      # 내부작업 선택가능
    VESSEL_RELEASED = 7   # 본선연계 job 신규 발생
    VESSEL_START = 8      # 본선 프로세스 개시
    PLAN_CHANGE = 9       # 완료시각/deadline/물량 변경
    ETA_UPDATED = 10
    HORIZON = 11


# 완료·자원해제 먼저 → 장비 → 이송/STS → 도착·release → 개시 → 계획변경 → ETA → marker
_PRIORITY: dict[EventKind, int] = {
    EventKind.JOB_COMPLETED: 0,
    EventKind.EQUIPMENT_DOWN: 1,
    EventKind.EQUIPMENT_UP: 1,
    EventKind.TRANSFER_ARRIVE: 2,
    EventKind.STS_MOVE: 2,
    EventKind.BLOCK_ARRIVAL: 3,
    EventKind.JOB_RELEASED: 3,
    EventKind.VESSEL_RELEASED: 3,
    EventKind.VESSEL_START: 4,
    EventKind.PLAN_CHANGE: 5,
    EventKind.ETA_UPDATED: 6,
    EventKind.HORIZON: 7,
}


@dataclass(order=True)
class _Entry:
    time: float
    priority: int
    seq: int
    payload: str = field(compare=False)              # job_id·crane_id·vessel_id
    data: object = field(default=None, compare=False)  # 구조화 payload (frozen dc/tuple 만)
    kind_name: str = field(default="", compare=False)


class EventQueue:
    def __init__(self):
        self._heap: list[_Entry] = []
        self._seq = 0

    def push(self, time: float, kind: EventKind, payload: str = "", data: object = None):
        self._seq += 1
        heapq.heappush(
            self._heap,
            _Entry(time, _PRIORITY[kind], self._seq, payload, data, kind.name))

    def pop(self) -> _Entry:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float | None:
        return self._heap[0].time if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
