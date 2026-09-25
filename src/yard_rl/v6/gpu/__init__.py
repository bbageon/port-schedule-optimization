"""v6 GPU 핵심 — 배열 기반 이산사건 세계 ([[YR-327]]).

v5 까지의 세계는 **파이썬 객체**로 돼 있다: 딕셔너리로 작업을 찾고, 힙으로 사건을
꺼내고, 객체를 제자리에서 고친다. 가속기는 이런 구조를 다루지 못한다 — 포인터를 따라갈
수 없고, 크기가 변하는 것을 컴파일할 수 없다.

v6 는 같은 규칙을 **고정 크기 배열 위에서** 다시 쓴다:

    파이썬 객체 세계              배열 세계
    ─────────────────────────────────────────────────
    heapq 우선순위 큐         →  (시각, 종류, 대상) 배열 + argmin
    dict[job_id] → Job        →  길이 N 배열들, 빈 칸은 -1
    if 조건: A else: B        →  A·B 를 다 계산하고 where 로 고름
    sim.assign(...) 제자리 수정 →  상태를 받아 **새 상태**를 돌려주는 순수 함수

■ 무엇을 노리나 — "하나를 빠르게" 가 아니라 "수천 개를 동시에"
  반사실 세계는 서로 **완전히 독립**이다([[YR-219]]). 지금은 프로세스 10~20개로
  나누는데, 배열로 바꾸면 `vmap` 한 줄로 수천 개를 한 번에 민다.

■ ⚠️ 이 묶음은 **아직 v5 세계를 대체하지 않는다**
  `world/` 의 규칙을 한 조각씩 옮기고, 조각마다 v5 와 **같은 답을 내는지** 시험으로
  붙든다(`tests/v6/gpu/`). 전부 옮기기 전까지 v5 경로가 정본이다.
"""
from __future__ import annotations

from .events import EventArray, next_event, push_event
from .state import WorldArrays, empty_world

__all__ = ["EventArray", "next_event", "push_event", "WorldArrays", "empty_world"]
