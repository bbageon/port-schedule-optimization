"""사건 큐를 **배열로** ([[YR-327]]).

v5 는 `heapq` 를 쓴다 — 파이썬 객체를 포인터로 잇는 힙이라 가속기가 못 다룬다.
여기서는 고정 길이 배열 넷으로 같은 일을 한다:

    time[i]     이 칸의 사건 시각          (빈 칸 = +inf)
    kind[i]     사건 종류                  (빈 칸 = -1)
    target[i]   대상 번호 (작업·크레인·배)  (빈 칸 = -1)
    seq[i]      넣은 순서                  ← **동시각 결정론**

■ ★같은 시각이면 넣은 순서대로 — 이게 없으면 답이 달라진다
  힙은 (시각, 순번) 짝으로 비교해 동시각을 **넣은 순서**로 푼다. 배열에서 시각만
  보고 `argmin` 하면 동점일 때 **아무거나** 걸려 실행마다 답이 달라진다.
  그래서 `(시각, 순번)` 을 하나의 정렬 키로 합쳐 비교한다.

■ 왜 힙이 아니라 선형 훑기인가
  힙은 O(log n) 이고 훑기는 O(n) 이다. 그런데 가속기에서는 **n 개를 동시에** 보므로
  훑기가 한 번의 병렬 축소(reduction)다. 대신 **칸이 고정**이라 꽉 차면 넘친다 —
  넘치면 조용히 버리지 않고 **표시**한다(`overflow`).

■ 지우기 — 꺼낸 칸은 +inf 로 되돌린다
  칸을 밀어내지 않는다(그러면 크기가 변한다). 빈 칸으로 표시만 하고 다음 `push` 가
  그 자리를 다시 쓴다.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

#: 빈 칸 표시 — 시각은 +inf, 나머지는 -1
EMPTY_TIME = jnp.inf
EMPTY_ID = -1


class EventArray(NamedTuple):
    """사건 큐 하나. 모든 칸이 **고정 길이**다."""

    time: jnp.ndarray      # (C,) float32 — 빈 칸은 +inf
    kind: jnp.ndarray      # (C,) int32   — 빈 칸은 -1
    target: jnp.ndarray    # (C,) int32   — 빈 칸은 -1
    seq: jnp.ndarray       # (C,) int32   — 넣은 순서 (동시각 타이브레이크)
    counter: jnp.ndarray   # ()   int32   — 다음에 줄 순번
    overflow: jnp.ndarray  # ()   int32   — 자리가 없어 못 넣은 수 (0 이어야 정상)

    @property
    def capacity(self) -> int:
        return int(self.time.shape[0])


def empty_queue(capacity: int) -> EventArray:
    """빈 큐 하나."""
    return EventArray(
        time=jnp.full((capacity,), EMPTY_TIME, jnp.float32),
        kind=jnp.full((capacity,), EMPTY_ID, jnp.int32),
        target=jnp.full((capacity,), EMPTY_ID, jnp.int32),
        seq=jnp.full((capacity,), EMPTY_ID, jnp.int32),
        counter=jnp.int32(0),
        overflow=jnp.int32(0))


def push_event(q: EventArray, time: float, kind: int, target: int) -> EventArray:
    """사건 하나를 넣는다. **빈 칸 중 가장 앞**을 쓴다 (결정론).

    자리가 없으면 넣지 않고 `overflow` 를 올린다 — 조용히 버리면 세계가 달라진 것을
    아무도 모른다.
    """
    free = q.time == EMPTY_TIME
    has_room = jnp.any(free)
    slot = jnp.argmax(free)            # 첫 번째 빈 칸 (없으면 0 이지만 아래서 막는다)
    put = lambda arr, v: jnp.where(has_room, arr.at[slot].set(v), arr)
    return EventArray(
        time=put(q.time, jnp.float32(time)),
        kind=put(q.kind, jnp.int32(kind)),
        target=put(q.target, jnp.int32(target)),
        seq=put(q.seq, q.counter),
        counter=q.counter + jnp.where(has_room, 1, 0),
        overflow=q.overflow + jnp.where(has_room, 0, 1))


def _sort_key(q: EventArray) -> jnp.ndarray:
    """정렬 키 = (시각, 넣은 순서). 동시각을 **넣은 순서**로 푼다 (힙과 같은 규약).

    두 값을 하나의 실수로 합친다 — 순번은 정수라 아주 작은 가중치를 주면 시각 비교를
    흐리지 않으면서 동점만 가른다. 빈 칸은 시각이 +inf 라 저절로 맨 뒤로 간다.
    """
    return q.time + jnp.where(q.seq >= 0, q.seq.astype(jnp.float32) * 1e-9, 0.0)


def next_event(q: EventArray):
    """가장 이른 사건을 **꺼낸다**. `(큐, 시각, 종류, 대상, 있나)` 를 돌려준다.

    꺼낸 칸은 +inf 로 되돌린다 — 밀어내지 않는다(크기가 변하면 컴파일이 안 된다).
    """
    i = jnp.argmin(_sort_key(q))
    alive = q.time[i] < EMPTY_TIME
    q2 = EventArray(
        time=jnp.where(alive, q.time.at[i].set(EMPTY_TIME), q.time),
        kind=jnp.where(alive, q.kind.at[i].set(EMPTY_ID), q.kind),
        target=jnp.where(alive, q.target.at[i].set(EMPTY_ID), q.target),
        seq=jnp.where(alive, q.seq.at[i].set(EMPTY_ID), q.seq),
        counter=q.counter, overflow=q.overflow)
    return q2, q.time[i], q.kind[i], q.target[i], alive


def peek_time(q: EventArray) -> jnp.ndarray:
    """다음 사건 시각. 비었으면 +inf — **꺼내지 않는다**."""
    return q.time[jnp.argmin(_sort_key(q))]


def n_pending(q: EventArray) -> jnp.ndarray:
    """남은 사건 수 — 시험·진단용."""
    return jnp.sum(q.time < EMPTY_TIME)


#: ★배치 판 — 세계 B 개를 한 번에. 반사실 세계가 서로 독립이라 이게 그대로 먹는다.
push_event_batch = jax.vmap(push_event, in_axes=(0, 0, 0, 0))
next_event_batch = jax.vmap(next_event)
peek_time_batch = jax.vmap(peek_time)
