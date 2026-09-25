"""사건 큐를 **배열로** ([[YR-327]]).

v5 는 `heapq` 를 쓴다 — 파이썬 객체를 포인터로 잇는 힙이라 가속기가 못 다룬다.
여기서는 고정 길이 배열로 같은 일을 한다:

    time[i]     이 칸의 사건 시각          (빈 칸 = +inf)
    kind[i]     사건 종류 0..11            (빈 칸 = -1)
    target[i]   대상 번호 (작업·크레인·배)  (빈 칸 = -1)
    seq[i]      넣은 순서                  (빈 칸 = -1)

■ ★정렬 키는 v5 와 같은 **3단 사전식** — (시각, 종류 우선순위, 넣은 순서)
  v5 `integrated/events.py:30-43` 의 힙 키가 `(time, _PRIORITY[kind], seq)` 다.
  같은 시각이면 **완료가 도착보다 먼저**(우선순위), 그것도 같으면 **넣은 순서**다.
  이 순서가 다르면 사건 하나하나가 같아도 세계가 갈린다.

  ⚠️ 처음엔 `시각 + 순번×1e-9` 한 수로 합쳤는데 **틀렸다** (반박 검증 2026-09-25):
    · float32 는 t=1 에서 이웃 간격이 1.19e-7 이라 1e-9 짜리 순번이 **사라진다**
      (t=1 초에서 seq 3 소실 · t≥100 초에서 seq 999 소실 — 실측)
    · 종류 우선순위 칸이 아예 없었다
  그래서 세 열을 **차례로** 좁힌다: 시각 최소 → 그중 우선순위 최소 → 그중 순번 최소.

■ ★시각은 float64 — 동등성은 x64 에서만
  v5 는 파이썬 float(=64비트)이고 `_EPS=1e-9` 로 비교한다. 하루 끝(86,400 초) 근처에서
  float32 의 이웃 간격은 **0.0078 초**라 1ms 도착시각을 못 담는다. `TIME_DTYPE` 이
  float64 인데 `jax_enable_x64` 가 꺼져 있으면 JAX 가 **조용히 float32 로 내려앉는다** —
  그래서 `empty_queue` 가 dtype 을 확인해 **큰 소리로** 실패한다. 학습 모드에서 float32 로
  가려면 이 상수를 바꾸고 동등성 시험은 건너뛴다는 것을 알고 해야 한다.

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

#: 시각 dtype — 동등성 시험은 float64 (본문 참조). 호출자가 x64 를 켜야 한다.
TIME_DTYPE = jnp.float64

#: 빈 칸 표시 — 시각은 +inf, 나머지는 -1
EMPTY_TIME = jnp.inf
EMPTY_ID = -1

#: 사건 종류 0..11 의 우선순위 — v5 `_PRIORITY` 그대로 (완료 0 → 장비 1 → 이송/STS 2
#: → 도착·해제 3 → 개시 4 → 계획변경 5 → ETA 6 → 지평 7). 빈 칸(-1)은 쓰이지 않는다.
PRIO = jnp.array([0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7], jnp.int32)
N_KINDS = int(PRIO.shape[0])
_BIG_PRIO = 1 << 20
_BIG_SEQ = 1 << 30


class EventArray(NamedTuple):
    """사건 큐 하나. 모든 칸이 **고정 길이**다."""

    time: jnp.ndarray      # (C,) TIME_DTYPE — 빈 칸은 +inf
    kind: jnp.ndarray      # (C,) int32     — 빈 칸은 -1
    target: jnp.ndarray    # (C,) int32     — 빈 칸은 -1
    seq: jnp.ndarray       # (C,) int32     — 넣은 순서 (동시각·동우선순위 타이브레이크)
    counter: jnp.ndarray   # ()   int32     — 다음에 줄 순번
    overflow: jnp.ndarray  # ()   int32     — 자리가 없어 못 넣은 수 (0 이어야 정상)

    @property
    def capacity(self) -> int:
        return int(self.time.shape[0])


def empty_queue(capacity: int) -> EventArray:
    """빈 큐 하나. ★시각 dtype 이 요구와 다르면(x64 꺼짐) 조용히 넘기지 않고 실패한다."""
    t = jnp.full((capacity,), EMPTY_TIME, TIME_DTYPE)
    if t.dtype != jnp.dtype(TIME_DTYPE):
        raise RuntimeError(
            f"사건 시각 dtype 이 {t.dtype} 다 — TIME_DTYPE={jnp.dtype(TIME_DTYPE).name} 를 "
            f"쓰려면 jax.config.update('jax_enable_x64', True) 를 먼저 불러야 한다. "
            f"(조용히 float32 로 내려앉으면 하루 끝에서 1ms 를 못 담아 v5 와 갈린다)")
    return EventArray(
        time=t,
        kind=jnp.full((capacity,), EMPTY_ID, jnp.int32),
        target=jnp.full((capacity,), EMPTY_ID, jnp.int32),
        seq=jnp.full((capacity,), EMPTY_ID, jnp.int32),
        counter=jnp.int32(0),
        overflow=jnp.int32(0))


def push_event(q: EventArray, time, kind, target) -> EventArray:
    """사건 하나를 넣는다. **빈 칸 중 가장 앞**을 쓴다 (결정론).

    자리가 없으면 넣지 않고 `overflow` 를 올린다 — 조용히 버리면 세계가 달라진 것을
    아무도 모른다.
    """
    free = q.time == EMPTY_TIME
    has_room = jnp.any(free)
    slot = jnp.argmax(free)            # 첫 번째 빈 칸 (없으면 0 이지만 아래서 막는다)
    put = lambda arr, v: jnp.where(has_room, arr.at[slot].set(v), arr)
    return EventArray(
        time=put(q.time, jnp.asarray(time, TIME_DTYPE)),
        kind=put(q.kind, jnp.asarray(kind, jnp.int32)),
        target=put(q.target, jnp.asarray(target, jnp.int32)),
        seq=put(q.seq, q.counter),
        counter=q.counter + jnp.where(has_room, 1, 0),
        overflow=q.overflow + jnp.where(has_room, 0, 1))


def _pick(q: EventArray) -> jnp.ndarray:
    """가장 이른 사건의 칸 번호 — (시각, 우선순위, 순번) **3단 사전식** 최소.

    세 열을 차례로 좁힌다. 빈 칸은 시각이 +inf 라 첫 단계에서 저절로 빠진다
    (큐가 통째로 비면 아무 칸이나 나오지만 호출자가 `alive` 로 막는다).
    """
    tmin = q.time.min()
    m1 = q.time == tmin
    p = jnp.where(q.kind >= 0, PRIO[jnp.clip(q.kind, 0, N_KINDS - 1)], 0)
    pmin = jnp.min(jnp.where(m1, p, _BIG_PRIO))
    m2 = m1 & (p == pmin)
    return jnp.argmin(jnp.where(m2, q.seq, _BIG_SEQ))


def next_event(q: EventArray):
    """가장 이른 사건을 **꺼낸다**. `(큐, 시각, 종류, 대상, 있나)` 를 돌려준다.

    꺼낸 칸은 +inf 로 되돌린다 — 밀어내지 않는다(크기가 변하면 컴파일이 안 된다).
    """
    i = _pick(q)
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
    return q.time[_pick(q)]


def n_pending(q: EventArray) -> jnp.ndarray:
    """남은 사건 수 — 시험·진단용."""
    return jnp.sum(q.time < EMPTY_TIME)


#: ★배치 판 — 세계 B 개를 한 번에. 반사실 세계가 서로 독립이라 이게 그대로 먹는다.
push_event_batch = jax.vmap(push_event, in_axes=(0, 0, 0, 0))
next_event_batch = jax.vmap(next_event)
peek_time_batch = jax.vmap(peek_time)
