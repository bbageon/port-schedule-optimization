"""세계 상태를 **고정 크기 배열로** ([[YR-327]]).

v5 는 `dict[job_id] → Job`, `list` 스택, 문자열 식별자를 쓴다. 배열 세계에서는
전부 번호와 배열이다 — 없는 것은 **-1**, 비어 있는 시각은 **+inf** 로 표시한다.

    문자열 "Y15:D-00042"  →  정수 번호 42 (블록 15)
    dict 조회             →  배열 색인
    리스트 길이 가변      →  길이 고정 + 개수 칸

■ ★왜 "전체 오더를 신경망으로" 가 이 구조에서 가능해지나
  오더가 배열이면 **전 오더의 특징이 하나의 행렬** (오더 수 × 특징 수)이 된다.
  망에 한 번 넣으면 **모든 오더의 점수가 한 번에** 나온다 — 지금처럼 후보마다
  따로 부르지 않는다. 그게 가속기가 빠른 이유이고, 반사실 기준선을
  **한 번의 순전파로** 만들 수 있는 근거이기도 하다([[YR-326]] ②).

■ 칸 수는 **미리 정한다**
  하루 최대 오더 15,000 · 블록 21 · 블록당 크레인 2 처럼 상한을 박아 둔다. 넘으면
  조용히 자르지 않고 표시한다 — 자르면 부하가 저절로 줄어 결과가 거짓이 된다
  ([[YR-316]] 에서 겪은 바로 그 사고).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

EMPTY_ID = -1
EMPTY_TIME = jnp.inf

#: 오더 상태 — 라이프사이클 다섯 단계와 같은 순서 (schema/lifecycle.py)
ST_NOTICE, ST_GATE_IN, ST_BLOCK_IN, ST_SERVING, ST_DONE, ST_GATE_OUT = 0, 1, 2, 3, 4, 5

#: 크레인 상태
CR_IDLE, CR_MOVING, CR_WORKING = 0, 1, 2


class OrderArrays(NamedTuple):
    """오더 N 개. 모든 칸이 같은 길이라 **한 번에 망에 넣을 수 있다**."""

    block: jnp.ndarray        # (N,) int32  어느 블록으로 가나 (-1 = 없는 오더)
    flow: jnp.ndarray         # (N,) int32  0 = 반입, 1 = 반출
    stage: jnp.ndarray        # (N,) int32  위 ST_* — 지금 어디까지 왔나
    #: ★시각 다섯 — 사용자 스키마 그대로 (2026-09-22). 아직 안 온 단계는 +inf
    notice_s: jnp.ndarray     # (N,) float32  통지 (Truck ETA)
    gate_in_s: jnp.ndarray    # (N,) float32  게이트 진입
    block_in_s: jnp.ndarray   # (N,) float32  블록 도착
    service_s: jnp.ndarray    # (N,) float32  작업 시작 (내부 관측 — 정책 입력 금지)
    done_s: jnp.ndarray       # (N,) float32  작업 완료
    gate_out_s: jnp.ndarray   # (N,) float32  게이트 아웃
    duration_s: jnp.ndarray   # (N,) float32  이 작업에 걸리는 시간
    travel_s: jnp.ndarray     # (N,) float32  게이트→블록 주행

    @property
    def n(self) -> int:
        return int(self.block.shape[0])


class CraneArrays(NamedTuple):
    """크레인 K 개."""

    block: jnp.ndarray         # (K,) int32   소속 블록
    status: jnp.ndarray        # (K,) int32   위 CR_*
    available_at: jnp.ndarray  # (K,) float32 언제부터 다시 고를 수 있나
    assigned: jnp.ndarray      # (K,) int32   지금 맡은 오더 (-1 = 없음)
    bay: jnp.ndarray           # (K,) float32 지금 위치 (bay)
    row: jnp.ndarray           # (K,) float32 지금 위치 (row)
    bay_min: jnp.ndarray       # (K,) int32   담당 구간
    bay_max: jnp.ndarray       # (K,) int32

    @property
    def k(self) -> int:
        return int(self.block.shape[0])


class WorldArrays(NamedTuple):
    """세계 하나 전체. `vmap` 으로 이 묶음을 B 개 쌓으면 세계 B 개가 된다."""

    clock: jnp.ndarray         # () float32  공용 시계
    end_s: jnp.ndarray         # () float32  이 세계를 언제까지 굴리나
    orders: OrderArrays
    cranes: CraneArrays
    #: 누적 비용 네 항 (원) — 항목을 갈라 둬야 어디서 났는지 보인다
    cost_wait: jnp.ndarray     # () float32
    cost_move: jnp.ndarray     # () float32
    cost_rehandle: jnp.ndarray # () float32
    cost_vessel: jnp.ndarray   # () float32
    #: ★넘침 표시 — 칸이 모자라 못 담은 수. 0 이 아니면 그 세계는 **실격**이다
    overflow: jnp.ndarray      # () int32

    @property
    def cost_total(self) -> jnp.ndarray:
        return self.cost_wait + self.cost_move + self.cost_rehandle + self.cost_vessel


def empty_orders(n: int) -> OrderArrays:
    i32 = lambda v: jnp.full((n,), v, jnp.int32)
    f32 = lambda v: jnp.full((n,), v, jnp.float32)
    return OrderArrays(
        block=i32(EMPTY_ID), flow=i32(0), stage=i32(ST_NOTICE),
        notice_s=f32(EMPTY_TIME), gate_in_s=f32(EMPTY_TIME),
        block_in_s=f32(EMPTY_TIME), service_s=f32(EMPTY_TIME),
        done_s=f32(EMPTY_TIME), gate_out_s=f32(EMPTY_TIME),
        duration_s=f32(0.0), travel_s=f32(0.0))


def empty_cranes(k: int) -> CraneArrays:
    i32 = lambda v: jnp.full((k,), v, jnp.int32)
    f32 = lambda v: jnp.full((k,), v, jnp.float32)
    return CraneArrays(
        block=i32(EMPTY_ID), status=i32(CR_IDLE), available_at=f32(0.0),
        assigned=i32(EMPTY_ID), bay=f32(1.0), row=f32(1.0),
        bay_min=i32(1), bay_max=i32(1))


def empty_world(n_orders: int, n_cranes: int, *, end_s: float = 86_400.0) -> WorldArrays:
    """빈 세계 하나. 칸 수는 **미리 정해** 넘어가면 표시한다(머리말 참조)."""
    z = jnp.float32(0.0)
    return WorldArrays(
        clock=jnp.float32(0.0), end_s=jnp.float32(end_s),
        orders=empty_orders(n_orders), cranes=empty_cranes(n_cranes),
        cost_wait=z, cost_move=z, cost_rehandle=z, cost_vessel=z,
        overflow=jnp.int32(0))


# ─────────────────────────────────────────────────── 성과지표
def turn_time_s(o: OrderArrays) -> jnp.ndarray:
    """턴타임 = 게이트 아웃 − 게이트 인. **나간 오더만** 유한한 값을 갖는다."""
    return o.gate_out_s - o.gate_in_s


def censored_turn_time_s(o: OrderArrays, end_s) -> jnp.ndarray:
    """못 나간 오더는 `end − 게이트인` 으로 검열한다.

    ★안 그러면 **못 나간 트럭이 표본에서 빠져** 정책이 이득을 본다
    (v4 `schema/lifecycle.py` 와 같은 규약).
    """
    entered = o.gate_in_s < EMPTY_TIME
    left = o.gate_out_s < EMPTY_TIME
    full = o.gate_out_s - o.gate_in_s
    part = jnp.maximum(0.0, end_s - o.gate_in_s)
    return jnp.where(entered, jnp.where(left, full, part), jnp.nan)
