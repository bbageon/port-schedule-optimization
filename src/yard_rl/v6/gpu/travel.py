"""크레인 이동시간을 **배열 함수로** ([[YR-327]] 조각 1 · 명세 §3·§5).

v5 `world/sim/travel_time.py:44-69` `move_container` 의 사본이다 — 파이썬 float 로
하던 계산을 JAX float64 스칼라로 한다. 정책·계획 단계가 크레인×오더로 vmap 해 부른다.

■ ★동등성 — v5 와 **비트까지 같은 답**이 목표
  한 사이클 소요시간은 10항의 합이다 (travel_time.py:56-67):

      빈 주행 gantry / 빈 주행 trolley / 하강(빈) / lock / 상승(적재)
      / 적재 주행 gantry / 적재 주행 trolley / 하강(적재) / unlock / 상승(빈)

  부동소수점 덧셈은 결합 순서에 따라 마지막 비트가 달라진다. 그래서 v5 의
  **좌결합 순서 그대로** `a + b + c + …` 한 식으로 적는다 — 파이썬이 왼쪽부터
  묶으니 JAX 도 같은 순서로 계산한다. 항을 모아 `sum()` 하거나 순서를 바꾸면 안 된다.
  여기의 곱(`|Δ|·길이`)은 전부 **나눗셈**으로 이어지고 덧셈으로 직접 이어지지 않아 FMA
  (곱셈-덧셈 융합) 대상이 아니다. 곱 뒤에 덧셈이 오는 식을 새로 쓰면 `exact.mul_exact` 를
  쓴다 — XLA_FLAGS 는 FMA 를 막지 못한다 (실측, exact.py 머리말).

■ ★나눗셈은 `div_exact` 로 — XLA 가 "broadcast 로 나누기" 를 역수 곱으로 바꾼다
  스펙(속도)은 크레인마다 하나라, 크레인×오더로 vmap 하면 `dist / broadcast(speed)`
  모양이 된다. XLA 대수 단순화기는 이것을 **무조건** `dist * broadcast(1/speed)` 로
  바꾸는데(플래그 없음 — 실측 2026-09-25, jax 0.11.2 CPU), 역수 곱은 나눗셈과 마지막
  비트가 다르다 (32건 중 3건, 최대 5.7e-14). 그래서 나눗셈마다 `div_exact` 를 쓴다:
  vmap 될 때 분모를 **명시적으로 펼치고 optimization_barrier 로 막아** 단순화기가
  broadcast 를 못 보게 한다. vmap 이 없으면 그냥 `a / b` 다.

■ 입력 모양
  spec   Spec7 — 크레인 1대의 스펙 7 스칼라 (cranes.spec_* 열에서 k 번째를 뽑아 만든다)
  start  (bay, row) float64 — 크레인 현위치 (연속 좌표 · 초기 위치는 lo+(k+0.5)(hi−lo)/n 라 정수가 아닐 수 있다)
  src/dst (3,) int32 — (bay, row, tier) · row 0 = 차선(트럭 인계지점) · tier 1-based
  g      기하 상수 — 속성 `tier_max` `bay_len` `row_w` `tier_h` 만 읽는다 (static · gpu/geom.py Geom 그대로 넘기면 된다)

■ 출력 MoveOut — (dur, loaded_m, empty_m, end_bay, end_row) 전부 float64
  loaded_m / empty_m 는 v5 처럼 **gantry 축 미터만** (trolley 는 KPI 에 안 잡힘).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.custom_batching import custom_vmap

#: 시각·거리 dtype — 동등성은 x64 에서만 (events.TIME_DTYPE 과 같은 이유)
F = jnp.float64


class Spec7(NamedTuple):
    """크레인 1대의 정적 스펙 7 스칼라 (models.py:126-132 순서). 각 칸 () float64.

    K 대 배열(cranes.spec_* (K,)) 에서 k 번째를 뽑아 만들거나, 그대로 (K,) 열을 넣고
    vmap 해도 된다 — NamedTuple 이라 pytree 다.
    """

    gantry_mps: jnp.ndarray       # gantry_speed_mps
    trolley_mps: jnp.ndarray      # trolley_speed_mps
    hoist_loaded_mps: jnp.ndarray # hoist_speed_loaded_mps
    hoist_empty_mps: jnp.ndarray  # hoist_speed_empty_mps
    lock_s: jnp.ndarray           # lock_time_s
    unlock_s: jnp.ndarray         # unlock_time_s
    truck_pos_s: jnp.ndarray      # truck_positioning_time_s (move_container 는 안 쓴다 — 계획이 더한다, engine.py 참조)


class MoveOut(NamedTuple):
    """v5 MoveResult 의 배열판 (travel_time.py:16-22). 각 칸 () float64."""

    dur: jnp.ndarray        # duration_s
    loaded_m: jnp.ndarray   # loaded_gantry_m  (src→dst gantry 미터)
    empty_m: jnp.ndarray    # empty_gantry_m   (현위치→src gantry 미터)
    end_bay: jnp.ndarray    # float(dst_bay)
    end_row: jnp.ndarray    # float(dst_row)


@dataclass(frozen=True)
class TravelGeom:
    """이동시간에 필요한 기하 상수 네 개 — 시험·단독 사용용. 엔진은 gpu/geom.py 의
    공용 `Geom` 을 넘기면 된다 (속성 이름 tier_max·bay_len·row_w·tier_h 가 같다).
    static 인자(jit 의 static_argnames / partial) 로 넘긴다."""

    tier_max: int     # T
    bay_len: float    # bay_length_m
    row_w: float      # row_width_m
    tier_h: float     # tier_height_m

    @classmethod
    def from_block_geometry(cls, geom) -> "TravelGeom":
        """v5 BlockGeometry (models.py:110-118) 에서."""
        return cls(tier_max=int(geom.tier_max), bay_len=float(geom.bay_length_m),
                   row_w=float(geom.row_width_m), tier_h=float(geom.tier_height_m))


def spec7_from_crane_spec(spec) -> Spec7:
    """v5 CraneSpec (models.py:122-132) → Spec7 (float64 스칼라). 호스트 변환용."""
    return Spec7(
        gantry_mps=jnp.asarray(spec.gantry_speed_mps, F),
        trolley_mps=jnp.asarray(spec.trolley_speed_mps, F),
        hoist_loaded_mps=jnp.asarray(spec.hoist_speed_loaded_mps, F),
        hoist_empty_mps=jnp.asarray(spec.hoist_speed_empty_mps, F),
        lock_s=jnp.asarray(spec.lock_time_s, F),
        unlock_s=jnp.asarray(spec.unlock_time_s, F),
        truck_pos_s=jnp.asarray(spec.truck_positioning_time_s, F))


# ───────────────────────────────────────────────── ★정확한 나눗셈 (머리말 참조)
@custom_vmap
def div_exact(a, b):
    """IEEE 나눗셈 `a / b` 그대로 — vmap 아래서도 역수 곱으로 안 바뀐다.

    vmap 이 없으면 보통 나눗셈이다. vmap 되면 아래 규칙이 분모를 배치 모양으로 펼치고
    optimization_barrier 뒤에서 나눈다 — XLA 가 `divide(a, broadcast(b))` 패턴을 못 본다.
    """
    return a / b


@div_exact.def_vmap
def _div_exact_vmap(axis_size, in_batched, a, b):
    a_b, b_b = in_batched
    if not a_b:
        a = jnp.broadcast_to(a[None], (axis_size,) + a.shape)
    if not b_b:
        b = jnp.broadcast_to(b[None], (axis_size,) + b.shape)
    a, b = lax.optimization_barrier((a, b))   # 펼친 분모를 단순화기에서 숨긴다
    return a / b, True


# ───────────────────────────────────────────────── 축별 항 (travel_time.py:29-41)
def gantry_m(g, from_bay, to_bay):
    """|from − to| × bay 길이 (travel_time.py:29-30). 인자 dtype 은 호출자 몫 — v5 도 int·float 섞어 넣는다."""
    return jnp.abs(from_bay - to_bay) * jnp.asarray(g.bay_len, F)


def trolley_m(g, from_row, to_row):
    """|from − to| × row 폭 (travel_time.py:33-34)."""
    return jnp.abs(from_row - to_row) * jnp.asarray(g.row_w, F)


def hoist_leg_s(g, spec: Spec7, tier, *, loaded: bool):
    """스프레더가 통과높이(T+1) ↔ tier 를 한 번 가는 시간 (travel_time.py:37-41).

    v5 는 `((T+1) − tier) * tier_h` 를 **정수 뺄셈 뒤 실수 곱**으로 한다 — 여기서도
    정수 상태로 빼고 나서 float64 로 올려 곱한다 (작은 정수라 어느 쪽이든 정확하지만
    순서를 v5 그대로 둔다). `loaded` 는 파이썬 bool (정적 분기).
    """
    dist = ((g.tier_max + 1) - tier).astype(F) * jnp.asarray(g.tier_h, F)
    speed = spec.hoist_loaded_mps if loaded else spec.hoist_empty_mps
    return div_exact(dist, speed)


# ───────────────────────────────────────────────── ★한 사이클 (travel_time.py:44-69)
def move_container(spec: Spec7, start_bay, start_row, src, dst, g) -> MoveOut:
    """컨테이너 1개를 src → dst 로 옮기는 1 사이클 — v5 44-69행과 **항 결합 순서까지 같다**.

    빈 주행(현위치→src) → 하강(빈)·lock·상승(적재) → 적재 주행(src→dst)
    → 하강(적재)·unlock·상승(빈).

    src/dst 는 (3,) int32 (bay, row, tier). start_bay/start_row 는 float64.
    """
    start_bay = jnp.asarray(start_bay, F)
    start_row = jnp.asarray(start_row, F)
    src = jnp.asarray(src, jnp.int32)
    dst = jnp.asarray(dst, jnp.int32)
    s_bay, s_row, s_tier = src[0], src[1], src[2]
    d_bay, d_row, d_tier = dst[0], dst[1], dst[2]

    # v5 52-55행 — 빈 주행은 float 현위치 − int 슬롯, 적재 주행은 int − int (정확한 정수 차)
    e_gantry = gantry_m(g, start_bay, s_bay)
    e_trolley = trolley_m(g, start_row, s_row)
    l_gantry = gantry_m(g, s_bay, d_bay)
    l_trolley = trolley_m(g, s_row, d_row)

    # ★v5 56-67행 — 10항 좌결합. 줄을 합치거나 순서를 바꾸지 말 것. 나눗셈은 div_exact (머리말).
    duration = (
        div_exact(e_gantry, spec.gantry_mps)
        + div_exact(e_trolley, spec.trolley_mps)
        + hoist_leg_s(g, spec, s_tier, loaded=False)   # 하강(빈)
        + spec.lock_s
        + hoist_leg_s(g, spec, s_tier, loaded=True)    # 상승(적재)
        + div_exact(l_gantry, spec.gantry_mps)
        + div_exact(l_trolley, spec.trolley_mps)
        + hoist_leg_s(g, spec, d_tier, loaded=True)    # 하강(적재)
        + spec.unlock_s
        + hoist_leg_s(g, spec, d_tier, loaded=False)   # 상승(빈)
    )
    return MoveOut(dur=duration, loaded_m=l_gantry, empty_m=e_gantry,
                   end_bay=d_bay.astype(F), end_row=d_row.astype(F))   # 68-69행 float(dst)


def estimate_reach_s(spec: Spec7, g, from_bay, from_row, to_bay, to_row):
    """빈 크레인이 목표 지점까지 가는 예상시간 (travel_time.py:72-76) — NEAREST_JOB 판정용."""
    return (div_exact(gantry_m(g, from_bay, to_bay), spec.gantry_mps)
            + div_exact(trolley_m(g, from_row, to_row), spec.trolley_mps))
