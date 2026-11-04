"""야드 스택을 **배열로** — 빈 칸 찾기·blocker·재조작 용량·적재·제거 ([[YR-327]] 조각 1).

v5 `world/sim/stack.py` 의 `YardStacks` 는 `dict[(bay,row)] → list[str]` 와 문자열 번호를
쓴다. 여기서는 전부 **고정 모양 배열**이고, 컨테이너는 정수 번호다 (형은 gpu/state.py):

    grid[b, r, t]   그 칸·그 단에 놓인 컨테이너 번호  (빈 단 = -1)     b,r,t 는 0-based
    height[b, r]    쌓인 단 수 (= len(pile))
    top_size[b, r]  맨 위 컨테이너 규격 (빈 칸 = -1)      ← find_slot 이 매번 쓰는 캐시
    c_bay/c_row/c_tier[c]   컨테이너 c 의 좌표 (v5 Container 와 같은 **1-based**, 없으면 -1)
    c_size[c]       규격 SZ_FT20 0 · SZ_FT40 1 · SZ_FT45 2 (v5 ContainerSize 선언 순)
    c_avail[c]      work_available (Hold·검사)      c_alive[c]  야드에 실재하나

■ ★find_slot 은 "가까운 bay 부터 보다가 자르는" 루프가 아니라 **격자 전체를 한 번에** 본다
  v5 (stack.py:139-168) 는 bay 를 가까운 순으로 돌며 거리 하한이 최선을 넘으면 멈춘다.
  잘리는 후보는 비용이 **엄격히** 크므로(stack.py:149-150) 전수 argmin 과 답이 같다.
  동률은 v5 의 `key = (cost, bay, row)` 사전식 — 여기서는 최소 비용 칸들 중 행우선
  평탄화의 **첫 칸**(bay 작은 것, 그다음 row 작은 것)이 정확히 그 순서다.

■ ★비용식은 v5 의 결합 순서 그대로 — `(g + |Δrow|·row_w) + top·tier_h` (stack.py:164)
  동률 판정이 float64 의 **마지막 비트**에 달려 있다. 그래서
    · 좌표·거리·비용은 전부 float64 (x64 모드 — `jax_enable_x64`)
    · 곱 세 개를 **반드시** 먼저 실체화한 뒤 더한다 (`exact.mul_exact` = optimization_barrier).
      XLA CPU 는 곱-덧셈을 한 번에 반올림(FMA)하고 **XLA_FLAGS 는 이를 막지 못한다**
      (실측 2026-09-25 · jax 0.11.2: 플래그 8종 전부 무효, barrier 만 유효 — exact.py 머리말).
      이 무대의 상수(6.5/2.9/2.6)에서는 20,000 질의(정확 동률 747건)에서 운 좋게 안 갈렸지만
      융합 자체는 확인됐으므로 장벽은 '보험' 이 아니라 **필수**다 — 끄는 스위치를 두지 않는다
      (학습 모드에서도 같다: 결정에 닿는 유일한 방어선이다).
    · gpu/geom.py 의 `row_cost`·`tier_cost` 표는 |Δrow| 가 정수일 때만 쓸 수 있어 여기서는
      안 쓴다 (기준점 near_row 가 연속값인 질의도 받는다).

■ 예외는 **위반 비트**로 — v5 가 `raise` 하는 자리(stack.py:55, 64, 66)에서 세계를
  실격 표시하고 상태는 **바꾸지 않는다** (배열 세계엔 예외가 없다).
    V_NOT_TOP 1  제거 대상이 맨 위가 아니다(또는 번호가 무효·야드에 없다)
    V_TIER    2  tier_max 를 넘겨 쌓으려 했다
    V_SIZE    4  다른 규격 위에 쌓으려 했다
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .exact import mul_exact
from .geom import Geom
from .state import (EMPTY_ID, SZ_FT20, SZ_FT40, SZ_FT45, V_NOT_TOP, V_SIZE, V_TIER,
                    ContArrays, StackArrays, empty_conts, empty_stacks)

__all__ = ["Geom", "StackArrays", "ContArrays", "empty_stacks", "empty_conts",
           "EMPTY_ID", "SZ_FT20", "SZ_FT40", "SZ_FT45", "SIZE_INDEX",
           "V_NOT_TOP", "V_TIER", "V_SIZE", "COORD_DTYPE",
           "find_slot", "blockers_above", "rehandle_capacity_ok", "place", "remove",
           "from_v5_stacks", "to_v5_piles"]

#: v5 `ContainerSize.value` → 규격 번호 (호스트 변환용)
SIZE_INDEX = {"FT20": SZ_FT20, "FT40": SZ_FT40, "FT45": SZ_FT45}

#: 좌표·거리·비용 dtype — 동등성은 float64 에서만 (머리말 참조)
COORD_DTYPE = jnp.float64


def _cells(B: int, R: int):
    """1-based bay 열벡터 (B,1) 와 row 행벡터 (1,R)."""
    bays = jnp.arange(1, B + 1, dtype=jnp.int32)[:, None]
    rows = jnp.arange(1, R + 1, dtype=jnp.int32)[None, :]
    return bays, rows


# ───────────────────────────────────────────────── 빈 칸 찾기
def find_slot(height, top_size, excluded, size, bay_min, bay_max, near_bay, near_row,
              g: Geom):
    """규격·tier·담당 구간을 만족하는 **최근접 칸** — v5 `YardStacks.find_slot` (stack.py:101-168).

    입력: height/top_size (B,R) int32 · excluded (B,R) bool · size int · bay_min/bay_max int
          (담당 구간, 1-based 닫힌 구간) · near_bay/near_row float (기준점, 연속 좌표)
          g: `tier_max` `bay_len` `row_w` 만 읽는다 (static)
    출력: (found bool, bay int32 1-based, row int32 1-based) — 없으면 (False, -1, -1)

    비용 = |Δbay|·bay_len + |Δrow|·row_w + top·tier_h, 동률은 (cost, bay, row) 사전식 최소.
    """
    B, R = height.shape
    T = g.tier_max
    F = COORD_DTYPE
    bays, rows = _cells(B, R)
    # ★스칼라 입력은 전부 배열로 — 파이썬 bool 에 `~` 를 쓰면 -2 가 돼 마스크가 깨진다
    size, bay_min, bay_max = (jnp.asarray(v, jnp.int32) for v in (size, bay_min, bay_max))
    nb = jnp.asarray(near_bay, F)
    nr = jnp.asarray(near_row, F)
    # ★곱 세 개 — v5 와 같은 피연산자 (142행 g · 164행 두 항). 각각 따로 반올림돼야 하므로
    #   mul_exact(장벽) 로 실체화한다 — 플래그가 아니라 이것이 FMA 방어다 (exact.py 머리말).
    gc = mul_exact(jnp.abs(nb - bays.astype(F)), jnp.asarray(g.bay_len, F))   # (B,1) 142행
    tc = mul_exact(jnp.abs(nr - rows.astype(F)), jnp.asarray(g.row_w, F))     # (1,R)
    hc = mul_exact(height.astype(F), jnp.asarray(g.tier_h, F))                # (B,R)
    cost = (gc + tc) + hc                                                # 164행 좌결합 그대로
    # 154행 exclude · 160행 tier · 162행 규격 · 139행 담당 구간
    valid = (~excluded & (height < T) & ((height == 0) | (top_size == size))
             & (bays >= bay_min) & (bays <= bay_max))
    c = jnp.where(valid, cost, jnp.inf)
    m = c.min()
    found = jnp.isfinite(m)
    tie = valid & (c == m)
    flat = jnp.argmax(tie.reshape(-1))            # 행우선 → bay 작은 것, 그다음 row (165-166행)
    bay = jnp.where(found, flat // R + 1, EMPTY_ID).astype(jnp.int32)
    row = jnp.where(found, flat % R + 1, EMPTY_ID).astype(jnp.int32)
    return found, bay, row


# ───────────────────────────────────────────────── blocker
def blockers_above(grid, height, c_bay, c_row, c_tier, c, g: Geom):
    """대상 c **위**에 놓인 컨테이너들 — 위에서부터 제거 순서 (stack.py:35-40 `reversed`).

    출력: (blk (T-1,) int32 — k < n_block 만 유효, 나머지 -1;  n_block int32)
    번호가 무효이거나 야드에 없는(좌표 -1) 컨테이너면 n_block = 0.
    """
    T = g.tier_max
    C = c_bay.shape[0]
    c = jnp.asarray(c, jnp.int32)
    cc = jnp.clip(c, 0, C - 1)
    ok = (c >= 0) & (c < C) & (c_bay[cc] > 0)
    b = jnp.clip(c_bay[cc] - 1, 0, grid.shape[0] - 1)
    r = jnp.clip(c_row[cc] - 1, 0, grid.shape[1] - 1)
    h = height[b, r]
    n_block = jnp.where(ok, jnp.clip(h - c_tier[cc], 0, T - 1), 0).astype(jnp.int32)
    k = jnp.arange(T - 1, dtype=jnp.int32)
    tier_idx = jnp.clip(h - 1 - k, 0, T - 1)       # 맨 위(h-1)부터 아래로
    blk = jnp.where(k < n_block, grid[b, r, tier_idx], EMPTY_ID).astype(jnp.int32)
    return blk, n_block


def rehandle_capacity_ok(height, top_size, tgt_bay, tgt_row, n_block, bay_min, bay_max,
                         g: Geom):
    """재조작 칸이 **충분히** 있나 — v5 `rehandle_capacity_ok` (stack.py:72-98).

    tgt_bay/tgt_row 는 대상 컨테이너의 1-based 좌표, n_block 은 `blockers_above` 의 수.
    스택은 항상 단일 규격이라 blocker 전부 = 맨 위 규격(top_size[대상 칸]).
    담당 구간 안·대상 칸 제외·tier 여유·규격 호환 칸의 잔여 단 수 합 ≥ n_block 이면 참.
    """
    B, R = height.shape
    T = g.tier_max
    bays, rows = _cells(B, R)
    tgt_bay, tgt_row, n_block, bay_min, bay_max = (
        jnp.asarray(v, jnp.int32) for v in (tgt_bay, tgt_row, n_block, bay_min, bay_max))
    b = jnp.clip(tgt_bay - 1, 0, B - 1)
    r = jnp.clip(tgt_row - 1, 0, R - 1)
    size_blk = top_size[b, r]
    mask = ((bays >= bay_min) & (bays <= bay_max)                     # 86행
            & ~((bays == tgt_bay) & (rows == tgt_row))                # 88행 원천 제외
            & (height < T)                                            # 91행
            & ((height == 0) | (top_size == size_blk)))               # 93행
    capacity = jnp.sum(jnp.where(mask, T - height, 0))                # 95행
    return (n_block == 0) | (capacity >= n_block)                     # 80-81, 96행


# ───────────────────────────────────────────────── 변형
def _select(ok, new, old):
    """ok 면 new, 아니면 old — 들어온 NamedTuple 형을 그대로 돌려준다."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(ok, x, y), new, old)


def place(stacks: StackArrays, conts: ContArrays, c, bay, row):
    """컨테이너 c 를 (bay,row) 맨 위에 쌓는다 — v5 `place` (stack.py:61-70).

    출력: (stacks', conts', viol int32). 위반(2 TIER · 4 SIZE · 1 무효 번호)이면 상태는 그대로.
    """
    B, R, T = stacks.grid.shape
    C = conts.c_bay.shape[0]
    c, bay, row = (jnp.asarray(v, jnp.int32) for v in (c, bay, row))
    cc = jnp.clip(c, 0, C - 1)
    b = jnp.clip(bay - 1, 0, B - 1)
    r = jnp.clip(row - 1, 0, R - 1)
    ok_c = (c >= 0) & (c < C) & (bay >= 1) & (bay <= B) & (row >= 1) & (row <= R)
    h = stacks.height[b, r]
    sz = conts.c_size[cc]
    ok_tier = h < T                                                    # 63행
    ok_size = (h == 0) | (stacks.top_size[b, r] == sz)                 # 65행 stack_size_ok
    ok = ok_c & ok_tier & ok_size
    viol = (jnp.where(~ok_c, V_NOT_TOP, 0)
            | jnp.where(ok_c & ~ok_tier, V_TIER, 0)
            | jnp.where(ok_c & ok_tier & ~ok_size, V_SIZE, 0)).astype(jnp.int32)
    ht = jnp.clip(h, 0, T - 1)
    new_stacks = stacks._replace(
        grid=stacks.grid.at[b, r, ht].set(c),
        height=stacks.height.at[b, r].set(h + 1),
        top_size=stacks.top_size.at[b, r].set(sz))
    new_conts = conts._replace(                                        # 68-69행
        c_bay=conts.c_bay.at[cc].set(bay),
        c_row=conts.c_row.at[cc].set(row),
        c_tier=conts.c_tier.at[cc].set(h + 1),
        c_alive=conts.c_alive.at[cc].set(True))
    return _select(ok, new_stacks, stacks), _select(ok, new_conts, conts), viol


def remove(stacks: StackArrays, conts: ContArrays, c):
    """컨테이너 c 를 스택 **맨 위**에서 뺀다 — v5 `remove` (stack.py:50-59).

    출력: (stacks', conts', viol int32). 맨 위가 아니거나 야드에 없으면 1 NOT_TOP, 상태 그대로.
    뺀 컨테이너는 좌표 -1 · alive False (v5 는 `containers` 에서 지운다 — 58행).
    """
    B, R, T = stacks.grid.shape
    C = conts.c_bay.shape[0]
    c = jnp.asarray(c, jnp.int32)
    cc = jnp.clip(c, 0, C - 1)
    b = jnp.clip(conts.c_bay[cc] - 1, 0, B - 1)
    r = jnp.clip(conts.c_row[cc] - 1, 0, R - 1)
    h = stacks.height[b, r]
    top = stacks.grid[b, r, jnp.clip(h - 1, 0, T - 1)]
    ok = ((c >= 0) & (c < C) & conts.c_alive[cc] & (conts.c_bay[cc] > 0)
          & (h > 0) & (top == c))                                     # 54행
    viol = jnp.where(ok, 0, V_NOT_TOP).astype(jnp.int32)
    below = stacks.grid[b, r, jnp.clip(h - 2, 0, T - 1)]
    new_top_size = jnp.where(h - 1 > 0, conts.c_size[jnp.clip(below, 0, C - 1)], EMPTY_ID)
    new_stacks = stacks._replace(
        grid=stacks.grid.at[b, r, jnp.clip(h - 1, 0, T - 1)].set(EMPTY_ID),
        height=stacks.height.at[b, r].set(h - 1),
        top_size=stacks.top_size.at[b, r].set(new_top_size.astype(jnp.int32)))
    new_conts = conts._replace(
        c_bay=conts.c_bay.at[cc].set(EMPTY_ID),
        c_row=conts.c_row.at[cc].set(EMPTY_ID),
        c_tier=conts.c_tier.at[cc].set(EMPTY_ID),
        c_alive=conts.c_alive.at[cc].set(False))
    return _select(ok, new_stacks, stacks), _select(ok, new_conts, conts), viol


# ───────────────────────────────────────────────── 호스트 변환 (파이썬 · numpy)
def from_v5_stacks(yard, g: Geom, *, cont_ids: list[str] | None = None,
                   n_cont: int | None = None):
    """v5 `YardStacks` → (StackArrays, ContArrays, cont_ids).

    컨테이너 번호 = `cont_ids` 의 위치. 기본은 `sorted(yard.containers)` (stack.py:18 과 같은
    결정론 순서). `n_cont` 를 더 크게 주면 뒤 칸은 반입 예비칸(좌표 -1·alive False·규격 -1
    — 규격은 호스트가 오더의 inbound_size 로 채운다).
    타입을 import 하지 않고 속성만 읽는다 — gpu/ 는 world/ 에 의존하지 않는다.
    """
    B, R, T = g.bay_count, g.row_count, g.tier_max
    if cont_ids is None:
        cont_ids = sorted(yard.containers)
    if n_cont is None:
        n_cont = len(cont_ids)
    idx = {cid: i for i, cid in enumerate(cont_ids)}
    grid = np.full((B, R, T), EMPTY_ID, np.int32)
    height = np.zeros((B, R), np.int32)
    top_size = np.full((B, R), EMPTY_ID, np.int32)
    c_bay = np.full((n_cont,), EMPTY_ID, np.int32)
    c_row = np.full((n_cont,), EMPTY_ID, np.int32)
    c_tier = np.full((n_cont,), EMPTY_ID, np.int32)
    c_size = np.full((n_cont,), EMPTY_ID, np.int32)
    c_avail = np.ones((n_cont,), bool)
    c_alive = np.zeros((n_cont,), bool)
    for (bay, row), pile in yard._stacks.items():
        for t, cid in enumerate(pile):
            grid[bay - 1, row - 1, t] = idx[cid]
        height[bay - 1, row - 1] = len(pile)
        if pile:
            top_size[bay - 1, row - 1] = SIZE_INDEX[yard.containers[pile[-1]].size.value]
    for cid, cont in yard.containers.items():
        i = idx[cid]
        c_bay[i], c_row[i], c_tier[i] = cont.bay, cont.row, cont.tier
        c_size[i] = SIZE_INDEX[cont.size.value]
        c_avail[i] = bool(cont.work_available)
        c_alive[i] = True
    stacks = StackArrays(jnp.asarray(grid), jnp.asarray(height), jnp.asarray(top_size))
    conts = ContArrays(jnp.asarray(c_bay), jnp.asarray(c_row), jnp.asarray(c_tier),
                       jnp.asarray(c_size), jnp.asarray(c_avail), jnp.asarray(c_alive))
    return stacks, conts, list(cont_ids)


def to_v5_piles(stacks: StackArrays, cont_ids: list[str]) -> dict[tuple[int, int], list[str]]:
    """격자 → v5 `_stacks` 모양의 사전 (비어 있지 않은 칸만) — 대조용."""
    grid = np.asarray(stacks.grid)
    height = np.asarray(stacks.height)
    out: dict[tuple[int, int], list[str]] = {}
    B, R, _ = grid.shape
    for b in range(B):
        for r in range(R):
            h = int(height[b, r])
            if h > 0:
                out[(b + 1, r + 1)] = [cont_ids[int(grid[b, r, t])] for t in range(h)]
    return out
