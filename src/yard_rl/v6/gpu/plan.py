"""SERVE 작업 계획을 **배열 함수로** ([[YR-327]] 조각 1 · 명세 §5).

v5 `world/integrated/engine.py:552-670` `_plan` 의 SERVE 두 경로(STORE·RETRIEVE) 를
옮긴 것이다. v5 는 파이썬 루프로 blocker 를 하나씩 치우며 계획을 세운다 — 여기서는
**고정 길이 scan** 으로 같은 일을 하고, 결과는 `PlanOut` 배열 묶음(모양 고정)이다.
REPOSITION·PRE_REHANDLE 은 조각 3 몫이라 여기 없다 (kind 는 항상 SERVE).

■ 두 경로 (engine.py:596-663)
  (a) STORE   반입 — 인계점(차선)에서 빈 칸으로 1 이동. 칸은 `find_slot` 이 크레인
              현위치를 기준점으로 고른다(제외 = 예약 칸 ∪ extra). 이동 1 · 재조작 0.
  (b) RETRIEVE 반출 — 대상 위 blocker 를 **위에서부터** 하나씩 다른 칸으로 옮기고(재조작),
              마지막에 대상을 차선으로 내린다. blocker 의 목적지는 blocker **자기 좌표**를
              기준점으로 고른다(626행). 이동 = blocker 수 + 1.

■ ★같은 답을 내기 위한 규칙 (동등성)
  · 시각·거리·좌표는 전부 float64. 누적합은 v5 와 같은 순서 — `total_s` 는 0.0 에서 시작해
    이동 순서대로 더하고, 외부트럭이면 **맨 마지막**에 `truck_positioning_time_s` 를 더한다
    (610-611, 656-657행). 항을 모아 더하거나 순서를 바꾸면 마지막 비트가 갈린다.
  · find_slot 의 제외 집합은 계획 중에 **자란다** — blocker 목적지가 정해지면 즉시 제외에
    들어가고(645행), 원천 칸(대상 pile)은 매 질의마다 제외된다(622행). 스택 자체는 계획
    중 불변이라 `height`/`top_size` 를 그대로 읽는다(YR-047 격리 전제).
  · 통로(corridor) = 크레인 시작 bay 와 계획이 건드린 모든 bay 의 min/max (646, 665행).
  · 레인은 계획이 아니라 `_jobref` 가 정한다(537-546행): RETRIEVE 는 **대상 bay**, STORE 는
    **예약 제외 없이** 크레인 위치에서 찾은 칸의 bay — 계획이 실제로 고른(제외 뒤) 칸과
    다를 수 있다. 그래서 STORE 는 find_slot 을 두 번 부른다(반박 검증 '완전성·누락' 항).
  · 예외 자리는 위반 비트로: 634-636행의 find_slot 후조건 위반은 `viol |= V_PLAN_POSTCOND`.
    find_slot 이 규칙을 지키는 한 절대 켜지지 않는다.

■ 고정 길이 scan
  blocker 는 최대 T−1 개(대상이 바닥일 때)라 scan 단계 수는 M−1 = tier_max−1 로 고정이고,
  `i < n_block` 인 단계만 살아 있다. 죽은 단계는 계산은 하되 carry 를 바꾸지 않는다
  (`where(active, 새값, 옛값)`). 이동 칸 M 개 중 [0, n_block) 이 재조작, n_block 이 대상.

■ 모양
  입력  world: BlockWorld (또는 `plan_serve_core` 에 낱개 배열), k·n 스칼라 int32,
        extra_excluded (B,R) bool (dry-run 순차 예약 — 명세 §8 배정 scan 이 carry.res 로 넘김)
  출력  PlanOut — 스칼라 열 + slots (B,R) + 이동 목록 (M,)·(M,3). ok 가 거짓이면 v5 None.
  크레인×오더는 `plan_serve_all` (vmap 두 겹) → 각 열 앞에 (K,N) 이 붙는다.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .geom import Geom
from .reserve import reserved_slots
from .stack_ops import blockers_above, find_slot
from .state import (EMPTY_ID, MV_REHANDLE, MV_RETRIEVE, MV_STORE, PK_SERVE, V_PLAN_POSTCOND,
                    BlockWorld, ContArrays, CraneArrays, StackArrays)
from .travel import F, Spec7, move_container

__all__ = ["PlanOut", "plan_serve", "plan_serve_core", "plan_serve_all", "crane_spec7",
           "onehot_cell", "lane_of_bay"]


class PlanOut(NamedTuple):
    """v5 JobPlan (jobplan.py:45-61) + Move 목록 (15-27) 의 배열판 — 한 (크레인, 오더) 쌍."""

    ok: jnp.ndarray         # () bool   계획 성립 (거짓 = v5 None)
    kind: jnp.ndarray       # () int32  PK_SERVE 고정 (조각 3 이 PRE/REPO 를 더한다)
    lo: jnp.ndarray         # () f64    corridor (lo, hi) — 건드린 bay 의 min/max
    hi: jnp.ndarray         # () f64
    dur: jnp.ndarray        # () f64    duration_s (이동 합 + 외부트럭 위치잡기)
    end_bay: jnp.ndarray    # () f64    마지막 이동의 목적지 (float)
    end_row: jnp.ndarray    # () f64
    rehandles: jnp.ndarray  # () int32  blocker 수
    loaded_m: jnp.ndarray   # () f64    loaded_gantry_m 합
    empty_m: jnp.ndarray    # () f64    empty_gantry_m 합
    lane: jnp.ndarray       # () int32  레인 번호 (-1 = 없음) — _jobref 규칙
    slots: jnp.ndarray      # (B,R) bool 예약 칸 (frozenset slots)
    n_moves: jnp.ndarray    # () int32  유효 이동 수 (i ≥ n_moves 는 빈 칸)
    mv_cont: jnp.ndarray    # (M,)   int32  옮길 컨테이너 (-1)
    mv_src: jnp.ndarray     # (M,3)  int32  (bay,row,tier) 1-based · 차선 row = transfer_row
    mv_dst: jnp.ndarray     # (M,3)  int32
    mv_kind: jnp.ndarray    # (M,)   int32  MV_REHANDLE 0 · MV_RETRIEVE 1 · MV_STORE 2 (-1)
    viol: jnp.ndarray       # () int32  위반 비트 (V_PLAN_POSTCOND 만)


# ───────────────────────────────────────────────── 작은 도구
def onehot_cell(bay, row, B: int, R: int):
    """1-based (bay,row) 한 칸의 (B,R) 마스크. bay/row 가 -1(없음) 이면 전부 False."""
    bays = jnp.arange(1, B + 1, dtype=jnp.int32)[:, None]
    rows = jnp.arange(1, R + 1, dtype=jnp.int32)[None, :]
    return (bays == jnp.asarray(bay, jnp.int32)) & (rows == jnp.asarray(row, jnp.int32))


def lane_of_bay(bay, g: Geom):
    """engine.py:271-273 `_lane_for`: ids[(bay−1) % L]. L=0 이면 -1 (v5 None). bay ≤ 0 도 -1."""
    bay = jnp.asarray(bay, jnp.int32)
    if int(g.n_lanes) <= 0:
        return jnp.full((), EMPTY_ID, jnp.int32)
    return jnp.where(bay >= 1, (bay - 1) % jnp.int32(g.n_lanes), EMPTY_ID).astype(jnp.int32)


def crane_spec7(cr: CraneArrays, k) -> Spec7:
    """cranes 의 스펙 7열에서 k 번째를 뽑아 Spec7 로 (travel.py 머리말)."""
    return Spec7(cr.spec_gantry[k], cr.spec_trolley[k], cr.spec_hoist_loaded[k],
                 cr.spec_hoist_empty[k], cr.spec_lock[k], cr.spec_unlock[k], cr.spec_truck_pos[k])


def _transfer_row(g: Geom) -> int:
    """차선 row — v5 BlockGeometry.transfer_row 는 int (models.py:118). 이동 슬롯은 int32 라
    정수여야 한다; 정수가 아니면 조용히 잘리지 않게 여기서 막는다."""
    tr = int(g.transfer_row)
    if float(tr) != float(g.transfer_row):
        raise ValueError(f"transfer_row={g.transfer_row!r} 가 정수가 아니다 — 이동 슬롯은 정수 좌표")
    return tr


def _empty_moves(M: int):
    return (jnp.full((M,), EMPTY_ID, jnp.int32), jnp.full((M, 3), EMPTY_ID, jnp.int32),
            jnp.full((M, 3), EMPTY_ID, jnp.int32), jnp.full((M,), EMPTY_ID, jnp.int32))


# ───────────────────────────────────────────────── (a) STORE (engine.py:596-616)
def _plan_store(stacks: StackArrays, excluded0, spec: Spec7, cur_bay, cur_row, bay_min, bay_max,
                is_ext, inbound_cont, inbound_size, g: Geom) -> PlanOut:
    B, R, T = stacks.grid.shape
    M = g.n_moves
    TR = _transfer_row(g)
    height, top_size = stacks.height, stacks.top_size

    # 597행 _store_slot(exclude=exclude) — 기준점은 크레인 현위치
    found, db, dr = find_slot(height, top_size, excluded0, inbound_size, bay_min, bay_max,
                              cur_bay, cur_row, g)
    bi = jnp.clip(db - 1, 0, B - 1)
    ri = jnp.clip(dr - 1, 0, R - 1)
    dtier = height[bi, ri] + 1                                        # 601행 top_tier+1
    src = jnp.stack([db, jnp.int32(TR), jnp.int32(1)]).astype(jnp.int32)   # 602행 (db, transfer_row, 1)
    dst = jnp.stack([db, dr, dtier]).astype(jnp.int32)
    mv = move_container(spec, cur_bay, cur_row, src, dst, g)         # 603행
    # 609-611행: total_s = 0.0 + dur (= dur 정확) ; 외부트럭이면 + truck_pos
    total = jnp.where(is_ext, mv.dur + spec.truck_pos_s, mv.dur)
    db_f = db.astype(F)
    lo = jnp.minimum(cur_bay, db_f)                                   # 614행 touched={cur_bay, db}
    hi = jnp.maximum(cur_bay, db_f)
    slots = onehot_cell(db, dr, B, R) & found                          # 615행

    # 레인: _jobref(540-543행) 는 **제외 없이** 찾은 칸의 bay 를 쓴다 — 계획 칸과 다를 수 있다
    found_l, lb, _ = find_slot(height, top_size, jnp.zeros((B, R), bool), inbound_size,
                               bay_min, bay_max, cur_bay, cur_row, g)
    lane = jnp.where(found_l, lane_of_bay(lb, g), EMPTY_ID).astype(jnp.int32)

    mv_cont, mv_src, mv_dst, mv_kind = _empty_moves(M)
    mv_cont = mv_cont.at[0].set(jnp.where(found, inbound_cont, EMPTY_ID))
    mv_src = mv_src.at[0].set(jnp.where(found, src, EMPTY_ID))
    mv_dst = mv_dst.at[0].set(jnp.where(found, dst, EMPTY_ID))
    mv_kind = mv_kind.at[0].set(jnp.where(found, MV_STORE, EMPTY_ID))
    return PlanOut(
        ok=found, kind=jnp.int32(PK_SERVE), lo=lo, hi=hi, dur=total,
        end_bay=mv.end_bay, end_row=mv.end_row, rehandles=jnp.int32(0),
        loaded_m=mv.loaded_m, empty_m=mv.empty_m, lane=lane, slots=slots,
        n_moves=jnp.where(found, 1, 0).astype(jnp.int32),
        mv_cont=mv_cont, mv_src=mv_src, mv_dst=mv_dst, mv_kind=mv_kind,
        viol=jnp.int32(0))


# ───────────────────────────────────────────────── (b) RETRIEVE (engine.py:618-663)
class _Carry(NamedTuple):
    """blocker scan 의 carry — v5 루프의 지역변수들 (593-600행 + 641-647행)."""

    cur_bay: jnp.ndarray    # () f64  크레인 가상 위치 (이동마다 목적지로)
    cur_row: jnp.ndarray
    excluded: jnp.ndarray   # (B,R) bool  자라는 제외 집합 (645행)
    total: jnp.ndarray      # () f64  total_s
    loaded: jnp.ndarray     # () f64  loaded_m
    empty: jnp.ndarray      # () f64  empty_m
    rehandles: jnp.ndarray  # () int32
    lo: jnp.ndarray         # () f64  touched_bays 의 min
    hi: jnp.ndarray         # () f64  touched_bays 의 max
    slots: jnp.ndarray      # (B,R) bool
    ok: jnp.ndarray         # () bool  전 blocker 가 칸을 찾았나
    viol: jnp.ndarray       # () int32
    mv_cont: jnp.ndarray    # (M,)
    mv_src: jnp.ndarray     # (M,3)
    mv_dst: jnp.ndarray     # (M,3)
    mv_kind: jnp.ndarray    # (M,)


def _plan_retrieve(stacks: StackArrays, conts: ContArrays, excluded0, spec: Spec7,
                   cur_bay, cur_row, bay_min, bay_max, is_ext, target_cont, g: Geom) -> PlanOut:
    B, R, T = stacks.grid.shape
    M = g.n_moves
    TR = _transfer_row(g)
    C = conts.c_bay.shape[0]
    height, top_size = stacks.height, stacks.top_size

    tc = jnp.asarray(target_cont, jnp.int32)
    tcc = jnp.clip(tc, 0, C - 1)
    # 대상이 야드에 없으면 v5 는 _dispatchable(503행) 이 미리 걸러 여기 못 온다 — 배열판은 ok=False
    valid = (tc >= 0) & (tc < C) & conts.c_alive[tcc] & (conts.c_bay[tcc] > 0)
    blk, n_block = blockers_above(stacks.grid, height, conts.c_bay, conts.c_row, conts.c_tier,
                                  tc, g)                              # 619행 (위에서부터)
    tb, tr, tt = conts.c_bay[tcc], conts.c_row[tcc], conts.c_tier[tcc]

    mv_cont, mv_src, mv_dst, mv_kind = _empty_moves(M)
    z = jnp.asarray(0.0, F)
    carry0 = _Carry(cur_bay=jnp.asarray(cur_bay, F), cur_row=jnp.asarray(cur_row, F),
                    excluded=excluded0, total=z, loaded=z, empty=z, rehandles=jnp.int32(0),
                    lo=jnp.asarray(cur_bay, F), hi=jnp.asarray(cur_bay, F),   # touched={cur_bay}
                    slots=jnp.zeros((B, R), bool), ok=valid, viol=jnp.int32(0),
                    mv_cont=mv_cont, mv_src=mv_src, mv_dst=mv_dst, mv_kind=mv_kind)

    def step(c: _Carry, x):
        i, b = x                                   # 단계 번호 · blocker 컨테이너 번호
        active = i < n_block
        bc = jnp.clip(b, 0, C - 1)
        sb, sr, st = conts.c_bay[bc], conts.c_row[bc], conts.c_tier[bc]   # 621행 src
        size = conts.c_size[bc]
        excl_k = c.excluded | onehot_cell(sb, sr, B, R)                    # 622행 exclude ∪ {원천}
        # 626행 — 기준점은 blocker **자기 좌표** (float(b.bay), float(b.row))
        found, db, dr = find_slot(height, top_size, excl_k, size, bay_min, bay_max,
                                  sb.astype(F), sr.astype(F), g)
        bi = jnp.clip(db - 1, 0, B - 1)
        ri = jnp.clip(dr - 1, 0, R - 1)
        h = height[bi, ri]
        dtier = h + 1                                                      # 631행
        # 634-636행 후조건 — excl_k 가 (exclude ∪ 원천) 이라 'dest in exclude'·'dest == 원천' 을 한 번에
        post_ok = (dtier <= T) & ((h == 0) | (top_size[bi, ri] == size)) & ~excl_k[bi, ri]
        src = jnp.stack([sb, sr, st]).astype(jnp.int32)
        dst = jnp.stack([db, dr, dtier]).astype(jnp.int32)
        mv = move_container(spec, c.cur_bay, c.cur_row, src, dst, g)      # 637행
        sb_f, db_f = sb.astype(F), db.astype(F)
        new = _Carry(
            cur_bay=mv.end_bay, cur_row=mv.end_row,                        # 643행
            excluded=c.excluded | onehot_cell(db, dr, B, R),               # 645행
            total=c.total + mv.dur,                                        # 640행
            loaded=c.loaded + mv.loaded_m, empty=c.empty + mv.empty_m,     # 641-642행
            rehandles=c.rehandles + 1,                                     # 644행
            lo=jnp.minimum(c.lo, jnp.minimum(sb_f, db_f)),                 # 646행 touched |= {b.bay, db}
            hi=jnp.maximum(c.hi, jnp.maximum(sb_f, db_f)),
            slots=c.slots | onehot_cell(sb, sr, B, R) | onehot_cell(db, dr, B, R),   # 647행
            ok=c.ok & found,                                               # 628-629행 None
            viol=c.viol | jnp.where(found & ~post_ok, V_PLAN_POSTCOND, 0).astype(jnp.int32),
            mv_cont=c.mv_cont.at[i].set(b), mv_src=c.mv_src.at[i].set(src),
            mv_dst=c.mv_dst.at[i].set(dst), mv_kind=c.mv_kind.at[i].set(jnp.int32(MV_REHANDLE)))
        c = jax.tree_util.tree_map(lambda a, o: jnp.where(active, a, o), new, c)
        return c, None

    xs = (jnp.arange(M - 1, dtype=jnp.int32), blk)
    c, _ = lax.scan(step, carry0, xs)

    # 648-663행 대상 반출 — dst = (target.bay, transfer_row, 1)
    src_t = jnp.stack([tb, tr, tt]).astype(jnp.int32)
    dst_t = jnp.stack([tb, jnp.int32(TR), jnp.int32(1)]).astype(jnp.int32)
    mv = move_container(spec, c.cur_bay, c.cur_row, src_t, dst_t, g)
    total = c.total + mv.dur                                               # 655행
    total = jnp.where(is_ext, total + spec.truck_pos_s, total)             # 656-657행
    loaded = c.loaded + mv.loaded_m
    empty = c.empty + mv.empty_m
    tb_f = tb.astype(F)
    lo = jnp.minimum(c.lo, tb_f)                                           # 662행
    hi = jnp.maximum(c.hi, tb_f)
    slots = c.slots | onehot_cell(tb, tr, B, R)                            # 663행
    j = jnp.clip(n_block, 0, M - 1)                                        # 대상 이동 칸 = n_block (≤ M−1)
    mv_cont = c.mv_cont.at[j].set(tc)
    mv_src = c.mv_src.at[j].set(src_t)
    mv_dst = c.mv_dst.at[j].set(dst_t)
    mv_kind = c.mv_kind.at[j].set(jnp.int32(MV_RETRIEVE))
    ok = c.ok
    lane = jnp.where(valid, lane_of_bay(tb, g), EMPTY_ID).astype(jnp.int32)   # 538행 대상 bay
    return PlanOut(
        ok=ok, kind=jnp.int32(PK_SERVE), lo=lo, hi=hi, dur=total,
        end_bay=mv.end_bay, end_row=mv.end_row, rehandles=c.rehandles,
        loaded_m=loaded, empty_m=empty, lane=lane, slots=slots & ok,
        n_moves=jnp.where(ok, n_block + 1, 0).astype(jnp.int32),
        mv_cont=mv_cont, mv_src=mv_src, mv_dst=mv_dst, mv_kind=mv_kind,
        viol=c.viol)


# ───────────────────────────────────────────────── 공개 함수
def plan_serve_core(stacks: StackArrays, conts: ContArrays, excluded0, spec: Spec7,
                    cur_bay, cur_row, bay_min, bay_max,
                    is_store, is_ext, target_cont, inbound_cont, inbound_size,
                    g: Geom) -> PlanOut:
    """낱개 배열로 받는 계획 함수 — `plan_serve` 가 world 에서 뽑아 여기로 넘긴다.

    excluded0 (B,R) bool = 예약 칸 ∪ extra (588행). 스칼라는 () 배열 또는 파이썬 값.
    STORE·RETRIEVE 두 경로를 **둘 다** 계산하고 is_store 로 고른다 — vmap 아래서는
    어차피 둘 다 계산되고(select), 단일 호출에서도 모양이 같아 단순하다.
    """
    is_store = jnp.asarray(is_store, bool)
    is_ext = jnp.asarray(is_ext, bool)
    cur_bay = jnp.asarray(cur_bay, F)
    cur_row = jnp.asarray(cur_row, F)
    excluded0 = jnp.asarray(excluded0, bool)
    a = _plan_store(stacks, excluded0, spec, cur_bay, cur_row, bay_min, bay_max,
                    is_ext, jnp.asarray(inbound_cont, jnp.int32), jnp.asarray(inbound_size, jnp.int32), g)
    b = _plan_retrieve(stacks, conts, excluded0, spec, cur_bay, cur_row, bay_min, bay_max,
                       is_ext, jnp.asarray(target_cont, jnp.int32), g)
    return jax.tree_util.tree_map(lambda x, y: jnp.where(is_store, x, y), a, b)


def plan_serve(world: BlockWorld, k, n, extra_excluded, g: Geom) -> PlanOut:
    """크레인 k 가 오더 n 을 SERVE 하는 계획 — v5 `_plan(crane_id, JobRef(SERVE), extra_exclude)`.

    excluded0 = reserved_slots(res) | extra_excluded (588행). start_s 는 엔진이 clock 으로 채운다.
    """
    cr, o = world.cranes, world.orders
    excluded0 = reserved_slots(world.res) | jnp.asarray(extra_excluded, bool)
    return plan_serve_core(world.stacks, world.conts, excluded0, crane_spec7(cr, k),
                           cr.bay[k], cr.row[k], cr.bay_min[k], cr.bay_max[k],
                           o.is_store[n], o.is_external[n], o.target_cont[n],
                           o.inbound_cont[n], o.inbound_size[n], g)


def plan_serve_all(world: BlockWorld, extra_excluded, g: Geom) -> PlanOut:
    """크레인 K × 오더 N 전부 — 각 열 앞에 (K,N). 명세 §8 의 P = vmap_k(vmap_n(plan_serve)).

    g 는 static 이라 partial 로 묶는다. jit 하려면 `jax.jit(plan_serve_all, static_argnames='g')`.
    """
    K, N = world.cranes.k, world.orders.n
    f = partial(plan_serve, g=g)
    over_n = jax.vmap(f, in_axes=(None, None, 0, None))
    over_kn = jax.vmap(over_n, in_axes=(None, 0, None, None))
    return over_kn(world, jnp.arange(K, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32),
                   jnp.asarray(extra_excluded, bool))
