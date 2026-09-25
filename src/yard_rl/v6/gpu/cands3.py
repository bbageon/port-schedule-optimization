"""조각 3 **확장 후보** — PRE_REHANDLE · REPOSITION · WAIT 를 후보 행렬에 더한다 ([[YR-327]] 조각 3 · key=cands3).

v5 정본 = `integrated/candidates.py` 의 `CandidateGenerator.generate` (LEGACY_DEFAULT: wait_mode=WAIT ·
safety_only=False · bound_repo=False · vessel_prep=False · block_pre_rehandle=False · k_max=12 ·
mandatory_wait_frac=0.8) 와 그것이 부르는 엔진 함수 (`_dispatchable`·`_jobref`·`_plan` 의 PRE/REPO 경로
552-576행·`interference_deadlock_corridors` 430-450행) 및 `_decision_cranes`(453-464행) 의 개방 술어.
조각 2 의 `dispatch.candidates`(SERVE 만) 는 서명을 그대로 두고, 여기 `candidates3` 이 **별도 함수**로
같은 재료(`escape.candidate_matrices`) 위에 세 종류를 더한다 — 통합자가 엔진 결정 국면에 바꿔 끼운다.

■ 네 종류가 언제 생기나 (generate 254-266행 · LEGACY_DEFAULT)
    크레인이 유휴·비양보(eligible) 가 아니면       → WAIT 하나뿐 (256-257행)
    SERVE        `_serve` 268-291행  — `_dispatchable` & `_jobref` 있음 & (계획 성립 | mandatory).
                 계획 실패 + mandatory 는 feasible=False(PLAN_FAILED) 로 **목록에 오른다** (prune 예산을 먹는다);
                 예약 거절(5-lock 코드 ≠ 0, 잡힌 오더의 DUP_JOB 포함) 도 목록에 오르되 feasible=False.
    PRE_REHANDLE `_pre_rehandle` 293-320행 — 게이트 `iter_pre_rehandle_jobs` (50-78행):
                 PRE_ADVICE & GATE_OUT & PLANNED & 대상 실재·가용 & 담당 구간 & blocker>0 & 재조작 여유 &
                 provided_eta 있음 & eta−now ≤ horizon (**eta 가 과거여도 통과** — 연착 신호).
                 계획 = `_plan(PRE)` 618-647행: blocker 만 치우고 대상은 남긴다 (트럭 위치잡기 없음, 655-657행은 SERVE 만).
                 계획 실패면 목록에 안 오른다(298-299행). token = job_id · lane = 대상 bay 의 레인.
    REPOSITION   `_reposition` 371-417행 — 목표 bay 집합 = `_future_target_bays`(ETA 주도 외부트럭, PRE_ADVICE 한정 ∪
                 release 주도 내부작업, **정보수준 무관**) ∪ `_escape_bays`(교착 술어 참일 때만, 334-363행).
                 set → sorted (407행). 탈출 목표가 아니면 |tb−pos| ≤ 1 은 버린다(412행). 계획 = 563-576행 (빈 주행).
                 token·lane·slots 없음 → 거절은 CRANE_INTERFERENCE 만 가능. score = −1000 + tb (417행).
    WAIT         항상 마지막 하나 (265행). LEGACY 는 defer 없음 (`defer_wait` 가 DEFER 두 모드의 재료를 따로 준다).
    결정 개방    `_decision_cranes`: eligible & (SERVE 후보 있음 | (armed & eta_opportunity)) — eta_opportunity 는
                 PRE 게이트만(계획 없이) 본다 (154-176행).

■ 후보 행렬 (K, C) · C = N + R + 1  (`flat_view`)
    [0, N)      SERVE 또는 PRE_REHANDLE — 한 오더는 둘 중 하나만 (SERVE 는 WAITING/RELEASED, PRE 는 PLANNED)
    [N, N+R)    REPOSITION — 목표 bay 를 **정렬·인접중복 제거한 고정 길이 R** (빈칸 +inf). R = B + 2·E
                (정수 목표는 bay 1..B 의 (B,) 마스크로 정확히 담기고, 탈출 목표는 통로마다 lo−gap·hi+gap 둘 → E 통로)
    [N+R]       WAIT
  `raw` = generate 의 원시 목록(prune 전) · `feasible` = raw & 거절 코드 0 · `mandatory`·`score`·계획 열.

■ prune (`prune`, 505-518행 `_prune` + 261행 `sorted(order_key)`)
    budget = k_max − 1. mandatory 는 전량, 나머지는 (−score, kind_rank, job_id 문자열, bay) 오름차순으로 budget−#mand 까지.
    ★REPO 의 job_id 는 "REPO:<crane>:<int(tb)>" 라 **정수의 문자열 순서**다 ("10" < "2") — 정적 표 `_str_rank`.
    남은 것을 order_key 로 다시 정렬한 위치가 candidate_id, WAIT 은 그 뒤 (= 남은 수).
    −score 의 −0.0 은 +0.0 으로 정규화한다 (파이썬 sort 는 둘을 같게 보지만 lax.sort 의 전순서는 −0 < +0).

■ ★같은 답을 내기 위한 규칙
  · 실수는 전부 float64. score 의 `0.1·dur + 50·rehandles` 는 두 곱을 `mul_exact` 로 실체화한 뒤 더한다 (FMA 방어).
  · REPO 소요 = `travel.estimate_reach_s` (v5 565-567행과 같은 식·같은 순서: dist/gantry + t_dist/trolley, 나눗셈은 div_exact).
  · PRE 계획의 blocker scan 은 `plan._plan_retrieve` 의 scan 과 **같은 본문**이다 (대상 이동만 없다) — 두 벌이 된 것은
    open_issues 에 적었다 (plan.py 가 scan 을 공개하면 여기서 부른다).
  · 예외 자리는 위반 비트 (V_PLAN_POSTCOND) — 값은 PlanOut.viol 에 남긴다.

■ 서명 (통합 단계가 그대로 쓴다)
    candidates3(world, g, *, horizon_s, pre_advice: bool, n_esc: int | None = None) -> CandOut3
        g·pre_advice·n_esc 는 static, horizon_s 는 traced 스칼라 (v5 profile.decision_horizon_s — Geom 에 없다).
    flat_view(out) -> FlatCands (K, N+R+1)
    prune(flat, k_max=12) -> PruneOut (keep·candidate_id·n_kept)
    defer_wait(world, g, pre_advice, *, t_max=600.0) -> DeferOut (YR-147 DEFER 재료 · LEGACY 는 안 쓴다)
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .escape import CandMats, blocked_corridors, candidate_matrices, deadlock_predicate
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .exact import mul_exact
from .geom import Geom
from .plan import PlanOut, _empty_moves, _transfer_row, crane_spec7, lane_of_bay, onehot_cell
from .reserve import OK, corridor_overlaps, reject_code, reserved_slots
from .stack_ops import blockers_above, find_slot, rehandle_capacity_ok
from .state import (FL_GATE_OUT, JS_DONE, JS_PLANNED, MV_REHANDLE, PK_PRE_REHANDLE, PK_REPOSITION,
                    PK_SERVE, PK_WAIT, V_PLAN_POSTCOND, BlockWorld, ContArrays, CraneArrays,
                    StackArrays)
from .travel import Spec7, estimate_reach_s, move_container

__all__ = ["CandOut3", "FlatCands", "PruneOut", "DeferOut", "K_MAX", "MANDATORY_WAIT_FRAC", "DEFER_T_MAX",
           "REPO_SCORE_BASE", "REPO_MIN_MOVE", "BUSY_NO_ORDER", "candidates3", "plan_pre_core", "plan_pre", "plan_repo",
           "pre_gate", "future_bays", "escape_targets", "repo_targets", "flat_view", "prune", "defer_wait",
           "n_repo_cols"]

F = TIME_DTYPE
#: v5 CandidateGenerator 기본값 (candidates.py:218-220)
K_MAX = 12
MANDATORY_WAIT_FRAC = 0.8
#: YR-147 DEFER 만료 (candidates.py:117)
DEFER_T_MAX = 600.0
#: REPO score = REPO_SCORE_BASE + tb (417행) · 일반 REPO 의 미세이동 하한 (412행)
REPO_SCORE_BASE = -1000.0
REPO_MIN_MOVE = 1.0
#: 탈출 목표의 "제자리" 판정 (357행)
_ESC_EPS = 1e-9
#: ★REPOSITION 실행 중 크레인의 `cranes.assigned` 값 **제안** — v5 는 `assigned_job = "REPO:<cid>:<bay>"` (699행) 라
#: 오더 번호가 없다. 엔진의 유휴 판정은 전부 `assigned < 0` 이므로 0 이상이면서 오더 번호가 아닌 값이면 된다
#: (`from_block_world` 의 `tables.job()` 은 범위 밖을 None 으로 돌려준다). 확정은 통합자 몫 — 시험이 이 값을 쓴다.
BUSY_NO_ORDER = 1 << 30
#: order_key 의 kind_rank (candidates.py:24-25)
_KIND_RANK = {PK_SERVE: 0, PK_PRE_REHANDLE: 1, PK_REPOSITION: 2, PK_WAIT: 3}


# ───────────────────────────────────────────────── 크기
def n_repo_cols(g: Geom, n_orders: int, n_esc: int | None = None) -> int:
    """R = B + 2·E — 정수 목표 (B,) + 탈출 목표 2E (E 기본 = N: 막힌 오더마다 통로 하나, 449행)."""
    E = int(n_orders) if n_esc is None else int(n_esc)
    return int(g.bay_count) + 2 * E


def _str_rank_table(B: int) -> jnp.ndarray:
    """int(tb) ∈ [0, B] 의 **문자열 정렬 순위** — "REPO:<cid>:<int(tb)>" 의 job_id 순서 (order_key 521행)."""
    order = sorted(range(B + 1), key=str)
    rank = [0] * (B + 1)
    for pos, i in enumerate(order):
        rank[i] = pos
    return jnp.asarray(rank, jnp.int32)


# ───────────────────────────────────────────────── PRE_REHANDLE 계획 (engine.py:618-647, 665-670)
class _PreCarry(NamedTuple):
    """blocker scan 의 carry — `plan._Carry` 와 같은 칸 (593-600행 + 641-647행)."""

    cur_bay: jnp.ndarray
    cur_row: jnp.ndarray
    excluded: jnp.ndarray
    total: jnp.ndarray
    loaded: jnp.ndarray
    empty: jnp.ndarray
    rehandles: jnp.ndarray
    lo: jnp.ndarray
    hi: jnp.ndarray
    slots: jnp.ndarray
    ok: jnp.ndarray
    viol: jnp.ndarray
    mv_cont: jnp.ndarray
    mv_src: jnp.ndarray
    mv_dst: jnp.ndarray
    mv_kind: jnp.ndarray


def plan_pre_core(stacks: StackArrays, conts: ContArrays, excluded0, spec: Spec7,
                  cur_bay, cur_row, bay_min, bay_max, target_cont, g: Geom) -> PlanOut:
    """PRE_REHANDLE 계획 — `_plan(ref.kind=PRE_REHANDLE)`: blocker 를 위에서부터 치우고 **대상은 남긴다**.

    `plan._plan_retrieve` 의 blocker scan 과 같은 본문 (618-647행) 이고, 648-663행(대상 반출)이 없다.
    duration 에 트럭 위치잡기가 **없다** (656-657행은 SERVE 분기 안). end = 마지막 blocker 의 목적지.
    lane 은 `_jobref` 가 아니라 `_pre_rehandle` 이 준다 — `_lane_for(c.bay)` (대상 bay, 305행).
    """
    B, R, T = stacks.grid.shape
    M = g.n_moves
    C = conts.c_bay.shape[0]
    height, top_size = stacks.height, stacks.top_size
    cur_bay = jnp.asarray(cur_bay, F)
    cur_row = jnp.asarray(cur_row, F)
    excluded0 = jnp.asarray(excluded0, bool)

    tc = jnp.asarray(target_cont, jnp.int32)
    tcc = jnp.clip(tc, 0, C - 1)
    valid = (tc >= 0) & (tc < C) & conts.c_alive[tcc] & (conts.c_bay[tcc] > 0)
    blk, n_block = blockers_above(stacks.grid, height, conts.c_bay, conts.c_row, conts.c_tier, tc, g)   # 619행
    tb = conts.c_bay[tcc]

    mv_cont, mv_src, mv_dst, mv_kind = _empty_moves(M)
    z = jnp.asarray(0.0, F)
    carry0 = _PreCarry(cur_bay=cur_bay, cur_row=cur_row, excluded=excluded0, total=z, loaded=z, empty=z,
                       rehandles=jnp.int32(0), lo=cur_bay, hi=cur_bay,            # 593행 touched={cur_bay}
                       slots=jnp.zeros((B, R), bool), ok=valid, viol=jnp.int32(0),
                       mv_cont=mv_cont, mv_src=mv_src, mv_dst=mv_dst, mv_kind=mv_kind)

    def step(c: _PreCarry, x):
        i, b = x
        active = i < n_block
        bc = jnp.clip(b, 0, C - 1)
        sb, sr, st = conts.c_bay[bc], conts.c_row[bc], conts.c_tier[bc]           # 621행 src
        size = conts.c_size[bc]
        excl_k = c.excluded | onehot_cell(sb, sr, B, R)                           # 622행 exclude ∪ {원천}
        found, db, dr = find_slot(height, top_size, excl_k, size, bay_min, bay_max,
                                  sb.astype(F), sr.astype(F), g)                  # 626행 기준점 = blocker 좌표
        bi = jnp.clip(db - 1, 0, B - 1)
        ri = jnp.clip(dr - 1, 0, R - 1)
        h = height[bi, ri]
        dtier = h + 1                                                             # 631행
        post_ok = (dtier <= T) & ((h == 0) | (top_size[bi, ri] == size)) & ~excl_k[bi, ri]   # 634-636행
        src = jnp.stack([sb, sr, st]).astype(jnp.int32)
        dst = jnp.stack([db, dr, dtier]).astype(jnp.int32)
        mv = move_container(spec, c.cur_bay, c.cur_row, src, dst, g)             # 637행
        sb_f, db_f = sb.astype(F), db.astype(F)
        new = _PreCarry(
            cur_bay=mv.end_bay, cur_row=mv.end_row,                               # 643행
            excluded=c.excluded | onehot_cell(db, dr, B, R),                      # 645행
            total=c.total + mv.dur,                                               # 640행
            loaded=c.loaded + mv.loaded_m, empty=c.empty + mv.empty_m,            # 641-642행
            rehandles=c.rehandles + 1,                                            # 644행
            lo=jnp.minimum(c.lo, jnp.minimum(sb_f, db_f)),                        # 646행
            hi=jnp.maximum(c.hi, jnp.maximum(sb_f, db_f)),
            slots=c.slots | onehot_cell(sb, sr, B, R) | onehot_cell(db, dr, B, R),   # 647행
            ok=c.ok & found,                                                      # 628-629행 None
            viol=c.viol | jnp.where(found & ~post_ok, V_PLAN_POSTCOND, 0).astype(jnp.int32),
            mv_cont=c.mv_cont.at[i].set(b), mv_src=c.mv_src.at[i].set(src),
            mv_dst=c.mv_dst.at[i].set(dst), mv_kind=c.mv_kind.at[i].set(jnp.int32(MV_REHANDLE)))
        c = jax.tree_util.tree_map(lambda a, o: jnp.where(active, a, o), new, c)
        return c, None

    xs = (jnp.arange(M - 1, dtype=jnp.int32), blk)
    c, _ = lax.scan(step, carry0, xs)
    ok = c.ok
    lane = jnp.where(valid, lane_of_bay(tb, g), EMPTY_ID).astype(jnp.int32)      # 305행 _lane_for(c.bay)
    return PlanOut(
        ok=ok, kind=jnp.int32(PK_PRE_REHANDLE), lo=c.lo, hi=c.hi, dur=c.total,   # 665행 min/max(touched)
        end_bay=c.cur_bay, end_row=c.cur_row, rehandles=c.rehandles,
        loaded_m=c.loaded, empty_m=c.empty, lane=lane, slots=c.slots & ok,
        n_moves=jnp.where(ok, n_block, 0).astype(jnp.int32),
        mv_cont=c.mv_cont, mv_src=c.mv_src, mv_dst=c.mv_dst, mv_kind=c.mv_kind,
        viol=c.viol)


def plan_pre(world: BlockWorld, k, n, g: Geom) -> PlanOut:
    """크레인 k 가 오더 n 의 대상 위 blocker 를 선처리하는 계획 — excluded = 현재 예약 칸 (588행)."""
    cr, o = world.cranes, world.orders
    return plan_pre_core(world.stacks, world.conts, reserved_slots(world.res), crane_spec7(cr, k),
                         cr.bay[k], cr.row[k], cr.bay_min[k], cr.bay_max[k], o.target_cont[n], g)


# ───────────────────────────────────────────────── REPOSITION 계획 (engine.py:563-576)
def plan_repo(cr: CraneArrays, k, tb, g: Geom) -> PlanOut:
    """빈 주행 계획 — `_plan(ref.kind=REPOSITION)`. tb 는 float64 목표 bay (이미 clamp 된 값이어도 다시 clamp 한다, 566행).

    dur = dist/gantry + t_dist/trolley (567행) = `travel.estimate_reach_s` (같은 식·같은 순서).
    corridor = (min(cur, tb), max(cur, tb)) · slots 없음 · lane 없음 · end = (tb, transfer_row) · empty_m = dist.
    """
    B, R = int(g.bay_count), int(g.row_count)
    M = g.n_moves
    spec = crane_spec7(cr, k)
    cur_bay, cur_row = cr.bay[k], cr.row[k]
    tb = jnp.asarray(tb, F)
    tb = jnp.minimum(jnp.maximum(tb, cr.bay_min[k].astype(F)), cr.bay_max[k].astype(F))   # 566행 clamp
    tr = jnp.asarray(float(g.transfer_row), F)
    dist = jnp.abs(cur_bay - tb) * jnp.asarray(g.bay_len, F)                     # 564행 gantry_m
    dur = estimate_reach_s(spec, g, cur_bay, cur_row, tb, tr)                    # 565-567행
    mv_cont, mv_src, mv_dst, mv_kind = _empty_moves(M)
    z = jnp.asarray(0.0, F)
    return PlanOut(
        ok=jnp.ones((), bool), kind=jnp.int32(PK_REPOSITION),
        lo=jnp.minimum(cur_bay, tb), hi=jnp.maximum(cur_bay, tb),               # 570행
        dur=dur, end_bay=tb, end_row=tr, rehandles=jnp.int32(0),                 # 572-573행
        loaded_m=z, empty_m=dist, lane=jnp.int32(EMPTY_ID),                      # 574행
        slots=jnp.zeros((B, R), bool), n_moves=jnp.int32(0),
        mv_cont=mv_cont, mv_src=mv_src, mv_dst=mv_dst, mv_kind=mv_kind, viol=jnp.int32(0))


# ───────────────────────────────────────────────── 게이트·목표 bay
def pre_gate(world: BlockWorld, g: Geom, horizon_s, pre_advice: bool) -> jnp.ndarray:
    """`iter_pre_rehandle_jobs` (candidates.py:50-78) 의 (K,N) 마스크 — plan 전 게이트 = `eta_opportunity` 의 재료.

    PRE_ADVICE & GATE_OUT & PLANNED & 대상 실재·가용 & 담당 구간 & blocker>0 & 재조작 여유 & eta 있음 & eta−now ≤ horizon.
    ★eta ≤ now 도 통과한다 (연착) — ETA 주도 REPO 의 `eta > now` 와 다르다.
    """
    o, cr, st, ct = world.orders, world.cranes, world.stacks, world.conts
    K, N = cr.k, o.n
    C = ct.c
    if not pre_advice:
        return jnp.zeros((K, N), bool)
    tcc = jnp.clip(o.target_cont, 0, C - 1)
    has_t = (o.target_cont >= 0) & (o.block >= 0)
    t_ok = ct.c_alive[tcc] & ct.c_avail[tcc]                                     # 65-67행
    tb, trow = ct.c_bay[tcc], ct.c_row[tcc]
    blk_fn = jax.vmap(partial(blockers_above, g=g), in_axes=(None, None, None, None, None, 0))
    _, n_block = blk_fn(st.grid, st.height, ct.c_bay, ct.c_row, ct.c_tier, o.target_cont)   # 70행
    eta = o.provided_eta_s
    hz = jnp.asarray(horizon_s, F)
    eta_ok = (eta < EMPTY_TIME) & ((eta - world.clock) <= hz)                    # 75-77행
    base = (has_t & (o.flow == FL_GATE_OUT) & (o.status == JS_PLANNED) & t_ok & (n_block > 0) & eta_ok)   # 62-64, 70행

    def per_kn(bmin, bmax, tb_, tr_, nb_):
        in_range = (bmin <= tb_) & (tb_ <= bmax)                                 # 68-69행
        cap = rehandle_capacity_ok(st.height, st.top_size, tb_, tr_, nb_, bmin, bmax, g)   # 72-73행
        return in_range & cap

    over_n = jax.vmap(per_kn, in_axes=(None, None, 0, 0, 0))
    over_kn = jax.vmap(over_n, in_axes=(0, 0, None, None, None))
    return base[None, :] & over_kn(cr.bay_min, cr.bay_max, tb, trow, n_block)


def future_bays(world: BlockWorld, g: Geom, horizon_s, pre_advice: bool):
    """`_future_target_bays` (candidates.py:540-554) 의 재료 — (K,N) 목표 bay (f64, clamp 뒤) 와 마스크.

    ETA 주도 (`iter_eta_reposition_jobs` 121-141행, PRE_ADVICE 한정): 외부트럭 & PLANNED & eta 있음 & eta > now & eta−now ≤ horizon
    release 주도 (546-553행, 정보수준 무관): 내부작업 & PLANNED(=DONE 아님·release_time 을 eta 로) & release > now & release−now ≤ horizon
    bay (`_future_bay_of` 41-49행): 대상 컨테이너가 야드에 있으면 그 bay, 아니면 STORE 면 크레인 현위치 기준 find_slot(제외 없음) 의 bay.
    반환 (mask_eta (K,N), mask_rel (K,N), bay (K,N) f64 clamp 된 값).
    """
    o, cr, st, ct = world.orders, world.cranes, world.stacks, world.conts
    K, N = cr.k, o.n
    B, R, _ = st.shape
    C = ct.c
    clock = world.clock
    hz = jnp.asarray(horizon_s, F)
    tcc = jnp.clip(o.target_cont, 0, C - 1)
    t_in_yard = (o.target_cont >= 0) & ct.c_alive[tcc]                           # 42행 `in sim.stacks.containers`
    t_bay = ct.c_bay[tcc].astype(F)
    eta = o.provided_eta_s
    m_eta = (o.block >= 0) & o.is_external & (o.status == JS_PLANNED) & (eta < EMPTY_TIME) \
        & (eta > clock) & ((eta - clock) <= hz)                                  # 128-134행
    if not pre_advice:
        m_eta = jnp.zeros_like(m_eta)
    rel = o.release_s
    m_rel = (o.block >= 0) & ~o.is_external & (o.status != JS_DONE) & (o.status == JS_PLANNED) \
        & (rel < EMPTY_TIME) & (rel > clock) & ((rel - clock) <= hz)             # 546-550행
    zero_ex = jnp.zeros((B, R), bool)

    def per_kn(bmin, bmax, kbay, krow, is_store_, size_, t_in_, t_bay_):
        found, sb, _ = find_slot(st.height, st.top_size, zero_ex, size_, bmin, bmax, kbay, krow, g)   # 45-47행
        store_ok = is_store_ & (size_ >= 0) & found
        has = t_in_ | store_ok
        bay = jnp.where(t_in_, t_bay_, sb.astype(F))
        bay = jnp.minimum(jnp.maximum(bay, bmin.astype(F)), bmax.astype(F))      # 141, 553행 clamp
        return has, bay

    over_n = jax.vmap(per_kn, in_axes=(None, None, None, None, 0, 0, 0, 0))
    over_kn = jax.vmap(over_n, in_axes=(0, 0, 0, 0, None, None, None, None))
    has, bay = over_kn(cr.bay_min, cr.bay_max, cr.bay, cr.row, o.is_store, o.inbound_size, t_in_yard, t_bay)
    return m_eta[None, :] & has, m_rel[None, :] & has, bay


def escape_targets(world: BlockWorld, g: Geom, m: CandMats, deadlock, blocked, n_esc: int | None = None):
    """`_escape_bays` (candidates.py:334-363) — (K, 2E) 목표 bay 와 마스크. E 통로 = `blocked_corridors` 의 앞 E 오더.

    술어가 거짓이면 전부 빈칸 (336-337행). 통로마다 lo−gap · hi+gap 을 clamp 하고, 이 크레인의 정지 위치가 그 통로를
    가릴 때만(352-353행) · 제자리가 아닐 때만(357-358행) · 물러난 자리가 통로를 안 가릴 때만(359-360행) 목표가 된다.
    반환 (bay (K,2E) f64, valid (K,2E) bool, overflow () bool — E 보다 뒤에도 막힌 오더가 있었다).
    """
    K, N = world.k, world.n
    cr = world.cranes
    E = N if n_esc is None else int(n_esc)
    if E > N or E < 0:
        raise ValueError(f"n_esc={E} 는 0..N({N}) 이어야 한다 — 통로는 막힌 오더마다 하나라 N 을 넘을 수 없다")
    lo, hi, has = blocked_corridors(m, blocked)                                  # 449행 (오더 축)
    overflow = (deadlock & jnp.any(has[E:])) if E < N else jnp.zeros((), bool)   # 술어 참일 때만 뜻이 있다
    lo, hi, has = lo[:E], hi[:E], has[:E] & deadlock
    gap = jnp.asarray(g.gap, F)

    def per_k(pos, bmin, bmax):
        block_it = corridor_overlaps(pos, pos, lo, hi, gap)                      # 352행 Corridor(pos,pos).overlaps
        cand = jnp.stack([lo - gap, hi + gap], axis=1)                           # 354행 (E,2)
        t = jnp.minimum(jnp.maximum(cand, bmin.astype(F)), bmax.astype(F))       # 355행 clamp
        not_here = jnp.abs(t - pos) >= _ESC_EPS                                  # 356-357행
        frees = ~corridor_overlaps(t, t, lo[:, None], hi[:, None], gap)          # 359행
        ok = (has & block_it)[:, None] & not_here & frees
        return t.reshape(-1), ok.reshape(-1)

    bay, valid = jax.vmap(per_k)(cr.bay, cr.bay_min, cr.bay_max)
    return bay, valid, overflow


def repo_targets(world: BlockWorld, g: Geom, m: CandMats, deadlock, blocked, horizon_s, pre_advice: bool,
                 n_esc: int | None = None):
    """REPO 목표 bay 집합 — `set(_future_target_bays) | escape_targets` → sorted (407행) 를 고정 길이 (K,R) 로.

    정수 목표(ETA·release 주도)는 bay 1..B 마스크로, 탈출 목표는 (2E,) 열로 모아 정렬한 뒤 인접 중복을 지운다
    (파이썬 set 의 float `==` 와 같다). 반환 (bay (K,R) f64 오름차순·빈칸 +inf, valid (K,R), is_escape (K,R), overflow).
    """
    K, N = world.k, world.n
    B = int(g.bay_count)
    m_eta, m_rel, fb = future_bays(world, g, horizon_s, pre_advice)
    fmask = m_eta | m_rel                                                        # 543행 합집합
    bays_f = jnp.arange(1, B + 1, dtype=jnp.int32).astype(F)
    fb_i = jnp.clip(fb.astype(jnp.int32), 1, B)                                 # 정수 목표 (clamp 뒤라 1..B)
    int_mask = jax.vmap(lambda mk, bi: jnp.any(mk[:, None] & (bi[:, None] == jnp.arange(1, B + 1)[None, :]), axis=0))(fmask, fb_i)
    e_bay, e_valid, overflow = escape_targets(world, g, m, deadlock, blocked, n_esc)
    vals = jnp.concatenate([jnp.where(int_mask, bays_f[None, :], EMPTY_TIME),
                            jnp.where(e_valid, e_bay, EMPTY_TIME)], axis=1)     # (K, B+2E)
    srt = jnp.sort(vals, axis=1)                                                 # 407행 sorted
    R = srt.shape[1]
    prev = jnp.concatenate([jnp.full((K, 1), -jnp.inf, F), srt[:, :-1]], axis=1)
    valid = (srt < EMPTY_TIME) & (srt != prev)                                   # set: 인접 중복 제거
    is_esc = valid & jax.vmap(lambda v, eb, ev: jnp.any((v[:, None] == eb[None, :]) & ev[None, :], axis=1))(srt, e_bay, e_valid)
    return srt, valid, is_esc, overflow


# ───────────────────────────────────────────────── score (candidates.py:493-503) · mandatory (485-487)
def _score(cum_term, cum_on, is_ves, deadline, clock, plan_ok, dur, rehandles, g: Geom):
    """`_score(sim, ref, plan, now, cum)`: s = 0; s += cum(있으면); s += max(0, sla − (deadline − now)) (본선연계·마감);
    s −= 0.1·dur + 50·rehandles (계획 있으면). 곱 둘은 mul_exact — 파이썬처럼 각각 반올림 뒤 더한다."""
    sla = jnp.asarray(g.sla_s, F)
    z = jnp.asarray(0.0, F)
    s = jnp.where(cum_on, z + cum_term, z)                                       # 496-497행
    has_dl = is_ves & (deadline < EMPTY_TIME)
    s = jnp.where(has_dl, s + jnp.maximum(0.0, sla - (deadline - clock)), s)     # 499-500행
    t = mul_exact(jnp.asarray(0.1, F), dur) + mul_exact(jnp.asarray(50.0, F), rehandles.astype(F))   # 502행
    return jnp.where(plan_ok, s - t, s)


# ───────────────────────────────────────────────── 본체
class CandOut3(NamedTuple):
    """결정 시작 시점의 확장 후보 — `CandidateGenerator.generate` 의 배열판 (머리말)."""

    m: CandMats               # SERVE 재료 (escape.candidate_matrices): P·disp·taken·code·feasible·idle·eligible·cand
    eligible: jnp.ndarray     # (K,)  bool  idle & ~yielded — 아니면 WAIT 뿐 (256행)
    # ── SERVE ──
    mandatory: jnp.ndarray    # (N,)  bool  외부트럭 & cum ≥ 0.8·sla (485-487행)
    serve_raw: jnp.ndarray    # (K,N) bool  eligible & disp & jobref 있음 & (계획 성립 | mandatory) — generate 목록
    serve: jnp.ndarray        # (K,N) bool  serve_raw & 거절 코드 0 (= m.cand)
    serve_score: jnp.ndarray  # (K,N) f64
    # ── PRE_REHANDLE ──
    pre_gate: jnp.ndarray     # (K,N) bool  iter_pre_rehandle_jobs (eta_opportunity 의 재료)
    P_pre: PlanOut            # (K,N) 계획 (kind PK_PRE_REHANDLE)
    pre_code: jnp.ndarray     # (K,N) int32 reject_reason
    pre_raw: jnp.ndarray      # (K,N) bool  eligible & gate & 계획 성립
    pre: jnp.ndarray          # (K,N) bool  pre_raw & 코드 0
    pre_score: jnp.ndarray    # (K,N) f64
    # ── REPOSITION ──
    repo_bay: jnp.ndarray     # (K,R) f64   정렬 목표 bay (+inf 빈칸)
    repo_valid: jnp.ndarray   # (K,R) bool  집합 원소
    repo_escape: jnp.ndarray  # (K,R) bool  탈출 목표
    P_repo: PlanOut           # (K,R) 계획 (kind PK_REPOSITION)
    repo_code: jnp.ndarray    # (K,R) int32
    repo_raw: jnp.ndarray     # (K,R) bool  eligible & valid & (탈출 | |tb−pos| > 1)
    repo: jnp.ndarray         # (K,R) bool  repo_raw & 코드 0
    repo_score: jnp.ndarray   # (K,R) f64   −1000 + tb
    esc_overflow: jnp.ndarray # ()    bool  E 칸보다 막힌 오더가 많았다 (n_esc < N 일 때만 가능)
    # ── 결정 개방·교착 ──
    eta_opp: jnp.ndarray      # (K,)  bool  any_n pre_gate — `eta_opportunity`
    open: jnp.ndarray         # (K,)  bool  eligible & (any serve | (armed & eta_opp)) — `_decision_cranes`
    deadlock: jnp.ndarray     # ()    bool
    blocked: jnp.ndarray      # (K,N) bool


def candidates3(world: BlockWorld, g: Geom, *, horizon_s, pre_advice: bool, n_esc: int | None = None) -> CandOut3:
    """확장 후보 한 번 — 머리말. g·pre_advice·n_esc 는 static (jit 하려면 `static_argnames=('g','pre_advice','n_esc')`)."""
    K, N = world.k, world.n
    B, R_, _ = world.stacks.shape
    o, cr = world.orders, world.cranes
    clock = world.clock
    gap = jnp.asarray(g.gap, F)
    m = candidate_matrices(world, g)
    deadlock, blocked = deadlock_predicate(world, m)
    eligible = m.eligible
    n_idx = jnp.arange(N, dtype=jnp.int32)

    # ── SERVE (268-291행) ──
    arrived = o.is_external & (o.block_in_s < EMPTY_TIME) & (o.block_in_s <= clock)
    cum = jnp.where(arrived, clock - o.block_in_s, 0.0)                          # engine.py:258-265 cum_wait
    thr = MANDATORY_WAIT_FRAC * float(g.sla_s)                                    # 파이썬 곱 — v5 486행과 같은 값
    mandatory = o.is_external & (cum >= thr)                                     # 485-487행 (cum 은 외부트럭만)
    has_ref = (o.target_cont >= 0) | o.is_store                                  # _jobref None 조건 (537-546행)
    serve_raw = eligible[:, None] & m.disp & has_ref[None, :] & (m.P.ok | mandatory[None, :])
    serve = m.cand
    serve_score = _score(cum[None, :], o.is_external[None, :], o.is_vessel[None, :], o.deadline_s[None, :],
                         clock, m.P.ok, m.P.dur, m.P.rehandles, g)

    # ── PRE_REHANDLE (293-320행) ──
    gate = pre_gate(world, g, horizon_s, pre_advice)
    f_pre = jax.vmap(partial(plan_pre, g=g), in_axes=(None, None, 0))
    P_pre = jax.vmap(f_pre, in_axes=(None, 0, None))(world, jnp.arange(K, dtype=jnp.int32), n_idx)
    code_fn = jax.vmap(reject_code, in_axes=(None, None, 0, 0, 0, 0, 0, None))
    pre_code = jax.vmap(code_fn, in_axes=(None, 0, None, 0, 0, 0, 0, None))(
        world.res, jnp.arange(K, dtype=jnp.int32), n_idx, P_pre.lo, P_pre.hi, P_pre.lane, P_pre.slots, gap)
    pre_raw = eligible[:, None] & gate & P_pre.ok
    pre = pre_raw & (pre_code == OK)
    pre_score = _score(jnp.zeros((K, N), F), jnp.zeros((K, N), bool), o.is_vessel[None, :], o.deadline_s[None, :],
                       clock, P_pre.ok, P_pre.dur, P_pre.rehandles, g)          # 318행 cum=None

    # ── REPOSITION (371-417행) ──
    rb, rvalid, resc, esc_overflow = repo_targets(world, g, m, deadlock, blocked, horizon_s, pre_advice, n_esc)
    f_repo = jax.vmap(partial(plan_repo, g=g), in_axes=(None, None, 0))
    P_repo = jax.vmap(f_repo, in_axes=(None, 0, 0))(cr, jnp.arange(K, dtype=jnp.int32), rb)
    zero_slots = jnp.zeros((B, R_), bool)
    rcode_fn = jax.vmap(reject_code, in_axes=(None, None, None, 0, 0, None, None, None))
    repo_code = jax.vmap(rcode_fn, in_axes=(None, 0, None, 0, 0, None, None, None))(
        world.res, jnp.arange(K, dtype=jnp.int32), jnp.int32(EMPTY_ID), P_repo.lo, P_repo.hi,
        jnp.int32(EMPTY_ID), zero_slots, gap)
    far = jnp.abs(rb - cr.bay[:, None]) > REPO_MIN_MOVE                          # 412행
    repo_raw = eligible[:, None] & rvalid & (resc | far) & P_repo.ok
    repo = repo_raw & (repo_code == OK)
    repo_score = jnp.asarray(REPO_SCORE_BASE, F) + rb                           # 417행

    # ── 결정 개방 (453-464행) ──
    eta_opp = jnp.any(gate, axis=1)
    open_ = eligible & (jnp.any(serve, axis=1) | (world.wake.eta_armed & eta_opp))
    return CandOut3(m=m, eligible=eligible, mandatory=mandatory, serve_raw=serve_raw, serve=serve,
                    serve_score=serve_score, pre_gate=gate, P_pre=P_pre, pre_code=pre_code, pre_raw=pre_raw,
                    pre=pre, pre_score=pre_score, repo_bay=rb, repo_valid=rvalid, repo_escape=resc,
                    P_repo=P_repo, repo_code=repo_code, repo_raw=repo_raw, repo=repo, repo_score=repo_score,
                    esc_overflow=esc_overflow, eta_opp=eta_opp, open=open_, deadlock=deadlock, blocked=blocked)


# ───────────────────────────────────────────────── 평면 행렬 (K, N+R+1)
class FlatCands(NamedTuple):
    """후보 행렬 한 장 — 열 [0,N) SERVE/PRE · [N,N+R) REPO · [N+R] WAIT (머리말)."""

    raw: jnp.ndarray        # (K,C) bool  generate 원시 목록 (prune 전; WAIT 열은 항상 True)
    feasible: jnp.ndarray   # (K,C) bool  raw & 거절 코드 0 (WAIT True)
    mandatory: jnp.ndarray  # (K,C) bool
    kind: jnp.ndarray       # (K,C) int32 PK_* (raw 아니면 -1; WAIT 열은 PK_WAIT)
    job: jnp.ndarray        # (K,C) int32 SERVE/PRE 의 오더 번호, 그 외 -1
    bay: jnp.ndarray        # (K,C) f64   REPO 목표 bay, 그 외 NaN
    score: jnp.ndarray      # (K,C) f64   WAIT −inf
    code: jnp.ndarray       # (K,C) int32 거절 코드 (계획 없는 mandatory SERVE 는 -1 = PLAN_FAILED)
    plan_ok: jnp.ndarray    # (K,C) bool
    dur: jnp.ndarray        # (K,C) f64
    rehandles: jnp.ndarray  # (K,C) int32
    lo: jnp.ndarray         # (K,C) f64
    hi: jnp.ndarray         # (K,C) f64
    end_bay: jnp.ndarray    # (K,C) f64
    end_row: jnp.ndarray    # (K,C) f64
    lane: jnp.ndarray       # (K,C) int32
    loaded_m: jnp.ndarray   # (K,C) f64
    empty_m: jnp.ndarray    # (K,C) f64
    n_moves: jnp.ndarray    # (K,C) int32


def flat_view(out: CandOut3) -> FlatCands:
    """CandOut3 → (K, N+R+1) 평면 행렬. SERVE 와 PRE 는 같은 열(오더)을 쓴다 — 상태로 배타."""
    K, N = out.serve_raw.shape
    R = out.repo_bay.shape[1]
    Ps, Pp, Pr = out.m.P, out.P_pre, out.P_repo
    sr, pr = out.serve_raw, out.pre_raw
    nan = jnp.full((K, N), jnp.nan, F)
    neg_inf = jnp.full((K, 1), -jnp.inf, F)
    one = jnp.ones((K, 1), bool)
    zero = jnp.zeros((K, 1), bool)

    def pick(a, b, default):
        return jnp.where(sr, a, jnp.where(pr, b, default))

    serve_code = jnp.where(Ps.ok, out.m.code, EMPTY_ID)                          # 계획 없는 mandatory → PLAN_FAILED(-1)
    raw = jnp.concatenate([sr | pr, out.repo_raw, one], axis=1)
    feasible = jnp.concatenate([out.serve | out.pre, out.repo, one], axis=1)
    mandatory = jnp.concatenate([sr & out.mandatory[None, :], jnp.zeros((K, R), bool), zero], axis=1)
    kind = jnp.concatenate([pick(jnp.int32(PK_SERVE), jnp.int32(PK_PRE_REHANDLE), jnp.int32(EMPTY_ID)),
                            jnp.where(out.repo_raw, PK_REPOSITION, EMPTY_ID).astype(jnp.int32),
                            jnp.full((K, 1), PK_WAIT, jnp.int32)], axis=1)
    job = jnp.concatenate([jnp.where(sr | pr, jnp.arange(N, dtype=jnp.int32)[None, :], EMPTY_ID),
                           jnp.full((K, R), EMPTY_ID, jnp.int32), jnp.full((K, 1), EMPTY_ID, jnp.int32)], axis=1)
    bay = jnp.concatenate([nan, jnp.where(out.repo_raw, out.repo_bay, jnp.nan), jnp.full((K, 1), jnp.nan, F)], axis=1)
    score = jnp.concatenate([pick(out.serve_score, out.pre_score, jnp.nan), out.repo_score, neg_inf], axis=1)
    code = jnp.concatenate([pick(serve_code, out.pre_code, jnp.int32(EMPTY_ID)), out.repo_code,
                            jnp.zeros((K, 1), jnp.int32)], axis=1)

    def col(a, b, c, fill):
        return jnp.concatenate([pick(a, b, jnp.asarray(fill, a.dtype)), c, jnp.full((K, 1), fill, a.dtype)], axis=1)

    fc = FlatCands(
        raw=raw, feasible=feasible, mandatory=mandatory, kind=kind, job=job, bay=bay, score=score, code=code,
        plan_ok=jnp.concatenate([pick(Ps.ok, Pp.ok, False), Pr.ok, zero], axis=1),
        dur=col(Ps.dur, Pp.dur, Pr.dur, jnp.nan), rehandles=col(Ps.rehandles, Pp.rehandles, Pr.rehandles, 0),
        lo=col(Ps.lo, Pp.lo, Pr.lo, jnp.nan), hi=col(Ps.hi, Pp.hi, Pr.hi, jnp.nan),
        end_bay=col(Ps.end_bay, Pp.end_bay, Pr.end_bay, jnp.nan), end_row=col(Ps.end_row, Pp.end_row, Pr.end_row, jnp.nan),
        lane=col(Ps.lane, Pp.lane, Pr.lane, EMPTY_ID), loaded_m=col(Ps.loaded_m, Pp.loaded_m, Pr.loaded_m, jnp.nan),
        empty_m=col(Ps.empty_m, Pp.empty_m, Pr.empty_m, jnp.nan), n_moves=col(Ps.n_moves, Pp.n_moves, Pr.n_moves, 0))
    return fc


# ───────────────────────────────────────────────── prune (505-521행)
class PruneOut(NamedTuple):
    keep: jnp.ndarray          # (K,C) bool   generate items 에 실린 후보 (WAIT 포함)
    candidate_id: jnp.ndarray  # (K,C) int32  items 위치 (order_key 정렬 · WAIT 마지막), 안 실리면 -1
    n_kept: jnp.ndarray        # (K,)  int32  WAIT 제외 개수 (= WAIT 의 candidate_id)
    rest_rank: jnp.ndarray     # (K,C) int32  비-mandatory 원시 후보의 (−score, order_key) 순위 (진단용; 아니면 C)


def prune(flat: FlatCands, g: Geom, *, k_max: int = K_MAX) -> PruneOut:
    """`_prune` + `sorted(order_key)` — 머리말 ■ prune. g 는 문자열 순위표(B)에 쓴다. 열 수 C 는 모양에서 읽는다."""
    K, C = flat.raw.shape
    B = int(g.bay_count)
    budget = int(k_max) - 1                                                      # 511행 WAIT 1칸 예약
    wait_col = jnp.arange(C) == C - 1
    is_wait = wait_col[None, :]
    raw = flat.raw & ~is_wait
    mand = flat.mandatory & raw
    rest = raw & ~mand
    kind_rank = jnp.where(flat.kind == PK_SERVE, 0, jnp.where(flat.kind == PK_PRE_REHANDLE, 1,
                          jnp.where(flat.kind == PK_REPOSITION, 2, 3))).astype(jnp.int32)
    str_rank = _str_rank_table(B)
    bay_i = jnp.clip(jnp.where(jnp.isnan(flat.bay), 0.0, flat.bay).astype(jnp.int32), 0, B)   # int(tb) — 양수라 절사
    name_rank = jnp.where(flat.job >= 0, flat.job, str_rank[bay_i]).astype(jnp.int32)   # job_id 문자열 순서
    bay_key = jnp.where(jnp.isnan(flat.bay), -1.0, flat.bay)                    # order_key 셋째 (523행 -1.0)
    neg = -flat.score
    neg = jnp.where(neg == 0.0, 0.0, neg)                                        # −0.0 → +0.0 (파이썬 sort 와 같게)
    neg = jnp.where(rest, neg, jnp.inf)                                          # rest 밖은 뒤로

    def per_k(rest_k, neg_k, kr, nr, bk, mand_k):
        perm = jnp.lexsort((bk, nr, kr, neg_k, ~rest_k))                         # 513행 (−score,)+order_key
        rank = jnp.zeros((C,), jnp.int32).at[perm].set(jnp.arange(C, dtype=jnp.int32))
        rank = jnp.where(rest_k, rank, C)
        n_mand = jnp.sum(mand_k).astype(jnp.int32)
        keep_k = mand_k | (rest_k & (rank < jnp.maximum(0, budget - n_mand)))    # 514행
        perm2 = jnp.lexsort((bk, nr, kr, ~keep_k))                              # 261행 sorted(order_key)
        pos = jnp.zeros((C,), jnp.int32).at[perm2].set(jnp.arange(C, dtype=jnp.int32))
        n_keep = jnp.sum(keep_k).astype(jnp.int32)
        cid = jnp.where(keep_k, pos, jnp.where(wait_col, n_keep, EMPTY_ID)).astype(jnp.int32)
        return keep_k | wait_col, cid, n_keep, rank

    keep, cid, n_kept, rank = jax.vmap(per_k)(rest, neg, kind_rank, name_rank, bay_key, mand)
    return PruneOut(keep=keep, candidate_id=cid, n_kept=n_kept, rest_rank=rank)


# ───────────────────────────────────────────────── DEFER (YR-147 · candidates.py:459-491) — LEGACY 미사용
class DeferOut(NamedTuple):
    trigger_s: jnp.ndarray     # () f64   가장 이른 관측 가능 미래 시각 (+inf = 없음)
    trigger_kind: jnp.ndarray  # () int32 0 ETA · 1 RELEASE · -1 없음
    trigger_job: jnp.ndarray   # () int32 오더 번호 (-1)
    defer_until: jnp.ndarray   # () f64   min(trigger, now + t_max) 또는 now + t_max


def defer_wait(world: BlockWorld, g: Geom, pre_advice: bool, *, t_max: float = DEFER_T_MAX) -> DeferOut:
    """`_defer_trigger_time` + `_wait` 의 DEFER 분기 — wait_mode ≠ WAIT 일 때 WAIT 후보에 붙는 재개방 시각.

    PLANNED 오더만: 외부트럭은 가시 ETA(PRE_ADVICE 의 provided_eta), 내부작업은 release_time. `t > now` 인 최소 —
    같은 값이면 오더 번호(job_id 정렬) 작은 것 (파이썬 `<` 갱신 = 첫 최소). 실현 미래(actual_*)는 안 읽는다.
    """
    o = world.orders
    clock = world.clock
    eta = o.provided_eta_s if pre_advice else jnp.full_like(o.provided_eta_s, EMPTY_TIME)
    t = jnp.where(o.is_external, eta, o.release_s)
    on = (o.block >= 0) & (o.status == JS_PLANNED) & (t < EMPTY_TIME) & (t > clock)
    vals = jnp.where(on, t, EMPTY_TIME)
    best = jnp.min(vals)
    has = best < EMPTY_TIME
    j = jnp.argmin(vals).astype(jnp.int32)
    kind = jnp.where(has, jnp.where(o.is_external[j], 0, 1), EMPTY_ID).astype(jnp.int32)
    expiry = clock + jnp.asarray(float(t_max), F)                                # 480행
    until = jnp.where(has, jnp.minimum(best, expiry), expiry)                    # 482, 485행
    return DeferOut(trigger_s=best, trigger_kind=kind, trigger_job=jnp.where(has, j, EMPTY_ID).astype(jnp.int32),
                    defer_until=until)
