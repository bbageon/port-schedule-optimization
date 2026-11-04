"""PRE_ADVICE 깨우기 — ETA wake · armed · DEFER wake 를 **배열로** ([[YR-327]] 조각 3 · key=wake).

v5 `world/integrated/engine.py` 의 다음 부분을 옮긴다 (줄 번호는 v5 사본 기준):
    159-179  reset — ETA wake 목록 시드: GATE_OUT + 대상 있음 + provided_eta 있음 + max(0, eta−horizon) < end,
             (시각, job_id) 정렬 · `_wake_idx = 0` · `_eta_armed = ∅` · `_defer_wakes = []`
    339-347  `schedule_defer_wake(t)` — 유한 DEFER 만료 예약 (clock+EPS < t ≤ end+EPS 만, insort)
    349-357  `_next_wake_time` — 미소비 wake 최소 시각 (ETA 는 PRE_ADVICE 한정 · DEFER 는 정보수준 무관)
    359-382  `_consume_due_wakes` — 도래 wake 전부 소비 (DEFER 먼저, 그다음 ETA) · 로그 · 전 크레인 arm + yield 해제
    453-464  `_decision_cranes` — 결정 개방 = eligible & (SERVE 후보 있음 | (armed & eta_opportunity))
    candidates.py:57-86 · 183-197  `iter_pre_rehandle_jobs` / `eta_opportunity` — 선제 재조작 기회 게이트 (plan 전)

■ 배열 표현 (state.WakeArrays)
    eta_wake_s   (W,) f64   오름차순, 빈 칸 +inf        eta_wake_job (W,) int32  오더 번호 (빈 칸 -1)
    wake_idx     ()   int32 다음에 소비할 칸 (v5 `_wake_idx`)
    eta_armed    (K,) bool  v5 `_eta_armed` 집합
    defer_wake_s (D,) f64   오름차순 · 유효 칸 [0, defer_n) · 나머지 +inf  (v5 `_defer_wakes` 리스트)
    defer_n      ()   int32
  v5 는 (시각, job_id 문자열) 로 정렬한다. 오더 번호 n 이 sorted(job_id) 순위이므로 **시각에 대한 안정 정렬**이
  같은 순서다 (`jnp.argsort(stable=True)`). W·D 는 모양(static) — 0 이면 그 종류는 없는 것으로 계산이 빠진다.

■ ★스텝 사슬에서의 자리 (v5 `run_until_decision` 276-336행) — 통합(engine_step) 몫, 여기서는 순수 함수만
    동시각 사건 소진(due_now) → [W] `consume_due_wakes` 가 fired 면 **그 스텝 끝** (continue, 291-292행)
    → [D] 결정 (개방은 `open_with_armed`, 460-463행) → [X] 탈출 → wt = `next_wake_in_window` (306-308행, 311-313행)
    → [F] nt·wt 둘 다 없음(+inf) → 종료 → [A] wt 있고 (nt 없거나 wt < nt − EPS) → `advance(wt)` 만 하고 스텝 끝 (333-335행)
    → [E] 사건 하나
  W 가 발화한 스텝에서는 결정을 열지 않는다 (명세 hard_parts ③): v5 는 continue 로 돌아와 소비를 다시 시도(이번엔
  False)한 뒤에야 결정을 본다. 상태는 같으니 결정 내용은 같지만 스텝 수·흔적이 달라지므로 별 스텝으로 둔다.
  armed 소진(297행 `_eta_armed -= idle`)은 `engine_step.close_decision(consume_armed=is_D)` 가 이미 한다.

■ 로그 — 소비 하나에 로그 한 줄. (clock, LOG_DEFER_WAKE=14, -1) 들 먼저, 이어 (clock, LOG_ETA_WAKE=13, 오더) 들
  (369-370행 → 375-377행 순서). 칸이 모자라면 `overflow` 에 부족 수를 더한다 (`log_event` 와 같은 규약).
  W 개 scatter 가 아니라 **로그 칸 E 에 대한 gather** 로 쓴다 — 같은 칸을 여러 원소가 겨냥하는 scatter 는 XLA 가
  순서를 정하지 않는다.

■ ★명세와 v5 코드가 다른 곳 — v5 가 정본
  piece3_spec 의 PRE 게이트 "0 < eta − clock ≤ horizon" 은 v5 `iter_pre_rehandle_jobs`(candidates.py:81-83) 와
  다르다. v5 는 `eta − now > horizon` 만 거르므로 **ETA 가 이미 지난(음수 gap) 미도착 트럭도 기회**다 —
  test_yr050 `negative_gap` 시험이 바로 그 경로를 요구한다. `eta <= now` 배제는 REPOSITION 쪽
  (`iter_eta_reposition_jobs` 165행)에만 있다. `eta_opportunity_mask` 는 v5 를 따른다.

■ 동등성
  시각 비교는 v5 식 그대로 (`x <= clock + EPS`, `eta − now > horizon`, `max(0.0, eta − horizon) < end`).
  곱셈은 없다. 정렬은 안정 정렬. 파이썬 `min(et, dw)` 는 `jnp.minimum` 과 같다 (NaN 없음).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .geom import Geom
from .stack_ops import blockers_above, rehandle_capacity_ok
from .state import (FL_GATE_OUT, JS_PLANNED, LOG_DEFER_WAKE, LOG_ETA_WAKE, BlockWorld, OrderArrays,
                    WakeArrays)

__all__ = ["EPS", "wake_times", "n_wakes", "seed_wakes", "schedule_defer_wake", "next_wake_time",
           "next_wake_in_window", "consume_due_wakes", "eta_opportunity_mask", "open_with_armed"]

F = TIME_DTYPE
#: v5 `_EPS` (engine.py:35) — engine_step.EPS 와 같은 값 (순환 import 를 피해 여기 다시 적는다)
EPS = 1e-9


def _i32(x):
    return jnp.asarray(x, jnp.int32)


# ───────────────────────────────────────────────── 시드 (166-171행)
def wake_times(o: OrderArrays, *, horizon_s, end_s):
    """오더마다 wake 시각과 자격 — v5 166-171행의 조건을 (N,) 으로.

        자격  flow == GATE_OUT & 대상 있음 & provided_eta 있음 & max(0, eta − horizon) < end
        시각  max(0.0, eta − horizon)     (자격 없으면 +inf)
    반환 (t (N,) f64, valid (N,) bool). GATE_IN 은 시드하지 않는다 (172-175행 주석: 선제 기회가 반출에서만 생긴다).
    """
    horizon = jnp.asarray(horizon_s, F)
    end = jnp.asarray(end_s, F)
    has_eta = o.provided_eta_s < EMPTY_TIME
    t_raw = jnp.maximum(0.0, o.provided_eta_s - horizon)                 # 167행 max(0.0, provided_eta − horizon)
    valid = ((o.block >= 0) & (o.flow == FL_GATE_OUT) & (o.target_cont >= 0)   # 169행
             & has_eta & (t_raw < end))                                  # 170-171행
    return jnp.where(valid, t_raw, EMPTY_TIME), valid


def n_wakes(o: OrderArrays, *, horizon_s, end_s) -> jnp.ndarray:
    """시드될 wake 수 () int32 — 호스트가 W(n_wake) 를 정할 때 쓴다."""
    _, valid = wake_times(o, horizon_s=horizon_s, end_s=end_s)
    return jnp.sum(valid).astype(jnp.int32)


def seed_wakes(o: OrderArrays, k: int, *, horizon_s, end_s, n_wake: int,
               n_defer: int = 0, n_review: int = 0):
    """reset 의 wake 상태 (166-179행) → `WakeArrays`. 반환 (wake, dropped () int32).

    (시각, 오더 번호) 오름차순 = v5 (시각, job_id) 정렬. W(n_wake) 칸에 앞에서부터 담고, 넘치는 뒤쪽 wake 수를
    `dropped` 로 돌려준다 — **조용히 자르지 않는다**. 호스트는 `n_wakes` 로 W 를 정하고 dropped 가 0 이 아니면
    실격(overflow) 처리해야 한다. wake_idx 0 · armed 전부 False · defer 비어 있음 (172-179행).
    """
    N = o.n
    W = int(n_wake)
    t, valid = wake_times(o, horizon_s=horizon_s, end_s=end_s)
    order = jnp.argsort(t, stable=True)                                  # 171행 sorted((t, job_id)) — 안정 정렬
    ts = t[order]
    js = jnp.where(valid[order], order, EMPTY_ID).astype(jnp.int32)
    count = jnp.sum(valid).astype(jnp.int32)
    if N >= W:
        ts, js = ts[:W], js[:W]
    else:
        ts = jnp.concatenate([ts, jnp.full((W - N,), EMPTY_TIME, F)])
        js = jnp.concatenate([js, jnp.full((W - N,), EMPTY_ID, jnp.int32)])
    dropped = jnp.maximum(count - W, 0).astype(jnp.int32)
    wake = WakeArrays(eta_wake_s=ts.astype(F), eta_wake_job=js,
                      wake_idx=_i32(0), eta_armed=jnp.zeros((int(k),), bool),
                      defer_wake_s=jnp.full((int(n_defer),), EMPTY_TIME, F), defer_n=_i32(0),
                      review_s=jnp.full((int(n_review),), EMPTY_TIME, F), review_idx=_i32(0))
    return wake, dropped


# ───────────────────────────────────────────────── DEFER 예약 (339-347행)
def schedule_defer_wake(world: BlockWorld, t) -> BlockWorld:
    """`schedule_defer_wake(t)` — clock+EPS < t ≤ end+EPS 일 때만 정렬 삽입 (insort). 아니면 무시.

    칸(D)이 모자라면 `overflow` +1 (v5 리스트는 무한). D == 0 이면 예약 자체가 넘침이다.
    """
    wk = world.wake
    t = jnp.asarray(t, F)
    ok = (t > world.clock + EPS) & (t <= world.end_s + EPS)              # 345행
    D = wk.defer_wake_s.shape[0]
    if D == 0:
        return world._replace(overflow=world.overflow + jnp.where(ok, 1, 0).astype(jnp.int32))
    room = wk.defer_n < D
    put = ok & room
    slot = jnp.clip(wk.defer_n, 0, D - 1)
    arr = wk.defer_wake_s.at[slot].set(jnp.where(put, t, wk.defer_wake_s[slot]))
    arr = jnp.sort(arr)                                                  # 346행 insort — +inf 는 뒤로
    wk2 = wk._replace(defer_wake_s=arr, defer_n=wk.defer_n + jnp.where(put, 1, 0).astype(jnp.int32))
    return world._replace(wake=wk2, overflow=world.overflow + jnp.where(ok & ~room, 1, 0).astype(jnp.int32))


# ───────────────────────────────────────────────── 다음 wake 시각 (349-357행)
def _first_defer(wk: WakeArrays):
    D = wk.defer_wake_s.shape[0]
    if D == 0:
        return jnp.asarray(EMPTY_TIME, F)
    return jnp.where(wk.defer_n > 0, wk.defer_wake_s[0], EMPTY_TIME)   # 351행


def _next_eta(wk: WakeArrays):
    W = wk.eta_wake_s.shape[0]
    if W == 0:
        return jnp.asarray(EMPTY_TIME, F)
    i = jnp.clip(wk.wake_idx, 0, W - 1)
    return jnp.where(wk.wake_idx < W, wk.eta_wake_s[i], EMPTY_TIME)     # 354-356행


def next_wake_time(world: BlockWorld, *, pre_advice: bool) -> jnp.ndarray:
    """`_next_wake_time` — () f64, 없으면 +inf (v5 None). pre_advice 는 static (info_level == PRE_ADVICE)."""
    wk = world.wake
    dw = _first_defer(wk)
    if not pre_advice:
        return dw                                                        # 352-353행
    return jnp.minimum(_next_eta(wk), dw)                                # 357행 min(et, dw)


def next_wake_in_window(world: BlockWorld, *, pre_advice: bool) -> jnp.ndarray:
    """306-308행: `_next_wake_time` 에 평가창 규칙(wt > end+EPS → 없음)을 적용한 값. 시계 전진 목표로 쓴다."""
    wt = next_wake_time(world, pre_advice=pre_advice)
    return jnp.where(wt > world.end_s + EPS, EMPTY_TIME, wt)


# ───────────────────────────────────────────────── 소비 (359-382행)
def consume_due_wakes(world: BlockWorld, *, pre_advice: bool):
    """`_consume_due_wakes` 한 번 — 순수 함수. 반환 (세계', fired () bool).

    DEFER: 유효 칸 앞에서부터 `≤ clock+EPS` 인 것을 전부 뺀다 (정렬돼 있어 접두 구간) · 로그 (clock, 14, -1) 씩.
    ETA  : pre_advice 일 때만, wake_idx 부터 `≤ clock+EPS` 인 것을 전부 소비 (접두 구간) · 로그 (clock, 13, 오더) 씩.
    fired 면 armed 전부 True · yielded 전부 False (380-382행). 아니면 세계 그대로.
    """
    wk, lg, cr = world.wake, world.log, world.cranes
    clock = world.clock
    due_t = clock + EPS                                                  # 368·374행 `<= self.clock + _EPS`
    W = wk.eta_wake_s.shape[0]
    D = wk.defer_wake_s.shape[0]

    # DEFER (368-371행)
    if D > 0:
        d_idx = jnp.arange(D, dtype=jnp.int32)
        d_due = (d_idx < wk.defer_n) & (wk.defer_wake_s <= due_t)
        n_dd = jnp.sum(d_due).astype(jnp.int32)
        src = jnp.clip(d_idx + n_dd, 0, D - 1)
        keep = (d_idx + n_dd) < wk.defer_n
        defer_s2 = jnp.where(keep, wk.defer_wake_s[src], EMPTY_TIME)    # pop(0) × n_dd = 왼쪽으로 당김
        defer_n2 = wk.defer_n - n_dd
    else:
        n_dd = _i32(0)
        defer_s2, defer_n2 = wk.defer_wake_s, wk.defer_n

    # ETA (372-378행) — PRE_ADVICE 한정
    if pre_advice and W > 0:
        w_idx = jnp.arange(W, dtype=jnp.int32)
        e_due = (w_idx >= wk.wake_idx) & (wk.eta_wake_s <= due_t)
        n_ed = jnp.sum(e_due).astype(jnp.int32)
    else:
        n_ed = _i32(0)
    wake_idx2 = wk.wake_idx + n_ed
    total = n_dd + n_ed
    fired = total > 0                                                    # 379행

    # 로그 — 칸 E 에 대한 gather (머리말 ■ 로그)
    E = lg.capacity
    e_idx = jnp.arange(E, dtype=jnp.int32)
    pos = e_idx - lg.n
    is_d = (pos >= 0) & (pos < n_dd)
    is_e = (pos >= n_dd) & (pos < total)
    if W > 0:
        wi = jnp.clip(wk.wake_idx + (pos - n_dd), 0, W - 1)
        job = wk.eta_wake_job[wi]
    else:
        job = jnp.full((E,), EMPTY_ID, jnp.int32)
    new = is_d | is_e
    t2 = jnp.where(new, clock, lg.t)
    kind2 = jnp.where(is_d, LOG_DEFER_WAKE, jnp.where(is_e, LOG_ETA_WAKE, lg.kind)).astype(jnp.int32)
    target2 = jnp.where(is_d, EMPTY_ID, jnp.where(is_e, job, lg.target)).astype(jnp.int32)
    n2 = jnp.minimum(lg.n + total, E).astype(jnp.int32)
    dropped = jnp.maximum(lg.n + total - E, 0).astype(jnp.int32)
    lg2 = lg._replace(t=t2, kind=kind2, target=target2, n=n2)

    wk2 = wk._replace(wake_idx=wake_idx2, defer_wake_s=defer_s2, defer_n=defer_n2,
                      eta_armed=jnp.where(fired, True, wk.eta_armed))   # 380행 set(fleet.ids())
    cr2 = cr._replace(yielded=jnp.where(fired, False, cr.yielded))       # 381행 _clear_yields
    return world._replace(wake=wk2, log=lg2, cranes=cr2, overflow=world.overflow + dropped), fired


# ───────────────────────────────────────────────── 선제 재조작 기회 (candidates.py:57-86, 183-197)
def eta_opportunity_mask(world: BlockWorld, g: Geom, *, horizon_s, pre_advice: bool) -> jnp.ndarray:
    """`eta_opportunity(sim, crane_id, level)` 를 (K,) bool 로 — `iter_pre_rehandle_jobs` 의 plan 전 게이트.

    오더 조건 (전 크레인 공통)  GATE_OUT & PLANNED & 대상 있음 & 야드에 있고 가용 & blocker ≥ 1 & ETA 있음 &
                                ~(eta − clock > horizon)   ← ★eta ≤ clock 도 기회 (머리말 ★명세와 다른 곳)
    크레인 조건                 대상 bay 가 담당 구간 안 & 재조작 칸 충분 (rehandle_capacity_ok, 구간 기준)
    pre_advice 가 아니면 전부 False (73행 · 195행).
    """
    K = world.k
    if not pre_advice:
        return jnp.zeros((K,), bool)
    o, cr, st, ct = world.orders, world.cranes, world.stacks, world.conts
    C = ct.c
    clock = world.clock
    horizon = jnp.asarray(horizon_s, F)
    has_t = o.target_cont >= 0
    tcc = jnp.clip(o.target_cont, 0, C - 1)
    exists = ct.c_alive[tcc] & ct.c_avail[tcc]                           # 71-73행 containers.get · work_available
    blk_fn = jax.vmap(partial(blockers_above, g=g), in_axes=(None, None, None, None, None, 0))
    _, n_block = blk_fn(st.grid, st.height, ct.c_bay, ct.c_row, ct.c_tier, o.target_cont)   # 76행
    eta_ok = (o.provided_eta_s < EMPTY_TIME) & ~((o.provided_eta_s - clock) > horizon)     # 81-83행
    job_ok = ((o.block >= 0) & (o.flow == FL_GATE_OUT) & (o.status == JS_PLANNED)          # 67행
              & has_t & exists & (n_block > 0) & eta_ok)
    tb, tr = ct.c_bay[tcc], ct.c_row[tcc]

    def per_kn(bmin, bmax, tb_, tr_, nb_):
        in_range = (bmin <= tb_) & (tb_ <= bmax)                         # 74행
        cap = rehandle_capacity_ok(st.height, st.top_size, tb_, tr_, nb_, bmin, bmax, g)   # 78행
        return in_range & cap

    over_n = jax.vmap(per_kn, in_axes=(None, None, 0, 0, 0))
    over_kn = jax.vmap(over_n, in_axes=(0, 0, None, None, None))
    m = over_kn(cr.bay_min, cr.bay_max, tb, tr, n_block) & job_ok[None, :]
    return jnp.any(m, axis=1)


# ───────────────────────────────────────────────── 결정 개방 (453-464행)
def open_with_armed(cand, eligible, armed, eta_opp) -> jnp.ndarray:
    """`_decision_cranes` — open[k] = eligible[k] & (any_n cand[k,n] | (armed[k] & eta_opp[k])). (K,) bool.

    cand (K,N) = candidates_for (escape.CandMats.cand) · eligible (K,) = idle & ~yielded (456행) ·
    armed (K,) = wake.eta_armed · eta_opp (K,) = `eta_opportunity_mask`.
    """
    return jnp.asarray(eligible, bool) & (jnp.any(jnp.asarray(cand, bool), axis=1)
                                          | (jnp.asarray(armed, bool) & jnp.asarray(eta_opp, bool)))
