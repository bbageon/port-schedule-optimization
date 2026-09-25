"""Φ — 터미널 총비용 네 항(원화)의 **배열판** ([[YR-327]] 조각 5).

v5 `reward/phi.py:terminal_cost_krw` 와 **같은 답**을 낸다 — 정수·분위수는 비트 일치, 원화 합은
v5 가 `+=` 로 더한 순서(기록 사전의 삽입 순서 = 배열 순서) 그대로 `lax.scan` 으로 더한다.

    Φ = C_wait + C_move + C_rehandle + C_vessel        [원]

    항1 wait      기록(A·O)에서 검열 턴타임 → 원화 (1시간 초과분 할증)     … (N,) 벡터 + 마스크 합
    항2 move      YC 빈 주행 초 (터미널 누적 계수기 값을 호출부가 준다)
    항3 rehandle  재조작 횟수 (계수기)
    항4 vessel    {배: (GT, 유휴 초)} 표 → 배마다 원화 → 표 순서대로 합

■ 검열은 `state.censored_turn_time_s` 를 **재사용**한다 (중복 구현 금지). 다만 v5 Φ 의 규칙
  (schema/lifecycle.py:129-142 `max(0, min(O, end) − A)` · 진단은 `A ≤ end` 인 트럭)과 한 곳이 다르다:
  `A == end` 이고 아직 안 나간 트럭을 state 판은 표본에서 빼지만(NaN) v5 Φ 는 **턴타임 0 인 검열
  표본**으로 센다 (n_trucks·n_censored 에 들어가고 분위수 표본이 0 이 된다). 무대에서 실제로
  생긴다 — 검토 격자 60초에 트럭이 60초에 게이트 진입하면 t=60 의 Φ 가 그 경우다. 그래서 여기서
  그 칸만 0 으로 되살린다. 대기 원화는 0 이라 어느 쪽이든 같다.

■ 부동소수점 규약 (exact.py 머리말)
  · 곱은 `mul_exact` — `K * tt / 3600` 의 곱이 뒤 덧셈과 FMA 로 묶이지 않게 실체화
  · 상수 나눗셈은 분모를 **같은 모양으로 펼쳐 장벽 뒤에** 둔다 — vmap 아래서도 `divide(a, broadcast(c))`
    패턴이 안 생겨 역수 곱으로 안 바뀐다 (`_div_c`, travel.div_exact 위에)
  · 합은 `lax.scan` 으로 원소 순서대로 (jnp.sum 은 축소 순서가 규정돼 있지 않다)
  · 분위수는 정렬 뒤 `index = min(n−1, floor(p·n))` **원소 자체** — 보간 금지 (v5 phi.py:115-118)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..reward.krw import (KRW_OVERTIME_START_S, KRW_REHANDLE_EACH, KRW_TRUCK_HOUR,
                          KRW_TRUCK_OVER_HOUR, KRW_VESSEL_GT_HOUR, KRW_YC_MOVE_HOUR)
from .events import TIME_DTYPE
from .exact import mul_exact, sum_seq
from .state import EMPTY_TIME, OrderArrays, censored_turn_time_s, empty_orders
from .travel import div_exact

__all__ = ["PhiArrays", "terminal_cost_krw", "terminal_cost_krw_batch", "truck_wait_krw",
           "vessel_idle_krw", "vessel_idle_by_ship", "orders_from_records", "PHI_KEYS"]

#: v5 `PhiBreakdown.as_dict()` 의 키 — 같은 순서
PHI_KEYS = ("phi_krw", "c_wait", "c_move", "c_rehandle", "c_vessel", "n_trucks", "n_censored",
            "mean_turn_time_s", "over_ratio", "p50_turn_time_s", "p90_turn_time_s", "p99_turn_time_s")


class PhiArrays(NamedTuple):
    """v5 `PhiBreakdown` 의 배열판 — 네 항·합계와 진단 열. 배치면 잎마다 앞축 B 가 붙는다."""

    wait: jnp.ndarray        # () f64  항1 트럭 대기 (원)
    move: jnp.ndarray        # () f64  항2 YC 추가 이동 (원)
    rehandle: jnp.ndarray    # () f64  항3 재조작 (원)
    vessel: jnp.ndarray      # () f64  항4 본선 유휴 (원)
    total: jnp.ndarray       # () f64  Φ = ((wait + move) + rehandle) + vessel — v5 `total` 순서
    # ── 진단 열 (Φ 에 안 들어간다) — 창 안 트럭(A ≤ end) 기준
    n_trucks: jnp.ndarray          # () int32
    n_censored: jnp.ndarray        # () int32  창 끝에 아직 안 나간 트럭
    mean_turn_time_s: jnp.ndarray  # () f64
    over_ratio: jnp.ndarray        # () f64   턴타임 > 3600 비율
    p50_turn_time_s: jnp.ndarray   # () f64
    p90_turn_time_s: jnp.ndarray   # () f64
    p99_turn_time_s: jnp.ndarray   # () f64

    def as_dict(self, i: int | None = None) -> dict:
        """v5 `as_dict()` 와 같은 키·파이썬 스칼라. 배치면 `i` 번째 세계를 고른다."""
        pick = (lambda x: x) if i is None else (lambda x: x[i])
        return {
            "phi_krw": float(pick(self.total)),
            "c_wait": float(pick(self.wait)), "c_move": float(pick(self.move)),
            "c_rehandle": float(pick(self.rehandle)), "c_vessel": float(pick(self.vessel)),
            "n_trucks": int(pick(self.n_trucks)), "n_censored": int(pick(self.n_censored)),
            "mean_turn_time_s": float(pick(self.mean_turn_time_s)),
            "over_ratio": float(pick(self.over_ratio)),
            "p50_turn_time_s": float(pick(self.p50_turn_time_s)),
            "p90_turn_time_s": float(pick(self.p90_turn_time_s)),
            "p99_turn_time_s": float(pick(self.p99_turn_time_s)),
        }


# ───────────────────────────────────────────────── 정확한 산술 도우미
def _div_c(a, c: float):
    """`a / c` (c 는 파이썬 상수) — 분모를 `a` 와 같은 모양으로 펼쳐 장벽 뒤에 둔다.

    `exact.div_const` 는 0-차원 분모라 (N,) 벡터를 나누면 `divide(a, broadcast(c))` 가 되고, vmap 아래서는
    한 번 더 펼쳐진다. 같은 모양의 장벽 출력으로 나누면 XLA 가 상수도 broadcast 도 못 본다.
    """
    d = lax.optimization_barrier(jnp.full(jnp.shape(a), float(c), TIME_DTYPE))
    return div_exact(a, d)


def _sum_in_order(terms, on):
    """파이썬 `acc += x` 를 **원소 순서대로** — 마스크가 꺼진 칸은 건너뛴다 (state.censored_exposure_s 꼴)."""
    def body(acc, x):
        t, m = x
        return jnp.where(m, acc + t, acc), None
    acc, _ = lax.scan(body, jnp.asarray(0.0, TIME_DTYPE), (terms, on))
    return acc


# ───────────────────────────────────────────────── 단가 (krw.py 의 배열판 — 식의 결합 순서 그대로)
def truck_wait_krw(turn_time_s):
    """항1 — `krw.truck_wait_krw`: `K·tt/3600 + K_over·max(0, tt−3600)/3600` (elementwise)."""
    tt = jnp.maximum(0.0, jnp.asarray(turn_time_s, TIME_DTYPE))
    base = _div_c(mul_exact(KRW_TRUCK_HOUR, tt), 3600.0)
    over = _div_c(mul_exact(KRW_TRUCK_OVER_HOUR, jnp.maximum(0.0, tt - KRW_OVERTIME_START_S)), 3600.0)
    return base + over


def vessel_idle_krw(gt, idle_s):
    """항4 — `krw.vessel_idle_krw`: `(2.99·gt)·max(0, idle)/3600` (elementwise)."""
    per_hour = mul_exact(KRW_VESSEL_GT_HOUR, jnp.asarray(gt, TIME_DTYPE))
    return _div_c(mul_exact(per_hour, jnp.maximum(0.0, jnp.asarray(idle_s, TIME_DTYPE))), 3600.0)


def vessel_idle_by_ship(stream_wait_s, stream_ship, ship_sts, n_ships: int):
    """스트림 유휴 → **배 단위** 유휴 표 — `episode.vessel_idle_of`/`month.month_vessel_idle` 의
    `sum(w) / max(1, sts)` 를 스트림 순서대로.

    ★파이썬 3.12 부터 `sum()` 은 실수 목록을 **Neumaier 보정합**으로 더한다 (CPython bltinmodule.c:
    첫 항은 `0 + w0`, 이후 `t = f + x; c += |f|≥|x| ? (f−t)+x : (x−t)+f; f = t`, 끝에 `c` 가 0 이 아니고
    유한하면 `f + c`). 순차 `+=` 로 더하면 세 항부터 마지막 비트가 갈린다 (실측 2026-09-25, 40 스트림·4척
    에서 1척 갈림). 그래서 같은 알고리즘을 배마다 (f, c, 첫항 여부) 로 들고 스트림 순서대로 scan 한다.
    (`terminal_cost_krw` 본체의 세 합은 v5 가 `+=` 루프라 보정 없이 순차다 — `_sum_in_order`.)

    `stream_wait_s` (S,) f64 · `stream_ship` (S,) int32 (배 번호, -1 = 표에 없는 배 → 안 센다)
    `ship_sts` (V,) int32 STS 대수. 결과 (V,) f64 유휴 초 — `terminal_cost_krw(vessel_idle_s=…)` 에 넣는다.
    """
    def body(carry, x):
        f, c, seen = carry
        w, s = x
        i = jnp.maximum(s, 0)
        fi, ci, first = f[i], c[i], ~seen[i]
        t = lax.optimization_barrier(fi + w)                     # 실체화 — (f−t)+w 가 대수적으로 안 지워진다
        comp = jnp.where(jnp.abs(fi) >= jnp.abs(w), (fi - t) + w, (w - t) + fi)
        nf = jnp.where(first, 0.0 + w, t)                        # 첫 항: int 0 + w0
        nc = jnp.where(first, ci, ci + comp)
        ok = s >= 0
        f = jnp.where(ok, f.at[i].set(nf), f)
        c = jnp.where(ok, c.at[i].set(nc), c)
        seen = jnp.where(ok, seen.at[i].set(True), seen)
        return (f, c, seen), None
    z = jnp.zeros((n_ships,), TIME_DTYPE)
    (f, c, _), _ = lax.scan(body, (z, z, jnp.zeros((n_ships,), jnp.bool_)),
                            (jnp.asarray(stream_wait_s, TIME_DTYPE), jnp.asarray(stream_ship, jnp.int32)))
    total = jnp.where((c != 0.0) & jnp.isfinite(c), f + c, f)
    return div_exact(total, jnp.maximum(1, jnp.asarray(ship_sts, jnp.int32)).astype(TIME_DTYPE))


# ───────────────────────────────────────────────── Φ 본체
def _phi_core(o: OrderArrays, end_s, truck_mask, vessel_gt, vessel_idle_s, vessel_mask,
              yc_extra_move_s, rehandles) -> PhiArrays:
    """모든 인자가 배열인 순수 함수 — `jax.vmap(_phi_core)` 가 배치판이다."""
    end_s = jnp.asarray(end_s, TIME_DTYPE)
    a, oo = o.gate_in_s, o.gate_out_s

    # ── 항1 원료: 검열 턴타임 (state 판 재사용) + `A == end` 경계 보정 (머리말)
    tt = censored_turn_time_s(o, end_s)
    tt = jnp.where(jnp.isnan(tt) & (a == end_s), 0.0, tt)
    in_sample = truck_mask & ~jnp.isnan(tt)                      # v5: A 있음 · A ≤ end
    tt0 = jnp.where(in_sample, tt, 0.0)

    wait = _sum_in_order(truck_wait_krw(tt0), in_sample)         # v5 `wait += truck_wait_krw(tt)` 순서
    n = jnp.sum(in_sample.astype(jnp.int32))
    censored = jnp.sum((in_sample & ((oo >= EMPTY_TIME) | (oo > end_s))).astype(jnp.int32))
    over = jnp.sum((in_sample & (tt0 > 3600.0)).astype(jnp.int32))
    tt_sum = _sum_in_order(tt0, in_sample)                       # v5 `tt_sum += tt` 순서
    n_f = n.astype(TIME_DTYPE)
    has = n > 0
    mean = jnp.where(has, div_exact(tt_sum, jnp.maximum(n_f, 1.0)), 0.0)
    over_ratio = jnp.where(has, div_exact(over.astype(TIME_DTYPE), jnp.maximum(n_f, 1.0)), 0.0)

    # ── 분위수: 표본 아닌 칸을 +inf 로 밀어 정렬 → 앞 n 칸이 v5 `samples.sort()` 와 같다
    srt = jnp.sort(jnp.where(in_sample, tt, jnp.inf))

    def q(p: float):
        if srt.shape[0] == 0:                                    # 오더 칸이 0 — 표본이 있을 수 없다
            return jnp.asarray(0.0, TIME_DTYPE)
        idx = jnp.minimum(n - 1, jnp.floor(p * n_f).astype(jnp.int32))   # int(p * len(samples))
        return jnp.where(has, srt[jnp.maximum(idx, 0)], 0.0)

    # ── 항4: 표 순서대로 `vessel += vessel_idle_krw(gt, idle)`
    vessel = _sum_in_order(vessel_idle_krw(vessel_gt, vessel_idle_s), vessel_mask)
    # ── 항2·항3
    move = _div_c(mul_exact(KRW_YC_MOVE_HOUR, jnp.maximum(0.0, jnp.asarray(yc_extra_move_s, TIME_DTYPE))),
                  3600.0)
    rehab = mul_exact(KRW_REHANDLE_EACH,
                      jnp.maximum(0, jnp.asarray(rehandles, jnp.int32)).astype(TIME_DTYPE))
    total = sum_seq([wait, move, rehab, vessel])                 # v5 `wait + move + rehandle + vessel`

    return PhiArrays(wait=wait, move=move, rehandle=rehab, vessel=vessel, total=total,
                     n_trucks=n, n_censored=censored, mean_turn_time_s=mean, over_ratio=over_ratio,
                     p50_turn_time_s=q(0.50), p90_turn_time_s=q(0.90), p99_turn_time_s=q(0.99))


def _fill(o: OrderArrays, truck_mask, vessel_gt, vessel_idle_s, vessel_mask, batch: int | None):
    """빈 선택 인자를 채운다. 배치면 앞축 B 를 붙인다."""
    lead = () if batch is None else (batch,)
    if truck_mask is None:
        truck_mask = o.is_external                                # v5 기록 사전 = 외부트럭 오더
    if vessel_gt is None:
        vessel_gt = jnp.zeros(lead + (1,), TIME_DTYPE)
        vessel_idle_s = jnp.zeros(lead + (1,), TIME_DTYPE)
        vessel_mask = jnp.zeros(lead + (1,), jnp.bool_)
    else:
        vessel_gt = jnp.asarray(vessel_gt, TIME_DTYPE)
        vessel_idle_s = jnp.asarray(vessel_idle_s, TIME_DTYPE)
        vessel_mask = (jnp.ones(vessel_gt.shape, jnp.bool_) if vessel_mask is None
                       else jnp.asarray(vessel_mask, jnp.bool_))
    return truck_mask, vessel_gt, vessel_idle_s, vessel_mask


def terminal_cost_krw(o: OrderArrays, end_s, *, vessel_gt=None, vessel_idle_s=None, vessel_mask=None,
                      yc_extra_move_s=0.0, rehandles=0, truck_mask=None) -> PhiArrays:
    """v5 `terminal_cost_krw(records, end_s=, vessel_idle=, yc_extra_move_s=, rehandles=)` 의 배열판.

    `o`               : 오더 배열 — `gate_in_s`(A)·`gate_out_s`(O) 가 v5 기록의 다섯 시각 열, 없으면 +inf.
                        **배열 순서 = v5 기록 사전의 삽입 순서**여야 원화 합이 비트 일치한다.
    `end_s`           : 평가창 오른쪽 끝 T — 미완료 트럭을 여기서 검열
    `vessel_gt`/`vessel_idle_s`/`vessel_mask` : (V,) 배별 (GT, 유휴 초) 표 — v5 dict 의 순서대로
    `yc_extra_move_s` : 터미널 YC 빈 주행 초 (계수기)   `rehandles` : 재조작 수 (계수기)
    `truck_mask`      : 어느 오더가 "기록" 인가 — 기본 `o.is_external`
    """
    args = _fill(o, truck_mask, vessel_gt, vessel_idle_s, vessel_mask, None)
    return _phi_core(o, end_s, *args, yc_extra_move_s, rehandles)


def terminal_cost_krw_batch(o: OrderArrays, end_s, *, vessel_gt=None, vessel_idle_s=None,
                            vessel_mask=None, yc_extra_move_s=None, rehandles=None,
                            truck_mask=None) -> PhiArrays:
    """세계 B 개를 한 번에 — 잎마다 앞축 B (오더 (B,N) · end_s (B,) · 본선 표 (B,V) · 계수기 (B,))."""
    b = int(o.gate_in_s.shape[0])
    args = _fill(o, truck_mask, vessel_gt, vessel_idle_s, vessel_mask, b)
    yc = jnp.zeros((b,), TIME_DTYPE) if yc_extra_move_s is None else jnp.asarray(yc_extra_move_s, TIME_DTYPE)
    rh = jnp.zeros((b,), jnp.int32) if rehandles is None else jnp.asarray(rehandles, jnp.int32)
    return jax.vmap(_phi_core)(o, jnp.asarray(end_s, TIME_DTYPE), *args, yc, rh)


# ───────────────────────────────────────────────── 호스트 변환 (v5 기록 → 배열)
def orders_from_records(records, *, n_max: int | None = None) -> OrderArrays:
    """v5 `{doc_key: ExecutionRecord}` → `OrderArrays`. **사전 삽입 순서가 배열 순서**다 (합산 순서).

    다섯 시각의 `None` 은 +inf. 기록은 전부 외부트럭이라 `is_external=True`, `block=0` 을 준다.
    `n_max` 로 칸을 더 잡으면 남는 칸은 빈 오더(+inf · is_external=False)다 — 배치 정렬용.
    """
    n = len(records) if n_max is None else int(n_max)
    if n < len(records):
        raise ValueError(f"칸 {n} < 기록 {len(records)} — 조용히 자르지 않는다")
    cols = {k: np.full(n, np.inf) for k in ("notice", "a", "b", "s", "c", "o")}
    ext, blk = np.zeros(n, bool), np.full(n, -1, np.int32)
    fields = (("notice", "copino_notice_s"), ("a", "gate_in_s"), ("b", "block_in_s"),
              ("s", "service_start_s"), ("c", "job_done_s"), ("o", "gate_out_s"))
    for i, rec in enumerate(records.values()):
        for col, name in fields:
            v = getattr(rec, name, None)
            if v is not None:
                cols[col][i] = float(v)
        ext[i], blk[i] = True, 0
    f = lambda k: jnp.asarray(cols[k], TIME_DTYPE)
    return empty_orders(n)._replace(
        block=jnp.asarray(blk), is_external=jnp.asarray(ext),
        notice_s=f("notice"), gate_in_s=f("a"), block_in_s=f("b"), service_s=f("s"),
        done_s=f("c"), gate_out_s=f("o"))
