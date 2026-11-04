"""자원 예약표를 **배열로** ([[YR-327]] 조각 1 · v5 `world/integrated/reservation.py` 의 배열판).

v5 는 `ReservationTable` 이 세 개의 dict 를 든다 — 크레인별 예약(`_by_crane`)·
작업토큰→크레인(`_tokens`)·예약 없는 크레인의 현재 bay(`_idle_pos`). 여기서는 전부
**고정 길이 배열**이고, 예약이 있는지는 `active` 마스크가 말한다.

    _by_crane[cid] = Reservation(...)   →  active[k]·token[k]·lo[k]·hi[k]·lane[k]·release_at[k]·slots[k]
    _tokens[token] = cid                →  token_owner[n]  (n = 오더 번호, -1 = 아무도 안 잡음)
    _idle_pos[cid] = bay                →  idle_pos[k]      (+inf = 등록 안 됨 → 장벽 없음)
    슬롯 frozenset {(bay,row)}          →  slots[k] (B,R) bool, 0-based (bay-1, row-1)

■ ★거절 사유는 **5-lock 순서 고정** (reservation.py:86-99) — 코드 표
    0 없음(성공)  1 DOUBLE_RESERVE  2 DUP_JOB  3 LANE_CONFLICT  4 CRANE_INTERFERENCE  5 SLOT_CONFLICT
  v5 는 위에서부터 첫 번째로 걸리는 이유 하나만 돌려준다. 다섯 조건을 **전부 계산한 뒤**
  `where` 사다리로 같은 순서를 재현한다 — 조건 둘이 동시에 참이어도 답이 같아야
  1·2·3차 방어가 v5 와 같은 소스가 된다(YR-037).

■ ★통로(corridor) 겹침은 v5 식 그대로 — `not (a.hi + gap <= b.lo or b.hi + gap <= a.lo)`
  (reservation.py:21-22). `+ gap` 은 float64 덧셈 하나라 파이썬 float 과 비트 동일하다.
  예약 없는 크레인은 현재 bay 를 **점 통로 [p, p]** 로 두어 관통을 막는다(YR-091, 70-76행).
  이때 자기 자신과 **예약이 있는 크레인은 뺀다** — 예약이 있으면 통로 검사가 이미 봤다.

■ 빈 칸 규약
  active False 인 행의 lo/hi/lane/token/release_at/slots 는 **읽지 않는다** (모든 검사가
  active 로 마스크). release 는 그래도 token -1 · lane -1 · slots False · release_at +inf 로
  되돌린다 — 진단 시 눈으로 봐도 v5 의 "pop 된 상태" 와 같게.

■ 음수 색인 주의
  `arr.at[-1].set(v)` 는 JAX 에서 **마지막 칸**을 고친다. 토큰 -1(없음)을 그대로 색인에
  쓰면 엉뚱한 오더가 잡힌 것으로 기록된다. 그래서 token_owner 갱신은 색인이 아니라
  `(arange(N) == token) & valid` 마스크로 한다.

■ 자료형은 `state.ReservationArrays` **하나뿐** (조각 2 통일)
  조각 1 에는 이 파일에 같은 이름의 사본이 있어 `world.res` 와 pytree 형이 달랐다. 지금은
  state.py 의 클래스를 import 해 그대로 받고 그대로 돌려준다 — 엔진의 `_as_res` 변환은 항등이다.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .events import TIME_DTYPE
from .state import EMPTY_ID, EMPTY_TIME, ReservationArrays, empty_reservations

__all__ = ["ReservationArrays", "empty_reservations", "COORD_DTYPE",
           "OK", "DOUBLE_RESERVE", "DUP_JOB", "LANE_CONFLICT", "CRANE_INTERFERENCE", "SLOT_CONFLICT",
           "REASON_TO_CODE", "CODE_TO_REASON", "corridor_overlaps", "reject_code", "can_reserve",
           "reject_code_over_orders", "reserve", "release", "set_idle_position", "reserved_slots",
           "lane_occupancy", "orphan_count"]

#: ★거절 코드 — v5 문자열 이유와 1:1 (순서가 곧 판정 순서)
OK, DOUBLE_RESERVE, DUP_JOB, LANE_CONFLICT, CRANE_INTERFERENCE, SLOT_CONFLICT = 0, 1, 2, 3, 4, 5
REASON_TO_CODE = {None: OK, "DOUBLE_RESERVE": DOUBLE_RESERVE, "DUP_JOB": DUP_JOB,
                  "LANE_CONFLICT": LANE_CONFLICT, "CRANE_INTERFERENCE": CRANE_INTERFERENCE,
                  "SLOT_CONFLICT": SLOT_CONFLICT}
CODE_TO_REASON = {v: k for k, v in REASON_TO_CODE.items()}

#: 좌표 dtype — 시각과 같이 float64 (x64). bay 좌표·gap 은 v5 에서 파이썬 float 이다.
COORD_DTYPE = TIME_DTYPE


# ───────────────────────────────────────────────── 판정
def _overlaps(a_lo, a_hi, b_lo, b_hi, gap):
    """v5 `Corridor.overlaps` (reservation.py:21-22) 그대로 — a 가 기존, b 가 질의.

        not (a.hi + gap <= b.lo  or  b.hi + gap <= a.lo)
    """
    return ~((a_hi + gap <= b_lo) | (b_hi + gap <= a_lo))


#: 공개 이름 — 엔진의 불변식 검사(engine_step.check_invariants, v5 1123-1137행 CORRIDOR_OVERLAP)가 같은 식을 쓴다.
corridor_overlaps = _overlaps


def reject_code(res: ReservationArrays, k, token, lo, hi, lane, slots, gap) -> jnp.ndarray:
    """크레인 k 가 (token, [lo,hi], lane, slots) 를 예약하려 할 때의 거절 코드 (0 = 성공).

    v5 `ReservationTable.reject_reason` (reservation.py:86-99) 을 다섯 조건 **전부 계산 →
    where 사다리** 로 옮긴 것. 순서: DOUBLE → DUP → LANE → INTERF → SLOT.

    인자: k int32 · token int32(-1 = None) · lo/hi f64 · lane int32(-1 = None) ·
          slots (B,R) bool · gap f64(safety_gap_bay). 반환 () int32.
    """
    K = res.active.shape[0]
    ks = jnp.arange(K, dtype=jnp.int32)
    lo = jnp.asarray(lo, COORD_DTYPE)
    hi = jnp.asarray(hi, COORD_DTYPE)
    gap = jnp.asarray(gap, COORD_DTYPE)
    token = jnp.asarray(token, jnp.int32)
    lane = jnp.asarray(lane, jnp.int32)

    # 1 DOUBLE_RESERVE — 이 크레인에 이미 예약이 있다 (89행)
    c1 = res.active[k]
    # 2 DUP_JOB — 토큰을 누가 잡고 있다 (91행; None 토큰은 건너뜀). 음수 색인 금지 → 마스크
    n_idx = jnp.arange(res.token_owner.shape[0], dtype=jnp.int32)
    c2 = (token >= 0) & jnp.any((n_idx == token) & (res.token_owner >= 0))
    # 3 LANE_CONFLICT — 그 레인을 가진 예약이 하나라도 (93행; lane_owner 는 자기 제외 안 함 —
    #   자기 예약이 있으면 c1 이 먼저 잡으니 결과는 같다)
    c3 = (lane >= 0) & jnp.any(res.active & (res.lane == lane))
    # 4 CRANE_INTERFERENCE — 다른 크레인의 통로(64-69행) 또는 예약 없는 다른 크레인의 점 장벽(70-76행)
    others = res.active & (ks != k)
    ov_res = _overlaps(res.lo, res.hi, lo, hi, gap)
    ov_idle = _overlaps(res.idle_pos, res.idle_pos, lo, hi, gap) & ~res.active & (ks != k)
    c4 = jnp.any(others & ov_res) | jnp.any(ov_idle)
    # 5 SLOT_CONFLICT — 활성 예약 칸과 겹침 (97행; reserved_slots 는 전 예약 합집합)
    c5 = jnp.any(res.slots & res.active[:, None, None] & slots[None])

    return jnp.where(c1, DOUBLE_RESERVE,
           jnp.where(c2, DUP_JOB,
           jnp.where(c3, LANE_CONFLICT,
           jnp.where(c4, CRANE_INTERFERENCE,
           jnp.where(c5, SLOT_CONFLICT, OK))))).astype(jnp.int32)


def can_reserve(res: ReservationArrays, k, token, lo, hi, lane, slots, gap) -> jnp.ndarray:
    """v5 `can_reserve` — 성공 여부만 (예약 없이). () bool."""
    return reject_code(res, k, token, lo, hi, lane, slots, gap) == OK


#: 오더 축 배치판 — 한 크레인 k 가 오더 N 개 각각을 예약할 수 있나. (N,) int32 코드.
#: 입력: token (N,), lo (N,), hi (N,), lane (N,), slots (N,B,R). res·k·gap 은 공유.
reject_code_over_orders = jax.vmap(reject_code, in_axes=(None, None, 0, 0, 0, 0, 0, None))


# ───────────────────────────────────────────────── 갱신
def reserve(res: ReservationArrays, k, token, lo, hi, lane, slots, release_at, gap):
    """예약을 시도한다 → `(res', code)`. code≠0 이면 res 는 **그대로**다.

    v5 `reserve` (reservation.py:101-108) 는 거절 시 예외를 던진다 — 여기서는 코드를
    돌려주고 호출자가 violation 비트(16)를 세운다. 성공 시:
      _by_crane[cid] = r        →  active[k]=T · token/lo/hi/lane/release_at/slots[k] 기록
      _tokens[token] = cid      →  token_owner[token] = k   (token ≥ 0 일 때만, 107-108행)
    """
    code = reject_code(res, k, token, lo, hi, lane, slots, gap)
    ok = code == OK
    token = jnp.asarray(token, jnp.int32)
    n_idx = jnp.arange(res.token_owner.shape[0], dtype=jnp.int32)
    take = ok & (token >= 0) & (n_idx == token)
    new = ReservationArrays(
        active=res.active.at[k].set(True),
        token=res.token.at[k].set(token),
        lo=res.lo.at[k].set(jnp.asarray(lo, COORD_DTYPE)),
        hi=res.hi.at[k].set(jnp.asarray(hi, COORD_DTYPE)),
        lane=res.lane.at[k].set(jnp.asarray(lane, jnp.int32)),
        release_at=res.release_at.at[k].set(jnp.asarray(release_at, TIME_DTYPE)),
        slots=res.slots.at[k].set(jnp.asarray(slots, bool)),
        token_owner=jnp.where(take, jnp.asarray(k, jnp.int32), res.token_owner),
        idle_pos=res.idle_pos)
    out = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, res)
    return out, code


def release(res: ReservationArrays, k) -> ReservationArrays:
    """크레인 k 의 예약을 푼다 (reservation.py:114-117). 예약이 없으면 무동작.

    v5: `_by_crane.pop(cid)`; 토큰이 있으면 `_tokens.pop(token)`. idle_pos 는 건드리지
    않는다 — 엔진이 완료 처리에서 `set_idle_position` 을 따로 부른다(engine.py:901-903).
    """
    had = res.active[k]
    tok = res.token[k]
    n_idx = jnp.arange(res.token_owner.shape[0], dtype=jnp.int32)
    drop = had & (tok >= 0) & (n_idx == tok)
    new = ReservationArrays(
        active=res.active.at[k].set(False),
        token=res.token.at[k].set(EMPTY_ID),
        lo=res.lo.at[k].set(jnp.asarray(0.0, COORD_DTYPE)),
        hi=res.hi.at[k].set(jnp.asarray(0.0, COORD_DTYPE)),
        lane=res.lane.at[k].set(EMPTY_ID),
        release_at=res.release_at.at[k].set(jnp.asarray(EMPTY_TIME, TIME_DTYPE)),
        slots=res.slots.at[k].set(jnp.zeros(res.slots.shape[1:], bool)),
        token_owner=jnp.where(drop, EMPTY_ID, res.token_owner),
        idle_pos=res.idle_pos)
    return jax.tree_util.tree_map(lambda a, b: jnp.where(had, a, b), new, res)


def set_idle_position(res: ReservationArrays, k, bay) -> ReservationArrays:
    """YR-091 상시 장벽 갱신 (reservation.py:80-81) — 크레인 k 의 현재 bay 를 적는다."""
    return res._replace(idle_pos=res.idle_pos.at[k].set(jnp.asarray(bay, COORD_DTYPE)))


# ───────────────────────────────────────────────── 읽기 (plan·refresh_rates 용)
def reserved_slots(res: ReservationArrays) -> jnp.ndarray:
    """전 활성 예약의 칸 합집합 (reservation.py:50-54) — (B,R) bool. plan 의 excluded 원천."""
    return jnp.any(res.slots & res.active[:, None, None], axis=0)


def lane_occupancy(res: ReservationArrays, n_lanes: int) -> jnp.ndarray:
    """레인별 점유 (engine.py:819 `occ = {r.lane_id for r in active() if r.lane_id}`) — (L,) bool."""
    ls = jnp.arange(n_lanes, dtype=jnp.int32)
    return jnp.any(res.active[None, :] & (res.lane[None, :] == ls[:, None]), axis=1)


def orphan_count(res: ReservationArrays) -> jnp.ndarray:
    """활성 예약 수 (reservation.py:119-120) — 불변식 검사용."""
    return jnp.sum(res.active).astype(jnp.int32)
