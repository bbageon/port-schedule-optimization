"""전역 트럭 원장 — v5 `MultiBlockTerminal.ledger`(TerminalLedger·JobRecord) 를 배열로 ([[YR-327]] 조각 6 · key=ledger).

⚠️ **조정자 경로 밖의 참조 구현이다** (2026-09-26 검증). 조각 6 의 조정자(`gpu/multiblock.py`)가 실제로 쓰는 원장은
   `host_terminal.TerminalLedgerArrays`(정본) 와 이송·이연 때만 잠깐 만드는 `transfer_txn.TxnLedger`(브리지) 둘이다.
   여기 `TruckLedger` 를 import 하는 곳은 자기 시험(tests/v6/test_gpu_ledger.py)뿐이므로, 이 파일의 시험이 통과해도
   **사다리 경로를 보증하지 않는다**. 같은 성질(런 중 sync_locks 타이밍 · harvest 뒤 records 전 필드)은 정본 경로에서
   따로 증명한다 — 사다리 ①②는 살아 있는 v5 와, ③은 두꺼운 정답 JSON(locked 전건·오더 전열)과 대조한다.
   정리(삭제 또는 TxnLedger 로 통합)는 별도 작업으로 남긴다.

v5 `world/integrated/multiblock.py` 30-94행의 `JobRecord`/`TerminalLedger` 와 조정자가 그것을 만지는 자리
(register 63 · harvest 66-83 · a_to_o_samples_s 84-94 · reassignable 52-54 · _sync_locks 212-227 ·
admit 290-291 · defer 363-366 · commit 611-616 · route_cost_s 132/616) 를 **고정 크기 배열 + 순수 함수**로 옮긴다.

    v5 (dict[job_id] → JobRecord)                   여기 (TruckLedger — (N,) 열)
    ─────────────────────────────────────────────────────────────────────────────
    records[jid] 존재                              registered[n]
    origin_block / owner (블록 이름)                origin_block / owner (블록 번호 · -1 = 없음)
    flow (문자열 값)                               flow (state.FL_* 정수)
    version · transfer_count                       version · transfer_count
    transfer_history ((src, dst, t), …) 튜플       transfer_src / transfer_dst / transfer_t (N, H) 고정 칸 + transfer_count
    a_gate_in · b_block_arrival · c_job_done ·      a_gate_in · b_block_arrival · c_job_done · o_gate_out (+inf = None)
      o_gate_out (None = 미발생)
    locked · entry_deferrals · entry_deferred_s     locked · entry_deferrals · entry_deferred_s
    dict 삽입 순서                                  reg_seq (등록 순번 · -1 = 미등록) + n_registered
    MultiBlockTerminal.route_cost_s                 route_cost_s () — 이송 추가주행 누적 (commit 616행)

■ 오더 번호 n 은 **터미널 전역**이다 — 블록 접두어("Y05:D-00042")는 origin_block 열이 대신한다 (명세 §multiblock
  _namespace_jobs → no-op). 조정자가 (B, N) 로 쌓은 블록 세계의 어느 행이 '지금 소유한' 값인지는 `owner` 가 말하고,
  `gather_owned(x_bn, owner)` 가 그 행을 (N,) 으로 모은다. 원장 자체는 vmap 축 밖(에폭 scan 의 carry)에 있다.
■ a_to_o 표본은 n 순서(= 명단 순서, 조정자가 트럭 n 을 명단 순으로 매길 때) — v5 dict 순서가 필요하면 `registration_order`.
■ 모든 갱신은 `ok` 스칼라로 조건부 — 조정자의 prepare→validate→commit 원자 확정은 `new = f(state); state' = where(ok, new, old)`
  (명세 hard_parts ④) 이고, 여기 함수들은 `ok=False` 면 입력을 그대로 돌려준다 (롤백 공짜).
■ 검사 술어(`pre_gate_ok`·`post_gate_ok`·`validate_ok`·`defer_ok`·`can_register`)는 v5 raise 검사 가운데 **원장이 답할 수 있는
  것만** 모아 (ok, 첫 실패 코드) 로 돌려준다. 블록 상태 검사(status==PLANNED · free_slots · find_slot · 도착시각 창)는
  조정자가 자기 배열로 판정해 AND 한다 — 코드는 진단용이며 v5 예외 문장의 순서와 1:1 이 아니다(그 사이에 조정자 검사가 끼어든다).
■ 실수 연산은 v5 와 같은 식·같은 순서: `a + delta`, `end − a` 뒤 `max(0, ·)`, `route_cost += route_s` (순차), `now + 1e-6` 비교.
  x64 전제 (state.check_x64). 곱셈이 없어 mul_exact 자리가 없다.
■ 이어 돌리기: `to_numpy`/`from_numpy` 가 원장 전체를 numpy dict 로 내보내고 되살린다 (WSL 80초 세션 사이 npz 저장용).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .events import TIME_DTYPE
from .state import (EMPTY_ID, EMPTY_TIME, FL_GATE_IN, JS_ASSIGNED, JS_DONE, JS_PLANNED, JS_RUNNING,
                    JS_WAITING)

__all__ = [
    "TruckLedger", "empty_truck_ledger", "can_register", "register", "register_rows",
    "sync_locks", "harvest", "a_to_o_samples_s", "reassignable", "registration_order",
    "commit_transfer", "defer_entry", "pre_gate_ok", "post_gate_ok", "validate_ok", "defer_ok",
    "invariant_errors", "gather_owned", "to_numpy", "from_numpy", "to_v5_records", "sample_list",
    "GATE_EPS", "LR_OK", "LR_UNREGISTERED", "LR_BAD_DST", "LR_NOT_REASSIGNABLE", "LR_TRANSFER_CAP",
    "LR_ALREADY_GATE_IN", "LR_BEFORE_GATE_IN", "LR_A_MISSING", "LR_STALE", "LR_DELTA_NONPOS",
    "LR_DEFER_CAP", "LR_DUPLICATE", "LR_NAMES",
]

#: v5 multiblock.py 316·390·438행 — gate-in 창 판정 허용오차 (`now + 1e-6`)
GATE_EPS = 1e-6

#: 술어 실패 코드 (0 = 통과). v5 의 TransferError 문장에 대응 — 진단용
LR_OK, LR_UNREGISTERED, LR_BAD_DST, LR_NOT_REASSIGNABLE, LR_TRANSFER_CAP = 0, 1, 2, 3, 4
LR_ALREADY_GATE_IN, LR_BEFORE_GATE_IN, LR_A_MISSING, LR_STALE = 5, 6, 7, 8
LR_DELTA_NONPOS, LR_DEFER_CAP, LR_DUPLICATE = 9, 10, 11
LR_NAMES: dict[int, str] = {
    LR_OK: "OK", LR_UNREGISTERED: "미등록", LR_BAD_DST: "수신 블록 부적격", LR_NOT_REASSIGNABLE: "lock/자격 위반",
    LR_TRANSFER_CAP: "이송 상한 초과", LR_ALREADY_GATE_IN: "이미 gate-in", LR_BEFORE_GATE_IN: "gate-in 전 (창 밖)",
    LR_A_MISSING: "gate-in 결측", LR_STALE: "version/owner 변경 (stale quote)", LR_DELTA_NONPOS: "이연량은 양수여야 함",
    LR_DEFER_CAP: "이연 상한 초과", LR_DUPLICATE: "이미 등록된 작업",
}


# ─────────────────────────────────────────────────── 원장
class TruckLedger(NamedTuple):
    """터미널 전역 작업 원장 (N 행) — v5 JobRecord 의 열판. 빈 시각은 +inf, 빈 번호는 -1."""

    registered: jnp.ndarray        # (N,) bool   records 에 있나
    origin_block: jnp.ndarray      # (N,) int32  TOS 최초 배정 (불변)
    owner: jnp.ndarray             # (N,) int32  현재 실행 블록
    flow: jnp.ndarray              # (N,) int32  state.FL_*
    version: jnp.ndarray           # (N,) int32  낙관적 동시성
    transfer_count: jnp.ndarray    # (N,) int32
    transfer_src: jnp.ndarray      # (N,H) int32  이송 이력 (src 블록)
    transfer_dst: jnp.ndarray      # (N,H) int32  (dst 블록)
    transfer_t: jnp.ndarray        # (N,H) f64    (확정 시각 = 조정자 now)
    a_gate_in: jnp.ndarray         # (N,) f64  A — 터미널 보유 (이송 무관)
    b_block_arrival: jnp.ndarray   # (N,) f64  B
    c_job_done: jnp.ndarray        # (N,) f64  C
    o_gate_out: jnp.ndarray        # (N,) f64  O
    locked: jnp.ndarray            # (N,) bool  block-in/배정 이후 = 재배정 금지
    entry_deferrals: jnp.ndarray   # (N,) int32  진입 전 이연 횟수
    entry_deferred_s: jnp.ndarray  # (N,) f64    이연 누적
    reg_seq: jnp.ndarray           # (N,) int32  등록 순번 (v5 dict 삽입 순서; -1 = 미등록)
    n_registered: jnp.ndarray      # ()   int32
    route_cost_s: jnp.ndarray      # ()   f64   이송 추가주행 누적 (MultiBlockTerminal.route_cost_s)
    overflow: jnp.ndarray          # ()   int32 이송 이력 칸(H) 부족 수 — 0 이어야 정상

    @property
    def n(self) -> int:
        return int(self.registered.shape[0])

    @property
    def h(self) -> int:
        return int(self.transfer_src.shape[1])


def empty_truck_ledger(n: int, h_max: int = 1) -> TruckLedger:
    """빈 원장 N 행 · 이송 이력 H 칸 (pre-gate 상한 max_transfers=1 이 기본)."""
    n, h = int(n), max(1, int(h_max))
    return TruckLedger(
        registered=jnp.zeros((n,), jnp.bool_),
        origin_block=jnp.full((n,), EMPTY_ID, jnp.int32), owner=jnp.full((n,), EMPTY_ID, jnp.int32),
        flow=jnp.full((n,), EMPTY_ID, jnp.int32),
        version=jnp.zeros((n,), jnp.int32), transfer_count=jnp.zeros((n,), jnp.int32),
        transfer_src=jnp.full((n, h), EMPTY_ID, jnp.int32), transfer_dst=jnp.full((n, h), EMPTY_ID, jnp.int32),
        transfer_t=jnp.full((n, h), EMPTY_TIME, TIME_DTYPE),
        a_gate_in=jnp.full((n,), EMPTY_TIME, TIME_DTYPE), b_block_arrival=jnp.full((n,), EMPTY_TIME, TIME_DTYPE),
        c_job_done=jnp.full((n,), EMPTY_TIME, TIME_DTYPE), o_gate_out=jnp.full((n,), EMPTY_TIME, TIME_DTYPE),
        locked=jnp.zeros((n,), jnp.bool_),
        entry_deferrals=jnp.zeros((n,), jnp.int32), entry_deferred_s=jnp.zeros((n,), TIME_DTYPE),
        reg_seq=jnp.full((n,), EMPTY_ID, jnp.int32), n_registered=jnp.zeros((), jnp.int32),
        route_cost_s=jnp.zeros((), TIME_DTYPE), overflow=jnp.zeros((), jnp.int32))


# ─────────────────────────────────────────────────── 작은 도구
def _i32(x):
    return jnp.asarray(x, jnp.int32)


def _f64(x):
    return jnp.asarray(x, TIME_DTYPE)


def _clip_row(arr, n):
    """n 이 -1 이면 `.at[-1]` 이 마지막 칸을 고치므로 clip 해 두고 유효성은 where 로 가른다."""
    return jnp.clip(_i32(n), 0, arr.shape[0] - 1)


def _set1(arr, n, v, ok):
    """arr[n] = v (ok 일 때만)."""
    nc = _clip_row(arr, n)
    return arr.at[nc].set(jnp.where(ok, jnp.asarray(v, arr.dtype), arr[nc]))


def _row_valid(L: TruckLedger, n):
    n = _i32(n)
    return (n >= 0) & (n < L.n)


def gather_owned(x_bn, owner, fill=None):
    """(B, N) 블록별 열에서 행 n 의 **소유 블록** 값을 (N,) 으로 — v5 `sim.jobs[jid]` (소유 블록에만 있다) 조회.

    owner < 0 (미등록·소유자 없음) 이면 fill (기본: 그 dtype 의 빈값 — 실수 +inf · 정수 -1 · bool False)."""
    x_bn = jnp.asarray(x_bn)
    B = x_bn.shape[0]
    owner = _i32(owner)
    n_idx = jnp.arange(owner.shape[0], dtype=jnp.int32)
    got = x_bn[jnp.clip(owner, 0, B - 1), n_idx]
    if fill is None:
        if jnp.issubdtype(x_bn.dtype, jnp.floating):
            fill = EMPTY_TIME
        elif x_bn.dtype == jnp.bool_:
            fill = False
        else:
            fill = EMPTY_ID
    return jnp.where(owner >= 0, got, jnp.asarray(fill, x_bn.dtype))


# ─────────────────────────────────────────────────── 등록 (register 63 · __init__ 137-140 · admit 290-291)
def can_register(L: TruckLedger, n):
    """admit 257행 `jid in self.ledger.records` → 거절. (ok, code) — 행 밖도 거절."""
    v = _row_valid(L, n)
    dup = L.registered[_clip_row(L.registered, n)]
    ok = v & ~dup
    code = jnp.where(ok, LR_OK, jnp.where(v, LR_DUPLICATE, LR_UNREGISTERED)).astype(jnp.int32)
    return ok, code


def register(L: TruckLedger, n, origin_block, owner, flow, a_gate_in, ok=True) -> TruckLedger:
    """행 하나 등록 — v5 `JobRecord(job_id, origin_block, owner, flow, a_gate_in)` + `records[jid] = rec`.

    a_gate_in 은 +inf 로 None (초기 본선 작업 137-140행 `getattr(j, 'actual_gate_in', None)`). 이미 등록된 행은
    (v5 가 admit 에서 먼저 거절하므로) 여기서도 바꾸지 않는다 — 조정자는 `can_register` 로 먼저 판정한다.
    """
    ok = jnp.asarray(ok, jnp.bool_) & can_register(L, n)[0]
    seq = L.n_registered
    return L._replace(
        registered=_set1(L.registered, n, True, ok),
        origin_block=_set1(L.origin_block, n, origin_block, ok), owner=_set1(L.owner, n, owner, ok),
        flow=_set1(L.flow, n, flow, ok),
        version=_set1(L.version, n, 0, ok), transfer_count=_set1(L.transfer_count, n, 0, ok),
        a_gate_in=_set1(L.a_gate_in, n, a_gate_in, ok),
        b_block_arrival=_set1(L.b_block_arrival, n, EMPTY_TIME, ok),
        c_job_done=_set1(L.c_job_done, n, EMPTY_TIME, ok), o_gate_out=_set1(L.o_gate_out, n, EMPTY_TIME, ok),
        locked=_set1(L.locked, n, False, ok),
        entry_deferrals=_set1(L.entry_deferrals, n, 0, ok), entry_deferred_s=_set1(L.entry_deferred_s, n, 0.0, ok),
        reg_seq=_set1(L.reg_seq, n, seq, ok),
        n_registered=jnp.where(ok, seq + 1, seq).astype(jnp.int32))


def register_rows(L: TruckLedger, mask, origin_block, owner, flow, a_gate_in) -> TruckLedger:
    """여러 행을 **n 오름차순으로** 한 번에 등록 (구성 시 초기 작업 137-140행 — 블록 순 × sim.jobs 순 = n 순일 때).

    이미 등록된 행은 건너뛴다. 등록 순번은 n 순으로 이어 붙는다."""
    mask = jnp.asarray(mask, jnp.bool_) & ~L.registered
    seq0 = L.n_registered
    seq = seq0 + jnp.cumsum(mask.astype(jnp.int32)) - 1
    inf = jnp.full_like(L.a_gate_in, EMPTY_TIME)

    def w(new, old):
        return jnp.where(mask, jnp.asarray(new, old.dtype), old)

    return L._replace(
        registered=L.registered | mask,
        origin_block=w(origin_block, L.origin_block), owner=w(owner, L.owner), flow=w(flow, L.flow),
        version=w(0, L.version), transfer_count=w(0, L.transfer_count),
        a_gate_in=w(a_gate_in, L.a_gate_in), b_block_arrival=w(inf, L.b_block_arrival),
        c_job_done=w(inf, L.c_job_done), o_gate_out=w(inf, L.o_gate_out),
        locked=w(False, L.locked), entry_deferrals=w(0, L.entry_deferrals), entry_deferred_s=w(0.0, L.entry_deferred_s),
        reg_seq=w(seq, L.reg_seq),
        n_registered=(seq0 + jnp.sum(mask.astype(jnp.int32))).astype(jnp.int32))


# ─────────────────────────────────────────────────── 런 중 lock (_sync_locks 212-227)
def sync_locks(L: TruckLedger, rows, status, block_in_s, tl_mask) -> TruckLedger:
    """v5 `_sync_locks(sim)` — ReviewEpoch 에 park 한 블록의 작업을 훑어 **처음** status≠PLANNED 인 행을 잠근다.

    rows      (N,) bool  이번에 훑는 행 (= 그 블록 sim.jobs = owner==b; 전 블록을 한 번에 훑으면 all True)
    status    (N,) int32 소유 블록의 오더 status (state.JS_*)
    block_in_s(N,) f64   소유 블록 장부의 B (time_contract TruckTimes.block_arrival; +inf = None)
    tl_mask   (N,) bool  소유 블록 시간 장부에 항목이 있나 (`tl.records.get(jid)` — 외부트럭·exit_travel 있음)

    잠그는 순간에만 B 를 옮겨 적는다 (226-227행) — 이미 잠긴 행은 건드리지 않는다 (221행 continue).
    행 단위 독립·멱등이라 블록별로 따로 불러도, 에폭에 전 블록을 한 번에 불러도 같은 답이다 (park 뒤 상태 불변).
    """
    rows = jnp.asarray(rows, jnp.bool_)
    newly = L.registered & rows & ~L.locked & (_i32(status) != JS_PLANNED)          # 221-224행
    b_in = _f64(block_in_s)
    take_b = newly & jnp.asarray(tl_mask, jnp.bool_) & (b_in < EMPTY_TIME)           # 225-227행
    return L._replace(locked=L.locked | newly,
                      b_block_arrival=jnp.where(take_b, b_in, L.b_block_arrival))


# ─────────────────────────────────────────────────── 종료 흡수 (harvest 66-82)
def harvest(L: TruckLedger, status, block_in_s, done_s, gate_out_s, tl_mask) -> TruckLedger:
    """v5 `harvest(blocks)` — 소유 블록의 실현 시각 B·C·O 를 흡수하고 WAITING/ASSIGNED/RUNNING/DONE 이면 잠근다.

    입력은 전부 (N,) '소유 블록의 값' (gather_owned). tl_mask 가 거짓인 행(본선 작업 — 블록 장부에 없음)은 시각을
    안 바꾼다 (74-79행 `r is None`). None 인 값은 기존을 유지 (76-79행 `if … is not None else rec.…`).
    ★RELEASED·CANCELLED 는 잠그지 않는다 — 마지막 에폭 뒤에 해제된 본선 작업은 v5 에서도 unlocked 로 남는다."""
    reg = L.registered
    tl = reg & jnp.asarray(tl_mask, jnp.bool_)
    b_in, c_dn, o_out = _f64(block_in_s), _f64(done_s), _f64(gate_out_s)
    st = _i32(status)
    lock = reg & ((st == JS_WAITING) | (st == JS_ASSIGNED) | (st == JS_RUNNING) | (st == JS_DONE))   # 80-82행
    return L._replace(
        b_block_arrival=jnp.where(tl & (b_in < EMPTY_TIME), b_in, L.b_block_arrival),
        c_job_done=jnp.where(tl & (c_dn < EMPTY_TIME), c_dn, L.c_job_done),
        o_gate_out=jnp.where(tl & (o_out < EMPTY_TIME), o_out, L.o_gate_out),
        locked=L.locked | lock)


# ─────────────────────────────────────────────────── 표본·자격 (84-94 · 52-54)
def a_to_o_samples_s(L: TruckLedger, end) -> jnp.ndarray:
    """터미널 턴타임 A→O — v5 `a_to_o_samples_s(end)` 86-93행. (N,) f64, 표본 아닌 행(미등록·A 없음)은 NaN.

    O 확정 → O − A ; 아니면 max(0, end − A) (미완료가 이득 보지 않게). 순서 = n (명단 순) — dict 순서는 registration_order."""
    end = _f64(end)
    has_a = L.registered & (L.a_gate_in < EMPTY_TIME)
    has_o = L.o_gate_out < EMPTY_TIME
    return jnp.where(has_a, jnp.where(has_o, L.o_gate_out - L.a_gate_in,
                                      jnp.maximum(0.0, end - L.a_gate_in)), jnp.nan)


def reassignable(L: TruckLedger) -> jnp.ndarray:
    """v5 JobRecord.reassignable 52-54행 — not locked · GATE_IN · B 없음. (N,) bool (미등록 행 False)."""
    return L.registered & ~L.locked & (L.flow == FL_GATE_IN) & (L.b_block_arrival >= EMPTY_TIME)


def registration_order(L: TruckLedger) -> jnp.ndarray:
    """v5 `records` dict 순회 순서 = 등록 순번 오름차순의 행 번호 (N,) int32 — 앞 n_registered 칸이 유효, 뒤는 미등록 행."""
    key = jnp.where(L.registered, L.reg_seq, jnp.iinfo(jnp.int32).max)
    return jnp.argsort(key, stable=True).astype(jnp.int32)


# ─────────────────────────────────────────────────── 이송 확정 (commit 611-616) · 이연 (defer 363-366)
def commit_transfer(L: TruckLedger, n, dst, now, route_s, ok=True) -> TruckLedger:
    """commit() 의 원장 부분 — owner=dst · version+1 · transfer_count+1 · 이력 (src, dst, now) 추가 · route_cost += route_s.

    src 는 지금 owner (615행 `txn.src` = prepare 시점 owner; validate 가 owner 불변을 보장). 이력 칸 H 가 다 찼으면
    overflow+1 (조용히 자르지 않는다 — 이력은 못 적지만 count·owner·route 는 v5 처럼 진행)."""
    ok = jnp.asarray(ok, jnp.bool_) & _row_valid(L, n)
    nc = _clip_row(L.owner, n)
    cnt = L.transfer_count[nc]
    slot_ok = ok & (cnt < L.h)
    hc = jnp.clip(cnt, 0, L.h - 1)
    src = L.owner[nc]
    return L._replace(
        owner=_set1(L.owner, n, dst, ok),
        version=_set1(L.version, n, L.version[nc] + 1, ok),
        transfer_count=_set1(L.transfer_count, n, cnt + 1, ok),
        transfer_src=L.transfer_src.at[nc, hc].set(jnp.where(slot_ok, _i32(src), L.transfer_src[nc, hc])),
        transfer_dst=L.transfer_dst.at[nc, hc].set(jnp.where(slot_ok, _i32(dst), L.transfer_dst[nc, hc])),
        transfer_t=L.transfer_t.at[nc, hc].set(jnp.where(slot_ok, _f64(now), L.transfer_t[nc, hc])),
        route_cost_s=jnp.where(ok, L.route_cost_s + _f64(route_s), L.route_cost_s),
        overflow=(L.overflow + jnp.where(ok & ~slot_ok, 1, 0)).astype(jnp.int32))


def defer_entry(L: TruckLedger, n, delta_s, ok=True) -> TruckLedger:
    """defer_admitted_entry() 의 원장 부분 (322-323 · 363-366행) — A ← A + delta · version+1 · 이연 횟수·누적."""
    ok = jnp.asarray(ok, jnp.bool_) & _row_valid(L, n)
    nc = _clip_row(L.a_gate_in, n)
    d = _f64(delta_s)
    return L._replace(
        a_gate_in=_set1(L.a_gate_in, n, L.a_gate_in[nc] + d, ok),                    # 322-323 · 363행
        version=_set1(L.version, n, L.version[nc] + 1, ok),
        entry_deferrals=_set1(L.entry_deferrals, n, L.entry_deferrals[nc] + 1, ok),
        entry_deferred_s=_set1(L.entry_deferred_s, n, L.entry_deferred_s[nc] + d, ok))


# ─────────────────────────────────────────────────── 검사 술어 (원장이 답할 수 있는 것만)
def _first_fail(checks):
    """[(통과 bool, 실패 코드), …] 순서대로 — 전부 통과면 (True, 0), 아니면 (False, 첫 실패 코드)."""
    ok = jnp.asarray(True)
    code = jnp.asarray(LR_OK, jnp.int32)
    for passed, c in checks:
        passed = jnp.asarray(passed, jnp.bool_)
        code = jnp.where(ok & ~passed, jnp.asarray(c, jnp.int32), code)
        ok = ok & passed
    return ok, code


def pre_gate_ok(L: TruckLedger, n, dst, now, *, n_blocks: int, max_transfers: int = 1):
    """prepare_pre_gate_transfer 421-441행 가운데 원장 검사 — (ok, code).

    미등록 · dst 부적격(범위 밖·owner 와 같음) · not reassignable · 이송 상한 · [status: 조정자] · 이미 gate-in
    (`A ≤ now + 1e-6`) · A 결측. 나머지(용량·도착시각 창)는 조정자."""
    nc = _clip_row(L.owner, n)
    dst = _i32(dst)
    a = L.a_gate_in[nc]
    return _first_fail([
        (_row_valid(L, n) & L.registered[nc], LR_UNREGISTERED),
        ((dst >= 0) & (dst < n_blocks) & (dst != L.owner[nc]), LR_BAD_DST),
        (reassignable(L)[nc], LR_NOT_REASSIGNABLE),
        (L.transfer_count[nc] < max_transfers, LR_TRANSFER_CAP),
        (~((a < EMPTY_TIME) & (a <= _f64(now) + GATE_EPS)), LR_ALREADY_GATE_IN),
        (a < EMPTY_TIME, LR_A_MISSING),
    ])


def post_gate_ok(L: TruckLedger, n, dst, now, *, n_blocks: int):
    """prepare_transfer 379-391행 가운데 원장 검사 — 이미 게이트를 지난 반입만 (`A ≤ now + 1e-6`, 390행)."""
    nc = _clip_row(L.owner, n)
    dst = _i32(dst)
    a = L.a_gate_in[nc]
    return _first_fail([
        (_row_valid(L, n) & L.registered[nc], LR_UNREGISTERED),
        ((dst >= 0) & (dst < n_blocks) & (dst != L.owner[nc]), LR_BAD_DST),
        (reassignable(L)[nc], LR_NOT_REASSIGNABLE),
        ((a < EMPTY_TIME) & (a <= _f64(now) + GATE_EPS), LR_BEFORE_GATE_IN),
    ])


def validate_ok(L: TruckLedger, n, src, seen_version):
    """validate 476-480행 원장 검사 — version·owner 가 prepare 시점 그대로인가 · 아직 reassignable 인가."""
    nc = _clip_row(L.owner, n)
    return _first_fail([
        (_row_valid(L, n) & L.registered[nc], LR_UNREGISTERED),
        ((L.version[nc] == _i32(seen_version)) & (L.owner[nc] == _i32(src)), LR_STALE),
        (reassignable(L)[nc], LR_NOT_REASSIGNABLE),
    ])


def defer_ok(L: TruckLedger, n, delta_s, now, *, max_deferrals: int = 1):
    """defer_admitted_entry 309-317행 원장 검사 — 미등록 · delta ≤ 0 · 이연 상한 · 이미 gate-in (`A ≤ now + 1e-6`)."""
    nc = _clip_row(L.owner, n)
    a = L.a_gate_in[nc]
    return _first_fail([
        (_row_valid(L, n) & L.registered[nc], LR_UNREGISTERED),
        (_f64(delta_s) > 0.0, LR_DELTA_NONPOS),
        (L.entry_deferrals[nc] < max_deferrals, LR_DEFER_CAP),
        ((a < EMPTY_TIME) & (a > _f64(now) + GATE_EPS), LR_ALREADY_GATE_IN),
    ])


# ─────────────────────────────────────────────────── 불변식 (check_invariants 653-666)
def invariant_errors(L: TruckLedger, world_owner) -> jnp.ndarray:
    """세계가 실제로 쥔 소유 (N,) int32 (-1 = 어느 블록에도 없음) 와 원장의 owner 를 맞춰 본다 — 위반 행 수 () int32.

    v5: 이중 소유(배열에선 구조적으로 불가) · 소유자 없음 · owner 불일치 · 보존 위반(작업 수). 여기서는
    등록 행의 world_owner ≠ owner (소유자 없음 포함) + 미등록인데 세계가 쥔 행."""
    wo = _i32(world_owner)
    bad = (L.registered & (wo != L.owner)) | (~L.registered & (wo >= 0))
    return jnp.sum(bad.astype(jnp.int32)).astype(jnp.int32)


# ─────────────────────────────────────────────────── 호스트 — 저장·복원·v5 모양
def to_numpy(L: TruckLedger) -> dict[str, np.ndarray]:
    """원장 전체 → numpy dict (np.savez 로 저장해 세션을 이어 돌린다)."""
    return {f: np.asarray(getattr(L, f)) for f in L._fields}


def from_numpy(d) -> TruckLedger:
    """`to_numpy` 의 역 (np.load 결과도 받는다). dtype 은 원장 규약으로 되돌린다."""
    tmpl = empty_truck_ledger(int(np.asarray(d["registered"]).shape[0]), int(np.asarray(d["transfer_src"]).shape[1]))
    return TruckLedger(*[jnp.asarray(np.asarray(d[f]), getattr(tmpl, f).dtype) for f in TruckLedger._fields])


def _opt(v: float):
    return None if not np.isfinite(v) else float(v)


def to_v5_records(L: TruckLedger, job_ids, block_ids, flow_names=None, rows=None) -> dict[str, dict]:
    """등록된 행 → v5 `TerminalLedger.records` 모양 (job_id → JobRecord 필드 dict), **dict 순서 = 등록 순서**.

    flow 는 v5 `JobFlow.value` 문자열 (host_convert.FLOW_NAMES 순); 이력은 (src 이름, dst 이름, t) 튜플.
    rows 를 주면 그 행들만 (준 순서대로; 미등록 행은 뺀다)."""
    if flow_names is None:
        from .host_convert import FLOW_NAMES
        flow_names = FLOW_NAMES
    d = to_numpy(L)
    if rows is None:
        order = np.asarray(registration_order(L))[: int(d["n_registered"])]
    else:
        order = [int(n) for n in rows if d["registered"][int(n)]]
    out: dict[str, dict] = {}
    for n in order:
        n = int(n)
        cnt = int(d["transfer_count"][n])
        hist = tuple((block_ids[int(d["transfer_src"][n, h])], block_ids[int(d["transfer_dst"][n, h])],
                      float(d["transfer_t"][n, h])) for h in range(min(cnt, L.h)))
        out[job_ids[n]] = {
            "job_id": job_ids[n], "origin_block": block_ids[int(d["origin_block"][n])],
            "owner": block_ids[int(d["owner"][n])], "flow": flow_names[int(d["flow"][n])],
            "version": int(d["version"][n]), "transfer_count": cnt, "transfer_history": hist,
            "a_gate_in": _opt(d["a_gate_in"][n]), "b_block_arrival": _opt(d["b_block_arrival"][n]),
            "c_job_done": _opt(d["c_job_done"][n]), "o_gate_out": _opt(d["o_gate_out"][n]),
            "locked": bool(d["locked"][n]), "entry_deferrals": int(d["entry_deferrals"][n]),
            "entry_deferred_s": float(d["entry_deferred_s"][n]),
        }
    return out


def sample_list(L: TruckLedger, end) -> list[float]:
    """v5 `a_to_o_samples_s(end)` 와 같은 **목록** (dict 순서) — 파이썬 float."""
    s = np.asarray(a_to_o_samples_s(L, end))
    order = np.asarray(registration_order(L))[: int(np.asarray(L.n_registered))]
    return [float(s[n]) for n in order if not np.isnan(s[n])]
