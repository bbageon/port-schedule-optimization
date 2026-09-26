"""이송(공간 판매)·이연(시간 판매)의 **원자 확정** — 배열판 ([[YR-327]] 조각 6 · key=txn).

v5 `world/integrated/multiblock.py` 의 다음 메서드를 **순수 함수**로 옮긴다:

    prepare_transfer          377-406   게이트 **진입 후** 재배정 준비 (예약·txn 발급)
    prepare_pre_gate_transfer 408-455   게이트 **진입 전** 재배정 준비
    try_pre_gate_transfer     457-473   prepare → commit, 실패면 rollback
    validate                  475-488   version/owner · 자격 · 상태 · 규격 슬롯
    _precommit/_snapshot/_restore 490-554 (원자성 흉내 — 여기서는 구조로 사라진다)
    commit                    556-620   이관 실행 (사건·오더 행·컨테이너 규격·원장)
    rollback/_release         622-629   예약 해제 (txn_id 로 멱등)
    try_transfer              631-650
    defer_admitted_entry      294-366   진입 시각 이연
    try_defer_admitted_entry  368-374
    free_slots                230-240   수신 용량
    _sync_locks               212-227   원장 lock 갱신
    check_invariants          653-666

■ 원자성 — new = f(state); state' = where(ok, new, old)
  v5 는 스냅샷·복원(try/except) 으로 "전부 되거나 전혀 안 되게" 를 흉내 낸다. 배열판은 실패 여부와
  무관하게 새 상태를 **전부 계산**하고 마지막에 잎마다 `where(ok, new, old)` 로 고른다 — 롤백이
  공짜다 (명세 hard_parts ④). 거절 사유는 v5 검사 **순서 그대로** 첫 실패 코드로 돌려준다
  (`R_*`, `REASONS` 에 v5 예외 문구의 부분 문자열이 있어 시험이 v5 메시지를 코드로 사상한다).

■ 상태 모양 — 블록 축 B 가 **맨 앞**
  `worlds`  : `state.BlockWorld` 를 `vmap` 으로 쌓은 것 — 모든 잎이 (B, …). 오더는 (B, N_rows).
              트럭 하나는 **어느 한 블록 사본의 한 행**에만 산다 (`orders.block[b, r] >= 0`). 다른 사본에는
              없거나 빈 행 (`empty_orders` 기본값: block -1 · 시각 +inf · exit_travel -1) — 그래야 그 블록의
              `advance`(터미널 점유 적분)·`ledger_mode`·후보 생성이 남의 트럭을 안 본다.
              ★ v5 의 `sim.jobs.pop / dst.jobs[jid] = j` (소유 이동) = 행을 **복사 뒤 원본 비움**.
  `ledger`  : `TxnLedger` — 원장 키 n (트럭 번호) 마다 `owner`(블록)·`row`(그 블록 사본의 행) + v5 JobRecord 중
              이송·이연이 읽고 쓰는 열 + 조정자 상태 (`_reserved_inbound`·`route_cost_s`·`_txn_seq`·`_open_txn`).
              ★행 번호 규약 두 가지를 **모두** 받는다:
                 · 전역 행  — 전 블록이 행 번호를 공유 (row[n] = n). 이송해도 행 번호 그대로 (dst_row = src_row).
                 · 블록별 행 — 사본마다 자기 번호 (host_terminal.py 규약: 트럭·본선 작업 뒤에 **여분 행** n_spare).
                   이송은 dst 사본의 빈 행으로 들어간다 — `commit(..., dst_row=first_free_row(worlds, d, n_used_d))`.
              v5 `JobRecord` 의 `a_gate_in`·`flow`·`b_block_arrival` 은 **OrderArrays 열에서 파생**한다 (중복 열을
              두지 않는다 — 명세 "원장 = OrderArrays 열 자체"):
                  a_gate_in        = orders.gate_in_s[owner, row]   (+inf = None)
                  flow             = orders.flow[owner, row]
                  reassignable     = ~locked & flow == GATE_IN
                     (v5 `b_block_arrival is None` 항은 `_sync_locks` 가 lock 과 **함께** 만 채우므로
                      ~locked 에 포함된다; `harvest` 는 런 끝 1회라 런 중 판정에 안 낀다)
  `now`     : v5 `MultiBlockTerminal.now` (146-147행) — `terminal_now(worlds)`. 검토 시각에는
              전 블록이 같은 t 라 t 와 같다. 함수는 이 값을 **인자로** 받는다 (조정자가 안다).

■ 컨테이너 예비칸 — `TxnLedger.cont_base` (B,) = 블록별 초기 컨테이너 수 c0_b
  반입 오더가 낳을 컨테이너 번호는 블록·행마다 다르다 (`host_convert` 규칙 c0 + row). 이송은 dst 사본의
  `inbound_cont[d, dst_row] = cont_base[d] + dst_row` 로 다시 매기고 `conts.c_size[d, ·]` 에 규격을 적는다
  (v5 는 컨테이너가 place 시점에 `IN_{job_id}` 로 생기므로 번호만 바뀌면 된다). src 사본의
  `c_size[cont_base[s] + src_row]` 은 **건드리지 않는다** — place 는 살아 있는 오더의 예비칸만 읽는다.

■ 부동소수점 — 이 파일의 실수 연산은 전부 **덧셈** (A + travel · A + travel + route · eta + Δ ·
  route_cost += route · deferred += Δ) 이라 v5 파이썬 float 와 같은 한 번 반올림이다. 곱셈이 없어
  FMA·상수 재결합이 끼어들 자리가 없다. 허용오차 상수는 v5 그대로: 창 판정 1e-6 (316·390·438행),
  도착 판정 1e-9 (398·447행).

■ 트랜잭션 — `TransferTxn` + `open_txn` 슬롯 (MAX_OPEN,)
  v5 `_open_txn: set[int]` 는 "rollback 한 txn 을 다시 commit 하면 예약 없이 이송이 성사" (게이트 D) 를
  막는다. 슬롯이 모자라면 `open_overflow` 를 올린다 (조용히 버리지 않는다 — 0 이어야 정상).
  `try_*` 는 v5 와 같은 순서 prepare → commit → (실패면) rollback 을 그대로 밟아 `txn_seq` 증가 횟수까지
  같다 (prepare 성공 시 1).

■ 시험 — tests/v6/test_gpu_transfer_txn.py: v5 무대(Y01+Y21·40대)를 실제로 굴리며 검토 시각마다 v5 상태를
  배열로 옮겨 같은 질의를 v5·배열 양쪽에 넣고 결과 상태·거절 코드를 == 로 대조한다.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE, EventArray, push_event
from .geom import Geom
from .stack_ops import find_slot
from .state import (EV_BLOCK_ARRIVAL, FL_GATE_IN, FL_VESSEL_DISCHARGE, JS_PLANNED, BlockWorld,
                    OrderArrays, empty_orders)

__all__ = [
    "CAPACITY_MARGIN", "EPS_WINDOW", "EPS_ARRIVAL",
    "OK", "R_UNREGISTERED", "R_BAD_DST", "R_NOT_REASSIGNABLE", "R_MAX_TRANSFERS", "R_SRC_STATUS",
    "R_ALREADY_GATE_IN", "R_NO_GATE_IN", "R_BEFORE_GATE_IN", "R_LEDGER_ASYM", "R_CAPACITY",
    "R_BAD_ARRIVAL", "R_STALE", "R_LOCKED_AFTER", "R_STATUS_AFTER", "R_NO_SLOT", "R_CLOSED_TXN",
    "R_NONPOSITIVE_DELTA", "R_MAX_DEFERRALS", "R_STATUS", "R_OUT_OF_WINDOW", "R_LEDGER_A_MISSING",
    "R_NO_LEDGER_RECORD", "R_NO_SPARE_ROW", "R_NO_TXN_SLOT", "REASONS", "REASON_NAMES", "reason_code",
    "TxnLedger", "TransferTxn", "empty_txn_ledger", "register_job", "no_txn",
    "tree_where", "terminal_now", "live_mask", "first_free_row", "free_slots", "sync_locks", "reassignable",
    "remove_block_arrival", "prepare_transfer", "prepare_pre_gate_transfer", "validate", "commit",
    "rollback", "try_transfer", "try_pre_gate_transfer", "defer_admitted_entry",
    "try_defer_admitted_entry", "check_invariants",
    "txn_ledger_from_v5", "txn_ledger_to_v5",
]

F = TIME_DTYPE
#: v5 multiblock.py:26 — 수신 블록에 남겨둘 여유 슬롯
CAPACITY_MARGIN = 2
#: v5 창 판정 여유 (316·390·438행) 와 도착 판정 여유 (267·398·447행)
EPS_WINDOW = 1e-6
EPS_ARRIVAL = 1e-9

# ───────────────────────────────────────────────── 거절 코드 (v5 TransferError 사유 · 검사 순서)
OK = 0
R_UNREGISTERED = 1        # "미등록"                        prepare*:380 · defer:310
R_BAD_DST = 2             # "수신 블록 부적격"               382 · 424
R_NOT_REASSIGNABLE = 3    # "lock/자격 위반"                 384 · 426
R_MAX_TRANSFERS = 4       # "이송 상한 초과"                 431 (pre-gate 만)
R_SRC_STATUS = 5          # "소스 상태 위반"                 388 · 435
R_ALREADY_GATE_IN = 6     # "이미 gate-in"                   438 (pre-gate) · 316 (defer)
R_NO_GATE_IN = 7          # "gate-in 결측"                   440 · _precommit 502
R_BEFORE_GATE_IN = 8      # "gate-in 전 (창 밖)"             390 (post-gate)
R_LEDGER_ASYM = 9         # "time_ledger 비대칭"             393 · 442 — 배열판은 전 블록 장부 있음 (구조상 불가)
R_CAPACITY = 10           # "용량 부족"                      395 · 444
R_BAD_ARRIVAL = 11        # "도착시각 무효"                  398 · 447
R_STALE = 12              # "version/owner 변경"             validate 477
R_LOCKED_AFTER = 13       # "준비 후 lock"                   479
R_STATUS_AFTER = 14       # "준비 후 상태 변경"              482
R_NO_SLOT = 15            # "규격 적합 슬롯 없음"            486
R_CLOSED_TXN = 16         # "닫힌 트랜잭션"                  commit 562
R_NONPOSITIVE_DELTA = 17  # "이연량은 양수여야"              defer 312
R_MAX_DEFERRALS = 18      # "이연 상한 초과"                 314
R_STATUS = 19             # "상태 위반 (PLANNED 아님)"       320 (defer)
R_OUT_OF_WINDOW = 20      # "관측창 밖"                      325
R_LEDGER_A_MISSING = 21   # "소스 장부에 A 부재"             _precommit 509 — 배열판 구조상 불가
R_NO_LEDGER_RECORD = 22   # "시간 장부 항목 없음"            defer 331 — 배열판 구조상 불가
R_NO_SPARE_ROW = 23       # (배열판 전용) dst 사본에 빈 행이 없다 — v5 dict 는 무한이라 자리가 없다; 칸 부족은 조용히 넘기지 않는다
R_NO_TXN_SLOT = 24        # (배열판 전용) 열린 트랜잭션 슬롯(open_txn) 소진 — v5 `_open_txn` 은 무한 set 이라 없는 경우

#: 코드 → v5 예외 문구의 **부분 문자열** (시험이 v5 메시지를 코드로 사상). ★검사 순서 = 이 순서 —
#: "소스 상태 위반" 이 "상태 위반" 보다 앞에 와야 한다.
REASONS: tuple[tuple[int, str], ...] = (
    (R_UNREGISTERED, "미등록"), (R_BAD_DST, "수신 블록 부적격"), (R_NOT_REASSIGNABLE, "lock/자격 위반"),
    (R_MAX_TRANSFERS, "이송 상한 초과"), (R_SRC_STATUS, "소스 상태 위반"), (R_ALREADY_GATE_IN, "이미 gate-in"),
    (R_NO_GATE_IN, "gate-in 결측"), (R_BEFORE_GATE_IN, "gate-in 전"), (R_LEDGER_ASYM, "time_ledger 비대칭"),
    (R_CAPACITY, "용량 부족"), (R_BAD_ARRIVAL, "도착시각 무효"), (R_STALE, "version/owner 변경"),
    (R_LOCKED_AFTER, "준비 후 lock"), (R_STATUS_AFTER, "준비 후 상태 변경"), (R_NO_SLOT, "규격 적합 슬롯 없음"),
    (R_CLOSED_TXN, "닫힌 트랜잭션"), (R_NONPOSITIVE_DELTA, "이연량은 양수여야"),
    (R_MAX_DEFERRALS, "이연 상한 초과"), (R_STATUS, "상태 위반"), (R_OUT_OF_WINDOW, "관측창 밖"),
    (R_LEDGER_A_MISSING, "소스 장부에 A 부재"), (R_NO_LEDGER_RECORD, "시간 장부 항목 없음"),
    (R_NO_SPARE_ROW, "수신 사본 빈 행 없음"), (R_NO_TXN_SLOT, "열린 트랜잭션 슬롯 없음"),
)
REASON_NAMES: dict[int, str] = {OK: "OK", **{c: s for c, s in REASONS}}


def reason_code(message: str) -> int:
    """v5 `TransferError` 문구 → 코드 (호스트·시험용). 모르는 문구는 -1 — 조용히 OK 로 두지 않는다."""
    for code, sub in REASONS:
        if sub in message:
            return code
    return -1


# ───────────────────────────────────────────────── 상태
class TxnLedger(NamedTuple):
    """전역 원장 중 이송·이연 열 + 조정자 상태 (머리말 ■ 상태 모양). N 오더 · B 블록."""

    owner: jnp.ndarray            # (N,) int32  현재 실행 블록 (-1 = 미등록 — v5 records 에 없음)
    row: jnp.ndarray              # (N,) int32  owner 사본 안의 행 (전역 행 규약이면 = n)
    origin: jnp.ndarray           # (N,) int32  최초 배정 블록 (감사용 불변)
    version: jnp.ndarray          # (N,) int32  낙관적 동시성
    transfer_count: jnp.ndarray   # (N,) int32
    hist_src: jnp.ndarray         # (N,T) int32  transfer_history[k] = (src, dst, t) — 칸 k < transfer_count 만 유효
    hist_dst: jnp.ndarray         # (N,T) int32
    hist_t: jnp.ndarray           # (N,T) f64
    locked: jnp.ndarray           # (N,) bool   block-in/배정 이후 = 재배정 금지 (sync_locks 가 켠다)
    entry_deferrals: jnp.ndarray  # (N,) int32
    entry_deferred_s: jnp.ndarray # (N,) f64
    reserved_inbound: jnp.ndarray # (B,) int32  미도착 예약분 (prepare +1 · release −1)
    cont_base: jnp.ndarray        # (B,) int32  블록별 반입 예비칸 원점 c0_b (머리말 ■ 컨테이너 예비칸)
    route_cost_s: jnp.ndarray     # ()   f64    이송 추가주행 누적 (critical-2)
    txn_seq: jnp.ndarray          # ()   int32  트랜잭션 고유 id 발급기
    open_txn: jnp.ndarray         # (MAX_OPEN,) int32  살아 있는 예약의 txn_id (-1 = 빈 칸)
    open_overflow: jnp.ndarray    # ()   int32  open_txn 칸 부족 수 (0 이어야 정상)
    hist_overflow: jnp.ndarray    # ()   int32  transfer_history 칸 부족 수 (0 이어야 정상)

    @property
    def n(self) -> int:
        return int(self.owner.shape[0])

    @property
    def b(self) -> int:
        return int(self.reserved_inbound.shape[0])


class TransferTxn(NamedTuple):
    """v5 `TransferTxn` (98-110행). 실패한 prepare 는 job/src/dst/txn_id 가 -1 (= 닫힌 txn)."""

    job: jnp.ndarray            # () int32
    src: jnp.ndarray            # () int32
    dst: jnp.ndarray            # () int32
    seen_version: jnp.ndarray   # () int32
    new_arrival_s: jnp.ndarray  # () f64
    prepared_at_s: jnp.ndarray  # () f64
    route_s: jnp.ndarray        # () f64   post-gate: route_s · pre-gate: route_delta_s
    txn_id: jnp.ndarray         # () int32  (-1 = 없음)


def _i32(x):
    return jnp.asarray(x, jnp.int32)


def _f(x):
    return jnp.asarray(x, F)


def empty_txn_ledger(n: int, b: int, *, max_t: int = 1, max_open: int = 4) -> TxnLedger:
    """빈 원장 — 아무 오더도 등록되지 않은 상태 (owner 전부 -1). `cont_base` 는 호스트가 채운다."""
    return TxnLedger(
        owner=jnp.full((n,), EMPTY_ID, jnp.int32), row=jnp.arange(n, dtype=jnp.int32),
        origin=jnp.full((n,), EMPTY_ID, jnp.int32),
        version=jnp.zeros((n,), jnp.int32), transfer_count=jnp.zeros((n,), jnp.int32),
        hist_src=jnp.full((n, max_t), EMPTY_ID, jnp.int32), hist_dst=jnp.full((n, max_t), EMPTY_ID, jnp.int32),
        hist_t=jnp.full((n, max_t), EMPTY_TIME, F),
        locked=jnp.zeros((n,), bool), entry_deferrals=jnp.zeros((n,), jnp.int32),
        entry_deferred_s=jnp.zeros((n,), F),
        reserved_inbound=jnp.zeros((b,), jnp.int32), cont_base=jnp.zeros((b,), jnp.int32),
        route_cost_s=jnp.zeros((), F), txn_seq=jnp.zeros((), jnp.int32),
        open_txn=jnp.full((max_open,), EMPTY_ID, jnp.int32),
        open_overflow=jnp.zeros((), jnp.int32), hist_overflow=jnp.zeros((), jnp.int32))


def register_job(ledger: TxnLedger, n, block, row=None) -> TxnLedger:
    """v5 `ledger.register(JobRecord(job_id, origin_block=bid, owner=bid, …))` — 원장 열만 (오더 행·큐는 투입 하위 시스템 몫).
    row = 그 블록 사본의 행 (기본 None = n, 전역 행 규약)."""
    n = _i32(n)
    nc = jnp.clip(n, 0, ledger.n - 1)
    valid = (n >= 0) & (n < ledger.n)
    b = _i32(block)
    r = nc if row is None else _i32(row)
    return ledger._replace(owner=ledger.owner.at[nc].set(jnp.where(valid, b, ledger.owner[nc])),
                           row=ledger.row.at[nc].set(jnp.where(valid, r, ledger.row[nc])),
                           origin=ledger.origin.at[nc].set(jnp.where(valid, b, ledger.origin[nc])))


def no_txn() -> TransferTxn:
    """닫힌(없는) 트랜잭션 — prepare 실패의 반환값과 같다."""
    return TransferTxn(job=_i32(EMPTY_ID), src=_i32(EMPTY_ID), dst=_i32(EMPTY_ID), seen_version=_i32(0),
                       new_arrival_s=_f(EMPTY_TIME), prepared_at_s=_f(EMPTY_TIME), route_s=_f(0.0),
                       txn_id=_i32(EMPTY_ID))


# ───────────────────────────────────────────────── 작은 도구
def tree_where(c, a, b):
    """c 면 a, 아니면 b — pytree 잎마다 where (engine_step.tree_where 와 같음; 무거운 import 회피)."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


def _row(tree, b):
    """(B, …) 잎 묶음에서 블록 b 의 행."""
    return jax.tree_util.tree_map(lambda x: x[b], tree)


def _set_row(tree, b, row):
    return jax.tree_util.tree_map(lambda x, r: x.at[b].set(r), tree, row)


def terminal_now(worlds: BlockWorld) -> jnp.ndarray:
    """v5 `MultiBlockTerminal.now` (146-147행): 끝나지 않은 블록 시계의 최소, 전부 끝났으면 최대."""
    live = ~worlds.terminal
    return jnp.where(jnp.any(live), jnp.min(jnp.where(live, worlds.clock, jnp.inf)), jnp.max(worlds.clock))


def live_mask(worlds: BlockWorld) -> jnp.ndarray:
    """(B,N_rows) bool — 행 r 이 블록 b 사본에 살아 있나 (= v5 `jid in sim.jobs`)."""
    return worlds.orders.block >= 0


def first_free_row(worlds: BlockWorld, b, min_row=0) -> jnp.ndarray:
    """블록 b 사본에서 `min_row` 이상인 첫 빈 행 (block < 0) — 없으면 -1. () int32.

    블록별 행 규약(host_terminal.py)에서 이송 수신 행을 고를 때 쓴다 (`min_row = n_used_b`). 전역 행 규약에서는
    필요 없다 (dst_row = src_row)."""
    B = worlds.clock.shape[0]
    bc = jnp.clip(_i32(b), 0, B - 1)
    free = (worlds.orders.block[bc] < 0) & (jnp.arange(worlds.orders.block.shape[1]) >= _i32(min_row))
    return jnp.where(jnp.any(free), jnp.argmax(free), EMPTY_ID).astype(jnp.int32)


def free_slots(worlds: BlockWorld, ledger: TxnLedger) -> jnp.ndarray:
    """v5 `free_slots(bid)` (230-240행) 를 전 블록 (B,) int32 로.

    물리 칸 − 야드 안 컨테이너 수 − PLANNED 반입/양하 수 − 미도착 예약분.
    """
    o = worlds.orders
    B_, R_, T_ = worlds.stacks.grid.shape[1:]
    phys = int(B_) * int(R_) * int(T_)
    used = jnp.sum(worlds.conts.c_alive, axis=1).astype(jnp.int32)                       # len(stk.containers)
    pending = jnp.sum(live_mask(worlds) & (o.status == JS_PLANNED)
                      & ((o.flow == FL_GATE_IN) | (o.flow == FL_VESSEL_DISCHARGE)), axis=1).astype(jnp.int32)
    return phys - used - pending - ledger.reserved_inbound


def sync_locks(worlds: BlockWorld, ledger: TxnLedger, blocks=None) -> TxnLedger:
    """v5 `_sync_locks(sim)` (212-227행) — 블록 사본에 살아 있고 status != PLANNED 인 등록 오더를 lock.

    blocks (B,) bool 은 v5 가 ReviewEpoch 를 돌려준 블록만 동기화하는 것을 재현할 때 쓴다 (기본 전 블록).
    `b_block_arrival` 은 따로 두지 않는다 (머리말 ■ 상태 모양 — lock 과 함께만 채워지는 열).
    """
    o = worlds.orders
    B, R = o.block.shape
    m = jnp.ones((B,), bool) if blocks is None else jnp.asarray(blocks, bool)
    reg = ledger.owner >= 0
    oc = jnp.clip(ledger.owner, 0, B - 1)
    rc = jnp.clip(ledger.row, 0, R - 1)
    newly = reg & m[oc] & (o.block[oc, rc] >= 0) & (o.status[oc, rc] != JS_PLANNED)
    return ledger._replace(locked=ledger.locked | newly)


def reassignable(worlds: BlockWorld, ledger: TxnLedger, n) -> jnp.ndarray:
    """v5 `JobRecord.reassignable` (51-54행): ~locked & flow == GATE_IN (& b 없음 — lock 에 포함)."""
    nc = jnp.clip(_i32(n), 0, ledger.n - 1)
    s = jnp.clip(ledger.owner[nc], 0, worlds.clock.shape[0] - 1)
    r = jnp.clip(ledger.row[nc], 0, worlds.orders.block.shape[1] - 1)
    return ~ledger.locked[nc] & (worlds.orders.flow[s, r] == FL_GATE_IN)


def remove_block_arrival(q: EventArray, n) -> EventArray:
    """큐에서 `BLOCK_ARRIVAL(target=n)` 을 전부 빈 칸으로 — v5 337-339·572-574행의 리스트 필터+heapify.
    (힙 재구성은 키를 안 바꾸므로 마스크 제거와 등가 — 명세 determinism.) 순번 카운터는 그대로."""
    hit = (q.kind == EV_BLOCK_ARRIVAL) & (q.target == _i32(n))
    return q._replace(time=jnp.where(hit, EMPTY_TIME, q.time), kind=jnp.where(hit, EMPTY_ID, q.kind),
                      target=jnp.where(hit, EMPTY_ID, q.target), seq=jnp.where(hit, EMPTY_ID, q.seq))


def _first_fail(checks: list[tuple]) -> jnp.ndarray:
    """[(ok_i, code_i), …] 를 순서대로 — 첫 실패의 코드, 전부 통과면 OK."""
    code = _i32(OK)
    for ok, c in reversed(checks):
        code = jnp.where(ok, code, _i32(c))
    return code


def _empty_order_row() -> OrderArrays:
    """빈 오더 한 행 (empty_orders(1) 의 0번 행) — 이송 뒤 src 사본의 행."""
    return jax.tree_util.tree_map(lambda x: x[0], empty_orders(1))


# ───────────────────────────────────────────────── 공통 조회
def _lookup(worlds: BlockWorld, ledger: TxnLedger, job, dst):
    """prepare 두 판이 같이 쓰는 조회. 색인은 전부 clip 하고 유효성은 bool 로 따로 든다."""
    N, B = ledger.n, worlds.clock.shape[0]
    o = worlds.orders
    R = o.block.shape[1]
    job = _i32(job)
    dst = _i32(dst)
    nc = jnp.clip(job, 0, N - 1)
    owner = ledger.owner[nc]
    registered = (job >= 0) & (job < N) & (owner >= 0)                                  # rec is None
    s = jnp.clip(owner, 0, B - 1)
    d = jnp.clip(dst, 0, B - 1)
    r = jnp.clip(ledger.row[nc], 0, R - 1)                                              # owner 사본의 행
    dst_ok = (dst >= 0) & (dst < B) & (dst != owner)                                    # 382·424행
    live_src = o.block[s, r] >= 0                                                       # src.jobs.get(jid) is not None
    reass = ~ledger.locked[nc] & (o.flow[s, r] == FL_GATE_IN)                           # 384·426행 reassignable
    planned = live_src & (o.status[s, r] == JS_PLANNED)                                 # 388·435행
    a = o.gate_in_s[s, r]                                                               # rec.a_gate_in (+inf = None)
    return dict(job=job, nc=nc, owner=owner, registered=registered, s=s, d=d, r=r, dst_ok=dst_ok,
                reass=reass, planned=planned, a=a, has_a=a < EMPTY_TIME)


def _issue(worlds: BlockWorld, ledger: TxnLedger, L: dict, ok, now, arr, route_s):
    """prepare 성공분: 예약 +1 · txn_seq +1 · open_txn 슬롯 등록 (400-406·449-455행).

    ★슬롯이 꽉 차면(has=False) **아무것도 바꾸지 않고** prepare 를 실패시킨다 (R_NO_TXN_SLOT). 구판은 예약만
      올리고 슬롯에는 못 넣어, commit 이 R_CLOSED_TXN 으로 거절하고 `_release` 도 못 찾아 그 블록의 수신 용량이
      영구히 1 줄었다 (v5 `_open_txn` 은 무한 set 이라 구조적으로 없는 누수). 반환 (원장', txn, ok').
    """
    d = L["d"]
    txn_id = ledger.txn_seq + 1
    free = ledger.open_txn < 0
    has = jnp.any(free)
    ok2 = ok & has
    slot = jnp.argmax(free)
    open2 = jnp.where(ok2, ledger.open_txn.at[slot].set(txn_id), ledger.open_txn)
    ledger2 = ledger._replace(
        reserved_inbound=ledger.reserved_inbound.at[d].add(jnp.where(ok2, 1, 0).astype(jnp.int32)),
        txn_seq=jnp.where(ok2, txn_id, ledger.txn_seq),
        open_txn=open2,
        open_overflow=ledger.open_overflow + jnp.where(ok & ~has, 1, 0).astype(jnp.int32))
    txn = TransferTxn(job=jnp.where(ok2, L["job"], EMPTY_ID).astype(jnp.int32),
                      src=jnp.where(ok2, L["owner"], EMPTY_ID).astype(jnp.int32),
                      dst=jnp.where(ok2, _i32(L["d"]), EMPTY_ID).astype(jnp.int32),
                      seen_version=ledger.version[L["nc"]],
                      new_arrival_s=_f(arr), prepared_at_s=_f(now), route_s=_f(route_s),
                      txn_id=jnp.where(ok2, txn_id, EMPTY_ID).astype(jnp.int32))
    return ledger2, txn, ok2


# ───────────────────────────────────────────────── prepare (377-406 · 408-455행)
def prepare_transfer(worlds: BlockWorld, ledger: TxnLedger, now, job, dst, *, route_s, travel_s,
                     capacity_margin: int = CAPACITY_MARGIN):
    """게이트 **진입 후** 재배정 준비 — v5 `prepare_transfer` (377-406행). 반환 (원장', txn, 코드).

    검사 순서 그대로: 미등록 → 수신 부적격 → lock/자격 → 소스 상태 → gate-in 전(창 밖) → [장부 비대칭:
    구조상 불가] → 용량 → 도착시각 (arr = A + travel + route, dst 시계·end 기준).
    """
    L = _lookup(worlds, ledger, job, dst)
    now = _f(now)
    in_window = L["has_a"] & ~(L["a"] > now + EPS_WINDOW)                               # 390행
    cap_ok = free_slots(worlds, ledger)[L["d"]] > _i32(capacity_margin)               # 395행
    arr = (L["a"] + _f(travel_s)) + _f(route_s)                                         # 397행 (좌결합)
    d = L["d"]
    arr_ok = (arr > worlds.clock[d] + EPS_ARRIVAL) & (arr <= worlds.end_s[d])           # 398행
    code = _first_fail([(L["registered"], R_UNREGISTERED), (L["dst_ok"], R_BAD_DST),
                        (L["reass"], R_NOT_REASSIGNABLE), (L["planned"], R_SRC_STATUS),
                        (in_window, R_BEFORE_GATE_IN), (cap_ok, R_CAPACITY), (arr_ok, R_BAD_ARRIVAL)])
    ok = code == OK
    ledger2, txn, ok2 = _issue(worlds, ledger, L, ok, now, arr, route_s)
    code = jnp.where(ok & ~ok2, _i32(R_NO_TXN_SLOT), code)          # 슬롯 소진 → 상태 무변경 + 명시 거절
    return ledger2, txn, code


def prepare_pre_gate_transfer(worlds: BlockWorld, ledger: TxnLedger, now, job, dst, *, travel_s,
                              route_delta_s=0.0, capacity_margin: int = CAPACITY_MARGIN,
                              max_transfers: int = 1):
    """게이트 **진입 전** 재배정 준비 — v5 `prepare_pre_gate_transfer` (408-455행). 반환 (원장', txn, 코드).

    검사 순서: 미등록 → 수신 부적격 → lock/자격 → 이송 상한 → 소스 상태 → 이미 gate-in → gate-in 결측
    → [장부 비대칭] → 용량 → 도착시각 (arr = A + travel — 게이트에서 새 블록 직행).
    """
    L = _lookup(worlds, ledger, job, dst)
    now = _f(now)
    under_max = ledger.transfer_count[L["nc"]] < _i32(max_transfers)                  # 431행
    not_in = ~(L["has_a"] & (L["a"] <= now + EPS_WINDOW))                               # 438행
    cap_ok = free_slots(worlds, ledger)[L["d"]] > _i32(capacity_margin)               # 444행
    arr = L["a"] + _f(travel_s)                                                         # 446행
    d = L["d"]
    arr_ok = (arr > worlds.clock[d] + EPS_ARRIVAL) & (arr <= worlds.end_s[d])           # 447행
    code = _first_fail([(L["registered"], R_UNREGISTERED), (L["dst_ok"], R_BAD_DST),
                        (L["reass"], R_NOT_REASSIGNABLE), (under_max, R_MAX_TRANSFERS),
                        (L["planned"], R_SRC_STATUS), (not_in, R_ALREADY_GATE_IN),
                        (L["has_a"], R_NO_GATE_IN), (cap_ok, R_CAPACITY), (arr_ok, R_BAD_ARRIVAL)])
    ok = code == OK
    ledger2, txn, ok2 = _issue(worlds, ledger, L, ok, now, arr, route_delta_s)
    code = jnp.where(ok & ~ok2, _i32(R_NO_TXN_SLOT), code)          # 슬롯 소진 → 상태 무변경 + 명시 거절
    return ledger2, txn, code


# ───────────────────────────────────────────────── validate (475-488행)
def validate(worlds: BlockWorld, ledger: TxnLedger, txn: TransferTxn, g: Geom, *, k0: int = 0) -> jnp.ndarray:
    """v5 `validate` — 코드 () int32 (OK = 통과). 상태를 바꾸지 않는다.

    k0 = v5 `profile.cranes[0]` 의 크레인 번호 (정렬 순위; h21 프로파일은 0). find_slot 기준점은 그 크레인의
    service_bay_min · row 1.0 (485-487행), 제외 칸 없음.
    """
    N, B = ledger.n, worlds.clock.shape[0]
    o = worlds.orders
    nc = jnp.clip(txn.job, 0, N - 1)
    registered = (txn.job >= 0) & (txn.job < N) & (ledger.owner[nc] >= 0)               # records[…] KeyError 자리
    fresh = (ledger.version[nc] == txn.seen_version) & (ledger.owner[nc] == txn.src)    # 477행
    s = jnp.clip(txn.src, 0, B - 1)
    d = jnp.clip(txn.dst, 0, B - 1)
    r = jnp.clip(ledger.row[nc], 0, o.block.shape[1] - 1)
    reass = ~ledger.locked[nc] & (o.flow[s, r] == FL_GATE_IN)                           # 479행
    planned = (o.block[s, r] >= 0) & (o.status[s, r] == JS_PLANNED)                     # 482행
    cr = worlds.cranes
    bmin, bmax = cr.bay_min[d, k0], cr.bay_max[d, k0]
    zero_ex = jnp.zeros(worlds.stacks.height.shape[1:], bool)
    found, _, _ = find_slot(worlds.stacks.height[d], worlds.stacks.top_size[d], zero_ex, o.inbound_size[s, r],
                            bmin, bmax, bmin.astype(F), _f(1.0), g)                     # 486-487행
    slot_ok = ~o.is_store[s, r] | found
    return _first_fail([(registered, R_UNREGISTERED), (fresh, R_STALE), (reass, R_LOCKED_AFTER),
                        (planned, R_STATUS_AFTER), (slot_ok, R_NO_SLOT)])


# ───────────────────────────────────────────────── commit · rollback (556-629행)
def _release(ledger: TxnLedger, txn: TransferTxn) -> TxnLedger:
    """v5 `_release` (625-629행) — open_txn 에 있을 때만 슬롯 비우고 예약 −1 (멱등)."""
    hit = (ledger.open_txn == txn.txn_id) & (txn.txn_id >= 0)
    released = jnp.any(hit)
    d = jnp.clip(txn.dst, 0, ledger.b - 1)
    r = ledger.reserved_inbound[d]
    return ledger._replace(open_txn=jnp.where(hit, EMPTY_ID, ledger.open_txn),
                           reserved_inbound=ledger.reserved_inbound.at[d].set(
                               jnp.where(released, jnp.maximum(0, r - 1), r)))


def rollback(ledger: TxnLedger, txn: TransferTxn) -> TxnLedger:
    """v5 `rollback` (622-623행) = `_release`."""
    return _release(ledger, txn)


def commit(worlds: BlockWorld, ledger: TxnLedger, txn: TransferTxn, now, g: Geom, *, k0: int = 0, dst_row=None):
    """v5 `commit` (556-620행) — 검증 통과분만 **원자적으로** 이관. 반환 (worlds', ledger', 코드).

    ① 닫힌 txn 거절 ② validate ③ _precommit (gate-in 결측 — 장부 A 부재는 구조상 불가) ④ 변경:
        src 큐에서 BLOCK_ARRIVAL 제거 · src 오더 행 비움 · dst 오더 행 = src 행 + (actual_arrival = 새 도착,
        provided_eta += route, block = dst, inbound_cont = cont_base[dst] + dst_row) · dst 큐 push(새 도착) ·
        dst c_size[예비칸] = 규격 · 원장 owner/row/version/transfer_count/history · route_cost += route ⑤ _release.
    dst_row = dst 사본에서 트럭이 앉을 행 (None = src 행 번호 그대로 — 전역 행 규약; 블록별 규약은 `first_free_row`).
    빈 행이 없으면(-1) 배열판 전용 코드 R_NO_SPARE_ROW 로 거절한다 (v5 는 dict 라 자리가 무한).
    실패면 아무것도 바꾸지 않는다 (txn 은 열린 채 — 호출자가 rollback; `try_*` 가 그렇게 한다).
    """
    N, B = ledger.n, worlds.clock.shape[0]
    o = worlds.orders
    R = o.block.shape[1]
    now = _f(now)
    is_open = jnp.any((ledger.open_txn == txn.txn_id) & (txn.txn_id >= 0))             # 562행
    vcode = validate(worlds, ledger, txn, g, k0=k0)
    nc = jnp.clip(txn.job, 0, N - 1)
    s = jnp.clip(txn.src, 0, B - 1)
    d = jnp.clip(txn.dst, 0, B - 1)
    sr = jnp.clip(ledger.row[nc], 0, R - 1)                                             # src 사본의 행
    dr_raw = sr if dst_row is None else _i32(dst_row)
    dr_ok = (dr_raw >= 0) & (dr_raw < R)
    dr = jnp.clip(dr_raw, 0, R - 1)
    has_a = o.gate_in_s[s, sr] < EMPTY_TIME                                             # 501-503행 _precommit
    code = jnp.where(~is_open, _i32(R_CLOSED_TXN),
                     jnp.where(vcode != OK, vcode,
                               jnp.where(~has_a, _i32(R_NO_GATE_IN),
                                         jnp.where(~dr_ok, _i32(R_NO_SPARE_ROW), _i32(OK)))))
    ok = code == OK

    # ── 변경 구간 (568-616행) — 전부 계산하고 마지막에 where ──
    row = jax.tree_util.tree_map(lambda x: x[s, sr], o)                                 # j = src.jobs.pop(jid)
    eta = row.provided_eta_s
    is_store = row.is_store
    new_row = row._replace(
        block=_i32(d),
        actual_arrival_s=txn.new_arrival_s,                                             # 586행 actual_block_arrival
        provided_eta_s=jnp.where(eta < EMPTY_TIME, eta + txn.route_s, eta),             # 587-595행 est·eta += route
        inbound_cont=jnp.where(is_store, ledger.cont_base[d] + dr, EMPTY_ID).astype(jnp.int32))
    blank = _empty_order_row()
    o2 = jax.tree_util.tree_map(lambda x, e, v: x.at[s, sr].set(e).at[d, dr].set(v), o, blank, new_row)
    # 큐: src 제거 (572-574행) · dst push (598행) — 사건 대상은 각 사본의 행 번호
    qs = remove_block_arrival(_row(worlds.queue, s), sr)
    q2 = _set_row(worlds.queue, s, qs)
    qd = push_event(_row(q2, d), txn.new_arrival_s, EV_BLOCK_ARRIVAL, dr)
    q2 = _set_row(q2, d, qd)
    # dst 컨테이너 예비칸 규격 (host_convert 규칙 c_size[c0+row] = inbound_size)
    C = worlds.conts.c_size.shape[1]
    ic = jnp.clip(ledger.cont_base[d] + dr, 0, C - 1)
    c_size2 = worlds.conts.c_size.at[d, ic].set(jnp.where(is_store, row.inbound_size, worlds.conts.c_size[d, ic]))
    worlds2 = worlds._replace(orders=o2, queue=q2, conts=worlds.conts._replace(c_size=c_size2))
    # 원장 (611-616행)
    tc = ledger.transfer_count[nc]
    T = ledger.hist_src.shape[1]
    tcc = jnp.clip(tc, 0, T - 1)
    fits = tc < T
    put = lambda h, v: h.at[nc, tcc].set(jnp.where(fits, jnp.asarray(v, h.dtype), h[nc, tcc]))
    ledger2 = ledger._replace(
        owner=ledger.owner.at[nc].set(_i32(d)), row=ledger.row.at[nc].set(dr),
        version=ledger.version.at[nc].add(1),
        transfer_count=ledger.transfer_count.at[nc].add(1),
        hist_src=put(ledger.hist_src, s), hist_dst=put(ledger.hist_dst, d), hist_t=put(ledger.hist_t, now),
        hist_overflow=ledger.hist_overflow + jnp.where(fits, 0, 1).astype(jnp.int32),
        route_cost_s=ledger.route_cost_s + txn.route_s)
    ledger2 = _release(ledger2, txn)                                                    # 620행
    return tree_where(ok, worlds2, worlds), tree_where(ok, ledger2, ledger), code


# ───────────────────────────────────────────────── try_* (457-473 · 631-650행)
def _try(worlds, ledger, prep, now, g, k0, dst_row=None):
    ledger1, txn, pcode = prep(ledger)
    worlds2, ledger2, ccode = commit(worlds, ledger1, txn, now, g, k0=k0, dst_row=dst_row)   # prepare 실패 → txn 닫힘 → 무변경
    ledger3 = rollback(ledger2, txn)                                                    # commit 실패분만 실제로 풀린다 (멱등)
    ok = (pcode == OK) & (ccode == OK)
    code = jnp.where(pcode != OK, pcode, ccode)
    return worlds2, ledger3, ok, code


def try_transfer(worlds: BlockWorld, ledger: TxnLedger, now, job, dst, *, route_s, travel_s, g: Geom,
                 k0: int = 0, capacity_margin: int = CAPACITY_MARGIN, dst_row=None):
    """v5 `try_transfer` (631-650행) — prepare → commit, 실패면 rollback. 반환 (worlds', ledger', ok, 코드).
    dst_row: commit 참조 (None = 전역 행 규약)."""
    return _try(worlds, ledger,
                lambda lg: prepare_transfer(worlds, lg, now, job, dst, route_s=route_s, travel_s=travel_s,
                                            capacity_margin=capacity_margin), now, g, k0, dst_row)


def try_pre_gate_transfer(worlds: BlockWorld, ledger: TxnLedger, now, job, dst, *, travel_s, route_delta_s=0.0,
                          g: Geom, k0: int = 0, capacity_margin: int = CAPACITY_MARGIN, max_transfers: int = 1,
                          dst_row=None):
    """v5 `try_pre_gate_transfer` (457-473행). 반환 (worlds', ledger', ok, 코드). dst_row: commit 참조."""
    return _try(worlds, ledger,
                lambda lg: prepare_pre_gate_transfer(worlds, lg, now, job, dst, travel_s=travel_s,
                                                     route_delta_s=route_delta_s, capacity_margin=capacity_margin,
                                                     max_transfers=max_transfers), now, g, k0, dst_row)


# ───────────────────────────────────────────────── 이연 (294-374행)
def defer_admitted_entry(worlds: BlockWorld, ledger: TxnLedger, now, job, delta_s, *, max_deferrals: int = 1):
    """v5 `defer_admitted_entry` — 진입 전 트럭의 진입 시각을 delta 만큼 미룬다. 반환 (worlds', ledger', 코드).

    검사 순서: 미등록 → 이연량 양수 → 이연 상한 → 이미 gate-in → 상태(PLANNED) → 이연 도착 창 밖 → [장부 항목: 구조상 불가].
    변경: 큐 BLOCK_ARRIVAL 제거·재push(new_arr) · gate_in_s = A+Δ (v5 actual_gate_in·notified_gate_in_s·장부 A 셋이
    같은 값 — 열 하나) · actual_arrival_s += Δ · provided_eta_s += Δ (est·eta) · 원장 version+1·deferrals+1·deferred+=Δ.
    ★`appt_s`(예약 원점 = v5 `Job.appointment_gate_time`)를 **건드리지 않는다** — 비용 은닉 금지 계약(302-304행).
      기사 외부 대기 = 새 `gate_in_s` − `appt_s` 로 계상된다 (host_terminal.from_terminal_world "deferrals").
      `notice_s`(= max(0, 도착 − lead))는 통지 시각이지 예약 원점이 아니다 — 섞으면 대기가 lead 만큼 부푼다.
    """
    N, B = ledger.n, worlds.clock.shape[0]
    o = worlds.orders
    job = _i32(job)
    now = _f(now)
    delta = _f(delta_s)
    nc = jnp.clip(job, 0, N - 1)
    owner = ledger.owner[nc]
    registered = (job >= 0) & (job < N) & (owner >= 0)                                  # 309-311행
    s = jnp.clip(owner, 0, B - 1)
    r = jnp.clip(ledger.row[nc], 0, o.block.shape[1] - 1)
    a = o.gate_in_s[s, r]
    not_in = (a < EMPTY_TIME) & (a > now + EPS_WINDOW)                                  # 316행
    planned = (o.block[s, r] >= 0) & (o.status[s, r] == JS_PLANNED)                     # 319-321행
    new_a = a + delta                                                                   # 323행
    new_arr = o.actual_arrival_s[s, r] + delta                                          # 324행
    in_win = new_arr <= worlds.end_s[s]                                                 # 325행
    code = _first_fail([(registered, R_UNREGISTERED), (delta > 0, R_NONPOSITIVE_DELTA),
                        (ledger.entry_deferrals[nc] < _i32(max_deferrals), R_MAX_DEFERRALS),
                        (not_in, R_ALREADY_GATE_IN), (planned, R_STATUS), (in_win, R_OUT_OF_WINDOW)])
    ok = code == OK
    # ── 변경 구간 (334-366행) ──
    qs = remove_block_arrival(_row(worlds.queue, s), r)
    qs = push_event(qs, new_arr, EV_BLOCK_ARRIVAL, r)
    q2 = _set_row(worlds.queue, s, qs)
    eta = o.provided_eta_s[s, r]
    o2 = o._replace(gate_in_s=o.gate_in_s.at[s, r].set(new_a),
                    actual_arrival_s=o.actual_arrival_s.at[s, r].set(new_arr),
                    provided_eta_s=o.provided_eta_s.at[s, r].set(jnp.where(eta < EMPTY_TIME, eta + delta, eta)))
    worlds2 = worlds._replace(orders=o2, queue=q2)
    ledger2 = ledger._replace(version=ledger.version.at[nc].add(1),
                              entry_deferrals=ledger.entry_deferrals.at[nc].add(1),
                              entry_deferred_s=ledger.entry_deferred_s.at[nc].add(delta))
    return tree_where(ok, worlds2, worlds), tree_where(ok, ledger2, ledger), code


def try_defer_admitted_entry(worlds: BlockWorld, ledger: TxnLedger, now, job, delta_s, *, max_deferrals: int = 1):
    """v5 `try_defer_admitted_entry` (368-374행). 반환 (worlds', ledger', ok, 코드)."""
    w2, l2, code = defer_admitted_entry(worlds, ledger, now, job, delta_s, max_deferrals=max_deferrals)
    return w2, l2, code == OK, code


# ───────────────────────────────────────────────── 불변식 (653-666행)
def check_invariants(worlds: BlockWorld, ledger: TxnLedger, *, conserve: bool = False,
                     n_extra_truck_rows: int = 0) -> jnp.ndarray:
    """v5 `check_invariants` (653-666행) 의 배열판. () bool (True = 통과).

    ① 등록 트럭마다 (owner, row) 에 살아 있다 (소유자 없음·owner 불일치)
    ② 두 트럭이 같은 (owner, row) 를 갖지 않는다 (이중 소유)
    ③ ★보존 — **살아 있는 외부트럭 행의 총수 = 등록 수**. v5 `len(seen) == len(records)` 에 해당한다.
       이것이 없으면 commit 이 src 행을 비우는 데 실패해 같은 트럭이 두 블록 사본에 동시에 살아 있어도(이중 소유),
       또 원장 밖의 트럭 행이 유령으로 남아도 True 가 나왔다 (v5 가 이 검사를 넣은 이유가 바로 그 경로다).
    ③은 `conserve=True` 일 때만 센다 — 원장이 **외부트럭만** 담는 경로(조정자 TerminalLedgerArrays)에서만 등식이
    성립하기 때문이다. 시나리오가 처음부터 외부트럭을 들고 있으면 그 수를 `n_extra_truck_rows` 로 알려 준다.
    (v5 원장을 통째로 옮긴 `txn_ledger_from_v5` 경로는 본선 작업까지 records 에 있어 등식이 다르다 → 기본 False.)
    """
    live = live_mask(worlds)                                                            # (B,R)
    B, R = live.shape
    reg = ledger.owner >= 0
    oc = jnp.clip(ledger.owner, 0, B - 1)
    rc = jnp.clip(ledger.row, 0, R - 1)
    at_owner = live[oc, rc] & (ledger.row >= 0) & (ledger.row < R)
    occ = jnp.zeros((B, R), jnp.int32).at[oc, rc].add(jnp.where(reg, 1, 0).astype(jnp.int32))
    base = jnp.all(~reg | at_owner) & jnp.all(occ <= 1)
    if not conserve:
        return base
    # ③ 살아 있는 외부트럭 행 (is_external & exit_travel 있음 = 장부 계약을 통과한 투입 트럭) 총수 == 등록 수
    truck_rows = live & worlds.orders.is_external & (worlds.orders.exit_travel_s >= 0.0)
    conserved = (jnp.sum(truck_rows.astype(jnp.int32))
                 == jnp.sum(reg.astype(jnp.int32)) + jnp.int32(int(n_extra_truck_rows)))
    return base & conserved


# ───────────────────────────────────────────────── 호스트 (v5 ↔ 배열)
def txn_ledger_from_v5(mbt, job_index: dict[str, int], block_index: dict[str, int], cont_base,
                       *, n: int, max_t: int = 1, max_open: int = 4, row_of: dict[str, int] | None = None) -> TxnLedger:
    """v5 `MultiBlockTerminal` 의 원장·조정자 상태 → `TxnLedger` (호스트 · 시험용).

    job_index: job_id → 원장 키 n · block_index: 블록 이름 → b · cont_base: 블록별 초기 컨테이너 수 (B,).
    row_of: job_id → owner 사본의 행 (기본 None = job_index 와 같음, 전역 행 규약).
    `_open_txn` 은 정렬해 앞 칸부터 넣는다 (집합이라 순서가 없다 — 슬롯 위치는 의미 없음).
    """
    B = len(block_index)
    lg = empty_txn_ledger(n, B, max_t=max_t, max_open=max_open)
    row = np.asarray(lg.row).copy()
    owner = np.full((n,), EMPTY_ID, np.int32); origin = np.full((n,), EMPTY_ID, np.int32)
    version = np.zeros((n,), np.int32); tcount = np.zeros((n,), np.int32)
    hs = np.full((n, max_t), EMPTY_ID, np.int32); hd = np.full((n, max_t), EMPTY_ID, np.int32)
    ht = np.full((n, max_t), EMPTY_TIME, np.float64)
    locked = np.zeros((n,), bool); ndef = np.zeros((n,), np.int32); sdef = np.zeros((n,), np.float64)
    for jid, rec in mbt.ledger.records.items():
        i = job_index[jid]
        owner[i] = block_index[rec.owner]; origin[i] = block_index[rec.origin_block]
        if row_of is not None:
            row[i] = row_of[jid]
        version[i] = rec.version; tcount[i] = rec.transfer_count
        if rec.transfer_count > max_t:
            raise ValueError(f"{jid}: transfer_count {rec.transfer_count} > max_t {max_t}")
        for k, (a, b, t) in enumerate(rec.transfer_history[:max_t]):
            hs[i, k], hd[i, k], ht[i, k] = block_index[a], block_index[b], t
        locked[i] = bool(rec.locked); ndef[i] = rec.entry_deferrals; sdef[i] = rec.entry_deferred_s
    reserved = np.zeros((B,), np.int32)
    for bid, r in mbt._reserved_inbound.items():
        reserved[block_index[bid]] = r
    open_ids = sorted(mbt._open_txn)
    if len(open_ids) > max_open:
        raise ValueError(f"열린 트랜잭션 {len(open_ids)} > max_open {max_open}")
    ot = np.full((max_open,), EMPTY_ID, np.int32)
    ot[:len(open_ids)] = open_ids
    return lg._replace(owner=jnp.asarray(owner), row=jnp.asarray(row), origin=jnp.asarray(origin), version=jnp.asarray(version),
                       transfer_count=jnp.asarray(tcount), hist_src=jnp.asarray(hs), hist_dst=jnp.asarray(hd),
                       hist_t=jnp.asarray(ht), locked=jnp.asarray(locked), entry_deferrals=jnp.asarray(ndef),
                       entry_deferred_s=jnp.asarray(sdef), reserved_inbound=jnp.asarray(reserved),
                       cont_base=jnp.asarray(np.asarray(cont_base, np.int32)),
                       route_cost_s=jnp.asarray(float(mbt.route_cost_s), F),
                       txn_seq=jnp.asarray(int(mbt._txn_seq), jnp.int32), open_txn=jnp.asarray(ot))


def txn_ledger_to_v5(ledger: TxnLedger, job_ids, block_ids) -> dict:
    """`TxnLedger` → v5 모양의 파이썬 값 (시험이 v5 JobRecord 와 항목별 대조).

    반환 {"records": {job_id: {owner, origin_block, version, transfer_count, transfer_history, locked,
    entry_deferrals, entry_deferred_s}}, "reserved_inbound": {block: n}, "route_cost_s", "txn_seq", "open_txn"}.
    """
    owner = np.asarray(ledger.owner); origin = np.asarray(ledger.origin); rowv = np.asarray(ledger.row)
    version = np.asarray(ledger.version); tcount = np.asarray(ledger.transfer_count)
    hs, hd, ht = np.asarray(ledger.hist_src), np.asarray(ledger.hist_dst), np.asarray(ledger.hist_t)
    locked = np.asarray(ledger.locked); ndef = np.asarray(ledger.entry_deferrals); sdef = np.asarray(ledger.entry_deferred_s)
    recs = {}
    for i, jid in enumerate(job_ids):
        if owner[i] < 0:
            continue
        k = int(tcount[i])
        recs[jid] = {
            "owner": block_ids[int(owner[i])], "row": int(rowv[i]), "origin_block": block_ids[int(origin[i])],
            "version": int(version[i]), "transfer_count": k,
            "transfer_history": tuple((block_ids[int(hs[i, j])], block_ids[int(hd[i, j])], float(ht[i, j]))
                                      for j in range(min(k, hs.shape[1]))),
            "locked": bool(locked[i]), "entry_deferrals": int(ndef[i]), "entry_deferred_s": float(sdef[i]),
        }
    return {"records": recs,
            "reserved_inbound": {b: int(v) for b, v in zip(block_ids, np.asarray(ledger.reserved_inbound))},
            "route_cost_s": float(ledger.route_cost_s), "txn_seq": int(ledger.txn_seq),
            "open_txn": sorted(int(x) for x in np.asarray(ledger.open_txn) if x >= 0),
            "open_overflow": int(ledger.open_overflow), "hist_overflow": int(ledger.hist_overflow)}
