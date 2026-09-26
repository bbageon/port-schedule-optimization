"""게이트 투입을 **배열로** — 조각 6 · key=admit ([[YR-327]]).

v5 는 두 층이다:
  · `terminal_stream.ScheduledAnnouncer.review(mbt, t)` (802-825행) — 검토 시각 t 가 60초 격자면
    그 격자 버킷의 명단 항목을 **명단 순서대로** 하나씩 투입 시도하고 ADMIT / SKIP / SKIP_TAIL 원장을 남긴다.
  · `multiblock.MultiBlockTerminal.admit_external_job(bid, job, gate_in_s, travel_s)` (243-291행) —
    검사 7개(등록 중복 · 블록 · 장부 · exit_travel · 시각 · 용량 · 반출 대상) 뒤 job 삽입 · BLOCK_ARRIVAL push ·
    시간 장부 A 등록 · 전역 원장 등록.

배열판에서는 **오더 행이 이미 있다** (호스트 `attach_schedule` 이 블록 오더 표의 뒤 칸에 명단 트럭을 미리 앉힌다 —
정적 속성만 채우고 `block=-1` 로 둬 엔진에는 없는 오더다). 투입 = 그 행의 `block`(= 소유 블록 · 원장 owner)·
`gate_in_s`(A · 시간 장부 등록)·`appt_s`(예약 원점)·`actual_arrival_s`·`provided_eta_s`·`notice_s`·`travel_s` 를
채우고 BLOCK_ARRIVAL 을 push 하는 것뿐이다. `_a_sorted` 포인터 보정(285-289행)은 엔진의 닫힌 식 적분(`engine_step._terminal_walk`)이
대신하므로 사라진다.

■ 순서 규약 (조사 지도 multiblock ■)
  · 같은 에폭 안 투입은 **순차 의존** — 앞 투입이 뒤 투입의 `free_slots`(pending 계수)에 영향 (806-825행).
    → 블록마다 그 에폭의 명단 항목을 **명단 색인 오름차순** `lax.scan` 으로 처리한다. 블록 사이는 독립이므로
    `vmap`. 같은 블록 안 명단 순서 = 도착 오름차순 = 큐 seq 순서가 그대로 재현된다 (v5 seq 는 블록별 카운터).
  · 원장 행 순서(전 블록 섞인 명단 순서)는 호스트 `ledger_rows` 가 명단 색인으로 되돌린다.

■ 검사 순서와 코드 (v5 raise 순서 그대로 — 첫 실패 사유가 코드)
    0 ADMIT
    1 SKIP_TAIL    arrival + travel > 스케줄러 end_s          (announcer 808행; end_s None → 검사 없음 = +inf)
    2 SKIP_DUP     이미 등록된 작업 — **전역 원장** `registered[t]` (257행 `jid in ledger.records`).
                  원장을 안 받으면 목적 블록 그 행의 `block ≥ 0` 만 본다 (id 가 `{block}:{job}` 라 답은 같다).
    3 SKIP_EXIT    exit_travel_s 결측 (행 값 < 0)               (264행)
    4 SKIP_TIME    gate_in < clock−1e-9 · arr ≤ clock+1e-9 · arr > end   (267행)
    5 SKIP_CAP     반입인데 free_slots ≤ margin                 (269행)
    6 SKIP_TARGET  반출인데 대상 없음(-1) 또는 야드에 없음      (271-274행 `tgt not in stacks.containers`)
   -1 없음 (버킷 빈 칸)
  259·262행(블록 없음·장부 없음)은 호스트 구성 계약이라 배열에 없다 (블록 색인은 항상 유효 · 장부는 열로 항상 있다).

■ free_slots (230-240행)
    phys − used − pending − reserved
    phys    = bay·row·tier (Geom)          used = Σ conts.c_alive (= len(stacks.containers))
    pending = #(block ≥ 0 & status PLANNED & flow ∈ {GATE_IN, VESSEL_DISCHARGE})   reserved = 이송 예약 (B,) (이송 하위 시스템)
  scan 안에서는 pending 을 carry 로 들고 반입 투입마다 +1 — 매 단계 다시 세는 것과 같은 값이다.

■ 왜 명단은 호스트에서 만드나
  by_epoch 버킷(`round((notify // period) * period, 6)` 을 dict 키로)은 `slot = int(notify // period)` 정수와 같다.
  명단 생성 자체(난수)는 v5 파이썬이 하고, 여기서는 (T,) 열과 (E,B,M) 버킷으로만 받는다 (조사 지도 ■ 난수).

■ 이어 돌리기
  전부 순수 함수 — 상태는 `BlockWorld` 묶음(앞축 B)과 `ScheduleArrays` 뿐. `tree_to_numpy`/`tree_to_jax` 로 세션 사이에
  저장·복원한다 (80초 WSL 창 대응 — README ⚠️ 환경).
"""
from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .geom import Geom
from .state import (EV_BLOCK_ARRIVAL, FL_GATE_IN, FL_GATE_OUT, FL_VESSEL_DISCHARGE, JS_PLANNED,
                    ST_NOTICE, SZ_FT20, SZ_FT40, BlockWorld)
from .vessel import push_if

__all__ = [
    "PERIOD_S", "CAPACITY_MARGIN", "ADMIT", "SKIP_TAIL", "SKIP_DUP", "SKIP_EXIT", "SKIP_TIME", "SKIP_CAP",
    "SKIP_TARGET", "NO_ENTRY", "CODE_NAMES", "code_of_reason", "ScheduleArrays",
    "admission_epochs_s", "slot_of", "attach_schedule", "schedule_arrays",
    "free_slots", "free_slots_batch", "admit_block", "admit_epoch", "admit_epoch_jit",
    "ledger_rows", "stack_worlds", "world_at", "tree_to_numpy", "tree_to_jax",
]

F = TIME_DTYPE
#: v5 `WIP_ADMISSION_PERIOD_S` (terminal_stream.py:380) · `EPOCH_S` (stage/orders.py:32)
PERIOD_S = 60.0
#: v5 `multiblock.CAPACITY_MARGIN` (26행)
CAPACITY_MARGIN = 2
#: v5 `_EPS` — multiblock.py:267 의 1e-9
EPS = 1e-9

#: 투입 결과 코드 (머리말 ■ 검사 순서)
ADMIT, SKIP_TAIL, SKIP_DUP, SKIP_EXIT, SKIP_TIME, SKIP_CAP, SKIP_TARGET = range(7)
NO_ENTRY = -1
CODE_NAMES: tuple[str, ...] = ("ADMIT", "SKIP_TAIL", "SKIP_DUP", "SKIP_EXIT", "SKIP_TIME", "SKIP_CAP", "SKIP_TARGET")
#: v5 TransferError 문구 → 코드 (announcer 원장의 reason 문자열을 분류한다 — 시험 대조용)
_REASON_KEYS: tuple[tuple[str, int], ...] = (
    ("이미 등록된 작업", SKIP_DUP), ("exit_travel_s 결측", SKIP_EXIT), ("투입시각 무효", SKIP_TIME),
    ("용량 부족", SKIP_CAP), ("반출 대상 부재", SKIP_TARGET),
)


def code_of_reason(reason: str) -> int:
    """v5 `ScheduledAnnouncer.ledger` 의 SKIP reason 문자열 → 코드. 모르는 문구는 -2 (조용히 넘기지 않는다)."""
    for key, code in _REASON_KEYS:
        if key in reason:
            return code
    return -2


# ───────────────────────────────────────────────── 명단 배열
class ScheduleArrays(NamedTuple):
    """명단 T 건 — **명단 순서 그대로** (= `diurnal_arrivals` 정렬 = stage/orders.py 의 Φ 합산 순서 정본).

    항목당 announcer 가 읽는 값만 든다 (`arrival_s`·`travel_s`·`end_s` 비교, `_job_from_entry` 의 통지·예측 시각).
    작업 속성(flow·규격·반출 대상·exit_travel)은 `attach_schedule` 이 오더 행에 미리 앉혔으므로 행에서 읽는다.
    """

    slot: jnp.ndarray          # (T,) int32  통지 격자 = int(max(0, arrival − lead) // period)  (announcer 795-796행)
    arrival_s: jnp.ndarray     # (T,) f64    도착(= gate-in) 시각 — `e["arrival_s"]` (round 3)
    notify_s: jnp.ndarray      # (T,) f64    통지 시각 max(0, arrival − lead)  → `orders.notice_s`
    travel_s: jnp.ndarray      # (T,) f64    게이트→블록 실현 주행 `e["travel_s"]` (블록 도착 = arrival + travel)
    travel_base_s: jnp.ndarray  # (T,) f64   기대 주행 `e["travel_base_s"]` → provided_eta = arrival + base (491행)
    block: jnp.ndarray         # (T,) int32  목적 블록 (layout.ids 순서 색인)
    row: jnp.ndarray           # (T,) int32  그 블록 오더 표의 행 (attach_schedule 이 정함)
    bucket: jnp.ndarray        # (E,B,M) int32  에폭 e · 블록 b 의 명단 색인들 (명단 순서 · 빈 칸 -1)
    epoch_s: jnp.ndarray       # (E,) f64    격자 시각 i·period (admission_epochs)

    @property
    def t(self) -> int:
        return int(self.slot.shape[0])

    @property
    def e(self) -> int:
        return int(self.bucket.shape[0])

    @property
    def b(self) -> int:
        return int(self.bucket.shape[1])

    @property
    def m(self) -> int:
        return int(self.bucket.shape[2])


def admission_epochs_s(observe_s: float, period_s: float = PERIOD_S) -> np.ndarray:
    """v5 `admission_epochs(obs)` (827-831행) — `i * period` 를 파이썬 float 로 (같은 값)."""
    n = int(float(observe_s) // period_s)
    return np.asarray([i * period_s for i in range(n + 1)], np.float64)


def slot_of(t_s: float, period_s: float = PERIOD_S) -> int:
    """검토 시각 → 격자 슬롯. 격자 밖이면 -1 (v5 `on_grid` 1e-6 허용 · 804행 `return`)."""
    r = float(t_s) % period_s
    if not (r < 1e-6 or period_s - r < 1e-6):
        return -1
    return int(round(float(t_s) / period_s))


def attach_schedule(world: BlockWorld, tables, entries: Sequence[dict], *, block_idx: int,
                    rows: Sequence[int] | None = None):
    """블록 하나의 명단 트럭을 오더 표 뒤 칸에 **미리 앉힌다** (호스트 · 한 번).

    `entries` = 이 블록으로 가는 명단 항목들(명단 순서). 행 = `tables.n0` 부터 차례로 (같은 job_id 가 다시 오면
    같은 행 — v5 는 그 항목을 '이미 등록' 으로 거른다). 정적 속성만 채운다:
        flow · is_external · is_store · inbound_size · inbound_cont(=c0+행, 예비칸 이름 IN_{job_id}) · target_cont ·
        exit_travel_s(None → -1) · travel_s · release_s 0 · status PLANNED
    `block` 은 -1 그대로 (엔진에 없는 오더) — 투입이 채운다. `gate_in_s`·`actual_arrival_s`·`provided_eta_s` 는 +inf.
    기존 행(시나리오 작업)의 `block` 은 `block_idx` 로 통일한다 (owner 열).

    ★장부 모드: is_external·exit_travel_s 를 미리 채우므로 `engine_step.ledger_mode` 가 처음부터 참이다 —
      v5 `ensure_time_ledger` 가 빈 장부를 켜 두는 것과 같다 (블록 도착 전엔 두 적분 모두 0 이라 관측 차이 없음).

    반환 (world, tables', row_of_job) — tables' 는 job_ids·cont_ids·n0 를 늘린 새 `IdTables`.
    """
    o = {f: np.asarray(getattr(world.orders, f)).copy() for f in world.orders._fields}
    c_size = np.asarray(world.conts.c_size).copy()
    n_max = int(o["block"].shape[0])
    n0, c0 = int(tables.n0), int(tables.c0)
    cont_index = tables.cont_index
    job_ids = list(tables.job_ids)
    cont_ids = list(tables.cont_ids)
    row_of_job: dict[str, int] = {j: i for i, j in enumerate(job_ids)}
    o["block"][:n0] = np.where(o["block"][:n0] >= 0, block_idx, o["block"][:n0])
    nxt = n0
    for k, e in enumerate(entries):
        jid = e["job_id"]
        if jid in row_of_job:
            if rows is not None and rows[k] != row_of_job[jid]:
                raise ValueError(f"{jid}: 같은 작업이 다른 행 {rows[k]}·{row_of_job[jid]} 에")
            continue                                    # 중복 항목 — 같은 행 (v5 는 SKIP_DUP)
        n = nxt if rows is None else int(rows[k])
        if n >= n_max:
            raise ValueError(f"오더 칸 부족 — 행 {n} ≥ n_max {n_max} (시나리오 {n0} + 명단 {len(entries)})")
        if c0 + n >= len(cont_ids):
            raise ValueError(f"컨테이너 칸 부족 — 예비칸 {c0 + n} ≥ {len(cont_ids)}")
        row_of_job[jid] = n
        nxt = max(nxt, n + 1)
        is_in = e["flow"] != "GATE_OUT"
        o["block"][n] = EMPTY_ID
        o["flow"][n] = FL_GATE_IN if is_in else FL_GATE_OUT
        o["stage"][n] = ST_NOTICE
        o["status"][n] = JS_PLANNED
        o["is_external"][n] = True
        o["is_vessel"][n] = False
        o["is_store"][n] = is_in                          # _job_from_entry 497-501행: 반입만 inbound_size
        if is_in:
            sz = SZ_FT40 if e.get("size_ft40") else SZ_FT20
            o["inbound_size"][n] = sz
            o["inbound_cont"][n] = c0 + n                 # engine.py:604 `IN_{job_id}` 예비칸
            c_size[c0 + n] = sz
        else:
            o["inbound_size"][n] = EMPTY_ID
            o["inbound_cont"][n] = EMPTY_ID
        tgt = e.get("target")
        o["target_cont"][n] = cont_index.get(tgt, EMPTY_ID) if (tgt is not None and not is_in) else EMPTY_ID
        ex = e.get("exit_travel_s")
        o["exit_travel_s"][n] = -1.0 if ex is None else float(ex)
        o["travel_s"][n] = float(e["travel_s"])
        o["release_s"][n] = 0.0                           # Job(release_time=0.0)
        o["vessel"][n] = EMPTY_ID
        o["assigned_crane"][n] = EMPTY_ID
        o["rehandles"][n] = 0
        o["deadline_s"][n] = EMPTY_TIME
        o["provided_eta_s"][n] = EMPTY_TIME
        o["gate_in_s"][n] = EMPTY_TIME
        o["actual_arrival_s"][n] = EMPTY_TIME
        o["notice_s"][n] = EMPTY_TIME
        o["appt_s"][n] = EMPTY_TIME                       # 예약 원점 — 투입 때 찍는다
        o["waiting"][n] = False
        o["in_block"][n] = False
        o["wait_sample_s"][n] = np.nan
        while len(job_ids) <= n:
            job_ids.append(f"#{len(job_ids)}")
        job_ids[n] = jid
        cont_ids[c0 + n] = f"IN_{jid}"
    world = world._replace(orders=world.orders._replace(**{f: jnp.asarray(v) for f, v in o.items()}),
                           conts=world.conts._replace(c_size=jnp.asarray(c_size)))
    tables2 = dataclasses.replace(tables, job_ids=tuple(job_ids), cont_ids=tuple(cont_ids), n0=nxt)
    return world, tables2, row_of_job


def schedule_arrays(schedule: Sequence[dict], *, block_index: dict[str, int], row_of_job: dict[str, int],
                    lead_s, n_epochs: int, period_s: float = PERIOD_S,
                    notify_s: Sequence[float] | None = None) -> tuple[ScheduleArrays, dict]:
    """v5 명단(dict 열) → `ScheduleArrays`. 버킷은 announcer 795-797행과 같은 규칙, 명단 순서 보존.

    `row_of_job`: job_id → 그 블록 오더 표의 행 (`attach_schedule` 이 돌려준 것을 블록별로 합친 dict).
    ★`lead_s` 는 **스칼라 하나 또는 항목별 (T,) 열**이다 — v2 `ScheduledAnnouncer` 는 전원 같은 리드(스칼라)지만
      v3 `stage/orders.V3Announcer` 는 트럭마다 자기 리드(`e["lead_s"]`, orders.py:62-69·116행)를 쓴다. 항목별
      열을 주면 통지 시각·에폭 버킷이 트럭마다 갈린다 (전원 같은 값을 주면 예전과 비트 동일).
      명단 항목이 `"lead_s"` 키를 들고 있으면 그것을 우선 쓴다 (v3 경로 — 호출자가 따로 안 넘겨도 된다).
    `notify_s`: 통지 시각을 직접 준다 (호스트가 이미 계산해 둔 값 — lead 재계산 없이 그대로 쓴다).
    `n_epochs`: 격자 수 E (= len(admission_epochs)). 슬롯이 E 이상인 항목은 v5 도 영원히 검토되지 않는다 —
    버킷에 넣지 않고 `info["dropped"]` 로 센다 (조용히 자르지 않는다).
    반환 (sched, info) — info: T · E · B · M · dropped.
    """
    T = len(schedule)
    B = len(block_index)
    E = int(n_epochs)
    slot = np.full((T,), EMPTY_ID, np.int32)
    arrival = np.zeros((T,), np.float64)
    notify = np.zeros((T,), np.float64)
    travel = np.zeros((T,), np.float64)
    base = np.zeros((T,), np.float64)
    block = np.full((T,), EMPTY_ID, np.int32)
    row = np.full((T,), EMPTY_ID, np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    dropped = 0
    lead_arr = np.asarray(lead_s, np.float64)
    if lead_arr.ndim == 0:
        lead_arr = np.full((T,), float(lead_arr), np.float64)
    elif lead_arr.shape != (T,):
        raise ValueError(f"lead_s 는 스칼라 또는 (T={T},) 열 — 받은 모양 {lead_arr.shape}")
    nt_arr = None if notify_s is None else np.asarray(notify_s, np.float64).reshape(-1)
    if nt_arr is not None and nt_arr.shape != (T,):
        raise ValueError(f"notify_s 는 (T={T},) 열 — 받은 모양 {nt_arr.shape}")
    for i, e in enumerate(schedule):
        a = float(e["arrival_s"])
        li = float(e["lead_s"]) if ("lead_s" in e and e["lead_s"] is not None) else float(lead_arr[i])
        nt = float(nt_arr[i]) if nt_arr is not None else max(0.0, a - li)    # 795행 · V3Announcer 116행
        s = int(nt // period_s)                             # 796행 round((notify // period) * period, 6) 의 정수판
        arrival[i], notify[i], travel[i] = a, nt, float(e["travel_s"])
        base[i] = float(e["travel_base_s"])
        block[i] = block_index[e["block"]]
        row[i] = row_of_job[e["job_id"]]
        slot[i] = s
        if s >= E:
            dropped += 1
            continue
        buckets.setdefault((s, int(block[i])), []).append(i)
    M = max([len(v) for v in buckets.values()] + [0])
    bucket = np.full((E, B, M), EMPTY_ID, np.int32)
    for (s, b), idx in buckets.items():
        bucket[s, b, :len(idx)] = idx
    epoch_s = np.asarray([i * period_s for i in range(E)], np.float64)
    sched = ScheduleArrays(slot=jnp.asarray(slot), arrival_s=jnp.asarray(arrival, F), notify_s=jnp.asarray(notify, F),
                           travel_s=jnp.asarray(travel, F), travel_base_s=jnp.asarray(base, F),
                           block=jnp.asarray(block), row=jnp.asarray(row), bucket=jnp.asarray(bucket),
                           epoch_s=jnp.asarray(epoch_s, F))
    return sched, {"T": T, "E": E, "B": B, "M": M, "dropped": dropped}


# ───────────────────────────────────────────────── 용량
def _pending_inbound(o) -> jnp.ndarray:
    """multiblock.py:237-239 — PLANNED 이고 flow ∈ {GATE_IN, VESSEL_DISCHARGE} 인 (존재하는) 오더 수. () int32."""
    m = (o.block >= 0) & (o.status == JS_PLANNED) & ((o.flow == FL_GATE_IN) | (o.flow == FL_VESSEL_DISCHARGE))
    return jnp.sum(m).astype(jnp.int32)


def free_slots(world: BlockWorld, g: Geom, reserved=0) -> jnp.ndarray:
    """v5 `MultiBlockTerminal.free_slots(bid)` (230-240행) — 블록 하나. () int32.

    `reserved` = `_reserved_inbound[bid]` (이송 prepare 의 예약, 이송 하위 시스템 소관; 기본 0)."""
    phys = int(g.bay_count) * int(g.row_count) * int(g.tier_max)                     # 236행
    used = jnp.sum(world.conts.c_alive).astype(jnp.int32)                            # 235행 len(stk.containers)
    return (jnp.int32(phys) - used - _pending_inbound(world.orders)
            - jnp.asarray(reserved, jnp.int32))                                      # 240행


def free_slots_batch(W: BlockWorld, g: Geom, reserved=None) -> jnp.ndarray:
    """블록 묶음 (B,…) → (B,) int32."""
    B = int(W.clock.shape[0])
    r = jnp.zeros((B,), jnp.int32) if reserved is None else jnp.asarray(reserved, jnp.int32)
    return jax.vmap(lambda w, rr: free_slots(w, g, rr))(W, r)


# ───────────────────────────────────────────────── 투입 (블록 하나 · 에폭 하나)
def _set1(arr, i, v, valid):
    """arr[i] = v (valid 일 때만). -1 은 clip + where 로 막는다 (engine_step._set1 규약)."""
    ic = jnp.clip(jnp.asarray(i, jnp.int32), 0, arr.shape[0] - 1)
    return arr.at[ic].set(jnp.where(valid, jnp.asarray(v, arr.dtype), arr[ic]))


def admit_block(world: BlockWorld, b, entries, sched: ScheduleArrays, g: Geom, *, margin: int = CAPACITY_MARGIN,
                reserved=0, end_ann=EMPTY_TIME, registered=None):
    """블록 `b` 의 이번 에폭 명단 `entries` (M,) int32 (명단 색인 · -1 빈 칸) 를 **순서대로** 투입 시도.

    = announcer 806-823행 루프 + admit_external_job 검사·변경 (머리말 ■ 검사 순서). 세계는 이 블록이 검토 시각에
    park 한 상태여야 한다 (`clock` = t). 반환 (world', codes (M,) int32).
    `end_ann` = announcer `end_s` (None 이면 +inf). `reserved` = 이 블록의 이송 예약 수.
    `registered` (T,) bool = **전역 원장** 등록 여부 (v5 257행 `jid in self.ledger.records`). 주면 중복 판정이
    v5 와 같은 의미가 된다 — 안 주면 목적 블록 자기 행만 본다 (id 가 `{block}:{job}` 라 현 명명 규약에서는 같은 답).
    """
    M = int(entries.shape[0])
    T = sched.t
    if M == 0 or T == 0:
        return world, jnp.full((M,), NO_ENTRY, jnp.int32)
    N = world.n
    C = world.conts.c
    b = jnp.asarray(b, jnp.int32)
    phys = jnp.int32(int(g.bay_count) * int(g.row_count) * int(g.tier_max))
    used = jnp.sum(world.conts.c_alive).astype(jnp.int32)                    # 에폭 안에서 불변 (스택은 안 바뀐다)
    reserved = jnp.asarray(reserved, jnp.int32)
    end_ann = jnp.asarray(end_ann, F)
    margin_i = jnp.int32(int(margin))
    clock, end = world.clock, world.end_s

    def body(carry, i):
        w, pending = carry
        o = w.orders
        t = entries[i]
        valid = t >= 0
        tc = jnp.clip(t, 0, T - 1)
        row = sched.row[tc]
        rc = jnp.clip(row, 0, N - 1)
        arrival = sched.arrival_s[tc]                                          # gate_in_s = e["arrival_s"]
        travel = sched.travel_s[tc]
        arr = arrival + travel                                                 # 266행 · announcer 808행 (같은 f64 덧셈)
        flow, target, exit_t = o.flow[rc], o.target_cont[rc], o.exit_travel_s[rc]
        # ── 검사 (첫 실패가 코드) ──
        tail = arr > end_ann                                                   # 808행
        # 257행 `jid in self.ledger.records` — 전역 원장이 정본. 원장을 안 받으면 자기 행 fallback.
        dup = o.block[rc] >= 0
        if registered is not None:
            dup = dup | registered[tc]
        exit_bad = exit_t < 0.0                                                # 264행
        time_bad = (arrival < clock - EPS) | (arr <= clock + EPS) | (arr > end)   # 267행
        free = phys - used - pending - reserved                                # 240행
        cap_bad = (flow == FL_GATE_IN) & (free <= margin_i)                    # 269행
        tcc = jnp.clip(target, 0, C - 1)
        tgt_bad = (flow == FL_GATE_OUT) & ((target < 0) | ~w.conts.c_alive[tcc])   # 271-274행
        code = jnp.where(tail, SKIP_TAIL,
               jnp.where(dup, SKIP_DUP,
               jnp.where(exit_bad, SKIP_EXIT,
               jnp.where(time_bad, SKIP_TIME,
               jnp.where(cap_bad, SKIP_CAP,
               jnp.where(tgt_bad, SKIP_TARGET, ADMIT)))))).astype(jnp.int32)
        code = jnp.where(valid, code, NO_ENTRY).astype(jnp.int32)
        ok = valid & (code == ADMIT)
        # ── 변경 구간 (276-291행 + _job_from_entry 489-506행) ──
        o2 = o._replace(
            block=_set1(o.block, row, b, ok),                                  # sim.jobs[jid] = job · 원장 owner=bid
            gate_in_s=_set1(o.gate_in_s, row, arrival, ok),                    # actual_gate_in · TruckTimes(gate_in) · a_gate_in
            appt_s=_set1(o.appt_s, row, arrival, ok),                          # ★예약 원점 (appointment_gate_time, 503행) — 이후 불변
            actual_arrival_s=_set1(o.actual_arrival_s, row, arr, ok),         # actual_block_arrival
            provided_eta_s=_set1(o.provided_eta_s, row, arrival + sched.travel_base_s[tc], ok),   # est = gate_in + base (491행)
            notice_s=_set1(o.notice_s, row, sched.notify_s[tc], ok),           # 통지 시각 (arrival − lead)
            travel_s=_set1(o.travel_s, row, travel, ok),
            status=_set1(o.status, row, JS_PLANNED, ok),
            stage=_set1(o.stage, row, ST_NOTICE, ok))
        q2 = push_if(w.queue, ok, arr, EV_BLOCK_ARRIVAL, row)                 # 280행 queue.push(arr, BLOCK_ARRIVAL, jid)
        pending2 = pending + jnp.where(ok & (flow == FL_GATE_IN), 1, 0).astype(jnp.int32)   # 다음 free_slots 의 pending
        return (w._replace(orders=o2, queue=q2), pending2), code

    (w_out, _), codes = lax.scan(body, (world, _pending_inbound(world.orders)), jnp.arange(M, dtype=jnp.int32))
    return w_out, codes


# ───────────────────────────────────────────────── 투입 (전 블록 · 에폭 하나)
def admit_epoch(W: BlockWorld, sched: ScheduleArrays, slot, g: Geom, *, margin: int = CAPACITY_MARGIN,
                reserved=None, end_ann=EMPTY_TIME, registered=None):
    """검토 시각 하나 — 전 블록 vmap × 블록 안 명단 scan. `W` 는 블록 묶음 (잎마다 앞축 B).

    `slot` = 격자 슬롯(`slot_of(t)`); -1 또는 범위 밖이면 무엇도 하지 않는다 (v5 804행 격자 밖 return).
    반환 (W', codes (B,M) int32). jit 판 `admit_epoch_jit` (g·margin static).
    """
    B = int(W.clock.shape[0])
    if sched.b != B:
        raise ValueError(f"명단 블록 수 {sched.b} ≠ 세계 블록 수 {B}")
    E, M = sched.e, sched.m
    if M == 0 or E == 0:
        return W, jnp.full((B, M), NO_ENTRY, jnp.int32)
    slot = jnp.asarray(slot, jnp.int32)
    ok_slot = (slot >= 0) & (slot < E)
    entries = jnp.where(ok_slot, sched.bucket[jnp.clip(slot, 0, E - 1)], NO_ENTRY)     # (B,M)
    r = jnp.zeros((B,), jnp.int32) if reserved is None else jnp.asarray(reserved, jnp.int32)
    fn = partial(admit_block, sched=sched, g=g, margin=margin, end_ann=jnp.asarray(end_ann, F),
                 registered=registered)
    return jax.vmap(lambda w, b, en, rr: fn(w, b, en, reserved=rr))(W, jnp.arange(B, dtype=jnp.int32), entries, r)


admit_epoch_jit = jax.jit(admit_epoch, static_argnames=("g", "margin"))


# ───────────────────────────────────────────────── 원장 (호스트)
def ledger_rows(codes, sched: ScheduleArrays, slot: int, t_s: float, schedule: Sequence[dict]) -> list[dict]:
    """에폭 하나의 코드 (B,M) → v5 `ScheduledAnnouncer.ledger` 모양의 행들 (명단 순서 · 마지막 EPOCH 행).

    v5 행: ADMIT {t, event, job_id, block, flow, arrival_s} · SKIP {t, event, job_id, reason} · SKIP_TAIL {t, event, job_id}
    여기서는 SKIP 에 `reason` 대신 `code`·`reason_code` 이름을 둔다 (문구는 v5 가 값을 섞어 넣어 문자열 비교가 안 된다).
    격자 밖(slot < 0)이면 빈 목록 (v5 는 원장에 아무것도 안 남긴다).
    """
    if slot < 0 or slot >= sched.e:
        return []
    bucket = np.asarray(sched.bucket[slot])
    codes = np.asarray(codes)
    pairs: list[tuple[int, int]] = []
    for b in range(bucket.shape[0]):
        for m in range(bucket.shape[1]):
            ti = int(bucket[b, m])
            if ti >= 0:
                pairs.append((ti, int(codes[b, m])))
    pairs.sort()
    out: list[dict] = []
    for ti, code in pairs:
        e = schedule[ti]
        if code == ADMIT:
            out.append({"t": t_s, "event": "ADMIT", "job_id": e["job_id"], "block": e["block"],
                        "flow": e["flow"], "arrival_s": e["arrival_s"]})
        elif code == SKIP_TAIL:
            out.append({"t": t_s, "event": "SKIP_TAIL", "job_id": e["job_id"]})
        else:
            out.append({"t": t_s, "event": "SKIP", "job_id": e["job_id"], "code": code,
                        "reason_code": CODE_NAMES[code] if 0 <= code < len(CODE_NAMES) else str(code)})
    out.append({"t": t_s, "event": "EPOCH"})
    return out


# ───────────────────────────────────────────────── 블록 묶음 · 저장
def stack_worlds(worlds: Sequence[BlockWorld]) -> BlockWorld:
    """블록 세계들 → 앞축 B 묶음 (같은 모양이어야 한다 — 같은 프로파일·같은 칸 수)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)


def world_at(W: BlockWorld, b: int) -> BlockWorld:
    """묶음의 b 번째 블록."""
    return jax.tree_util.tree_map(lambda x: x[b], W)


def tree_to_numpy(tree: Any) -> Any:
    """세션 사이 저장용 — 잎 전부 numpy (np.savez 로 그대로 쓸 수 있게 잎 목록은 tree_leaves 순서)."""
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def tree_to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)
