"""본선·이송을 **배열로** ([[YR-327]] 조각 4 · key=vessel — 명세 outputs/v6/piece4_spec.md).

v5 `world/integrated/engine.py` 의 본선 6 처리기(948-1065행)·완료의 본선 훅(929-942행)·STS 대기
적분·요율(806-811, 816-817행)·clearout(1072-1080행)과 `integrated/transfer.py`(이송차 큐)·
`integrated/vessel.py:20-54`(VesselPlan·VesselProcess)·`scenario.py:14-22`(InjectedEvent) 를
고정 크기 배열 위의 **순수 함수**로 옮긴다. 처리기는 전부 `world → world` 이고, 통합자가
`engine_step._handlers` 의 3/4/7/8/9 칸에 그대로 끼운다.

■ 배열 세 묶음
    VesselArrays  (V,) 배 V 척 — 계획 8열(VesselPlan) + 진행 7열(VesselProcess) + 오더 측 `release_rank` (N,)
    TransferArrays 이송차 U 대 busy_until + 대기 요청 **링버퍼** (P,) head/tail + 대기 적분
    PlanChangeArrays 주입 PLAN_CHANGE 표 (I,) — completion/basis/etd 덮어쓰기 + 작업 마감 쌍 (I,J)

  세계는 이 세 묶음을 `vessels`·`transfer`·`plan_change` 라는 **이름의 칸**으로 갖고 있어야 한다
  (`BlockWorld` 에 세 칸을 더하는 것은 통합자 몫). 통합 전 시험은 아래 `VesselWorld` 로 돈다.

■ ★v5 와 같은 답 — 규약 (README ★)
  · 시각·적분은 float64. `clock + cadence` 는 파이썬 float 덧셈과 같은 f64 덧셈(누적 가산, 988행) —
    scenario_gen 의 `start + k·cadence` 곱셈과 마지막 비트가 다를 수 있는 것이 **v5 의 동작**이고 그대로 둔다.
  · 곱은 `exact.mul_exact` — `len(pending) * dt` (transfer.py:59), `rate * dt` 는 통합자의 advance 가 한다.
  · v5 가 `+=` 로 하나씩 더하는 clearout(1076-1079행)은 배 번호 순 scan — `jnp.sum` 을 쓰지 않는다.
  · **막힌 STS 는 자기 STS_MOVE 를 재예약하지 않는다** (971-974행) — 다른 사건(TRANSFER_ARRIVE 1019-1021행,
    양하 적재 완료 939-942행 handover 모드)이 clock 에 push 한다. '넣지 않음' 은 `push_if(False, …)` 로,
    큐 칸을 먹지 않는다 (events.push_event 를 고치지 않고 tree_where 로 감쌌다).
  · 양하 해제 순서 = `sorted(self.jobs)` 첫 PLANNED·STORE (1001-1006행) = **job_id 문자열 사전식**.
    호스트가 오더마다 `release_rank`(사전식 순위)를 굽고 처리기는 `argmin(where(mask, rank, BIG))`.
    host_convert 번호(n = sorted(job_id) 순위)에서는 rank == n 이지만, 스트림 물량 ≥100 ('-100' < '-11')
    처럼 번호가 사전식이 아닌 세계를 위해 열을 따로 둔다.
  · `yard_handover_cap` (YR-088 opt-in, 기본 None) 은 옮기지 않았다 — Y01 정답·fixtures 전부 None.
    handover 모드가 필요해지면 `_discharge_pipeline` (V,) 열 하나와 can 판정 한 항이 더 든다.

■ 처리기 (v5 줄)
    h_vessel_start   948-957   do=~started; remaining=total; STS_MOVE 를 clock+cadence 에
    h_sts_move       966-989   act=started&~done; ok=(양하: buffer<cap · 적하: buffer>0); blocked_since 갱신;
                               양하 buffer+1 → transfer_request(TRANSFER_ARRIVE 먼저 push) · 적하 buffer−1;
                               remaining−1; ≤0 → _vessel_finish (vessel_delay·berth_overrun·depart_delay)
                               아니면 STS_MOVE 를 clock+cadence 에 (막혔으면 아무것도 push 안 함)
    h_transfer_arrive 1008-1022 양하 buffer=max(0,b−1)·VESSEL_RELEASED(clock, rank 최소 PLANNED·STORE) ·
                               적하 buffer+1; pending 재배차(TRANSFER_ARRIVE); 막힌 STS 재개(STS_MOVE clock);
                               yielded 전부 해제
    h_vessel_released 877-881  status[n]=RELEASED; yielded 해제
    h_plan_change    1048-1065 표에서 그 배의 첫 미소비 행: completion/basis/etd 를 '키 있음' 이면 덮어쓰기
                               (+inf/-1 = None 으로 세팅도 키 있음), 작업 마감 쌍 산포; 모르는 배(-1) 는 무동작
    on_load_completed 929-935  SERVE 완료 job 이 VESSEL_LOAD & 배 있음 → transfer_request
    integrate_vessel_wait 806-811 · transfer.py:57-60   blocked&~done 배마다 wait_accum += dt · 이송 대기 += n·dt
    vessel_rates     816-817   (Σ blocked&~done, len(pending)) → cost.rate[0:2]
    clearout_vessels 1072-1080 미완·계획완료 있음·end>pc → vessel_delay·berth_overrun (배 번호 순 `+=`)

■ 통합자가 부르는 자리 (engine_step.py)
    _handlers: 3→h_transfer_arrive · 4→h_sts_move · 7→h_vessel_released(또는 기존 h_released) · 8→h_vessel_start · 9→h_plan_change
    h_completed: 오더 DONE 처리 뒤 `on_load_completed(world, n, nvalid)` (935행 자리, down_pending 처리 앞)
    advance: `go` 안에서 `vessels, transfer = integrate_vessel_wait(vessels, transfer, dt)` (811행 자리, cost.advance 앞)
    refresh_rates: `sts, tw = vessel_rates(vessels, transfer)` 를 rate[0], rate[1] 에 (816-817행)
    _fin: advance 뒤 `clearout_vessels(world)` (1072-1080행)
    host_convert: `vessel_arrays_from_scenario`·`transfer_arrays_from_profile`·`plan_change_from_scenario` (아래 호스트 절)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE, EventArray, push_event
from .exact import mul_exact
from .state import (C_DEPART_DELAY, C_VESSEL_DELAY, EV_STS_MOVE, EV_TRANSFER_ARRIVE, EV_VESSEL_RELEASED,
                    FL_VESSEL_LOAD, JS_PLANNED, JS_RELEASED, CostArrays, CraneArrays, KpiArrays, OrderArrays)

__all__ = ["EPS", "BASIS_NAMES", "BASIS_NONE", "BASIS_ABSENT", "VesselArrays", "TransferArrays",
           "PlanChangeArrays", "VesselWorld", "empty_vessels", "empty_transfer", "empty_plan_change",
           "attach", "push_if", "can_sts_process", "transfer_request", "dispatch_pending",
           "h_vessel_start", "h_sts_move", "h_transfer_arrive", "h_vessel_released", "h_plan_change",
           "on_load_completed", "integrate_vessel_wait", "vessel_rates", "clearout_vessels",
           "pending_count", "pending_entries",
           "release_rank_from_ids", "vessel_arrays_from_scenario", "vessel_arrays_from_v5",
           "transfer_arrays_from_profile", "transfer_arrays_from_v5", "plan_change_from_scenario",
           "vessels_to_v5", "transfer_to_v5"]

F = TIME_DTYPE
#: v5 `_EPS` (engine.py:35 · transfer.py:11) — 이송차 free 판정 `busy_until ≤ now + EPS`
EPS = 1e-9
_BIG = 1 << 30

#: CompletionBasis 선언 순 (contract/vessel.py:20-26). -1 = None, -2 = (PLAN_CHANGE 표에서) 키 없음
BASIS_NAMES: tuple[str, ...] = ("TOS_TARGET", "PLAN_COMPUTED", "ATD_MINUS_BUFFER", "OPERATOR_TEMP")
BASIS_NONE, BASIS_ABSENT = -1, -2


# ───────────────────────────────────────────────── 배열
class VesselArrays(NamedTuple):
    """배 V 척 — VesselPlan (vessel.py:20-33) + VesselProcess (44-54) + 오더 측 release_rank."""

    # ── 계획 (VesselPlan) — PLAN_CHANGE 가 completion/basis/etd 를 바꾼다 (1052-1060행) ──
    is_discharge: jnp.ndarray          # (V,) bool  DISCHARGE(양하) / LOAD(적하)
    total_moves: jnp.ndarray           # (V,) int32
    cadence_s: jnp.ndarray             # (V,) f64   sts_move_interval_s
    buffer_cap: jnp.ndarray            # (V,) int32 quay_buffer_cap
    planned_start_s: jnp.ndarray       # (V,) f64
    planned_completion_s: jnp.ndarray  # (V,) f64   +inf = None
    completion_basis: jnp.ndarray      # (V,) int32 BASIS_NAMES 순, -1 = None (is_symptom 용 — 엔진은 안 읽음)
    etd_s: jnp.ndarray                 # (V,) f64   +inf = None
    # ── 진행 (VesselProcess) ──
    started: jnp.ndarray               # (V,) bool
    remaining: jnp.ndarray             # (V,) int32  -1 = 미개시
    buffer: jnp.ndarray                # (V,) int32  안벽 버퍼 점유
    blocked_since_s: jnp.ndarray       # (V,) f64    +inf = None (sts_blocked ⟺ < inf)
    wait_accum_s: jnp.ndarray          # (V,) f64    sts_wait_accum_s
    done: jnp.ndarray                  # (V,) bool
    actual_completion_s: jnp.ndarray   # (V,) f64    truth.actual_completion_s (+inf = None) — 비용정산 전용
    alive: jnp.ndarray                 # (V,) bool   실제 배인 칸 (V 가 실제 수보다 크면 뒤는 False)
    # ── 오더 측 ──
    release_rank: jnp.ndarray          # (N,) int32  job_id 사전식 순위 (양하 해제 순서, 1001행 sorted(self.jobs))

    @property
    def v(self) -> int:
        return int(self.is_discharge.shape[0])


class TransferArrays(NamedTuple):
    """TransferFleet (transfer.py:16-60) — 유닛 U 대 + 대기 요청 링버퍼 P 칸."""

    busy_until: jnp.ndarray    # (U,) f64  유닛별 바쁜 끝 시각 (0.0 시작)
    move_time_s: jnp.ndarray   # ()   f64  한 트립
    pend_time: jnp.ndarray     # (P,) f64  대기 요청 시각 (링버퍼)
    pend_vessel: jnp.ndarray   # (P,) int32 대기 요청 배
    head: jnp.ndarray          # ()   int32 다음에 꺼낼 칸 (mod P)
    tail: jnp.ndarray          # ()   int32 다음에 넣을 칸 (mod P); 대기 수 = tail − head
    wait_accum_s: jnp.ndarray  # ()   f64  transfer_wait_accum_s (진단용 적분)
    overflow: jnp.ndarray      # ()   int32 링버퍼가 차서 못 넣은 수 (0 이어야 정상)

    @property
    def u(self) -> int:
        return int(self.busy_until.shape[0])

    @property
    def p(self) -> int:
        return int(self.pend_time.shape[0])


class PlanChangeArrays(NamedTuple):
    """주입 PLAN_CHANGE 표 (scenario.py:14-22 InjectedEvent, kind=PLAN_CHANGE) — 시드 순서(시각·종류·대상 정렬) 그대로.

    같은 배의 행은 시각 순이고 사건도 그 순서로 오므로, 처리기는 **그 배의 첫 미소비 행**을 쓴다.
    """

    vessel: jnp.ndarray      # (I,) int32  대상 배 (-1 = 모르는 배 → v5 도 무동작)
    time_s: jnp.ndarray      # (I,) f64    주입 시각 (기록용)
    completion: jnp.ndarray  # (I,) f64    NaN = 키 없음(유지) · +inf = None 으로 · 값
    basis: jnp.ndarray       # (I,) int32  BASIS_ABSENT(-2) = 키 없음 · BASIS_NONE(-1) · 0..3
    etd: jnp.ndarray         # (I,) f64    NaN = 키 없음 · +inf = None · 값
    dl_job: jnp.ndarray      # (I,J) int32 작업 마감 쌍의 오더 번호 (-1 = 빈칸 · 모르는 job → v5 도 건너뜀)
    dl_s: jnp.ndarray        # (I,J) f64   마감 (+inf = None)
    consumed: jnp.ndarray    # (I,) bool   처리된 행

    @property
    def i(self) -> int:
        return int(self.vessel.shape[0])


class VesselWorld(NamedTuple):
    """통합 전 시험용 최소 세계 — 처리기가 읽고 쓰는 칸만. 통합자의 `BlockWorld` 도 같은 이름의 칸을 가진다."""

    clock: jnp.ndarray
    end_s: jnp.ndarray
    queue: EventArray
    orders: OrderArrays
    cranes: CraneArrays
    cost: CostArrays
    kpi: KpiArrays
    vessels: VesselArrays
    transfer: TransferArrays
    plan_change: PlanChangeArrays
    violation: jnp.ndarray


def _i32(x):
    return jnp.asarray(x, jnp.int32)


def _f64(shape, v):
    return jnp.full(shape, v, F)


def empty_vessels(v: int, n_orders: int) -> VesselArrays:
    """빈 배 V 칸 (alive 전부 False) + 오더 N 칸의 release_rank (전부 BIG)."""
    return VesselArrays(
        is_discharge=jnp.zeros((v,), bool), total_moves=_i32(jnp.zeros((v,))),
        cadence_s=_f64((v,), 0.0), buffer_cap=_i32(jnp.zeros((v,))),
        planned_start_s=_f64((v,), EMPTY_TIME), planned_completion_s=_f64((v,), EMPTY_TIME),
        completion_basis=jnp.full((v,), BASIS_NONE, jnp.int32), etd_s=_f64((v,), EMPTY_TIME),
        started=jnp.zeros((v,), bool), remaining=jnp.full((v,), -1, jnp.int32),
        buffer=_i32(jnp.zeros((v,))), blocked_since_s=_f64((v,), EMPTY_TIME),
        wait_accum_s=_f64((v,), 0.0), done=jnp.zeros((v,), bool),
        actual_completion_s=_f64((v,), EMPTY_TIME), alive=jnp.zeros((v,), bool),
        release_rank=jnp.full((n_orders,), _BIG, jnp.int32))


def empty_transfer(u: int, p: int, move_time_s: float) -> TransferArrays:
    """`TransferFleet(...)` 직후 — busy_until 전부 0.0 (transfer.py:26-28), 대기 없음."""
    return TransferArrays(
        busy_until=_f64((u,), 0.0), move_time_s=jnp.asarray(float(move_time_s), F),
        pend_time=_f64((p,), EMPTY_TIME), pend_vessel=jnp.full((p,), EMPTY_ID, jnp.int32),
        head=_i32(0), tail=_i32(0), wait_accum_s=jnp.asarray(0.0, F), overflow=_i32(0))


def empty_plan_change(i: int, j: int) -> PlanChangeArrays:
    return PlanChangeArrays(
        vessel=jnp.full((i,), EMPTY_ID, jnp.int32), time_s=_f64((i,), EMPTY_TIME),
        completion=_f64((i,), jnp.nan), basis=jnp.full((i,), BASIS_ABSENT, jnp.int32),
        etd=_f64((i,), jnp.nan), dl_job=jnp.full((i, j), EMPTY_ID, jnp.int32),
        dl_s=_f64((i, j), EMPTY_TIME), consumed=jnp.zeros((i,), bool))


def attach(world, vessels: VesselArrays, transfer: TransferArrays, plan_change: PlanChangeArrays) -> VesselWorld:
    """`BlockWorld`(또는 같은 칸을 가진 것) 에서 처리기가 쓰는 칸만 떼어 `VesselWorld` 를 만든다 (시험용)."""
    return VesselWorld(clock=world.clock, end_s=world.end_s, queue=world.queue, orders=world.orders,
                       cranes=world.cranes, cost=world.cost, kpi=world.kpi, vessels=vessels,
                       transfer=transfer, plan_change=plan_change, violation=world.violation)


# ───────────────────────────────────────────────── 작은 도구
def _where_tree(c, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(c, x, y), a, b)


def push_if(q: EventArray, cond, time, kind, target) -> EventArray:
    """`cond` 일 때만 push — 아니면 큐를 **그대로** (칸·순번을 먹지 않는다). '넣지 않음' 의 배열 표현."""
    return _where_tree(cond, push_event(q, time, kind, target), q)


def _set1(arr, i, v, valid):
    """arr[i] = v (valid 일 때만). -1 은 clip + where 로 막는다 (engine_step._set1 과 같은 규약)."""
    ic = jnp.clip(_i32(i), 0, arr.shape[0] - 1)
    return arr.at[ic].set(jnp.where(valid, jnp.asarray(v, arr.dtype), arr[ic]))


def _vessel_index(ves: VesselArrays, v):
    """(vc, valid) — 범위 안·실제 배."""
    v = _i32(v)
    V = ves.v
    vc = jnp.clip(v, 0, V - 1)
    valid = (v >= 0) & (v < V) & ves.alive[vc]
    return vc, valid


def _accrue(cost: CostArrays, idx: int, amount, cond) -> CostArrays:
    """`cost.accrue(term, amount)` (cost.py:63-76) — pending·episode 에 같은 값을 더한다. cond 거짓이면 손대지 않는다."""
    pend = jnp.where(cond, cost.pending.at[idx].add(amount), cost.pending)
    epi = jnp.where(cond, cost.episode.at[idx].add(amount), cost.episode)
    return cost._replace(pending=pend, episode=epi)


def _clear_yields(cr: CraneArrays, cond) -> CraneArrays:
    """`_clear_yields` (840-842행) — cond 일 때만."""
    return cr._replace(yielded=jnp.where(cond, jnp.zeros_like(cr.yielded), cr.yielded))


def pending_count(tr: TransferArrays) -> jnp.ndarray:
    """`waiting_count()` = len(pending) (transfer.py:53-54). () int32."""
    return tr.tail - tr.head


def can_sts_process(ves: VesselArrays) -> jnp.ndarray:
    """`_can_sts_process` (959-964행, handover None) 를 (V,) 로 — 양하: buffer < cap · 적하: buffer > 0."""
    return jnp.where(ves.is_discharge, ves.buffer < ves.buffer_cap, ves.buffer > 0)


# ───────────────────────────────────────────────── 이송차 (transfer.py)
def _free_unit(tr: TransferArrays, now):
    """`_free_index(now)` (30-34행): busy_until ≤ now+EPS 인 **첫** 유닛. (i, 있나)."""
    free = tr.busy_until <= now + EPS
    return jnp.argmax(free), jnp.any(free)


def transfer_request(world, v, cond):
    """`_transfer_request(vid)` (991-994행) + `TransferFleet.request` (36-43행) — cond 일 때만.

    유닛이 비면 busy_until[i] = now + move_time 하고 TRANSFER_ARRIVE 를 그 시각에 push, 아니면 pending 링버퍼
    꼬리에 (now, v) 를 넣는다 (칸이 없으면 overflow++ · 넣지 않음).
    """
    tr = world.transfer
    now = world.clock
    v = _i32(v)
    i, has_free = _free_unit(tr, now)
    arrive = now + tr.move_time_s                                          # 42행 now + move_time_s
    dispatch = cond & has_free
    busy = jnp.where(dispatch, tr.busy_until.at[i].set(arrive), tr.busy_until)
    enqueue = cond & ~has_free                                             # 39-40행 pending.append
    P = tr.p
    room = (tr.tail - tr.head) < P
    idx = jnp.mod(tr.tail, P)
    put = enqueue & room
    tr2 = tr._replace(
        busy_until=busy,
        pend_time=jnp.where(put, tr.pend_time.at[idx].set(now), tr.pend_time),
        pend_vessel=jnp.where(put, tr.pend_vessel.at[idx].set(v), tr.pend_vessel),
        tail=tr.tail + jnp.where(put, 1, 0).astype(jnp.int32),
        overflow=tr.overflow + jnp.where(enqueue & ~room, 1, 0).astype(jnp.int32))
    q2 = push_if(world.queue, dispatch, arrive, EV_TRANSFER_ARRIVE, v)     # 993-994행
    return world._replace(transfer=tr2, queue=q2)


def dispatch_pending(world, cond):
    """`dispatch_pending(now)` (45-51행) + 1016-1018행 push — 대기 요청이 있고 유닛이 비면 머리를 꺼내 배차."""
    tr = world.transfer
    now = world.clock
    has_p = (tr.tail - tr.head) > 0
    i, has_free = _free_unit(tr, now)
    do = cond & has_p & has_free
    P = tr.p
    idx = jnp.mod(tr.head, P)
    nv = tr.pend_vessel[idx]
    arrive = now + tr.move_time_s                                          # 50-51행
    tr2 = tr._replace(busy_until=jnp.where(do, tr.busy_until.at[i].set(arrive), tr.busy_until),
                      head=tr.head + jnp.where(do, 1, 0).astype(jnp.int32))
    q2 = push_if(world.queue, do, arrive, EV_TRANSFER_ARRIVE, nv)          # 1018행
    return world._replace(transfer=tr2, queue=q2)


# ───────────────────────────────────────────────── 본선 처리기
def h_vessel_start(world, v):
    """VESSEL_START (948-957행): 미개시면 started·remaining=total, STS_MOVE 를 clock+cadence 에 push."""
    ves = world.vessels
    vc, valid = _vessel_index(ves, v)
    do = valid & ~ves.started[vc]                                          # 950-951행
    ves2 = ves._replace(started=ves.started.at[vc].set(ves.started[vc] | do),
                        remaining=ves.remaining.at[vc].set(jnp.where(do, ves.total_moves[vc], ves.remaining[vc])))
    q2 = push_if(world.queue, do, world.clock + ves.cadence_s[vc], EV_STS_MOVE, vc)   # 957행
    return world._replace(vessels=ves2, queue=q2)


def _vessel_finish(world, vc, cond):
    """`_vessel_finish` (1024-1039행) — cond 일 때만: done·remaining 0·blocked 해제·actual_completion=clock·
    vessel_delay(+berth_overrun 같은 값)·depart_delay."""
    ves = world.vessels
    clock = world.clock
    ves2 = ves._replace(
        done=ves.done.at[vc].set(ves.done[vc] | cond),
        remaining=ves.remaining.at[vc].set(jnp.where(cond, 0, ves.remaining[vc])),
        blocked_since_s=ves.blocked_since_s.at[vc].set(jnp.where(cond, EMPTY_TIME, ves.blocked_since_s[vc])),
        actual_completion_s=ves.actual_completion_s.at[vc].set(jnp.where(cond, clock, ves.actual_completion_s[vc])))
    pc = ves.planned_completion_s[vc]
    late = cond & (pc < EMPTY_TIME) & (clock > pc)                         # 1031행
    over = clock - pc                                                      # 1034행 clock − pc
    cost = _accrue(world.cost, C_VESSEL_DELAY, over, late)
    kpi = world.kpi._replace(berth_overrun=jnp.where(late, world.kpi.berth_overrun + jnp.maximum(0.0, over),
                                                     world.kpi.berth_overrun))   # kpis.py:101 max(0, s)
    etd = ves.etd_s[vc]
    late2 = cond & (etd < EMPTY_TIME) & (clock > etd)                      # 1037행
    cost = _accrue(cost, C_DEPART_DELAY, clock - etd, late2)               # 1038행
    return world._replace(vessels=ves2, cost=cost, kpi=kpi)


def h_sts_move(world, v):
    """STS_MOVE (966-989행) — 머리말 표. 막히면 blocked_since 만 찍고 **아무것도 push 하지 않는다**."""
    ves = world.vessels
    clock = world.clock
    vc, valid = _vessel_index(ves, v)
    act = valid & ves.started[vc] & ~ves.done[vc]                          # 968-969행
    disch = ves.is_discharge[vc]
    can = can_sts_process(ves)[vc]                                         # 970행
    ok = act & can
    was_blocked = ves.blocked_since_s[vc] < EMPTY_TIME
    blocked = jnp.where(act & ~can & ~was_blocked, clock,                  # 971-973행
                        jnp.where(ok, EMPTY_TIME, ves.blocked_since_s[vc]))   # 975-976행
    buffer = ves.buffer[vc] + jnp.where(ok, jnp.where(disch, 1, -1), 0).astype(jnp.int32)   # 978·983행
    remaining = ves.remaining[vc] - jnp.where(ok, 1, 0).astype(jnp.int32)  # 984행
    finish = ok & (remaining <= 0)                                         # 985행
    ves2 = ves._replace(blocked_since_s=ves.blocked_since_s.at[vc].set(blocked),
                        buffer=ves.buffer.at[vc].set(buffer),
                        remaining=ves.remaining.at[vc].set(remaining))
    w1 = world._replace(vessels=ves2)
    w1 = transfer_request(w1, vc, ok & disch)                              # 981행 (STS_MOVE push 보다 먼저 — seq 순서)
    w1 = _vessel_finish(w1, vc, finish)                                    # 986행
    q2 = push_if(w1.queue, ok & ~finish, clock + ves.cadence_s[vc], EV_STS_MOVE, vc)   # 988행
    return w1._replace(queue=q2)


def _release_next_discharge(world, vc, cond):
    """`_release_next_discharge` (995-1006행): 그 배의 PLANNED·STORE 오더 중 release_rank 최소를 VESSEL_RELEASED(clock) 로."""
    o = world.orders
    mask = (o.vessel == vc) & (o.status == JS_PLANNED) & o.is_store        # 1003-1004행 (STORE ⟺ inbound_size 있음)
    has = jnp.any(mask)
    n = jnp.argmin(jnp.where(mask, world.vessels.release_rank, _BIG)).astype(jnp.int32)
    return world._replace(queue=push_if(world.queue, cond & has, world.clock, EV_VESSEL_RELEASED, n))   # 1005행


def h_transfer_arrive(world, v):
    """TRANSFER_ARRIVE (1008-1022행) — 머리말 표. push 순서: VESSEL_RELEASED → 대기 재배차 TRANSFER_ARRIVE → STS_MOVE."""
    ves = world.vessels
    clock = world.clock
    vc, valid = _vessel_index(ves, v)
    disch = ves.is_discharge[vc]
    buffer = jnp.where(disch, jnp.maximum(0, ves.buffer[vc] - 1), ves.buffer[vc] + 1)   # 1011행 · 1014행
    buffer = jnp.where(valid, buffer, ves.buffer[vc]).astype(jnp.int32)
    ves2 = ves._replace(buffer=ves.buffer.at[vc].set(buffer))
    w1 = world._replace(vessels=ves2)
    w1 = _release_next_discharge(w1, vc, valid & disch)                    # 1012행
    w1 = dispatch_pending(w1, valid)                                       # 1015-1018행
    can = can_sts_process(w1.vessels)[vc]                                  # 1019행 (버퍼 갱신 뒤)
    resume = valid & ~ves.done[vc] & (ves.blocked_since_s[vc] < EMPTY_TIME) & can
    ves3 = w1.vessels._replace(blocked_since_s=w1.vessels.blocked_since_s.at[vc].set(
        jnp.where(resume, EMPTY_TIME, w1.vessels.blocked_since_s[vc])))    # 1020행
    q2 = push_if(w1.queue, resume, clock, EV_STS_MOVE, vc)                 # 1021행
    return w1._replace(vessels=ves3, queue=q2, cranes=_clear_yields(world.cranes, valid))   # 1022행


def h_vessel_released(world, n):
    """VESSEL_RELEASED (877-881행): status[n]=RELEASED + yielded 해제 (engine_step.h_released 와 같은 일)."""
    o = world.orders
    n = _i32(n)
    valid = (n >= 0) & (n < o.n)
    return world._replace(orders=o._replace(status=_set1(o.status, n, JS_RELEASED, valid)),
                          cranes=_clear_yields(world.cranes, True))


def h_plan_change(world, v):
    """PLAN_CHANGE (1048-1065행): 그 배의 첫 미소비 표 행을 적용. 모르는 배(-1)·행 없음 → 무동작 (1049-1051행)."""
    ves, pc = world.vessels, world.plan_change
    vc, valid = _vessel_index(ves, v)
    I = pc.i
    if I == 0:
        return world
    rows = jnp.arange(I, dtype=jnp.int32)
    cand = (pc.vessel == vc) & ~pc.consumed & valid
    has = jnp.any(cand)
    i = jnp.argmin(jnp.where(cand, rows, _BIG)).astype(jnp.int32)
    apply = valid & has
    comp, basis, etd = pc.completion[i], pc.basis[i], pc.etd[i]
    ves2 = ves._replace(                                                   # 1053-1060행 upd.get(키, 기존)
        planned_completion_s=ves.planned_completion_s.at[vc].set(
            jnp.where(apply & ~jnp.isnan(comp), comp, ves.planned_completion_s[vc])),
        completion_basis=ves.completion_basis.at[vc].set(
            jnp.where(apply & (basis != BASIS_ABSENT), basis, ves.completion_basis[vc])),
        etd_s=ves.etd_s.at[vc].set(jnp.where(apply & ~jnp.isnan(etd), etd, ves.etd_s[vc])))
    o = world.orders
    dl = o.deadline_s
    for j in range(int(pc.dl_job.shape[1])):                               # 1062-1064행 순서대로 (뒤 쌍이 이긴다)
        job = pc.dl_job[i, j]
        dl = _set1(dl, job, pc.dl_s[i, j], apply & (job >= 0) & (job < o.n))
    pc2 = pc._replace(consumed=pc.consumed.at[i].set(pc.consumed[i] | apply))
    return world._replace(vessels=ves2, orders=o._replace(deadline_s=dl), plan_change=pc2)


def on_load_completed(world, n, cond):
    """`_complete` 의 본선 훅 (929-935행): SERVE 완료 오더가 VESSEL_LOAD 이고 배가 있으면 transfer_request.
    (양하 handover 분기 936-942행은 yard_handover_cap=None 이라 없음 — 머리말.)"""
    o = world.orders
    n = _i32(n)
    nc = jnp.clip(n, 0, o.n - 1)
    v = o.vessel[nc]
    c = cond & (n >= 0) & (n < o.n) & (o.flow[nc] == FL_VESSEL_LOAD) & (v >= 0)   # 934행
    return transfer_request(world, jnp.clip(v, 0, world.vessels.v - 1), c)


# ───────────────────────────────────────────────── 적분·요율·clearout
def integrate_vessel_wait(ves: VesselArrays, tr: TransferArrays, dt):
    """`_advance` 의 본선·이송 몫 (806-811행 · transfer.py:57-60) — dt = hi − lo > 0 일 때 통합자가 부른다.

    blocked & ~done 인 배마다 `sts_wait_accum_s += (hi − lo)`; 이송 `transfer_wait_accum_s += len(pending) * dt`.
    """
    dt = jnp.asarray(dt, F)
    blocked = (ves.blocked_since_s < EMPTY_TIME) & ~ves.done & ves.alive    # 809행
    ves2 = ves._replace(wait_accum_s=jnp.where(blocked, ves.wait_accum_s + dt, ves.wait_accum_s))   # 810행
    n_pend = (tr.tail - tr.head).astype(F)
    tr2 = tr._replace(wait_accum_s=tr.wait_accum_s + mul_exact(n_pend, dt))   # 60행 len(pending) * dt
    return ves2, tr2


def vessel_rates(ves: VesselArrays, tr: TransferArrays):
    """`_refresh_rates` 의 앞 두 항 (816-817행): (Σ blocked&~done, len(pending)) — set_rate 의 max(0,·) 포함. () f64 둘."""
    sts = jnp.sum((ves.blocked_since_s < EMPTY_TIME) & ~ves.done & ves.alive).astype(F)
    tw = (tr.tail - tr.head).astype(F)
    return jnp.maximum(0.0, sts), jnp.maximum(0.0, tw)


def clearout_vessels(world):
    """`_finalize` 의 본선 clearout (1072-1080행): 미완·계획완료 있음·end > pc 인 배마다 배 번호 순으로
    vessel_delay += end − pc · berth_overrun += max(0, end − pc). v5 `+=` 순서 그대로 scan."""
    ves = world.vessels
    end = world.end_s
    pc = ves.planned_completion_s
    cond = ves.alive & ~ves.done & (pc < EMPTY_TIME) & (end > pc)          # 1077행
    amt = end - pc                                                         # 1078행 self.end − pc

    def body(carry, x):
        pend, epi, berth = carry
        a, c = x
        pend = jnp.where(c, pend + a, pend)
        epi = jnp.where(c, epi + a, epi)
        berth = jnp.where(c, berth + jnp.maximum(0.0, a), berth)
        return (pend, epi, berth), None

    cost, kpi = world.cost, world.kpi
    (pend, epi, berth), _ = lax.scan(body, (cost.pending[C_VESSEL_DELAY], cost.episode[C_VESSEL_DELAY],
                                            kpi.berth_overrun), (amt, cond))
    cost2 = cost._replace(pending=cost.pending.at[C_VESSEL_DELAY].set(pend),
                          episode=cost.episode.at[C_VESSEL_DELAY].set(epi))
    return world._replace(cost=cost2, kpi=kpi._replace(berth_overrun=berth))


# ───────────────────────────────────────────────── 호스트 (v5 ↔ 배열) — gpu 순수 경로에는 안 들어온다
def release_rank_from_ids(job_ids, n_orders: int):
    """오더 n 의 job_id 사전식 순위 (1001행 `sorted(self.jobs)`). 빈 칸(n ≥ len(job_ids)) 은 BIG."""
    import numpy as np
    order = sorted(range(len(job_ids)), key=lambda n: job_ids[n])
    rank = np.full((n_orders,), _BIG, np.int32)
    for r, n in enumerate(order):
        rank[n] = r
    return jnp.asarray(rank)


def _basis_code(b) -> int:
    if b is None:
        return BASIS_NONE
    name = b.value if hasattr(b, "value") else str(b)
    return BASIS_NAMES.index(name)


def _or_inf(x) -> float:
    return EMPTY_TIME if x is None else float(x)


def vessel_arrays_from_v5(vessels: dict, job_ids, vessel_ids, n_orders: int, *, v_max: int | None = None) -> VesselArrays:
    """살아 있는 v5 `sim.vessels` (vessel_id → VesselProcess) → 배열. 배 번호 = `vessel_ids` (sorted) 의 위치."""
    import numpy as np
    V = len(vessel_ids) if v_max is None else int(v_max)
    if len(vessel_ids) > V:
        raise ValueError(f"배 {len(vessel_ids)} 척 > v_max {V}")
    e = empty_vessels(V, n_orders)
    cols = {f: np.asarray(getattr(e, f)).copy() for f in e._fields if f != "release_rank"}
    for i, vid in enumerate(vessel_ids):
        vp = vessels[vid]
        p = vp.plan
        cols["is_discharge"][i] = vp.work_type.value == "DISCHARGE"
        cols["total_moves"][i] = int(p.total_moves)
        cols["cadence_s"][i] = float(p.sts_move_interval_s)
        cols["buffer_cap"][i] = int(p.quay_buffer_cap)
        cols["planned_start_s"][i] = float(p.planned_start_s)
        cols["planned_completion_s"][i] = _or_inf(p.planned_completion_s)
        cols["completion_basis"][i] = _basis_code(p.completion_basis)
        cols["etd_s"][i] = _or_inf(p.etd_s)
        cols["started"][i] = bool(vp.started)
        cols["remaining"][i] = int(vp.remaining_moves)
        cols["buffer"][i] = int(vp.buffer_level)
        cols["blocked_since_s"][i] = _or_inf(vp.sts_blocked_since_s)
        cols["wait_accum_s"][i] = float(vp.sts_wait_accum_s)
        cols["done"][i] = bool(vp.done)
        cols["actual_completion_s"][i] = _or_inf(vp.truth.actual_completion_s)
        cols["alive"][i] = True
    return e._replace(**{f: jnp.asarray(a) for f, a in cols.items()},
                      release_rank=release_rank_from_ids(list(job_ids), n_orders))


def vessel_arrays_from_scenario(scenario, job_ids, vessel_ids, n_orders: int, *, v_max: int | None = None) -> VesselArrays:
    """시나리오의 `vessels` (reset 직후 = 미개시) → 배열. `job_ids`·`vessel_ids` 는 host_convert.IdTables 의 것."""
    return vessel_arrays_from_v5({v.vessel_id: v for v in scenario.vessels}, job_ids, vessel_ids, n_orders, v_max=v_max)


def transfer_arrays_from_v5(tr, vessel_ids, *, p_cap: int) -> TransferArrays:
    """살아 있는 v5 `sim.transfer` (TransferFleet) → 배열 (pending 은 링버퍼 앞부터)."""
    import numpy as np
    if len(tr.pending) > p_cap:
        raise ValueError(f"대기 요청 {len(tr.pending)} 건 > p_cap {p_cap}")
    e = empty_transfer(int(tr.n_units), int(p_cap), float(tr.move_time_s))
    vidx = {v: i for i, v in enumerate(vessel_ids)}
    pt = np.full((p_cap,), EMPTY_TIME, np.float64)
    pv = np.full((p_cap,), EMPTY_ID, np.int32)
    for i, (t, vid) in enumerate(tr.pending):
        pt[i], pv[i] = float(t), vidx[vid]
    return e._replace(busy_until=jnp.asarray(np.asarray(tr.busy_until, np.float64)),
                      pend_time=jnp.asarray(pt), pend_vessel=jnp.asarray(pv),
                      tail=_i32(len(tr.pending)), wait_accum_s=jnp.asarray(float(tr.transfer_wait_accum_s), F))


def transfer_arrays_from_profile(profile, *, p_cap: int) -> TransferArrays:
    """`IntegratedProfile.transfer` (TransferFleetSpec) → reset 직후 배열 (engine.py:135-137)."""
    sp = profile.transfer
    return empty_transfer(int(sp.n_units), int(p_cap), float(sp.move_time_s))


def plan_change_from_scenario(scenario, job_ids, vessel_ids, *, i_max: int | None = None,
                              j_max: int | None = None) -> PlanChangeArrays:
    """`injected_events` 의 PLAN_CHANGE 행 — 시드 정렬 (time, kind, target) 순서 (engine.py:238) 그대로."""
    import numpy as np
    rows = [ie for ie in sorted(scenario.injected_events, key=lambda x: (x.time, x.kind, x.target))
            if ie.kind == "PLAN_CHANGE"]
    I = len(rows) if i_max is None else int(i_max)
    if len(rows) > I:
        raise ValueError(f"PLAN_CHANGE {len(rows)} 건 > i_max {I}")
    dls = [list(dict(ie.data or ()).get("job_deadlines", ())) for ie in rows]
    J = max([len(d) for d in dls] + [0]) if j_max is None else int(j_max)
    if any(len(d) > J for d in dls):
        raise ValueError(f"작업 마감 쌍이 j_max {J} 를 넘는다")
    e = empty_plan_change(I, J)
    vidx = {v: i for i, v in enumerate(vessel_ids)}
    jidx = {j: i for i, j in enumerate(job_ids)}
    vessel = np.full((I,), EMPTY_ID, np.int32); time_s = np.full((I,), EMPTY_TIME, np.float64)
    comp = np.full((I,), np.nan, np.float64); basis = np.full((I,), BASIS_ABSENT, np.int32)
    etd = np.full((I,), np.nan, np.float64)
    dl_job = np.full((I, J), EMPTY_ID, np.int32); dl_s = np.full((I, J), EMPTY_TIME, np.float64)
    for i, ie in enumerate(rows):
        upd = dict(ie.data or ())
        vessel[i] = vidx.get(ie.target, EMPTY_ID)
        time_s[i] = float(ie.time)
        if "planned_completion_s" in upd:
            comp[i] = _or_inf(upd["planned_completion_s"])
        if "completion_basis" in upd:
            basis[i] = _basis_code(upd["completion_basis"])
        if "etd_s" in upd:
            etd[i] = _or_inf(upd["etd_s"])
        for j, (jid, dl) in enumerate(dls[i]):
            dl_job[i, j] = jidx.get(jid, EMPTY_ID)
            dl_s[i, j] = _or_inf(dl)
    return e._replace(vessel=jnp.asarray(vessel), time_s=jnp.asarray(time_s), completion=jnp.asarray(comp),
                      basis=jnp.asarray(basis), etd=jnp.asarray(etd), dl_job=jnp.asarray(dl_job),
                      dl_s=jnp.asarray(dl_s))


def _opt(x) -> float | None:
    import numpy as np
    v = float(np.asarray(x))
    return None if not np.isfinite(v) else v


def vessels_to_v5(ves: VesselArrays, vessel_ids) -> dict[str, dict]:
    """배열 → v5 VesselProcess 모양의 값 (시험이 항목별로 대조)."""
    import numpy as np
    out = {}
    for i, vid in enumerate(vessel_ids):
        b = int(ves.completion_basis[i])
        out[vid] = {
            "work_type": "DISCHARGE" if bool(ves.is_discharge[i]) else "LOAD",
            "total_moves": int(ves.total_moves[i]), "sts_move_interval_s": float(ves.cadence_s[i]),
            "quay_buffer_cap": int(ves.buffer_cap[i]), "planned_start_s": float(ves.planned_start_s[i]),
            "planned_completion_s": _opt(ves.planned_completion_s[i]),
            "completion_basis": (None if b < 0 else BASIS_NAMES[b]), "etd_s": _opt(ves.etd_s[i]),
            "started": bool(ves.started[i]), "remaining_moves": int(ves.remaining[i]),
            "buffer_level": int(ves.buffer[i]), "sts_blocked_since_s": _opt(ves.blocked_since_s[i]),
            "sts_wait_accum_s": float(np.asarray(ves.wait_accum_s[i])), "done": bool(ves.done[i]),
            "actual_completion_s": _opt(ves.actual_completion_s[i]),
        }
    return out


def pending_entries(tr: TransferArrays, vessel_ids) -> list[tuple[float, str]]:
    """링버퍼 → v5 `pending` 리스트 (요청시각, vessel_id) 순서대로."""
    import numpy as np
    P = tr.p
    h, t = int(tr.head), int(tr.tail)
    pt, pv = np.asarray(tr.pend_time), np.asarray(tr.pend_vessel)
    return [(float(pt[k % P]), vessel_ids[int(pv[k % P])]) for k in range(h, t)]


def transfer_to_v5(tr: TransferArrays, vessel_ids) -> dict:
    """배열 → v5 TransferFleet 모양 (busy_until 리스트 · pending 리스트 · 대기 적분 · overflow)."""
    import numpy as np
    return {"busy_until": [float(x) for x in np.asarray(tr.busy_until)],
            "pending": pending_entries(tr, vessel_ids),
            "transfer_wait_accum_s": float(np.asarray(tr.wait_accum_s)),
            "overflow": int(tr.overflow)}
