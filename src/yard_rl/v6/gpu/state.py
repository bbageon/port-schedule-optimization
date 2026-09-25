"""세계 상태를 **고정 크기 배열로** ([[YR-327]]).

v5 는 `dict[job_id] → Job`, `list` 스택, 문자열 식별자를 쓴다. 배열 세계에서는
전부 번호와 배열이다 — 없는 것은 **-1**, 비어 있는 시각은 **+inf** 로 표시한다.

    문자열 "Y15:D-00042"  →  정수 번호 42 (블록 15)
    dict 조회             →  배열 색인
    리스트 길이 가변      →  길이 고정 + 개수 칸

■ ★왜 "전체 오더를 신경망으로" 가 이 구조에서 가능해지나
  오더가 배열이면 **전 오더의 특징이 하나의 행렬** (오더 수 × 특징 수)이 된다.
  망에 한 번 넣으면 **모든 오더의 점수가 한 번에** 나온다 — 지금처럼 후보마다
  따로 부르지 않는다. 그게 가속기가 빠른 이유이고, 반사실 기준선을
  **한 번의 순전파로** 만들 수 있는 근거이기도 하다([[YR-326]] ②).

■ 칸 수는 **미리 정한다**
  하루 최대 오더 15,000 · 블록 21 · 블록당 크레인 2 처럼 상한을 박아 둔다. 넘으면
  조용히 자르지 않고 표시한다 — 자르면 부하가 저절로 줄어 결과가 거짓이 된다
  ([[YR-316]] 에서 겪은 바로 그 사고).

■ 조각 1 (단일 블록 엔진) 이 더한 것 — `BlockWorld` 와 하위 묶음
  v5 `TerminalSimulator` 가 객체로 들고 있던 것을 전부 배열 열로 편다:

    v5 객체                           배열 묶음                 출처
    ─────────────────────────────────────────────────────────────────────────────
    Job (dict)                         OrderArrays 엔진 열      domain/models.py:32-75
    YcRuntime + CraneState + CraneSpec CraneArrays 런타임·스펙 열  integrated/cranes.py:14-33
    YardStacks (_stacks dict→list)     StackArrays (grid)        sim/stack.py:17-26
    Container                          ContArrays                domain/models.py:14-24
    ReservationTable                   ReservationArrays         integrated/reservation.py:25-54
    JobPlan + Move                     PlanArrays                integrated/jobplan.py:15-61
    KpiTracker                         KpiArrays                 sim/kpis.py:27-45
    TimeLedger                         LedgerArrays              integrated/time_contract.py:39-49
    CostAccumulator                    CostArrays                integrated/cost.py:43-47
    LaneNetwork                        LaneArrays                integrated/lane.py:14-56
    event_log                          LogArrays                 engine.py:157
    _pending/_assigned                 DecisionArrays            engine.py:146, 155
    _eta_wakes/_defer_wakes            WakeArrays (조각 3)       engine.py:166-179
    vessels (VesselProcess dict)       vessel.VesselArrays (조각 4)   integrated/vessel.py:20-54
    transfer (TransferFleet)           vessel.TransferArrays (조각 4) integrated/transfer.py
    injected PLAN_CHANGE               vessel.PlanChangeArrays (조각 4) scenario.py:14-22

  ★조각 3·4 통합 (2026-09-26): `BlockWorld` 에 `vessels`·`transfer`·`plan_change` 세 칸이 있다. 그 형은
    `gpu/vessel.py` 가 정의한다 — 이 파일은 순환 import 를 피해 **호출 시점에** 그 모듈을 불러 빈 배열을 만든다
    (`empty_block_world`). 배가 없는 세계도 V=1·U=1 의 **가짜 칸**(alive False · busy_until +inf) 을 갖는다 —
    처리기가 빈 배열을 색인하지 않게 하기 위해서다 (호스트 `IdTables.n_units`·`vessel_ids` 가 실제 수를 안다).
  ★사건 로그에 `aux` 열이 있다 — DISPATCH 줄의 **무엇을** 배정했나 (오더 번호 ≥ 0 · REPOSITION 은 −(2+bay)).
    v5 는 payload 문자열 "crane:job" 에 그 정보가 있고, PRE_REHANDLE(오더는 PLANNED 잔존)·REPOSITION(오더 없음)은
    오더 표에서 되찾을 수 없어 로그가 직접 든다 (host_convert.event_log_from_arrays).

  ★두 종류의 열을 **나눠 둔다** — `stage`(라이프사이클 6단, 정책 관측·시장용) 와
    `status`(v5 JobStatus 7단, 엔진 진실). 정책이 보는 열과 엔진이 굴리는 열이 섞이면
    실현 미래(actual_*)가 정책으로 새는 길이 생긴다.

■ ★시각·좌표·거리·비용은 전부 float64 (`TIME_DTYPE`, events.py 머리말)
  동등성은 x64 에서만 성립한다. `empty_block_world` 가 dtype 을 확인해 x64 가 꺼져
  있으면 **큰 소리로** 실패한다.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .events import TIME_DTYPE, EventArray, empty_queue
from .geom import Geom

EMPTY_ID = -1
EMPTY_TIME = jnp.inf

#: 오더 상태 — 라이프사이클 다섯 단계와 같은 순서 (schema/lifecycle.py) — 정책 관측용
ST_NOTICE, ST_GATE_IN, ST_BLOCK_IN, ST_SERVING, ST_DONE, ST_GATE_OUT = 0, 1, 2, 3, 4, 5

#: 크레인 상태 — v5 CraneStatus 선언 순 (domain/enums.py:58-63). 엔진은 IDLE·HANDLING 만 쓴다.
CR_IDLE, CR_MOVING, CR_HANDLING, CR_BLOCKED, CR_DOWN = 0, 1, 2, 3, 4
CR_WORKING = CR_HANDLING   # 기존 이름 유지 (= v5 HANDLING)

#: ★엔진 작업 상태 — v5 JobStatus 선언 순 (domain/enums.py:48-55)
JS_PLANNED, JS_RELEASED, JS_WAITING, JS_ASSIGNED, JS_RUNNING, JS_DONE, JS_CANCELLED = range(7)

#: 작업 흐름 — v5 JobFlow 선언 순 (domain/enums.py:10-16). 기존 `flow` 열의 0=반입 1=반출 과 같다.
FL_GATE_IN, FL_GATE_OUT, FL_VESSEL_LOAD, FL_VESSEL_DISCHARGE, FL_TRANSSHIPMENT, FL_REHANDLE = range(6)

#: 컨테이너 규격 — v5 ContainerSize 선언 순 (domain/enums.py:42-45)
SZ_FT20, SZ_FT40, SZ_FT45 = 0, 1, 2

#: 계획·행동 종류 — v5 CandidateKind 선언 순 (contract/schema.py:85-91)
PK_SERVE, PK_PRE_REHANDLE, PK_REPOSITION, PK_WAIT = 0, 1, 2, 3

#: 이동 종류 (jobplan.py:15-27 의 Move 를 세 갈래로) — 0 재배치(remove+place) 1 반출(remove) 2 반입(place)
MV_REHANDLE, MV_RETRIEVE, MV_STORE = 0, 1, 2

#: 사건 종류 — v5 EventKind (integrated/events.py:14-26) 그대로. 우선순위표는 events.PRIO
(EV_JOB_COMPLETED, EV_EQUIPMENT_DOWN, EV_EQUIPMENT_UP, EV_TRANSFER_ARRIVE, EV_STS_MOVE,
 EV_BLOCK_ARRIVAL, EV_JOB_RELEASED, EV_VESSEL_RELEASED, EV_VESSEL_START, EV_PLAN_CHANGE,
 EV_ETA_UPDATED, EV_HORIZON) = range(12)
#: 로그 전용 종류 (큐에 안 들어감) — engine.py:721, 370, 377, 408
LOG_DISPATCH, LOG_ETA_WAKE, LOG_DEFER_WAKE, LOG_DEADLOCK_ESCAPE = 12, 13, 14, 15

#: ★비용 13항 — v5 contract/schema.py:283-287 COST_TERMS 와 **같은 순서**여야 한다
#: (시험이 v5 를 읽어 대조한다). 호스트가 문자열로 찾지 않도록 색인을 굽는다.
COST_TERMS: tuple[str, ...] = (
    "truck_wait", "long_wait", "crane_travel", "empty_travel", "rehandle",
    "sts_wait", "transfer_wait", "vessel_delay", "depart_delay",
    "lane_cong", "interference", "resequence", "imbalance",
)
N_COST = len(COST_TERMS)
COST_IDX: dict[str, int] = {t: i for i, t in enumerate(COST_TERMS)}
#: rate(율×시간) 로 쌓는 5항 — v5 cost.py:16-21 RATE_TERMS_ORDERED (COST_TERMS 순서로 거른 것)
RATE_TERMS: tuple[str, ...] = ("sts_wait", "transfer_wait", "lane_cong", "interference", "imbalance")
N_RATE = len(RATE_TERMS)
#: rate 5칸 → 13칸 위치. `pending = pending.at[RATE_IDX].add(rate * dt)` 로 쓴다.
RATE_IDX = jnp.array([COST_IDX[t] for t in RATE_TERMS], jnp.int32)
#: 비용 색인 상수 — 엔진이 자주 쓰는 것
(C_TRUCK_WAIT, C_LONG_WAIT, C_CRANE_TRAVEL, C_EMPTY_TRAVEL, C_REHANDLE, C_STS_WAIT,
 C_TRANSFER_WAIT, C_VESSEL_DELAY, C_DEPART_DELAY, C_LANE_CONG, C_INTERFERENCE,
 C_RESEQUENCE, C_IMBALANCE) = range(N_COST)


def cost_index(term: str) -> int:
    """비용항 이름 → 13칸 색인. 모르는 이름은 조용히 넘기지 않는다."""
    try:
        return COST_IDX[term]
    except KeyError:
        raise KeyError(f"미지원 비용항 {term!r} — COST_TERMS={COST_TERMS}") from None


#: 위반 비트 — 0 이 아니면 그 세계는 **실격** (v5 는 예외를 던지는 자리)
V_NOT_TOP, V_TIER, V_SIZE, V_PLAN_POSTCOND, V_RESERVE_REJECT, V_NEG_COST = 1, 2, 4, 8, 16, 32
V_TIME_BACKWARD, V_COMPLETE_NO_PLAN, V_STEPS_EXHAUSTED, V_DECISION_COVERAGE = 64, 128, 256, 512
V_BUSY_NO_EVENT = 1024   # 큐는 비었는데 작업 중 크레인이 있다 (engine.py:324-326 RuntimeError)
V_UNSUPPORTED_EVENT = 2048   # 본선·이송·계획변경 사건이 실제 대상과 함께 왔다 (조각 4 전)
V_OVERFLOW = 4096            # 큐·로그 칸 부족 (world.overflow + queue.overflow > 0) — run 끝에서 켠다
V_LEDGER_UNREGISTERED = 8192 # 장부 모드인데 등록 안 된 트럭이 블록에 도착 (time_contract.py:61 KeyError)
#: 조각 2 불변식 (engine.py:1107-1137 check_invariants — v5 는 ConstraintViolation 을 던진다; engine_step.check_invariants)
V_CRANE_ORDER_SWAP = 16384   # 레일 순서(rail_order)가 뒤집혔다 = 크레인이 서로를 관통했다 (CRANE_ORDER_SWAP)
V_CRANE_MIN_GAP = 32768      # 레일 이웃 크레인 간격 < safety_gap (CRANE_MIN_GAP)
V_PAIRWISE_LOCK = 65536      # 활성 예약 쌍이 토큰·레인·통로·칸을 공유 (TOKEN_DOUBLE·LANE_DOUBLE·CORRIDOR_OVERLAP·SLOT_DOUBLE)
#: 조각 3·7 결정 계층 (dispatch.resolve_central) — 실린 후보 쌍이 고정 길이 scan 보다 많아 뒤가 잘렸다 (조용히 자르지 않는다)
V_RESOLVER_TRUNC = 131072
#: 비트 → 이름 (진단·보고용 역표). `violation_names(v)` 로 푼다.
VIOLATION_NAMES: dict[int, str] = {
    V_NOT_TOP: "NOT_TOP", V_TIER: "TIER", V_SIZE: "SIZE", V_PLAN_POSTCOND: "PLAN_POSTCOND",
    V_RESERVE_REJECT: "RESERVE_REJECT", V_NEG_COST: "NEG_COST", V_TIME_BACKWARD: "TIME_BACKWARD",
    V_COMPLETE_NO_PLAN: "COMPLETE_NO_PLAN", V_STEPS_EXHAUSTED: "STEPS_EXHAUSTED",
    V_DECISION_COVERAGE: "DECISION_COVERAGE", V_BUSY_NO_EVENT: "BUSY_NO_EVENT",
    V_UNSUPPORTED_EVENT: "UNSUPPORTED_EVENT", V_OVERFLOW: "OVERFLOW",
    V_LEDGER_UNREGISTERED: "LEDGER_UNREGISTERED",
    V_CRANE_ORDER_SWAP: "CRANE_ORDER_SWAP", V_CRANE_MIN_GAP: "CRANE_MIN_GAP",
    V_PAIRWISE_LOCK: "PAIRWISE_LOCK", V_RESOLVER_TRUNC: "RESOLVER_TRUNC",
}


def violation_names(v: int) -> tuple[str, ...]:
    """위반 비트합 → 켜진 비트 이름들 (모르는 비트는 'BIT_<n>')."""
    v = int(v)
    out = []
    bit = 1
    while v:
        if v & 1:
            out.append(VIOLATION_NAMES.get(bit, f"BIT_{bit}"))
        v >>= 1
        bit <<= 1
    return tuple(out)


# ─────────────────────────────────────────────────── 오더
class OrderArrays(NamedTuple):
    """오더 N 개. 모든 칸이 같은 길이라 **한 번에 망에 넣을 수 있다**.

    앞 열은 정책 관측(라이프사이클), `status` 부터가 엔진 진실 열이다.
    """

    block: jnp.ndarray        # (N,) int32  어느 블록으로 가나 (-1 = 없는 오더)
    flow: jnp.ndarray         # (N,) int32  FL_* (v5 JobFlow 순; -1 = 없음)
    stage: jnp.ndarray        # (N,) int32  위 ST_* — 지금 어디까지 왔나 (정책 관측)
    #: ★시각 다섯 — 사용자 스키마 그대로 (2026-09-22). 아직 안 온 단계는 +inf
    notice_s: jnp.ndarray     # (N,) f64  통지 (Truck ETA)
    gate_in_s: jnp.ndarray    # (N,) f64  게이트 진입 A — reset 에 시나리오 값 (engine.py:132)
    block_in_s: jnp.ndarray   # (N,) f64  블록 도착 B (engine.py:852)
    service_s: jnp.ndarray    # (N,) f64  작업 시작 S (내부 관측 — 정책 입력 금지, engine.py:713)
    done_s: jnp.ndarray       # (N,) f64  작업 완료 C (engine.py:920)
    gate_out_s: jnp.ndarray   # (N,) f64  게이트 아웃 O = C + exit_travel (engine.py:927-928)
    duration_s: jnp.ndarray   # (N,) f64  이 작업에 걸리는 시간
    travel_s: jnp.ndarray     # (N,) f64  게이트→블록 주행
    # ── 엔진 진실 열 (조각 1) — v5 Job 필드 (domain/models.py:32-75) ──
    status: jnp.ndarray          # (N,) int32  JS_* (v5 JobStatus 순)
    is_external: jnp.ndarray     # (N,) bool   외부트럭 (GATE_IN·GATE_OUT)
    is_vessel: jnp.ndarray       # (N,) bool   본선연계
    is_store: jnp.ndarray        # (N,) bool   STORE 모드 (= inbound_size 있음)
    target_cont: jnp.ndarray     # (N,) int32  반출 대상 컨테이너 번호 (-1 = 없음)
    inbound_cont: jnp.ndarray    # (N,) int32  반입 시 낳을 컨테이너 번호 = C0+n (engine.py:604-606 대체)
    inbound_size: jnp.ndarray    # (N,) int32  SZ_* (-1 = 없음)
    vessel: jnp.ndarray          # (N,) int32  소속 선박 (-1)
    assigned_crane: jnp.ndarray  # (N,) int32  맡은 크레인 (-1, engine.py:708)
    rehandles: jnp.ndarray       # (N,) int32  재조작 횟수 (engine.py:921)
    release_s: jnp.ndarray       # (N,) f64  해제 시각
    provided_eta_s: jnp.ndarray  # (N,) f64  제공 ETA (+inf = None)
    deadline_s: jnp.ndarray      # (N,) f64  마감 (+inf = None)
    exit_travel_s: jnp.ndarray   # (N,) f64  완료→출문 소요. ★-1 = None (장부 모드 판별, engine.py:127-133)
    actual_arrival_s: jnp.ndarray  # (N,) f64  실현 블록도착 — 시드용 진실값, **정책 입력 금지**
    waiting: jnp.ndarray         # (N,) bool  S−B 대기 중 (kpis._waiting)
    in_block: jnp.ndarray        # (N,) bool  B≤t<C 블록 점유 중 (ledger._in_block)
    wait_sample_s: jnp.ndarray   # (N,) f64  대기 표본 (NaN = 아직; 서비스 시작 또는 검열 시 확정)

    @property
    def n(self) -> int:
        return int(self.block.shape[0])


# ─────────────────────────────────────────────────── 크레인
class CraneArrays(NamedTuple):
    """크레인 K 개 — CraneState(models.py:85-96) + YcRuntime(cranes.py:14-33) + CraneSpec(models.py:121-132)."""

    block: jnp.ndarray         # (K,) int32   소속 블록
    status: jnp.ndarray        # (K,) int32   CR_* (엔진은 IDLE·HANDLING 만)
    available_at: jnp.ndarray  # (K,) f64     언제부터 다시 고를 수 있나
    assigned: jnp.ndarray      # (K,) int32   지금 맡은 오더 (-1 = 없음)
    bay: jnp.ndarray           # (K,) f64     지금 위치 (bay, 연속 좌표)
    row: jnp.ndarray           # (K,) f64     지금 위치 (row, 연속 좌표)
    bay_min: jnp.ndarray       # (K,) int32   담당 구간
    bay_max: jnp.ndarray       # (K,) int32
    # ── YcRuntime 확장 (cranes.py:14-33) ──
    down: jnp.ndarray          # (K,) bool    고장
    down_pending: jnp.ndarray  # (K,) bool    작업 중 고장 → 완료 후 DOWN (비선점)
    yielded: jnp.ndarray       # (K,) bool    WAIT 후 다음 상태변경까지 결정 제외
    yield_count: jnp.ndarray   # (K,) int32   경합 패배 양보 누적 (recent_yield_count, engine.py:687-688) — v5 는 resolver
                               #              (resolver.py:96) 가 yield_reason='LOST_CONTENTION' 을 세울 때만 올린다. 배열판은
                               #              `assign_scan(lost=…)` 이 같은 규칙으로 올리고, 정본 구동(ReferenceDispatcher)에서는 항상 0.
    completions: jnp.ndarray   # (K,) int32   recent_completions
    served: jnp.ndarray        # (K,) int32   served_count
    is_loaded: jnp.ndarray     # (K,) bool
    loaded_m: jnp.ndarray      # (K,) f64     누적 적재 주행 (CraneState.loaded_travel_m)
    empty_m: jnp.ndarray       # (K,) f64     누적 빈 주행 (CraneState.empty_travel_m)
    # ── 정적 스펙 (models.py:121-132) — 이동시간 10항의 분모·상수 ──
    spec_gantry: jnp.ndarray        # (K,) f64  gantry_speed_mps
    spec_trolley: jnp.ndarray       # (K,) f64  trolley_speed_mps
    spec_hoist_loaded: jnp.ndarray  # (K,) f64  hoist_speed_loaded_mps
    spec_hoist_empty: jnp.ndarray   # (K,) f64  hoist_speed_empty_mps
    spec_lock: jnp.ndarray          # (K,) f64  lock_time_s
    spec_unlock: jnp.ndarray        # (K,) f64  unlock_time_s
    spec_truck_pos: jnp.ndarray     # (K,) f64  truck_positioning_time_s
    rail_order: jnp.ndarray         # (K,) int32  레일 물리 순서 순열 (engine.py:105-118)

    @property
    def k(self) -> int:
        return int(self.block.shape[0])


# ─────────────────────────────────────────────────── 스택·컨테이너
class StackArrays(NamedTuple):
    """격자 (B,R,T) — v5 `_stacks: dict[(bay,row)] → list[container_id]` (stack.py:17-26)."""

    grid: jnp.ndarray      # (B,R,T) int32  0-based 칸·단 → 컨테이너 번호 (-1 = 빈 단)
    height: jnp.ndarray    # (B,R)   int32  쌓인 단 수 (= len(pile))
    top_size: jnp.ndarray  # (B,R)   int32  맨 위 규격 SZ_* (빈 칸 -1) — stack.stack_size_ok 캐시

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.grid.shape)


class ContArrays(NamedTuple):
    """컨테이너 C 개 — v5 Container (models.py:14-24). 좌표는 v5 와 같이 **1-based**."""

    c_bay: jnp.ndarray    # (C,) int32  (-1 = 야드 밖)
    c_row: jnp.ndarray    # (C,) int32
    c_tier: jnp.ndarray   # (C,) int32  바닥 = 1
    c_size: jnp.ndarray   # (C,) int32  SZ_*
    c_avail: jnp.ndarray  # (C,) bool   work_available (Hold·검사)
    c_alive: jnp.ndarray  # (C,) bool   야드에 있나 (반입 예비칸·반출된 것은 False)

    @property
    def c(self) -> int:
        return int(self.c_bay.shape[0])


# ─────────────────────────────────────────────────── 예약
class ReservationArrays(NamedTuple):
    """크레인별 예약 한 건 — v5 ReservationTable (reservation.py:25-54, 101-117).

    ★이 클래스가 **유일한** 정의다 (조각 2 · key=dispatch 에서 통일). `gpu/reserve.py` 는 여기서
    import 해 쓰고, 예약 판정·갱신 함수(`reject_code`·`reserve`·`release`…)는 그쪽에 있다.
    (조각 1 에는 reserve.py 에 같은 이름의 사본이 따로 있어 pytree 형이 달랐고, 엔진이 매 호출
    `_as_res` 로 바꿔 넣어야 했다.)

    빈 칸 규약: `active` 가 거짓인 행의 lo/hi/lane/token/release_at/slots 는 읽지 않는다.
    `idle_pos` 는 +inf 가 '미등록'(장벽 없음) — v5 `_idle_pos` 에 없는 크레인.
    """

    active: jnp.ndarray       # (K,) bool
    token: jnp.ndarray        # (K,) int32  작업 토큰 = 오더 번호 (-1)
    lo: jnp.ndarray           # (K,) f64    통로 [lo, hi] (Corridor)
    hi: jnp.ndarray           # (K,) f64
    lane: jnp.ndarray         # (K,) int32  레인 (-1 = 없음)
    release_at: jnp.ndarray   # (K,) f64    해제 예정 시각 (기록용 — 판정엔 안 쓴다; 빈 칸 +inf)
    slots: jnp.ndarray        # (K,B,R) bool  예약 칸 마스크 (frozenset slots)
    token_owner: jnp.ndarray  # (N,) int32   토큰 → 크레인 역표 (_tokens; -1 = 없음)
    idle_pos: jnp.ndarray     # (K,) f64     예약 없는 크레인의 현재 bay 장벽 (YR-091, reservation.py:40-42; +inf = 미등록)

    @property
    def k(self) -> int:
        return int(self.active.shape[0])


# ─────────────────────────────────────────────────── 계획
class PlanArrays(NamedTuple):
    """크레인이 지금 실행 중인 JobPlan (jobplan.py:45-61) + Move 목록 (15-27)."""

    kind: jnp.ndarray       # (K,) int32  PK_* (-1 = 계획 없음)
    job: jnp.ndarray        # (K,) int32  오더 번호 (-1)
    lo: jnp.ndarray         # (K,) f64    corridor
    hi: jnp.ndarray         # (K,) f64
    dur: jnp.ndarray        # (K,) f64    duration_s
    end_bay: jnp.ndarray    # (K,) f64
    end_row: jnp.ndarray    # (K,) f64
    rehandles: jnp.ndarray  # (K,) int32
    loaded_m: jnp.ndarray   # (K,) f64    loaded_gantry_m
    empty_m: jnp.ndarray    # (K,) f64    empty_gantry_m
    n_moves: jnp.ndarray    # (K,) int32  유효 이동 수 (i ≥ n_moves 는 무효)
    start_s: jnp.ndarray    # (K,) f64
    mv_cont: jnp.ndarray    # (K,M)   int32  이동할 컨테이너 (-1 = 빈 칸)
    mv_src: jnp.ndarray     # (K,M,3) int32  (bay,row,tier) 1-based
    mv_dst: jnp.ndarray     # (K,M,3) int32
    mv_kind: jnp.ndarray    # (K,M)   int32  MV_* (-1)

    @property
    def m(self) -> int:
        return int(self.mv_cont.shape[1])


# ─────────────────────────────────────────────────── 지표·장부·비용·레인
class KpiArrays(NamedTuple):
    """KpiTracker 누적 (sim/kpis.py:27-45). wait_samples 는 orders.wait_sample_s 가 대신한다."""

    queue_area: jnp.ndarray     # () f64   ∫ 대기 트럭 수 dt
    tail_area: jnp.ndarray      # () f64   그중 SLA 넘긴 구간
    loaded_m: jnp.ndarray       # () f64
    empty_m: jnp.ndarray        # () f64
    rehandles: jnp.ndarray      # () int32
    completed_ext: jnp.ndarray  # () int32
    completed_ves: jnp.ndarray  # () int32
    berth_overrun: jnp.ndarray  # () f64
    vessel_delay_s: jnp.ndarray  # () f64  본선 야드작업 마감 지각 합 (kpis.py:90-96; 반박 검증 누락분)
    pre_rehandle_count: jnp.ndarray  # () int32  선재조작 수 (kpis.py:39) — ★v5 는 어디서도 올리지 않는다 (항상 0;
                                     #           정답 궤적 Y01 도 REPO 3회 실행에 positioning_count 0). 배열판도 0 으로 둔다.
    positioning_count: jnp.ndarray   # () int32  REPOSITION 수 (kpis.py:40) — 위와 같이 v5 가 안 올리므로 0


class LedgerArrays(NamedTuple):
    """TimeLedger 적분 셋 (time_contract.py:39-49). 포인터·힙은 닫힌 식으로 대체."""

    block_area: jnp.ndarray     # () f64  ∫ N_block dt (B≤t<C)
    block_tail: jnp.ndarray     # () f64  그중 (t−B) > sla
    terminal_area: jnp.ndarray  # () f64  ∫ N_inside dt (A≤t<O)
    closed_end: jnp.ndarray     # () f64  +inf = None (아직 안 닫힘)


class CostArrays(NamedTuple):
    """CostAccumulator (cost.py:43-47) — 13항 raw. 항 순서 = COST_TERMS."""

    rate: jnp.ndarray     # (5,)  f64  RATE_TERMS 순 — advance 에서 rate×dt
    pending: jnp.ndarray  # (13,) f64  현재 결정구간 누적 (cut 으로 비움)
    episode: jnp.ndarray  # (13,) f64  에피소드 누적


class LaneArrays(NamedTuple):
    """LaneNetwork (lane.py:14-56) — 인접표와 혼잡 적분. 레인 수 L 은 Geom.n_lanes."""

    adj: jnp.ndarray          # (L,L) bool  무방향 인접 (대각 False)
    cong_area_s: jnp.ndarray  # ()    f64   ∫ 평균 혼잡률 dt (lane.py:52-56)


class LogArrays(NamedTuple):
    """사건 흐름 기록 (engine.py:157 event_log). 해시는 호스트가 만든다."""

    t: jnp.ndarray       # (E,) f64
    kind: jnp.ndarray    # (E,) int32  0..11 큐 종류 + 12..15 로그 전용 (-1 = 빈 칸)
    target: jnp.ndarray  # (E,) int32  대상 (오더·크레인 번호). ★DEADLOCK_ESCAPE(15) 는 유휴 크레인 **비트마스크**(bit k) —
                         #             int32 라 블록당 K ≤ 31 (escape.crane_bits; 다중블록은 블록별 K 라 실무상 무관)
    aux: jnp.ndarray     # (E,) int32  보조 — DISPATCH(12) 의 배정 내용: 오더 번호(SERVE·PRE_REHANDLE) · −(2+int(bay)) (REPOSITION) · 그 밖 -1
    n: jnp.ndarray       # ()   int32  적힌 수 (E 를 넘으면 world.overflow++)

    @property
    def capacity(self) -> int:
        return int(self.t.shape[0])


class DecisionArrays(NamedTuple):
    """열린 결정의 대상과 답 (engine.py:146, 155, 295-299, 679-736)."""

    pending: jnp.ndarray   # (K,) bool   이번 결정에서 물은 크레인
    answered: jnp.ndarray  # (K,) bool
    act_kind: jnp.ndarray  # (K,) int32  PK_* (-1)
    act_job: jnp.ndarray   # (K,) int32  (-1 = WAIT)
    act_bay: jnp.ndarray   # (K,) f64    REPOSITION 목표 bay (NaN)


class WakeArrays(NamedTuple):
    """PRE_ADVICE 깨우기·DEFER·검토 시각 (engine.py:166-179, 339-382, 314-322). 조각 1 은 0 칸."""

    eta_wake_s: jnp.ndarray    # (W,) f64   정렬된 wake 시각
    eta_wake_job: jnp.ndarray  # (W,) int32
    wake_idx: jnp.ndarray      # ()   int32
    eta_armed: jnp.ndarray     # (K,) bool
    defer_wake_s: jnp.ndarray  # (D,) f64
    defer_n: jnp.ndarray       # ()   int32
    review_s: jnp.ndarray      # (Rv,) f64
    review_idx: jnp.ndarray    # ()   int32


# ─────────────────────────────────────────────────── 세계 (조각 1)
class BlockWorld(NamedTuple):
    """단일 블록 세계 하나 — 명세 §2. `vmap` 으로 쌓으면 세계 여러 개."""

    clock: jnp.ndarray             # () f64   공용 시계
    end_s: jnp.ndarray             # () f64   평가창 끝 = horizon + drain (scenario.py:37-38)
    terminal: jnp.ndarray          # () bool
    last_decision_at: jnp.ndarray  # () f64   결정 시각 엄격증가 표식 (-inf = None, engine.py:184)
    escape_at: jnp.ndarray         # () f64   같은 시각 탈출 재발화 방지 (-inf = None, engine.py:181)
    escape_count: jnp.ndarray      # () int32
    steps: jnp.ndarray             # () int32 소비한 스텝 수
    queue: EventArray
    orders: OrderArrays
    cranes: CraneArrays
    stacks: StackArrays
    conts: ContArrays
    res: ReservationArrays
    plan: PlanArrays
    kpi: KpiArrays
    ledger: LedgerArrays
    cost: CostArrays
    lane: LaneArrays
    log: LogArrays
    decision: DecisionArrays
    wake: WakeArrays
    #: 조각 4 — 본선·이송·계획변경 (형은 gpu/vessel.py: VesselArrays · TransferArrays · PlanChangeArrays)
    vessels: "VesselArrays"
    transfer: "TransferArrays"
    plan_change: "PlanChangeArrays"
    violation: jnp.ndarray         # () int32  위반 비트합 V_* (0 이어야 정상)
    overflow: jnp.ndarray          # () int32  칸 부족 수 (0 이어야 정상)

    @property
    def n(self) -> int:
        return self.orders.n

    @property
    def k(self) -> int:
        return self.cranes.k


# ─────────────────────────────────────────────────── 생성기 (빈 배열)
def _i32(shape, v=EMPTY_ID):
    return jnp.full(shape, v, jnp.int32)


def _f64(shape, v=EMPTY_TIME):
    return jnp.full(shape, v, TIME_DTYPE)


def _bool(shape, v=False):
    return jnp.full(shape, v, jnp.bool_)


def check_x64() -> None:
    """★x64 가 꺼져 있으면 float64 요청이 조용히 float32 로 내려앉는다 — 큰 소리로 막는다."""
    t = jnp.zeros((), TIME_DTYPE)
    if t.dtype != jnp.dtype(TIME_DTYPE):
        raise RuntimeError(
            f"시각 dtype 이 {t.dtype} 다 — TIME_DTYPE={jnp.dtype(TIME_DTYPE).name} 를 쓰려면 "
            f"jax.config.update('jax_enable_x64', True) 를 배열 생성 전에 불러야 한다.")


def empty_orders(n: int) -> OrderArrays:
    """빈 오더 N 칸. 없는 오더는 block=-1 로 판별한다."""
    return OrderArrays(
        block=_i32((n,)), flow=_i32((n,)), stage=_i32((n,), ST_NOTICE),
        notice_s=_f64((n,)), gate_in_s=_f64((n,)), block_in_s=_f64((n,)),
        service_s=_f64((n,)), done_s=_f64((n,)), gate_out_s=_f64((n,)),
        duration_s=_f64((n,), 0.0), travel_s=_f64((n,), 0.0),
        status=_i32((n,), JS_PLANNED), is_external=_bool((n,)), is_vessel=_bool((n,)),
        is_store=_bool((n,)), target_cont=_i32((n,)), inbound_cont=_i32((n,)),
        inbound_size=_i32((n,)), vessel=_i32((n,)), assigned_crane=_i32((n,)),
        rehandles=_i32((n,), 0), release_s=_f64((n,)), provided_eta_s=_f64((n,)),
        deadline_s=_f64((n,)), exit_travel_s=_f64((n,), -1.0), actual_arrival_s=_f64((n,)),
        waiting=_bool((n,)), in_block=_bool((n,)), wait_sample_s=_f64((n,), jnp.nan))


def empty_cranes(k: int) -> CraneArrays:
    """빈 크레인 K 칸. 스펙은 호스트가 채운다 (0 으로 두면 이동시간이 0 나누기 → inf)."""
    return CraneArrays(
        block=_i32((k,)), status=_i32((k,), CR_IDLE), available_at=_f64((k,), 0.0),
        assigned=_i32((k,)), bay=_f64((k,), 1.0), row=_f64((k,), 1.0),
        bay_min=_i32((k,), 1), bay_max=_i32((k,), 1),
        down=_bool((k,)), down_pending=_bool((k,)), yielded=_bool((k,)),
        yield_count=_i32((k,), 0), completions=_i32((k,), 0), served=_i32((k,), 0),
        is_loaded=_bool((k,)), loaded_m=_f64((k,), 0.0), empty_m=_f64((k,), 0.0),
        spec_gantry=_f64((k,), 0.0), spec_trolley=_f64((k,), 0.0),
        spec_hoist_loaded=_f64((k,), 0.0), spec_hoist_empty=_f64((k,), 0.0),
        spec_lock=_f64((k,), 0.0), spec_unlock=_f64((k,), 0.0), spec_truck_pos=_f64((k,), 0.0),
        rail_order=jnp.arange(k, dtype=jnp.int32))


def empty_stacks(b: int, r: int, t: int) -> StackArrays:
    return StackArrays(grid=_i32((b, r, t)), height=_i32((b, r), 0), top_size=_i32((b, r)))


def empty_conts(c: int) -> ContArrays:
    return ContArrays(c_bay=_i32((c,)), c_row=_i32((c,)), c_tier=_i32((c,)), c_size=_i32((c,)),
                      c_avail=_bool((c,), True), c_alive=_bool((c,)))


def empty_reservations(k: int, n: int, b: int, r: int) -> ReservationArrays:
    """빈 예약표 — v5 `ReservationTable(gap)` 직후와 같다 (idle_pos 도 비어 있음 = +inf).

    (조각 1 의 state 판은 release_at·idle_pos 를 0.0 으로 뒀다 — idle_pos 0.0 은 'bay 0 에 장벽'
    으로 읽히므로 reserve.py 판의 +inf 로 통일했다. 호스트 `to_block_world` 는 어차피 전 크레인의
    idle_pos 를 덮어쓴다(engine.py:114-115).)"""
    return ReservationArrays(
        active=_bool((k,)), token=_i32((k,)), lo=_f64((k,), 0.0), hi=_f64((k,), 0.0),
        lane=_i32((k,)), release_at=_f64((k,)), slots=_bool((k, b, r)),
        token_owner=_i32((n,)), idle_pos=_f64((k,)))


def empty_plans(k: int, m: int) -> PlanArrays:
    return PlanArrays(
        kind=_i32((k,)), job=_i32((k,)), lo=_f64((k,), 0.0), hi=_f64((k,), 0.0),
        dur=_f64((k,), 0.0), end_bay=_f64((k,), 0.0), end_row=_f64((k,), 0.0),
        rehandles=_i32((k,), 0), loaded_m=_f64((k,), 0.0), empty_m=_f64((k,), 0.0),
        n_moves=_i32((k,), 0), start_s=_f64((k,), 0.0),
        mv_cont=_i32((k, m)), mv_src=_i32((k, m, 3)), mv_dst=_i32((k, m, 3)), mv_kind=_i32((k, m)))


def empty_kpi() -> KpiArrays:
    z = _f64((), 0.0)
    return KpiArrays(queue_area=z, tail_area=z, loaded_m=z, empty_m=z,
                     rehandles=_i32((), 0), completed_ext=_i32((), 0), completed_ves=_i32((), 0),
                     berth_overrun=z, vessel_delay_s=z,
                     pre_rehandle_count=_i32((), 0), positioning_count=_i32((), 0))


def empty_ledger() -> LedgerArrays:
    z = _f64((), 0.0)
    return LedgerArrays(block_area=z, block_tail=z, terminal_area=z, closed_end=_f64(()))


def empty_cost() -> CostArrays:
    return CostArrays(rate=_f64((N_RATE,), 0.0), pending=_f64((N_COST,), 0.0),
                      episode=_f64((N_COST,), 0.0))


def empty_lanes(n_lanes: int) -> LaneArrays:
    return LaneArrays(adj=_bool((n_lanes, n_lanes)), cong_area_s=_f64((), 0.0))


def empty_log(capacity: int) -> LogArrays:
    return LogArrays(t=_f64((capacity,)), kind=_i32((capacity,)), target=_i32((capacity,)),
                     aux=_i32((capacity,)), n=_i32((), 0))


def empty_decision(k: int) -> DecisionArrays:
    return DecisionArrays(pending=_bool((k,)), answered=_bool((k,)), act_kind=_i32((k,)),
                          act_job=_i32((k,)), act_bay=_f64((k,), jnp.nan))


def empty_wake(k: int, *, n_wake: int = 0, n_defer: int = 0, n_review: int = 0) -> WakeArrays:
    return WakeArrays(eta_wake_s=_f64((n_wake,)), eta_wake_job=_i32((n_wake,)),
                      wake_idx=_i32((), 0), eta_armed=_bool((k,)),
                      defer_wake_s=_f64((n_defer,)), defer_n=_i32((), 0),
                      review_s=_f64((n_review,)), review_idx=_i32((), 0))


def empty_block_world(g: Geom, *, n_orders: int, n_cranes: int, n_conts: int,
                      q_cap: int, log_cap: int, end_s: float,
                      n_wake: int = 0, n_defer: int = 0, n_review: int = 0,
                      v_max: int = 0, n_units: int = 0, p_cap: int = 0, move_time_s: float = 0.0,
                      i_max: int = 0, j_max: int = 0) -> BlockWorld:
    """빈 단일 블록 세계. 격자 모양은 `g` 에서, 나머지 칸 수는 인자로 **미리 정한다**.

    조각 4 칸 수: v_max 배 · n_units 이송차 · p_cap 대기 링버퍼 · i_max PLAN_CHANGE 행 · j_max 행당 마감 쌍.
    v_max·n_units 가 0 이면 **가짜 칸 하나**(alive False · busy_until +inf)를 둔다 (머리말 ★) — 배가 없는 세계에서도
    처리기가 빈 배열을 색인하지 않고, 이송 요청은 언제나 대기 링버퍼로 간다(v5 n_units=0 의 `_free_index` None).
    ★x64 가 꺼져 있으면 여기서 실패한다 (events.empty_queue 와 같은 규약).
    """
    from .vessel import empty_plan_change, empty_transfer, empty_vessels   # 호출 시점 import (순환 방지, 머리말 ★)
    check_x64()
    b, r, t = int(g.bay_count), int(g.row_count), int(g.tier_max)
    ves = empty_vessels(max(1, int(v_max)), n_orders)
    tr = empty_transfer(max(1, int(n_units)), max(1, int(p_cap)), float(move_time_s))
    if int(n_units) <= 0:
        tr = tr._replace(busy_until=jnp.full((1,), EMPTY_TIME, TIME_DTYPE))   # 가짜 유닛 — 영원히 바쁨
    return BlockWorld(
        clock=_f64((), 0.0), end_s=_f64((), float(end_s)), terminal=_bool(()),
        last_decision_at=_f64((), -jnp.inf), escape_at=_f64((), -jnp.inf),
        escape_count=_i32((), 0), steps=_i32((), 0),
        queue=empty_queue(q_cap),
        orders=empty_orders(n_orders), cranes=empty_cranes(n_cranes),
        stacks=empty_stacks(b, r, t), conts=empty_conts(n_conts),
        res=empty_reservations(n_cranes, n_orders, b, r),
        plan=empty_plans(n_cranes, g.n_moves),
        kpi=empty_kpi(), ledger=empty_ledger(), cost=empty_cost(),
        lane=empty_lanes(int(g.n_lanes)), log=empty_log(log_cap),
        decision=empty_decision(n_cranes),
        wake=empty_wake(n_cranes, n_wake=n_wake, n_defer=n_defer, n_review=n_review),
        vessels=ves, transfer=tr, plan_change=empty_plan_change(int(i_max), int(j_max)),
        violation=_i32((), 0), overflow=_i32((), 0))


# ─────────────────────────────────────────────────── 기존 (정책·시장용) 세계
class WorldArrays(NamedTuple):
    """세계 하나 전체 (조각 1 이전의 얇은 판). `vmap` 으로 B 개 쌓으면 세계 B 개."""

    clock: jnp.ndarray         # () f64  공용 시계
    end_s: jnp.ndarray         # () f64  이 세계를 언제까지 굴리나
    orders: OrderArrays
    cranes: CraneArrays
    #: 누적 비용 네 항 (원) — 항목을 갈라 둬야 어디서 났는지 보인다
    cost_wait: jnp.ndarray     # () f64
    cost_move: jnp.ndarray     # () f64
    cost_rehandle: jnp.ndarray # () f64
    cost_vessel: jnp.ndarray   # () f64
    #: ★넘침 표시 — 칸이 모자라 못 담은 수. 0 이 아니면 그 세계는 **실격**이다
    overflow: jnp.ndarray      # () int32

    @property
    def cost_total(self) -> jnp.ndarray:
        return self.cost_wait + self.cost_move + self.cost_rehandle + self.cost_vessel


def empty_world(n_orders: int, n_cranes: int, *, end_s: float = 86_400.0) -> WorldArrays:
    """빈 세계 하나. 칸 수는 **미리 정해** 넘어가면 표시한다(머리말 참조)."""
    z = _f64((), 0.0)
    return WorldArrays(
        clock=_f64((), 0.0), end_s=_f64((), float(end_s)),
        orders=empty_orders(n_orders), cranes=empty_cranes(n_cranes),
        cost_wait=z, cost_move=z, cost_rehandle=z, cost_vessel=z,
        overflow=_i32((), 0))


# ─────────────────────────────────────────────────── 성과지표
def turn_time_s(o: OrderArrays) -> jnp.ndarray:
    """턴타임 = 게이트 아웃 − 게이트 인. **나간 오더만** 유한한 값을 갖는다."""
    return o.gate_out_s - o.gate_in_s


def censored_turn_time_s(o: OrderArrays, end_s) -> jnp.ndarray:
    """터미널 턴타임 표본 — v5 `TimeLedger.terminal_turntime_samples_s` (time_contract.py:125-134) 그대로.

        O 가 확정됐고 O ≤ end     →  O − A          (나간 트럭)
        그 밖에 A < end           →  end − A        (검열 — 못 나갔거나 O > end)
        나머지 (A ≥ end, 미등록)  →  NaN (표본 아님)

    ★안 그러면 **못 나간 트럭이 표본에서 빠져** 정책이 이득을 본다. `gate_in_s` 는 reset 에
    시나리오 값으로 차므로 "등록됐다(A<inf)" 가 "평가창 안에 들어왔다" 는 뜻이 아니다 —
    A ≥ end 인 트럭은 표본에서 **빠져야** 하고(반박 검증 finding), O > end 는 end−A 로 검열한다.
    """
    end_s = jnp.asarray(end_s, TIME_DTYPE)
    left = (o.gate_out_s < EMPTY_TIME) & (o.gate_out_s <= end_s)     # 129행 (O 확정 · O ≤ end)
    part = o.gate_in_s < end_s                                        # 131행 (A < end)
    return jnp.where(left, o.gate_out_s - o.gate_in_s,
                     jnp.where(part, end_s - o.gate_in_s, jnp.nan))


def censored_exposure_s(o: OrderArrays, end_s) -> jnp.ndarray:
    """종료시점 터미널 안(O 미확정 또는 O > end) 트럭의 `end − A` 합 — v5 `censored_exposure_s`
    (time_contract.py:136-142) `sum(end − r.gate_in for …)`.

    ★파이썬 3.12 `sum()` 은 **Neumaier 보정합**이다 (exact.py 머리말 실측 2) — 항이 파이썬 float(end = 시나리오
    `horizon_s + drain_window_s`, gate_in = engine.py:132 `actual_gate_in or 0.0`, 둘 다 random.Random 산 float) 이므로
    `exact.sum_python` 으로 더한다 (순차 `+=` 는 3항부터 마지막 비트가 갈릴 수 있다 — 탐침 40항 20회 중 18회).
    순서 = records 사전 삽입 순서 = engine.py:131 `sorted(_v2, key=job_id)` = 오더 번호 순 (host_convert 번호 규칙)."""
    from .exact import sum_python
    end_s = jnp.asarray(end_s, TIME_DTYPE)
    inside = (o.gate_in_s < end_s) & ((o.gate_out_s >= EMPTY_TIME) | (o.gate_out_s > end_s))
    terms = jnp.where(inside, end_s - o.gate_in_s, 0.0)
    return sum_python(terms, inside)


def block_turn_time_s(o: OrderArrays, end_s) -> jnp.ndarray:
    """블록 턴타임 표본 — v5 `block_turntime_samples_s` (time_contract.py:112-123):
    도착한(B 있음) 등록 트럭만, 완료면 C − B, 아니면 max(0, end − B). 나머지 NaN."""
    end_s = jnp.asarray(end_s, TIME_DTYPE)
    rec = (o.gate_in_s < EMPTY_TIME) & (o.block_in_s < EMPTY_TIME)
    done = o.done_s < EMPTY_TIME
    return jnp.where(rec, jnp.where(done, o.done_s - o.block_in_s,
                                    jnp.maximum(0.0, end_s - o.block_in_s)), jnp.nan)
