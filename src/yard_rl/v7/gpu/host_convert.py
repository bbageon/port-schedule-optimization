"""v5 입력(프로파일·시나리오) ↔ 배열 세계 — **호스트에서 한 번** ([[YR-327]] 조각 1 §1).

v5 `TerminalSimulator.reset()` (integrated/engine.py:93-243) 이 객체로 만드는 초기 상태를
`BlockWorld` 배열로 **똑같이** 만든다. 반대 방향(`from_block_world`)은 배열을 v5 모양의
파이썬 값으로 되돌려 시험이 v5 와 나란히 놓고 비교하게 한다.

■ ★번호 규칙 — 전부 파이썬 문자열 정렬 (= v5 `sorted`) 이라 v5 순회 순서와 같다
    오더 n       = sorted(job_id) 순위            engine.py:227 (`_seed_events`), 477, 1001
    컨테이너 c   = sorted(container_id) 순위      sim/stack.py:18 (`YardStacks.__init__`)
                   반입 예비칸 = C0 + n  (C0 = 초기 컨테이너 수, engine.py:604 `IN_{job_id}` 를 미리 잡음)
    크레인 k     = sorted(crane_id) 순위          integrated/cranes.py:53-54 (`CraneFleet.ids`)
    선박 v       = sorted(vessel_id) 순위         engine.py:236
    레인 l       = lane_graph.lane_ids 의 위치    engine.py:271-273 (`_lane_for`: ids[(bay-1) % L])

■ reset 재현 (v5 줄 ↔ 여기)
    격자·컨테이너            96행 YardStacks             → stack_ops.from_v5_stacks
    크레인 초기 위치         98-113행 (bay = service_bay_min; 같은 구간 n>1 이면 lo+(k+0.5)(hi−lo)/n)
    idle 장벽·레일 순서      114-118행                   → res.idle_pos · cranes.rail_order
    시간계약 v2 장부 등록    127-133행 (외부트럭·exit_travel 있음 → gate_in = actual_gate_in or 0.0)
    레인 인접표              137행 LaneNetwork           → lane.adj
    평가창 끝                144행 end = horizon + drain → end_s
    사건 큐 시드             226-243행 `_seed_events`     → queue (같은 순서·같은 seq)
    입력 검증                198-221행 `_validate`        → **v5 함수를 그대로 부른다**
    rate 초기화              196행 `_refresh_rates`       → 전부 0 (예약·양보·본선 없음)

■ ★위치 산식은 파이썬 float 로 계산한다 — `lo + (k + 0.5) * (hi - lo) / n` (113행) 을 numpy 로
  옮기면 결합 순서는 같아도 dtype 승격이 끼어들 수 있다. v5 가 파이썬 float 로 낸 값을 그대로 담는다.

■ 사건 로그 역변환 (`event_log_from_arrays`) — 배열 (t, kind, target, aux) → v5 (t, kind_name, payload)
    큐 사건 0..11   payload = 대상 문자열 (크레인·오더·선박, HORIZON 은 "HORIZON")   engine.py:845
    12 DISPATCH     "crane:job" — target 은 크레인, **aux 가 무엇을 배정했나** (state.LogArrays.aux):
                    aux ≥ 0 → 오더 id (SERVE·PRE_REHANDLE — PRE 는 오더가 PLANNED 잔존이라 오더 표로는 못 찾는다),
                    aux ≤ −2 → "REPO:<crane>:<bay>" (bay = −aux−2; v5 는 job_id=f"REPO:{cid}:{int(tb)}" 413행)
    13 ETA_WAKE     payload = 오더 id (engine.py:377)     14 DEFER_WAKE  payload = "" (370행)
    15 DEADLOCK_ESCAPE  target 은 유휴 크레인 **비트마스크** (bit k = 크레인 k) → ",".join(정렬 id) (408행)

■ 조각 3·4 통합 (2026-09-26) — 여기서 같이 만든다
    ETA wake     `n_wake=None`(기본) 이면 시드될 wake 수만큼 칸을 잡는다 (engine.py:166-171 조건 그대로; 0 이면 W=0).
    본선·이송    `vessel.vessel_arrays_from_scenario`·`transfer_arrays_from_profile`·`plan_change_from_scenario`
                 (engine.py:134-138 reset). 배가 없으면 V=1 가짜 칸 (state.py 머리말), 이송차 0 대면 U=1 가짜 유닛.
                 p_cap(대기 링버퍼) 기본 = max(8, Σ total_moves) — 한 배의 요청이 전부 밀려도 넘치지 않는 폭.
    REPOSITION 중 크레인  `cranes.assigned = cands3.BUSY_NO_ORDER` (오더 번호 아님) → `from_block_world` 가 활성 계획의
                 end_bay 로 v5 문자열 "REPO:<cid>:<int(bay)>" 를 되돌린다 (engine.py:699 assigned_job = plan.job_id).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .events import EMPTY_ID, EMPTY_TIME, PRIO, TIME_DTYPE, EventArray, empty_queue
from .geom import Geom
from .stack_ops import SIZE_INDEX, from_v5_stacks, to_v5_piles
from .state import (CR_IDLE, EV_BLOCK_ARRIVAL, EV_EQUIPMENT_DOWN, EV_EQUIPMENT_UP,
                    EV_HORIZON, EV_JOB_RELEASED, EV_PLAN_CHANGE, EV_VESSEL_START,
                    JS_PLANNED, LOG_DEADLOCK_ESCAPE, LOG_DEFER_WAKE, LOG_DISPATCH,
                    LOG_ETA_WAKE, COST_TERMS, PK_REPOSITION, RATE_TERMS, BlockWorld, empty_block_world,
                    violation_names)
from .vessel import (plan_change_from_scenario, transfer_arrays_from_profile, transfer_to_v5,
                     vessel_arrays_from_scenario, vessels_to_v5)

__all__ = ["IdTables", "to_block_world", "from_block_world", "event_log_from_arrays",
           "queue_entries", "event_stream_hash", "repo_job_id", "EV_NAMES", "FLOW_NAMES", "STATUS_NAMES"]

#: v5 enum 이름 — 선언 순 (domain/enums.py). 배열 정수 ↔ 문자열 복원용.
EV_NAMES: tuple[str, ...] = (
    "JOB_COMPLETED", "EQUIPMENT_DOWN", "EQUIPMENT_UP", "TRANSFER_ARRIVE", "STS_MOVE",
    "BLOCK_ARRIVAL", "JOB_RELEASED", "VESSEL_RELEASED", "VESSEL_START", "PLAN_CHANGE",
    "ETA_UPDATED", "HORIZON", "DISPATCH", "ETA_WAKE", "DEFER_WAKE", "DEADLOCK_ESCAPE")
FLOW_NAMES: tuple[str, ...] = ("GATE_IN", "GATE_OUT", "VESSEL_LOAD", "VESSEL_DISCHARGE",
                               "TRANSSHIPMENT", "REHANDLE")
STATUS_NAMES: tuple[str, ...] = ("PLANNED", "RELEASED", "WAITING", "ASSIGNED", "RUNNING",
                                 "DONE", "CANCELLED")
CRANE_STATUS_NAMES: tuple[str, ...] = ("IDLE", "MOVING", "HANDLING", "BLOCKED", "DOWN")

#: 사건 종류 → payload 가 가리키는 표 (engine.py:229-243, 720, 957-1021)
_TARGET_KIND: dict[int, str] = {
    0: "crane", 1: "crane", 2: "crane", 3: "vessel", 4: "vessel", 5: "job", 6: "job",
    7: "job", 8: "vessel", 9: "vessel", 10: "job", 11: "horizon"}
_INJECT_KIND = {"EQUIPMENT_DOWN": EV_EQUIPMENT_DOWN, "EQUIPMENT_UP": EV_EQUIPMENT_UP,
                "PLAN_CHANGE": EV_PLAN_CHANGE}


# ───────────────────────────────────────────────── 번호표
@dataclass(frozen=True)
class IdTables:
    """번호 ↔ v5 문자열 id. 배열 세계는 번호만 들고, 이름은 여기서만 찾는다."""

    job_ids: tuple[str, ...]      # n → job_id  (n < n0; n ≥ n0 은 빈 칸)
    cont_ids: tuple[str, ...]     # c → container_id (c < c0 초기, c0+n 은 "IN_{job_id}")
    crane_ids: tuple[str, ...]    # k → crane_id
    vessel_ids: tuple[str, ...]   # v → vessel_id
    lane_ids: tuple[str, ...]     # l → lane_id
    n0: int                       # 실제 오더 수 (≤ N)
    c0: int                       # 초기 컨테이너 수
    ledger_mode: bool             # 시간계약 v2 장부 활성 (engine.py:127-129)
    n_units: int = 0              # 실제 이송차 수 (profile.transfer.n_units; 배열 U 는 max(1, ·) — 조각 4)

    # 역표 — 자주 쓰이므로 속성으로 (frozen 이라 dict 필드 대신 매번 만든다; 크기 작음)
    @property
    def job_index(self) -> dict[str, int]:
        return {j: i for i, j in enumerate(self.job_ids)}

    @property
    def cont_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.cont_ids)}

    @property
    def crane_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.crane_ids)}

    @property
    def vessel_index(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.vessel_ids)}

    def job(self, n: int) -> str | None:
        return self.job_ids[n] if 0 <= n < len(self.job_ids) else None

    def crane(self, k: int) -> str | None:
        return self.crane_ids[k] if 0 <= k < len(self.crane_ids) else None

    def vessel(self, v: int) -> str | None:
        return self.vessel_ids[v] if 0 <= v < len(self.vessel_ids) else None


class _ValidateShim:
    """v5 `TerminalSimulator._validate` (engine.py:198-221) 가 읽는 속성 셋만 흉내 낸다.

    엔진 인스턴스를 만들지 않고(=reset 을 돌리지 않고) 검증 함수 **그 자체**를 부른다.
    """

    def __init__(self, profile, scenario):
        self.profile = profile
        self.scenario = scenario

    def jobs_input(self):
        return self.scenario.jobs


def _validate_v5(profile, scenario) -> None:
    from ..world.integrated.engine import TerminalSimulator   # 호스트 전용 — gpu/ 순수 경로엔 안 들어옴
    TerminalSimulator._validate(_ValidateShim(profile, scenario))


def _flow_index(flow) -> int:
    return FLOW_NAMES.index(flow.value if hasattr(flow, "value") else str(flow))


def _or(v, empty):
    """None → 빈칸 표시."""
    return empty if v is None else float(v)


# ───────────────────────────────────────────────── v5 → 배열
def _wake_seeds(profile, scenario, job_ids) -> list[tuple[float, str]]:
    """engine.py:166-171 — (시각, job_id) 정렬 목록 (GATE_OUT · 대상 있음 · provided_eta 있음 · max(0, eta−horizon) < end)."""
    horizon = float(profile.decision_horizon_s)
    end = float(scenario.end_time)
    return sorted((max(0.0, j.provided_eta - horizon), j.job_id)
                  for j in scenario.jobs
                  if j.flow.value == "GATE_OUT" and j.target_container is not None
                  and j.provided_eta is not None
                  and max(0.0, j.provided_eta - horizon) < end)


def to_block_world(profile, scenario, *, n_max: int, q_cap: int, log_cap: int,
                   n_wake: int | None = None, n_defer: int = 0, n_review: int = 0,
                   c_max: int | None = None, v_max: int | None = None,
                   p_cap: int | None = None) -> tuple[BlockWorld, IdTables]:
    """v5 `IntegratedProfile` + `TerminalScenario` → reset 직후의 `BlockWorld` 와 번호표.

    n_max  오더 칸 N (실제 수보다 크면 뒤는 빈 칸 block=-1)
    q_cap  사건 큐 칸 — 시드 사건이 넘치면 조용히 overflow 하지 않고 **여기서 실패**한다
    log_cap 사건 로그 칸
    c_max  컨테이너 칸 C (기본 = 초기 수 + n_max). ★서로 다른 야드(초기 컨테이너 수가 다름)를
           `vmap` 으로 쌓으려면 모양이 같아야 하므로 공통 상한을 준다 — 뒤 칸은 'PAD_#i' 예비칸
           (좌표 -1·alive False). 반입 예비칸 번호 C0+n 은 세계별 값이라 그대로다. 같은 프로파일
           (K·L·B·R·T 동일)이어야 쌓인다.
    n_wake ETA wake 칸 W (기본 None = 시드될 수만큼; 모자라면 여기서 실패). n_defer DEFER 칸 D (LEGACY 는 0).
    v_max  배 칸 V (기본 = 실제 수, 0 이면 가짜 1칸) · p_cap 이송 대기 링버퍼 (기본 max(8, Σ total_moves)).
    """
    _validate_v5(profile, scenario)                                   # 94행
    g = Geom.from_profile(profile)
    geom = profile.block
    B, R, T = g.bay_count, g.row_count, g.tier_max

    # ── 번호표 ────────────────────────────────────────────────
    jobs = sorted(scenario.jobs, key=lambda j: j.job_id)               # 227행
    n0 = len(jobs)
    if n0 > n_max:
        raise ValueError(f"오더 {n0} 개 > n_max {n_max}")
    crane_specs = sorted(profile.cranes, key=lambda s: s.crane_id)     # cranes.py:54
    crane_ids = tuple(s.crane_id for s in crane_specs)
    K = len(crane_ids)
    vessel_ids = tuple(sorted(v.vessel_id for v in scenario.vessels))  # 236행
    lane_ids = tuple(profile.lane_graph.lane_ids)
    init_ids = sorted(scenario.containers)                             # stack.py:18
    c0 = len(init_ids)
    # 반입 예비칸: 실제 오더는 v5 이름 IN_{job_id}, 빈 오더 칸은 자리표시 이름
    cont_ids = (init_ids
                + [f"IN_{j.job_id}" for j in jobs]
                + [f"IN_#{n}" for n in range(n0, n_max)])
    if c_max is not None:
        if c_max < len(cont_ids):
            raise ValueError(f"c_max {c_max} < 필요한 컨테이너 칸 {len(cont_ids)} (초기 {c0} + n_max {n_max})")
        cont_ids += [f"PAD_#{i}" for i in range(len(cont_ids), c_max)]
    cont_ids = tuple(cont_ids)
    job_ids = tuple(j.job_id for j in jobs)
    # ★장부 모드(외부트럭 중 exit_travel 있는 것이 하나라도 있음)에 exit_travel 없는 외부트럭이 섞이면
    #   v5 는 그 트럭의 BLOCK_ARRIVAL 에서 KeyError(time_contract.py:61) 로 죽는다 — 여기서 큰 소리로 막는다
    #   (엔진도 V_LEDGER_UNREGISTERED 비트로 방어하지만 입력 검증은 호스트 몫).
    ext_jobs = [j for j in jobs if j.is_external_truck]
    if any(j.exit_travel_s is not None for j in ext_jobs) and any(j.exit_travel_s is None for j in ext_jobs):
        bad = [j.job_id for j in ext_jobs if j.exit_travel_s is None]
        raise ValueError(f"장부 모드인데 exit_travel_s 가 없는 외부트럭 {bad} — v5 는 BLOCK_ARRIVAL 에서 KeyError")
    cont_idx = {c: i for i, c in enumerate(cont_ids)}
    crane_idx = {c: i for i, c in enumerate(crane_ids)}
    vessel_idx = {v: i for i, v in enumerate(vessel_ids)}

    # ── 격자·컨테이너 (96행 → from_v5_stacks) ─────────────────────
    from ..world.sim.stack import YardStacks                          # 호스트 전용
    yard = YardStacks(geom, copy.deepcopy(scenario.containers))
    stacks, conts, _ = from_v5_stacks(yard, g, cont_ids=list(cont_ids), n_cont=len(cont_ids))

    # 시간계약 v2 장부 활성 여부 (127-129행)
    ext_v2 = [j for j in jobs if j.is_external_truck and j.exit_travel_s is not None]
    ledger_mode = bool(ext_v2)

    wakes = _wake_seeds(profile, scenario, job_ids)                  # 166-171행
    if n_wake is None:
        n_wake = len(wakes)
    if len(wakes) > n_wake:
        raise ValueError(f"ETA wake {len(wakes)} 건 > n_wake {n_wake}")
    n_units = int(profile.transfer.n_units)
    if p_cap is None:
        p_cap = max(8, sum(int(v.plan.total_moves) for v in scenario.vessels))
    V = len(vessel_ids) if v_max is None else int(v_max)
    if len(vessel_ids) > V:
        raise ValueError(f"배 {len(vessel_ids)} 척 > v_max {V}")
    pc_rows = [ie for ie in scenario.injected_events if ie.kind == "PLAN_CHANGE"]
    j_max = max([len(list(dict(ie.data or ()).get("job_deadlines", ()))) for ie in pc_rows] + [0])
    world = empty_block_world(g, n_orders=n_max, n_cranes=K, n_conts=len(cont_ids),
                              q_cap=q_cap, log_cap=log_cap, end_s=float(scenario.end_time),
                              n_wake=n_wake, n_defer=n_defer, n_review=n_review,
                              v_max=V, n_units=n_units, p_cap=int(p_cap),
                              move_time_s=float(profile.transfer.move_time_s),
                              i_max=len(pc_rows), j_max=j_max)                     # 144행 end

    # ── 오더 (domain/models.py:28-56 → OrderArrays 엔진 열) ──────────
    o = {f: np.asarray(getattr(world.orders, f)).copy() for f in world.orders._fields}
    c_size = np.asarray(conts.c_size).copy()
    for n, j in enumerate(jobs):
        o["block"][n] = 0                                              # 단일 블록
        o["flow"][n] = _flow_index(j.flow)
        o["status"][n] = JS_PLANNED                                    # models.py:52
        o["is_external"][n] = bool(j.is_external_truck)
        o["is_vessel"][n] = bool(j.is_vessel_linked)
        is_store = j.inbound_size is not None                          # models.py:74-75
        o["is_store"][n] = is_store
        o["target_cont"][n] = cont_idx[j.target_container] if j.target_container is not None else EMPTY_ID
        if is_store:
            o["inbound_cont"][n] = c0 + n                              # engine.py:604 사전 배정
            o["inbound_size"][n] = SIZE_INDEX[j.inbound_size.value]
            c_size[c0 + n] = SIZE_INDEX[j.inbound_size.value]
        o["vessel"][n] = vessel_idx.get(j.vessel_id, EMPTY_ID) if j.vessel_id is not None else EMPTY_ID
        o["assigned_crane"][n] = EMPTY_ID
        o["rehandles"][n] = 0
        o["release_s"][n] = float(j.release_time)
        o["provided_eta_s"][n] = _or(j.provided_eta, EMPTY_TIME)
        o["deadline_s"][n] = _or(j.deadline, EMPTY_TIME)
        o["exit_travel_s"][n] = _or(j.exit_travel_s, -1.0)
        o["actual_arrival_s"][n] = _or(j.actual_block_arrival, EMPTY_TIME)
        #: v5 `Job.appointment_gate_time` (models.py:41) — 예약 원점. 없으면 +inf.
        o["appt_s"][n] = _or(getattr(j, "appointment_gate_time", None), EMPTY_TIME)
        # 131-132행: 장부 등록 = 외부트럭·exit_travel 있음 → A = actual_gate_in or 0.0
        if ledger_mode and j.is_external_truck and j.exit_travel_s is not None:
            o["gate_in_s"][n] = float(j.actual_gate_in or 0.0)
        o["waiting"][n] = False
        o["in_block"][n] = False
        o["wait_sample_s"][n] = np.nan
    orders = world.orders._replace(**{f: jnp.asarray(v) for f, v in o.items()})
    conts = conts._replace(c_size=jnp.asarray(c_size))

    # ── 크레인 (98-118행 + models.py:121-132 스펙) ────────────────
    cr = {f: np.asarray(getattr(world.cranes, f)).copy() for f in world.cranes._fields}
    pos: list[float] = [float(s.service_bay_min) for s in crane_specs]    # cranes.py:38
    groups: dict[tuple[int, int], list[int]] = {}
    for k, s in enumerate(crane_specs):                                  # 105-108행 (id 순 = k 순)
        groups.setdefault((s.service_bay_min, s.service_bay_max), []).append(k)
    for (lo, hi), ks in groups.items():
        n = len(ks)
        if n > 1:
            for i, k in enumerate(ks):                                    # 112-113행 — 파이썬 float
                pos[k] = lo + (i + 0.5) * (hi - lo) / n
    for k, s in enumerate(crane_specs):
        cr["block"][k] = 0
        cr["status"][k] = CR_IDLE
        cr["available_at"][k] = 0.0                                       # models.py:89
        cr["assigned"][k] = EMPTY_ID
        cr["bay"][k] = pos[k]
        cr["row"][k] = float(geom.transfer_row)                           # cranes.py:39
        cr["bay_min"][k] = int(s.service_bay_min)
        cr["bay_max"][k] = int(s.service_bay_max)
        cr["spec_gantry"][k] = float(s.gantry_speed_mps)
        cr["spec_trolley"][k] = float(s.trolley_speed_mps)
        cr["spec_hoist_loaded"][k] = float(s.hoist_speed_loaded_mps)
        cr["spec_hoist_empty"][k] = float(s.hoist_speed_empty_mps)
        cr["spec_lock"][k] = float(s.lock_time_s)
        cr["spec_unlock"][k] = float(s.unlock_time_s)
        cr["spec_truck_pos"][k] = float(s.truck_positioning_time_s)
    for k, s in enumerate(crane_specs):
        if not (s.gantry_speed_mps > 0 and s.trolley_speed_mps > 0
                and s.hoist_speed_loaded_mps > 0 and s.hoist_speed_empty_mps > 0):
            raise ValueError(f"크레인 {s.crane_id} 속도 스펙이 0 이하 — 이동시간이 inf 가 된다")
    # 116-118행: 레일 순서 = (초기 bay, id) 오름차순. rail_order[i] = i 번째 크레인 번호
    cr["rail_order"][:] = np.asarray(sorted(range(K), key=lambda k: (pos[k], crane_ids[k])), np.int32)
    cranes = world.cranes._replace(**{f: jnp.asarray(v) for f, v in cr.items()})

    # ── 예약표: idle 장벽 (114-115행) ─────────────────────────────
    res = world.res._replace(idle_pos=jnp.asarray(np.asarray(pos, dtype=np.float64)))

    # ── 레인 인접표 (137행 LaneNetwork.__post_init__, lane.py:19-24) ──
    L = len(lane_ids)
    adj = np.zeros((L, L), bool)
    lane_pos = {l: i for i, l in enumerate(lane_ids)}
    for a, b in profile.lane_graph.edges:
        if a in lane_pos and b in lane_pos:
            adj[lane_pos[a], lane_pos[b]] = adj[lane_pos[b], lane_pos[a]] = True
    lane = world.lane._replace(adj=jnp.asarray(adj))

    # ── 사건 큐 시드 (226-243행 `_seed_events`) — 같은 순서로 넣는다 ──
    seeds: list[tuple[float, int, int]] = []
    from ..world.domain.enums import ServiceMode                       # 233행 판정용
    for n, j in enumerate(jobs):
        if j.is_external_truck:
            seeds.append((float(j.actual_block_arrival), EV_BLOCK_ARRIVAL, n))       # 229행
        else:
            if j.is_vessel_linked and j.service_mode == ServiceMode.STORE:         # 233-234행
                continue
            seeds.append((float(j.release_time), EV_JOB_RELEASED, n))                # 235행
    for v in sorted(scenario.vessels, key=lambda x: x.vessel_id):                    # 236-237행
        seeds.append((float(v.plan.planned_start_s), EV_VESSEL_START, vessel_idx[v.vessel_id]))
    for ie in sorted(scenario.injected_events, key=lambda x: (x.time, x.kind, x.target)):  # 238-242행
        kind = _INJECT_KIND[ie.kind]
        if kind == EV_PLAN_CHANGE:
            tgt = vessel_idx.get(ie.target, EMPTY_ID)
        else:
            tgt = crane_idx.get(ie.target, EMPTY_ID)          # 없는 id → -1 (처리기가 무시)
        seeds.append((float(ie.time), kind, tgt))
    seeds.append((float(scenario.horizon_s), EV_HORIZON, EMPTY_ID))                  # 243행
    if len(seeds) > q_cap:
        raise ValueError(f"시드 사건 {len(seeds)} 건 > q_cap {q_cap} — 큐 칸을 늘려라")
    queue = _seed_queue(q_cap, seeds)

    # ── ETA wake (166-171행) — 조각 3 ─────────────────────────────
    wake = world.wake
    if n_wake > 0:
        ws = np.full((n_wake,), EMPTY_TIME, np.float64)
        wj = np.full((n_wake,), EMPTY_ID, np.int32)
        jidx = {j: i for i, j in enumerate(job_ids)}
        for i, (t, jid) in enumerate(wakes):
            ws[i], wj[i] = t, jidx[jid]
        wake = wake._replace(eta_wake_s=jnp.asarray(ws), eta_wake_job=jnp.asarray(wj))

    # ── 본선·이송·계획변경 (134-138행) — 조각 4 ────────────────────
    vessels = vessel_arrays_from_scenario(scenario, job_ids, vessel_ids, n_max, v_max=max(1, V))
    transfer = world.transfer
    if n_units > 0:
        transfer = transfer_arrays_from_profile(profile, p_cap=int(p_cap))
    plan_change = plan_change_from_scenario(scenario, job_ids, vessel_ids, i_max=len(pc_rows), j_max=j_max)

    # rate 는 reset 의 `_refresh_rates` (196행) 결과 = 전부 0 (예약·양보·본선 없음)
    world = world._replace(orders=orders, cranes=cranes, stacks=stacks, conts=conts, res=res,
                           lane=lane, queue=queue, wake=wake, vessels=vessels, transfer=transfer,
                           plan_change=plan_change)
    tables = IdTables(job_ids=job_ids, cont_ids=cont_ids, crane_ids=crane_ids,
                      vessel_ids=vessel_ids, lane_ids=lane_ids, n0=n0, c0=c0,
                      ledger_mode=ledger_mode, n_units=n_units)
    return world, tables


def _seed_queue(q_cap: int, seeds: list[tuple[float, int, int]]) -> EventArray:
    """`push_event` 를 차례로 부른 것과 **같은 배열** — 칸 i 에 i 번째 시드, seq = 0,1,2,… (counter 0 시작).

    numpy 로 한 번에 만든다 (오더 만 단위에서 push 를 하나씩 부르면 느리다).
    """
    q = empty_queue(q_cap)
    time = np.full((q_cap,), EMPTY_TIME, np.float64)
    kind = np.full((q_cap,), EMPTY_ID, np.int32)
    target = np.full((q_cap,), EMPTY_ID, np.int32)
    seq = np.full((q_cap,), EMPTY_ID, np.int32)
    for i, (t, k, tg) in enumerate(seeds):
        time[i], kind[i], target[i], seq[i] = t, k, tg, i
    return q._replace(time=jnp.asarray(time, TIME_DTYPE), kind=jnp.asarray(kind),
                      target=jnp.asarray(target), seq=jnp.asarray(seq),
                      counter=jnp.asarray(len(seeds), jnp.int32))


# ───────────────────────────────────────────────── 배열 → v5 모양
def _payload(kind: int, target: int, tables: IdTables) -> str:
    """큐 사건 0..11 의 payload 문자열 (engine.py:845 가 적는 그대로)."""
    what = _TARGET_KIND.get(kind)
    if what == "crane":
        return tables.crane(target) or ""
    if what == "job":
        return tables.job(target) or ""
    if what == "vessel":
        return tables.vessel(target) or ""
    if what == "horizon":
        return "HORIZON"
    return ""


def queue_entries(queue: EventArray, tables: IdTables) -> list[tuple[float, int, int, str, str]]:
    """큐 배열 → (time, priority, seq, kind_name, payload) 를 **꺼내는 순서**로 정렬한 목록.

    v5 `EventQueue._heap` 의 `_Entry(time, priority, seq, payload, …, kind_name)` 와 나란히 놓는다.
    """
    time = np.asarray(queue.time); kind = np.asarray(queue.kind)
    target = np.asarray(queue.target); seq = np.asarray(queue.seq)
    prio = np.asarray(PRIO)
    out = []
    for i in range(time.shape[0]):
        if not np.isfinite(time[i]):
            continue
        k = int(kind[i])
        out.append((float(time[i]), int(prio[k]), int(seq[i]), EV_NAMES[k],
                    _payload(k, int(target[i]), tables)))
    out.sort(key=lambda e: (e[0], e[1], e[2]))
    return out


def repo_job_id(cid: str, bay: float) -> str:
    """v5 REPOSITION 후보의 job_id — candidates.py:413 `f"REPO:{cid}:{int(tb)}"` (int 는 절사)."""
    return f"REPO:{cid}:{int(bay)}"


def event_log_from_arrays(world: BlockWorld, tables: IdTables) -> list[tuple[float, str, str]]:
    """로그 배열 (t, kind, target, aux) → v5 `event_log` 의 (time, kind_name, payload) 목록 (머리말 참조)."""
    log = world.log
    n = int(log.n)
    t = np.asarray(log.t)[:n]; kind = np.asarray(log.kind)[:n]; target = np.asarray(log.target)[:n]
    aux = np.asarray(log.aux)[:n]
    service_s = np.asarray(world.orders.service_s)
    assigned = np.asarray(world.orders.assigned_crane)
    out: list[tuple[float, str, str]] = []
    for i in range(n):
        k, tg, ti, ax = int(kind[i]), int(target[i]), float(t[i]), int(aux[i])
        if k < 0:
            continue
        if k < LOG_DISPATCH:
            out.append((ti, EV_NAMES[k], _payload(k, tg, tables)))
        elif k == LOG_DISPATCH:
            cid = tables.crane(tg) or ""
            if ax >= 0:
                jid = tables.job(ax)
            elif ax <= -2:
                jid = repo_job_id(cid, -ax - 2)
            else:                                                    # aux 없음(-1, 손으로 만든 로그) — 오더 표에서 되찾는다
                hit = np.nonzero((assigned == tg) & (service_s == ti))[0]   # (service_s == t & assigned_crane == k)
                jid = tables.job(int(hit[0])) if hit.size else None
            out.append((ti, "DISPATCH", f"{cid}:{jid}"))
        elif k == LOG_ETA_WAKE:
            out.append((ti, "ETA_WAKE", tables.job(tg) or ""))
        elif k == LOG_DEFER_WAKE:
            out.append((ti, "DEFER_WAKE", ""))
        elif k == LOG_DEADLOCK_ESCAPE:
            esc = [tables.crane_ids[kk] for kk in range(len(tables.crane_ids)) if (tg >> kk) & 1]
            out.append((ti, "DEADLOCK_ESCAPE", ",".join(esc)))
        else:
            out.append((ti, f"UNKNOWN_{k}", str(tg)))
    return out


def event_stream_hash(world: BlockWorld, tables: IdTables) -> str:
    """v5 `event_stream_hash` (engine.py:1147-1149) 와 같은 식으로 로그를 해시한다."""
    import hashlib
    blob = "|".join(f"{round(t, 6)}:{k}:{p}" for (t, k, p) in event_log_from_arrays(world, tables))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _f(x) -> float:
    return float(np.asarray(x))


def _opt_time(x) -> float | None:
    """+inf(빈 시각) → None."""
    v = _f(x)
    return None if not np.isfinite(v) else v


def from_block_world(world: BlockWorld, tables: IdTables) -> dict[str, Any]:
    """배열 세계 → v5 객체 모양의 파이썬 값 (시험이 v5 와 항목별로 대조).

    반환 키: clock end terminal last_decision_at jobs cranes piles containers reservations
             idle_positions rail_order queue event_log kpi ledger cost_episode cost_pending
             cost_rate lane_cong_area violation overflow steps
    """
    o, cr, rs, pl = world.orders, world.cranes, world.res, world.plan
    n0, K = tables.n0, len(tables.crane_ids)
    jobs: dict[str, dict] = {}
    for n in range(n0):
        jobs[tables.job_ids[n]] = {
            "status": STATUS_NAMES[int(o.status[n])],
            "assigned_crane": tables.crane(int(o.assigned_crane[n])),
            "service_start": _opt_time(o.service_s[n]),
            "service_end": _opt_time(o.done_s[n]),
            "actual_gate_out": _opt_time(o.gate_out_s[n]),
            "rehandle_count": int(o.rehandles[n]),
            "block_arrival": _opt_time(o.block_in_s[n]),
            "gate_in": _opt_time(o.gate_in_s[n]),
            "waiting": bool(o.waiting[n]),
            "in_block": bool(o.in_block[n]),
            "wait_sample": (None if np.isnan(_f(o.wait_sample_s[n])) else _f(o.wait_sample_s[n])),
        }
    cranes: dict[str, dict] = {}
    for k in range(K):
        a_job = tables.job(int(cr.assigned[k]))
        if int(cr.assigned[k]) >= 0 and a_job is None and int(pl.kind[k]) == PK_REPOSITION:
            a_job = repo_job_id(tables.crane_ids[k], _f(pl.end_bay[k]))      # engine.py:699 assigned_job = plan.job_id
        cranes[tables.crane_ids[k]] = {
            "position_bay": _f(cr.bay[k]), "trolley_row": _f(cr.row[k]),
            "available_at": _f(cr.available_at[k]),
            "assigned_job": a_job,
            "status": CRANE_STATUS_NAMES[int(cr.status[k])],
            "down": bool(cr.down[k]), "down_pending": bool(cr.down_pending[k]),
            "yielded": bool(cr.yielded[k]), "is_loaded": bool(cr.is_loaded[k]),
            "recent_yield_count": int(cr.yield_count[k]),
            "recent_completions": int(cr.completions[k]), "served_count": int(cr.served[k]),
            "loaded_travel_m": _f(cr.loaded_m[k]), "empty_travel_m": _f(cr.empty_m[k]),
            "service_bay_min": int(cr.bay_min[k]), "service_bay_max": int(cr.bay_max[k]),
            "plan_kind": int(pl.kind[k]),
            "plan_job": (repo_job_id(tables.crane_ids[k], _f(pl.end_bay[k])) if int(pl.kind[k]) == PK_REPOSITION
                         else tables.job(int(pl.job[k]))),
        }
    containers: dict[str, tuple[int, int, int]] = {}
    cb, crow, ct, alive = (np.asarray(world.conts.c_bay), np.asarray(world.conts.c_row),
                           np.asarray(world.conts.c_tier), np.asarray(world.conts.c_alive))
    for c in range(len(tables.cont_ids)):
        if alive[c]:
            containers[tables.cont_ids[c]] = (int(cb[c]), int(crow[c]), int(ct[c]))
    reservations: dict[str, dict] = {}
    for k in range(K):
        if bool(rs.active[k]):
            slots = np.argwhere(np.asarray(rs.slots[k]))
            reservations[tables.crane_ids[k]] = {
                "job_token": tables.job(int(rs.token[k])),
                "corridor": (_f(rs.lo[k]), _f(rs.hi[k])),
                "lane_id": (tables.lane_ids[int(rs.lane[k])] if int(rs.lane[k]) >= 0 else None),
                "release_at": _f(rs.release_at[k]),
                "slots": frozenset((int(b) + 1, int(r) + 1) for b, r in slots),
            }
    idle_positions = {tables.crane_ids[k]: _f(rs.idle_pos[k]) for k in range(K)
                      if np.isfinite(_f(rs.idle_pos[k]))}
    kp, ld, cs = world.kpi, world.ledger, world.cost
    return {
        "clock": _f(world.clock), "end": _f(world.end_s), "terminal": bool(world.terminal),
        "last_decision_at": (None if _f(world.last_decision_at) == -np.inf else _f(world.last_decision_at)),
        # 조각 2 탈출 표식 (engine.py:181-184) — None 은 -inf
        "escape_at": (None if _f(world.escape_at) == -np.inf else _f(world.escape_at)),
        "escape_count": int(world.escape_count),
        "jobs": jobs, "cranes": cranes,
        "piles": to_v5_piles(world.stacks, list(tables.cont_ids)),
        "containers": containers,
        "reservations": reservations, "idle_positions": idle_positions,
        "rail_order": tuple(tables.crane_ids[int(k)] for k in np.asarray(cr.rail_order)),
        "queue": queue_entries(world.queue, tables),
        "event_log": event_log_from_arrays(world, tables),
        "kpi": {"queue_area_s": _f(kp.queue_area), "tail_area_s": _f(kp.tail_area),
                "loaded_gantry_m": _f(kp.loaded_m), "empty_gantry_m": _f(kp.empty_m),
                "rehandle_count": int(kp.rehandles), "completed_external": int(kp.completed_ext),
                "completed_vessel": int(kp.completed_ves), "berth_overrun_s": _f(kp.berth_overrun),
                "vessel_delay_s": _f(kp.vessel_delay_s),
                "pre_rehandle_count": int(kp.pre_rehandle_count),
                "positioning_count": int(kp.positioning_count)},
        "ledger": {"block_area_s": _f(ld.block_area), "block_tail_area_s": _f(ld.block_tail),
                   "terminal_area_s": _f(ld.terminal_area), "closed_end_s": _opt_time(ld.closed_end)},
        "cost_episode": {t: _f(cs.episode[i]) for i, t in enumerate(COST_TERMS)},
        "cost_pending": {t: _f(cs.pending[i]) for i, t in enumerate(COST_TERMS)},
        "cost_rate": {t: _f(cs.rate[i]) for i, t in enumerate(RATE_TERMS)},
        "lane_cong_area_s": _f(world.lane.cong_area_s),
        # 조각 4 — 배·이송차 (vessel.py 호스트 함수; 가짜 칸은 vessel_ids 밖이라 안 나온다)
        "vessels": vessels_to_v5(world.vessels, tables.vessel_ids),
        "transfer": _transfer_view(world, tables),
        "violation": int(world.violation), "violation_names": violation_names(int(world.violation)),
        "overflow": int(world.overflow) + int(world.queue.overflow) + int(world.transfer.overflow),
        "steps": int(world.steps),
    }


def _transfer_view(world: BlockWorld, tables: IdTables) -> dict:
    """transfer_to_v5 에서 가짜 유닛(n_units=0 일 때의 U=1 +inf) 을 뺀다."""
    d = transfer_to_v5(world.transfer, tables.vessel_ids)
    d["busy_until"] = d["busy_until"][:tables.n_units]
    return d
