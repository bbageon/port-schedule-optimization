"""터미널 변환기 — v5 다중블록 무대(`build_diurnal` 21블록) ↔ 배열 터미널 세계 ([[YR-327]] 조각 6 · key=convert).

`gpu/host_convert.to_block_world` 가 블록 **하나**를 v5 `reset()` 직후와 같게 만든다면, 여기서는 그것을
블록마다 부르고 **블록 축 B 를 맨 앞에** 쌓은 뒤, v5 `MultiBlockTerminal`(world/integrated/multiblock.py) 과
`ScheduledAnnouncer`(terminal_stream.py:780-825) 가 런타임에 만들던 것 — 트럭 명단·검토 시각·전역 원장 —
을 **호스트에서 한 번** 배열로 만든다. 조정자(조각 6 통합)는 이 배열만 읽고 쓴다. 반대 방향
(`from_terminal_world`)은 배열을 `scripts/v6/dump_ground_truth.py run_terminal` 의 결과 모양으로 되돌린다.

■ ★블록 번호 규약 — b = `built["scenarios"]` 의 **삽입 순서** (= `layout.ids` 순 = Y01…Y21)
    v5 `MultiBlockTerminal.__init__` 은 `dict(blocks)` 로 받고 `run` 은 `min(movable, key=clock)` (192행) 으로 다음
    블록을 고른다 — 파이썬 `min` 은 동률에서 **첫 원소** = dict 삽입 순서를 돌려준다. 그 dict 는
    `{b: sim for b, s in built["scenarios"].items()}` 라 삽입 순서 = `layout.ids` 순서 (terminal_stream.py:700-708)
    = 문자열 정렬 순서(Y01 < … < Y21) 이기도 하다. 그래서 번호 b = `list(built["scenarios"]).index(bid)` 로 두면
    "삽입순" 과 "정렬순" 두 규약이 같은 번호를 준다 (`TerminalTables.block_ids` 가 그 순서를 보존한다).
    영향: totals 의 `sum(totals.values())` 덧셈 순서 (from_terminal_world 가 같은 순서로 더한다). 검토 시각 사이
    블록은 독립이라 나머지 답에는 영향이 없다 (piece6_spec ■ 블록 전진 순서).

■ ★오더 행 번호 규약 — 블록 안에서 **sorted(namespaced job_id) 순위** (트럭이 본선 작업보다 앞)
    v5 는 구성 시 `_namespace_jobs` (multiblock.py:669-690) 로 job_id 를 `{block}:{id}` 로 통일하고, 런타임의
    모든 순서 의존 순회가 `sorted(self.jobs)` 다 (engine.py:477 candidates_for · 1001 양하 해제 · candidates.py:62,
    127, 185, 280, 432). 배열 엔진은 그 순서를 **오더 번호 n** 으로 재현한다 (cands3.py:573 `name_rank = n` ·
    engine_step.first_by_id · vessel.release_rank). 트럭 명단 id 는 `"Y01:D-00035"`, 본선 작업은 `"Y01:J-V-…"`
    이므로 `"D" < "J"` → **트럭이 앞 번호**를 받는다 — 그래야 트럭이 투입된 뒤의 tie-break 가 v5 와 같다.
    · 트럭 행 (n < n_trucks_b)   : 명단 순 (= `D-{i:05d}` 사전식 = 도착 순). 변환 직후에는 **없는 오더**
                                  (`block = -1`, `gate_in_s = +inf`) — v5 도 admit_external_job 전에는 sim.jobs 에 없다.
                                  나머지 열(flow·규격·대상·반입 예비칸·exit_travel·provided_eta·actual_arrival·travel)은
                                  `_job_from_entry(e, arrival)` 가 만드는 Job 그대로 미리 채워 둔다 → 투입 = block·gate_in·
                                  큐 push·원장 등록만 (조정자 몫).
    · 본선 작업 행                 : 그 뒤에 sorted 순. `block = b`.
    · 여분 행 (n ≥ n_used_b)       : 비어 있음 (block = -1) — 이송으로 들어오는 트럭 자리 (`n_spare`).
    ★`ledger_mode` (engine_step.py:214) 는 `any(is_external & exit_travel_s ≥ 0)` 라 미리 채운 트럭 행이 변환 직후부터
      장부 모드를 켠다 — v5 `ensure_time_ledger` (terminal_stream.py:767-777) 가 빈 장부를 처음부터 붙이는 것과 같다.
    ★반입 예비칸 = C0_b + n (host_convert 규칙 그대로), 이름은 v5 가 런타임에 만드는 `IN_{namespaced job_id}` (engine.py:604).
    ★`dispatch.resolver_params(tables_b, g)` 는 **namespaced 이름**으로 순위표를 굽는다 — REPO 이름 `"REPO:…"` 와 오더
      이름의 순서가 블록 단독(`"J-V…" < "REPO"`)과 터미널(`"Y01:J-V…" > "REPO"`)에서 **다르다**. 조정자는 반드시
      `TerminalTables.tables[b]` (namespaced) 로 순위표를 만들어야 v5 터미널 답이 나온다.

■ 명단·검토 시각·원장 (ScheduledAnnouncer · admission_epochs · _schedule_review_epochs · TerminalLedger)
    ScheduleArrays (S,)  명단 한 줄 = 트럭 하나. `notify_epoch` = `round((max(0, arr−lead) // period) · period, 6)` 가
                         검토 시각 배열의 몇 번째 칸인가 (-1 = 검토 시각이 아님 → v5 도 영원히 투입 안 됨).
                         같은 에폭 안 순서 = 명단 순서 (by_epoch 리스트 = schedule 순 append, 806행) = S 번호 오름차순.
                         `tail_skip` = `arr + travel > end_s` (808-811행 SKIP_TAIL, 호스트 파이썬 float 덧셈).
    EpochArrays   (E,)   v5 `_schedule_review_epochs` (149-167행): {round(A,6) | 반입 외부트럭 A∈[0,end]} ∪ {round(t,6) |
                         extra ≥ 0} 을 정렬. 블록별 유효 개수 `n_valid[b]` = #(t ≤ end_b) (접두 구간). `on_grid` = time_grid.on_grid.
                         같은 값이 `blocks.wake.review_s[b]` 에도 들어간다 (v5 `sim.review_epochs`; 소비는 엔진/조정자 몫).
    TerminalLedgerArrays 트럭 S 건의 JobRecord (multiblock.py:31-54) — 투입 전 `registered=False`. 본선 작업의 JobRecord 는
                         a_gate_in=None·GATE_IN 아님이라 표본·이송에 영향이 없어 배열로 두지 않는다(문서화). 조정자 상태
                         (`reserved_inbound`·`route_cost_s`·`txn_seq`) 도 여기.
    `epoch_idx` () int32  다음에 처리할 검토 시각 칸 — 세션을 나눠 이어 돌릴 때의 커서 (save/load 참조).

■ 모양 통일 (vmap 전제) — 블록마다 다른 것은 전부 **최대값**으로 맞추고 빈 칸을 마스크한다
    N = n_max (기본 max_b(n_trucks_b + n_vessel_jobs_b) + n_spare) · C = max_b C0_b + N · V = max_b 배 수 · P = max(8, max_b Σ moves)
    · W = max_b ETA wake 수(본선 작업만 — v5 는 투입 시 wake 를 안 만든다, multiblock.py:243-291) · Rv = E · PLAN_CHANGE (I,J) 최대.

■ 저장·복원 — `terminal_world_to_numpy` / `terminal_world_from_numpy` (잎 순서 = tree_flatten 순서; 형은 템플릿에서)
    `save_terminal_world(path, tw)` 는 npz 한 파일. 복원은 같은 입력으로 만든 템플릿(`to_terminal_world`) 의 treedef 를 쓴다.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import wake as WK
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .host_convert import (_INJECT_KIND, IdTables, _seed_queue, event_stream_hash, from_block_world,
                           to_block_world)
from .stack_ops import SIZE_INDEX
from .state import (EV_BLOCK_ARRIVAL, EV_HORIZON, EV_JOB_RELEASED, EV_PLAN_CHANGE, EV_VESSEL_START,
                    FL_GATE_IN, FL_GATE_OUT, JS_CANCELLED, JS_DONE, BlockWorld)
from .vessel import PlanChangeArrays, empty_plan_change

__all__ = ["ScheduleArrays", "EpochArrays", "TerminalLedgerArrays", "TerminalWorld", "TerminalTables",
           "to_terminal_world", "from_terminal_world", "block_slice", "stack_blocks", "block_row_ids",
           "block_tables", "review_epochs_v5", "on_grid",
           "terminal_world_to_numpy", "terminal_world_from_numpy", "save_terminal_world", "load_terminal_world",
           "DEFAULT_PERIOD_S", "CAPACITY_MARGIN"]

F = TIME_DTYPE
#: terminal_stream.WIP_ADMISSION_PERIOD_S — 투입 검토 격자
DEFAULT_PERIOD_S = 60.0
#: multiblock.CAPACITY_MARGIN — 수신 블록 여유 슬롯
CAPACITY_MARGIN = 2


# ───────────────────────────────────────────────── 배열
class ScheduleArrays(NamedTuple):
    """트럭 명단 S 건 — `ScheduledAnnouncer` 가 읽는 값 전부 (정적; 런에서 바뀌지 않는다)."""

    block: jnp.ndarray          # (S,) int32  최초 배정 블록 b (e["block"])
    row: jnp.ndarray            # (S,) int32  그 블록 안의 행 번호 n (머리말 행 번호 규약)
    flow: jnp.ndarray           # (S,) int32  FL_GATE_IN / FL_GATE_OUT
    size: jnp.ndarray           # (S,) int32  반입 규격 SZ_* (반출은 -1)
    target_cont: jnp.ndarray    # (S,) int32  반출 대상 컨테이너 (블록 b 의 번호; -1)
    inbound_cont: jnp.ndarray   # (S,) int32  반입 예비칸 = C0_b + row (반출은 -1)
    arrival_s: jnp.ndarray      # (S,) f64    게이트 진입 A = e["arrival_s"] (= 투입 시 gate_in_s)
    block_in_s: jnp.ndarray     # (S,) f64    블록 도착 = A + travel_s (Job.actual_block_arrival — 파이썬 float 덧셈 그대로)
    eta_s: jnp.ndarray          # (S,) f64    공개 예측 = A + travel_base_s (provided_eta = estimated_block_arrival)
    travel_s: jnp.ndarray       # (S,) f64
    travel_base_s: jnp.ndarray  # (S,) f64
    exit_travel_s: jnp.ndarray  # (S,) f64
    notice_s: jnp.ndarray       # (S,) f64    통지 = max(0, A − lead_s[s]) (announcer 795행 · V3Announcer 116행)
                                #             ★예약 원점(appointment)이 **아니다** — 그것은 orders.appt_s (state.py)
    notify_epoch: jnp.ndarray   # (S,) int32  검토 시각 칸 (-1 = 없음)
    tail_skip: jnp.ndarray      # (S,) bool   A + travel > end_s → SKIP_TAIL (투입 시도 자체를 안 한다)

    @property
    def s(self) -> int:
        return int(self.block.shape[0])


class EpochArrays(NamedTuple):
    """검토 시각 E 개 (전 블록 공통 정렬 목록) + 블록별 유효 개수."""

    t: jnp.ndarray         # (E,) f64   오름차순 (round(·,6) 적용된 값)
    on_grid: jnp.ndarray   # (E,) bool  time_grid.on_grid(t, period)
    n_valid: jnp.ndarray   # (B,) int32 블록별 #(t ≤ end_b) — 접두 구간
    period_s: jnp.ndarray  # ()   f64


class TerminalLedgerArrays(NamedTuple):
    """전역 원장 (multiblock.TerminalLedger · JobRecord) — 트럭 S 건 + 조정자 상태."""

    registered: jnp.ndarray        # (S,) bool  admit 됐나 (v5: records 에 있나)
    owner: jnp.ndarray             # (S,) int32 현재 실행 블록 (변환 직후 = 최초 배정 블록)
    row: jnp.ndarray               # (S,) int32 owner 블록 안의 행 (이송하면 수신 블록의 여분 행)
    origin: jnp.ndarray            # (S,) int32 최초 배정 블록 (감사용 불변)
    version: jnp.ndarray           # (S,) int32
    transfer_count: jnp.ndarray    # (S,) int32
    transfer_src: jnp.ndarray      # (S,T) int32  transfer_history (src, dst, t) — 빈 칸 -1 / +inf
    transfer_dst: jnp.ndarray      # (S,T) int32
    transfer_t: jnp.ndarray        # (S,T) f64
    a_gate_in: jnp.ndarray         # (S,) f64   A (+inf = None)
    entry_deferrals: jnp.ndarray   # (S,) int32
    entry_deferred_s: jnp.ndarray  # (S,) f64
    reserved_inbound: jnp.ndarray  # (B,) int32 `_reserved_inbound` (prepare 예약)
    route_cost_s: jnp.ndarray      # ()   f64   이송 추가주행 누적
    txn_seq: jnp.ndarray           # ()   int32 트랜잭션 id 발급기


class TerminalWorld(NamedTuple):
    """터미널 하나 — 블록 세계 B 개(잎 앞에 (B,)) + 명단 + 검토 시각 + 원장 + 에폭 커서. 전부 배열(pytree)."""

    blocks: BlockWorld
    sched: ScheduleArrays
    epochs: EpochArrays
    ledger: TerminalLedgerArrays
    epoch_idx: jnp.ndarray   # () int32  다음에 처리할 검토 시각 칸 (이어 돌리기 커서)

    @property
    def b(self) -> int:
        return int(self.blocks.clock.shape[0])


@dataclass(frozen=True)
class TerminalTables:
    """번호 ↔ 이름과 정적 상수 (호스트 전용). 배열 세계는 번호만 든다."""

    block_ids: tuple[str, ...]            # b → 블록 이름 (built["scenarios"] 삽입 순서)
    tables: tuple[IdTables, ...]          # b → 그 블록의 번호표 (job_ids 는 **namespaced**, 사용 행 n_used_b 개)
    n_trucks: tuple[int, ...]             # b → 명단 트럭 수 (행 0..n_trucks−1)
    n_vessel_jobs: tuple[int, ...]        # b → 시나리오(본선) 작업 수
    n_used: tuple[int, ...]               # b → n_trucks + n_vessel_jobs (그 뒤가 여분 행)
    truck_ids: tuple[str, ...]            # s → 명단 job_id ("Y01:D-00035")
    n_max: int
    c_max: int
    v_max: int
    p_cap: int
    n_wake: int
    n_review: int
    q_cap: int
    log_cap: int
    lead_s: float                         # 대표 리드 (항목별이면 lead_of[0] — 기록용)
    period_s: float
    end_s: float                          # built["sim_end_s"] — 투입 창 끝 (SKIP_TAIL 기준)
    observe_s: float                      # obs.observe_s — 턴타임 표본의 검열 기준 (a_to_o_samples_s(end))
    horizon_s: float                      # profile.decision_horizon_s
    gate_to_block_s: tuple[float, ...]    # (B,) layout.gate_to_block_s
    block_to_block_s: tuple[tuple[float, ...], ...]   # (B,B) layout.block_to_block_s
    phys_capacity: int                    # bay·row·tier (free_slots 의 phys)
    capacity_margin: int
    max_transfers: int
    lead_of: tuple[float, ...] = ()       # s → 그 트럭의 통지 리드 (V3Announcer 경로; 비어 있으면 전원 lead_s)

    @property
    def block_index(self) -> dict[str, int]:
        return {b: i for i, b in enumerate(self.block_ids)}

    @property
    def truck_index(self) -> dict[str, int]:
        return {j: i for i, j in enumerate(self.truck_ids)}


# ───────────────────────────────────────────────── 작은 도구
def on_grid(t: float, grid_s: float = DEFAULT_PERIOD_S) -> bool:
    """v5 time_grid.on_grid — 격자 시각 판정 (부동소수점 오차 1e-6 허용)."""
    r = t % grid_s
    return r < 1e-6 or grid_s - r < 1e-6


def _flow_code(name: str) -> int:
    return FL_GATE_OUT if name == "GATE_OUT" else FL_GATE_IN


def review_epochs_v5(scenarios: dict, extra_review_epochs=()) -> tuple[list[float], dict[str, int]]:
    """v5 `MultiBlockTerminal._schedule_review_epochs` (multiblock.py:149-167) 를 그대로 — (정렬 목록, 블록별 유효 개수).

    `scenarios` 는 블록 이름 → TerminalScenario. 반입(GATE_IN) 외부트럭의 A ∈ [0, end] 와 extra ≥ 0 을 round(·,6) 으로
    모아 정렬하고, 블록마다 `t <= end_b` 인 접두 개수를 센다.
    """
    ts: set[float] = set()
    for scn in scenarios.values():
        end = scn.end_time
        for j in scn.jobs:
            a = getattr(j, "actual_gate_in", None)
            if a is not None and j.flow.value == "GATE_IN" and 0.0 <= a <= end:
                ts.add(round(a, 6))
    ts.update(round(float(t), 6) for t in extra_review_epochs if float(t) >= 0.0)
    eps = sorted(ts)
    n_valid = {b: sum(1 for t in eps if t <= scn.end_time) for b, scn in scenarios.items()}
    return eps, n_valid


def _pad_plan_change(pc: PlanChangeArrays, i_max: int, j_max: int) -> PlanChangeArrays:
    """PLAN_CHANGE 표를 (I,J) 최대 모양으로 — 블록마다 주입 행 수가 달라도 쌓이게."""
    I, J = pc.i, int(pc.dl_job.shape[1])
    if (I, J) == (i_max, j_max):
        return pc
    e = empty_plan_change(i_max, j_max)
    out = {}
    for f in pc._fields:
        v = getattr(pc, f)
        base = getattr(e, f)
        if v.ndim == 1:
            out[f] = base.at[:I].set(v)
        else:
            out[f] = base.at[:I, :J].set(v)
    return PlanChangeArrays(**out)


def _seed_events_vessel_only(jobs_sorted, is_truck, scenario, vessel_idx, crane_idx) -> list[tuple[float, int, int]]:
    """host_convert.to_block_world 의 사건 시드 (engine.py:226-243 `_seed_events`) 에서 **트럭 행만 뺀** 목록.

    v5 터미널은 구성 시 sim.jobs 에 본선 작업만 있어 시드가 그것뿐이고, 트럭의 BLOCK_ARRIVAL 은 투입 때 push 된다.
    seq 는 이 목록의 위치 (0,1,2,…) — v5 큐의 seq 와 순서가 같다 (host_convert._seed_queue).
    """
    from ..world.domain.enums import ServiceMode                    # 호스트 전용 (host_convert 와 같은 판정)
    seeds: list[tuple[float, int, int]] = []
    for n, j in enumerate(jobs_sorted):
        if is_truck[n]:
            continue
        if j.is_external_truck:                                     # 시나리오에 처음부터 있던 외부트럭 (229행)
            seeds.append((float(j.actual_block_arrival), EV_BLOCK_ARRIVAL, n))
        else:
            if j.is_vessel_linked and j.service_mode == ServiceMode.STORE:    # 233-234행
                continue
            seeds.append((float(j.release_time), EV_JOB_RELEASED, n))          # 235행
    for v in sorted(scenario.vessels, key=lambda x: x.vessel_id):               # 236-237행
        seeds.append((float(v.plan.planned_start_s), EV_VESSEL_START, vessel_idx[v.vessel_id]))
    for ie in sorted(scenario.injected_events, key=lambda x: (x.time, x.kind, x.target)):   # 238-242행
        kind = _INJECT_KIND[ie.kind]
        tgt = vessel_idx.get(ie.target, EMPTY_ID) if kind == EV_PLAN_CHANGE else crane_idx.get(ie.target, EMPTY_ID)
        seeds.append((float(ie.time), kind, tgt))
    seeds.append((float(scenario.horizon_s), EV_HORIZON, EMPTY_ID))            # 243행
    return seeds


# ───────────────────────────────────────────────── v5 → 배열
def to_terminal_world(profile, built: dict, *, lead_s, n_max: int | None = None,
                      q_cap: int | None = None, log_cap: int | None = None,
                      extra_review_epochs=(), end_s: float | None = None,
                      period_s: float = DEFAULT_PERIOD_S, observe_s: float | None = None,
                      layout=None, n_spare: int = 0, max_transfers: int = 1,
                      capacity_margin: int = CAPACITY_MARGIN,
                      retarget=None, resolve_entry=None) -> tuple[TerminalWorld, TerminalTables]:
    """v5 무대 묶음 `built` (`terminal_stream.build_diurnal` 결과) → 변환 직후의 `TerminalWorld` 와 번호표.

    profile   IntegratedProfile (전 블록 공통 — 같은 Geom·크레인·이송 스펙이어야 쌓인다)
    built     {"scenarios": {블록: TerminalScenario}, "schedule": [명단], "sim_end_s", "observation", "layout", …}
    lead_s    통지 리드 — **스칼라 하나**(v2 `ScheduledAnnouncer`) 또는 **명단 길이 (S,) 열**(v3 `stage/orders.V3Announcer`
              — 트럭마다 자기 리드 `e["lead_s"]`). 명단 항목이 `"lead_s"` 키를 들고 있으면 그 값이 우선이다.
              통지 = max(0, 도착 − 그 트럭의 리드) → 통지 에폭 버킷(notify_epoch)·notice_s 가 트럭마다 갈린다.
    retarget / resolve_entry  v3 `V3Announcer` 전용 호스트 훅 — 배열판은 표현하지 못한다. None 이 아니면 **거절**한다
              (조용히 무시하면 NO_TARGET·CONTAINER_ID_CHANGED SKIP 이 통째로 사라진다).
    n_max     블록당 오더 칸 N (기본 = 필요 최대 + n_spare). 모자라면 여기서 실패한다.
    q_cap / log_cap  사건 큐·로그 칸 (기본 max(32, 4N) / 9N + 256 — test_gpu_y01 의 상한과 같은 식)
    extra_review_epochs  MultiBlockTerminal(extra_review_epochs=…) — 보통 admission_epochs(obs)
    end_s     ScheduledAnnouncer(end_s) — 기본 built["sim_end_s"] (SKIP_TAIL 기준)
    period_s  투입 격자 (ScheduledAnnouncer.period_s) — extra_review_epochs 와 같은 격자여야 한다
    observe_s 턴타임 표본의 검열 기준 (기본 built["observation"]["observe_s"])
    layout    YardLayout — 기본은 built["layout"] (as_dict, 좌표 round 6) 로 되살린다
    n_spare   블록당 여분 행 (이송 수신용) · max_transfers  이송 이력 칸 T · capacity_margin  free_slots 여유
    반환 (TerminalWorld, TerminalTables). 변환은 순수하다 — `built` 를 고치지 않는다 (Job 은 replace 로 복사).
    """
    from ..world.integrated.terminal_stream import _job_from_entry     # 호스트 전용 — v5 가 투입 때 쓰는 그 함수
    scns: dict = built["scenarios"]
    block_ids = tuple(scns)                                            # ★삽입 순서 (머리말)
    B = len(block_ids)
    bidx = {b: i for i, b in enumerate(block_ids)}
    schedule: list[dict] = built["schedule"]
    S = len(schedule)
    if retarget is not None or resolve_entry is not None:
        raise ValueError("retarget/resolve_entry 는 V3Announcer 전용 호스트 훅이라 배열판이 표현하지 못한다 "
                         "— 조용히 무시하지 않는다 (SKIP 사유 NO_TARGET·CONTAINER_ID_CHANGED 미이식)")
    _lead = np.asarray(lead_s, np.float64)
    if _lead.ndim == 0:
        lead_of = np.full((S,), float(_lead), np.float64)
    elif _lead.shape == (S,):
        lead_of = _lead.astype(np.float64)
    else:
        raise ValueError(f"lead_s 는 스칼라 또는 (S={S},) 열 — 받은 모양 {_lead.shape}")
    for i, e in enumerate(schedule):                       # 명단 자신의 리드가 있으면 그것이 정본 (V3Announcer 116행)
        if e.get("lead_s") is not None:
            lead_of[i] = float(e["lead_s"])
    end_s = float(built["sim_end_s"]) if end_s is None else float(end_s)
    observe_s = float(built["observation"]["observe_s"]) if observe_s is None else float(observe_s)
    if layout is None:
        from ..world.integrated.yard_layout import YardLayout
        ld = built["layout"]
        layout = YardLayout(tuple(ld["ids"]), tuple(float(p) for p in ld["positions_m"]), float(ld["speed_mps"]))
    geom = profile.block
    K = len(profile.cranes)
    horizon_s = float(profile.decision_horizon_s)

    # ── 명단을 블록별로 (schedule 순서 = 명단 순서 = 같은 에폭 안 투입 순서) ──
    per_block: dict[str, list[int]] = {b: [] for b in block_ids}
    for s, e in enumerate(schedule):
        if e["block"] not in per_block:
            raise ValueError(f"명단 {e['job_id']} 의 블록 {e['block']} 이 scenarios 에 없다")
        per_block[e["block"]].append(s)
    n_trucks = tuple(len(per_block[b]) for b in block_ids)
    n_vessel_jobs = tuple(len(scns[b].jobs) for b in block_ids)
    n_used = tuple(a + c for a, c in zip(n_trucks, n_vessel_jobs))
    need = max(n_used) + int(n_spare)
    if n_max is None:
        n_max = need
    if int(n_max) < need:
        raise ValueError(f"n_max {n_max} < 필요한 오더 칸 {need} (= max_b(트럭+본선작업) {max(n_used)} + 여분 {n_spare})")
    n_max = int(n_max)
    if q_cap is None:
        q_cap = max(32, 4 * n_max)
    if log_cap is None:
        log_cap = 8 * n_max + 256 + n_max
    c0s = [len(scns[b].containers) for b in block_ids]
    c_max = max(c0s) + n_max
    v_max = max(1, max(len(scns[b].vessels) for b in block_ids))
    p_cap = max(8, max(sum(int(v.plan.total_moves) for v in scns[b].vessels) for b in block_ids))

    # ── 검토 시각 (v5 _schedule_review_epochs) ──
    eps, n_valid_map = review_epochs_v5(scns, extra_review_epochs)
    E = len(eps)
    eps_index = {t: i for i, t in enumerate(eps)}

    # ── 트럭 Job (v5 투입이 만드는 그 객체) 과 namespaced 본선 작업 ──
    job_of: list = [_job_from_entry(e, e["arrival_s"]) for e in schedule]          # s → Job (812행)
    for e, j in zip(schedule, job_of):
        if j.exit_travel_s is None:
            raise ValueError(f"{e['job_id']}: exit_travel_s 결측 — v2 장부 계약 위반 (admit_external_job 이 거부한다)")
        # ★배열은 `estimated_block_arrival` 과 `provided_eta` 를 `provided_eta_s` 한 열로 합쳤다 (state.py 주석).
        #   v5 는 둘을 따로 밀어 주므로, 생성기가 둘을 다르게 만들면 조용히 한쪽을 버리는 셈이 된다 — 큰 소리로 막는다.
        est = getattr(j, "estimated_block_arrival", None)
        if est is not None and j.provided_eta is not None and est != j.provided_eta:
            raise ValueError(f"{e['job_id']}: estimated_block_arrival {est} ≠ provided_eta {j.provided_eta} — "
                             f"배열은 두 값을 한 열(provided_eta_s)로 합쳤다. 열을 분리해야 한다 (state.py 주석)")
    truck_jobs: dict[str, list] = {b: [job_of[s] for s in per_block[b]] for b in block_ids}

    # ── ETA wake 수 (본선 작업만 — 투입은 wake 를 만들지 않는다) → 공통 W ──
    def _n_wakes(scn) -> int:
        end = float(scn.end_time)
        return sum(1 for j in scn.jobs
                   if j.flow.value == "GATE_OUT" and j.target_container is not None and j.provided_eta is not None
                   and max(0.0, j.provided_eta - horizon_s) < end)
    W = max(_n_wakes(scns[b]) for b in block_ids)

    # ── 블록마다 변환 ──
    worlds: list[BlockWorld] = []
    tables: list[IdTables] = []
    row_of_truck = np.full((S,), EMPTY_ID, np.int32)
    for b, bid in enumerate(block_ids):
        scn = scns[bid]
        jobs_ns = [dataclasses.replace(j, job_id=f"{bid}:{j.job_id}") for j in scn.jobs]   # _namespace_jobs 669-690행
        scn_aug = dataclasses.replace(scn, jobs=jobs_ns + truck_jobs[bid])
        w, tb = to_block_world(profile, scn_aug, n_max=n_max, q_cap=q_cap, log_cap=log_cap,
                               n_wake=None, n_defer=0, n_review=E, c_max=c_max, v_max=v_max, p_cap=p_cap)
        truck_id_set = {j.job_id for j in truck_jobs[bid]}
        is_truck = np.asarray([jid in truck_id_set for jid in tb.job_ids], bool)
        m = int(is_truck.sum())
        if m != len(truck_id_set) or not is_truck[:m].all():
            raise AssertionError(f"{bid}: 트럭 행이 앞 번호가 아니다 — 번호 규약 위반 (머리말)")
        jobs_sorted = sorted(scn_aug.jobs, key=lambda j: j.job_id)
        for n, jid in enumerate(tb.job_ids):
            assert jobs_sorted[n].job_id == jid
        job_index = tb.job_index
        # 오더 열: 트럭 행은 '없는 오더' (block=-1 · gate_in +inf), 본선 작업 행은 block=b
        o = {f: np.asarray(getattr(w.orders, f)).copy() for f in w.orders._fields}
        for n in range(tb.n0):
            if is_truck[n]:
                o["block"][n] = EMPTY_ID
                o["gate_in_s"][n] = EMPTY_TIME
            else:
                o["block"][n] = b
        # 명단 → 행 · notice/travel 열
        for s in per_block[bid]:
            n = job_index[schedule[s]["job_id"]]
            row_of_truck[s] = n
            o["notice_s"][n] = max(0.0, schedule[s]["arrival_s"] - float(lead_of[s]))
            o["appt_s"][n] = EMPTY_TIME                    # 예약 원점은 **투입 때** 찍힌다 (v5 도 투입 전엔 Job 이 없다)
            o["travel_s"][n] = float(schedule[s]["travel_s"])
        orders = w.orders._replace(**{f: jnp.asarray(v) for f, v in o.items()})
        cranes = w.cranes._replace(block=jnp.full((K,), b, jnp.int32))
        # 큐: 본선 작업 시드만 (트럭 BLOCK_ARRIVAL 은 투입 때)
        seeds = _seed_events_vessel_only(jobs_sorted, is_truck, scn, tb.vessel_index, tb.crane_index)
        if len(seeds) > q_cap:
            raise ValueError(f"{bid}: 시드 사건 {len(seeds)} 건 > q_cap {q_cap}")
        queue = _seed_queue(q_cap, seeds)
        # wake: 마스크된 오더(트럭 행 block=-1)로 다시 시드 — 공통 W · 검토 시각 Rv=E
        wake, dropped = WK.seed_wakes(orders, K, horizon_s=horizon_s, end_s=float(scn.end_time),
                                      n_wake=W, n_defer=0, n_review=E)
        if int(dropped) != 0:
            raise AssertionError(f"{bid}: wake {int(dropped)} 건이 W={W} 를 넘었다")
        nv = n_valid_map[bid]
        rv = np.full((E,), EMPTY_TIME, np.float64)
        rv[:nv] = np.asarray(eps[:nv], np.float64)
        wake = wake._replace(review_s=jnp.asarray(rv, F))
        w = w._replace(orders=orders, cranes=cranes, queue=queue, wake=wake)
        worlds.append(w)
        tables.append(tb)
    # PLAN_CHANGE 표 모양 통일
    i_max = max(w.plan_change.i for w in worlds)
    j_max = max(int(w.plan_change.dl_job.shape[1]) for w in worlds)
    worlds = [w._replace(plan_change=_pad_plan_change(w.plan_change, i_max, j_max)) for w in worlds]
    blocks = stack_blocks(worlds)

    # ── 명단 배열 ──
    cols: dict[str, np.ndarray] = {
        "block": np.zeros((S,), np.int32), "row": row_of_truck, "flow": np.zeros((S,), np.int32),
        "size": np.full((S,), EMPTY_ID, np.int32), "target_cont": np.full((S,), EMPTY_ID, np.int32),
        "inbound_cont": np.full((S,), EMPTY_ID, np.int32),
        "arrival_s": np.zeros((S,), np.float64), "block_in_s": np.zeros((S,), np.float64),
        "eta_s": np.zeros((S,), np.float64), "travel_s": np.zeros((S,), np.float64),
        "travel_base_s": np.zeros((S,), np.float64), "exit_travel_s": np.zeros((S,), np.float64),
        "notice_s": np.zeros((S,), np.float64), "notify_epoch": np.full((S,), EMPTY_ID, np.int32),
        "tail_skip": np.zeros((S,), bool)}
    truck_ids: list[str] = []
    cont_index = [tb.cont_index for tb in tables]
    for s, e in enumerate(schedule):
        b = bidx[e["block"]]
        tb = tables[b]
        j = job_of[s]
        n = int(row_of_truck[s])
        truck_ids.append(e["job_id"])
        cols["block"][s] = b
        cols["flow"][s] = _flow_code(e["flow"])
        if e["flow"] == "GATE_OUT":
            cols["target_cont"][s] = cont_index[b][e["target"]]
        else:
            cols["size"][s] = SIZE_INDEX[j.inbound_size.value]
            cols["inbound_cont"][s] = tb.c0 + n
        arr = float(e["arrival_s"])
        cols["arrival_s"][s] = j.actual_gate_in                       # = arr
        cols["block_in_s"][s] = j.actual_block_arrival                # arr + travel_s (파이썬 float)
        cols["eta_s"][s] = j.provided_eta                             # arr + travel_base_s
        cols["travel_s"][s] = float(e["travel_s"])
        cols["travel_base_s"][s] = float(e["travel_base_s"])
        cols["exit_travel_s"][s] = float(e["exit_travel_s"])
        notify = max(0.0, arr - float(lead_of[s]))                    # 795행 · V3Announcer 116행
        cols["notice_s"][s] = notify
        slot = round((notify // period_s) * period_s, 6)              # 796행
        cols["notify_epoch"][s] = eps_index.get(slot, EMPTY_ID)
        cols["tail_skip"][s] = arr + float(e["travel_s"]) > end_s    # 808행
    sched = ScheduleArrays(**{f: jnp.asarray(v) for f, v in cols.items()})

    # ── 검토 시각 배열 ──
    epochs = EpochArrays(t=jnp.asarray(np.asarray(eps, np.float64), F),
                         on_grid=jnp.asarray(np.asarray([on_grid(t, period_s) for t in eps], bool)),
                         n_valid=jnp.asarray(np.asarray([n_valid_map[b] for b in block_ids], np.int32)),
                         period_s=jnp.asarray(float(period_s), F))

    # ── 원장 ──
    T = max(1, int(max_transfers))
    ledger = TerminalLedgerArrays(
        registered=jnp.zeros((S,), bool), owner=jnp.asarray(cols["block"]), row=jnp.asarray(row_of_truck),
        origin=jnp.asarray(cols["block"]), version=jnp.zeros((S,), jnp.int32),
        transfer_count=jnp.zeros((S,), jnp.int32),
        transfer_src=jnp.full((S, T), EMPTY_ID, jnp.int32), transfer_dst=jnp.full((S, T), EMPTY_ID, jnp.int32),
        transfer_t=jnp.full((S, T), EMPTY_TIME, F), a_gate_in=jnp.full((S,), EMPTY_TIME, F),
        entry_deferrals=jnp.zeros((S,), jnp.int32), entry_deferred_s=jnp.zeros((S,), F),
        reserved_inbound=jnp.zeros((B,), jnp.int32), route_cost_s=jnp.zeros((), F), txn_seq=jnp.zeros((), jnp.int32))

    tw = TerminalWorld(blocks=blocks, sched=sched, epochs=epochs, ledger=ledger, epoch_idx=jnp.zeros((), jnp.int32))
    tt = TerminalTables(
        block_ids=block_ids, tables=tuple(tables), n_trucks=n_trucks, n_vessel_jobs=n_vessel_jobs, n_used=n_used,
        truck_ids=tuple(truck_ids), n_max=n_max, c_max=c_max, v_max=v_max, p_cap=p_cap, n_wake=W, n_review=E,
        q_cap=int(q_cap), log_cap=int(log_cap), lead_s=float(lead_of[0]) if S else 0.0,
        lead_of=tuple(float(x) for x in lead_of), period_s=float(period_s), end_s=end_s,
        observe_s=observe_s, horizon_s=horizon_s,
        gate_to_block_s=tuple(layout.gate_to_block_s(b) for b in block_ids),
        block_to_block_s=tuple(tuple(layout.block_to_block_s(a, c) for c in block_ids) for a in block_ids),
        phys_capacity=int(geom.bay_count * geom.row_count * geom.tier_max),
        capacity_margin=int(capacity_margin), max_transfers=T)
    return tw, tt


# ───────────────────────────────────────────────── 쌓기·자르기
def stack_blocks(worlds) -> BlockWorld:
    """BlockWorld 목록 → 잎마다 앞에 (B,) 축. 모양·dtype 이 다르면 jnp.stack 이 실패한다 (조용히 안 맞춘다)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)


def block_slice(tw: TerminalWorld, b: int) -> BlockWorld:
    """블록 b 하나를 잘라 낸 BlockWorld (잎마다 x[b])."""
    return jax.tree_util.tree_map(lambda x: x[b], tw.blocks)


def block_row_ids(tw: TerminalWorld, tables: TerminalTables, b: int) -> tuple[str | None, ...]:
    """블록 b 의 행 n → job_id (없는 행 None) — **원장 기준** (이송으로 들어온 트럭은 여분 행에서 찾는다).

    정적 표(변환 직후 배치)로 시작해, 원장에서 owner==b 인 트럭을 (row) 에 덮어쓴다. 최초 배정 행에 있던 트럭이
    다른 블록으로 갔으면 그 행은 None.
    """
    tb = tables.tables[b]
    ids: list[str | None] = list(tb.job_ids) + [None] * (tables.n_max - len(tb.job_ids))
    owner = np.asarray(tw.ledger.owner); row = np.asarray(tw.ledger.row)
    reg = np.asarray(tw.ledger.registered)
    home_b = np.asarray(tw.sched.block); home_r = np.asarray(tw.sched.row)
    for s in range(len(tables.truck_ids)):
        if home_b[s] == b and (owner[s] != b or row[s] != home_r[s]):
            ids[int(home_r[s])] = None                              # 떠났거나 자리를 옮김
    for s in range(len(tables.truck_ids)):
        if owner[s] == b and (reg[s] or home_b[s] == b):
            ids[int(row[s])] = tables.truck_ids[s]
    return tuple(ids)


def block_tables(tw: TerminalWorld, tables: TerminalTables, b: int) -> IdTables:
    """블록 b 의 **현재** 번호표 — 여분 행에 들어온 트럭의 이름과 그 반입 예비칸 이름(IN_…)까지 반영."""
    tb = tables.tables[b]
    ids = block_row_ids(tw, tables, b)
    last = max([n for n, j in enumerate(ids) if j is not None] + [-1])
    job_ids = tuple(j if j is not None else f"#{n}" for n, j in enumerate(ids[:last + 1]))
    cont = list(tb.cont_ids)
    for n, j in enumerate(job_ids):
        if not j.startswith("#"):
            cont[tb.c0 + n] = f"IN_{j}"
    return dataclasses.replace(tb, job_ids=job_ids, cont_ids=tuple(cont), n0=len(job_ids))


# ───────────────────────────────────────────────── 배열 → v5 모양 (dump_ground_truth.run_terminal 의 결과 모양)
def _f(x) -> float:
    return float(np.asarray(x))


def from_terminal_world(tw: TerminalWorld, tables: TerminalTables, *, totals=None) -> dict[str, Any]:
    """배열 터미널 → `scripts/v6/dump_ground_truth.py run_terminal` 과 같은 모양의 dict (정답 JSON 과 직접 대조).

    키: mode load? 없음 · terminal_total · route_cost_s · end · admitted · n_turns · turn_sum_s · turn_samples_s ·
        blocks{bid: block·n_jobs·end_s·clock_s·n_cranes·event_hash·n_events·events·cost_raw·kpis·deadlock_escapes·unfinished
               + jobs·cranes·piles·containers·vessels·transfer·ledger·violation·overflow·terminal}
    totals  (B,) 블록별 구간비용 누적 (cost_fn 이 없으면 None = 전부 0.0). terminal_total = round(sum(순서대로), 6) —
            v5 209행 `sum(totals.values())` (파이썬 sum) 과 같은 순서·같은 함수.
    턴타임 표본 = v5 `TerminalLedger.a_to_o_samples_s(observe_s)` (84-94행): 등록된 트럭마다 O−A (O 확정) 또는
    max(0, observe_s − A). O 는 소유 블록 행의 gate_out_s. 표본 순서 = 명단 순서(= 등록 순서).
    """
    B = tw.b
    K = int(tw.blocks.cranes.block.shape[1])
    blocks_out: dict[str, dict] = {}
    gate_out_rows: list[np.ndarray] = []
    for b, bid in enumerate(tables.block_ids):
        w = block_slice(tw, b)
        tb = block_tables(tw, tables, b)
        d = from_block_world(w, tb)
        blk = np.asarray(w.orders.block); st = np.asarray(w.orders.status)
        gate_out_rows.append(np.asarray(w.orders.gate_out_s))
        present = [n for n in range(len(tb.job_ids)) if blk[n] >= 0]
        jobs = {tb.job_ids[n]: d["jobs"][tb.job_ids[n]] for n in present}
        log = d["event_log"]
        blocks_out[bid] = {
            "block": bid, "n_jobs": len(present), "end_s": d["end"], "clock_s": d["clock"], "n_cranes": K,
            "event_hash": event_stream_hash(w, tb), "n_events": len(log),
            "events": [[round(t, 6), k, p] for (t, k, p) in log],
            "cost_raw": d["cost_episode"],
            "kpis": {k: d["kpi"][k] for k in ("queue_area_s", "tail_area_s", "loaded_gantry_m", "empty_gantry_m",
                                             "rehandle_count", "pre_rehandle_count", "completed_external",
                                             "completed_vessel", "vessel_delay_s", "positioning_count")},
            "deadlock_escapes": d["escape_count"],
            "unfinished": int(sum(1 for n in present if int(st[n]) not in (JS_DONE, JS_CANCELLED))),   # 1140-1145행
            "jobs": jobs, "cranes": d["cranes"], "piles": d["piles"], "containers": d["containers"],
            "vessels": d["vessels"], "transfer": d["transfer"], "ledger": d["ledger"],
            "violation": d["violation"], "violation_names": d["violation_names"], "overflow": d["overflow"],
            "terminal": d["terminal"], "queue": d["queue"],
        }
    reg = np.asarray(tw.ledger.registered); owner = np.asarray(tw.ledger.owner); row = np.asarray(tw.ledger.row)
    a_in = np.asarray(tw.ledger.a_gate_in)
    appt_rows = np.asarray(tw.blocks.orders.appt_s)
    flow_rows = np.asarray(tw.blocks.orders.flow)
    ndef = np.asarray(tw.ledger.entry_deferrals); sdef = np.asarray(tw.ledger.entry_deferred_s)
    turns: list[float] = []
    for s in range(reg.shape[0]):
        if not reg[s]:
            continue
        a = float(a_in[s])
        o = float(gate_out_rows[int(owner[s])][int(row[s])])
        turns.append(o - a if np.isfinite(o) else max(0.0, tables.observe_s - a))
    ts = sorted(turns)
    # ── 이연 원장 (v5 time_sell.deferral_ledger 105-134행) — 원점은 **appt_s**, 통지(notice_s) 가 아니다 ──
    deferrals: list[dict] = []
    for s in sorted(range(reg.shape[0]), key=lambda i: tables.truck_ids[i]):
        if not reg[s] or int(ndef[s]) == 0:
            continue
        ow, rw = int(owner[s]), int(row[s])
        ap = float(appt_rows[ow, rw]); a = float(a_in[s])
        deferrals.append({
            "job_id": tables.truck_ids[s], "block": tables.block_ids[ow],
            "flow": "GATE_OUT" if int(flow_rows[ow, rw]) == FL_GATE_OUT else "GATE_IN",
            "n_deferrals": int(ndef[s]), "deferred_total_s": float(sdef[s]),
            "original_appointment_s": (None if not np.isfinite(ap) else ap),
            "actual_gate_in_s": (None if not np.isfinite(a) else a),
            "driver_outside_wait_s": (None if not (np.isfinite(ap) and np.isfinite(a)) else a - ap)})
    tot = [0.0] * B if totals is None else [float(x) for x in np.asarray(totals)]
    return {
        "mode": "terminal",
        "terminal_total": round(sum(tot), 6), "route_cost_s": _f(tw.ledger.route_cost_s),
        "end": max(float(x) for x in np.asarray(tw.blocks.end_s)),
        "admitted": int(reg.sum()), "n_turns": len(ts), "turn_sum_s": round(sum(ts), 6),
        "turn_samples_s": [round(t, 6) for t in ts],
        "totals": {bid: tot[b] for b, bid in enumerate(tables.block_ids)},
        "deferrals": deferrals,
        "epoch_idx": int(tw.epoch_idx),
        "blocks": blocks_out,
    }


# ───────────────────────────────────────────────── 저장·복원 (세션을 나눠 이어 돌리기)
def terminal_world_to_numpy(tw: TerminalWorld) -> dict[str, np.ndarray]:
    """잎 전부를 numpy 로 — 키 `leaf_00000…` 는 `jax.tree_util.tree_flatten` 순서."""
    leaves, _ = jax.tree_util.tree_flatten(tw)
    return {f"leaf_{i:05d}": np.asarray(x) for i, x in enumerate(leaves)}


def terminal_world_from_numpy(template: TerminalWorld, arrays: dict[str, np.ndarray]) -> TerminalWorld:
    """`terminal_world_to_numpy` 의 역 — 템플릿(같은 입력으로 만든 TerminalWorld)의 treedef·모양·dtype 을 따른다.

    모양이나 dtype 이 하나라도 다르면 조용히 캐스팅하지 않고 실패한다.
    """
    leaves, treedef = jax.tree_util.tree_flatten(template)
    if len(arrays) != len(leaves):
        raise ValueError(f"잎 수가 다르다: 저장 {len(arrays)} · 템플릿 {len(leaves)}")
    new = []
    for i, ref in enumerate(leaves):
        a = np.asarray(arrays[f"leaf_{i:05d}"])
        if a.shape != tuple(ref.shape) or a.dtype != np.dtype(ref.dtype):
            raise ValueError(f"잎 {i}: 모양/dtype 불일치 저장 {a.shape}/{a.dtype} · 템플릿 {tuple(ref.shape)}/{ref.dtype}")
        new.append(jnp.asarray(a, dtype=ref.dtype))
    return jax.tree_util.tree_unflatten(treedef, new)


def save_terminal_world(path, tw: TerminalWorld) -> None:
    """npz 한 파일로 (압축 없음 — 복원 속도 우선)."""
    np.savez(path, **terminal_world_to_numpy(tw))


def load_terminal_world(path, template: TerminalWorld) -> TerminalWorld:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return terminal_world_from_numpy(template, arrays)
