"""30일 무대 — v5 `stage/month_engine.MonthTerminal` + `stage/month_run.run_month` **기본 갈래**를 배열로
([[YR-327]] 조각 8 · key=month).

■ 한 줄 요약
    조각 6 의 조정자(`gpu/multiblock.py`)를 **그대로 재사용**하고, 30일 무대가 더하는 것만 여기서 얹는다.
    더하는 것은 셋뿐이다 — ① 루프 상한(배열판은 `Engine.steps_*` 라 상수 한 줄) ② **런 중 본선 붙이기**
    (`inject_vessel`) ③ **하루 경계 부기**(계수기 사진·날별 Φ·완료분 정리·끝난 배 치우기).

■ ★왜 조정자를 복사하지 않아도 되나 (v5 를 읽고 확인한 것)
    v5 `MonthTerminal(MultiBlockTerminal)` 은 `run()` 을 **재정의하지만 바뀐 줄은 `guard` 상한 하나**다
    (month_engine.py:75-111 머리말이 그렇게 못 박고, `tests/v6/test_month_stage.py:49` 가 사본과 대조해
    표류를 막는다). 배열판의 상한은 `lax.while_loop` 의 `max_steps`(`Engine.steps_per_epoch`·`steps_final`)
    이고 그것은 **무대 크기에서** 나온다(`multiblock.steps_for`) — 그래서 30일이라고 바꿀 것이 없다.
    ⇒ 여기서는 `multiblock.epoch_step_jit` / `finish_run_jit` 을 **부르기만** 한다 (재작성 없음).

■ 하루가 지나도 세계가 이어진다 — 그게 이 무대의 존재 이유다
    하루 무대는 아침마다 인공 초기 적재에서 출발한다. 30일 무대는 **첫날만** 그 적재를 쓰고, 이튿날은
    전날이 남긴 야드 위에서 시작한다. 배열판에서 이것은 저절로 성립한다 — 상태(`TerminalRun`)가 에폭마다
    이어지는 carry 이고 날 경계에서 리셋하는 코드가 없기 때문이다. 시험
    (`tests/v6/test_gpu_month.py::test_day_boundary_carries_yesterdays_yard`)이 그 지점을 v5 와 대조한다.

■ ★범위 — `run_month` 의 **기본 갈래(MonthTerminal)만** 옮긴다
    v5 `run_month` 은 두 갈래다 (month_run.py:273-281):
      · 기본 (`seed_data=None`)  → `MonthTerminal` + `V3Announcer(retarget=make_retarget(seed))`  ← **이번 범위**
      · 고정 화물 (`seed_data`)  → `stage/cargo_runtime.CargoTerminal` (전역 큐 4단 키 · affected 마스크 ·
        원격 본선 인계 · 끝의 재고 등식 · `cargo_moves.move_dependent`)  ← **범위 밖**
    `seed_data`(또는 그것이 낳는 `targets` 를 든 본선 행)를 주면 조용히 다른 답을 내지 않고 `MonthScopeError`
    로 **큰 소리로 거절**한다. 이유: 그 갈래는 블록 간 결합이 있어 조각 6 의 "에폭 사이 블록은 독립" 논증이
    성립하지 않는다 (multiblock.py 머리말 ■ 적용 범위).
    ⚠️ ★**그런데 연구선이 실제로 쓰는 갈래가 바로 그 범위 밖 갈래다** (2026-09-27 통합·수정 단계에서 확인):
      연구 드라이버 `ppo/workload_experiment.py:74-97` 은 `--seed-bundle` 이 없어도 `build_fixed_seed` 로
      문서를 만들어 **언제나** `run_month(seed_data=document)` 로 부른다 (최신 실행 yr331 manifest 의
      `arguments.seed_bundle`). 그래서 **배열 학습은 지금 연구선 설정으로는 돌 수 없다** — 기본 갈래는
      v5 자신의 컨테이너 계약 게이트가 막고(`unbound_vessel_load_streams`), 고정 화물 갈래는 이 파일이
      거절한다. 이식은 다음 조각의 일이다 (CargoTerminal 갈래 = 전역 큐 4단 키 · `V3Announcer.resolve_entry` ·
      원격 본선 인계 · 끝의 재고 등식 · `cargo_moves.move_dependent`).

■ 호스트에서 할 일과 배열 안에서 할 일 (경계를 분명히 둔다)
    호스트 (파이썬 · 난수 · 집합)                     배열 (jit · 순수)
    ─────────────────────────────────────────────────────────────────────────────
    `free_targets` 의 **메르센 셔플** (궤적 의존)      블록 전진 · 투입 · 적분 · 비용 (조각 1~6 그대로)
    양하 규격 난수 흐름 (`Random(f"…:size")`)          오더 행·배 칸 쓰기 (여기서 만든 값을 `_replace`)
    `retarget` 의 NO_TARGET 판정 (집합 연산)           사건 큐 push (v5 와 같은 순서 → 같은 seq)
    날 경계 부기 (계수기 사진첩 · 날별 Φ)              Φ 네 항 계산 (`gpu/phi`)
    끝난 배 archive 순서 (Neumaier 합의 순서)

■ ★`prune_completed` / `retire_done_vessels` 는 배열판에서 **무동작**이다 (증명, 시험이 지킨다)
    v5 의 두 함수는 **성능 대책**이고 세계를 바꾸지 않는다:
      · `prune_completed` 는 `sim.jobs`·원장·시간장부에서 **끝난 것**을 뺀다. 배열판의 오더는 고정 행이고
        순회 순서(=행 번호)가 끊기지 않으므로 빼도 상대 순서가 그대로다. 훑는 비용도 없다(벡터 연산).
      · `retire_done_vessels` 는 `v.done` 인 배를 치운다. 그 배는 `sts_blocked` 가 거짓이라
        `_refresh_rates` 의 `sts_wait` 계수에 안 들어가고(engine.py:816-817), 유휴 적립도 `not v.done`
        가드로 멈춘다(810-811행) — 즉 **치워도 안 치워도 같은 값**이다. 배열판은 `alive` 를 유지한 채
        `month_vessel_idle` 이 archive∪live 를 한 번에 센다.
    그래서 두 함수를 **감사 수치만** 돌려주는 `prune_audit`·`retire_audit` 으로 둔다 — v5
    `DayReport.pruned` / 치운 척수와 `==` 대조해 "무동작" 주장 자체를 시험이 검사한다.

■ 동등성 (규칙 1) — 이 파일의 실수 연산은 v5 호스트 코드와 **같은 파이썬 float 식**이다
    `pc = start + moves*cadence*deadline_mult` · `etd = start + moves*cadence*(mult+1)` ·
    `yc_empty_travel_s` 의 `tot += empty_m / v` (블록 dict 순 순차 덧셈) — v5 줄과 순서까지 같게 둔다.
    배열 안 연산은 손대지 않는다 (조각 1~6 이 이미 v5 결합 순서를 지킨다).
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import admission as AD
from . import host_terminal as HT
from . import multiblock as MB
from . import phi as PH
from . import vessel as VS
from .events import EMPTY_ID, EMPTY_TIME, TIME_DTYPE
from .geom import Geom
from .state import (EV_JOB_RELEASED, EV_VESSEL_START, FL_VESSEL_DISCHARGE, FL_VESSEL_LOAD,
                    JS_DONE, JS_PLANNED, empty_orders)
from .stack_ops import SIZE_INDEX

__all__ = [
    "MonthScopeError", "TransferError", "MonthDay", "MonthLayout", "VesselAdmission", "MonthTape",
    "MonthResult",
    "DAY_S", "EPOCH_S", "SNAP_S", "VESSEL_DEADLINE_MULT", "SIZE_MIX_FT40", "RETIRE_LAG_S",
    "month_days", "to_month_world", "month_engine", "make_month_run",
    "block_boxes", "free_targets", "inject_vessel", "open_day", "no_target_skips", "month_epoch",
    "gate_out_slots", "epoch_grid",
    "month_vessel_idle", "live_vessel_order", "yc_empty_travel_s", "rehandles_of",
    "prune_audit", "retire_audit",
    "day_orders", "day_phi", "close_day_at", "run_month",
]

F = TIME_DTYPE

#: v5 `stage/month.DAY_S`
DAY_S = 86_400.0
#: v5 `stage/orders.EPOCH_S` (= terminal_stream.WIP_ADMISSION_PERIOD_S) — 투입 검토 격자
EPOCH_S = 60.0
#: v5 `stage/month_run.SNAP_S` — 계수기 사진 간격 (날 경계는 **항상** 따로 찍는다)
SNAP_S = 900.0
#: v5 `stage/month_engine.VESSEL_DEADLINE_MULT`
VESSEL_DEADLINE_MULT = 2.0
#: v5 `stage/month_engine.SIZE_MIX_FT40`
SIZE_MIX_FT40 = 0.6
#: v5 `stage/month.RETIRE_LAG_S`
RETIRE_LAG_S = 3600.0


class MonthScopeError(NotImplementedError):
    """범위 밖 갈래를 부른 것 — 조용히 다른 답을 내지 않는다 (머리말 ■ 범위)."""


class TransferError(RuntimeError):
    """v5 `world/integrated/multiblock.TransferError` 의 자리 — 투입 거절 (fail-closed).

    ⚠️ v5 의 그 예외를 **상속하지 않는다** — `gpu/` 는 모듈 수준에서 v5 를 부르지 않는다는 규약 때문이다
    (호스트 전용 함수 안에서만 import 한다). v5 쪽 예외를 같이 잡아야 하는 호출부는 두 클래스를 함께
    적어야 한다. 사유 문자열은 v5 와 같게 만든다 (주입 원장 `why` 열이 그대로 대조된다).
    """


# ───────────────────────────────────────────────── 호스트 표 (번호 ↔ 이름 · 예약 배치)
@dataclass(frozen=True)
class MonthDay:
    """하루 한 줄 — v5 `stage/month.DayPlan` 이 들고 있는 값만 (배열판은 시드를 안 쓴다)."""

    index: int
    load: int
    label: str
    t0: float
    train: bool

    @property
    def t1(self) -> float:
        return self.t0 + DAY_S


@dataclass(frozen=True)
class MonthLayout:
    """월 전체 예약 배치 — **어느 본선 작업이 어느 오더 행·배 칸에 앉는가**.

    ★행 배치 규약 (host_terminal 머리말 ■ 오더 행 번호 규약을 30일로 넓힌 것)
      블록 b 의 행은 [트럭 n_trucks_b 개] → [월 전체 본선 작업 (사전식)] → [여분] 이다.
      트럭 이름은 `"Y01:D07-D-00035"`, 본선 작업은 `"Y01:J-D07-M1-s2-0000"` 이라 `"D" < "J"` →
      **트럭이 언제나 앞 번호**다. 본선 작업끼리는 `key` 사전식이고 `key = "D{일:02d}-{배}-s{스트림}"`
      이라 **날짜가 앞**이므로 "사전식 순서 = 붙는 날짜 순서" 가 된다 ⇒ 날마다 **이어지는 구간**을 쓴다.
      v5 는 런타임 `sorted(self.jobs)` 로 같은 순서를 얻는다. 아직 안 붙은 행은 `block = -1` 이라
      후보·순회에서 빠지고, 붙는 순간 같은 상대 순서로 끼어든다 (`name_rank = 행 번호`).
    """

    block_ids: tuple
    n_trucks: tuple
    #: (b, key) → 그 스트림이 쓸 오더 행 (moves 개 · 오름차순)
    rows: dict = field(default_factory=dict)
    #: (b, key) → 배 칸 v
    slot: dict = field(default_factory=dict)
    #: (b, key) → 양하 규격 SZ_* (moves 개) — `Random(f"{size_seed}:size")` 를 호스트에서 미리 소비
    sizes: dict = field(default_factory=dict)
    #: key → 본선 행 (plan_month_vessels 의 한 줄) · key → 날짜 · key → 블록 번호
    row_of_key: dict = field(default_factory=dict)
    day_of_key: dict = field(default_factory=dict)
    block_of_key: dict = field(default_factory=dict)
    #: 스트림 이름 → (배 이름, GT, STS) — v5 `stage/month.vessel_meta`
    meta: dict = field(default_factory=dict)
    #: s → 그 트럭이 게이트를 들어오는 날 (Φ 항1 의 귀속 — `_day_records` 의 `D{일:02d}-` 접두)
    truck_day: tuple = ()
    #: s → 명단 한 줄 (NO_TARGET 판정에 `target` 이름이 필요하다)
    sched: tuple = ()
    days: tuple = ()
    month_s: float = 0.0
    sim_end_s: float = 0.0
    v_by_day: dict = field(default_factory=dict)

    @property
    def b(self) -> int:
        return len(self.block_ids)


@dataclass(frozen=True)
class VesselAdmission:
    """투입 결과 한 줄 — v5 `stage/month_engine.VesselAdmission` 과 같은 열."""

    vessel_key: str
    block: str
    work: str
    asked_moves: int
    moves: int
    start_s: float
    planned_completion_s: float
    reason: str = ""

    @property
    def clipped(self) -> bool:
        return self.moves < self.asked_moves

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def month_days(days) -> tuple:
    """v5 `DayPlan` 목록 → `MonthDay` 목록 (필요한 열만 옮긴다)."""
    return tuple(MonthDay(index=int(d.index), load=int(d.load), label=str(d.label),
                          t0=float(d.t0), train=bool(d.is_train)) for d in days)


def _vessel_job_id(bid: str, key: str, m: int) -> str:
    """v5 `inject_vessel` 232행 — 블록 접두 `_namespace_jobs` 와 같은 꼴."""
    return f"{bid}:J-{key}-{m:04d}"


# ───────────────────────────────────────────────── 무대 세우기
def to_month_world(profile, built: dict, v_by_day: dict, *, days, seed: int, layout=None,
                   lead_s=None, n_spare: int = 0, period_s: float = EPOCH_S,
                   drain_s: float | None = None, n_max: int | None = None,
                   q_cap: int | None = None, log_cap: int | None = None,
                   max_transfers: int = 1, seed_data=None):
    """v5 `run_month` 의 무대 조립(month_run.py:242-283) 을 배열로 — `(tw, tt, MonthLayout)`.

    built     `stage/month.build_month(seed, days=…)` 결과 (`day0`·30일 `schedule`)
    v_by_day  `stage/month.plan_month_vessels(...)` 결과 — **날마다 붙일 본선 스트림 행**
    days      `DayPlan` 목록 (`stage/month.plan_month`)
    seed      `run_month(seed=…)` — 양하 규격·적하 대상 난수의 이름에 들어간다
              (v5 447행 `size_seed=f"v3:month:{seed}:{key}"`)
    lead_s    통지 리드 — 기본은 명단의 `lead_s` 열 (V3Announcer 116행과 같은 뜻)

    ■ v5 와 같게 두는 것
      · 배경은 **첫날 것만** (`built["day0"]["scenarios"]`), 본선·야드작업은 **넣지 않는다** (268-273행)
      · `horizon_s = n_days·DAY_S` · `drain_window_s = DIURNAL_DRAIN_S`
      · 검토 시각 = `extra_review_epochs = (0, 60, …, month_s)` (275행) — 시나리오 jobs 가 비어 있으므로
        `_schedule_review_epochs` 의 도착 집합은 공집합이고 격자만 남는다. 날 경계 `i·86400` 은
        60 의 배수라 **반드시 검토 시각**이다 (날 경계 훅이 걸릴 자리).

    ■ 배열판이 더 하는 것 — **월 전체 본선의 자리를 미리 잡는다**
      오더 칸 N 에 월 전체 본선 작업 행을, 배 칸 V 에 월 전체 스트림을 예약한다 (`MonthLayout`).
      예약 행은 `block = -1` 이라 붙기 전에는 v5 `sim.jobs` 에 없는 것과 같다.
    """
    if seed_data is not None:
        raise MonthScopeError(
            "seed_data(고정 화물)는 CargoTerminal 갈래다 — 전역 큐 4단 키·원격 본선 인계·끝의 재고 등식이 "
            "있어 조각 6 의 '에폭 사이 블록 독립' 논증이 성립하지 않는다. 이번 범위 밖 (머리말 ■ 범위)")
    from ..world.integrated.terminal_stream import DIURNAL_DRAIN_S

    days = list(days)
    n_days = len(days)
    month_s = n_days * DAY_S
    drain = float(DIURNAL_DRAIN_S if drain_s is None else drain_s)
    sim_end = month_s + drain
    scns = {b: dataclasses.replace(s, jobs=[], vessels=[], horizon_s=month_s, drain_window_s=drain)
            for b, s in built["day0"]["scenarios"].items()}
    block_ids = tuple(scns)
    bidx = {b: i for i, b in enumerate(block_ids)}
    schedule = built["schedule"]

    # ── 월 전체 본선을 블록별로 모은다 (행·칸 예약) ──
    row_of_key: dict = {}
    day_of_key: dict = {}
    block_of_key: dict = {}
    meta: dict = {}
    keys_of: dict = {b: [] for b in range(len(block_ids))}
    for d in days:
        for r in v_by_day.get(int(d.index), []):
            if "targets" in r:
                raise MonthScopeError(
                    f"{r['key']}: 본선 행에 `targets` 가 있다 — 고정 명단(CargoTerminal) 갈래이고 이번 범위 밖")
            b = bidx[r["block"]]
            keys_of[b].append(r["key"])
            row_of_key[r["key"]] = r
            day_of_key[r["key"]] = int(d.index)
            block_of_key[r["key"]] = b
            meta[r["key"]] = (r["ship"], float(r["gt"]), int(r["sts"]))

    n_trucks = tuple(sum(1 for e in schedule if bidx[e["block"]] == b) for b in range(len(block_ids)))
    rows: dict = {}
    slot: dict = {}
    sizes: dict = {}
    vessel_ids_of: dict = {}
    vjob_ids_of: dict = {}
    for b in range(len(block_ids)):
        bid = block_ids[b]
        keys = sorted(keys_of[b])                                  # 사전식 = 붙는 날짜 순 (머리말 ★행 배치)
        vessel_ids_of[b] = tuple(keys)
        n = n_trucks[b]
        jids: list = []
        for i, key in enumerate(keys):
            r = row_of_key[key]
            asked = int(r["moves"])
            rows[(b, key)] = tuple(range(n, n + asked))            # 사전식 = m 오름차순
            slot[(b, key)] = i
            jids += [_vessel_job_id(bid, key, m) for m in range(asked)]
            n += asked
            if r["work"] == "DISCHARGE":                           # v5 230-240행 규격 난수 (적하는 안 뽑는다)
                rng = random.Random(f"{_size_seed(seed, key)}:size")
                sizes[(b, key)] = tuple(SIZE_INDEX["FT40"] if rng.random() < SIZE_MIX_FT40
                                        else SIZE_INDEX["FT20"] for _ in range(asked))
        vjob_ids_of[b] = tuple(jids)
    need = max(n_trucks[b] + len(vjob_ids_of[b]) for b in range(len(block_ids))) + int(n_spare)
    N = need if n_max is None else int(n_max)
    if N < need:
        raise ValueError(f"n_max {N} < 필요한 오더 칸 {need} (트럭 + 월 전체 본선작업 + 여분 {n_spare})")

    extra = tuple(i * period_s for i in range(int(month_s // period_s) + 1))
    lead = (np.asarray([float(e["lead_s"]) for e in schedule], np.float64) if lead_s is None else lead_s)
    if layout is None:                       # `build_month` 는 layout 을 안 싣는다 — 첫날 무대에서 꺼낸다
        from ..world.integrated.yard_layout import YardLayout
        ld = built["day0"]["layout"]
        layout = YardLayout(tuple(ld["ids"]), tuple(float(p) for p in ld["positions_m"]),
                            float(ld["speed_mps"]))
    tw, tt = HT.to_terminal_world(
        profile, {**built, "scenarios": scns, "sim_end_s": sim_end},
        lead_s=lead, n_max=N, q_cap=q_cap, log_cap=log_cap, extra_review_epochs=extra,
        end_s=sim_end, observe_s=month_s, layout=layout,
        period_s=period_s, n_spare=int(n_spare), max_transfers=int(max_transfers))

    # ── 번호표를 월 전체 배치로 갈아 끼운다 (name_rank·컨테이너 이름이 여기서 나온다) ──
    tables = []
    for b in range(len(block_ids)):
        tb = tt.tables[b]
        jids = tuple(tb.job_ids) + vjob_ids_of[b]
        cont = list(tb.cont_ids)
        for n, j in enumerate(jids):
            cont[tb.c0 + n] = f"IN_{j}"
        tables.append(dataclasses.replace(tb, job_ids=jids, cont_ids=tuple(cont), n0=len(jids),
                                          vessel_ids=vessel_ids_of[b]))
    V = max(1, max(len(vessel_ids_of[b]) for b in range(len(block_ids))))
    P = max(8, max(sum(int(row_of_key[k]["moves"]) for k in vessel_ids_of[b])
                   for b in range(len(block_ids))))
    tt = dataclasses.replace(tt, tables=tuple(tables), v_max=V, p_cap=P)

    # ── 배 칸 V · 이송 링버퍼 P 를 월 규모로 (변환 직후엔 배가 하나도 없으니 빈 배열로 갈아 끼운다) ──
    U = max(1, int(profile.transfer.n_units))
    move_t = float(profile.transfer.move_time_s)
    ves = [VS.empty_vessels(V, N)._replace(
        release_rank=VS.release_rank_from_ids(list(tables[b].job_ids), N)) for b in range(len(block_ids))]
    tr = [VS.empty_transfer(U, P, move_t) for _ in block_ids]
    stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)
    tw = tw._replace(blocks=tw.blocks._replace(vessels=stack(ves), transfer=stack(tr)))

    lay = MonthLayout(
        block_ids=block_ids, n_trucks=n_trucks, rows=rows, slot=slot, sizes=sizes,
        row_of_key=row_of_key, day_of_key=day_of_key, block_of_key=block_of_key, meta=meta,
        truck_day=tuple(int(e.get("day", 0)) for e in schedule), sched=tuple(schedule),
        days=month_days(days), month_s=month_s, sim_end_s=sim_end,
        v_by_day={int(k): tuple(v) for k, v in v_by_day.items()})
    return tw, tt, lay


def _size_seed(seed: int, key: str) -> str:
    """v5 `run_month` 447행 `size_seed=f"v3:month:{seed}:{r['key']}"`."""
    return f"v3:month:{seed}:{key}"


def month_engine(tt: HT.TerminalTables, g: Geom, *, policy_fn, check: bool = True,
                 pre_advice: bool = True, horizon_s: float = 0.0, joint: bool = True,
                 k0: int = 0, margin: int | None = None) -> MB.Engine:
    """조정자 설정 — **상한은 무대 크기에서** (`multiblock.steps_for`). v5 `MonthTerminal.LOOP_GUARD`
    (90,000,000) 의 자리이고, 배열판은 에폭 하나 안의 스텝만 재므로 30일이라고 커지지 않는다."""
    steps = MB.steps_for(tt)
    return MB.Engine(g=g, policy_fn=policy_fn, check=check, pre_advice=pre_advice,
                     horizon_s=float(horizon_s), joint=joint,
                     margin=AD.CAPACITY_MARGIN if margin is None else int(margin),
                     end_ann=float(tt.end_s), steps_per_epoch=steps, steps_final=steps, k0=int(k0))


def make_month_run(tw: HT.TerminalWorld, tt: HT.TerminalTables, g: Geom, *, params=None) -> MB.TerminalRun:
    """`multiblock.make_run` 그대로 — 순위표는 월 전체 배치로 구운 `tt.tables` 에서 나온다."""
    if params is None:
        params = MB.terminal_resolver_params(tt, g)
    return MB.make_run(tw, tt, params=params)


# ───────────────────────────────────────────────── 런 중 ① 적하 대상 (호스트 · 궤적 의존)
def block_boxes(run: MB.TerminalRun, tt: HT.TerminalTables, b: int) -> list:
    """블록 b 의 **지금 야드에 있는 상자 이름** — v5 `sim.stacks.containers` 의 열쇠 집합.

    `conts.c_alive` 가 정본이다 (반입 예비칸·반출된 것은 False). 이름은 월 전체 배치로 구운
    `tt.tables[b].cont_ids` 에서 — 양하가 낳는 상자도 `IN_{본선작업 id}` 로 v5 와 같은 이름이다.
    """
    alive = np.asarray(run.tw.blocks.conts.c_alive[b])
    names = tt.tables[b].cont_ids
    return [names[c] for c in np.nonzero(alive)[0]]


def _taken_targets(run: MB.TerminalRun, tt: HT.TerminalTables, b: int) -> set:
    """v5 `{j.target_container for j in sim.jobs.values() if …}` — 이미 누가 찍은 상자.

    배열판은 오더 행이 남아 있으므로 `block >= 0` (= 붙어 있는 오더) 인 행의 `target_cont` 를 센다.
    v5 는 `prune_completed` 로 끝난 job 을 치우지만, 끝난 반출의 대상 상자는 **야드에서 사라져**
    `c_alive` 가 거짓이라 어느 쪽으로 세도 결과가 같다 (머리말 ■ prune 무동작).
    """
    o = run.tw.blocks.orders
    blk = np.asarray(o.block[b])
    tgt = np.asarray(o.target_cont[b])
    names = tt.tables[b].cont_ids
    return {names[t] for t in tgt[(blk >= 0) & (tgt >= 0)]}


def free_targets(run: MB.TerminalRun, tt: HT.TerminalTables, b: int, *, limit: int, seed: str) -> list:
    """v5 `stage/month_engine.free_targets` 그대로 — **지금 야드에 있고 아무도 안 찍은 상자**를
    시드 고정 메르센 셔플로 고른다.

    ★궤적 의존이라 호스트 몫이다. 같은 무대라도 어제 정책이 어디를 비웠는지에 따라 집합이 달라지고,
    `random.Random(seed).shuffle` 은 **집합의 정렬 순서**에 걸려 있다 (같은 집합·같은 시드 → 같은 목록).
    """
    taken = _taken_targets(run, tt, b)
    cand = [c for c in sorted(block_boxes(run, tt, b)) if c not in taken]
    random.Random(seed).shuffle(cand)
    return cand[:limit]


# ───────────────────────────────────────────────── 런 중 ② 본선 붙이기
def inject_vessel(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, *, key: str,
                  seed: int, profile, deadline_mult: float = VESSEL_DEADLINE_MULT):
    """v5 `stage/month_engine.inject_vessel` 의 배열판 — 배 한 척(STS 스트림 하나)을 **런 중에** 붙인다.

    검사에 걸리면 `TransferError` 를 던지고 **아무것도 안 바꾼다** (v5 와 같은 fail-closed 계약).
    통과하면 (run', VesselAdmission).

    ■ v5 와 같은 것
      · 양하(DISCHARGE) — 야드 재고를 안 쓴다. job 해제는 시각이 아니라 **박스의 물리 도착**이라
        `JOB_RELEASED` 를 안 건다 (250-251행 `is_vessel_linked and STORE` 조건).
      · 적하(LOAD) — `free_targets` 로 고른 상자를 쓰고, 모자라면 물량을 깎고 사유를 남긴다.
      · 계획 시각 `pc = start + moves·cadence·mult` · `etd = start + moves·cadence·(mult+1)` ·
        `phys_min_completion_s` 가 더 늦으면 그것으로 밀고 `etd = pc + moves·cadence`.
      · 사건 push 순서 = VESSEL_START → (적하면) 해제 moves 건 — 같은 시각 타이브레이크(seq)가 같아진다.
    """
    from ..world.domain.models import Container
    from ..world.integrated.scenario_gen import phys_min_completion_s
    from ..world.integrated.vessel import VesselWorkType

    r = lay.row_of_key.get(key)
    if r is None:
        raise TransferError(f"{key}: 계획에 없는 스트림")
    b = lay.block_of_key[key]
    bid = lay.block_ids[b]
    W = run.tw.blocks
    if bool(np.asarray(W.vessels.alive[b, lay.slot[(b, key)]])):
        raise TransferError(f"{key}: 이미 붙어 있는 배")
    start = float(r["start_s"])
    clock = float(np.asarray(W.clock[b]))
    end = float(np.asarray(W.end_s[b]))
    if start < clock - 1e-9:
        raise TransferError(f"{key}: 과거에 붙일 수 없다 start={start:.1f} clock={clock:.1f}")
    if start > end:
        raise TransferError(f"{key}: 창 밖 start={start:.1f} end={end:.1f}")

    discharge = r["work"] == "DISCHARGE"
    work = VesselWorkType.DISCHARGE if discharge else VesselWorkType.LOAD
    cadence = float(r["cadence_s"])
    asked = int(r["moves"])
    targets: list = []
    reason = ""
    if not discharge:
        targets = free_targets(run, tt, b, limit=asked, seed=f"{_size_seed(seed, key)}:tgt")
        if len(targets) < asked:
            reason = f"재고 부족 {len(targets)}/{asked}"
    moves = asked if discharge else len(targets)
    if moves <= 0:
        raise TransferError(f"{key}: 실을 물량이 0 — {reason or '물량 0'}")

    pc = start + moves * cadence * deadline_mult
    etd = start + moves * cadence * (deadline_mult + 1.0)
    cont_idx = tt.tables[b].cont_index
    tgt_c = None
    if not discharge:
        cb = np.asarray(W.conts.c_bay[b]); cr = np.asarray(W.conts.c_row[b])
        ct = np.asarray(W.conts.c_tier[b]); cs = np.asarray(W.conts.c_size[b])
        from ..world.domain.enums import ContainerSize, LoadStatus
        sz = ("FT20", "FT40", "FT45")
        tgt_c = [Container(container_id=c, size=ContainerSize(sz[int(cs[cont_idx[c]])]),
                           load_status=LoadStatus.FULL, block=bid, bay=int(cb[cont_idx[c]]),
                           row=int(cr[cont_idx[c]]), tier=int(ct[cont_idx[c]])) for c in targets]
    phys_min = phys_min_completion_s(profile, work=work, start_s=start, moves=moves,
                                     cadence_s=cadence, load_targets=tgt_c)
    if phys_min > pc:
        pc = phys_min
        etd = pc + moves * cadence

    # ── 여기부터 실패하지 않는 연산만 (원자성 — v5 와 같은 계약) ──
    v = lay.slot[(b, key)]
    ns = lay.rows[(b, key)][:moves]
    ves = {f: np.asarray(getattr(W.vessels, f)).copy() for f in W.vessels._fields}
    ves["is_discharge"][b, v] = discharge
    ves["total_moves"][b, v] = moves
    ves["cadence_s"][b, v] = cadence
    ves["buffer_cap"][b, v] = 3                     # VesselPlan.quay_buffer_cap 기본값
    ves["planned_start_s"][b, v] = start
    ves["planned_completion_s"][b, v] = pc
    ves["etd_s"][b, v] = etd
    ves["started"][b, v] = False
    ves["remaining"][b, v] = -1
    ves["buffer"][b, v] = 0
    ves["blocked_since_s"][b, v] = EMPTY_TIME
    ves["wait_accum_s"][b, v] = 0.0
    ves["done"][b, v] = False
    ves["actual_completion_s"][b, v] = EMPTY_TIME
    ves["alive"][b, v] = True

    o = {f: np.asarray(getattr(W.orders, f)).copy() for f in W.orders._fields}
    c_size = np.asarray(W.conts.c_size).copy()
    c0 = tt.tables[b].c0
    flow = FL_VESSEL_DISCHARGE if discharge else FL_VESSEL_LOAD
    szs = lay.sizes.get((b, key), ())
    for m, n in enumerate(ns):
        o["block"][b, n] = b
        o["flow"][b, n] = flow
        o["status"][b, n] = JS_PLANNED
        o["is_external"][b, n] = False
        o["is_vessel"][b, n] = True
        o["is_store"][b, n] = discharge
        o["vessel"][b, n] = v
        o["release_s"][b, n] = start + m * cadence
        o["deadline_s"][b, n] = pc + 1800.0
        o["exit_travel_s"][b, n] = -1.0
        if discharge:
            o["inbound_cont"][b, n] = c0 + n
            o["inbound_size"][b, n] = szs[m]
            c_size[b, c0 + n] = szs[m]
        else:
            o["target_cont"][b, n] = cont_idx[targets[m]]

    # ── 사건 큐 (v5 227행 VESSEL_START · 250-251행 해제) ──
    q = {f: np.asarray(getattr(W.queue, f)).copy() for f in W.queue._fields}
    pushes = [(start, EV_VESSEL_START, v)]
    if not discharge:                                   # 양하는 **박스 물리 도착**이 해제한다
        pushes += [(start + m * cadence, EV_JOB_RELEASED, int(n)) for m, n in enumerate(ns)]
    for t, kind, tgt in pushes:
        free = np.nonzero(q["time"][b] == np.inf)[0]
        if free.size == 0:
            q["overflow"][b] += 1
            continue
        i = int(free[0])
        q["time"][b, i] = t
        q["kind"][b, i] = kind
        q["target"][b, i] = tgt
        q["seq"][b, i] = q["counter"][b]
        q["counter"][b] += 1

    W2 = W._replace(
        vessels=W.vessels._replace(**{f: jnp.asarray(a) for f, a in ves.items()}),
        orders=W.orders._replace(**{f: jnp.asarray(a) for f, a in o.items()}),
        conts=W.conts._replace(c_size=jnp.asarray(c_size)),
        queue=W.queue._replace(**{f: jnp.asarray(a) for f, a in q.items()}))
    run = run._replace(tw=run.tw._replace(blocks=W2))
    return run, VesselAdmission(vessel_key=key, block=bid, work=work.value, asked_moves=asked,
                                moves=moves, start_s=start, planned_completion_s=pc, reason=reason)


def open_day(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, day: int, *,
             seed: int, profile, deadline_mult: float = VESSEL_DEADLINE_MULT, t0: float | None = None):
    """v5 `run_month.open_day` (446-475행) — 그날 배를 **계획 순서대로** 붙인다.

    못 붙인 배는 조용히 넘기지 않고 `ok=False` 줄로 남긴다 (v5 와 같은 원장 열).
    반환 `(run, rows, (n, moves, skipped))`.
    """
    d = lay.days[day]
    rows: list = []
    n = moves = skipped = 0
    for r in lay.v_by_day.get(int(day), ()):            # v5 는 v_by_day 목록 순서 그대로 (블록 정렬 순)
        try:
            run, a = inject_vessel(run, tt, lay, key=r["key"], seed=seed,
                                   deadline_mult=deadline_mult, profile=profile)
        except TransferError as ex:
            skipped += 1
            rows.append({"key": r["key"], "day": int(day), "block": r["block"],
                         "time_s": float(d.t0 if t0 is None else t0), "asked": int(r["moves"]),
                         "moves": 0, "ok": False, "why": str(ex)})
            continue
        n += 1
        moves += a.moves
        rows.append({"key": a.vessel_key, "day": int(day), "block": r["block"],
                     "time_s": float(d.t0 if t0 is None else t0), "ok": True, "moves": a.moves,
                     "asked": a.asked_moves, "why": a.reason})
    return run, rows, (n, moves, skipped)


# ───────────────────────────────────────────────── 런 중 ③ 투입 (retarget 의 NO_TARGET)
def no_target_skips(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, e: int, t: float):
    """v5 `V3Announcer.review` 164-178행 + `stage/month.make_retarget` — **고정 대상 확인**.

    `make_retarget` 은 대상을 새로 고르지 않고 *"그 상자가 지금 야드에 있고 아무도 안 찍었나"* 만 본다.
    아니면 `NO_TARGET` 으로 **투입 시도 자체를 건너뛴다**. 30일이면 날마다 같은 이름이 다시 나오므로
    (YR-239 · YR-306) 이 판정이 실제로 트럭을 떨어뜨린다.

    배열판의 `admission.admit_block` 은 `SKIP_TARGET`(대상 부재)만 보고 "이미 찍혔나" 는 안 본다 —
    그 한 조건이 여기 호스트 몫이다. 반환 `(blocked (S,) bool, skips 목록)`.

    ■ ★한 에폭 **안에서도** 찍힌 상자가 늘어난다 (v5 는 트럭마다 `sim.jobs` 를 새로 읽는다)
      같은 에폭·같은 블록에 **같은 대상 이름**의 반출 트럭이 둘 오면, v5 는 앞 트럭을 넣고 나서
      뒤 트럭의 `taken` 에 그 상자를 넣어 `NO_TARGET` 을 낸다. 30일이면 반출 이름이 날마다 겹치므로
      (YR-239) 실제로 생길 수 있다. 그래서 여기서도 명단 순서대로 돌며 `taken` 을 **키워 간다**.
      앞 트럭이 실제로 들어갈지는 `admit_block` 의 나머지 검사에 달렸는데, 대상 확인을 통과한 반출
      트럭이 떨어지는 사유는 **시각(SKIP_TIME)** 하나뿐이다 (DUP 은 위에서 걸렀고, EXIT 는 변환 때
      막고, CAP 은 반입 전용, TARGET 은 방금 본 조건이다) — 그 한 검사만 같은 식으로 미리 본다.
    """
    S = int(run.tw.sched.s)
    blocked = np.zeros((S,), bool)
    skips: list = []
    slot = int(np.asarray(MB._grid_slot(run, e)))
    Eg = int(run.adm.bucket.shape[0])
    if slot < 0 or slot >= Eg:
        return blocked, skips
    bucket = np.asarray(run.adm.bucket[slot])                     # (B,M) 명단 색인
    reg = np.asarray(run.tw.ledger.registered)
    clock = np.asarray(run.tw.blocks.clock)
    end_b = np.asarray(run.tw.blocks.end_s)
    eps = float(AD.EPS)
    for b in range(lay.b):
        idx = [int(s) for s in bucket[b] if int(s) >= 0]
        if not idx:
            continue
        boxes = set(block_boxes(run, tt, b))
        taken = _taken_targets(run, tt, b)
        for s in idx:                                             # 명단 순 (= by_epoch 리스트 순)
            e_row = lay.sched[s]
            if e_row["flow"] != "GATE_OUT" or reg[s]:
                continue
            arr = float(e_row["arrival_s"])
            blk_in = arr + float(e_row["travel_s"])
            if blk_in > lay.sim_end_s:                            # 160-162행 TAIL 이 먼저다
                continue
            want = e_row.get("target")
            if want is None or want not in boxes or want in taken:
                blocked[s] = True
                skips.append({"t": float(t), "job_id": e_row["job_id"], "reason": "NO_TARGET"})
                continue
            # 통과했다 — 시각 검사까지 넘으면 v5 는 이 상자를 **찍는다** (다음 트럭이 못 쓴다)
            time_bad = (arr < float(clock[b]) - eps or blk_in <= float(clock[b]) + eps
                        or blk_in > float(end_b[b]))
            if not time_bad:
                taken.add(want)
    return blocked, skips


def _advance_to_epoch(run: MB.TerminalRun, eng: MB.Engine):
    """전 블록을 다음 검토 시각까지 (`multiblock.epoch_step` 의 앞 절반). **jit 한 벌**로 재사용된다."""
    from .state import V_STEPS_EXHAUSTED
    W, parked, _, stuck = MB.run_to_epoch(run.tw.blocks, run.params, eng, max_steps=eng.steps_per_epoch)
    W = W._replace(violation=W.violation | jnp.where(stuck, V_STEPS_EXHAUSTED, 0).astype(jnp.int32))
    return run._replace(tw=run.tw._replace(blocks=W), exhausted=run.exhausted + stuck.astype(jnp.int32))


def _review_with_veto(run: MB.TerminalRun, e, blocked, eng: MB.Engine):
    """`multiblock.review_epoch` + **NO_TARGET 거부권**.

    v5 는 대상이 없는 트럭에서 `continue` 하므로, 배열판은 그 트럭을 **등록된 것처럼 보이게** 해서
    `admit_block` 의 중복 검사에 걸리게 한다 (코드는 `SKIP_DUP` 으로 남지만 세계 변화는 v5 와 같은
    '아무것도 안 함' 이다). 그 뒤 켠 비트를 되돌린다 — 그 트럭은 **투입되지 않았다**.
    """
    L = run.tw.ledger
    run2, codes = MB.review_epoch(
        run._replace(tw=run.tw._replace(ledger=L._replace(registered=L.registered | blocked))), e, eng)
    L2 = run2.tw.ledger
    return run2._replace(tw=run2.tw._replace(ledger=L2._replace(registered=L2.registered & ~blocked))), codes


_advance_jit = jax.jit(_advance_to_epoch, static_argnames=("eng",))
_review_jit = jax.jit(_review_with_veto, static_argnames=("eng",))


def gate_out_slots(run: MB.TerminalRun, lay: MonthLayout) -> set:
    """반출 트럭이 통지되는 **격자 슬롯** — 그 에폭에서만 호스트가 NO_TARGET 을 따진다 (나머지는 건너뛴다).

    명단은 정적이라 한 번 계산해 두면 된다. 이게 없으면 에폭마다 (B,C) 배열을 호스트로 내려 30일이
    43,201번 장치 왕복을 한다.
    """
    bucket = np.asarray(run.adm.bucket)                       # (E_grid, B, M)
    out: set = set()
    for slot in range(bucket.shape[0]):
        idx = bucket[slot][bucket[slot] >= 0]
        if any(lay.sched[int(s)]["flow"] == "GATE_OUT" for s in idx):
            out.add(slot)
    return out


def epoch_grid(run: MB.TerminalRun) -> tuple:
    """검토 시각 목록과 격자 슬롯을 **호스트 numpy 로 한 번에** — 에폭마다 장치 왕복을 없앤다.
    (`multiblock.epoch_time`·`_grid_slot` 과 같은 식: on_grid 면 round(t/period), 아니면 -1.)"""
    t = np.asarray(run.tw.epochs.t)
    on = np.asarray(run.tw.epochs.on_grid)
    period = float(np.asarray(run.tw.epochs.period_s))
    slot = np.where(on, np.rint(t / period).astype(np.int32), EMPTY_ID)
    return t, slot


def month_epoch(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, e: int, eng: MB.Engine,
                *, slots: set | None = None, grid: tuple | None = None):
    """검토 시각 하나 — 전 블록 전진 → `_sync_locks` → 투입(NO_TARGET 가려낸 뒤) → 원장 등록.

    반환 `(run, codes (B,M), skips)`. `slots` 를 주면 그 격자 슬롯에서만 NO_TARGET 을 따진다
    (반출 트럭이 없는 에폭에서는 호스트가 아무것도 내려받지 않는다).
    """
    run = _advance_jit(run, eng)
    ts, sl = epoch_grid(run) if grid is None else grid
    t = float(ts[int(e)])
    slot = int(sl[int(e)])
    S = int(run.tw.sched.s)
    if slots is not None and slot not in slots:
        blocked, skips = np.zeros((S,), bool), []
    else:
        blocked, skips = no_target_skips(run, tt, lay, e, t)
    run, codes = _review_jit(run, jnp.asarray(e, jnp.int32), jnp.asarray(blocked), eng)
    return run, codes, skips


# ───────────────────────────────────────────────── 계수기 사진첩 · 날별 Φ
def yc_empty_travel_s(run: MB.TerminalRun, profile) -> float:
    """v5 `stage/episode.yc_empty_travel_s` — Σ_b (빈 갠트리 m / 갠트리 속도). **블록 dict 순 순차 덧셈**."""
    v = float(profile.cranes[0].gantry_speed_mps) if profile.cranes else 1.0
    em = np.asarray(run.tw.blocks.kpi.empty_m)
    tot = 0.0
    for b in range(em.shape[0]):
        tot += float(em[b]) / max(1e-9, v)
    return tot


def rehandles_of(run: MB.TerminalRun) -> int:
    """v5 `stage/episode.rehandles_of` — 터미널 전체 재조작 수 (정수 합)."""
    return int(sum(int(x) for x in np.asarray(run.tw.blocks.kpi.rehandles)))


def month_vessel_idle(run: MB.TerminalRun, lay: MonthLayout, order=None, archive=()) -> dict:
    """v5 `stage/month.month_vessel_idle` — `{배: (GT, 유휴 초)}`. **배 단위**로 묶는다.

    ★배열판은 끝난 배도 칸에 남겨 두므로(머리말 ■ retire 무동작) archive 와 live 를 가릴 필요가 없다.
    다만 `sum(w)/sts` 의 **덧셈 순서**가 v5 와 같아야 마지막 비트가 맞는다 — 파이썬 3.12 `sum()` 은
    실수 목록을 Neumaier 보정합으로 더하므로 **세 항부터** 순서가 마지막 비트를 바꾼다 (배 한 척에
    STS 가 3대 이상이면, 즉 스트림이 셋 이상 모이면 실제로 갈린다).

    v5 순서 = `archive`(치운 순) → **블록 dict 순** × `sim.vessels` 삽입(=붙은) 순이다.
    그래서 `archive` (치운 스트림 이름, 치운 순) 를 주면 그것을 앞에 두고, 나머지는
    `_live_order` (블록 순 × 붙은 순)로 센다. `order` 를 직접 주면 그 목록을 그대로 쓴다.
    """
    wait = np.asarray(run.tw.blocks.vessels.wait_accum_s)
    alive = np.asarray(run.tw.blocks.vessels.alive)
    if order is None:
        seen = set(archive)
        order = list(archive) + [k for k in live_vessel_order(lay) if k not in seen]
    keys = order
    acc: dict = {}
    gt_of: dict = {}
    sts_of: dict = {}
    for key in keys:
        m = lay.meta.get(key)
        if m is None:
            continue
        b = lay.block_of_key[key]
        v = lay.slot[(b, key)]
        if not bool(alive[b, v]):
            continue
        ship, gt, sts = m
        acc.setdefault(ship, []).append(float(wait[b, v]))
        gt_of[ship], sts_of[ship] = gt, sts
    return {k: (gt_of[k], sum(w) / max(1, sts_of[k])) for k, w in acc.items()}


def _inject_order(lay: MonthLayout) -> tuple:
    """붙는 순서 — 날마다 `v_by_day` 목록 순 (v5 `open_day` 가 그 순서로 `sim.vessels` 에 넣는다)."""
    out: list = []
    for d in lay.days:
        for r in lay.v_by_day.get(int(d.index), ()):
            out.append(r["key"])
    return tuple(out)


def live_vessel_order(lay: MonthLayout) -> tuple:
    """v5 `for sim in mbt.blocks.values(): for key in sim.vessels` — **블록 순 × 붙은 순**."""
    inj = _inject_order(lay)
    return tuple(k for b in range(lay.b) for k in inj if lay.block_of_key[k] == b)


class MonthTape:
    """v5 `stage/month_run._MonthTape` — 날 경계의 **계수기 사진첩**. 하루치 항2·3·4 는 차분에서 나온다."""

    def __init__(self, lay: MonthLayout, profile, archive=None):
        self.lay, self.profile = lay, profile
        self.at: dict = {}
        #: 치운 스트림 이름 (치운 순) — v5 `archive` dict 의 순서. `run_month` 가 채운다.
        self.archive: list = [] if archive is None else archive

    def snap(self, run: MB.TerminalRun, t: float) -> None:
        self.at[round(t, 6)] = (month_vessel_idle(run, self.lay, archive=tuple(self.archive)),
                                yc_empty_travel_s(run, self.profile), rehandles_of(run))

    def read(self, t: float):
        key = round(t, 6)
        if key in self.at:
            return self.at[key]
        keys = [k for k in self.at if k <= key]
        return self.at[max(keys)] if keys else ({}, 0.0, 0)

    def diff(self, t0: float, t1: float):
        """`t0 → t1` 사이에 **늘어난 만큼** — v5 `_MonthTape.diff` 와 같은 식·같은 순서."""
        v0, yc0, rh0 = self.read(t0)
        v1, yc1, rh1 = self.read(t1)
        v = {}
        for k, (gt, idle) in v1.items():
            v[k] = (gt, max(0.0, idle - (v0.get(k, (gt, 0.0))[1])))
        return v, max(0.0, yc1 - yc0), max(0, rh1 - rh0)


def day_orders(run: MB.TerminalRun, lay: MonthLayout, day: int, *, sync_t: float):
    """날 `day` 의 **트럭 기록**을 Φ 가 읽는 평평한 오더 배열로 — `gpu/phi.orders_from_records` 와 같은 모양.

    v5 `records` 는 명단 순서(= 도착 정렬 순)로 만들어지고 `bridge._sync` 가 **`값 ≤ t` 인 단계만** 찍는다
    (bridge.py:109-118). 그래서 A(게이트 진입)·O(게이트 아웃)도 `sync_t` 이하만 채운다 — 그러지 않으면
    잠정 Φ 가 미래를 읽는다. 반환 `(OrderArrays (S,), truck_mask (S,) bool)`.
    """
    S = int(run.tw.sched.s)
    reg = np.asarray(run.tw.ledger.registered)
    a_in = np.asarray(run.tw.ledger.a_gate_in)
    owner = np.asarray(run.tw.ledger.owner)
    row = np.asarray(run.tw.ledger.row)
    go = np.asarray(run.tw.blocks.orders.gate_out_s)
    a = np.full((S,), np.inf)
    o = np.full((S,), np.inf)
    for s in range(S):
        if not reg[s]:
            continue
        av = float(a_in[s])
        if np.isfinite(av) and av <= sync_t + 1e-9:
            a[s] = av
        ov = float(go[int(owner[s]), int(row[s])])
        if np.isfinite(ov) and ov <= sync_t + 1e-9:
            o[s] = ov
    mask = np.asarray([d == int(day) for d in lay.truck_day], bool)
    flat = empty_orders(S)._replace(
        block=jnp.zeros((S,), jnp.int32), is_external=jnp.asarray(mask),
        gate_in_s=jnp.asarray(a, F), gate_out_s=jnp.asarray(o, F))
    return flat, jnp.asarray(mask)


def day_phi(run: MB.TerminalRun, lay: MonthLayout, day: int, *, end_s: float, tape: MonthTape,
            t0: float, t1: float, sync_t: float | None = None):
    """v5 `stage/month_run._phi_of_day` — 그 날 트럭(항1) + 계수기 차분(항2·3·4)으로 Φ 한 개.

    항1 은 **게이트를 들어온 날**로 귀속한다 (자정을 넘겨 끝나도 그 날 몫 — 세계가 이어지므로 실제로
    끝날 때까지 센다). 항2·3·4 는 누적 계수기라 시각이 없어 날 경계 사진의 차분을 쓴다.
    """
    v, yc, rh = tape.diff(t0, t1)
    flat, mask = day_orders(run, lay, day, sync_t=end_s if sync_t is None else sync_t)
    ships = [k for k in v]
    gt = jnp.asarray([v[k][0] for k in ships] or [0.0], F)
    idle = jnp.asarray([v[k][1] for k in ships] or [0.0], F)
    vm = jnp.asarray([True] * len(ships) or [False])
    return PH.terminal_cost_krw(flat, end_s, vessel_gt=gt, vessel_idle_s=idle, vessel_mask=vm,
                                yc_extra_move_s=yc, rehandles=rh, truck_mask=mask)


# ───────────────────────────────────────────────── 감사 (v5 정리 함수의 수치만)
def _prunable(run: MB.TerminalRun, t: float) -> np.ndarray:
    """v5 `prune_completed` 의 조건 — `status == DONE` 이고 (외부트럭이면) `gate_out ≤ t`. (B,N) bool."""
    o = run.tw.blocks.orders
    st = np.asarray(o.status)
    ext = np.asarray(o.is_external)
    blk = np.asarray(o.block)
    go = np.asarray(o.gate_out_s)
    return (st == JS_DONE) & (blk >= 0) & (~ext | (np.isfinite(go) & (go <= t)))


def prune_audit(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, t: float,
                seen: np.ndarray | None = None) -> dict:
    """v5 `stage/month.prune_completed` 가 **치웠을** 건수 — 배열판은 아무것도 치우지 않는다 (머리말).

    ★v5 는 실제로 지우므로 다음 날에는 **그 날 새로 끝난 것만** 센다. 배열판은 행이 남아 있으니
    `seen` (B,N) 에 이미 센 행을 표시해 **차분**을 낸다 — 그래야 v5 `DayReport.pruned` 와 `==` 다.
    ⚠️ v5 의 `a_sorted` 열은 **배열판에 대응물이 없다** — 그것은 시간장부의 정렬 리스트에서 이미 지나간
    앞부분 길이(`tl._a_idx`)이고, 배열판은 그 리스트 자체를 닫힌 식으로 대체했다. 그래서 이 감사는
    `a_sorted` 를 내지 않는다 (실측: 부하 150 에서 v5 150 vs 나간 트럭 149 로 값도 다르다).
    """
    gone = _prunable(run, t)
    if seen is not None:
        gone = gone & ~seen
        seen |= gone
    ext = np.asarray(run.tw.blocks.orders.is_external)
    n_jobs = int(gone.sum())
    n_tl = int((gone & ext).sum())
    return {"jobs": n_jobs, "ledger": n_jobs, "time_ledger": n_tl}


def retire_audit(run: MB.TerminalRun, lay: MonthLayout, t: float,
                 lag_s: float = RETIRE_LAG_S, seen: np.ndarray | None = None,
                 archive: list | None = None) -> int:
    """v5 `stage/month.retire_done_vessels` 가 **치웠을** 척수 — 조건 셋 (done · 남은 job 0 · lag 지남).

    배열판은 치우지 않는다 (머리말 ■ retire 무동작). 수치만 v5 와 대조해 그 주장을 시험한다.
    ★`prune_audit` 뒤에 부르는 것이 v5 순서다 — 끝난 job 이 치워져야 '남은 job 0' 이 참이 된다.
    `seen` (B,V) 를 주면 이미 센 배는 다시 세지 않는다 (v5 는 한 번 치우면 끝이다).
    `archive` 목록을 주면 치운 스트림 이름을 **v5 순서**(블록 순 × 붙은 순)로 덧붙인다 —
    그 순서가 `month_vessel_idle` 의 Neumaier 합 순서를 정한다.
    """
    V = run.tw.blocks.vessels
    o = run.tw.blocks.orders
    alive = np.asarray(V.alive); done = np.asarray(V.done)
    ac = np.asarray(V.actual_completion_s)
    ves = np.asarray(o.vessel); blk = np.asarray(o.block)
    pruned = _prunable(run, t)
    keys_of: dict = {}
    for key in live_vessel_order(lay):
        keys_of.setdefault(lay.block_of_key[key], []).append(key)
    n = 0
    for b in range(alive.shape[0]):
        live = set(int(x) for x in ves[b][(blk[b] >= 0) & (ves[b] >= 0) & ~pruned[b]])
        for key in keys_of.get(b, ()):                    # v5 `sim.vessels.items()` = 붙은 순
            v = lay.slot[(b, key)]
            if not alive[b, v] or not done[b, v] or v in live:
                continue
            if seen is not None and seen[b, v]:
                continue
            if np.isfinite(ac[b, v]) and (t - float(ac[b, v])) < lag_s:
                continue
            if seen is not None:
                seen[b, v] = True
            if archive is not None:
                archive.append(key)
            n += 1
    return n


# ───────────────────────────────────────────────── 30일 런
@dataclass
class MonthResult:
    """v5 `stage/month_run.MonthResult` 의 배열판 — 같은 열 이름으로 둔다 (대조가 기계적이 되게)."""

    days: list = field(default_factory=list)
    live: list = field(default_factory=list)
    admitted: int = 0
    skipped: int = 0
    decisions: int = 0
    vessel_admissions: list = field(default_factory=list)
    truck_skips: list = field(default_factory=list)
    pruned: list = field(default_factory=list)
    retired: list = field(default_factory=list)
    n_epochs: int = 0
    #: ★이어 돌리기 커서 — 날 번호·사진 시각·이미 센 행/배. `run_month` 를 `e0/e1` 로 쪼개 부를 때
    #:  같은 `res` 를 다시 주면 그대로 이어진다 (안 주면 날 경계가 처음부터 다시 돈다).
    state: dict | None = None


def close_day_at(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, tape: "MonthTape",
                 res: "MonthResult", state: dict, d: "MonthDay", t: float, *, opened=None,
                 on_day=None) -> dict:
    """어제를 닫는다 (v5 `month_run.run_month` 안의 `close_day`) — **모듈 수준 함수**.

    ★왜 밖에 있나 (2026-09-27 통합·수정): 학습 드라이버(`gpu/train.train_month`)도 날 경계마다 같은
      일을 해야 하는데, 닫힘 안에 있으면 부를 수 없어 **복제**가 생겼다. 한쪽만 고치면 조용히 갈리므로
      (검증 반박 '`_close_day` 복제') 두 호출자가 이 함수 **하나**를 쓴다.
    `opened` 를 주지 않으면 `state["opened"]` 를 읽는다 (학습 드라이버 쪽 관례).
    """
    op = state["opened"] if opened is None else opened
    rep = {"index": int(d.index), "load": int(d.load), "label": d.label, "train": bool(d.train),
           "vessels": op[0], "vessel_moves": op[1], "vessel_skipped": op[2],
           "provisional": True}
    phi = day_phi(run, lay, int(d.index), end_s=t, tape=tape, t0=d.t0, t1=t)
    rep.update(_phi_cols(phi))
    rep["pruned"] = prune_audit(run, tt, lay, t, seen=state["pruned_seen"])
    rep["retired"] = retire_audit(run, lay, t, seen=state["retired_seen"], archive=tape.archive)
    res.pruned.append(rep["pruned"])
    res.retired.append(rep["retired"])
    res.live.append(rep)
    if on_day is not None:
        on_day(rep)
    return rep


def run_month(run: MB.TerminalRun, tt: HT.TerminalTables, lay: MonthLayout, eng: MB.Engine, *,
              seed: int, profile, deadline_mult: float = VESSEL_DEADLINE_MULT,
              e0: int = 0, e1: int | None = None, finish: bool = True,
              on_day=None, on_epoch=None, on_boundary=None, tape: MonthTape | None = None,
              res: MonthResult | None = None):
    """v5 `stage/month_run.run_month` **기본 갈래**의 런 루프 — 에폭마다 호스트로 돌아온다.

    v5 `review(m, t)` 의 순서를 그대로 지킨다 (month_run.py:513-539):
      ① 투입(`ann.review`) → ② 시장(범위 밖 — NO_REALLOC 과 같다) → ③ 계수기 사진(`SNAP_S` 지나면)
      → ④ 날 경계면 `tape.snap(d.t0)` · 어제 닫기(`close_day`) · 오늘 배 붙이기(`open_day`)
    끝에 `tape.snap(month_s)` · 마지막 날 닫기 · `finish_run` (v5 541-548행).

    반환 `(run, MonthResult, MonthTape)`. 중간에 끊고 이어 돌릴 수 있게 `e0/e1` 과 `res/tape` 를 받는다.
    """
    E = run.n_epochs
    e1 = E if e1 is None else int(e1)
    res = res if res is not None else MonthResult()
    tape = tape if tape is not None else MonthTape(lay, profile)
    n_days = len(lay.days)
    W0 = run.tw.blocks
    if res.state is None:
        res.state = {"day": 0, "snap": 0.0, "opened": (0, 0, 0),
                     # v5 는 치우면 사라지므로 **차분**을 내려면 이미 센 것을 기억해야 한다 (prune_audit 머리말)
                     "pruned_seen": np.zeros(np.asarray(W0.orders.status).shape, bool),
                     "retired_seen": np.zeros(np.asarray(W0.vessels.alive).shape, bool)}
    state = res.state

    def close_day(run, d, t: float, opened):
        close_day_at(run, tt, lay, tape, res, state, d, t, opened=opened, on_day=on_day)

    slots = gate_out_slots(run, lay)
    grid = epoch_grid(run)
    for e in range(int(e0), e1):
        run, codes, skips = month_epoch(run, tt, lay, e, eng, slots=slots, grid=grid)
        t = float(grid[0][e])
        res.truck_skips += skips
        res.n_epochs += 1
        if on_epoch is not None:
            on_epoch(run, e, t, codes)
        if t >= state["snap"]:
            tape.snap(run, t)
            state["snap"] = t + SNAP_S
        while state["day"] < n_days and t >= lay.days[state["day"]].t0 - 1e-9:
            d = lay.days[state["day"]]
            tape.snap(run, d.t0)                       # 경계는 **정확히** 찍는다
            if on_boundary is not None:                # ★하루 경계의 세계 — 어제 야드 위에 오늘 (시험 고리)
                on_boundary(run, int(d.index), float(d.t0))
            if state["day"] > 0:
                close_day(run, lay.days[state["day"] - 1], d.t0, state["opened"])
            run, rows, opened = open_day(run, tt, lay, int(d.index), seed=seed, profile=profile,
                                         deadline_mult=deadline_mult, t0=d.t0)
            res.vessel_admissions += rows
            state["opened"] = opened
            state["day"] += 1
            state["snap"] = t + SNAP_S

    if finish and e1 >= E:
        run = MB.finish_run_jit(run, eng)
        tape.snap(run, lay.month_s)
        close_day(run, lay.days[-1], lay.month_s, state["opened"])
        # ── 확정 — 끝까지 기다린 값으로 다시 낸다 (판정은 이쪽 · v5 561-575행)
        for d in lay.days:
            phi = day_phi(run, lay, int(d.index), end_s=lay.sim_end_s, tape=tape,
                          t0=d.t0, t1=min(d.t1, lay.month_s))
            live = dict(res.live[int(d.index)])
            live.update(_phi_cols(phi), provisional=False)
            res.days.append(live)
        res.admitted = int(np.asarray(run.tw.ledger.registered).sum())
        res.skipped = len(res.truck_skips)
    return run, res, tape


def _phi_cols(phi) -> dict:
    return {"phi_krw": float(phi.total), "c_wait": float(phi.wait), "c_move": float(phi.move),
            "c_rehandle": float(phi.rehandle), "c_vessel": float(phi.vessel),
            "n_trucks": int(phi.n_trucks), "n_censored": int(phi.n_censored),
            "mean_turn_time_s": float(phi.mean_turn_time_s),
            "p90_turn_time_s": float(phi.p90_turn_time_s), "over_ratio": float(phi.over_ratio)}
