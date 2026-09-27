"""37칸 특징 벡터(gpu/v5feat.py)가 v5 학습 정책망에 들어가는 값과 **같은가** ([[YR-327]] 조각 7 · key=feat).

■ 무엇을 지키나 — 기대값을 손으로 적지 않는다. v5 를 실제로 굴려 **결정마다 · 크레인마다** v5 가
  망에 넣는 행렬을 그 자리에서 받아 배열판과 `==` 로 맞춘다.
    ① v5 구동 = `ppo/crane.py:CraneActor.__call__` 을 그대로 흉내 낸다 —
       블록 요약 `block_features(n_cands=None)` 한 번 → 크레인 사전순으로
       `generate` → `joint_mask` → `candidate_row(…, 직전 선택)` → 선택 → `_apply`.
       선택은 **정해진 규칙으로 돌려 가며** 고른다(결정 번호·크레인 위치로 순환) — 그래야 직전 크레인
       칸(19~23)에 SERVE·PRE_REHANDLE·REPOSITION·WAIT 가 모두 실린다.
       예외가 나면 `stage/episode.py:212-216` 의 대체 규칙(전원 WAIT)을 따르고 그 수를 보고한다.
    ② 블록 요약 8칸의 두 입력은 터미널 층 값이라 시험이 만든다 — `ExecutionRecord` 는 **v5 자신의**
       `MarketBridge._sync` 로 채우고(사건 ≤ t 만 찍는 v5 규칙 그대로), `Order` 는 명단 규약
       (`stage/orders.py:88` = `round(도착예정, 3)`)대로 만든다. 판정 대상은 v5 `block_features` 대 배열판이다.
    ③ 대조 — 24개 값 칸을 float64 로 `==`, 그리고 `ppo/model.py:encode(rows, "crane")` 를 **실제로 불러**
       float32 37칸까지 `==`. 어느 칸이 갈리는지 **칸 하나하나** 세어 보고하고, 처음 갈리는 결정·크레인·
       후보·칸을 이름으로 적는다.
    ④ 공허 방지 — 값 칸 24개마다 **0 이 아닌 횟수**를 센다. 하나라도 항상 0 이면 실패다
       (대조가 전부 0==0 이면 아무것도 확인하지 못한다). 구조적 0 칸(24~31·32·33·35·36)과
       항상 1 인 칸(34)도 따로 확인한다.

■ 무대 (전부 크레인 2 · 10×4×4 · 레인 2 · 결정 지평 1800 · 트럭은 exit_travel 있음 = 시간장부 활성)
  feat-basic       blocker 있는 반출 + ETA(선재조작) · ETA 반입 · 본선연계 → SERVE/PRE/REPO/WAIT 모두.
                   도착예정 4000초 트럭 하나로 '곧 올 통지분' 창(t+1800) **밖** 경우도 밟는다
  feat-plan-failed 야드 만재 + 크레인 둘 고장/복구 → **필수인데 계획 실패** SERVE (계획 칸이 0 이어야 한다)
  feat-crowded     트럭 10대 동시 도착(재조작 3단 → 필수 후보) + 선재조작 4 + 반입 2 → 추리기 작동
  feat-vessel      실제 본선 1척(적하 · 계획완료 있음) → 본선 여유 칸이 −2~2 안에서 실제로 움직인다
  feat-rand-s*     무작위 야드·오더 4 시드

  경계 탐침 넷(`probes`)이 무대 전체에서 0 이면 실패한다 — '이미 나간 트럭'·'창 밖 통지분'·'계획 실패
  필수 후보'·'작업 중 크레인(WAIT 하나뿐)' 이 한 번도 안 나오면 그 조건들은 대조되지 않은 것이다.
  후보 수(`n_items`)는 결정에 **안 부른 크레인까지** 전 크레인을 v5 `generate` 와 맞춘다.

실행 (Git Bash · Windows 파이썬):
    PYTHONPATH="src;tests/v6" PYTHONIOENCODING=utf-8 .venv-jax/Scripts/python.exe \
        -m pytest tests/v6/test_gpu_v5feat.py -q -p no:cacheprovider
"""
from __future__ import annotations

import importlib.util
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)       # ★float64 — 동등성은 x64 에서만
jnp = jax.numpy
torch = pytest.importorskip("torch")

from yard_rl.v6.gpu import cands3 as C3                                              # noqa: E402
from yard_rl.v6.gpu import v5feat as VF                                              # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                 # noqa: E402
from yard_rl.v6.gpu.host_convert import to_block_world                               # noqa: E402
from yard_rl.v6.gpu.state import PK_PRE_REHANDLE, PK_REPOSITION, PK_SERVE, PK_WAIT   # noqa: E402
# ── v5 정본 (읽기만 한다) ───────────────────────────────────────────────────────
from yard_rl.v6.features.block import block_features                                 # noqa: E402
from yard_rl.v6.ppo.crane import candidate_row, joint_mask                           # noqa: E402
from yard_rl.v6.ppo.model import encode                                              # noqa: E402
from yard_rl.v6.schema.order import Order                                            # noqa: E402
from yard_rl.v6.schema.record import ExecutionRecord                                 # noqa: E402
from yard_rl.v6.stage.bridge import MarketBridge                                      # noqa: E402
from yard_rl.v6.world.contract.schema import CandidateKind                            # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, InformationLevel, JobFlow, LoadStatus  # noqa: E402
from yard_rl.v6.world.domain.models import Job                                        # noqa: E402
from yard_rl.v6.world.integrated.baselines import _apply                              # noqa: E402
from yard_rl.v6.world.integrated.candidates import CandidateGenerator                 # noqa: E402
from yard_rl.v6.world.integrated.engine import TerminalSimulator                      # noqa: E402
from yard_rl.v6.world.integrated.policy_config import LEGACY_DEFAULT                  # noqa: E402
from yard_rl.v6.world.integrated.scenario import TerminalScenario                     # noqa: E402
from yard_rl.v6.world.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType  # noqa: E402
from yard_rl.v6.world.contract.vessel import CompletionBasis                          # noqa: E402

FT20, FT40, FT45 = ContainerSize.FT20, ContainerSize.FT40, ContainerSize.FT45
PA = InformationLevel.PRE_ADVICE
BID = "B1"
#: 특징 행렬의 정적 폭 — v5 items 는 k_max(12) 이하 + WAIT 이라 넉넉하다 (넘치면 overflow 로 실패)
C_MAX = 20
#: 한 무대의 결정 수 상한 (돌려 가며 고르면 REPOSITION 이 반복돼 끝이 멀어질 수 있다)
MAX_DECISIONS = 240

#: 시험 전체 집계 (마지막 시험이 보고)
REPORT: dict[str, dict] = {}
#: 값 칸별 "0 이 아닌 횟수" 누적 (37칸)
NONZERO = [0] * VF.INPUT_DIM
#: 값 칸별 불일치 누적
MISMATCH = [0] * VF.INPUT_DIM


def _load(name: str, fname: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C3T = _load("_c3_for_feat", "test_gpu_cands3.py")     # world_from_sim(REPO 번호표 포함) · prof_k2 · _c
world_from_sim, prof_k2, _c = _C3T.world_from_sim, _C3T.prof_k2, _C3T._c
_caps = _C3T._caps


# ───────────────────────────────────────────────── 무대 빌더
def _out(jid, target, gate_in, arrival, eta=None, exit_s=60.0):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, provided_eta=eta, target_container=target, exit_travel_s=exit_s)


def _in_(jid, gate_in, arrival, eta=None, size=FT40, exit_s=60.0):
    return Job(job_id=jid, flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, provided_eta=eta, inbound_size=size,
               inbound_load=LoadStatus.FULL, exit_travel_s=exit_s)


def _vl(jid, target, release, deadline, vid=None):
    return Job(job_id=jid, flow=JobFlow.VESSEL_LOAD, release_time=release, actual_gate_in=None,
               actual_block_arrival=None, target_container=target, deadline=deadline,
               priority_class=1, vessel_id=vid)


def _scn(sid, containers, jobs, horizon=7200.0, drain=0.0, vessels=()):
    return TerminalScenario(scenario_id=sid, seed=0, horizon_s=horizon, drain_window_s=drain,
                            containers=containers, jobs=jobs, vessels=list(vessels), injected_events=[])


def basic_scenario():
    """C-T(5,1,1) 위 blocker → 선재조작. 게이트인과 블록도착 간격을 크게 벌려 '오는 중' 칸을 살린다."""
    containers = {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2), "C-X": _c("C-X", 2, 1, 1),
                  "C-Y": _c("C-Y", 9, 1, 1), "C-Z": _c("C-Z", 9, 2, 1), "C-W": _c("C-W", 7, 3, 1),
                  "C-L": _c("C-L", 6, 4, 1)}
    jobs = [_out("J-OUT-T", "C-T", 600.0, 1500.0, 900.0), _out("J-OUT-X", "C-X", 0.0, 300.0),
            _out("J-OUT-Y", "C-Y", 900.0, 2000.0, 1900.0), _out("J-OUT-W", "C-W", 200.0, 1000.0, 1000.0),
            # ★도착예정이 한참 뒤(4000초) — 이른 결정에서 '곧 올 통지분(t+1800)' 창 **밖**이라 세면 안 된다
            _out("J-OUT-L", "C-L", 4000.0, 4600.0, 4500.0),
            _in_("J-IN-1", 1200.0, 2500.0, 2400.0), _vl("J-VL-1", "C-Z", 700.0, 3000.0)]
    return _scn("feat-basic", containers, jobs)


def crowded_scenario():
    """트럭 10대가 300초에 동시 도착(뒤쪽은 재조작 3단 → 대기 > 1440 = 필수 후보) + 선재조작 대상 4 + 반입 2."""
    containers, jobs = {}, []
    for b in range(1, 7):
        containers[f"S{b}"] = _c(f"S{b}", b, 1, 1)
        jobs.append(_out(f"J-S{b}", f"S{b}", 0.0, 300.0))
    for b in range(7, 11):
        containers[f"H{b}"] = _c(f"H{b}", b, 1, 1)
        for t in (2, 3, 4):
            containers[f"H{b}B{t}"] = _c(f"H{b}B{t}", b, 1, t)
        jobs.append(_out(f"J-H{b}", f"H{b}", 0.0, 300.0))
    for i, b in enumerate(range(2, 6)):
        containers[f"P{b}"] = _c(f"P{b}", b, 2, 1)
        containers[f"P{b}B"] = _c(f"P{b}B", b, 2, 2)
        jobs.append(_out(f"J-P{b}", f"P{b}", 500.0, 2100.0 + 100.0 * i, 1600.0 + 100.0 * i))
    jobs += [_in_("J-IN-1", 400.0, 1200.0, 1000.0, FT40), _in_("J-IN-2", 400.0, 1250.0, 1000.0, FT20)]
    containers["V9"] = _c("V9", 9, 3, 1)
    containers["V10"] = _c("V10", 10, 3, 1)
    jobs += [_vl("J-VL-1", "V9", 600.0, 5000.0), _vl("J-VL-2", "V10", 1000.0, 600.0)]
    return _scn("feat-crowded", containers, jobs, horizon=7200.0)


def vessel_scenario():
    """실제 본선 1척(적하 4 moves · 계획완료 4000 · cadence 300) — 본선 여유 칸이 시간에 따라 줄어든다."""
    containers = {f"C-VL{i}": _c(f"C-VL{i}", 3 + i, 2, 1) for i in range(4)}
    containers["C-T"] = _c("C-T", 8, 1, 1)
    containers["C-TB"] = _c("C-TB", 8, 1, 2)
    containers["C-U"] = _c("C-U", 2, 1, 1)
    jobs = [_vl(f"J-V-L{m}", f"C-VL{m}", 600.0 + 300.0 * m, 4000.0, "V-LOAD") for m in range(4)]
    jobs += [_out("J-OUT-T", "C-T", 400.0, 1400.0, 900.0), _out("J-OUT-U", "C-U", 0.0, 300.0)]
    ves = [VesselProcess("V-LOAD", VesselWorkType.LOAD,
                         VesselPlan(planned_start_s=600.0, planned_completion_s=4000.0,
                                    completion_basis=CompletionBasis.PLAN_COMPUTED, etd_s=5000.0,
                                    total_moves=4, sts_move_interval_s=300.0, quay_buffer_cap=3))]
    return _scn("feat-vessel", containers, jobs, horizon=7200.0, drain=1800.0, vessels=ves)


def rand_scenario(seed: int, B: int = 10, R: int = 4, T: int = 4) -> TerminalScenario:
    """무작위 야드(35%) · 반출 3~6(ETA 75%) · 반입 1~3(ETA 50%) · 본선연계 1~2."""
    rng = random.Random(9100 + seed)
    containers, n = {}, 0
    for bay in range(1, B + 1):
        for row in range(1, R + 1):
            if rng.random() < 0.35:
                size = rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0]
                for t in range(1, rng.randint(1, T) + 1):
                    containers[f"C{n:03d}"] = _c(f"C{n:03d}", bay, row, t, size)
                    n += 1
    pool = sorted(containers)
    jobs = []
    for i in range(rng.randint(3, 6)):
        tgt = pool[rng.randrange(len(pool))]
        arr = float(rng.randint(200, 2400))
        eta = arr + rng.uniform(-300.0, 600.0) if rng.random() < 0.75 else None
        jobs.append(_out(f"J-OUT-{i}", tgt, max(0.0, arr - 900.0), arr, eta))
    for i in range(rng.randint(1, 3)):
        arr = float(rng.randint(200, 2400))
        eta = arr + rng.uniform(-200.0, 400.0) if rng.random() < 0.5 else None
        jobs.append(_in_(f"J-IN-{i}", max(0.0, arr - 900.0), arr, eta))
    for i in range(rng.randint(1, 2)):
        jobs.append(_vl(f"J-VL-{i}", pool[rng.randrange(len(pool))], float(rng.randint(100, 3000)),
                        float(rng.randint(1500, 5000))))
    seen, uniq = set(), []
    for j in jobs:                                    # 같은 대상 두 번은 v5 검증이 막는다
        if j.target_container is not None and j.target_container in seen:
            continue
        if j.target_container is not None:
            seen.add(j.target_container)
        uniq.append(j)
    return _scn(f"feat-rand-s{seed}", containers, uniq, horizon=7200.0)


def plan_failed_scenario():
    """필수인데 계획이 실패하는 SERVE (계획 칸 14·16·17·18 이 0 이어야 한다 — 머리말 함정 ③).

    `test_gpu_cands3.plan_failed_mandatory_scenario` 를 시간장부 켠 판(출문주행 60초)으로 옮겼다.
    야드는 (5,1) 만 비고 나머지는 4단 만재, (4,1) 은 대상 T + blocker. 트럭 셋이 2초에 도착하는데
    두 크레인이 1초부터 고장 → YC-A 1450 복구(대기 1448 ≥ 1440 = 필수) → A-OUT 배정이 (5,1) 을 예약 →
    YC-B 1500 복구 때 B-IN 은 배차 가능하지만 계획이 실패해 `PLAN_FAILED` 로 목록에 오른다.
    """
    from yard_rl.v6.world.integrated.scenario import InjectedEvent
    containers, n = {}, 0
    for bay in range(1, 11):
        for row in range(1, 5):
            if (bay, row) in ((4, 1), (5, 1), (9, 1)):
                continue
            for t in (1, 2, 3, 4):
                containers[f"F{n:03d}"] = _c(f"F{n:03d}", bay, row, t)
                n += 1
    for t in (1, 2, 3, 4):
        containers[f"G{t}"] = _c(f"G{t}", 9, 1, t)
    containers["T"] = _c("T", 4, 1, 1)
    containers["TB"] = _c("TB", 4, 1, 2)
    jobs = [_out("A-OUT", "T", 0.0, 2.0), _in_("B-IN", 0.0, 2.0, None, FT40), _out("C-OUT", "G4", 0.0, 2.0)]
    inj = [InjectedEvent(1.0, "EQUIPMENT_DOWN", "YC-A"), InjectedEvent(1.0, "EQUIPMENT_DOWN", "YC-B"),
           InjectedEvent(1450.0, "EQUIPMENT_UP", "YC-A"), InjectedEvent(1500.0, "EQUIPMENT_UP", "YC-B")]
    return TerminalScenario(scenario_id="feat-plan-failed", seed=0, horizon_s=3600.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=inj)


STAGES = {
    "feat-basic": lambda: (prof_k2(2.0), basic_scenario()),
    "feat-plan-failed": lambda: (_C3T.prof_stepped(0.0), plan_failed_scenario()),
    "feat-crowded": lambda: (prof_k2(2.0), crowded_scenario()),
    "feat-vessel": lambda: (prof_k2(2.0), vessel_scenario()),
}
for _s in (1, 2, 3, 4):
    STAGES[f"feat-rand-s{_s}"] = (lambda s: lambda: (prof_k2(2.0 if s % 2 else 3.0), rand_scenario(s)))(_s)


# ───────────────────────────────────────────────── 터미널 층 입력 (블록 요약의 두 인자)
def _orders_of(sim) -> dict:
    """명단 규약대로 `Order` 를 만든다 — `in_out_reserve_s = round(도착예정, 3)` (stage/orders.py:88).

    외부트럭만이 오더다 (본선 작업은 `records`·`orders` 어디에도 없다 — bridge.py:66-67).
    """
    out = {}
    for jid, j in sim.jobs.items():
        if not j.is_external_truck:
            continue
        arr = float(j.actual_gate_in or 0.0)
        notice = round(max(0.0, arr - 1800.0), 3)
        out[jid] = Order(doc_key=jid, in_out=(0 if j.flow == JobFlow.GATE_OUT else 1),
                         copino_notice_s=notice, in_out_reserve_s=round(arr, 3), con_loc=BID,
                         con_no=(j.target_container if j.target_container is not None else f"IN-{jid}"))
    return out


def _reserve_arr(orders: dict, tb, n: int) -> jnp.ndarray:
    a = np.full((n,), np.inf)
    jidx = tb.job_index
    for jid, o in orders.items():
        a[jidx[jid]] = o.in_out_reserve_s
    return jnp.asarray(a)


def _with_gate_out(pre, sim, tb):
    """`world_from_sim` 이 안 채우는 `gate_out_s`(O = C + 출문주행) 를 v5 시간장부에서 옮긴다.

    통합 엔진(`engine_step.py:541`)은 완료 처리기에서 이 열을 채운다 — 여기서는 v5 진행 상태를
    옮기는 중이라 같은 값을 손으로 놓는다.
    """
    go = np.asarray(pre.orders.gate_out_s).copy()
    tl = getattr(sim, "time_ledger", None)
    if tl is not None:
        jidx = tb.job_index
        for jid, tt in tl.records.items():
            if tt.gate_out is not None:
                go[jidx[jid]] = float(tt.gate_out)
    return pre._replace(orders=pre.orders._replace(gate_out_s=jnp.asarray(go)))


# ───────────────────────────────────────────────── v5 구동 (CraneActor 흉내)
def _key(gc):
    if gc.kind == CandidateKind.WAIT:
        return ("WAIT",)
    if gc.kind == CandidateKind.REPOSITION:
        return ("REPOSITION", float(gc.job_ref.reposition_target_bay))
    return (gc.kind.value, gc.job_ref.job_id)


def _wait_of(items):
    for gc in items:
        if gc.kind == CandidateKind.WAIT:
            return gc
    raise AssertionError("WAIT 후보가 없다 — v5 generate 규약 위반")


def run_v5(prof, scn, level, w0, tb, g, end_s: float):
    """`CraneActor.__call__` 을 그대로 흉내 내며 결정마다 (블록요약·크레인별 행렬·선택) 을 남긴다."""
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    orders = _orders_of(sim)
    records = {jid: ExecutionRecord(doc_key=jid, copino_notice_s=o.copino_notice_s)
               for jid, o in orders.items()}
    mbt = SimpleNamespace(blocks={BID: sim})
    # v5 `MarketBridge._sync` 를 그대로 부른다 — 쓰는 속성은 `records` 와 `_stamp_past` 뿐이다
    syncer = SimpleNamespace(records=records, _stamp_past=MarketBridge._stamp_past)
    recs, exc_n, d = [], 0, 0
    while (dp := sim.run_until_decision()) is not None:
        MarketBridge._sync(syncer, mbt, sim.now)            # ★v5 자신의 기록 동기화 (사건 ≤ t 만)
        bf = block_features(mbt, BID, sim.now, n_cands=None, records=records, orders=orders, end_s=end_s)
        pre = _with_gate_out(world_from_sim(sim, w0, tb, g), sim, tb)
        # ★결정에 안 부른 크레인까지 후보 수를 적어 둔다 — '유휴·비양보 아니면 WAIT 하나뿐'
        #   (candidates.py:256-257) 분기도 대조하려면 전 크레인이 필요하다. generate 는 상태를 안 바꾼다.
        counts = {c: len(gen.generate(sim, c, level).items) for c in sim.fleet.ids()}
        busy = {c: not (sim.fleet.get(c).idle and not sim.fleet.get(c).yielded) for c in sim.fleet.ids()}
        selected, per_crane = {}, []
        for pos, cid in enumerate(sorted(dp.crane_ids)):
            items = gen.generate(sim, cid, level).items
            mask = [bool(m) for m in joint_mask(sim, [(cid, gc) for gc in items], selected)]
            rows = [candidate_row(sim, gc, bf, selected) for gc in items]
            feas = [i for i, m in enumerate(mask) if m]
            idx = feas[(d * 7 + pos * 3) % len(feas)]       # 돌려 가며 고른다 (직전 칸 네 종류 다 나오게)
            per_crane.append(dict(cid=cid, keys=[_key(gc) for gc in items], rows=rows, mask=mask,
                                  idx=idx, chosen=_key(items[idx])))
            selected[cid] = items[idx]
        try:
            _apply(sim, selected)
        except Exception:                                   # stage/episode.py:212-216 대체 규칙
            exc_n += 1
            _apply(sim, {c: _wait_of(gen.generate(sim, c, level).items) for c in dp.crane_ids})
        recs.append(dict(t=sim.now, pre=pre, block=bf, cranes=per_crane, orders=orders,
                         counts=counts, busy=busy))
        d += 1
        if d >= MAX_DECISIONS:
            break
    return sim, recs, exc_n, orders


# ───────────────────────────────────────────────── 배열 쪽
_JIT: dict = {}


def _cands_jit(g: Geom, pre_advice: bool):
    key = (g, pre_advice)
    if key not in _JIT:
        def f(w, hz):
            out = C3.candidates3(w, g, horizon_s=hz, pre_advice=pre_advice)
            fl = C3.flat_view(out)
            return fl, C3.prune(fl, g)
        _JIT[key] = jax.jit(f)
    return _JIT[key]


def _col_of(fl, k: int, key, N: int, C: int, jidx: dict) -> int:
    """v5 후보 키 → `flat` 열 번호 (SERVE·PRE 는 오더 번호, REPO 는 목표 칸 일치, WAIT 은 마지막)."""
    if key == ("WAIT",):
        return C - 1
    if key[0] == "REPOSITION":
        for c in range(N, C - 1):
            if int(fl.kind[k, c]) == PK_REPOSITION and float(fl.bay[k, c]) == key[1]:
                return c
        raise AssertionError(f"배열판에 REPOSITION 목표 {key[1]} 열이 없다 (크레인 {k})")
    return jidx[key[1]]


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def check_stage(label: str) -> dict:
    prof, scn = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    assert tb.ledger_mode, f"{label}: 시간장부가 꺼져 있다 — 트럭에 exit_travel_s 를 줘야 기록 칸이 산다"
    hz = float(prof.decision_horizon_s)
    end_s = float(scn.horizon_s) + float(scn.drain_window_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)   # v5 가 더하는 순서
    sim, recs, exc_n, orders = run_v5(prof, scn, PA, w0, tb, g, end_s)
    assert recs, f"{label}: 결정이 하나도 없다 — 무대가 공허하다"
    res_arr = _reserve_arr(orders, tb, w0.n)
    jidx, cidx = tb.job_index, tb.crane_index
    fn = _cands_jit(g, True)

    n_rows = n_block_cmp = 0
    first_bad = None
    kinds_seen, prior_seen = set(), set()
    #: 블록 요약의 경계 조건이 실제로 밟히는지 — 안 밟히면 그 조건은 공허하다
    probes = dict(left=0, far=0, plan_failed=0, busy_wait_only=0)
    res_np = np.asarray(res_arr)
    for i, rec in enumerate(recs):
        pre = rec["pre"]
        N, K = pre.n, pre.k
        t = float(rec["t"])
        present = np.asarray(pre.orders.block) >= 0
        if bool((present & (np.asarray(pre.orders.gate_out_s) <= t)).any()):
            probes["left"] += 1            # 이미 나간 트럭 → '블록 안 전부' 에서 빠져야 한다
        if bool((present & (res_np > t + 1800.0) & np.isfinite(res_np)).any()):
            probes["far"] += 1             # 30분 창 밖 통지분 → '곧 올 통지분' 에서 빠져야 한다
        fl, pr = _np(fn(pre, hz))
        C = fl.raw.shape[1]

        # ── 블록 요약 8칸 (v5 block_features 대 배열판) ──
        blk = np.asarray(VF.block_row(pre, g, end_s=end_s, reserve_s=res_arr, crane_order=crane_order))
        want_blk = np.asarray(rec["block"], np.float64)
        assert want_blk.shape == (VF.BLOCK_DIM,), f"{label} #{i}: v5 블록 요약 폭 {want_blk.shape}"
        for c in range(VF.BLOCK_DIM):
            if blk[c] != want_blk[c]:
                MISMATCH[c] += 1
                if first_bad is None:
                    first_bad = (i, rec["t"], "-", "블록요약", c, VF.FEATURE_NAMES[c],
                                 float(want_blk[c]), float(blk[c]))
        n_block_cmp += 1

        # ── 직전 크레인 재료: 크레인 사전순으로 앞 크레인의 **선택**을 배열 열에서 읽는다 ──
        pk = np.full((K,), VF.NO_PRIOR, np.int32)
        pe = np.full((K,), np.nan, np.float64)
        prev_col = prev_k = None
        for cr in rec["cranes"]:
            k = cidx[cr["cid"]]
            if prev_col is not None:
                kind, eb = VF.prior_from_choice(fl, prev_k, prev_col)
                pk[k], pe[k] = int(kind), float(eb)
                prior_seen.add(int(kind))
            prev_k, prev_col = k, _col_of(fl, k, cr["chosen"], N, C, jidx)

        fo = VF.features(pre, g, fl, pr, block=jnp.asarray(blk), prior_kind=jnp.asarray(pk),
                         prior_end_bay=jnp.asarray(pe), c_max=C_MAX)
        assert int(fo.overflow) == 0, f"{label} #{i}: candidate_id 가 c_max={C_MAX} 를 넘었다"
        x = np.asarray(fo.x)
        x32 = np.asarray(VF.as_net_input(fo.x))
        mask = np.asarray(fo.mask)
        n_items_arr = np.asarray(fo.n_items)

        # ── 전 크레인 후보 수 (결정에 안 부른 크레인 포함) ──
        for cid, want_n in rec["counts"].items():
            k = cidx[cid]
            assert int(n_items_arr[k]) == want_n, (
                f"{label} #{i} {cid}: 후보 수 arr={int(n_items_arr[k])} v5={want_n} (부른 크레인 밖)")
            if rec["busy"][cid]:
                probes["busy_wait_only"] += 1
                assert want_n == 1, f"{label} #{i} {cid}: 유휴가 아닌 크레인인데 후보가 {want_n} 개"

        # ── 후보 행렬 대조 ──
        for cr in rec["cranes"]:
            k = cidx[cr["cid"]]
            want = np.asarray(cr["rows"], np.float64)              # (n, 24) — v5 candidate_row 그대로
            n = want.shape[0]
            assert want.shape[1] == VF.CRANE_ROW_DIM, f"{label} #{i}: v5 행 폭 {want.shape[1]}"
            assert int(n_items_arr[k]) == n, (
                f"{label} #{i} {cr['cid']}: 후보 수 arr={int(n_items_arr[k])} v5={n}")
            assert mask[k, :n].all() and not mask[k, n:].any(), f"{label} #{i} {cr['cid']}: 실후보 마스크 어긋남"
            assert (x[k, n:] == 0.0).all(), f"{label} #{i} {cr['cid']}: 안 실린 행이 0 이 아니다"

            # 배열판의 후보 순서가 v5 items 순서(candidate_id)인지 — 열 번호로 되짚어 확인
            for j, key in enumerate(cr["keys"]):
                col = _col_of(fl, k, key, N, C, jidx)
                assert int(pr.candidate_id[k, col]) == j, (
                    f"{label} #{i} {cr['cid']}: {key} 의 candidate_id arr={int(pr.candidate_id[k, col])} v5={j}")
                kinds_seen.add(int(fl.kind[k, col]))
                if int(fl.kind[k, col]) != PK_WAIT and not bool(fl.plan_ok[k, col]):
                    probes["plan_failed"] += 1       # 계획 실패 필수 후보 → 계획 칸이 0 이어야 한다

            got = x[k, :n]
            for j in range(n):
                n_rows += 1
                for c in range(VF.INPUT_DIM):
                    w = float(want[j, c]) if c < VF.CRANE_ROW_DIM else (
                        1.0 if VF.FEATURE_NAMES[c] == "role_crane" else 0.0)
                    if w != 0.0:
                        NONZERO[c] += 1
                    if got[j, c] != w:
                        MISMATCH[c] += 1
                        if first_bad is None:
                            first_bad = (i, rec["t"], cr["cid"], cr["keys"][j], c,
                                         VF.FEATURE_NAMES[c], w, float(got[j, c]))
            # float32 (망이 실제로 보는 값) — v5 encode 를 실제로 부른다
            enc = encode(cr["rows"], "crane").numpy()
            assert enc.shape == (n, VF.INPUT_DIM)
            bad32 = np.argwhere(x32[k, :n] != enc)
            assert not len(bad32), (
                f"{label} #{i} {cr['cid']}: float32 캐스팅 후 {len(bad32)} 칸 불일치 "
                f"(첫 자리 후보 {bad32[0][0]} 칸 {VF.FEATURE_NAMES[bad32[0][1]]})")

    assert first_bad is None, (
        f"{label}: 특징 불일치 — 결정 #{first_bad[0]} t={first_bad[1]:.3f} 크레인 {first_bad[2]} "
        f"후보 {first_bad[3]} 칸 {first_bad[4]}({first_bad[5]}) v5={first_bad[6]!r} 배열={first_bad[7]!r}")
    return dict(decisions=len(recs), rows=n_rows, block_points=n_block_cmp, exceptions=exc_n,
                kinds=sorted(kinds_seen), priors=sorted(prior_seen), probes=probes)


# ───────────────────────────────────────────────── 시험
def test_feature_name_table():
    """37칸 이름표가 v5 폭·역할 순서와 맞물리는가 (이 모듈의 정본 산출물)."""
    assert VF.INPUT_DIM == 37 and VF.RAW_DIM == 32 and VF.CRANE_ROW_DIM == 24
    from yard_rl.v6.ppo import model as M
    assert VF.INPUT_DIM == M.INPUT_DIM and VF.RAW_DIM == M.RAW_DIM and VF.ROLES == M.ROLES
    from yard_rl.v6.features.block import BLOCK_DIM_BUYER
    assert VF.BLOCK_DIM == BLOCK_DIM_BUYER == 8          # n_cands=None 판 (block_state)
    assert len(VF.FEATURE_NAMES) == 37 and len(set(VF.FEATURE_NAMES)) == 37
    assert VF.FEATURE_NAMES[32:] == ("role_seller", "role_buyer", "role_crane", "role_state", "buyer_first_row")
    assert VF.KIND_ORDER == (PK_SERVE, PK_PRE_REHANDLE, PK_REPOSITION, PK_WAIT)
    # 종류 원핫 칸 순서가 v5 KINDS 와 같은가
    from yard_rl.v6.ppo.crane import KINDS
    names = ("cand_kind_serve", "cand_kind_pre_rehandle", "cand_kind_reposition", "cand_kind_wait")
    assert tuple(k.value for k in KINDS) == ("SERVE", "PRE_REHANDLE", "REPOSITION", "WAIT")
    assert tuple(VF.FEATURE_NAMES[8:12]) == names


@pytest.mark.parametrize("d", [1, 7, 16, 24, 32])
def test_encode_rows_matches_v5(d):
    """`encode_rows` 가 v5 `ppo/model.py:encode` 와 **비트 동일** (역할 crane·state · 폭 1~32)."""
    rng = np.random.default_rng(1234 + d)
    rows = rng.normal(size=(5, d)) * 3.0
    for role in ("crane", "state"):
        want = encode(rows.tolist(), role).numpy()
        got = np.asarray(VF.as_net_input(VF.encode_rows(jnp.asarray(rows), role)))
        assert got.shape == want.shape == (5, 37)
        bad = np.argwhere(got != want)
        assert not len(bad), f"role={role} d={d}: {len(bad)} 칸 불일치 (첫 칸 {bad[0][1]})"


def test_state_rows_matches_v5():
    """`state_rows` 가 `ppo/runtime.py:112-113 states_at` 과 같은가 (블록 요약 8칸 · 역할 state)."""
    rng = np.random.default_rng(99)
    blocks = rng.normal(size=(3, 8))
    want = encode(blocks.tolist(), "state").numpy()
    got = np.asarray(VF.as_net_input(VF.state_rows(jnp.asarray(blocks))))
    assert np.array_equal(got, want)
    assert got[:, 35].tolist() == [1.0, 1.0, 1.0] and got[:, 34].tolist() == [0.0, 0.0, 0.0]


def test_encode_rows_rejects_buyer_and_overwide():
    with pytest.raises(NotImplementedError):
        VF.encode_rows(jnp.zeros((2, 5)), "buyer")
    with pytest.raises(ValueError):
        VF.encode_rows(jnp.zeros((2, 33)), "crane")
    with pytest.raises(ValueError):
        VF.encode_rows(jnp.zeros((2, 5)), "nobody")


@pytest.mark.parametrize("label", list(STAGES))
def test_features_match_v5(label):
    """결정마다·크레인마다 v5 가 망에 넣는 37칸과 배열판이 `==` 인가."""
    REPORT[label] = check_stage(label)
    r = REPORT[label]
    assert r["rows"] > 0
    assert PK_WAIT in r["kinds"], f"{label}: WAIT 후보가 안 나왔다"


def test_div_is_exact_in_four_modes():
    """★상수 나눗셈 함정 — `_div` 가 eager·jit·vmap·scan 네 경로에서 파이썬과 비트 동일한가.

    배열을 상수로 그냥 나누면 XLA 가 역수 곱으로 바꿔 `3.0/10` 이 0.3 대신 0.30000000000000004 가 된다.
    `cand_rehandles_10` 이 실제로 이 한 비트에서 갈렸다 (2026-09-26).
    """
    raw = [3.0, 7.0, 1.0, 9.0, 11.0, 13.0, 17.0, 49.0]
    want = np.asarray([v / 10.0 for v in raw])
    a = jnp.asarray(raw)
    b = jnp.stack([a, a])
    d = VF._div
    got = {
        "eager": np.asarray(d(a, 10.0)),
        "jit": np.asarray(jax.jit(lambda x: d(x, 10.0))(a)),
        "vmap": np.asarray(jax.jit(jax.vmap(lambda x: d(x, 10.0)))(b))[0],
        "scan": np.asarray(jax.jit(lambda z: jax.lax.scan(
            lambda c, x: (c, d(x, 10.0)), 0.0, z)[1])(b))[0],
    }
    for mode, v in got.items():
        bad = np.argwhere(v != want)
        assert not len(bad), f"{mode}: {len(bad)} 개 값이 파이썬 나눗셈과 다르다 (첫 자리 {bad[0][0]})"
    naive_still_wrong = not np.array_equal(np.asarray(a / 10.0), want)
    print(f"\n  [탐침] 그냥 `배열/10.0` 은 여전히 어긋난다: {naive_still_wrong}")


def test_jit_and_vmap_match_eager():
    """`block_row`·`features` 가 jit·vmap 아래서도 eager 와 **비트 동일**한가 (조각 8 의 배치 학습 전제)."""
    label = "feat-crowded"
    prof, scn = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    end_s = float(scn.horizon_s) + float(scn.drain_window_s)
    crane_order = tuple(tb.crane_index[c.crane_id] for c in prof.cranes)
    _sim, recs, _exc, orders = run_v5(prof, scn, PA, w0, tb, g, end_s)
    res_arr = _reserve_arr(orders, tb, w0.n)
    pre = recs[len(recs) // 2]["pre"]
    fl, pr = _cands_jit(g, True)(pre, hz)
    K = pre.k
    pk = jnp.asarray([PK_SERVE] + [VF.NO_PRIOR] * (K - 1), jnp.int32)
    pe = jnp.asarray([7.0] + [np.nan] * (K - 1))

    def blk_of(w, res):
        return VF.block_row(w, g, end_s=end_s, reserve_s=res, crane_order=crane_order)

    def feat_of(w, f, p, b):
        return VF.features(w, g, f, p, block=b, prior_kind=pk, prior_end_bay=pe, c_max=C_MAX).x

    b_eager = blk_of(pre, res_arr)
    x_eager = feat_of(pre, fl, pr, b_eager)
    b_jit = jax.jit(blk_of)(pre, res_arr)
    x_jit = jax.jit(feat_of)(pre, fl, pr, b_jit)
    assert np.array_equal(np.asarray(b_jit), np.asarray(b_eager)), "block_row 가 jit 에서 갈렸다"
    assert np.array_equal(np.asarray(x_jit), np.asarray(x_eager)), "features 가 jit 에서 갈렸다"

    stack = lambda tree: jax.tree_util.tree_map(lambda v: jnp.stack([v, v]), tree)
    b_v = jax.jit(jax.vmap(blk_of))(stack(pre), stack(res_arr))
    x_v = jax.jit(jax.vmap(feat_of))(stack(pre), stack(fl), stack(pr), b_v)
    for i in (0, 1):
        assert np.array_equal(np.asarray(b_v[i]), np.asarray(b_eager)), "block_row 가 vmap 에서 갈렸다"
        assert np.array_equal(np.asarray(x_v[i]), np.asarray(x_eager)), "features 가 vmap 에서 갈렸다"


def test_crane_backlog_uses_python_sum_and_profile_order():
    """크레인 여유 합(칸 2)이 v5 `sum(제너레이터)` 와 비트 동일한가 — 크레인 5대·보정합이 갈리는 값.

    v5 는 `sum(max(0, 여유 − t) for c in sim.profile.cranes)` 다. 파이썬 3.12 의 `sum` 은 **보정합**
    (Neumaier)이라 3항부터 순차 `+=` 와 마지막 비트가 갈릴 수 있고, 더하는 순서는 프로파일의 크레인
    나열 순서다. 그래서 `exact.sum_python` + `crane_order` 를 쓴다 — 여기서 그 둘을 직접 못박는다.
    """
    from yard_rl.v6.gpu.state import empty_block_world
    g = Geom(bay_count=10, row_count=4, tier_max=4, bay_len=6.5, row_w=2.9, tier_h=2.6,
             transfer_row=0.0, sla_s=1800.0, shift_len_s=28800.0, gap=2.0, n_lanes=2)
    K = 5
    w = empty_block_world(g, n_orders=4, n_cranes=K, n_conts=4, q_cap=8, log_cap=8, end_s=9000.0)
    t = 100.0
    # 크기 차가 큰 값들 — 보정합과 순차 합이 갈리기 쉬운 배치
    avail = [t + 1e16, t + 1.0, t + 1.0, t - 5.0, t + 2.0 ** -20]
    w = w._replace(clock=jnp.asarray(t), cranes=w.cranes._replace(available_at=jnp.asarray(avail)))
    res = jnp.full((w.n,), jnp.inf)
    for order in [tuple(range(K)), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3)]:
        terms = [max(0.0, avail[k] - t) for k in order]
        want = sum(terms) / 3600.0                      # ★v5 와 똑같은 파이썬 식
        got = float(VF.block_row(w, g, end_s=9000.0, reserve_s=res, crane_order=order)[2])
        assert got == want, f"순서 {order}: 배열 {got!r} vs v5 파이썬 {want!r}"
    # 순차 합과 실제로 갈리는 배치인지 — 안 갈리면 이 시험은 보정합을 확인하지 못한다
    seq = 0.0
    for v in [max(0.0, avail[k] - t) for k in range(K)]:
        seq += v
    print()
    print(f"  [탐침] 보정합 {sum(max(0.0, avail[k] - t) for k in range(K))!r} vs 순차합 {seq!r}")


def test_block_row_needs_terminal_inputs():
    """블록 요약의 두 터미널 입력(`end_s`·`reserve_s`)은 기본값이 없다 — 조용히 틀린 0 을 막는다."""
    prof, scn = STAGES["feat-basic"]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    with pytest.raises(TypeError):
        VF.block_row(w0, g)                                   # end_s·reserve_s 없음
    with pytest.raises(ValueError):
        VF.block_row(w0, g, end_s=1.0, reserve_s=jnp.full((w0.n,), jnp.inf), crane_order=(0, 0))


def test_report():
    """마지막 — 칸별 0 이 아닌 횟수와 불일치 수를 사람 말로 보고하고, 공허한 칸이 없음을 확인한다."""
    assert len(REPORT) == len(STAGES), "무대 시험이 다 안 돌았다 (통과 수를 확인하라)"
    total_rows = sum(r["rows"] for r in REPORT.values())
    print("\n■ 무대별")
    for label, r in REPORT.items():
        print(f"  {label:16s} 결정 {r['decisions']:4d} · 행 {r['rows']:5d} · 블록요약 {r['block_points']:4d}점 "
              f"· 예외대체 {r['exceptions']} · 후보종류 {r['kinds']} · 직전종류 {r['priors']} · 탐침 {r['probes']}")
    print(f"\n■ 칸별 '0 이 아닌 횟수' / 전체 {total_rows} 행")
    for c, name in enumerate(VF.FEATURE_NAMES):
        print(f"  {c:2d} {name:28s} {NONZERO[c]:6d}   불일치 {MISMATCH[c]}")
    assert sum(MISMATCH) == 0, f"불일치 칸 {[VF.FEATURE_NAMES[c] for c, m in enumerate(MISMATCH) if m]}"

    value_cols = list(range(VF.CRANE_ROW_DIM))
    dead = [VF.FEATURE_NAMES[c] for c in value_cols if NONZERO[c] == 0]
    assert not dead, f"항상 0 인 값 칸이 있다 — 그 칸의 대조는 공허하다: {dead}"
    for c in range(VF.CRANE_ROW_DIM, 32):
        assert NONZERO[c] == 0, f"패딩 칸 {VF.FEATURE_NAMES[c]} 에 값이 들어갔다"
    assert NONZERO[VF.IDX["role_crane"]] == total_rows, "역할 칸(crane)이 모든 행에서 1 이어야 한다"
    # 경계 조건 탐침 — 하나라도 0 이면 그 조건이 어느 무대에서도 안 밟혔다는 뜻이다
    probe_sum = {k: sum(r["probes"][k] for r in REPORT.values())
                 for k in ("left", "far", "plan_failed", "busy_wait_only")}
    print()
    print("■ 경계 탐침 합계:", probe_sum)
    assert probe_sum["left"] > 0, "게이트를 이미 나간 트럭이 없었다 — '블록 안 전부' 의 상한 조건이 공허하다"
    assert probe_sum["far"] > 0, "30분 창 밖 통지분이 없었다 — '곧 올 통지분' 의 창 조건이 공허하다"
    assert probe_sum["plan_failed"] > 0, "계획 실패 필수 후보가 없었다 — 계획 칸의 '0' 규칙이 공허하다"
    assert probe_sum["busy_wait_only"] > 0, "작업 중인 크레인이 한 번도 없었다 — 'WAIT 하나뿐' 분기가 공허하다"
    for name in ("role_seller", "role_buyer", "role_state", "buyer_first_row"):
        assert NONZERO[VF.IDX[name]] == 0, f"{name} 칸이 크레인 행에서 0 이 아니다"
