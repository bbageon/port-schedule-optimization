"""확장 후보(gpu/cands3.py) 가 v5 `CandidateGenerator.generate` 와 **같은 답**을 내는가 ([[YR-327]] 조각 3 · key=cands3).

■ 무엇을 지키나 — v5 를 CentralResolver(선호 규칙) 로 완주시키며 **결정마다** 생성기를 가로채 대조한다
  결정 시점마다 v5 진행 중 상태를 배열 세계로 옮기고(test_gpu_escape.world_from_sim + armed) `candidates3` 를 부른다:
    ① 결정 개방 — `_decision_cranes` 가 연 크레인 집합 == open (정상 결정; armed 는 소진 **전** 값을 가로챈다 297행)
       · `eta_opportunity` (전 크레인) == eta_opp · `interference_deadlock_corridors()` 비어 있지 않음 == deadlock
    ② 원시 목록 (prune 전) — 유휴·비양보 크레인마다 `_serve`·`_pre_rehandle`·`_reposition` 가 만든 후보의
       (종류, 오더 | 목표 bay) 집합 == raw, 후보마다 feasible·거절 코드·mandatory·score(`==`)·계획
       (dur·rehandles·corridor·end·lane·slots·loaded/empty·moves 전부 `==`). 비-eligible 크레인은 WAIT 뿐.
    ③ 목표 bay 집합 — `_future_target_bays ∪ _escape_bays` == repo_valid 의 bay, `_escape_bays` == repo_escape
    ④ prune·순서 — `generate().items` 의 (후보 → candidate_id) == prune 의 keep·candidate_id, WAIT 의 id == n_kept
    ⑤ DEFER 재료 — `_defer_trigger_time`·DEFER_ALL 의 `_wait` == defer_wait (LEGACY 는 안 쓰지만 같은 값)
  정보수준 둘: PRE_ADVICE (PRE·ETA 주도 REPO·wake) 와 BLOCK_ARRIVAL (PRE 0 · release 주도 REPO 만).
  선호 둘: BaselinePreference · ServiceFirstSPTPreference (정답 궤적 Y01 의 규칙).

■ 무대 (K=2 · 10×4×4 · YC-A/YC-B 1..10 · 레인 L1,L2 · 결정 지평 1800)
  eta-basic      blocker 있는 반출 대상 + ETA (wake 0) · ETA 있는 반출/반입 · 본선연계 release → PRE·ETA REPO·release REPO
  crowded-eta    트럭 10대 동시 도착(재조작 3단 4대 → SLA 0.8 초과 mandatory) + PRE 4 + ETA 반입 2 + 본선 2 → 원시 후보 > 11 → prune
  dead-first-eta 첫 도착이 두 크레인 사각지대 → 교착 탈출 결정 → `_escape_bays` 목표 (REPO 로 풀린다)
  eta-random-s*  무작위 야드·오더 (ETA 75%·반입 ETA 50%·본선연계 1~2) 6 시드 · spt-s* 2 시드 · block-arrival-s* 2 시드

기대값은 손으로 적지 않는다 — v5 를 같은 규칙으로 실제로 굴려 얻는다. 실행: WSL venv · x64 CPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
from dataclasses import fields, replace
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu import cands3 as C3                                             # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import IdTables, to_block_world                    # noqa: E402
from yard_rl.v6.gpu.reserve import REASON_TO_CODE                                   # noqa: E402
from yard_rl.v6.gpu.state import (EMPTY_ID, MV_REHANDLE, MV_RETRIEVE, MV_STORE, PK_PRE_REHANDLE,   # noqa: E402
                                  PK_REPOSITION, PK_SERVE, PK_WAIT)
# v5 정본
from yard_rl.v6.world.contract.schema import CandidateKind                          # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, InformationLevel, JobFlow, LoadStatus   # noqa: E402
from yard_rl.v6.world.domain.models import Job                                      # noqa: E402
from yard_rl.v6.world.integrated.baselines import ServiceFirstSPTPreference        # noqa: E402
from yard_rl.v6.world.integrated.candidates import CandidateGenerator, eta_opportunity   # noqa: E402
from yard_rl.v6.world.integrated.engine import TerminalSimulator                    # noqa: E402
from yard_rl.v6.world.integrated.policy_config import LEGACY_DEFAULT                # noqa: E402
from yard_rl.v6.world.integrated.resolver import BaselinePreference, CentralResolver   # noqa: E402
from yard_rl.v6.world.integrated.scenario import TerminalScenario                   # noqa: E402

FT20, FT40, FT45 = ContainerSize.FT20, ContainerSize.FT40, ContainerSize.FT45
PA, BA = InformationLevel.PRE_ADVICE, InformationLevel.BLOCK_ARRIVAL
KIND_NAME = {PK_SERVE: "SERVE", PK_PRE_REHANDLE: "PRE_REHANDLE", PK_REPOSITION: "REPOSITION", PK_WAIT: "WAIT"}
MV_NAME = {MV_REHANDLE: "REHANDLE", MV_RETRIEVE: "RETRIEVE", MV_STORE: "STORE"}
PLAN_FAILED = -1
#: 시험 전체 집계 (마지막 시험이 보고)
REPORT: dict[str, dict] = {}
#: 첫 무대의 배열 세계 (jit/vmap 시험 재사용) — label → [(world, horizon)]
WORLDS: dict[str, list] = {}


def _load(name: str, fname: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ESC = _load("_tge_for_cands3", "test_gpu_escape.py")      # world_from_sim · prof_k2 · dead_first_scenario
_EQ = _ESC._EQ                                                # _caps · _c · _out · _in
prof_k2, dead_first_scenario = _ESC.prof_k2, _ESC.dead_first_scenario
_caps, _c = _EQ._caps, _EQ._c


class _JobIndex(dict):
    """`world_from_sim` 이 `jidx[st.assigned_job]` 로 읽는 오더 번호표 — REPOSITION 실행 중 크레인은 v5 가
    `assigned_job = "REPO:<cid>:<bay>"` (engine.py:699) 라 오더가 없다 → BUSY_NO_ORDER (cands3 머리말 제안)."""

    def __missing__(self, key):
        if isinstance(key, str) and key.startswith("REPO:"):
            return C3.BUSY_NO_ORDER
        raise KeyError(key)


class _Tables(IdTables):
    @property
    def job_index(self):
        return _JobIndex(super().job_index)


def world_from_sim(sim, w0, tb, g):
    """test_gpu_escape.world_from_sim + REPO 실행 중 크레인 처리 (번호표만 감싼다)."""
    return _ESC.world_from_sim(sim, w0, _Tables(*[getattr(tb, f.name) for f in fields(tb)]), g)


# ───────────────────────────────────────────────── 무대 빌더
def _out_eta(jid, target, gate_in, arrival, eta, exit_s=None):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, provided_eta=eta, target_container=target, exit_travel_s=exit_s)


def _in_eta(jid, gate_in, arrival, eta, size=FT40, exit_s=None):
    return Job(job_id=jid, flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, provided_eta=eta, inbound_size=size, inbound_load=LoadStatus.FULL,
               exit_travel_s=exit_s)


def _vl(jid, target, release, deadline):
    return Job(job_id=jid, flow=JobFlow.VESSEL_LOAD, release_time=release, actual_gate_in=None,
               actual_block_arrival=None, target_container=target, deadline=deadline)


def _scn(sid, containers, jobs, horizon=3600.0, drain=0.0):
    return TerminalScenario(scenario_id=sid, seed=0, horizon_s=horizon, drain_window_s=drain,
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


def eta_basic_scenario():
    """C-T(5,1,1) 위 blocker C-B → J-OUT-T(ETA 900·도착 1500) 는 wake 0 에서 PRE. J-OUT-Y(ETA 1900) · J-IN-1(ETA 2400)
    은 ETA 주도 REPO 목표, J-VL-1(release 700·마감 3000) 은 release 주도 REPO 목표 + 본선 마감 score 항."""
    containers = {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2), "C-X": _c("C-X", 2, 1, 1),
                  "C-Y": _c("C-Y", 9, 1, 1), "C-Z": _c("C-Z", 9, 2, 1)}
    jobs = [_out_eta("J-OUT-T", "C-T", 900.0, 1500.0, 900.0), _out_eta("J-OUT-X", "C-X", 0.0, 300.0, None),
            _out_eta("J-OUT-Y", "C-Y", 1400.0, 2000.0, 1900.0), _in_eta("J-IN-1", 2000.0, 2500.0, 2400.0),
            _vl("J-VL-1", "C-Z", 700.0, 3000.0)]
    return _scn("eta-basic", containers, jobs)


def crowded_eta_scenario():
    """트럭 10대가 300초에 동시 도착 (bay 7..10 은 blocker 3단 → 뒤쪽은 대기 > 1440 = mandatory) + PRE 대상 4 (bay 2..5 row 2,
    ETA 1600..1900 → wake 0) + ETA 반입 2 (ETA 1000) + 본선연계 2 (release 600/1000 · 마감 5000/600) → 원시 후보 > 11."""
    containers = {}
    jobs = []
    for b in range(1, 7):
        containers[f"S{b}"] = _c(f"S{b}", b, 1, 1)
        jobs.append(_out_eta(f"J-S{b}", f"S{b}", 0.0, 300.0, None, 60.0))
    for b in range(7, 11):
        containers[f"H{b}"] = _c(f"H{b}", b, 1, 1)
        for t in (2, 3, 4):
            containers[f"H{b}B{t}"] = _c(f"H{b}B{t}", b, 1, t)
        jobs.append(_out_eta(f"J-H{b}", f"H{b}", 0.0, 300.0, None, 60.0))
    for i, b in enumerate(range(2, 6)):
        containers[f"P{b}"] = _c(f"P{b}", b, 2, 1)
        containers[f"P{b}B"] = _c(f"P{b}B", b, 2, 2)
        jobs.append(_out_eta(f"J-P{b}", f"P{b}", 1000.0, 2100.0 + 100.0 * i, 1600.0 + 100.0 * i, 60.0))
    jobs += [_in_eta("J-IN-1", 800.0, 1200.0, 1000.0, FT40, 60.0), _in_eta("J-IN-2", 800.0, 1250.0, 1000.0, FT20, 60.0)]
    containers["V9"] = _c("V9", 9, 3, 1)
    containers["V10"] = _c("V10", 10, 3, 1)
    jobs += [_vl("J-VL-1", "V9", 600.0, 5000.0), _vl("J-VL-2", "V10", 1000.0, 600.0)]
    return _scn("crowded-eta", containers, jobs, horizon=7200.0)


def random_eta_scenario(seed: int, B: int = 10, R: int = 4, T: int = 4) -> TerminalScenario:
    """무작위 야드(35%)·반출 3~6 (ETA 75%: 도착 + U(−300,600))·반입 1~3 (ETA 50%)·본선연계 1~2 (release U(100,3000))."""
    rng = random.Random(7000 + seed)
    containers = {}
    n = 0
    for bay in range(1, B + 1):
        for row in range(1, R + 1):
            if rng.random() < 0.35:
                size = rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0]
                for t in range(1, rng.randint(1, T) + 1):
                    containers[f"C{n:03d}"] = _c(f"C{n:03d}", bay, row, t, size)
                    n += 1
    ex = 60.0 if rng.random() < 0.5 else None
    pool = sorted(containers)
    targets = rng.sample(pool, min(len(pool), rng.randint(3, 6)))
    jobs = []
    for i, tgt in enumerate(targets):
        arr = round(rng.uniform(0.0, 2500.0), 3)
        eta = round(arr + rng.uniform(-300.0, 600.0), 3) if rng.random() < 0.75 else None
        jobs.append(_out_eta(f"J-OUT-{i:02d}", tgt, max(0.0, arr - rng.uniform(0.0, 600.0)), arr, eta, ex))
    for i in range(rng.randint(1, 3)):
        arr = round(rng.uniform(0.0, 2500.0), 3)
        eta = round(arr + rng.uniform(-300.0, 600.0), 3) if rng.random() < 0.5 else None
        jobs.append(_in_eta(f"J-IN-{i:02d}", max(0.0, arr - rng.uniform(0.0, 600.0)), arr, eta,
                            rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0], ex))
    rest = [c for c in pool if c not in targets]
    for i in range(rng.randint(1, 2)):
        if not rest:
            break
        tgt = rest.pop(rng.randrange(len(rest)))
        jobs.append(_vl(f"J-VL-{i:02d}", tgt, float(rng.randint(100, 3000)), rng.choice([600.0, 5000.0, None])))
    return TerminalScenario(scenario_id=f"eta-random-{seed}", seed=seed, horizon_s=rng.choice([3600.0, 7200.0]),
                            drain_window_s=rng.choice([0.0, 600.0]), containers=containers, jobs=jobs,
                            vessels=[], injected_events=[])


def prof_stepped(gap: float, a=(1, 5), b=(5, 10)):
    """계단식 담당구간 (YC-A a · YC-B b) — 그룹 크기 1 이라 초기 위치 = service_bay_min."""
    from yard_rl.v6.world.integrated import fixtures
    base = prof_k2(gap)
    return replace(base, cranes=(replace(fixtures._spec("YC-A"), service_bay_min=a[0], service_bay_max=a[1]),
                                 replace(fixtures._spec("YC-B"), service_bay_min=b[0], service_bay_max=b[1])))


def plan_failed_mandatory_scenario():
    """mandatory 인데 계획이 실패하는 SERVE (feasible=False · PLAN_FAILED · 목록에는 오른다, candidates.py:280-284).

    야드는 (5,1) 만 비고 나머지는 FT40 4단 만재, (4,1) 은 대상 T + blocker. 트럭 셋(A-OUT 반출 · B-IN FT40 반입 · C-OUT
    (9,1) 꼭대기 반출)이 2초에 도착하지만 두 크레인이 1초부터 고장 → YC-A 만 1450 에 복구 → 결정: A-OUT·B-IN 둘 다
    대기 1448 ≥ 1440 = mandatory, job_id 순으로 A-OUT 배정 (blocker 목적지 = (5,1) 예약). YC-B 가 1500 에 복구 → C-OUT
    이 feasible 이라 결정이 열리고, B-IN 은 dispatchable(제외 없는 find_slot 은 (5,1)) 이지만 계획은 (5,1) 이 예약돼 실패
    → PLAN_FAILED mandatory 후보가 목록에 오른다 (v5 는 SERVE 후보가 하나도 없으면 결정 자체를 안 연다 — 그래서 C-OUT).
    gap 0 이라 YC-B 정지 위치 5 가 YC-A 통로 [1,5] 를 막지 않는다. A 완료 뒤 (5,1) 의 TB(FT40) 위에 B-IN 이 실린다."""
    from yard_rl.v6.world.integrated.scenario import InjectedEvent
    containers = {}
    n = 0
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
    jobs = [_out_eta("A-OUT", "T", 0.0, 2.0, None), _in_eta("B-IN", 0.0, 2.0, None, FT40),
            _out_eta("C-OUT", "G4", 0.0, 2.0, None)]
    inj = [InjectedEvent(1.0, "EQUIPMENT_DOWN", "YC-A"), InjectedEvent(1.0, "EQUIPMENT_DOWN", "YC-B"),
           InjectedEvent(1450.0, "EQUIPMENT_UP", "YC-A"), InjectedEvent(1500.0, "EQUIPMENT_UP", "YC-B")]
    return TerminalScenario(scenario_id="plan-failed-mandatory", seed=0, horizon_s=3600.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=inj)


PREFS = {"baseline": BaselinePreference, "spt": ServiceFirstSPTPreference}
#: label → () → (profile, scenario, info_level, 선호 이름)
STAGES = {
    "eta-basic": lambda: (prof_k2(2.0), eta_basic_scenario(), PA, "baseline"),
    "crowded-eta": lambda: (prof_k2(2.0), crowded_eta_scenario(), PA, "baseline"),
    "dead-first-eta": lambda: (prof_k2(3.0), dead_first_scenario(), PA, "baseline"),
    "plan-failed-mandatory": lambda: (prof_stepped(0.0), plan_failed_mandatory_scenario(), PA, "baseline"),
}
for _s in range(1, 7):
    STAGES[f"eta-random-s{_s}"] = (lambda s: lambda: (prof_k2(2.0 if s % 2 else 3.0), random_eta_scenario(s), PA, "baseline"))(_s)
for _s in (7, 8):
    STAGES[f"spt-s{_s}"] = (lambda s: lambda: (prof_k2(2.0), random_eta_scenario(s), PA, "spt"))(_s)
for _s in (9, 10):
    STAGES[f"block-arrival-s{_s}"] = (lambda s: lambda: (prof_k2(2.0), random_eta_scenario(s), BA, "baseline"))(_s)


# ───────────────────────────────────────────────── v5 쪽 — 가로채기 구동
class Rec(NamedTuple):
    """결정 하나의 v5 기록 (생성기·개방 술어를 그 시점 상태에서 부른 값)."""

    pre: object          # 배열 세계 (world_from_sim + armed 소진 전 값)
    t: float
    open_ids: tuple      # TerminalDecision.crane_ids
    escape: bool         # 탈출로 열린 결정인가
    eligible: dict       # crane_id → idle & ~yielded
    eta_opp: dict        # crane_id → eta_opportunity
    cors: tuple          # interference_deadlock_corridors()
    raw: dict            # crane_id → {key: (feasible, code, mandatory, score, plan_tuple|None)} (eligible 만)
    targets: dict        # crane_id → (future bays set, escape bays set) (eligible 만)
    items: dict          # crane_id → {key: candidate_id} (generate 전 크레인) + ("WAIT",) → id
    defer: tuple         # (_defer_trigger_time, DEFER_ALL _wait.defer_until, trigger)
    chosen: tuple        # ((crane, action, token), …)


def _key(gc):
    if gc.kind == CandidateKind.WAIT:
        return ("WAIT",)
    if gc.kind == CandidateKind.REPOSITION:
        return ("REPOSITION", float(gc.job_ref.reposition_target_bay))
    return (gc.kind.value, gc.job_ref.job_id)


def _v5_plan(plan, lane_ids):
    if plan is None:
        return None
    moves = tuple((m.container_id, tuple(m.src), tuple(m.dst),
                   "STORE" if m.inbound is not None else ("RETRIEVE" if m.depart else "REHANDLE")) for m in plan.moves)
    lane = lane_ids.index(plan.lane_id) if plan.lane_id is not None else EMPTY_ID
    return (plan.duration_s, plan.rehandles, plan.corridor[0], plan.corridor[1], plan.end_bay, plan.end_row, lane,
            frozenset(plan.slots), plan.loaded_gantry_m, plan.empty_gantry_m, moves)


def _v5_code(gc):
    return PLAN_FAILED if gc.mask_reason == "PLAN_FAILED" else REASON_TO_CODE[gc.mask_reason]


def run_v5_hooked(prof, scn, level, pref_name, w0, tb, g):
    """v5 완주 (CentralResolver + generate) — 결정마다 Rec. armed 는 `_decision_cranes` 호출 시점(소진 전)을 가로챈다."""
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    gen_defer = CandidateGenerator(config=LEGACY_DEFAULT.with_changes(name="DEFER_ALL", wait_mode="DEFER_ALL"))
    resolver = CentralResolver(PREFS[pref_name]())
    snap = {"armed": set()}
    orig_dc = sim._decision_cranes

    def hooked_dc():
        snap["armed"] = set(sim._eta_armed)
        return orig_dc()

    sim._decision_cranes = hooked_dc
    recs: list[Rec] = []
    lane_ids = list(tb.lane_ids)
    while (dp := sim.run_until_decision()) is not None:
        now = sim.now
        escape = bool(sim.event_log) and sim.event_log[-1][1] == "DEADLOCK_ESCAPE" and sim.event_log[-1][0] == now
        pre = world_from_sim(sim, w0, tb, g)
        armed = jnp.asarray([cid in snap["armed"] for cid in tb.crane_ids])
        pre = pre._replace(wake=pre.wake._replace(eta_armed=armed))
        eligible, eta_opp, raw, targets, items = {}, {}, {}, {}, {}
        gens = {}
        for cid in sim.fleet.ids():
            yc = sim.fleet.get(cid)
            eligible[cid] = bool(yc.idle and not yc.yielded)
            eta_opp[cid] = bool(eta_opportunity(sim, cid, level))
            gens[cid] = gen.generate(sim, cid, level)
            items[cid] = {_key(gc): gc.candidate_id for gc in gens[cid].items}
            if eligible[cid]:
                lst = (gen._serve(sim, cid, now) + gen._pre_rehandle(sim, cid, now, level)
                       + gen._reposition(sim, cid, now, level))
                raw[cid] = {_key(gc): (gc.feasible, _v5_code(gc), gc.mandatory, gc.score, _v5_plan(gc.plan, lane_ids))
                            for gc in lst}
                assert len(raw[cid]) == len(lst), f"v5 원시 목록에 같은 키가 둘 ({cid} t={now})"
                targets[cid] = (set(gen._future_target_bays(sim, cid, now, level)), set(gen._escape_bays(sim, cid)))
        cors = sim.interference_deadlock_corridors()
        dt = gen_defer._defer_trigger_time(sim, now, level)
        dw = gen_defer._wait(sim, now, level)
        gen_by = {c: gens[c] for c in dp.crane_ids}
        resn = resolver.resolve(sim, dp, gen_by)
        resolver.apply(sim, resn, gen_by)
        chosen = tuple((r.crane_id, r.action.value, r.chosen_token) for r in resn.resolutions)
        recs.append(Rec(pre, now, tuple(dp.crane_ids), escape, eligible, eta_opp, cors, raw, targets, items,
                        (dt, dw.defer_until, dw.defer_trigger, dw.defer_trigger_jid), chosen))
    return sim, recs


# ───────────────────────────────────────────────── 배열 쪽
_JIT_CACHE: dict = {}


def all3_jit(g: Geom, pre_advice: bool):
    key = (g, pre_advice)
    if key not in _JIT_CACHE:
        def f(w, hz):
            out = C3.candidates3(w, g, horizon_s=hz, pre_advice=pre_advice)
            fl = C3.flat_view(out)
            pr = C3.prune(fl, g)
            df = C3.defer_wait(w, g, pre_advice)
            return out, fl, pr, df
        _JIT_CACHE[key] = jax.jit(f)
    return _JIT_CACHE[key]


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _arr_plan(out, fl, k, col, N, tb):
    """배열 계획 → v5 계획 tuple 과 같은 모양."""
    if col < N:
        P = out.m.P if fl.kind[k, col] == PK_SERVE else out.P_pre
        n = col
    else:
        P, n = out.P_repo, col - N
    slots = frozenset((int(b) + 1, int(r) + 1) for b, r in np.argwhere(P.slots[k, n]))
    moves = tuple((tb.cont_ids[int(P.mv_cont[k, n, i])], tuple(int(v) for v in P.mv_src[k, n, i]),
                   tuple(int(v) for v in P.mv_dst[k, n, i]), MV_NAME[int(P.mv_kind[k, n, i])])
                  for i in range(int(P.n_moves[k, n])))
    return (float(P.dur[k, n]), int(P.rehandles[k, n]), float(P.lo[k, n]), float(P.hi[k, n]), float(P.end_bay[k, n]),
            float(P.end_row[k, n]), int(P.lane[k, n]), slots, float(P.loaded_m[k, n]), float(P.empty_m[k, n]), moves)


def _arr_key(fl, k, col, N, tb):
    kind = int(fl.kind[k, col])
    if kind == PK_WAIT:
        return ("WAIT",)
    if kind == PK_REPOSITION:
        return ("REPOSITION", float(fl.bay[k, col]))
    return (KIND_NAME[kind], tb.job_ids[int(fl.job[k, col])])


def check_record(label, i, rec: Rec, tb, g, horizon, pre_advice) -> dict:
    K, n0 = len(tb.crane_ids), tb.n0
    N = rec.pre.n
    tag = f"[{label} #{i} t={rec.t:.3f}{' ESC' if rec.escape else ''}]"
    out_j, fl_j, pr_j, df_j = all3_jit(g, pre_advice)(rec.pre, horizon)
    out, fl, pr, df = _np(out_j), _np(fl_j), _np(pr_j), _np(df_j)
    C = fl.raw.shape[1]
    wait_col = C - 1
    assert not np.any(out.serve_raw & out.pre_raw), f"{tag} 한 오더가 SERVE 이면서 PRE"
    assert not bool(out.esc_overflow)

    # ① 개방 술어
    got_el = {cid: bool(out.eligible[k]) for k, cid in enumerate(tb.crane_ids)}
    assert got_el == rec.eligible, f"{tag} ① eligible arr={got_el} v5={rec.eligible}"
    got_opp = {cid: bool(out.eta_opp[k]) for k, cid in enumerate(tb.crane_ids)}
    assert got_opp == rec.eta_opp, f"{tag} ① eta_opportunity arr={got_opp} v5={rec.eta_opp}"
    assert bool(out.deadlock) == bool(rec.cors), f"{tag} ① deadlock arr={bool(out.deadlock)} v5={rec.cors}"
    if not rec.escape:
        got_open = tuple(cid for k, cid in enumerate(tb.crane_ids) if bool(out.open[k]))
        assert got_open == rec.open_ids, f"{tag} ① open arr={got_open} v5={rec.open_ids} (armed={np.asarray(rec.pre.wake.eta_armed)})"
    else:   # 탈출 결정 = 유휴 전원 (399행) — 개방 술어가 아니라 idle 마스크
        got_idle = tuple(cid for k, cid in enumerate(tb.crane_ids) if bool(out.m.idle[k]))
        assert got_idle == rec.open_ids, f"{tag} ① 탈출 결정 대상 arr={got_idle} v5={rec.open_ids}"

    flags = dict(escape=rec.escape, deadlock=bool(rec.cors), pre_raw=0, pre_feas=0, repo_raw=0, repo_feas=0,
                 esc_targets=0, mandatory=0, pruned=0, raw_max=0, plan_failed=0, code_hist={})
    for k, cid in enumerate(tb.crane_ids):
        # ② 원시 목록 (eligible 크레인) · 비-eligible 은 WAIT 뿐
        cols = [c for c in range(C) if fl.raw[k, c] and c != wait_col]
        assert bool(fl.raw[k, wait_col]) and bool(fl.feasible[k, wait_col]) and int(fl.kind[k, wait_col]) == PK_WAIT
        if not rec.eligible[cid]:
            assert not cols, f"{tag} ② {cid} 비-eligible 인데 후보 {cols}"
            assert rec.items[cid] == {("WAIT",): 0}
            assert int(pr.n_kept[k]) == 0 and int(pr.candidate_id[k, wait_col]) == 0
            continue
        got = {}
        for c in cols:
            key = _arr_key(fl, k, c, N, tb)
            assert key not in got, f"{tag} ② {cid} 배열 원시 목록에 같은 키 둘 {key}"
            plan = _arr_plan(out, fl, k, c, N, tb) if bool(fl.plan_ok[k, c]) else None
            got[key] = (bool(fl.feasible[k, c]), int(fl.code[k, c]), bool(fl.mandatory[k, c]), float(fl.score[k, c]), plan)
        exp = rec.raw[cid]
        if set(got) != set(exp):
            pytest.fail(f"{tag} ② {cid} 원시 후보 집합이 다르다:\n  arr 만={sorted(set(got) - set(exp), key=str)}\n"
                        f"  v5 만={sorted(set(exp) - set(got), key=str)}")
        for key in sorted(exp, key=str):
            a, b = got[key], exp[key]
            for j, name in enumerate(("feasible", "code", "mandatory", "score")):
                assert a[j] == b[j], f"{tag} ② {cid} {key} {name} arr={a[j]!r} v5={b[j]!r}"
            if a[4] is None or b[4] is None:
                assert a[4] is None and b[4] is None, f"{tag} ② {cid} {key} 계획 유무 arr={a[4] is not None} v5={b[4] is not None}"
            else:
                names = ("dur", "rehandles", "lo", "hi", "end_bay", "end_row", "lane", "slots", "loaded_m", "empty_m", "moves")
                for j, name in enumerate(names):
                    assert a[4][j] == b[4][j], f"{tag} ② {cid} {key} 계획.{name} arr={a[4][j]!r} v5={b[4][j]!r}"
            flags["code_hist"][a[1]] = flags["code_hist"].get(a[1], 0) + 1
            flags["plan_failed"] += int(a[1] == PLAN_FAILED)
            flags["mandatory"] += int(a[2])
        # ③ 목표 bay 집합
        fut, esc = rec.targets[cid]
        got_t = {float(out.repo_bay[k, r]) for r in range(out.repo_bay.shape[1]) if out.repo_valid[k, r]}
        got_e = {float(out.repo_bay[k, r]) for r in range(out.repo_bay.shape[1]) if out.repo_escape[k, r]}
        assert got_t == (fut | esc), f"{tag} ③ {cid} 목표 bay arr={sorted(got_t)} v5={sorted(fut | esc)}"
        assert got_e == esc, f"{tag} ③ {cid} 탈출 목표 arr={sorted(got_e)} v5={sorted(esc)}"
        # ④ prune · candidate_id
        got_items = {_arr_key(fl, k, c, N, tb): int(pr.candidate_id[k, c]) for c in range(C) if pr.keep[k, c]}
        assert got_items == rec.items[cid], f"{tag} ④ {cid} generate items arr={got_items} v5={rec.items[cid]}"
        assert int(pr.n_kept[k]) == len(rec.items[cid]) - 1
        n_raw = len(cols)
        flags["raw_max"] = max(flags["raw_max"], n_raw)
        flags["pruned"] += int(n_raw > int(pr.n_kept[k]))
        kinds = fl.kind[k]
        flags["pre_raw"] += int(np.sum(fl.raw[k] & (kinds == PK_PRE_REHANDLE)))
        flags["pre_feas"] += int(np.sum(fl.feasible[k] & (kinds == PK_PRE_REHANDLE)))
        flags["repo_raw"] += int(np.sum(fl.raw[k] & (kinds == PK_REPOSITION)))
        flags["repo_feas"] += int(np.sum(fl.feasible[k] & (kinds == PK_REPOSITION)))
        flags["esc_targets"] += len(esc)
    # ⑤ DEFER 재료
    dt, until, trig, tjid = rec.defer
    t5 = None if dt[0] is None else float(dt[0])
    got5 = None if not np.isfinite(df.trigger_s) else float(df.trigger_s)
    assert got5 == t5, f"{tag} ⑤ defer trigger arr={got5} v5={dt}"
    assert float(df.defer_until) == until, f"{tag} ⑤ defer_until arr={float(df.defer_until)} v5={until}"
    exp_kind = {None: -1, "ETA": 0, "RELEASE": 1}[trig]
    assert int(df.trigger_kind) == exp_kind
    exp_job = EMPTY_ID if tjid is None else tb.job_index[tjid]
    assert int(df.trigger_job) == exp_job, f"{tag} ⑤ trigger job arr={int(df.trigger_job)} v5={tjid}"
    return flags


def run_stage(label: str):
    prof, scn, level, pref = STAGES[label]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    horizon = float(prof.decision_horizon_s)
    pre_advice = level == PA
    sim, recs = run_v5_hooked(prof, scn, level, pref, w0, tb, g)
    assert recs, f"[{label}] 결정이 한 번도 안 열렸다"
    flags = [check_record(label, i, r, tb, g, horizon, pre_advice) for i, r in enumerate(recs)]
    WORLDS.setdefault(label, [(r.pre, horizon, pre_advice) for r in recs[:3]])
    chosen = {}
    for r in recs:
        for (_c, act, _t) in r.chosen:
            chosen[act] = chosen.get(act, 0) + 1
    code_hist: dict = {}
    for f in flags:
        for c, n in f["code_hist"].items():
            code_hist[c] = code_hist.get(c, 0) + n
    tot = lambda key: sum(f[key] for f in flags)
    REPORT[label] = dict(
        level=level.value, pref=pref, K=len(tb.crane_ids), decisions=len(recs), escapes=tot("escape"),
        deadlock=tot("deadlock"), pre_raw=tot("pre_raw"), pre_feas=tot("pre_feas"), repo_raw=tot("repo_raw"),
        repo_feas=tot("repo_feas"), esc_targets=tot("esc_targets"), mandatory=tot("mandatory"), pruned=tot("pruned"),
        raw_max=max(f["raw_max"] for f in flags), plan_failed=tot("plan_failed"), code_hist=code_hist, chosen=chosen,
        eta_wakes=sum(1 for (_, k, _) in sim.event_log if k == "ETA_WAKE"),
        backlog=sim.unfinished_backlog(), events=len(sim.event_log), hash=sim.event_stream_hash())
    return sim, recs, tb, g


# ───────────────────────────────────────────────── 시험
@pytest.mark.parametrize("label", sorted(STAGES))
def test_stage_equivalence(label):
    sim, recs, tb, g = run_stage(label)
    r = REPORT[label]
    if label.startswith("block-arrival"):
        assert r["pre_raw"] == 0 and r["eta_wakes"] == 0, "BLOCK_ARRIVAL 에서 PRE·wake 가 나왔다 (누출)"
        assert all(not any(rec.eta_opp.values()) for rec in recs)
    if label == "eta-basic":
        assert r["pre_feas"] >= 1 and r["eta_wakes"] >= 2 and r["chosen"].get("PRE_REHANDLE", 0) >= 1
        assert r["repo_feas"] >= 1
    if label == "crowded-eta":
        assert r["pruned"] >= 1 and r["raw_max"] > C3.K_MAX - 1, f"prune 이 실제로 자르지 않았다 raw_max={r['raw_max']}"
        assert r["mandatory"] >= 1, "SLA 0.8 초과 mandatory 후보가 없었다"
    if label == "dead-first-eta":
        assert r["escapes"] >= 1 and r["esc_targets"] >= 1 and r["chosen"].get("REPOSITION", 0) >= 1
        assert sim.unfinished_backlog() == 0, "REPO 로 교착이 풀려 전부 처리돼야 한다"
    if label == "plan-failed-mandatory":
        assert r["plan_failed"] >= 1 and r["mandatory"] >= 2, f"PLAN_FAILED mandatory 경로가 안 열렸다 {r}"
        assert sim.unfinished_backlog() == 0 and sim.jobs["B-IN"].status.name == "DONE"


#: 정답 궤적 (scripts/v6/dump_ground_truth.py run_block · 블록 Y01 · 본선 240건 · 크레인 2 · 사건 1,318)
_TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "reports", "yr327_v6_port",
                      "ground_truth", "block_Y01_load30_seed9900777.json")


@pytest.mark.skipif(not os.path.exists(_TRUTH), reason="정답 궤적 파일 없음")
def test_y01_ground_truth_candidates_match_every_decision():
    """정답 궤적 Y01 을 만든 것과 **같은 절차**(dump_ground_truth.run_block: ResolverPolicy(ServiceFirstSPT) +
    CandidateGenerator(LEGACY_DEFAULT) + baselines._apply) 로 v5 를 굴리며, 결정마다 `generate().items` 전부
    (candidate_id·feasible·mandatory·score·계획) 가 candidates3+flat_view+prune 와 같은지 본다. 구동기가 정답과 같은
    절차임은 끝의 사건 해시로 확인한다. (조각 3 통합 전이라 결정·사건 진행은 v5 가 하고 배열은 후보만 낸다.)"""
    from yard_rl.v6.world.integrated import baselines as bl, engine as eng
    D = _load("_dgt_for_cands3", os.path.join("..", "..", "scripts", "v6", "dump_ground_truth.py"))
    truth = json.load(open(_TRUTH, encoding="utf-8"))
    prof, built = D._build(truth["load"], truth["seed"])
    scn = built["scenarios"][truth["block"]]
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    sim = D._sim(prof, scn)
    level = sim.info_level
    assert level == PA
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    pol = bl.ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    snap = {"armed": set()}
    orig_dc = sim._decision_cranes

    def hooked_dc():
        snap["armed"] = set(sim._eta_armed)
        return orig_dc()

    sim._decision_cranes = hooked_dc
    lane_ids = list(tb.lane_ids)
    f3 = all3_jit(g, True)
    n_dec = n_items = 0
    stats = dict(pruned=0, raw_max=0, repo_feas=0, pre_raw=0, chosen={})
    while (out := sim.run_until_decision()) is not None:
        if isinstance(out, eng.ReviewEpoch):
            continue
        dp = out
        n_dec += 1
        now = sim.now
        escape = bool(sim.event_log) and sim.event_log[-1][1] == "DEADLOCK_ESCAPE" and sim.event_log[-1][0] == now
        pre = world_from_sim(sim, w0, tb, g)
        pre = pre._replace(wake=pre.wake._replace(eta_armed=jnp.asarray([c in snap["armed"] for c in tb.crane_ids])))
        o3, fl, pr, _ = (_np(x) for x in f3(pre, hz))
        gens = {c: gen.generate(sim, c, level) for c in sim.fleet.ids()}
        N, C = pre.n, fl.raw.shape[1]
        tag = f"[Y01 #{n_dec} t={now:.3f}]"
        if not escape:
            got_open = tuple(c for k, c in enumerate(tb.crane_ids) if bool(o3.open[k]))
            assert got_open == tuple(dp.crane_ids), f"{tag} open arr={got_open} v5={dp.crane_ids}"
        for k, c in enumerate(tb.crane_ids):
            exp = {_key(gc): (gc.candidate_id, gc.feasible, gc.mandatory, gc.score, _v5_plan(gc.plan, lane_ids))
                   for gc in gens[c].items}
            got = {}
            for col in range(C):
                if not pr.keep[k, col]:
                    continue
                key = _arr_key(fl, k, col, N, tb)
                if key == ("WAIT",):
                    got[key] = (int(pr.candidate_id[k, col]), True, False, float("-inf"), None)
                    continue
                plan = _arr_plan(o3, fl, k, col, N, tb) if bool(fl.plan_ok[k, col]) else None
                got[key] = (int(pr.candidate_id[k, col]), bool(fl.feasible[k, col]), bool(fl.mandatory[k, col]),
                            float(fl.score[k, col]), plan)
            assert set(got) == set(exp), (f"{tag} {c} 후보 집합: arr 만={sorted(set(got) - set(exp), key=str)} "
                                          f"v5 만={sorted(set(exp) - set(got), key=str)}")
            for key in exp:
                assert got[key] == exp[key], f"{tag} {c} {key}\n  arr={got[key]}\n  v5 ={exp[key]}"
            n_items += len(exp)
            n_raw = int(np.sum(fl.raw[k])) - 1
            stats["raw_max"] = max(stats["raw_max"], n_raw)
            stats["pruned"] += int(n_raw > int(pr.n_kept[k]))
            stats["repo_feas"] += int(np.sum(fl.feasible[k] & (fl.kind[k] == PK_REPOSITION)))
            stats["pre_raw"] += int(np.sum(fl.raw[k] & (fl.kind[k] == PK_PRE_REHANDLE)))
        gb = {c: gens[c] for c in dp.crane_ids}
        assign = pol.decide(sim, dp, gb)
        for gc in assign.values():
            stats["chosen"][gc.kind.value] = stats["chosen"].get(gc.kind.value, 0) + 1
        bl._apply(sim, assign)
    h = sim.event_stream_hash()
    assert n_dec == truth["n_decisions"] and len(sim.event_log) == truth["n_events"]
    assert h == truth["event_hash"], f"시험 구동기가 정답 절차와 다르다: {h} != {truth['event_hash']}"
    assert stats["pruned"] >= 1 and stats["repo_feas"] >= 1
    REPORT["Y01-truth"] = dict(level="PRE_ADVICE", pref="spt", K=len(tb.crane_ids), decisions=n_dec, escapes=0, deadlock=0,
                               pre_raw=stats["pre_raw"], pre_feas=0, repo_raw=0, repo_feas=stats["repo_feas"], esc_targets=0,
                               mandatory=0, pruned=stats["pruned"], raw_max=stats["raw_max"], plan_failed=0, code_hist={},
                               chosen=stats["chosen"], eta_wakes=sum(1 for (_, k, _) in sim.event_log if k == "ETA_WAKE"),
                               backlog=sim.unfinished_backlog(), events=len(sim.event_log), hash=h, items=n_items)


def test_jit_eager_vmap_agree():
    """같은 세계에서 eager == jit == vmap (잎 전부 비트) — 앞 무대의 기록 세계 셋."""
    if "eta-basic" not in WORLDS:
        run_stage("eta-basic")
    worlds = WORLDS["eta-basic"]
    assert len(worlds) >= 2
    prof = prof_k2(2.0)
    g = Geom.from_profile(prof)
    hz = worlds[0][1]

    def f(w):
        out = C3.candidates3(w, g, horizon_s=hz, pre_advice=True)
        fl = C3.flat_view(out)
        return out, fl, C3.prune(fl, g)

    eager = [f(w) for (w, _, _) in worlds]
    jit_f = jax.jit(f)
    jitted = [jit_f(w) for (w, _, _) in worlds]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[w for (w, _, _) in worlds])
    vm = jax.jit(jax.vmap(f))(batched)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(eager[0])]
    for i in range(len(worlds)):
        le = jax.tree_util.tree_leaves(eager[i])
        lj = jax.tree_util.tree_leaves(jitted[i])
        lb = [np.asarray(l)[i] for l in jax.tree_util.tree_leaves(vm)]
        bad_j = [names[j] for j, (a, b) in enumerate(zip(le, lj)) if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)]
        bad_b = [names[j] for j, (a, b) in enumerate(zip(le, lb)) if not np.array_equal(np.asarray(a), b, equal_nan=True)]
        assert not bad_j and not bad_b, f"세계 {i}: jit 다름 {bad_j} · vmap 다름 {bad_b}"


def test_n_esc_smaller_than_n_flags_overflow_only_when_needed():
    """n_esc < N: 막힌 오더가 E 개를 넘으면 esc_overflow, 아니면 같은 목표 집합 (dead-first 교착 세계)."""
    if "dead-first-eta" not in WORLDS:
        run_stage("dead-first-eta")
    prof = prof_k2(3.0)
    g = Geom.from_profile(prof)
    for (w, hz, pa) in WORLDS["dead-first-eta"]:
        full = C3.candidates3(w, g, horizon_s=hz, pre_advice=pa)
        small = C3.candidates3(w, g, horizon_s=hz, pre_advice=pa, n_esc=1)
        assert small.repo_bay.shape[1] == C3.n_repo_cols(g, w.n, 1)
        full_t = {(k, float(b)) for k in range(w.k) for b, v in zip(np.asarray(full.repo_bay[k]), np.asarray(full.repo_valid[k])) if v}
        small_t = {(k, float(b)) for k in range(w.k) for b, v in zip(np.asarray(small.repo_bay[k]), np.asarray(small.repo_valid[k])) if v}
        has = np.any(np.asarray(full.blocked), axis=0)                 # 오더별 통로 (blocked_corridors 의 has)
        if bool(full.deadlock) and bool(has[1:].any()):                # E=1 칸 뒤에도 막힌 오더가 있다
            assert bool(small.esc_overflow)
        else:
            assert not bool(small.esc_overflow) and small_t == full_t
    with pytest.raises(ValueError):
        w = WORLDS["dead-first-eta"][0][0]
        C3.candidates3(w, g, horizon_s=1800.0, pre_advice=True, n_esc=w.n + 1)


def test_zz_report(capsys):
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    tot = lambda key: sum(r[key] for r in REPORT.values())
    with capsys.disabled():
        print("\n[cands3 equiv report]  dec=결정 수 · esc=탈출 결정 · pre=PRE 원시/feasible · repo=REPO 원시/feasible · "
              "escT=탈출 목표 · mand=mandatory · pruned=잘린 결정-크레인 · rawmax=원시 최대 · pf=PLAN_FAILED")
        for k, r in REPORT.items():
            print(f"  {k:18s} {r['level'][:3]} {r['pref']:8s} dec={r['decisions']:3d} esc={r['escapes']} "
                  f"pre={r['pre_raw']}/{r['pre_feas']} repo={r['repo_raw']}/{r['repo_feas']} escT={r['esc_targets']} "
                  f"mand={r['mandatory']} pruned={r['pruned']} rawmax={r['raw_max']:2d} pf={r['plan_failed']} "
                  f"codes={r['code_hist']} chosen={r['chosen']} wakes={r['eta_wakes']} backlog={r['backlog']}")
        print(f"  stages={len(REPORT)} · decisions={tot('decisions')} · pre_feas={tot('pre_feas')} · repo_feas={tot('repo_feas')}"
              f" · esc_targets={tot('esc_targets')} · mandatory={tot('mandatory')} · pruned={tot('pruned')} · plan_failed={tot('plan_failed')}")
    assert tot("pre_feas") >= 5 and tot("repo_feas") >= 5
    assert tot("esc_targets") >= 1, "탈출 목표 bay 가 한 번도 안 나왔다"
    assert tot("mandatory") >= 1 and tot("pruned") >= 1
    assert tot("plan_failed") >= 1, "mandatory 인데 계획 실패(PLAN_FAILED) 후보가 시험되지 않았다"
    if "Y01-truth" in REPORT:
        assert REPORT["Y01-truth"]["items"] >= 2000
    assert sum(r["chosen"].get("PRE_REHANDLE", 0) for r in REPORT.values()) >= 1
    assert sum(r["chosen"].get("REPOSITION", 0) for r in REPORT.values()) >= 1
    codes = {}
    for r in REPORT.values():
        for c, n in r["code_hist"].items():
            codes[c] = codes.get(c, 0) + n
    assert 4 in codes and 2 in codes, f"CRANE_INTERFERENCE·DUP_JOB 거절이 후보 목록에 실린 적이 없다 {codes}"
    assert any(r["level"] == "BLOCK_ARRIVAL" for r in REPORT.values())
    assert any(r["pref"] == "spt" for r in REPORT.values())
