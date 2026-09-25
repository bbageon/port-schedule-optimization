"""배열 계획(gpu/plan.py) 이 v5 `TerminalSimulator._plan` 과 **같은 답**을 내는가 ([[YR-327]] 조각 1).

■ 무엇을 지키나
  ① 명세 §10 무대(10×4×4 · 크레인 1대) 와 그 확장판(blocker 0·1·2·3개 · 규격 섞임 · bay 5 를
     꽉 채워 blocker 가 **다른 bay 로** 가게 함) 에서, v5 를 실제로 굴리며 **여러 시점**
     (초기 · 결정마다 · 배정 직후=예약 칸 제외 중) 에 모든 (크레인, 오더) 계획을 대조한다.
     대조 항목: ok · dur · rehandles · loaded_m · empty_m · corridor(lo,hi) · end · slots ·
     lane · 이동 목록(컨테이너·src·dst·종류) 전부. 실수는 `==` (근사 아님), 다르면 최대 차이 보고.
  ② extra_excluded(dry-run 순차 예약) 를 무작위로 넣어도 같다.
  ③ 무작위 야드(채움 5종) × 무작위 오더 × 무작위 크레인 위치 — t=0 에서 전부 대조.
  ④ (K,N) vmap+jit 판(`plan_serve_all`) 이 낱개 eager 판과 비트까지 같다.
  ⑤ 동률 — v5 find_slot 이 실제로 부른 질의에서 **정확 동률**이 몇 건인지 세어 보고한다
     (동률이 0 이면 tie-break 는 시험되지 않은 것이다).

기대값은 손으로 적지 않는다 — v5 `sim._plan(crane_id, JobRef)` 을 실제로 부른다.
실행: WSL venv · x64 CPU (XLA_FLAGS=--xla_allow_excess_precision=false 는 관례 — FMA 방어는 gpu/exact.py 의 barrier)
"""
from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — plan.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu.geom import Geom  # noqa: E402
from yard_rl.v6.gpu.plan import PlanOut, plan_serve, plan_serve_all  # noqa: E402
from yard_rl.v6.gpu.stack_ops import SIZE_INDEX, from_v5_stacks  # noqa: E402
from yard_rl.v6.gpu.state import (MV_REHANDLE, MV_RETRIEVE, MV_STORE, PK_SERVE,  # noqa: E402
                                  empty_block_world)
from yard_rl.v6.world.contract.schema import CandidateKind  # noqa: E402
from yard_rl.v6.world.contract.state import LaneGraph  # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, JobFlow, LoadStatus  # noqa: E402
from yard_rl.v6.world.domain.models import BlockGeometry, Container, Job  # noqa: E402
from yard_rl.v6.world.integrated import fixtures  # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator  # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec  # noqa: E402
from yard_rl.v6.world.integrated.scenario import TerminalScenario  # noqa: E402

FT20, FT40, FT45 = ContainerSize.FT20, ContainerSize.FT40, ContainerSize.FT45

#: 동률·불일치 집계 (모듈 전체) — 마지막 시험이 보고한다
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── 무대 (명세 §10)
def _profile(n_lanes: int = 1, *, two_cranes: bool = False):
    """명세 §10 프로파일. two_cranes 면 담당 구간·속도가 다른 둘째 크레인을 더한다
    (t=0 계획 대조 전용 — 결정 구동은 크레인 1대로만 한다)."""
    base = fixtures.build_integrated_profile()
    lanes = tuple(f"L{i + 1}" for i in range(n_lanes))
    cranes = (replace(fixtures._spec("YC-A"), service_bay_max=10),)
    if two_cranes:
        cranes += (replace(fixtures._spec("YC-B"), service_bay_min=4, service_bay_max=10,
                           gantry_speed_mps=1.7, trolley_speed_mps=0.8, hoist_speed_loaded_mps=0.45,
                           hoist_speed_empty_mps=0.95, lock_time_s=28.0, unlock_time_s=19.0,
                           truck_positioning_time_s=31.0),)
    return replace(base,
                   block=BlockGeometry("B1", 10, 4, 4, 6.5, 2.9, 2.6, 0),
                   cranes=cranes,
                   lane_graph=LaneGraph(lanes, ()),
                   transfer=TransferFleetSpec("TF1", "YT", n_units=0, move_time_s=180.0))


def _c(cid, bay, row, tier, size=FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _out(jid, target, gate_in, arrival):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, target_container=target, exit_travel_s=60.0)


def _in(jid, gate_in, arrival, size=FT40):
    return Job(job_id=jid, flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, inbound_size=size, inbound_load=LoadStatus.FULL,
               exit_travel_s=60.0)


def _scenario_spec10() -> TerminalScenario:
    """명세 §10 그대로 — 컨테이너 3 · 오더 5 (blocker 0·1개, STORE 2건)."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 5, 1, 2), "C3": _c("C3", 8, 2, 1)}
    jobs = [_out("J-OUT-1", "C1", 0.0, 300.0), _out("J-OUT-2", "C3", 0.0, 300.0),
            _in("J-IN-1", 100.0, 700.0), _out("J-OUT-3", "C2", 900.0, 1500.0),
            _in("J-IN-2", 900.0, 1500.0)]
    return TerminalScenario(scenario_id="p1-spec10", seed=0, horizon_s=7200.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


def _scenario_rich() -> TerminalScenario:
    """확장판 — blocker 0·1·2·3개, FT20/FT40/FT45 섞임, bay 5 를 꽉 채워 blocker 가 다른 bay 로.

    bay 5 row 1 = C1,C2,C4,C5 (아래→위, FT40) · row 2~4 는 꼭대기까지 FT40 (갈 곳 없음)
    bay 4 row 1 = E1 (FT20) → 규격이 달라 blocker 가 못 간다 → (4,1)/(6,1) 동률이 규격 규칙으로 갈림
    bay 2 row 3 = F1,F2 (FT20) → F1 반출은 blocker 1개(FT20)
    """
    containers = {
        "C1": _c("C1", 5, 1, 1), "C2": _c("C2", 5, 1, 2), "C4": _c("C4", 5, 1, 3), "C5": _c("C5", 5, 1, 4),
        "C3": _c("C3", 8, 2, 1),
        "E1": _c("E1", 4, 1, 1, FT20),
        "F1": _c("F1", 2, 3, 1, FT20), "F2": _c("F2", 2, 3, 2, FT20),
    }
    for r in (2, 3, 4):
        for t in (1, 2, 3, 4):
            cid = f"D{r}{t}"
            containers[cid] = _c(cid, 5, r, t)
    jobs = [
        _out("J-OUT-C1", "C1", 0.0, 300.0),      # blocker 3
        _out("J-OUT-C2", "C2", 0.0, 300.0),      # blocker 2
        _out("J-OUT-C4", "C4", 100.0, 700.0),    # blocker 1
        _out("J-OUT-C5", "C5", 300.0, 900.0),    # blocker 0
        _out("J-OUT-C3", "C3", 900.0, 1500.0),   # blocker 0 (다른 bay)
        _out("J-OUT-F1", "F1", 1400.0, 2000.0),  # blocker 1 (FT20)
        _in("J-IN-1", 100.0, 700.0, FT40), _in("J-IN-2", 900.0, 1500.0, FT40),
        _in("J-IN-3", 1900.0, 2500.0, FT20), _in("J-IN-4", 2400.0, 3000.0, FT45),
    ]
    return TerminalScenario(scenario_id="p1-rich", seed=0, horizon_s=7200.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


# ───────────────────────────────────────────────── v5 쪽 도구
def _v5_sorted_keys(stk, geom, size, spec, nb, nr, excl):
    """v5 find_slot 과 같은 식으로 모든 합법 칸의 (cost, bay, row) — 동률 집계 전용."""
    keys = []
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        gc = abs(nb - bay) * geom.bay_length_m
        for row in range(1, geom.row_count + 1):
            if (bay, row) in excl:
                continue
            pile = stk._stacks.get((bay, row))
            top = len(pile) if pile else 0
            if top >= geom.tier_max:
                continue
            if pile and stk.containers[pile[-1]].size != size:
                continue
            keys.append((gc + abs(nr - row) * geom.row_width_m + top * geom.tier_height_m, bay, row))
    keys.sort()
    return keys


class _TieCounter:
    """v5 `YardStacks.find_slot` 을 감싸 실제 질의마다 정확 동률(최선 비용 == 차선 비용)을 센다."""

    def __init__(self):
        self.calls = 0
        self.ties = 0

    @contextmanager
    def watching(self, sim):
        stk, geom = sim.stacks, sim.profile.block
        orig = stk.find_slot

        def wrapped(size, spec, near_bay, near_row, exclude=frozenset()):
            keys = _v5_sorted_keys(stk, geom, size, spec, near_bay, near_row, exclude)
            self.calls += 1
            self.ties += int(len(keys) >= 2 and keys[0][0] == keys[1][0])
            return orig(size, spec, near_bay, near_row, exclude=exclude)

        stk.find_slot = wrapped
        try:
            yield self
        finally:
            del stk.find_slot


def _v5_plan(sim, cid, j, extra=frozenset()):
    """v5 `_jobref` → `_plan` 그대로. 대상이 이미 반출됐으면 v5 는 `_dispatchable` 이 미리 걸러
    `_plan` 에 못 오므로(KeyError) None 으로 두고, 배열판 ok=False 를 기대한다."""
    yc, spec = sim.fleet.get(cid), sim.fleet.spec(cid)
    if j.target_container is not None and j.target_container not in sim.stacks.containers:
        return None
    ref = sim._jobref(j, spec, yc)
    if ref is None:
        return None
    return sim._plan(cid, ref, extra_exclude=frozenset(extra))


# ───────────────────────────────────────────────── 호스트 변환 (시험 전용 최소판)
def _cont_ids(scn: TerminalScenario) -> list[str]:
    """번호 규칙(명세 §1): 초기 컨테이너 = sorted(container_id) 순위, 반입 예비칸 = C0 + n."""
    return sorted(scn.containers) + [f"IN_{jid}" for jid in sorted(j.job_id for j in scn.jobs)]


def _world_from_sim(sim, g: Geom, cont_ids: list[str], job_ids: list[str]):
    """v5 sim 의 **지금** 상태에서 plan 이 읽는 열만 채운 BlockWorld."""
    N, K, C = len(job_ids), len(sim.fleet.ids()), len(cont_ids)
    w = empty_block_world(g, n_orders=N, n_cranes=K, n_conts=C, q_cap=8, log_cap=8, end_s=sim.end)
    stacks, conts, _ = from_v5_stacks(sim.stacks, g, cont_ids=cont_ids, n_cont=C)
    idx = {cid: i for i, cid in enumerate(cont_ids)}

    is_store = np.zeros(N, bool); is_ext = np.zeros(N, bool)
    tgt = np.full(N, -1, np.int32); inb_c = np.full(N, -1, np.int32); inb_s = np.full(N, -1, np.int32)
    for n, jid in enumerate(job_ids):
        j = sim.jobs[jid]
        is_store[n] = j.inbound_size is not None
        is_ext[n] = j.is_external_truck
        tgt[n] = idx[j.target_container] if j.target_container is not None else -1
        inb_c[n] = idx[f"IN_{jid}"]
        inb_s[n] = SIZE_INDEX[j.inbound_size.value] if j.inbound_size is not None else -1
    orders = w.orders._replace(is_store=jnp.asarray(is_store), is_external=jnp.asarray(is_ext),
                               target_cont=jnp.asarray(tgt), inbound_cont=jnp.asarray(inb_c),
                               inbound_size=jnp.asarray(inb_s))

    ids = sim.fleet.ids()
    col = lambda f: jnp.asarray([f(cid) for cid in ids], jnp.float64)  # noqa: E731
    coli = lambda f: jnp.asarray([f(cid) for cid in ids], jnp.int32)   # noqa: E731
    sp = sim.fleet.spec
    cranes = w.cranes._replace(
        bay=col(lambda c: sim.fleet.get(c).state.position_bay),
        row=col(lambda c: sim.fleet.get(c).state.trolley_row),
        bay_min=coli(lambda c: sp(c).service_bay_min), bay_max=coli(lambda c: sp(c).service_bay_max),
        spec_gantry=col(lambda c: sp(c).gantry_speed_mps), spec_trolley=col(lambda c: sp(c).trolley_speed_mps),
        spec_hoist_loaded=col(lambda c: sp(c).hoist_speed_loaded_mps),
        spec_hoist_empty=col(lambda c: sp(c).hoist_speed_empty_mps),
        spec_lock=col(lambda c: sp(c).lock_time_s), spec_unlock=col(lambda c: sp(c).unlock_time_s),
        spec_truck_pos=col(lambda c: sp(c).truck_positioning_time_s))

    active = np.zeros(K, bool)
    slots = np.zeros((K, g.bay_count, g.row_count), bool)
    for k, cid in enumerate(ids):
        r = sim.reservations._by_crane.get(cid)
        if r is not None:
            active[k] = True
            for (b, rr) in r.slots:
                slots[k, b - 1, rr - 1] = True
    res = w.res._replace(active=jnp.asarray(active), slots=jnp.asarray(slots))
    return w._replace(stacks=stacks, conts=conts, orders=orders, cranes=cranes, res=res)


def _excl_mask(g: Geom, cells):
    m = np.zeros((g.bay_count, g.row_count), bool)
    for b, r in cells:
        m[b - 1, r - 1] = True
    return m


# ───────────────────────────────────────────────── 대조
def _take(out_np: PlanOut, k: int, n: int) -> PlanOut:
    return PlanOut(*[x[k, n] for x in out_np])


def _compare_one(plan, out: PlanOut, cont_ids, lane_ids):
    """v5 JobPlan|None ↔ PlanOut 하나 → 불일치 [(항목, v5, v6)]. 실수는 `==` 다."""
    bad = []
    f = lambda name, a, b: (None if a == b else bad.append((name, a, b)))  # noqa: E731
    f("viol", 0, int(out.viol))
    ok = bool(out.ok)
    if (plan is not None) != ok:
        bad.append(("ok", plan is not None, ok))
        return bad
    if plan is None:
        f("n_moves(None)", 0, int(out.n_moves))
        return bad
    f("kind", PK_SERVE, int(out.kind))
    f("dur", plan.duration_s, float(out.dur))
    f("rehandles", plan.rehandles, int(out.rehandles))
    f("loaded_m", plan.loaded_gantry_m, float(out.loaded_m))
    f("empty_m", plan.empty_gantry_m, float(out.empty_m))
    f("lo", float(plan.corridor[0]), float(out.lo))
    f("hi", float(plan.corridor[1]), float(out.hi))
    f("end_bay", float(plan.end_bay), float(out.end_bay))
    f("end_row", float(plan.end_row), float(out.end_row))
    got_slots = {(int(b) + 1, int(r) + 1) for b, r in zip(*np.nonzero(out.slots))}
    f("slots", set(plan.slots), got_slots)
    f("lane", lane_ids.index(plan.lane_id) if plan.lane_id is not None else -1, int(out.lane))
    f("n_moves", len(plan.moves), int(out.n_moves))
    M = out.mv_cont.shape[0]
    for i, m in enumerate(plan.moves):
        c = int(out.mv_cont[i])
        f(f"mv{i}.cont", m.container_id, cont_ids[c] if 0 <= c < len(cont_ids) else None)
        f(f"mv{i}.src", tuple(int(x) for x in m.src), tuple(int(x) for x in out.mv_src[i]))
        f(f"mv{i}.dst", tuple(int(x) for x in m.dst), tuple(int(x) for x in out.mv_dst[i]))
        want = MV_STORE if m.inbound is not None else (MV_RETRIEVE if m.depart else MV_REHANDLE)
        f(f"mv{i}.kind", want, int(out.mv_kind[i]))
    for i in range(len(plan.moves), M):
        f(f"mv{i}.empty", -1, int(out.mv_cont[i]))
    return bad


def _compare_point(sim, g, cont_ids, job_ids, tag, *, extra_cells=(), rec):
    """지금 시점의 모든 (크레인, 오더) 를 v5 ↔ 배열판으로 대조. 불일치를 rec 에 쌓는다."""
    ids = sim.fleet.ids()
    lane_ids = list(sim.profile.lane_graph.lane_ids)
    world = _world_from_sim(sim, g, cont_ids, job_ids)
    extra = jnp.asarray(_excl_mask(g, extra_cells))
    out = jax.device_get(plan_serve_all(world, extra, g))
    n_ok = 0
    for k, cid in enumerate(ids):
        for n, jid in enumerate(job_ids):
            plan = _v5_plan(sim, cid, sim.jobs[jid], frozenset(extra_cells))
            n_ok += plan is not None
            for (name, a, b) in _compare_one(plan, _take(out, k, n), cont_ids, lane_ids):
                rec["bad"].append((tag, cid, jid, name, a, b))
    rec["points"] += 1
    rec["plans"] += len(ids) * len(job_ids)
    rec["plans_ok"] += n_ok
    return world, out


def _drive(sim, on_point):
    """v5 를 결정마다 굴린다 (명세 §10: 후보 첫째 SERVE, 없으면 WAIT). 시점마다 on_point."""
    on_point("t0")
    n_dec = 0
    while (dp := sim.run_until_decision()) is not None:
        n_dec += 1
        on_point(f"decision#{n_dec}@{dp.time:g}")
        acts = []
        for c in dp.crane_ids:
            cands = sim.candidates_for(c)
            acts.append(CraneAssignment(c, CandidateKind.SERVE, cands[0]) if cands
                        else CraneAssignment(c, CandidateKind.WAIT))
        sim.commit_decisions(acts)
        on_point(f"after-commit#{n_dec}@{dp.time:g}")
    return n_dec


def _report(rec):
    bad = rec["bad"]
    if not bad:
        return "불일치 0"
    fl = [abs(a - b) for (_, _, _, _, a, b) in bad
          if isinstance(a, float) and isinstance(b, float)]
    worst = f", 실수 최대차 {max(fl):.3e}" if fl else ""
    head = "\n".join(str(x) for x in bad[:12])
    return f"불일치 {len(bad)}건{worst}\n{head}"


def _new_rec():
    return {"bad": [], "points": 0, "plans": 0, "plans_ok": 0, "calls": 0, "ties": 0}


# ───────────────────────────────────────────────── ① 명세 §10 무대 — 초기·결정마다·배정 직후
@pytest.mark.parametrize("name,make", [("spec10", _scenario_spec10), ("rich", _scenario_rich)])
def test_plan_matches_v5_along_run(name, make):
    """★v5 를 실제로 굴리며 모든 시점·모든 (크레인,오더) 계획이 같다 — STORE·RETRIEVE(blocker 0~3)."""
    prof, scn = _profile(), make()
    g = Geom.from_profile(prof)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    cont_ids, job_ids = _cont_ids(scn), sorted(j.job_id for j in scn.jobs)
    rec = _new_rec()
    ties = _TieCounter()
    rng = random.Random(11)

    def on_point(tag):
        with ties.watching(sim):
            _compare_point(sim, g, cont_ids, job_ids, tag, rec=rec)
            # ② extra_excluded — dry-run 순차 예약을 흉내낸 무작위 칸 1~6개
            cells = {(rng.randint(1, g.bay_count), rng.randint(1, g.row_count))
                     for _ in range(rng.randint(1, 6))}
            _compare_point(sim, g, cont_ids, job_ids, tag + "+extra", extra_cells=cells, rec=rec)

    n_dec = _drive(sim, on_point)
    rec["calls"], rec["ties"] = ties.calls, ties.ties
    REPORT[name] = rec
    print(f"\n[{name}] 결정 {n_dec} · 시점 {rec['points']} · 계획 {rec['plans']} (성립 {rec['plans_ok']}) · "
          f"v5 find_slot 질의 {rec['calls']} (정확동률 {rec['ties']}) · {_report(rec)}")
    assert n_dec >= 4, "결정이 너무 적다 — 무대가 의도대로 굴러가지 않았다"
    assert sim.terminal and all(j.status.value == "DONE" for j in sim.jobs.values()), "v5 가 끝까지 못 갔다"
    assert not rec["bad"], f"[{name}] {_report(rec)}"


def test_rich_stage_actually_covers_blockers():
    """확장판 무대가 blocker 0·1·2·3 개와 다른 bay 로 가는 재조작(loaded gantry > 0) 을 정말 담는가."""
    prof, scn = _profile(), _scenario_rich()
    g = Geom.from_profile(prof)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    cont_ids, job_ids = _cont_ids(scn), sorted(j.job_id for j in scn.jobs)
    world = _world_from_sim(sim, g, cont_ids, job_ids)
    out = jax.device_get(plan_serve_all(world, jnp.zeros((g.bay_count, g.row_count), bool), g))
    by_job = {jid: _take(out, 0, n) for n, jid in enumerate(job_ids)}
    reh = {jid: int(p.rehandles) for jid, p in by_job.items() if not bool(world.orders.is_store[job_ids.index(jid)])}
    assert {reh["J-OUT-C1"], reh["J-OUT-C2"], reh["J-OUT-C4"], reh["J-OUT-C5"]} == {3, 2, 1, 0}
    assert reh["J-OUT-F1"] == 1 and reh["J-OUT-C3"] == 0
    assert float(by_job["J-OUT-C1"].loaded_m) > 0.0, "blocker 가 같은 bay 안에서만 움직였다 — crane_travel 식이 죽는다"
    assert all(bool(p.ok) for p in by_job.values())
    assert all(int(p.viol) == 0 for p in by_job.values())
    # STORE 는 규격별로 다른 칸 — FT45 는 빈 바닥으로
    assert int(by_job["J-IN-4"].n_moves) == 1 and int(by_job["J-IN-4"].mv_kind[0]) == MV_STORE
    assert out.dur.dtype == np.float64 and out.lo.dtype == np.float64


# ───────────────────────────────────────────────── ③ 무작위 야드 × 무작위 오더 (t=0)
def _random_scenario(rng: random.Random, geom: BlockGeometry, fill: float):
    """적재 규칙(같은 규격·tier 연속)을 지켜 무작위로 채운 야드 + 반출 8·반입 4 오더."""
    piles: dict[tuple[int, int], list[Container]] = {}
    target = int(geom.bay_count * geom.row_count * geom.tier_max * fill)
    made = 0
    for _ in range(target * 3):
        if made >= target:
            break
        bay, row = rng.randint(1, geom.bay_count), rng.randint(1, geom.row_count)
        size = rng.choice(list(ContainerSize))
        pile = piles.setdefault((bay, row), [])
        if len(pile) >= geom.tier_max or (pile and pile[-1].size != size):
            continue
        made += 1
        pile.append(_c(f"R{made:04d}", bay, row, len(pile) + 1, size))
    conts = {c.container_id: c for pile in piles.values() for c in pile}
    ids = sorted(conts)
    jobs = []
    for i in range(min(8, len(ids))):
        jobs.append(_out(f"J-OUT-{i:02d}", rng.choice(ids), 0.0, 300.0 + 100.0 * i))
    for i in range(4):
        jobs.append(_in(f"J-IN-{i:02d}", 0.0, 400.0 + 100.0 * i, rng.choice(list(ContainerSize))))
    return TerminalScenario(scenario_id=f"rnd-{fill}", seed=0, horizon_s=7200.0, drain_window_s=0.0,
                            containers=conts, jobs=jobs, vessels=[], injected_events=[])


@pytest.mark.parametrize("fill", [0.0, 0.30, 0.45, 0.75, 0.95])
def test_plan_matches_v5_random_yards(fill):
    """채움 5종 × 시드 4 × 크레인 2대(담당 구간·스펙 다름) — 크레인 위치(연속 좌표 포함)·
    제외 칸도 무작위. 전부 v5 와 같다 (크레인 축 vmap 도 여기서 대조된다)."""
    prof = _profile(n_lanes=2, two_cranes=True)   # 레인 2개 — (bay−1)%2 규칙도 본다
    geom = prof.block
    g = Geom.from_profile(prof)
    rec = _new_rec()
    ties = _TieCounter()
    n_none = 0
    for seed in range(4):
        rng = random.Random(500 + seed + int(fill * 100))
        scn = _random_scenario(rng, geom, fill)
        sim = TerminalSimulator(prof, scn, check_invariants=True)
        for cid in sim.fleet.ids():
            yc, sp = sim.fleet.get(cid), sim.fleet.spec(cid)
            yc.state.position_bay = (rng.uniform(float(sp.service_bay_min), float(sp.service_bay_max))
                                     if rng.random() < 0.5
                                     else float(rng.randint(sp.service_bay_min, sp.service_bay_max)))
            yc.state.trolley_row = float(rng.choice([0, 0, rng.randint(1, geom.row_count)]))
        cont_ids, job_ids = _cont_ids(scn), sorted(j.job_id for j in scn.jobs)
        cells = {(rng.randint(1, geom.bay_count), rng.randint(1, geom.row_count))
                 for _ in range(rng.randint(0, 6))}
        with ties.watching(sim):
            _compare_point(sim, g, cont_ids, job_ids, f"seed{seed}", rec=rec)
            _compare_point(sim, g, cont_ids, job_ids, f"seed{seed}+extra", extra_cells=cells, rec=rec)
        n_none += sum(_v5_plan(sim, c, sim.jobs[j]) is None
                      for c in sim.fleet.ids() for j in job_ids)
    rec["calls"], rec["ties"] = ties.calls, ties.ties
    REPORT[f"random-{fill}"] = rec
    print(f"\n[random fill={fill}] 계획 {rec['plans']} (성립 {rec['plans_ok']}, v5 None {n_none}) · "
          f"find_slot 질의 {rec['calls']} (정확동률 {rec['ties']}) · {_report(rec)}")
    assert not rec["bad"], f"[random fill={fill}] {_report(rec)}"


# ───────────────────────────────────────────────── ④ (K,N) jit 판 == 낱개 eager 판
def test_batched_jit_equals_eager():
    prof, scn = _profile(), _scenario_rich()
    g = Geom.from_profile(prof)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    cont_ids, job_ids = _cont_ids(scn), sorted(j.job_id for j in scn.jobs)
    world = _world_from_sim(sim, g, cont_ids, job_ids)
    extra = jnp.asarray(_excl_mask(g, {(6, 1), (4, 2)}))
    batched = jax.jit(plan_serve_all, static_argnames="g")
    out_b = jax.device_get(batched(world, extra, g=g))
    bad = []
    for n in range(len(job_ids)):
        e = jax.device_get(plan_serve(world, jnp.int32(0), jnp.int32(n), extra, g))
        b = _take(out_b, 0, n)
        for name, x, y in zip(PlanOut._fields, e, b):
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                bad.append((job_ids[n], name, x, y))
    assert not bad, f"jit+vmap ≠ eager: {bad[:5]}"
    assert out_b.slots.shape == (1, len(job_ids), g.bay_count, g.row_count)
    assert out_b.mv_src.shape == (1, len(job_ids), g.n_moves, 3)


# ───────────────────────────────────────────────── ⑤ 동률 보고
def test_ties_were_exercised():
    """정확 동률이 실제로 시험됐는지 — 무대 두 개와 무작위 야드의 합이 0 이면 tie-break 는 미검증이다."""
    total_ties = sum(r["ties"] for r in REPORT.values())
    total_calls = sum(r["calls"] for r in REPORT.values())
    summary = {k: (v["plans"], v["calls"], v["ties"], len(v["bad"])) for k, v in REPORT.items()}
    print(f"\n[동률 집계] (계획 수, find_slot 질의, 정확동률, 불일치) = {summary}")
    assert REPORT, "앞 시험이 돌지 않았다"
    assert total_calls > 0 and total_ties > 0, f"동률 0/{total_calls} — tie-break 가 시험되지 않았다"
