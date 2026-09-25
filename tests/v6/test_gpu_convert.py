"""호스트 변환기(gpu/host_convert.py) 가 v5 `reset()` 직후 상태를 **그대로** 만드는가 ([[YR-327]] 조각 1 §1).

■ 무엇을 지켜야 하나
  ① 번호 규칙 — 오더·컨테이너·크레인·선박이 v5 `sorted` 순위와 같다 (명세 §10 의 표를 포함)
  ② 격자·컨테이너 좌표 — v5 `YardStacks` 와 칸별 일치
  ③ 크레인 초기 위치(등간격 분산)·trolley_row·idle 장벽·레일 순서 — v5 fleet/reservations 와 일치
  ④ 사건 큐 시드 — v5 힙을 꺼내는 순서로 (time, priority, kind, payload) 가 같고 seq 순서도 같다
  ⑤ 장부 등록(gate_in_s) — v5 TimeLedger.records 와 같다 (장부 없는 시나리오는 전부 +inf)
  ⑥ 입력 검증 — v5 가 거부하는 입력은 변환기도 같은 예외로 거부한다
  ⑦ 로그 역변환 — 배열 (t,kind,target) → v5 'crane:job' 등 문자열, 해시식이 v5 와 같다
  ⑧ from_block_world 가 reset 직후 v5 객체 값과 같은 파이썬 값을 돌려준다

기대값은 손으로 적지 않는다 — **v5 `TerminalSimulator` 를 실제로 reset** 해 읽는다.
x64 CPU 에서 돌린다 (`XLA_FLAGS=--xla_allow_excess_precision=false` 는 관례 — FMA 를 막지는 못한다, gpu/exact.py).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import host_convert as HC                                       # noqa: E402
from yard_rl.v6.gpu.events import PRIO, push_event                                  # noqa: E402
from yard_rl.v6.gpu.host_convert import (IdTables, event_log_from_arrays,           # noqa: E402
                                         event_stream_hash, from_block_world,
                                         queue_entries, to_block_world)
from yard_rl.v6.gpu.state import (EV_BLOCK_ARRIVAL, EV_HORIZON, EV_JOB_COMPLETED,  # noqa: E402
                                  EV_JOB_RELEASED, EV_VESSEL_START, LOG_DEADLOCK_ESCAPE,
                                  LOG_DEFER_WAKE, LOG_DISPATCH, LOG_ETA_WAKE, SZ_FT40)
# v5 정본
from yard_rl.v6.world.contract.state import LaneGraph                               # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, JobFlow, LoadStatus        # noqa: E402
from yard_rl.v6.world.domain.models import BlockGeometry, Container, Job           # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                    # noqa: E402
from yard_rl.v6.world.integrated.engine import TerminalSimulator                    # noqa: E402
from yard_rl.v6.world.integrated.events import _PRIORITY, EventKind                 # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec                   # noqa: E402
from yard_rl.v6.world.integrated.scenario import TerminalScenario                   # noqa: E402
from yard_rl.v6.world.sim.constraints import ConstraintViolation                    # noqa: E402


# ───────────────────────────────────────────────── 시험 입력 (명세 §10)
def piece1_profile():
    """크레인 1대 · 10×4×4 · 레인 1개 · 이송 0대 (SLA 1800·지평 1800·gap 2.0 은 fixture 값)."""
    base = fixtures.build_integrated_profile()
    return dataclasses.replace(
        base,
        block=BlockGeometry("B1", 10, 4, 4, 6.5, 2.9, 2.6, 0),
        cranes=(dataclasses.replace(fixtures._spec("YC-A"), service_bay_max=10),),
        lane_graph=LaneGraph(("L1",), ()),
        transfer=TransferFleetSpec("TF1", "YT", n_units=0, move_time_s=180.0))


def _c(cid, bay, row, tier):
    return Container(container_id=cid, size=ContainerSize.FT40, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _out(jid, gate_in, arrival, target):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, target_container=target, exit_travel_s=60.0)


def _in(jid, gate_in, arrival):
    return Job(job_id=jid, flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, inbound_size=ContainerSize.FT40,
               inbound_load=LoadStatus.FULL, exit_travel_s=60.0)


def piece1_scenario() -> TerminalScenario:
    """§10: 컨테이너 3 (C2 가 C1 위 blocker) · 반출 3 · 반입 2 · 선박·주입 없음 · 지평 7200·drain 0."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 5, 1, 2), "C3": _c("C3", 8, 2, 1)}
    jobs = [_out("J-OUT-1", 0.0, 300.0, "C1"), _out("J-OUT-2", 0.0, 300.0, "C3"),
            _in("J-IN-1", 100.0, 700.0), _out("J-OUT-3", 900.0, 1500.0, "C2"),
            _in("J-IN-2", 900.0, 1500.0)]
    return TerminalScenario(scenario_id="piece1", seed=0, horizon_s=7200.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


CASES = {
    "piece1": (piece1_profile, piece1_scenario, dict(n_max=8, q_cap=32, log_cap=128)),
    "fixture": (fixtures.build_integrated_profile, fixtures.build_minimal_terminal_scenario,
                dict(n_max=16, q_cap=64, log_cap=256)),
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request):
    make_prof, make_scn, caps = CASES[request.param]
    prof, scn = make_prof(), make_scn()
    sim = TerminalSimulator(prof, scn, check_invariants=True)      # v5 reset 이 여기서 돈다
    world, tables = to_block_world(prof, scn, **caps)
    return request.param, prof, scn, sim, world, tables


# ───────────────────────────────────────────────── ① 번호 규칙
def test_numbering_follows_v5_sorted(case):
    name, prof, scn, sim, w, tb = case
    assert tb.job_ids == tuple(sorted(j.job_id for j in scn.jobs))
    assert tb.cont_ids[:tb.c0] == tuple(sorted(scn.containers)) == tuple(sorted(sim.stacks.containers))
    assert tb.crane_ids == sim.fleet.ids()
    assert tb.vessel_ids == tuple(sorted(sim.vessels))
    assert tb.lane_ids == prof.lane_graph.lane_ids
    # 반입 예비칸 = C0 + n, 이름은 v5 가 런타임에 만드는 IN_{job_id}
    for n, jid in enumerate(tb.job_ids):
        assert tb.cont_ids[tb.c0 + n] == f"IN_{jid}"
        j = sim.jobs[jid]
        ic = int(w.orders.inbound_cont[n])
        if j.inbound_size is not None:
            assert ic == tb.c0 + n
            assert int(w.conts.c_size[ic]) == HC.SIZE_INDEX[j.inbound_size.value]
            assert not bool(w.conts.c_alive[ic])
        else:
            assert ic == -1
    # 빈 오더 칸은 block=-1
    assert all(int(b) == -1 for b in np.asarray(w.orders.block)[tb.n0:])
    assert all(int(b) == 0 for b in np.asarray(w.orders.block)[:tb.n0])


def test_piece1_spec_table():
    """명세 §10 이 손으로 적어 둔 정렬 결과 — 오더 0..4 · 컨테이너 0..2 · 반입 3,4 · 시드 순서."""
    prof, scn = piece1_profile(), piece1_scenario()
    w, tb = to_block_world(prof, scn, n_max=8, q_cap=32, log_cap=128)
    assert tb.job_ids == ("J-IN-1", "J-IN-2", "J-OUT-1", "J-OUT-2", "J-OUT-3")
    assert tb.cont_ids[:3] == ("C1", "C2", "C3") and tb.c0 == 3
    assert int(w.orders.inbound_cont[0]) == 3 and int(w.orders.inbound_cont[1]) == 4
    seeds = [(t, k, p) for (t, _, _, k, p) in queue_entries(w.queue, tb)]
    assert seeds == [(300.0, "BLOCK_ARRIVAL", "J-OUT-1"), (300.0, "BLOCK_ARRIVAL", "J-OUT-2"),
                     (700.0, "BLOCK_ARRIVAL", "J-IN-1"), (1500.0, "BLOCK_ARRIVAL", "J-IN-2"),
                     (1500.0, "BLOCK_ARRIVAL", "J-OUT-3"), (7200.0, "HORIZON", "HORIZON")]
    # 넣은 순서(seq)는 오더 정렬순: J-IN-1(700) J-IN-2(1500) J-OUT-1 J-OUT-2 J-OUT-3 HORIZON
    by_seq = sorted(queue_entries(w.queue, tb), key=lambda e: e[2])
    assert [e[4] for e in by_seq] == ["J-IN-1", "J-IN-2", "J-OUT-1", "J-OUT-2", "J-OUT-3", "HORIZON"]
    assert tb.ledger_mode and int(w.orders.inbound_size[0]) == SZ_FT40


# ───────────────────────────────────────────────── ② 격자·컨테이너
def test_grid_matches_v5_stacks(case):
    name, prof, scn, sim, w, tb = case
    d = from_block_world(w, tb)
    assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}
    assert d["containers"] == {cid: (c.bay, c.row, c.tier) for cid, c in sim.stacks.containers.items()}
    B, R, T = w.stacks.shape
    assert (B, R, T) == (prof.block.bay_count, prof.block.row_count, prof.block.tier_max)
    height = np.asarray(w.stacks.height)
    for (bay, row), pile in sim.stacks._stacks.items():
        assert int(height[bay - 1, row - 1]) == len(pile)
        if pile:
            top = sim.stacks.containers[pile[-1]].size.value
            assert int(w.stacks.top_size[bay - 1, row - 1]) == HC.SIZE_INDEX[top]
    assert int(height.sum()) == len(sim.stacks.containers)


# ───────────────────────────────────────────────── ③ 크레인
def test_cranes_match_v5_fleet(case):
    name, prof, scn, sim, w, tb = case
    d = from_block_world(w, tb)
    for k, cid in enumerate(tb.crane_ids):
        yc, spec = sim.fleet.get(cid), sim.fleet.spec(cid)
        got = d["cranes"][cid]
        assert got["position_bay"] == yc.state.position_bay          # 비트 동일 (파이썬 float 산식)
        assert got["trolley_row"] == yc.state.trolley_row
        assert got["available_at"] == yc.state.available_at == 0.0
        assert got["assigned_job"] is None and got["status"] == "IDLE"
        assert (got["service_bay_min"], got["service_bay_max"]) == (spec.service_bay_min, spec.service_bay_max)
        assert not got["down"] and not got["yielded"] and not got["is_loaded"]
        assert got["served_count"] == got["recent_completions"] == 0
        assert got["loaded_travel_m"] == got["empty_travel_m"] == 0.0
        for arr, attr in (("spec_gantry", "gantry_speed_mps"), ("spec_trolley", "trolley_speed_mps"),
                          ("spec_hoist_loaded", "hoist_speed_loaded_mps"),
                          ("spec_hoist_empty", "hoist_speed_empty_mps"), ("spec_lock", "lock_time_s"),
                          ("spec_unlock", "unlock_time_s"), ("spec_truck_pos", "truck_positioning_time_s")):
            assert float(getattr(w.cranes, arr)[k]) == getattr(spec, attr)
    assert d["idle_positions"] == sim.reservations.idle_positions()
    assert d["rail_order"] == sim._rail_order
    assert d["reservations"] == {}
    if name == "fixture":   # 같은 구간 2대 → 등간격 분산이 실제로 일어났나
        assert d["cranes"]["YC-A"]["position_bay"] != d["cranes"]["YC-B"]["position_bay"]


# ───────────────────────────────────────────────── ④ 큐 시드
def _v5_heap_sorted(sim):
    return sorted(sim.queue._heap, key=lambda e: (e.time, e.priority, e.seq))


def test_queue_seed_matches_v5_heap(case):
    name, prof, scn, sim, w, tb = case
    got = queue_entries(w.queue, tb)
    want = _v5_heap_sorted(sim)
    assert len(got) == len(want) == int(w.queue.counter)
    assert int(w.queue.overflow) == 0
    for g, e in zip(got, want):
        assert (g[0], g[1], g[3], g[4]) == (e.time, e.priority, e.kind_name, e.payload)
        assert g[1] == _PRIORITY[EventKind[g[3]]] == int(PRIO[EventKind[g[3]]])
    # seq 는 v5 가 1 부터, 배열이 0 부터 — 순서(상대값)가 같아야 한다
    assert [g[2] for g in got] == [e.seq - 1 for e in want]
    assert w.queue.time.dtype == jnp.float64
    if name == "fixture":   # 양하(STORE·본선연계) job 은 시드에서 빠진다 (engine.py:233-234)
        names = {g[4] for g in got}
        assert "J-VES-D0" not in names and "J-VES-D1" not in names
        assert {g[3] for g in got} >= {"BLOCK_ARRIVAL", "JOB_RELEASED", "VESSEL_START",
                                       "EQUIPMENT_DOWN", "EQUIPMENT_UP", "PLAN_CHANGE", "HORIZON"}


def test_seed_queue_equals_sequential_push(case):
    """numpy 로 한 번에 채운 큐가 `push_event` 를 차례로 부른 것과 **배열까지** 같다."""
    name, prof, scn, sim, w, tb = case
    q = HC.empty_queue(w.queue.capacity)
    for e in sorted(sim.queue._heap, key=lambda e: e.seq):
        kind = int(EventKind[e.kind_name])
        what = HC._TARGET_KIND[kind]
        tgt = {"crane": tb.crane_index.get, "job": tb.job_index.get,
               "vessel": tb.vessel_index.get}.get(what, lambda p, d=-1: -1)(e.payload, -1)
        q = push_event(q, e.time, kind, tgt)
    for f in q._fields:
        assert np.array_equal(np.asarray(getattr(q, f)), np.asarray(getattr(w.queue, f))), f


# ───────────────────────────────────────────────── ⑤ 장부·시각·나머지 열
def test_orders_match_v5_jobs(case):
    name, prof, scn, sim, w, tb = case
    d = from_block_world(w, tb)
    assert tb.ledger_mode == (sim.time_ledger is not None)
    for n, jid in enumerate(tb.job_ids):
        j = sim.jobs[jid]
        got = d["jobs"][jid]
        assert got["status"] == j.status.name == "PLANNED"
        assert got["assigned_crane"] is None and got["service_start"] is None
        assert got["rehandle_count"] == 0 and got["wait_sample"] is None
        assert not got["waiting"] and not got["in_block"]
        assert HC.FLOW_NAMES[int(w.orders.flow[n])] == j.flow.name
        assert bool(w.orders.is_external[n]) == j.is_external_truck
        assert bool(w.orders.is_vessel[n]) == j.is_vessel_linked
        assert bool(w.orders.is_store[n]) == (j.inbound_size is not None)
        assert float(w.orders.release_s[n]) == j.release_time
        assert float(w.orders.actual_arrival_s[n]) == (math.inf if j.actual_block_arrival is None else j.actual_block_arrival)
        assert float(w.orders.deadline_s[n]) == (math.inf if j.deadline is None else j.deadline)
        assert float(w.orders.exit_travel_s[n]) == (-1.0 if j.exit_travel_s is None else j.exit_travel_s)
        tc = int(w.orders.target_cont[n])
        assert (tb.cont_ids[tc] if tc >= 0 else None) == j.target_container
        v = int(w.orders.vessel[n])
        assert (tb.vessel_ids[v] if v >= 0 else None) == j.vessel_id
        # 장부 등록: 외부트럭·exit_travel 있음 → A = actual_gate_in or 0.0, 아니면 +inf
        if sim.time_ledger is not None and jid in sim.time_ledger.records:
            assert got["gate_in"] == sim.time_ledger.records[jid].gate_in
        else:
            assert got["gate_in"] is None
    assert d["clock"] == sim.clock == 0.0 and d["end"] == sim.end == scn.end_time
    assert d["terminal"] is False and d["last_decision_at"] is None
    assert d["cost_rate"] == {t: sim.cost._rate[t] for t in d["cost_rate"]}
    assert all(v == 0.0 for v in d["cost_episode"].values()) and sim.cost.episode_raw() == d["cost_episode"]
    assert d["ledger"]["closed_end_s"] is None and d["lane_cong_area_s"] == sim.lanes.cong_area_s == 0.0
    assert d["violation"] == 0 and d["overflow"] == 0 and d["steps"] == 0
    assert d["event_log"] == sim.event_log == []
    # 레인 인접표 = v5 LaneNetwork._adj
    adj = np.asarray(w.lane.adj)
    for i, a in enumerate(tb.lane_ids):
        for j2, b in enumerate(tb.lane_ids):
            assert bool(adj[i, j2]) == (b in sim.lanes.neighbors(a))


def test_dtypes_are_x64(case):
    name, prof, scn, sim, w, tb = case
    for arr in (w.clock, w.end_s, w.orders.gate_in_s, w.orders.release_s, w.cranes.bay,
                w.cranes.spec_gantry, w.res.idle_pos, w.queue.time, w.cost.episode):
        assert arr.dtype == jnp.float64, arr
    for arr in (w.orders.status, w.orders.target_cont, w.cranes.rail_order, w.stacks.grid,
                w.queue.kind, w.queue.seq):
        assert arr.dtype == jnp.int32, arr


# ───────────────────────────────────────────────── ⑥ 입력 검증 = v5 함수 그대로
@pytest.mark.parametrize("bad", ["floating", "mixed_size", "unmatched", "out_of_range"])
def test_validate_rejects_like_v5(bad):
    scn = piece1_scenario()
    if bad == "floating":
        scn.containers["C9"] = _c("C9", 3, 3, 2)                       # 아래 빔
    elif bad == "mixed_size":
        scn.containers["C9"] = dataclasses.replace(_c("C9", 5, 1, 3), size=ContainerSize.FT20)
    elif bad == "unmatched":
        scn.jobs.append(_out("J-X", 0.0, 10.0, "NOPE"))
    else:
        scn.containers["C9"] = _c("C9", 11, 1, 1)
    prof = piece1_profile()
    with pytest.raises(ConstraintViolation) as v5_err:
        TerminalSimulator(prof, scn)
    with pytest.raises(ConstraintViolation) as v6_err:
        to_block_world(prof, scn, n_max=8, q_cap=32, log_cap=128)
    assert v6_err.value.args == v5_err.value.args


def test_capacity_is_loud():
    prof, scn = piece1_profile(), piece1_scenario()
    with pytest.raises(ValueError):
        to_block_world(prof, scn, n_max=4, q_cap=32, log_cap=128)      # 오더 5 > 4
    with pytest.raises(ValueError):
        to_block_world(prof, scn, n_max=8, q_cap=5, log_cap=128)       # 시드 6 > 5


# ───────────────────────────────────────────────── ⑦ 로그 역변환
def test_event_log_reverse_matches_v5_format():
    prof, scn = piece1_profile(), piece1_scenario()
    sim = TerminalSimulator(prof, scn)
    w, tb = to_block_world(prof, scn, n_max=8, q_cap=32, log_cap=128)
    n_out1 = tb.job_index["J-OUT-1"]
    # DISPATCH 복원용: 오더 2 를 크레인 0 이 t=300 에 배정한 것처럼 표를 채운다
    w = w._replace(orders=w.orders._replace(
        service_s=w.orders.service_s.at[n_out1].set(300.0),
        assigned_crane=w.orders.assigned_crane.at[n_out1].set(0)))
    rows = [(300.0, EV_BLOCK_ARRIVAL, n_out1), (300.0, LOG_DISPATCH, 0),
            (512.5, EV_JOB_COMPLETED, 0), (600.0, LOG_ETA_WAKE, n_out1), (600.0, LOG_DEFER_WAKE, -1),
            (700.0, LOG_DEADLOCK_ESCAPE, 0b1), (900.0, EV_JOB_RELEASED, n_out1),
            (1000.0, EV_VESSEL_START, -1), (7200.0, EV_HORIZON, -1), (7300.0, LOG_DISPATCH, 0)]
    t = np.asarray(w.log.t).copy(); kd = np.asarray(w.log.kind).copy(); tg = np.asarray(w.log.target).copy()
    for i, (ti, ki, gi) in enumerate(rows):
        t[i], kd[i], tg[i] = ti, ki, gi
    w = w._replace(log=w.log._replace(t=jnp.asarray(t), kind=jnp.asarray(kd), target=jnp.asarray(tg),
                                      n=jnp.asarray(len(rows), jnp.int32)))
    want = [(300.0, "BLOCK_ARRIVAL", "J-OUT-1"), (300.0, "DISPATCH", "YC-A:J-OUT-1"),
            (512.5, "JOB_COMPLETED", "YC-A"), (600.0, "ETA_WAKE", "J-OUT-1"), (600.0, "DEFER_WAKE", ""),
            (700.0, "DEADLOCK_ESCAPE", "YC-A"), (900.0, "JOB_RELEASED", "J-OUT-1"),
            (1000.0, "VESSEL_START", ""), (7200.0, "HORIZON", "HORIZON"),
            (7300.0, "DISPATCH", "YC-A:None")]                       # 오더를 못 찾으면 REPOSITION 꼴
    assert event_log_from_arrays(w, tb) == want
    sim.event_log = list(want)                                        # v5 해시식 (engine.py:1147-1149)
    assert event_stream_hash(w, tb) == sim.event_stream_hash()


def test_escape_mask_lists_sorted_crane_ids():
    prof, scn = fixtures.build_integrated_profile(), fixtures.build_minimal_terminal_scenario()
    w, tb = to_block_world(prof, scn, n_max=16, q_cap=64, log_cap=256)
    w = w._replace(log=w.log._replace(t=w.log.t.at[0].set(5.0), kind=w.log.kind.at[0].set(LOG_DEADLOCK_ESCAPE),
                                      target=w.log.target.at[0].set(0b11), n=jnp.asarray(1, jnp.int32)))
    assert event_log_from_arrays(w, tb) == [(5.0, "DEADLOCK_ESCAPE", "YC-A,YC-B")]


# ───────────────────────────────────────────────── ⑧ 왕복 — 배열은 jit/vmap pytree
def test_world_is_pytree_and_idtables_frozen():
    prof, scn = piece1_profile(), piece1_scenario()
    w, tb = to_block_world(prof, scn, n_max=8, q_cap=32, log_cap=128)
    leaves = jax.tree_util.tree_leaves(w)
    assert all(hasattr(x, "dtype") for x in leaves)
    w2 = jax.jit(lambda x: x)(w)
    assert from_block_world(w2, tb) == from_block_world(w, tb)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tb.n0 = 1
    assert isinstance(tb, IdTables)
