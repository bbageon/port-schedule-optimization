"""v6 조각 1 — 상태 묶음(`BlockWorld`)·기하(`Geom`) 계약 ([[YR-327]]).

■ 무엇을 지켜야 하나
  ① 정수 상수(JobStatus·CraneStatus·JobFlow·ContainerSize·CandidateKind·EventKind)가
     **v5 enum 선언 순서와 같다** — v5 를 실제로 import 해 대조한다
  ② 비용 13항 순서·rate 5항 순서가 v5 contract/schema·cost 와 같다
  ③ 모든 열의 **모양·dtype** 이 명세 array_layout 대로다 (시각·좌표·거리는 float64)
  ④ 빈칸 표시가 규약대로다 — 정수 -1 · 시각 +inf · 표본 NaN · -inf 표식
  ⑤ Geom 이 v5 프로파일에서 그대로 읽히고, 비용표가 파이썬 곱과 비트 동일하다
  ⑥ BlockWorld 가 jit/vmap 의 pytree 로 통과한다 (Geom 은 static 인자)
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import state as S
from yard_rl.v6.gpu.events import PRIO, TIME_DTYPE
from yard_rl.v6.gpu.geom import Geom
from yard_rl.v6.gpu.state import BlockWorld, empty_block_world, empty_world

# v5 정본
from yard_rl.v6.world.contract.schema import COST_TERMS as V5_COST_TERMS
from yard_rl.v6.world.contract.schema import CandidateKind
from yard_rl.v6.world.domain.enums import (ContainerSize, CraneStatus, JobFlow,
                                          JobStatus)
from yard_rl.v6.world.integrated.cost import RATE_TERMS_ORDERED as V5_RATE_TERMS
from yard_rl.v6.world.integrated.events import _PRIORITY, EventKind

# 조각 1 시험 크기 (명세 capacities)
N, K, B, R, T, C, Q, E = 8, 1, 10, 4, 4, 8, 32, 128


def _geom() -> Geom:
    return Geom(bay_count=B, row_count=R, tier_max=T, bay_len=6.5, row_w=2.9, tier_h=2.6,
                transfer_row=0.0, sla_s=1800.0, shift_len_s=28800.0, gap=2.0, n_lanes=1)


def _world() -> BlockWorld:
    return empty_block_world(_geom(), n_orders=N, n_cranes=K, n_conts=C,
                             q_cap=Q, log_cap=E, end_s=7200.0)


# ───────────────────────────────────────────────── ① 정수 상수 ↔ v5 enum
def test_job_status_matches_v5_order():
    names = ["PLANNED", "RELEASED", "WAITING", "ASSIGNED", "RUNNING", "DONE", "CANCELLED"]
    assert [m.name for m in JobStatus] == names
    for i, nm in enumerate(names):
        assert getattr(S, f"JS_{nm}") == i == list(JobStatus).index(JobStatus[nm])


def test_crane_status_matches_v5_order():
    names = ["IDLE", "MOVING", "HANDLING", "BLOCKED", "DOWN"]
    assert [m.name for m in CraneStatus] == names
    for i, nm in enumerate(names):
        assert getattr(S, f"CR_{nm}") == i
    assert S.CR_WORKING == list(CraneStatus).index(CraneStatus.HANDLING)


def test_job_flow_matches_v5_order():
    for i, m in enumerate(JobFlow):
        assert getattr(S, f"FL_{m.name}") == i
    assert (S.FL_GATE_IN, S.FL_GATE_OUT) == (0, 1)   # 기존 flow 열 규약(0 반입·1 반출)과 일치


def test_container_size_and_candidate_kind():
    for i, m in enumerate(ContainerSize):
        assert getattr(S, f"SZ_{m.name}") == i
    assert [m.name for m in CandidateKind] == ["SERVE", "PRE_REHANDLE", "REPOSITION", "WAIT"]
    assert (S.PK_SERVE, S.PK_PRE_REHANDLE, S.PK_REPOSITION, S.PK_WAIT) == (0, 1, 2, 3)


def test_event_kinds_match_v5():
    for m in EventKind:
        assert getattr(S, f"EV_{m.name}") == int(m)
        assert int(PRIO[int(m)]) == _PRIORITY[m]
    assert (S.LOG_DISPATCH, S.LOG_ETA_WAKE, S.LOG_DEFER_WAKE, S.LOG_DEADLOCK_ESCAPE) == (12, 13, 14, 15)
    assert S.LOG_DISPATCH == len(EventKind)   # 로그 전용은 큐 종류 바로 뒤


# ───────────────────────────────────────────────── ② 비용 13항
def test_cost_terms_match_v5_schema():
    assert S.COST_TERMS == tuple(V5_COST_TERMS)
    assert S.N_COST == 13
    for i, t in enumerate(V5_COST_TERMS):
        assert S.cost_index(t) == i
    assert S.C_TRUCK_WAIT == 0 and S.C_LANE_CONG == 9 and S.C_IMBALANCE == 12
    with pytest.raises(KeyError):
        S.cost_index("no_such_term")


def test_rate_terms_match_v5_order():
    assert S.RATE_TERMS == tuple(V5_RATE_TERMS)
    assert [int(i) for i in S.RATE_IDX] == [V5_COST_TERMS.index(t) for t in V5_RATE_TERMS]
    # rate 5칸 → 13칸 산포가 v5 accrue 와 같은 자리에 들어간다
    rate = jnp.arange(1, 6, dtype=TIME_DTYPE)
    pending = jnp.zeros((S.N_COST,), TIME_DTYPE).at[S.RATE_IDX].add(rate * 2.0)
    want = {t: 2.0 * (i + 1) for i, t in enumerate(V5_RATE_TERMS)}
    for t in V5_COST_TERMS:
        assert float(pending[S.cost_index(t)]) == want.get(t, 0.0)


# ───────────────────────────────────────────────── ③ 모양·dtype
def _leaves_with_names(tree, prefix=""):
    out = {}
    if hasattr(tree, "_fields"):
        for f in tree._fields:
            out.update(_leaves_with_names(getattr(tree, f), f"{prefix}{f}."))
    else:
        out[prefix[:-1]] = tree
    return out


def test_block_world_shapes_and_dtypes():
    w = _world()
    f64, i32, b_ = jnp.dtype(TIME_DTYPE), jnp.dtype(jnp.int32), jnp.dtype(jnp.bool_)
    assert f64 == jnp.dtype("float64")   # x64 가 실제로 켜졌다
    want = {
        "clock": ((), f64), "end_s": ((), f64), "terminal": ((), b_),
        "last_decision_at": ((), f64), "escape_at": ((), f64),
        "escape_count": ((), i32), "steps": ((), i32),
        "queue.time": ((Q,), f64), "queue.kind": ((Q,), i32), "queue.seq": ((Q,), i32),
        "orders.status": ((N,), i32), "orders.flow": ((N,), i32),
        "orders.is_external": ((N,), b_), "orders.is_store": ((N,), b_),
        "orders.target_cont": ((N,), i32), "orders.inbound_cont": ((N,), i32),
        "orders.gate_in_s": ((N,), f64), "orders.block_in_s": ((N,), f64),
        "orders.service_s": ((N,), f64), "orders.done_s": ((N,), f64),
        "orders.gate_out_s": ((N,), f64), "orders.exit_travel_s": ((N,), f64),
        "orders.actual_arrival_s": ((N,), f64), "orders.wait_sample_s": ((N,), f64),
        "orders.waiting": ((N,), b_), "orders.in_block": ((N,), b_),
        "cranes.status": ((K,), i32), "cranes.available_at": ((K,), f64),
        "cranes.bay": ((K,), f64), "cranes.row": ((K,), f64),
        "cranes.bay_min": ((K,), i32), "cranes.down": ((K,), b_), "cranes.yielded": ((K,), b_),
        "cranes.served": ((K,), i32), "cranes.loaded_m": ((K,), f64),
        "cranes.spec_gantry": ((K,), f64), "cranes.spec_truck_pos": ((K,), f64),
        "cranes.rail_order": ((K,), i32),
        "stacks.grid": ((B, R, T), i32), "stacks.height": ((B, R), i32), "stacks.top_size": ((B, R), i32),
        "conts.c_bay": ((C,), i32), "conts.c_size": ((C,), i32), "conts.c_alive": ((C,), b_),
        "res.active": ((K,), b_), "res.token": ((K,), i32), "res.lo": ((K,), f64),
        "res.lane": ((K,), i32), "res.slots": ((K, B, R), b_), "res.token_owner": ((N,), i32),
        "res.idle_pos": ((K,), f64),
        "plan.kind": ((K,), i32), "plan.dur": ((K,), f64), "plan.n_moves": ((K,), i32),
        "plan.mv_cont": ((K, T), i32), "plan.mv_src": ((K, T, 3), i32),
        "plan.mv_dst": ((K, T, 3), i32), "plan.mv_kind": ((K, T), i32),
        "kpi.queue_area": ((), f64), "kpi.rehandles": ((), i32), "kpi.completed_ext": ((), i32),
        "kpi.vessel_delay_s": ((), f64),
        "ledger.block_area": ((), f64), "ledger.closed_end": ((), f64),
        "cost.rate": ((5,), f64), "cost.pending": ((13,), f64), "cost.episode": ((13,), f64),
        "lane.adj": ((1, 1), b_), "lane.cong_area_s": ((), f64),
        "log.t": ((E,), f64), "log.kind": ((E,), i32), "log.target": ((E,), i32), "log.n": ((), i32),
        "decision.pending": ((K,), b_), "decision.act_job": ((K,), i32), "decision.act_bay": ((K,), f64),
        "wake.eta_wake_s": ((0,), f64), "wake.eta_armed": ((K,), b_), "wake.wake_idx": ((), i32),
        "violation": ((), i32), "overflow": ((), i32),
    }
    got = _leaves_with_names(w)
    for name, (shape, dt) in want.items():
        assert name in got, name
        assert got[name].shape == shape, (name, got[name].shape, shape)
        assert got[name].dtype == dt, (name, got[name].dtype, dt)
    # ★모든 실수 잎이 float64 — float32 가 하나라도 섞이면 동등성이 깨진다
    for name, leaf in got.items():
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == f64, name
    assert w.n == N and w.k == K and w.plan.m == T and w.stacks.shape == (B, R, T)


# ───────────────────────────────────────────────── ④ 빈칸 표시
def test_empty_markers():
    w = _world()
    o, c, r, p = w.orders, w.cranes, w.res, w.plan
    # 정수 -1
    for arr in (o.block, o.flow, o.target_cont, o.inbound_cont, o.inbound_size, o.vessel,
                o.assigned_crane, c.assigned, r.token, r.lane, r.token_owner, p.kind, p.job,
                p.mv_cont, p.mv_kind, w.stacks.grid, w.stacks.top_size, w.conts.c_bay,
                w.log.kind, w.log.target, w.decision.act_job, w.queue.kind):
        assert bool(jnp.all(arr == -1))
    # 시각 +inf
    for arr in (o.notice_s, o.gate_in_s, o.block_in_s, o.service_s, o.done_s, o.gate_out_s,
                o.release_s, o.provided_eta_s, o.deadline_s, o.actual_arrival_s,
                w.ledger.closed_end, w.log.t, w.queue.time):
        assert bool(jnp.all(jnp.isposinf(arr)))
    # 특수 표식
    assert bool(jnp.all(o.exit_travel_s == -1.0))          # -1 = None (장부 모드 판별)
    assert bool(jnp.all(jnp.isnan(o.wait_sample_s)))        # 표본 미확정
    assert bool(jnp.all(jnp.isnan(w.decision.act_bay)))
    assert bool(jnp.isneginf(w.last_decision_at)) and bool(jnp.isneginf(w.escape_at))
    # 상태 초기값
    assert bool(jnp.all(o.status == S.JS_PLANNED)) and bool(jnp.all(c.status == S.CR_IDLE))
    assert bool(jnp.all(w.stacks.height == 0)) and not bool(jnp.any(w.conts.c_alive))
    assert bool(jnp.all(w.conts.c_avail))
    assert not bool(jnp.any(r.active)) and not bool(jnp.any(r.slots))
    assert not bool(w.terminal) and int(w.violation) == 0 and int(w.overflow) == 0
    assert int(w.steps) == 0 and int(w.log.n) == 0 and int(w.queue.overflow) == 0
    assert float(w.clock) == 0.0 and float(w.end_s) == 7200.0
    assert bool(jnp.all(w.cost.rate == 0)) and bool(jnp.all(w.cost.episode == 0))
    assert [int(x) for x in c.rail_order] == list(range(K))


# ───────────────────────────────────────────────── ⑤ Geom ↔ v5 프로파일
def test_geom_from_v5_profile_and_cost_tables():
    from yard_rl.v6.world.contract.state import LaneGraph
    from yard_rl.v6.world.domain.models import BlockGeometry
    from yard_rl.v6.world.integrated import fixtures

    prof = fixtures.build_integrated_profile()
    prof = dataclasses.replace(
        prof, block=BlockGeometry("B1", 10, 4, 4, 6.5, 2.9, 2.6, 0),
        lane_graph=LaneGraph(("L1",), ()))
    g = Geom.from_profile(prof)
    assert (g.bay_count, g.row_count, g.tier_max) == (10, 4, 4)
    assert (g.bay_len, g.row_w, g.tier_h, g.transfer_row) == (6.5, 2.9, 2.6, 0.0)
    assert g.sla_s == prof.long_wait_sla_s == 1800.0
    assert g.shift_len_s == prof.shift_len_s and g.gap == prof.safety_gap_bay == 2.0
    assert g.n_lanes == 1 and g.n_moves == 4 and g.cells == 40
    # static 인자로 쓸 수 있다 (hashable · 값 동등)
    assert hash(g) == hash(_geom()) and g == _geom()
    # ★비용표 = v5 stack.py:164 의 파이썬 곱과 **비트 동일** (곱을 한 번만 반올림)
    assert len(g.row_cost) == 5 and len(g.tier_cost) == 5
    for d in range(5):
        assert g.row_cost[d] == abs(0.0 - float(d)) * 2.9      # near_row(=transfer_row) − row
        assert g.tier_cost[d] == d * 2.6                          # top(int) * tier_h
    # 배열로 올려도 값이 안 바뀐다
    assert np.array_equal(np.asarray(jnp.asarray(g.tier_cost, TIME_DTYPE)), np.array(g.tier_cost))
    # 근접 동률 사례(반박 검증 finding 2): 파이썬은 5·2.6 을 13.0 으로 **먼저 반올림**해
    # 2·6.5 와 정확히 같아진다 — 표가 이 반올림을 보존한다 (융합하면 13.00000000000000044 로 갈림)
    assert 5 * 2.6 == 13.0 == 2 * 6.5
    g5 = dataclasses.replace(g, tier_max=5)
    assert g5.tier_cost[5] == 13.0 and math.isclose(g5.tier_cost[5], 13.0, rel_tol=0.0, abs_tol=0.0)


# ───────────────────────────────────────────────── ⑥ pytree — jit·vmap 통과
def test_block_world_is_a_pytree_for_jit_and_vmap():
    g = _geom()
    w = _world()

    @jax.jit
    def bump(w: BlockWorld) -> BlockWorld:
        return w._replace(clock=w.clock + 1.0, steps=w.steps + 1)

    w1 = bump(w)
    assert float(w1.clock) == 1.0 and int(w1.steps) == 1 and w1.clock.dtype == jnp.float64

    def with_geom(w: BlockWorld, g: Geom):
        return w._replace(violation=w.violation + g.tier_max)

    w2 = jax.jit(with_geom, static_argnums=1)(w, g)
    assert int(w2.violation) == T
    # vmap: 세계 둘을 쌓아도 잎마다 앞축 2 가 붙는다
    stacked = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), w, w1)
    out = jax.vmap(bump)(stacked)
    assert out.clock.shape == (2,) and [float(x) for x in out.clock] == [1.0, 2.0]
    assert out.stacks.grid.shape == (2, B, R, T)
    # 잎 수가 고정이다 (칸 추가는 곧 계약 변경 — 여기서 드러난다)
    assert len(jax.tree_util.tree_leaves(w)) == len(jax.tree_util.tree_leaves(w1))


def test_legacy_world_still_works_and_is_float64():
    w = empty_world(3, 1, end_s=1_000.0)
    assert w.orders.gate_in_s.dtype == jnp.float64 and w.clock.dtype == jnp.float64
    assert bool(jnp.all(w.orders.flow == -1))
    assert float(w.cost_total) == 0.0 and int(w.overflow) == 0
