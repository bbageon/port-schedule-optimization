"""배정 scan(gpu/dispatch.py) 이 v5 `ReferenceDispatcher.run` + `commit_decisions` 와 **같은 답**을 내는가
([[YR-327]] 조각 2 · key=dispatch).

■ 무엇을 지키나 — v5 와 배열 세계를 **결정마다 나란히**(lockstep) 굴리며 대조한다
  결정 시점마다 (v5 `run_until_decision` ↔ 배열 '사건만 처리해 결정 직전까지'):
    ① 결정 시각·물은 크레인 집합 (v5 TerminalDecision.crane_ids ↔ open)
    ② 크레인 순서대로 **live 후보 집합** (v5 `candidates_for` — 앞 크레인 배정 반영 ↔ cand_live 행)
    ③ (크레인, 오더) 마다 **거절 사유 범주** — INELIGIBLE / NOT_DISPATCHABLE / TAKEN / NO_PLAN /
       DOUBLE_RESERVE·DUP_JOB·LANE_CONFLICT·CRANE_INTERFERENCE·SLOT_CONFLICT / OK — v5 판정 순서 그대로
    ④ 정책의 답 (act, job) · 수용된 계획 dur/rehandles/corridor/end/lane/slots/moves 전부 (실수 `==`)
    ⑤ 결정 뒤 예약표 (크레인별 token·corridor·lane·release_at·slots · 토큰 역표 · idle 장벽) · 위반 0
  끝까지 간 뒤에는 조각 1 `test_gpu_engine_equiv.compare` (사건 로그 전열·결정열·오더·계획·크레인·kpi·격자·
  실수 비트 동일) 를 그대로 빌려 최종 상태를 대조한다.
  K=1 무대에서는 결정마다 조각 1 `engine_step.decide` 와 `decide_seq` 의 결과 잎 전부가 비트 같은지도 본다.

■ 무대 — 무작위 30 (seed 0..29): K = seed%3+1 ∈ {1,2,3} · 격자 3종 · 오더 5~10 · 야드 35% 채움 ·
  같은 담당구간(초기 위치 등간격 분산) 60% / 계단식 구간 40% · 레인 1~3 · 동시각 도착 짝 60% ·
  고장/복구 주입 40% · 장부 모드 60% · 본선연계(선박 없는 VESSEL_LOAD) 30% · 정책 4종 순환:
    reference (v5 ReferenceDispatcher.select = 본선 우선 → 최장 대기 → job_id) · first · last · wait2
  기대값은 손으로 적지 않는다 — v5 를 같은 정책으로 실제로 굴린다.

■ 교착 탈출(v5 `_try_escape`, 384-411행)은 조각 2 의 다른 담당(탈출)이다. 여기 무대(같은 구간 2~3대
  + gap 2)에서는 실제로 자주 발화하므로, 시험 쪽 구동 루프가 v5 `_try_escape` 를 **호스트에서 그대로
  흉내 내어** (술어는 dispatch.candidates 의 deadlock, 결정은 decide_seq(open_override=유휴 전원))
  lockstep 이 끊기지 않게 한다 — 이 흉내는 시험 발판이고 엔진 통합이 아니다.

실행: WSL venv · x64 CPU.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy
from jax import lax                                                                  # noqa: E402

from yard_rl.v6.gpu import engine_step as ES                                        # noqa: E402
from yard_rl.v6.gpu.dispatch import (CandOut, DispatchOut, NO_PLAN, candidates, decide_seq,   # noqa: E402
                                     dispatch, dry_run as DRY_RUN)
from yard_rl.v6.gpu.engine_step import (EPS, DecideOut, advance, first_by_id, ledger_mode,   # noqa: E402
                                        log_event, refresh_rates)
from yard_rl.v6.gpu.events import next_event                                        # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import from_block_world, to_block_world            # noqa: E402
from yard_rl.v6.gpu.reserve import CODE_TO_REASON                                   # noqa: E402
from yard_rl.v6.gpu.state import (EMPTY_ID, LOG_DEADLOCK_ESCAPE, MV_REHANDLE, MV_RETRIEVE, MV_STORE,   # noqa: E402
                                  V_BUSY_NO_EVENT)
from yard_rl.v6.gpu import reserve as RS                                            # noqa: E402
from yard_rl.v6.gpu import state as ST                                              # noqa: E402
# v5 정본
from yard_rl.v6.world.contract.schema import CandidateKind                          # noqa: E402
from yard_rl.v6.world.contract.state import LaneGraph                               # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, JobFlow, LoadStatus        # noqa: E402
from yard_rl.v6.world.domain.models import BlockGeometry, Container, Job           # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                    # noqa: E402
from yard_rl.v6.world.integrated.dispatcher import ReferenceDispatcher              # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator   # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec                   # noqa: E402
from yard_rl.v6.world.integrated.scenario import InjectedEvent, TerminalScenario    # noqa: E402

FT20, FT40, FT45 = ContainerSize.FT20, ContainerSize.FT40, ContainerSize.FT45
MV_NAME = {MV_REHANDLE: "REHANDLE", MV_RETRIEVE: "RETRIEVE", MV_STORE: "STORE"}
POLICY_NAMES = ("reference", "first", "last", "wait2")
#: 무대별 집계 (마지막 시험이 보고)
REPORT: dict[str, dict] = {}


def _load_engine_test():
    """tests/v6/test_gpu_engine_equiv.py 의 최종 대조 `compare` 를 경로로 불러온다 (tests/ 에 __init__ 없음)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_gpu_engine_equiv.py")
    spec = importlib.util.spec_from_file_location("_tge_for_dispatch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TGE = _load_engine_test()
# compare ② 는 StepTrace 에서 결정열을 뽑는다 — 여기서는 lockstep 이 만든 결정열을 그대로 넘긴다
_TGE._array_decisions = lambda w, trace, tb: trace


# ───────────────────────────────────────────────── 시험 규약: ReservationArrays 통일
def test_reservation_arrays_is_single_class():
    """reserve.py 와 state.py 가 **같은** ReservationArrays 를 쓴다 (조각 1 의 사본 중복 제거)."""
    assert RS.ReservationArrays is ST.ReservationArrays
    r = RS.empty_reservations(2, 4, 3, 2)
    assert type(r) is ST.ReservationArrays and r.k == 2
    assert bool(jnp.all(jnp.isposinf(r.idle_pos))) and bool(jnp.all(jnp.isposinf(r.release_at)))
    # 형이 같으니 reserve.py 결과와 state.py 세계의 예약표를 tree_map 으로 섞어도 깨지지 않는다
    jax.tree_util.tree_map(lambda a, b: a, r, RS.release(r, 0))
    assert type(ES._as_res(r, RS.release(r, 0))) is ST.ReservationArrays     # 엔진의 변환 도우미는 항등


# ───────────────────────────────────────────────── 무대 (무작위)
GEOMS = [(10, 4, 4, 1), (12, 3, 4, 2), (10, 4, 3, 3)]   # (B, R, T, 레인 수)
N_MAX, Q_CAP, LOG_CAP = 16, 64, 512


def _c(cid, bay, row, tier, size=FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _out(jid, target, gate_in, arrival, exit_s):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, target_container=target, exit_travel_s=exit_s)


def _in(jid, gate_in, arrival, size, exit_s):
    return Job(job_id=jid, flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, inbound_size=size, inbound_load=LoadStatus.FULL,
               exit_travel_s=exit_s)


def _crane(cid, lo, hi, rng):
    sp = replace(fixtures._spec(cid), service_bay_min=lo, service_bay_max=hi)
    if rng.random() < 0.5:                                   # 속도가 다른 크레인도 섞는다
        sp = replace(sp, gantry_speed_mps=rng.choice([1.7, 2.0, 2.3]),
                     hoist_speed_loaded_mps=rng.choice([0.45, 0.5]), lock_time_s=rng.choice([28.0, 30.0]))
    return sp


def make_stage(seed: int):
    """(profile, scenario, K, 정책 이름). 머리말 '무대' 참조."""
    rng = random.Random(1000 + seed)
    K = seed % 3 + 1
    B, R, T, L = GEOMS[(seed // 3) % 3]
    policy = POLICY_NAMES[seed % 4]
    base = fixtures.build_integrated_profile()
    ids = ["YC-A", "YC-B", "YC-C"][:K]
    if rng.random() < 0.6:                                   # 같은 담당구간 → 초기 위치 등간격 분산 (engine.py:105-113)
        cranes = tuple(_crane(cid, 1, B, rng) for cid in ids)
        layout = "shared"
    else:                                                    # 계단식 구간 (초기 위치 = service_bay_min, 간격 3 ≥ gap)
        cranes = tuple(_crane(cid, 1 + 3 * i, B - 3 * (K - 1 - i), rng) for i, cid in enumerate(ids))
        layout = "stepped"
    lanes = tuple(f"L{i + 1}" for i in range(L))
    edges = tuple((lanes[i], lanes[i + 1]) for i in range(L - 1) if rng.random() < 0.5)
    prof = replace(base, block=BlockGeometry("B1", B, R, T, 6.5, 2.9, 2.6, 0), cranes=cranes,
                   lane_graph=LaneGraph(lanes, edges),
                   transfer=TransferFleetSpec("TF1", "YT", n_units=0, move_time_s=180.0),
                   safety_gap_bay=rng.choice([2.0, 2.0, 1.5]))
    # 야드
    containers = {}
    n = 0
    for bay in range(1, B + 1):
        for row in range(1, R + 1):
            if rng.random() < 0.35:
                size = rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0]
                for t in range(1, rng.randint(1, T) + 1):
                    containers[f"C{n:03d}"] = _c(f"C{n:03d}", bay, row, t, size)
                    n += 1
    ledger = rng.random() < 0.6
    ex = 60.0 if ledger else None
    tight = rng.random() < 0.5                                # 도착을 몰아 크레인 경합을 만든다

    def arrival():
        return round(rng.uniform(0.0, 500.0 if tight else 2500.0), 3)

    jobs = []
    targets = rng.sample(sorted(containers), min(len(containers), rng.randint(3, 6)))
    for i, tgt in enumerate(targets):
        arr = arrival()
        jobs.append(_out(f"J-OUT-{i:02d}", tgt, max(0.0, arr - rng.uniform(0.0, 600.0)), arr, ex))
    for i in range(rng.randint(2, 4)):
        arr = arrival()
        jobs.append(_in(f"J-IN-{i:02d}", max(0.0, arr - rng.uniform(0.0, 600.0)), arr,
                        rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0], ex))
    if rng.random() < 0.6 and len(jobs) >= 2:                # 동시각 도착 짝
        a, b = rng.sample(range(len(jobs)), 2)
        jobs[b] = replace(jobs[b], actual_block_arrival=jobs[a].actual_block_arrival,
                          actual_gate_in=min(jobs[b].actual_gate_in, jobs[a].actual_block_arrival))
    if rng.random() < 0.3:                                   # 선박 없는 본선연계 반출 (reference 정책의 '본선 우선' 키)
        pool = [c for c in sorted(containers) if c not in targets]
        if pool:
            jobs.append(Job(job_id="J-VL-00", flow=JobFlow.VESSEL_LOAD, release_time=float(rng.randint(0, 1200)),
                            actual_gate_in=None, actual_block_arrival=None, target_container=rng.choice(pool),
                            deadline=float(rng.choice([600, 5000]))))
    injected = []
    if rng.random() < 0.4:                                   # 고장/복구
        t1 = float(rng.randint(150, 1500))
        cid = rng.choice(ids)
        injected = [InjectedEvent(t1, "EQUIPMENT_DOWN", cid), InjectedEvent(t1 + rng.randint(200, 900), "EQUIPMENT_UP", cid)]
    scn = TerminalScenario(scenario_id=f"dispatch-{seed}", seed=seed,
                           horizon_s=rng.choice([3600.0, 7200.0]), drain_window_s=rng.choice([0.0, 600.0]),
                           containers=containers, jobs=jobs, vessels=[], injected_events=injected)
    assert len(jobs) <= N_MAX
    return prof, scn, K, policy, dict(layout=layout, tight=tight, ledger=ledger, lanes=L, inject=bool(injected))


# ───────────────────────────────────────────────── 정책 (v5 chooser ↔ 배열 policy_fn 짝)
def v5_choose(name: str, sim, cid, cands):
    if not cands:
        return None
    if name == "reference":
        return ReferenceDispatcher().select(sim, cid, cands)
    if name == "first":
        return cands[0]
    if name == "last":
        return cands[-1]
    if name == "wait2":
        return None if len(cands) == 2 else cands[-1]
    raise ValueError(name)


def policy_mixed(params, x, mask):
    """네 정책을 하나의 static 함수로 — mode 는 params 에 들어와 재컴파일이 없다.

    reference = v5 `ReferenceDispatcher.select`: min by (0 if vessel else 1, −cum_wait, job_id).
    cum_wait (engine.py:258-265) = 외부트럭이고 도착 ≤ clock 이면 clock − 도착, 아니면 0.
    job_id 오름차순 = 오더 번호 오름차순 (host_convert 번호 규칙). 행마다 독립.
    """
    mode, arr, ext, ves, clock = params
    K, N = mask.shape
    has = jnp.any(mask, axis=1)
    first = jnp.argmax(mask, axis=1)
    last = N - 1 - jnp.argmax(mask[:, ::-1], axis=1)
    n_c = jnp.sum(mask, axis=1)
    wait2 = jnp.where(n_c == 2, EMPTY_ID, last)
    cum = jnp.where(ext & (arr <= clock), clock - arr, 0.0)                # cum_wait
    k1 = jnp.where(ves, 0, 1).astype(jnp.int32)
    k2 = -cum
    big = jnp.int32(9)
    m1 = mask & (k1[None, :] == jnp.min(jnp.where(mask, k1[None, :], big), axis=1)[:, None])
    m2 = m1 & (k2[None, :] == jnp.min(jnp.where(m1, k2[None, :], jnp.inf), axis=1)[:, None])
    ref = jnp.argmax(m2, axis=1)
    pick = jnp.where(mode == 0, ref, jnp.where(mode == 1, first, jnp.where(mode == 2, last, wait2)))
    return jnp.where(has, pick, EMPTY_ID).astype(jnp.int32)


def _params(w, policy: str):
    o = w.orders
    return (jnp.int32(POLICY_NAMES.index(policy)), o.actual_arrival_s, o.is_external, o.is_vessel, w.clock)


# ───────────────────────────────────────────────── 배열 쪽 구동 (사건만 처리 — decide 는 부르지 않는다)
_RUNNERS: dict[Geom, tuple] = {}


def _make_runners(g: Geom):
    """engine_step.step 의 E(사건)·F(종료) 국면만 떼어 jit — D(결정)는 이 시험이 dispatch/decide_seq 로 한다.

    g(격자·레인 수) 별로 캐시한다 — 같은 g 의 무대들이 컴파일을 나눠 쓴다 (K 가 다르면 jit 이 알아서 재추적).
    반환 (event, fin, cand, disp, dec, dec1).
    """
    if g in _RUNNERS:
        return _RUNNERS[g]
    handlers = ES._handlers(g)

    @jax.jit
    def event(w):
        q2, t, kind, tgt, _ = next_event(w.queue)
        w1 = w._replace(queue=q2)
        w1 = advance(w1, t, g)
        w1 = log_event(w1, t, kind, tgt)
        w1 = lax.switch(kind, handlers, w1, tgt)
        w1 = refresh_rates(w1, g)
        return w1._replace(steps=w.steps + 1), kind, tgt

    @jax.jit
    def fin(w):
        raw_nt = jnp.min(w.queue.time)
        alive = raw_nt < jnp.inf
        w1 = advance(w, jnp.maximum(w.clock, w.end_s), g)
        busy = jnp.any(w1.cranes.assigned >= 0) & ~alive
        o = w1.orders
        ws = jnp.where(o.waiting & jnp.isnan(o.wait_sample_s),
                       jnp.maximum(0.0, w1.end_s - o.block_in_s), o.wait_sample_s)
        lm = ledger_mode(o)
        ld = w1.ledger._replace(closed_end=jnp.where(lm, w1.end_s, w1.ledger.closed_end))
        return w1._replace(orders=o._replace(wait_sample_s=ws), ledger=ld, terminal=jnp.ones((), bool),
                           violation=w1.violation | jnp.where(busy, V_BUSY_NO_EVENT, 0).astype(jnp.int32))

    cand = jax.jit(partial(candidates, g=g))
    disp = jax.jit(partial(dispatch, g=g, policy_fn=policy_mixed, with_trace=True))
    dec = jax.jit(partial(decide_seq, g=g, policy_fn=policy_mixed))
    dec1 = jax.jit(partial(ES.decide, g=g, policy_fn=policy_mixed))
    _RUNNERS[g] = (event, fin, cand, disp, dec, dec1)
    return _RUNNERS[g]


def _escape_arrays(w, c: CandOut):
    """v5 `_try_escape` (384-411행) 를 **호스트에서** 흉내 낸다 — 시험 발판 (머리말). 반환 (w, 유휴 마스크|None)."""
    clock, end = float(w.clock), float(w.end_s)
    if clock >= end - EPS or float(w.escape_at) == clock:
        return w, None
    lda = float(w.last_decision_at)
    if lda != -math.inf and clock <= lda + EPS:
        return w, None
    if not bool(c.deadlock):
        return w, None
    w = w._replace(cranes=w.cranes._replace(yielded=jnp.zeros_like(w.cranes.yielded)))   # _clear_yields (398행)
    idle = np.asarray((w.cranes.assigned < 0) & ~w.cranes.down)
    if not idle.any():
        return w, None
    bits = int(sum(1 << k for k in range(idle.shape[0]) if idle[k]))
    w = w._replace(escape_at=jnp.asarray(clock, jnp.float64), last_decision_at=jnp.asarray(clock, jnp.float64),
                   escape_count=w.escape_count + 1)
    w = log_event(w, clock, LOG_DEADLOCK_ESCAPE, bits)                                # 408행
    return w, jnp.asarray(idle)


def advance_to_decision(w, g, runners):
    """v5 `run_until_decision` (276-336행; wake·review 없음) 의 배열판 — 결정 직전에 멈춘다.

    반환 (w, 'decision'|'escape'|'terminal', CandOut|None, open_override|None).
    """
    event, fin, cand, _, _, _ = runners
    while True:
        if bool(w.terminal):
            return w, "terminal", None, None
        raw_nt = float(jnp.min(w.queue.time))
        clock, end = float(w.clock), float(w.end_s)
        nt_ok = raw_nt < math.inf and raw_nt <= end + EPS                    # 282-287행
        due_now = nt_ok and raw_nt <= clock + EPS                            # 288행
        if not due_now:
            c = cand(w)
            if clock < end - EPS:
                if bool(jnp.any(c.open)):                                    # 293-299행
                    return w, "decision", c, None
                w, esc = _escape_arrays(w, c)                                # 302-305행 (immediate)
                if esc is not None:
                    return w, "escape", c, esc
        if not nt_ok:                                                        # 323-332행 → _finalize
            return fin(w), "terminal", None, None
        w, _, _ = event(w)                                                   # 336행


# ───────────────────────────────────────────────── v5 쪽 도구
def v5_reason_row(sim, cid) -> dict[str, str]:
    """v5 판정 순서(candidates_for 466-489행) 그대로 오더마다 '왜 후보가 아닌가' 범주 — OK 면 후보."""
    yc, spec = sim.fleet.get(cid), sim.fleet.spec(cid)
    out = {}
    for jid in sorted(sim.jobs):
        j = sim.jobs[jid]
        if not yc.idle or yc.yielded:
            out[jid] = "INELIGIBLE"
        elif not sim._dispatchable(j, cid):
            out[jid] = "NOT_DISPATCHABLE"
        elif sim.reservations.job_taken(j.job_id) is not None:
            out[jid] = "TAKEN"
        else:
            ref = sim._jobref(j, spec, yc)
            if ref is None:
                out[jid] = "NO_REF"
                continue
            plan = sim._plan(cid, ref)
            if plan is None:
                out[jid] = "NO_PLAN"
            else:
                out[jid] = sim.reservations.reject_reason(sim._reservation(plan)) or "OK"
    return out


def _v5_moves(plan):
    return [(m.container_id, tuple(m.src), tuple(m.dst),
             "STORE" if m.inbound is not None else ("RETRIEVE" if m.depart else "REHANDLE"))
            for m in plan.moves]


def _v5_plan_tuple(plan):
    return (plan.duration_s, plan.rehandles, float(plan.corridor[0]), float(plan.corridor[1]),
            float(plan.end_bay), float(plan.end_row), plan.loaded_gantry_m, plan.empty_gantry_m,
            plan.lane_id, set(plan.slots), _v5_moves(plan))


def _arr_plan_tuple(w, k, tb):
    pl = w.plan
    moves = []
    for m in range(int(pl.n_moves[k])):
        moves.append((tb.cont_ids[int(pl.mv_cont[k, m])], tuple(int(v) for v in pl.mv_src[k, m]),
                      tuple(int(v) for v in pl.mv_dst[k, m]), MV_NAME[int(pl.mv_kind[k, m])]))
    rs = w.res
    lane = tb.lane_ids[int(rs.lane[k])] if int(rs.lane[k]) >= 0 else None
    slots = {(int(b) + 1, int(r) + 1) for b, r in np.argwhere(np.asarray(rs.slots[k]))}
    return (float(pl.dur[k]), int(pl.rehandles[k]), float(pl.lo[k]), float(pl.hi[k]),
            float(pl.end_bay[k]), float(pl.end_row[k]), float(pl.loaded_m[k]), float(pl.empty_m[k]),
            lane, slots, moves)


def _v5_reservations(sim, tb):
    out = {}
    for cid, r in sim.reservations._by_crane.items():
        out[cid] = {"job_token": r.job_token, "corridor": (r.corridor.lo, r.corridor.hi),
                    "lane_id": r.lane_id, "release_at": r.release_at, "slots": frozenset(r.slots)}
    return out


def _arr_category(d: DispatchOut, disp, eligible, k, n) -> str:
    """배열판 '왜 후보가 아닌가' 범주 — v5 판정 순서(v5_reason_row)와 같은 사다리.

    ★v5 의 TAKEN 은 실제로는 나올 수 없다 — 토큰이 잡힌 오더는 같은 assign 에서 RUNNING·assigned_crane 이
    되므로 `_dispatchable` 이 먼저 거른다(495-497행). 배열판은 dispatchable 을 결정 시작 시점에 한 번 계산하고
    '이번 결정에서 앞 크레인이 잡은 오더' 는 taken_live 로 가리므로, 그 둘의 합이 v5 의 NOT_DISPATCHABLE 이다.
    """
    if not eligible[k]:
        return "INELIGIBLE"
    if not bool(disp[k, n]) or bool(d.taken_live[k, n]):
        return "NOT_DISPATCHABLE"
    if not bool(d.plan_ok_live[k, n]):
        return "NO_PLAN"
    return CODE_TO_REASON[int(d.code_live[k, n])] or "OK"



# ───────────────────────────────────────────────── lockstep
def lockstep(seed: int):
    prof, scn, K, policy, meta = make_stage(seed)
    label = f"s{seed:02d}-K{K}-{policy}-{meta['layout']}"
    g = Geom.from_profile(prof)
    B, R, T, L = prof.block.bay_count, prof.block.row_count, prof.block.tier_max, len(prof.lane_graph.lane_ids)
    w, tb = to_block_world(prof, scn, n_max=N_MAX, q_cap=Q_CAP, log_cap=LOG_CAP, c_max=B * R * T + N_MAX)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    runners = _make_runners(g)
    _, _, _, disp_jit, dec_jit, dec1_jit = runners
    v5_dec, ar_dec = [], []
    stats = dict(decisions=0, escapes=0, waits=0, live_differs=0, taken_live=0, reasons={}, k1_checked=0,
                 same_time_open=0, multi_open=0)
    while True:
        w, status, c, esc = advance_to_decision(w, g, runners)
        esc_before = sim._escape_count
        dp = sim.run_until_decision()
        # ① 결정 시각·종류·물은 크레인
        if dp is None:
            assert status == "terminal", f"[{label}] v5 는 끝났는데 배열은 {status} (clock={float(w.clock)})"
            break
        v5_esc = sim._escape_count > esc_before
        assert status != "terminal", f"[{label}] 배열은 끝났는데 v5 는 결정 {dp} (escape={v5_esc})"
        assert v5_esc == (status == "escape"), f"[{label}] 탈출 여부 v5={v5_esc} arr={status} t={dp.time}"
        assert float(w.clock) == dp.time, f"[{label}] 결정 시각 v5={dp.time!r} arr={float(w.clock)!r}"
        open_ids = tuple(tb.crane_ids[k] for k in range(K) if bool((c.open if esc is None else esc)[k]))
        assert open_ids == tuple(dp.crane_ids), f"[{label}] t={dp.time} 물은 크레인 v5={dp.crane_ids} arr={open_ids}"
        stats["decisions"] += 1
        stats["escapes"] += int(v5_esc)
        stats["multi_open"] += int(len(open_ids) >= 2)
        # 배열 결정 (진단용 dispatch + 실제 갱신 decide_seq — 같은 입력)
        params = _params(w, policy)
        eligible = np.asarray((w.cranes.assigned < 0) & ~w.cranes.down & ~w.cranes.yielded)
        disp_now = np.asarray(c.disp)                                 # yielded 와 무관 — 탈출의 _clear_yields 전후 같다
        d: DispatchOut = disp_jit(w, params, open_override=esc)
        out: DecideOut = dec_jit(w, params, open_override=esc)
        if K == 1 and esc is None:                                    # 조각 1 decide 와 같은 답인가 (잎 전부)
            out1 = dec1_jit(w, params)
            names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(out.world)]
            bad = [names[i] for i, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(out.world),
                                                            jax.tree_util.tree_leaves(out1.world)))
                   if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)]
            assert not bad, f"[{label}] t={dp.time} K=1 인데 decide_seq 와 조각 1 decide 가 다른 잎 {bad}"
            assert np.array_equal(np.asarray(out.pick), np.asarray(out1.pick)) and bool(out.decided) == bool(out1.decided)
            stats["k1_checked"] += 1
        # ②③④ 크레인 순서대로 — v5 를 한 크레인씩 배정하며 live 후보·사유·답·계획을 대조
        rec5, reca = [], []
        for cid in dp.crane_ids:
            k = tb.crane_index[cid]
            cands = sim.candidates_for(cid)
            live5 = [j.job_id for j in cands]
            livea = [tb.job_ids[n] for n in range(tb.n0) if bool(d.cand_live[k, n])]
            assert live5 == livea, f"[{label}] t={dp.time} {cid} live 후보 v5={live5} arr={livea}"
            cand0a = [tb.job_ids[n] for n in range(tb.n0) if bool(d.cand0[k, n])]
            stats["live_differs"] += int(cand0a != livea)
            row5 = v5_reason_row(sim, cid)
            rowa = {tb.job_ids[n]: _arr_category(d, disp_now, eligible, k, n) for n in range(tb.n0)}
            assert row5 == rowa, f"[{label}] t={dp.time} {cid} 거절 사유 범주 v5={row5} arr={rowa}"
            for r in row5.values():
                stats["reasons"][r] = stats["reasons"].get(r, 0) + 1
            # 이번 결정에서 앞 크레인이 잡아 뒤 크레인 후보에서 빠진 오더 수 (순차 전파가 실제로 일어났나)
            stats["taken_live"] += int(np.sum(disp_now[k] & np.asarray(d.taken_live[k])))
            ref = v5_choose(policy, sim, cid, cands)
            pick = int(d.pick[k])
            if ref is None:
                assert pick == EMPTY_ID, f"[{label}] t={dp.time} {cid} v5=WAIT arr={tb.job(pick)}"
                sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))
                stats["waits"] += 1
                rec5.append((cid, None, [], None, None)); reca.append((cid, None, [], None, None))
                continue
            assert pick >= 0 and tb.job_ids[pick] == ref.job_id, \
                f"[{label}] t={dp.time} {cid} 답 v5={ref.job_id} arr={tb.job(pick)} (live={live5})"
            assert int(d.commit_code[k]) == 0, f"[{label}] t={dp.time} {cid} reserve 코드 {int(d.commit_code[k])}"
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, ref))
            p5 = sim.active_plan(cid)
            t5, ta = _v5_plan_tuple(p5), _arr_plan_tuple(out.world, k, tb)
            assert t5 == ta, f"[{label}] t={dp.time} {cid}:{ref.job_id} 계획\n  v5 ={t5}\n  arr={ta}"
            rec5.append((cid, ref.job_id, _v5_moves(p5), p5.duration_s, p5.rehandles))
            reca.append((cid, ref.job_id, ta[-1], ta[0], ta[1]))
        sim.close_decision()
        v5_dec.append((dp.time, tuple(dp.crane_ids), rec5))
        ar_dec.append((float(w.clock), open_ids, reca))
        # ⑤ 결정 뒤 예약표·위반
        w = out.world
        dd = from_block_world(w, tb)
        assert dd["violation"] == 0, f"[{label}] t={dp.time} violation {dd['violation_names']}"
        assert dd["reservations"] == _v5_reservations(sim, tb), \
            f"[{label}] t={dp.time} 예약표\n  v5 ={_v5_reservations(sim, tb)}\n  arr={dd['reservations']}"
        assert dd["idle_positions"] == sim.reservations.idle_positions()
        owner = {tb.job_ids[n]: tb.crane_ids[int(k)] for n, k in enumerate(np.asarray(w.res.token_owner)[:tb.n0]) if k >= 0}
        assert owner == dict(sim.reservations._tokens), f"[{label}] t={dp.time} 토큰 역표 v5={sim.reservations._tokens} arr={owner}"
        assert dd["last_decision_at"] == sim._last_decision_at
        assert int(w.escape_count) == sim._escape_count
        kinds_now = {kk for (t, kk, _) in sim.event_log if t == dp.time and kk not in ("DISPATCH", "DEADLOCK_ESCAPE")}
        stats["same_time_open"] += int(len(kinds_now) >= 2)
    # 최종 — 조각 1 의 compare 를 그대로 (사건 로그·결정열·오더·계획·크레인·kpi·격자·실수 비트 동일)
    _TGE.compare(sim, v5_dec, w, ar_dec, tb, label)
    stats.update(meta, K=K, policy=policy, events=len(sim.event_log), steps=int(w.steps),
                 backlog=sim.unfinished_backlog(),
                 cost={k: round(v, 3) for k, v in sim.cost.episode_raw().items() if v})
    REPORT[label] = stats
    return sim, w, tb, stats


@pytest.mark.parametrize("seed", list(range(30)))
def test_lockstep_equivalence(seed):
    lockstep(seed)


# ───────────────────────────────────────────────── 구조 시험 — 정책 호출이 scan 안에 있다
def test_policy_is_called_per_crane_with_single_live_row():
    """K=3 결정 하나에서 policy_fn 이 **크레인마다** 불리고, 그때 mask 는 그 크레인 행만 살아 있다."""
    prof, scn, K, _, _ = make_stage(2)          # seed 2 → K=3
    assert K == 3
    g = Geom.from_profile(prof)
    B, R, T = prof.block.bay_count, prof.block.row_count, prof.block.tier_max
    w, tb = to_block_world(prof, scn, n_max=N_MAX, q_cap=Q_CAP, log_cap=LOG_CAP, c_max=B * R * T + N_MAX)
    runners = _make_runners(g)
    w, status, c, esc = advance_to_decision(w, g, runners)
    assert status in ("decision", "escape")
    seen = []

    def spy(params, x, mask):
        rows = jnp.any(mask, axis=1)
        # scan 본문은 한 번만 추적된다 — 단계마다 실제 값을 받으려면 런타임 callback
        jax.debug.callback(lambda r: seen.append(np.asarray(r)), rows)
        return first_by_id(params, x, mask)

    d = dispatch(w, None, g, spy, open_override=esc)
    jax.block_until_ready(d.world.clock)
    assert len(seen) == K, f"정책이 {len(seen)} 번 불렸다 (K={K})"
    for k, rows in enumerate(seen):
        rows = np.asarray(rows)
        assert not rows[np.arange(K) != k].any(), f"단계 {k}: 다른 행 {np.nonzero(rows)[0]} 이 살아 있다"
    assert int(d.world.violation) == 0


def test_same_order_chosen_twice_is_resolved_not_dup_job():
    """조각 1 의 한계가 통합으로 사라졌는지: first_by_id 로 두 크레인이 같은 첫 후보를 보는 결정에서, (조각 1 decide 는
    DUP_JOB(16) 을 켜고 배정을 잃었다) 통합된 엔진 decide 와 decide_seq 둘 다 둘째 크레인이 다음 후보를 고른다(또는 WAIT)
    — 위반 0 · 같은 오더를 두 번 고르지 않음 · 두 결과 잎 전부 동일."""
    found = False
    for seed in range(30):
        prof, scn, K, _, _ = make_stage(seed)
        if K < 2:
            continue
        g = Geom.from_profile(prof)
        B, R, T = prof.block.bay_count, prof.block.row_count, prof.block.tier_max
        w, tb = to_block_world(prof, scn, n_max=N_MAX, q_cap=Q_CAP, log_cap=LOG_CAP, c_max=B * R * T + N_MAX)
        runners = _make_runners(g)
        for _ in range(6):
            w, status, c, esc = advance_to_decision(w, g, runners)
            if status == "terminal":
                break
            if status == "decision" and int(jnp.sum(c.open)) >= 2:
                firsts = [int(jnp.argmax(c.cand[k])) for k in range(K) if bool(c.open[k])]
                if len(set(firsts)) < len(firsts):
                    _, _, _, _, dec_jit, dec1_jit = runners
                    d1 = dec1_jit(w, _params(w, "first"))
                    d2 = dec_jit(w, _params(w, "first"))
                    assert int(d1.world.violation) == 0, f"엔진 decide 가 위반 {int(d1.world.violation)} 을 냈다 (재선택이 안 됐다)"
                    assert int(d2.world.violation) == 0
                    bad = _TGE._leaf_diff(d1.world, d2.world)
                    assert not bad, f"엔진 decide 와 decide_seq 가 다른 잎 {bad}"
                    picks = [int(p) for p in np.asarray(d2.pick) if p >= 0]
                    assert len(picks) == len(set(picks)) and picks, picks
                    assert np.array_equal(np.asarray(d1.pick), np.asarray(d2.pick))
                    found = True
                    break
            d = runners[4](w, _params(w, "first"), open_override=esc)
            w = d.world
        if found:
            break
    assert found, "30 무대 안에 '두 크레인이 같은 첫 후보를 보는' 결정이 없었다"


# ───────────────────────────────────────────────── dry-run 오라클 (engine.py:738-763)
def _v5_ref(sim, cid, jid):
    """v5 JobRef (dry_run_commit 의 입력). 대상 컨테이너가 이미 야드에 없거나(반출 완료 → KeyError) 반입 칸이
    없으면(None) v5 는 ref 자체를 못 만든다 — 그런 오더는 (c) 변형에서 뺀다 (배열판은 NO_PLAN 으로 답할 뿐)."""
    j, yc, sp = sim.jobs[jid], sim.fleet.get(cid), sim.fleet.spec(cid)
    try:
        return sim._jobref(j, sp, yc)
    except KeyError:
        return None


@pytest.mark.parametrize("seed", [1, 2, 4, 5, 8, 11, 14, 17, 20, 23])
def test_dry_run_matches_v5_dry_run_commit(seed):
    """`dispatch.dry_run` == v5 `dry_run_commit`: 결정마다 세 가지 joint 선택 — (a) 열린 크레인 전원이 **같은** 첫 후보
    (DUP_JOB 이 나와야 한다) · (b) 각자 첫 후보 · (c) 후보 밖 오더(대상 없음·구간 밖 등 → NO_PLAN 또는 거절 코드) — 를
    넣어 사유 코드열과 수용 계획(dur·rehandles·corridor·lane·slots·moves)을 대조한다. 세계·v5 상태는 바뀌지 않는다."""
    prof, scn, K, policy, _ = make_stage(seed)
    if K < 2:
        pytest.skip("K=1 무대")
    g = Geom.from_profile(prof)
    B, R, T = prof.block.bay_count, prof.block.row_count, prof.block.tier_max
    w, tb = to_block_world(prof, scn, n_max=N_MAX, q_cap=Q_CAP, log_cap=LOG_CAP, c_max=B * R * T + N_MAX)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    runners = _make_runners(g)
    dry = jax.jit(partial(DRY_RUN, g=g))
    n_checked, n_dup, n_noplan, n_reject, n_accept = 0, 0, 0, 0, 0
    while True:
        w, status, c, esc = advance_to_decision(w, g, runners)
        dp = sim.run_until_decision()
        if dp is None or status == "terminal":
            assert dp is None and status == "terminal"
            break
        open_ids = [cid for cid in dp.crane_ids]
        cand = {cid: [r.job_id for r in sim.candidates_for(cid)] for cid in open_ids}
        variants = []
        firsts = [cand[cid][0] for cid in open_ids if cand[cid]]
        if firsts:
            variants.append({cid: firsts[0] for cid in open_ids})                       # (a) 같은 오더
        variants.append({cid: cand[cid][0] for cid in open_ids if cand[cid]})          # (b) 각자 첫 후보
        others = [jid for jid in tb.job_ids if all(jid not in cand[cid] for cid in open_ids)
                  and all(_v5_ref(sim, cid, jid) is not None for cid in open_ids)]
        if others:
            variants.append({cid: others[(i * 3) % len(others)] for i, cid in enumerate(open_ids)})   # (c) 후보 밖
        for choice in variants:
            if not choice:
                continue
            snap = _v5_reservations(sim, tb)
            proj = sim.dry_run_commit({cid: _v5_ref(sim, cid, jid) for cid, jid in choice.items()})
            assert _v5_reservations(sim, tb) == snap                                     # 라이브 상태 불변
            arr = jnp.full((K,), EMPTY_ID, jnp.int32)
            for cid, jid in choice.items():
                arr = arr.at[tb.crane_index[cid]].set(tb.job_index[jid])
            plans, reasons = dry(w, arr)
            for cid in tb.crane_ids:
                k = tb.crane_index[cid]
                r = int(reasons[k])
                if cid not in choice:
                    assert r == -1, (cid, r)
                    continue
                if cid in proj.plans:
                    assert r == 0, f"[s{seed}] t={dp.time} {cid}:{choice[cid]} v5 수용인데 arr 사유 {r}"
                    p5 = proj.plans[cid]
                    ta = (float(plans.dur[k]), int(plans.rehandles[k]), float(plans.lo[k]), float(plans.hi[k]),
                          (tb.lane_ids[int(plans.lane[k])] if int(plans.lane[k]) >= 0 else None),
                          {(int(b) + 1, int(rr) + 1) for b, rr in np.argwhere(np.asarray(plans.slots[k]))},
                          [(tb.cont_ids[int(plans.mv_cont[k, m])], tuple(int(v) for v in plans.mv_src[k, m]),
                            tuple(int(v) for v in plans.mv_dst[k, m]), MV_NAME[int(plans.mv_kind[k, m])])
                           for m in range(int(plans.n_moves[k]))])
                    t5 = (p5.duration_s, p5.rehandles, float(p5.corridor[0]), float(p5.corridor[1]), p5.lane_id,
                          set(p5.slots), _v5_moves(p5))
                    assert ta == t5, f"[s{seed}] t={dp.time} {cid} dry-run 계획 v5={t5} arr={ta}"
                    n_accept += 1
                else:
                    exp = proj.reasons[cid]
                    got = "NO_PLAN" if r == NO_PLAN else CODE_TO_REASON[r]
                    assert got == exp, f"[s{seed}] t={dp.time} {cid}:{choice[cid]} 사유 v5={exp} arr={got}"
                    n_noplan += int(exp == "NO_PLAN"); n_dup += int(exp == "DUP_JOB"); n_reject += int(exp != "NO_PLAN")
                n_checked += 1
        # 실제 결정은 정책대로 진행 (lockstep 과 같은 발판)
        out = runners[4](w, _params(w, policy), open_override=esc)
        for cid in dp.crane_ids:
            ref = v5_choose(policy, sim, cid, sim.candidates_for(cid))
            sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT) if ref is None else CraneAssignment(cid, CandidateKind.SERVE, ref))
        sim.close_decision()
        w = out.world
    REPORT[f"dry-s{seed:02d}"] = dict(dry_checked=n_checked, dry_dup=n_dup, dry_noplan=n_noplan, dry_reject=n_reject, dry_accept=n_accept)
    assert n_checked >= 2 and n_accept >= 1, (n_checked, n_accept)


# ───────────────────────────────────────────────── 보고
def test_zz_report(capsys):
    """무대별 결정·탈출·WAIT·live 재선택 횟수·사유 범주 분포. 5-lock 의 c3·c4 가 실제로 나왔는지 단언한다."""
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    dry = {k: r for k, r in REPORT.items() if k.startswith("dry-")}
    REPORT_LOCK = {k: r for k, r in REPORT.items() if not k.startswith("dry-")}
    tot = {}
    for r in REPORT_LOCK.values():
        for k, v in r["reasons"].items():
            tot[k] = tot.get(k, 0) + v
    n_multi = sum(r["multi_open"] for r in REPORT_LOCK.values())
    n_live = sum(r["live_differs"] for r in REPORT_LOCK.values())
    n_esc = sum(r["escapes"] for r in REPORT_LOCK.values())
    n_wait = sum(r["waits"] for r in REPORT_LOCK.values())
    n_k1 = sum(r["k1_checked"] for r in REPORT_LOCK.values())
    n_taken = sum(r["taken_live"] for r in REPORT_LOCK.values())
    with capsys.disabled():
        print("\n[dispatch equiv report]")
        if dry:
            agg = {k: sum(r[k] for r in dry.values()) for k in next(iter(dry.values()))}
            print(f"  dry_run vs v5 dry_run_commit: stages={len(dry)} {agg}")
        for k, r in REPORT_LOCK.items():
            print(f"  {k:30s} dec={r['decisions']:3d} multi={r['multi_open']:2d} esc={r['escapes']:2d} wait={r['waits']:2d} "
                  f"live≠cand0={r['live_differs']:2d} ev={r['events']:3d} backlog={r['backlog']} tight={int(r['tight'])} "
                  f"inj={int(r['inject'])} lanes={r['lanes']} cost≠0={r['cost']}")
        print(f"  reasons(전 결정·전 (크레인,오더) 범주) = {dict(sorted(tot.items()))}")
        print(f"  multi-open decisions = {n_multi} · live≠cand0 = {n_live} · taken-by-earlier-crane = {n_taken} "
              f"· escapes = {n_esc} · WAIT = {n_wait} · K=1 decide 대조 = {n_k1}")
    assert n_multi >= 5, "크레인 2대 이상이 동시에 열린 결정이 너무 적다"
    assert n_live >= 1, "앞 크레인 배정으로 뒤 크레인 후보가 바뀐 결정이 없었다 — 순차 재선택이 시험되지 않았다"
    assert n_taken >= 1, "앞 크레인이 잡은 오더가 뒤 크레인 후보에서 빠진 일이 한 번도 없었다"
    assert tot.get("TAKEN", 0) == 0, "v5 TAKEN 범주가 나왔다 — _dispatchable 이 먼저 거른다는 전제가 깨졌다"
    assert tot.get("CRANE_INTERFERENCE", 0) >= 1, "c4 CRANE_INTERFERENCE 가 엔진 경로에서 한 번도 안 나왔다"
    assert tot.get("LANE_CONFLICT", 0) >= 1, "c3 LANE_CONFLICT 가 엔진 경로에서 한 번도 안 나왔다"
    assert n_k1 >= 1 and n_wait >= 1
    if dry:
        assert sum(r["dry_dup"] for r in dry.values()) >= 1, "dry_run 에서 DUP_JOB 사유가 한 번도 안 나왔다"
        assert sum(r["dry_accept"] for r in dry.values()) >= 5
