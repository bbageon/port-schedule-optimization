"""배열 엔진(gpu/engine_step.py) 이 v5 `TerminalSimulator` 와 **같은 답**을 내는가 ([[YR-327]] 조각 1 §10).

■ 무엇을 지키나 — 비교는 순서대로, 앞이 깨지면 뒤는 보지 않고 **어디서 처음 갈리는지** 보고한다
  ① 사건 로그 (round(t,6), 종류, 대상) 전열   ② 결정열 (시각·크레인·오더/WAIT)
  ③ 오더 status/assigned_crane/rehandles     ④ 계획 moves (컨테이너·src·dst·종류) + 소요·재조작
  ⑤ 크레인 bay/row/served/completions         ⑥ kpi 정수 (재조작·완료 수·선재조작·재배치)
  ⑦ 격자 (piles·컨테이너 좌표)                ⑧ violation==0 & overflow==0 & terminal
  ⑨ 실수 **비트 동일(==)**: service/done/gate_out, loaded/empty_m, available_at, queue/tail_area, wait 표본,
     장부 적분 셋(block_area·block_tail·terminal_area)·closed_end, 레인 혼잡 적분, 비용 13항·rate 5항,
     vessel_delay·berth_overrun, 턴타임 표본(터미널·블록)·검열 노출
     — 허용오차(1e-6)는 **없다**. 갈리는 항목은 전부 모아 개수·사례와 함께 실패 메시지로 낸다.
       (이전 판은 1e-6 허용이라 advance 의 FMA 로 block_area·truck_wait 가 3.64e-12 갈린 것을
        못 봤고, 그 차이를 'terminal_area 합산 순서' 로 잘못 귀속했다 — 반박 검증 2026-09-25.)
  + jit 없이(파이썬 루프) 돌린 세계가 jit 판과 **잎 전부** 같다 (시험 4)
  + v5 find_slot 이 실제로 부른 질의의 정확 동률 수를 세어 "동률이 시험됐는지" 보고한다

■ 무대
  spec10   명세 §10 그대로 (test_gpu_convert.piece1_profile/scenario 재사용)
  fixture  fixtures.build_minimal_terminal_scenario 를 크레인 1대·본선 없음으로 줄인 것 (레인 2개·인접 1,
           drain 3600, 장부 없음) — 고장/복구 주입 3변형: 없는 크레인(무시) · 유휴 중 DOWN · 작업 중 DOWN(down_pending)
  crowded  장기 대기(SLA 1800 초과)·다른 bay 로 가는 blocker(loaded gantry>0)·규격 셋 — long_wait·crane_travel 식 검증
  censored crowded 를 지평 2000 으로 잘라 종료시점 RUNNING/WAITING 검열 경로 (평가창 밖 완료 사건은 큐에 남는다)
  random   무작위 야드·오더 6 시드 (장부 모드 유무·지평·drain 섞음)
  wait2    "후보가 정확히 2개면 WAIT, 아니면 마지막 후보" 정책 — yielded·_clear_yields·interference rate 경로
  last     "마지막 후보" 정책 — first_by_id 와 다른 선택 순서
  same-time 첫 완료 시각에 도착 2건을 겹쳐 JOB_COMPLETED(우선 0)·BLOCK_ARRIVAL(3) 동시각 큐 순서·due_now 규칙
  dup-range-cap 같은 컨테이너를 겨냥한 반출 2건·담당 구간 밖 대상·재조작 용량 부족 → 미배차 잔존(backlog)
  edge     도착이 정확히 end / end+5e-10 / end 뒤 / drain 창 안 — 평가창 경계
  turntime O > end (end−A 검열) · A ≥ end (표본 제외) — 턴타임 표본 규칙
  vl-deadline 선박 없는 VESSEL_LOAD 2건(마감 100·5000) — vessel_delay_s 적립 경로 (v5 는 JOB_RELEASED 로 시드)

기대값은 손으로 적지 않는다 — v5 를 같은 규칙 정책으로 끝까지 돌려 얻는다.
실행: WSL venv · x64 CPU. `XLA_FLAGS=--xla_allow_excess_precision=false` 는 관례로 붙이지만 **FMA 를 막지
못한다** (실측) — 동등성은 gpu/exact.py 의 optimization_barrier 규약이 지킨다.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu import engine_step as ES                                        # noqa: E402
from yard_rl.v6.gpu.engine_step import first_by_id, run_jit, run_python             # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import (event_log_from_arrays, event_stream_hash,  # noqa: E402
                                         from_block_world, to_block_world)
from yard_rl.v6.gpu.state import (COST_TERMS, EMPTY_ID, MV_REHANDLE, MV_RETRIEVE, MV_STORE,   # noqa: E402
                                  V_DECISION_COVERAGE, V_LEDGER_UNREGISTERED, V_RESERVE_REJECT,
                                  V_STEPS_EXHAUSTED, block_turn_time_s, censored_exposure_s,
                                  censored_turn_time_s, violation_names)
# v5 정본
from yard_rl.v6.world.contract.schema import COST_TERMS as V5_COST_TERMS, CandidateKind   # noqa: E402
from yard_rl.v6.world.contract.state import LaneGraph                               # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, JobFlow, LoadStatus        # noqa: E402
from yard_rl.v6.world.domain.models import BlockGeometry, Container, Job           # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                    # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator   # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec                   # noqa: E402
from yard_rl.v6.world.integrated.scenario import TerminalScenario                   # noqa: E402

FT20, FT40, FT45 = ContainerSize.FT20, ContainerSize.FT40, ContainerSize.FT45
MV_NAME = {MV_REHANDLE: "REHANDLE", MV_RETRIEVE: "RETRIEVE", MV_STORE: "STORE"}

#: 시험 전체 집계 (마지막 시험이 보고) — 무대별 (스텝 수, find_slot 질의 수, 정확 동률 수, 실수 최대 오차, WAIT 수 …)
REPORT: dict[str, dict] = {}


def _load_convert_test():
    """tests/v6/test_gpu_convert.py 의 §10 빌더를 경로로 불러온다 (tests/ 에 __init__ 이 없다)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_gpu_convert.py")
    spec = importlib.util.spec_from_file_location("_tgc_for_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TGC = _load_convert_test()
piece1_profile, piece1_scenario = _TGC.piece1_profile, _TGC.piece1_scenario


# ───────────────────────────────────────────────── 무대 빌더
def _c(cid, bay, row, tier, size=FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _out(jid, target, gate_in, arrival, exit_s=60.0):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, target_container=target, exit_travel_s=exit_s)


def _in(jid, gate_in, arrival, size=FT40, exit_s=60.0):
    return Job(job_id=jid, flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=gate_in,
               actual_block_arrival=arrival, inbound_size=size, inbound_load=LoadStatus.FULL,
               exit_travel_s=exit_s)


def _scn(sid, containers, jobs, horizon=7200.0, drain=0.0):
    return TerminalScenario(scenario_id=sid, seed=0, horizon_s=horizon, drain_window_s=drain,
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


def fixture_profile_k1():
    """fixtures 프로파일에서 크레인을 YC-A 하나로 (40×4×4 · 레인 L1,L2 인접 · 이송 2대는 안 쓰임)."""
    base = fixtures.build_integrated_profile()
    return replace(base, cranes=(fixtures._spec("YC-A"),))


def fixture_scenario_k1(injected_target: str | None, down_up: tuple[float, float] = (2000.0, 2600.0)):
    """본선·연계 작업·PLAN_CHANGE 를 뺀 fixture 시나리오. 고장/복구 주입 대상만 바꾼다."""
    from yard_rl.v6.world.integrated.scenario import InjectedEvent
    base = fixtures.build_minimal_terminal_scenario()
    jobs = [j for j in base.jobs if j.vessel_id is None]
    # PLAN_CHANGE 는 그대로 둔다 — 선박이 없으니 v5 도 무동작(1049-1051행), 배열판은 target -1 → 무동작·실격 아님
    injected = [ie for ie in base.injected_events if ie.kind == "PLAN_CHANGE"]
    if injected_target is not None:
        injected += [InjectedEvent(down_up[0], "EQUIPMENT_DOWN", injected_target),
                     InjectedEvent(down_up[1], "EQUIPMENT_UP", injected_target)]
    return TerminalScenario(scenario_id=f"fixture-k1-{injected_target}-{down_up[0]}", seed=0,
                            horizon_s=base.horizon_s, drain_window_s=base.drain_window_s,
                            containers=dict(base.containers), jobs=jobs, vessels=[],
                            injected_events=injected)


def crowded_scenario(horizon_s: float = 7200.0) -> TerminalScenario:
    """장기 대기·타 bay 재조작·규격 셋. bay 5 row 1 = A1..A4(4단), row 2~4 는 꼭대기까지(갈 곳 없음),
    (4,1) FT20 · (2,3) FT20×2 · (7,1) G1~G3 · (8,2) C3. 트럭 12대가 300~800 초에 몰려 뒤쪽은 SLA 를 넘긴다."""
    containers = {"A1": _c("A1", 5, 1, 1), "A2": _c("A2", 5, 1, 2), "A3": _c("A3", 5, 1, 3), "A4": _c("A4", 5, 1, 4),
                  "C3": _c("C3", 8, 2, 1), "E1": _c("E1", 4, 1, 1, FT20),
                  "F1": _c("F1", 2, 3, 1, FT20), "F2": _c("F2", 2, 3, 2, FT20),
                  "G1": _c("G1", 7, 1, 1), "G2": _c("G2", 7, 1, 2), "G3": _c("G3", 7, 1, 3)}
    for r in (2, 3, 4):
        for t in (1, 2, 3, 4):
            containers[f"D{r}{t}"] = _c(f"D{r}{t}", 5, r, t)
    jobs = [_out("J-OUT-A1", "A1", 0.0, 300.0), _out("J-OUT-A2", "A2", 0.0, 310.0),
            _out("J-OUT-A3", "A3", 0.0, 320.0), _out("J-OUT-A4", "A4", 0.0, 330.0),
            _out("J-OUT-G1", "G1", 100.0, 340.0), _out("J-OUT-F1", "F1", 100.0, 350.0),
            _out("J-OUT-C3", "C3", 100.0, 360.0), _in("J-IN-1", 100.0, 400.0, FT40),
            _in("J-IN-2", 200.0, 500.0, FT20), _in("J-IN-3", 300.0, 600.0, FT45),
            _out("J-OUT-G2", "G2", 400.0, 700.0), _in("J-IN-4", 500.0, 800.0, FT40)]
    return _scn(f"crowded-{int(horizon_s)}", containers, jobs, horizon=horizon_s)


def random_scenario(seed: int, B: int = 10, R: int = 4, T: int = 4) -> TerminalScenario:
    """무작위 야드(칸 35%·단일 규격 pile)·반출 3~7·반입 1~4·장부 모드 60%·지평/drain 섞음."""
    rng = random.Random(seed)
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
    targets = rng.sample(sorted(containers), min(len(containers), rng.randint(3, 7)))
    jobs = []
    for i, tgt in enumerate(targets):
        arr = round(rng.uniform(0.0, 2500.0), 3)
        jobs.append(_out(f"J-OUT-{i:02d}", tgt, max(0.0, arr - rng.uniform(0.0, 600.0)), arr, ex))
    for i in range(rng.randint(1, 4)):
        arr = round(rng.uniform(0.0, 2500.0), 3)
        jobs.append(_in(f"J-IN-{i:02d}", max(0.0, arr - rng.uniform(0.0, 600.0)), arr,
                        rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0], ex))
    return TerminalScenario(scenario_id=f"random-{seed}", seed=seed,
                            horizon_s=rng.choice([1800.0, 3600.0, 7200.0]),
                            drain_window_s=rng.choice([0.0, 600.0]),
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


def same_time_scenario() -> TerminalScenario:
    """spec10 의 첫 JOB_COMPLETED 시각에 도착 2건(반출 C9·반입)을 정확히 겹친다 — 큐 키 (시각, 우선순위, seq)."""
    base = piece1_scenario()
    sim0, _, _ = run_v5(piece1_profile(), base)
    t_c = next(t for (t, k, _) in sim0.event_log if k == "JOB_COMPLETED")
    containers = dict(base.containers)
    containers["C9"] = _c("C9", 3, 3, 1)
    jobs = list(base.jobs) + [_out("J-OUT-9", "C9", 0.0, t_c), _in("J-IN-9", 0.0, t_c, FT40)]
    return _scn("same-time", containers, jobs)


def dup_range_cap_scenario() -> TerminalScenario:
    """같은 대상(C1) 반출 2건 · bay 8 대상(구간 밖일 수 있음) · bay 5 row 2~4 꼭대기(재조작 칸 부족)."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 5, 1, 2), "C3": _c("C3", 8, 2, 1)}
    for r in (2, 3, 4):
        for t in (1, 2, 3, 4):
            containers[f"D{r}{t}"] = _c(f"D{r}{t}", 5, r, t)
    jobs = [_out("J-OUT-1", "C1", 0.0, 300.0), _out("J-OUT-1b", "C1", 0.0, 320.0),
            _out("J-OUT-2", "C3", 0.0, 300.0),
            _out("J-OUT-3", "C2", 0.0, 900.0), _in("J-IN-1", 0.0, 1500.0, FT40)]
    return _scn("dup-range-cap", containers, jobs)


def _narrow_profile(lo: int, hi: int):
    return replace(piece1_profile(), cranes=(replace(fixtures._spec("YC-A"), service_bay_min=lo, service_bay_max=hi),))


def edge_end_scenario() -> TerminalScenario:
    """도착이 정확히 end(7200) · end+5e-10 (EPS 안) · end 뒤(7300)."""
    base = piece1_scenario()
    containers = dict(base.containers)
    containers["C9"] = _c("C9", 3, 3, 1)
    containers["C8"] = _c("C8", 2, 2, 1)
    jobs = list(base.jobs) + [_out("J-OUT-9", "C9", 0.0, 7200.0), _out("J-OUT-8", "C8", 0.0, 7300.0),
                              _in("J-IN-9", 0.0, 7200.0 + 5e-10, FT40)]
    return _scn("edge-end", containers, jobs)


def edge_drain_scenario() -> TerminalScenario:
    """drain 600: HORIZON 7200 · end 7800 · 도착 7500(HORIZON 뒤·end 앞) · 7800 정확."""
    base = piece1_scenario()
    containers = dict(base.containers)
    containers["C9"] = _c("C9", 3, 3, 1)
    containers["C8"] = _c("C8", 2, 2, 1)
    jobs = list(base.jobs) + [_out("J-OUT-9", "C9", 0.0, 7500.0), _out("J-OUT-8", "C8", 0.0, 7800.0)]
    return _scn("edge-drain", containers, jobs, drain=600.0)


def turntime_edge_scenario() -> TerminalScenario:
    """O > end (exit_travel 8000 → end−A 검열) · A ≥ end (표본 제외) · 보통 1건."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 8, 2, 1), "C3": _c("C3", 3, 3, 1)}
    jobs = [_out("J-OUT-1", "C1", 0.0, 300.0, 8000.0),
            _out("J-OUT-2", "C2", 7500.0, 7600.0, 60.0),
            _out("J-OUT-3", "C3", 100.0, 500.0, 60.0)]
    return _scn("turntime-edge", containers, jobs)


def vessel_deadline_scenario() -> TerminalScenario:
    """선박 없는 VESSEL_LOAD 2건(마감 100 → 지각, 5000 → 정시) + 반출 1건 — kpis.job_completed 의 지각 합."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 5, 1, 2), "C3": _c("C3", 8, 2, 1)}
    jobs = [Job(job_id="J-VL-1", flow=JobFlow.VESSEL_LOAD, release_time=0.0, actual_gate_in=None,
                actual_block_arrival=None, deadline=100.0, target_container="C1"),
            Job(job_id="J-VL-2", flow=JobFlow.VESSEL_LOAD, release_time=50.0, actual_gate_in=None,
                actual_block_arrival=None, deadline=5000.0, target_container="C3"),
            _out("J-OUT-1", "C2", 0.0, 300.0)]
    return _scn("vl-deadline", containers, jobs)


# ───────────────────────────────────────────────── 정책 (v5 chooser ↔ 배열 policy_fn 짝)
def chooser_first(cands):
    return cands[0] if cands else None


def chooser_last(cands):
    return cands[-1] if cands else None


def chooser_wait2(cands):
    """후보가 정확히 2개면 WAIT, 아니면 마지막 후보."""
    if not cands or len(cands) == 2:
        return None
    return cands[-1]


def policy_last(params, x, mask):
    N = mask.shape[1]
    last = N - 1 - jnp.argmax(mask[:, ::-1], axis=1)
    return jnp.where(jnp.any(mask, axis=1), last, EMPTY_ID).astype(jnp.int32)


def policy_wait2(params, x, mask):
    N = mask.shape[1]
    n_c = jnp.sum(mask, axis=1)
    last = N - 1 - jnp.argmax(mask[:, ::-1], axis=1)
    return jnp.where(n_c == 2, EMPTY_ID, jnp.where(n_c > 0, last, EMPTY_ID)).astype(jnp.int32)


POLICIES = {"first": (chooser_first, first_by_id), "last": (chooser_last, policy_last),
            "wait2": (chooser_wait2, policy_wait2)}


# ───────────────────────────────────────────────── v5 쪽 도구
class _TieCounter:
    """v5 `YardStacks.find_slot` 을 감싸 실제 질의마다 정확 동률(최선 비용 == 차선 비용)을 센다."""

    def __init__(self):
        self.calls = 0
        self.ties = 0

    @contextmanager
    def watching(self, sim):
        stk, geom = sim.stacks, sim.profile.block
        orig = stk.find_slot

        def keys(size, spec, nb, nr, excl):
            out = []
            for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
                gc = abs(nb - bay) * geom.bay_length_m
                for row in range(1, geom.row_count + 1):
                    if (bay, row) in excl:
                        continue
                    pile = stk._stacks.get((bay, row))
                    top = len(pile) if pile else 0
                    if top >= geom.tier_max or (pile and stk.containers[pile[-1]].size != size):
                        continue
                    out.append((gc + abs(nr - row) * geom.row_width_m + top * geom.tier_height_m, bay, row))
            out.sort()
            return out

        def wrapped(size, spec, near_bay, near_row, exclude=frozenset()):
            ks = keys(size, spec, near_bay, near_row, exclude)
            self.calls += 1
            self.ties += int(len(ks) >= 2 and ks[0][0] == ks[1][0])
            return orig(size, spec, near_bay, near_row, exclude=exclude)

        stk.find_slot = wrapped
        try:
            yield self
        finally:
            del stk.find_slot


def _v5_moves(plan):
    return [(m.container_id, tuple(m.src), tuple(m.dst),
             "STORE" if m.inbound is not None else ("RETRIEVE" if m.depart else "REHANDLE"))
            for m in plan.moves]


def run_v5(prof, scn, chooser=chooser_first):
    """명세 §10 구동 (기본 first-by-id). 반환 (sim, 결정열 [(t, crane_ids, [(crane, job|None, moves, dur, rehandles)])], 동률 계수)."""
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    decisions = []
    tc = _TieCounter()
    with tc.watching(sim):
        while (dp := sim.run_until_decision()) is not None:
            assigns = []
            for c in dp.crane_ids:
                ref = chooser(sim.candidates_for(c))
                assigns.append(CraneAssignment(c, CandidateKind.SERVE, ref) if ref is not None
                               else CraneAssignment(c, CandidateKind.WAIT))
            sim.commit_decisions(assigns)
            rec = []
            for a in assigns:
                if a.action == CandidateKind.SERVE:
                    p = sim.active_plan(a.crane_id)
                    rec.append((a.crane_id, a.job_ref.job_id, _v5_moves(p), p.duration_s, p.rehandles))
                else:
                    rec.append((a.crane_id, None, [], None, None))
            decisions.append((dp.time, tuple(dp.crane_ids), rec))
    return sim, decisions, tc


# ───────────────────────────────────────────────── 배열 쪽 도구
def _caps(scn, prof):
    """명세 capacities: S_max = 8N+256 · Q = 4N(≥32) · E = S_max + N (큐 사건 + 결정마다 DISPATCH K줄)."""
    n0 = len(scn.jobs)
    n_max = max(8, 1 << (n0 - 1).bit_length())
    s_max = 8 * n_max + 256
    return dict(n_max=n_max, q_cap=max(32, 4 * n_max), log_cap=s_max + n_max), s_max


def run_array(prof, scn, *, use_jit=True, policy_fn=first_by_id):
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    runner = run_jit if use_jit else run_python
    w, trace = runner(w0, None, g, policy_fn, s_max)
    return w, trace, tb


def _array_decisions(w, trace, tb):
    """StepTrace → v5 결정열 모양 [(t, crane_ids, [(crane, job|None, moves, dur, rehandles)])]."""
    n_steps = int(w.steps)
    tr = jax.tree_util.tree_map(lambda a: np.asarray(a)[:n_steps], trace)
    out = []
    K = len(tb.crane_ids)
    for i in range(n_steps):
        if not tr.decided[i]:
            continue
        rec = []
        ks = [k for k in range(K) if tr.open[i][k]]
        for k in ks:
            n = int(tr.pick[i][k])
            if n < 0:
                rec.append((tb.crane_ids[k], None, [], None, None))
                continue
            moves = []
            for m in range(int(tr.plan_n_moves[i][k])):
                moves.append((tb.cont_ids[int(tr.plan_mv_cont[i][k][m])],
                              tuple(int(v) for v in tr.plan_mv_src[i][k][m]),
                              tuple(int(v) for v in tr.plan_mv_dst[i][k][m]),
                              MV_NAME[int(tr.plan_mv_kind[i][k][m])]))
            rec.append((tb.crane_ids[k], tb.job_ids[n], moves, float(tr.plan_dur[i][k]),
                        int(tr.plan_rehandles[i][k])))
        out.append((float(tr.clock[i]), tuple(tb.crane_ids[k] for k in ks), rec))
    return out


# ───────────────────────────────────────────────── 비교 (①~⑨ 순서)
def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), (b[len(a)] if len(b) > len(a) else None)
    return None


def _finite_sorted(arr):
    a = np.asarray(arr)
    return sorted(float(x) for x in a[~np.isnan(a)])


def compare(sim, v5_dec, w, trace, tb, label: str) -> float:
    """앞 항목이 깨지면 그 자리에서 실패 메시지(어디서 처음 갈리는지)를 낸다.

    ⑨ 는 항목을 **전부** 대조한 뒤 갈린 것을 한꺼번에 보고한다 (개수·이름·양쪽 값·차이).
    반환 = 실수 최대 절대오차 (통과하면 0.0).
    """
    d = from_block_world(w, tb)
    # ① 사건 로그. ★없는 크레인을 겨냥한 고장/복구 주입은 배열 세계가 id 문자열을 못 담는다(호스트가 -1 로,
    #   host_convert 294행) — v5 는 원문 'YC-B' 를 로그에 남기고 처리는 둘 다 무시. 그 payload 만 '' 로 맞춰 본다.
    n_norm = 0
    v5log = []
    for (t, k, p) in sim.event_log:
        if k in ("EQUIPMENT_DOWN", "EQUIPMENT_UP") and p not in tb.crane_ids:
            p, n_norm = "", n_norm + 1
        if k == "PLAN_CHANGE" and p not in tb.vessel_ids:      # 모르는 선박 — 같은 표현 한계
            p, n_norm = "", n_norm + 1
        v5log.append((round(t, 6), k, p))
    arlog = [(round(t, 6), k, p) for (t, k, p) in d["event_log"]]
    diff = _first_diff(v5log, arlog)
    if diff is not None:
        i, x, y = diff
        ctx = "\n".join(f"    v5 {v5log[j] if j < len(v5log) else '-'}  |  arr {arlog[j] if j < len(arlog) else '-'}"
                        for j in range(max(0, i - 3), min(max(len(v5log), len(arlog)), i + 3)))
        pytest.fail(f"[{label}] ① 사건 로그가 {i}번째에서 갈린다: v5={x} arr={y}\n{ctx}\n"
                    f"  (v5 {len(v5log)}건 · arr {len(arlog)}건, violation={d['violation']} {d['violation_names']})")
    if n_norm == 0:
        assert event_stream_hash(w, tb) == sim.event_stream_hash(), f"[{label}] ① 해시"
    REPORT.setdefault(label, {})["log_payload_normalized"] = n_norm
    # ② 결정열 (시각·크레인·오더/WAIT)
    ar_dec = _array_decisions(w, trace, tb)
    v5_short = [(round(t, 6), cs, [(c, j) for (c, j, *_r) in rec]) for (t, cs, rec) in v5_dec]
    ar_short = [(round(t, 6), cs, [(c, j) for (c, j, *_r) in rec]) for (t, cs, rec) in ar_dec]
    diff = _first_diff(v5_short, ar_short)
    if diff is not None:
        i, x, y = diff
        pytest.fail(f"[{label}] ② 결정열이 {i}번째에서 갈린다: v5={x} arr={y}")
    # ③ 오더 status/assigned_crane/rehandles
    for jid, j in sim.jobs.items():
        a = d["jobs"][jid]
        assert (a["status"], a["assigned_crane"], a["rehandle_count"]) == (j.status.name, j.assigned_crane, j.rehandle_count), \
            f"[{label}] ③ 오더 {jid}: arr={(a['status'], a['assigned_crane'], a['rehandle_count'])} v5={(j.status.name, j.assigned_crane, j.rehandle_count)}"
    # ④ 계획 moves + 소요·재조작 (소요는 travel 이 비트 동일이라 == 로)
    for i, ((t5, _c5, r5), (ta, _ca, ra)) in enumerate(zip(v5_dec, ar_dec)):
        for (c5, j5, m5, d5, h5), (ca, ja, ma, da, ha) in zip(r5, ra):
            assert m5 == ma, f"[{label}] ④ 결정 {i} ({c5}:{j5}) moves 다름:\n  v5={m5}\n  arr={ma}"
            assert h5 == ha, f"[{label}] ④ 결정 {i} rehandles v5={h5} arr={ha}"
            if d5 is not None:
                assert d5 == da, f"[{label}] ④ 결정 {i} duration v5={d5!r} arr={da!r}"
    # ⑤ 크레인
    for cid in sim.fleet.ids():
        yc, a = sim.fleet.get(cid), d["cranes"][cid]
        got = (a["position_bay"], a["trolley_row"], a["served_count"], a["recent_completions"], a["down"], a["down_pending"], a["yielded"], a["assigned_job"])
        exp = (yc.state.position_bay, yc.state.trolley_row, yc.served_count, yc.recent_completions, yc.down, yc.down_pending, yc.yielded, yc.state.assigned_job)
        assert got == exp, f"[{label}] ⑤ 크레인 {cid}: arr={got} v5={exp}"
    # ⑥ kpi 정수
    ks = sim.kpis.snapshot()
    got6 = (d["kpi"]["rehandle_count"], d["kpi"]["completed_external"], d["kpi"]["completed_vessel"],
            d["kpi"]["pre_rehandle_count"], d["kpi"]["positioning_count"])
    exp6 = (ks.rehandle_count, ks.completed_external, ks.completed_vessel, ks.pre_rehandle_count, ks.positioning_count)
    assert got6 == exp6, f"[{label}] ⑥ kpi 정수 arr={got6} v5={exp6}"
    # ⑦ 격자
    assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}, f"[{label}] ⑦ piles"
    assert d["containers"] == {cid: (c.bay, c.row, c.tier) for cid, c in sim.stacks.containers.items()}, f"[{label}] ⑦ 컨테이너 좌표"
    # ⑧ 실격 없음·종료
    assert d["violation"] == 0 and d["overflow"] == 0 and d["terminal"], \
        f"[{label}] ⑧ violation={d['violation']} {d['violation_names']} overflow={d['overflow']} terminal={d['terminal']}"
    assert sim.terminal and d["clock"] == sim.clock and d["last_decision_at"] == sim._last_decision_at
    # ⑨ 실수 — 전부 비트 동일(==). 갈린 항목은 모아서 한 번에 보고한다.
    bad: list[tuple[str, object, object, float]] = []
    errs: list[float] = []

    def exact(name, a, b):
        if a is None or b is None:
            if a != b:
                bad.append((name, a, b, math.inf))
            return
        err = abs(a - b)
        errs.append(err)
        if a != b:
            bad.append((name, a, b, err))

    for jid, j in sim.jobs.items():
        a = d["jobs"][jid]
        exact(f"{jid}.service_start", a["service_start"], j.service_start)
        exact(f"{jid}.service_end", a["service_end"], j.service_end)
        exact(f"{jid}.actual_gate_out", a["actual_gate_out"], j.actual_gate_out)
    for cid in sim.fleet.ids():
        yc, a = sim.fleet.get(cid), d["cranes"][cid]
        exact(f"{cid}.loaded_m", a["loaded_travel_m"], yc.state.loaded_travel_m)
        exact(f"{cid}.empty_m", a["empty_travel_m"], yc.state.empty_travel_m)
        exact(f"{cid}.available_at", a["available_at"], yc.state.available_at)
    exact("queue_area", d["kpi"]["queue_area_s"], ks.queue_area_s)
    exact("tail_area", d["kpi"]["tail_area_s"], ks.tail_area_s)
    exact("kpi.loaded", d["kpi"]["loaded_gantry_m"], ks.loaded_gantry_m)
    exact("kpi.empty", d["kpi"]["empty_gantry_m"], ks.empty_gantry_m)
    exact("kpi.vessel_delay_s", d["kpi"]["vessel_delay_s"], ks.vessel_delay_s)
    exact("kpi.berth_overrun_s", d["kpi"]["berth_overrun_s"], ks.berth_overrun_s)
    ws_v5 = sorted(sim.kpis.wait_samples_s)
    ws_ar = sorted(v["wait_sample"] for v in d["jobs"].values() if v["wait_sample"] is not None)
    assert len(ws_v5) == len(ws_ar), f"[{label}] ⑨ wait 표본 수 arr={len(ws_ar)} v5={len(ws_v5)}"
    for i, (a, b) in enumerate(zip(ws_ar, ws_v5)):
        exact(f"wait_sample[{i}]", a, b)
    if sim.time_ledger is not None:
        exact("block_area", d["ledger"]["block_area_s"], sim.time_ledger.block_area_s)
        exact("block_tail", d["ledger"]["block_tail_area_s"], sim.time_ledger.block_tail_area_s)
        exact("terminal_area", d["ledger"]["terminal_area_s"], sim.time_ledger.terminal_area_s)
        exact("closed_end", d["ledger"]["closed_end_s"], sim.time_ledger.closed_end_s)
        # 턴타임 표본 규칙 (time_contract.py:112-142) — 정렬열 대조 + 검열 노출
        tt_v5 = sorted(sim.time_ledger.terminal_turntime_samples_s())
        tt_ar = _finite_sorted(censored_turn_time_s(w.orders, w.end_s))
        assert len(tt_v5) == len(tt_ar), f"[{label}] ⑨ 터미널 턴타임 표본 수 arr={len(tt_ar)} v5={len(tt_v5)}: arr={tt_ar} v5={tt_v5}"
        for i, (a, b) in enumerate(zip(tt_ar, tt_v5)):
            exact(f"terminal_turntime[{i}]", a, b)
        bt_v5 = sorted(sim.time_ledger.block_turntime_samples_s())
        bt_ar = _finite_sorted(block_turn_time_s(w.orders, w.end_s))
        assert len(bt_v5) == len(bt_ar), f"[{label}] ⑨ 블록 턴타임 표본 수 arr={len(bt_ar)} v5={len(bt_v5)}"
        for i, (a, b) in enumerate(zip(bt_ar, bt_v5)):
            exact(f"block_turntime[{i}]", a, b)
        exact("censored_exposure", float(censored_exposure_s(w.orders, w.end_s)), sim.time_ledger.censored_exposure_s())
    else:
        assert d["ledger"]["block_area_s"] == 0.0 and d["ledger"]["terminal_area_s"] == 0.0 \
            and d["ledger"]["closed_end_s"] is None, f"[{label}] ⑨ 장부 없는데 적분"
    exact("lane_cong_area", d["lane_cong_area_s"], sim.lanes.cong_area_s)
    ep = sim.cost.episode_raw()
    assert tuple(COST_TERMS) == tuple(V5_COST_TERMS)
    for t in COST_TERMS:
        exact(f"cost.{t}", d["cost_episode"][t], ep[t])
    for t, v in sim.cost._rate.items():
        exact(f"rate.{t}", d["cost_rate"][t], v)
    REPORT.setdefault(label, {})["float_mismatches"] = len(bad)
    if bad:
        lines = "\n".join(f"    {n}: arr={a!r} v5={b!r} (차 {e:.3e})" for (n, a, b, e) in bad)
        pytest.fail(f"[{label}] ⑨ 실수 {len(bad)}/{len(errs)} 항목이 비트 수준에서 갈린다:\n{lines}")
    return max(errs) if errs else 0.0


def _run_and_compare(prof, scn, label, *, use_jit=True, policy="first"):
    chooser, policy_fn = POLICIES[policy]
    sim, dec, tc = run_v5(prof, scn, chooser)
    w, trace, tb = run_array(prof, scn, use_jit=use_jit, policy_fn=policy_fn)
    err = compare(sim, dec, w, trace, tb, label)
    n_wait = sum(1 for (_, _, rec) in dec for (_, j, *_r) in rec if j is None)
    times = [t for (t, _, _) in sim.event_log]
    same_time_kinds = 0
    for t in set(times):
        kinds = {k for (tt, k, _) in sim.event_log if tt == t and k != "DISPATCH"}
        same_time_kinds += int(len(kinds) >= 2)
    REPORT.setdefault(label, {}).update(
        steps=int(w.steps), decisions=len(dec), waits=n_wait, events=len(sim.event_log),
        same_time=same_time_kinds, find_slot_calls=tc.calls, exact_ties=tc.ties, max_float_err=err,
        nonzero_cost={k: round(v, 3) for k, v in sim.cost.episode_raw().items() if v},
        backlog=sim.unfinished_backlog(), interference=sim.cost.episode_raw()["interference"])
    return sim, w, trace, tb


# ───────────────────────────────────────────────── ① 명세 §10 무대
def test_spec10_equivalence_jit():
    sim, w, trace, tb = _run_and_compare(piece1_profile(), piece1_scenario(), "spec10")
    # 명세 §10 이 손으로 적은 기대 거동 — 결정 5회, 첫 결정 J-OUT-1(재조작 1), 두 번째 J-OUT-2, 셋째 J-IN-1, 넷째 J-IN-2, 다섯째 J-OUT-3
    assert [j for (_, _, rec) in _array_decisions(w, trace, tb) for (_, j, *_r) in rec] == \
        ["J-OUT-1", "J-OUT-2", "J-IN-1", "J-IN-2", "J-OUT-3"]
    assert sim.jobs["J-OUT-1"].rehandle_count == 1
    # 반박 검증이 지적한 값 — lane_cong 은 0 이 아니다 (크레인 점유 시간 합), crane_travel·long_wait 은 0
    ep = sim.cost.episode_raw()
    assert ep["lane_cong"] > 0 and ep["crane_travel"] == 0.0 and ep["long_wait"] == 0.0


def test_spec10_python_loop_matches_jit():
    """시험 4) — jit 없이 파이썬 루프로 step 을 반복해도 잎 전부(비트) 같고, v5 와도 같다."""
    prof, scn = piece1_profile(), piece1_scenario()
    w_j, _, tb = run_array(prof, scn, use_jit=True)
    w_p, tr_p, _ = run_array(prof, scn, use_jit=False)
    lj, lp = jax.tree_util.tree_leaves(w_j), jax.tree_util.tree_leaves(w_p)
    assert len(lj) == len(lp)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(w_j)]
    bad = [names[i] for i, (a, b) in enumerate(zip(lj, lp))
           if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)]
    assert not bad, f"jit 판과 파이썬 루프 판이 다른 잎: {bad}"
    sim, dec, _ = run_v5(prof, scn)
    compare(sim, dec, w_p, tr_p, tb, "spec10-python")


# ───────────────────────────────────────────────── ② fixture 축소판 (레인 2개·drain·장부 없음·고장/복구)
@pytest.mark.parametrize("target,down_up", [
    ("YC-B", (2000.0, 2600.0)),     # 없는 크레인 → v5 는 로그만 남기고 무시 (866, 869행)
    ("YC-A", (2000.0, 2600.0)),     # 유휴 중 DOWN → down, UP 까지 결정 없음
    ("YC-A", (310.0, 400.0)),       # 작업 중 DOWN → down_pending → 완료 후 down → UP
], ids=["ignored-crane", "idle-down", "busy-down-pending"])
def test_fixture_k1_equivalence(target, down_up):
    prof, scn = fixture_profile_k1(), fixture_scenario_k1(target, down_up)
    sim, w, trace, tb = _run_and_compare(prof, scn, f"fixture-{target}-{int(down_up[0])}")
    assert sim.time_ledger is None                      # 장부 없는 경로 (queue_area 가 truck_wait)
    kinds = [k for (_, k, _) in sim.event_log]
    assert "EQUIPMENT_DOWN" in kinds and "EQUIPMENT_UP" in kinds and "PLAN_CHANGE" in kinds
    if target == "YC-A":
        assert any(k == "DISPATCH" for k in kinds)


# ───────────────────────────────────────────────── ③ 혼잡 무대 — long_wait·crane_travel 이 실제로 0 아님
def test_crowded_equivalence_exercises_all_integrals():
    prof = piece1_profile()
    sim, w, trace, tb = _run_and_compare(prof, crowded_scenario(), "crowded")
    ep = sim.cost.episode_raw()
    assert ep["long_wait"] > 0, "무대가 SLA 초과 대기를 만들지 못했다 — long_wait 식이 검증되지 않는다"
    assert ep["crane_travel"] > 0, "무대가 다른 bay 로 가는 blocker 를 만들지 못했다 — crane_travel 식이 검증되지 않는다"
    assert ep["rehandle"] >= 3 and ep["lane_cong"] > 0 and ep["truck_wait"] > 0
    assert sim.kpis.snapshot().tail_area_s > 0


def test_censored_end_equivalence():
    """지평 2000: 종료시점 RUNNING/WAITING 이 남고 평가창 밖 완료 사건이 큐에 남는다 (V_BUSY_NO_EVENT 아님)."""
    prof = piece1_profile()
    sim, w, trace, tb = _run_and_compare(prof, crowded_scenario(horizon_s=2000.0), "censored")
    assert sim.unfinished_backlog() > 0
    assert any(c.state.assigned_job is not None for c in sim.fleet.all())   # 종료시점 작업 중
    assert int(jnp.sum(w.queue.time < jnp.inf)) >= 1                         # 평가창 밖 사건이 큐에 남음
    assert len(sim.kpis.wait_samples_s) == sum(1 for j in sim.jobs.values() if j.status.name not in ("PLANNED",))


# ───────────────────────────────────────────────── ④ 무작위 무대
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_random_equivalence(seed):
    prof = piece1_profile()
    _run_and_compare(prof, random_scenario(seed), f"random-{seed}")


# ───────────────────────────────────────────────── ⑤ 시험 공백 메우기 (반박 검증 probe3 의 무대 4종)
@pytest.mark.parametrize("stage", ["spec10", "crowded", "random-6"])
def test_wait_policy_equivalence(stage):
    """WAIT 를 실제로 내는 정책 — yielded 설정(686행)·_clear_yields 해제·interference rate(Σyielded)·재질문 규칙."""
    prof = piece1_profile()
    scn = {"spec10": piece1_scenario, "crowded": crowded_scenario, "random-6": lambda: random_scenario(6)}[stage]()
    sim, w, trace, tb = _run_and_compare(prof, scn, f"wait2-{stage}", policy="wait2")
    assert REPORT[f"wait2-{stage}"]["waits"] >= 1, "이 무대에서 WAIT 가 한 번도 안 나왔다"
    if stage != "random-6":
        assert sim.cost.episode_raw()["interference"] > 0      # yielded 가 다음 사건까지 rate 로 적립됐다


@pytest.mark.parametrize("stage", ["crowded", "random-6"])
def test_last_candidate_policy_equivalence(stage):
    """first_by_id 와 다른 선택 순서 — 결정열이 정책에 따라 바뀌어도 v5 와 같은가."""
    prof = piece1_profile()
    scn = {"crowded": crowded_scenario, "random-6": lambda: random_scenario(6)}[stage]()
    _run_and_compare(prof, scn, f"last-{stage}", policy="last")


@pytest.mark.parametrize("policy", ["first", "wait2"])
def test_same_time_completion_and_arrival(policy):
    """같은 시각 JOB_COMPLETED(우선 0) + BLOCK_ARRIVAL(우선 3) ×2 — 큐 3단 키와 due_now 규칙이 엔진 수준에서."""
    prof = piece1_profile()
    scn = same_time_scenario()
    sim, w, trace, tb = _run_and_compare(prof, scn, f"same-time-{policy}", policy=policy)
    if policy == "first":      # 겹친 시각은 first-by-id 의 첫 완료 시각 — wait2 는 첫 완료가 달라 겹치지 않는다
        t_c = next(t for (t, k, _) in sim.event_log if k == "JOB_COMPLETED")
        at_tc = [k for (t, k, _) in sim.event_log if t == t_c]
        assert at_tc[:3] == ["JOB_COMPLETED", "BLOCK_ARRIVAL", "BLOCK_ARRIVAL"], at_tc
        assert REPORT["same-time-first"]["same_time"] >= 1


@pytest.mark.parametrize("window", [(5, 5), (4, 6)], ids=["window-5..5", "window-4..6"])
def test_dup_target_out_of_range_and_capacity(window):
    """같은 대상 반출 2건 · 담당 구간 밖 대상 · 재조작 용량 부족 → 미배차 잔존(backlog)."""
    prof = _narrow_profile(*window)
    sim, w, trace, tb = _run_and_compare(prof, dup_range_cap_scenario(), f"dup-range-cap-{window[0]}..{window[1]}")
    assert sim.unfinished_backlog() >= 2, "무대가 미배차 잔존을 만들지 못했다"


@pytest.mark.parametrize("stage", ["edge-end", "edge-drain"])
def test_evaluation_window_edges(stage):
    """도착이 정확히 end / end+5e-10 / end 뒤 / drain 창 안 — 287행 `nt <= end+EPS` 경계."""
    prof = piece1_profile()
    scn = {"edge-end": edge_end_scenario, "edge-drain": edge_drain_scenario}[stage]()
    sim, w, trace, tb = _run_and_compare(prof, scn, stage)
    assert sim.unfinished_backlog() >= 1


def test_turntime_edge_samples():
    """O > end 는 end−A 로 검열, A ≥ end 는 표본 제외 — censored_turn_time_s 가 v5 규칙과 같다 (compare 안에서 대조)."""
    prof = piece1_profile()
    sim, w, trace, tb = _run_and_compare(prof, turntime_edge_scenario(), "turntime-edge")
    tt = _finite_sorted(censored_turn_time_s(w.orders, w.end_s))
    assert len(tt) == 2 and tt[-1] == 7200.0, tt            # J-OUT-1 검열(end−0) · J-OUT-3 보통 · J-OUT-2 제외
    assert float(censored_exposure_s(w.orders, w.end_s)) == sim.time_ledger.censored_exposure_s() == 7200.0


def test_vessel_load_deadline_delay():
    """선박 없는 VESSEL_LOAD 2건 + 마감 — h_completed 의 지각 경로(kpis.py:90-96)가 실제로 실행되고 v5 와 같다."""
    prof = piece1_profile()
    sim, w, trace, tb = _run_and_compare(prof, vessel_deadline_scenario(), "vl-deadline")
    ks = sim.kpis.snapshot()
    assert ks.completed_vessel == 2 and ks.vessel_delay_s > 0


# ───────────────────────────────────────────────── ⑥ v5 가 예외를 던지는 자리 — 호스트 큰 소리 실패 · 위반 비트
def test_mixed_exit_travel_rejected_by_host_and_flagged_by_engine():
    """장부 모드에 exit_travel 없는 외부트럭이 섞이면 v5 는 BLOCK_ARRIVAL 에서 KeyError.
    호스트는 ValueError 로 막고, 그래도 들어오면 엔진은 8192 비트를 켜고 그 트럭의 블록 점유를 적분에 넣지 않는다."""
    prof, base = piece1_profile(), piece1_scenario()
    jobs = list(base.jobs)
    jobs[1] = _out("J-OUT-2", "C3", 0.0, 300.0, exit_s=None)
    scn = _scn("mixed-exit", dict(base.containers), jobs)
    with pytest.raises(KeyError):
        run_v5(prof, scn)
    caps, s_max = _caps(scn, prof)
    with pytest.raises(ValueError, match="장부 모드"):
        to_block_world(prof, scn, **caps)
    # 엔진 방어: 정상 세계에서 J-OUT-2 의 장부 등록(gate_in)만 지워 넣는다
    w0, tb = to_block_world(prof, base, **caps)
    n = tb.job_index["J-OUT-2"]
    w0 = w0._replace(orders=w0.orders._replace(gate_in_s=w0.orders.gate_in_s.at[n].set(jnp.inf)))
    w, _ = run_jit(w0, None, Geom.from_profile(prof), first_by_id, s_max)
    assert int(w.violation) & V_LEDGER_UNREGISTERED, violation_names(int(w.violation))
    assert not bool(w.orders.in_block[n])            # 등록 안 된 트럭은 블록 점유 적분에 들어가지 않는다
    assert bool(w.terminal)


def test_off_candidate_pick_sets_coverage_bit_and_skips():
    """정책이 후보 밖 오더(아직 도착 전 J-OUT-3)를 고르면: v5 는 KeyError, 배열은 512 비트 + 배정 건너뜀
    (그 트럭이 DISPATCH 되지 않는다). 같은 시각 재질문이 반복돼 스텝 소진(256)까지 가는 것은 실격 뒤의 일이다."""
    prof, scn = piece1_profile(), piece1_scenario()
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    n_off = tb.job_index["J-OUT-3"]

    def policy_off(params, x, mask):
        return jnp.where(jnp.any(mask, axis=1), n_off, EMPTY_ID).astype(jnp.int32)

    w, trace = run_jit(w0, None, Geom.from_profile(prof), policy_off, s_max)
    v = int(w.violation)
    assert v & V_DECISION_COVERAGE, violation_names(v)
    assert not (v & V_RESERVE_REJECT), violation_names(v)
    d = from_block_world(w, tb)
    assert d["jobs"]["J-OUT-3"]["status"] == "PLANNED" and d["jobs"]["J-OUT-3"]["service_start"] is None
    assert not any(p.endswith(":J-OUT-3") for (_, k, p) in d["event_log"] if k == "DISPATCH")
    # v5: 같은 배정을 넣으면 죽는다
    from yard_rl.v6.world.integrated.engine import TerminalSimulator as _TS
    sim = _TS(prof, scn, check_invariants=True)
    sim.run_until_decision()
    j, yc, sp = sim.jobs["J-OUT-3"], sim.fleet.get("YC-A"), sim.fleet.spec("YC-A")
    with pytest.raises(KeyError):
        sim.commit_decisions([CraneAssignment("YC-A", CandidateKind.SERVE, sim._jobref(j, sp, yc))])


# ───────────────────────────────────────────────── ⑦ 배치(vmap)·크레인 2대 연기
def test_vmap_batch_matches_single_worlds():
    """여러 세계를 `vmap(run)` 으로 한 번에 — 종료 국면이 다른 두 세계(crowded 완주·censored 검열)가
    각각 단일 판과 잎 전부 같다 (cond/switch 가 select 로 풀려도 답이 안 바뀐다는 학습 경로의 전제)."""
    prof = piece1_profile()
    caps, s_max = dict(n_max=16, q_cap=64, log_cap=400), 384
    g = Geom.from_profile(prof)
    worlds = [to_block_world(prof, s, **caps)[0] for s in (crowded_scenario(), crowded_scenario(2000.0))]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    run_b = jax.jit(jax.vmap(lambda w: ES.run(w, None, g, first_by_id, s_max)))
    wb, _ = run_b(batched)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(worlds[0])]
    for i, w0 in enumerate(worlds):
        ws, _ = run_jit(w0, None, g, first_by_id, s_max)
        lb = [np.asarray(l)[i] for l in jax.tree_util.tree_leaves(wb)]
        ls = [np.asarray(l) for l in jax.tree_util.tree_leaves(ws)]
        bad = [names[j] for j, (a, b) in enumerate(zip(lb, ls)) if not np.array_equal(a, b, equal_nan=True)]
        assert not bad, f"세계 {i}: vmap 판과 단일 판이 다른 잎 {bad}"
        assert bool(ws.terminal) and int(ws.violation) == 0


def test_vmap_across_different_yards_with_c_max():
    """초기 컨테이너 수가 다른 두 야드(random-1·random-2)를 `c_max` 로 같은 모양에 맞춰 쌓는다 —
    vmap 판 = 단일 판(잎 전부), 그리고 c_max 를 준 세계의 v5 모양 결과 = 안 준 세계의 결과."""
    prof = piece1_profile()
    scns = [random_scenario(1), random_scenario(2)]
    assert len(scns[0].containers) != len(scns[1].containers)
    caps, s_max = dict(n_max=16, q_cap=64, log_cap=400), 384
    c_max = max(len(s.containers) for s in scns) + caps["n_max"] + 3
    g = Geom.from_profile(prof)
    built = [to_block_world(prof, s, c_max=c_max, **caps) for s in scns]
    worlds = [w for w, _ in built]
    assert len({tuple(l.shape for l in jax.tree_util.tree_leaves(w)) for w in worlds}) == 1
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    run_b = jax.jit(jax.vmap(lambda w: ES.run(w, None, g, first_by_id, s_max)))
    wb, _ = run_b(batched)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(worlds[0])]
    for i, (w0, tb) in enumerate(built):
        ws, _ = run_jit(w0, None, g, first_by_id, s_max)
        lb = [np.asarray(l)[i] for l in jax.tree_util.tree_leaves(wb)]
        ls = [np.asarray(l) for l in jax.tree_util.tree_leaves(ws)]
        bad = [names[j] for j, (a, b) in enumerate(zip(lb, ls)) if not np.array_equal(a, b, equal_nan=True)]
        assert not bad, f"세계 {i}: vmap 판과 단일 판이 다른 잎 {bad}"
        assert bool(ws.terminal) and int(ws.violation) == 0
        # c_max 없이 만든 세계와 v5 모양 결과가 같다 (PAD 칸은 alive 아님 → containers 에 안 나옴)
        w1, tb1 = to_block_world(prof, scns[i], **caps)
        w1r, _ = run_jit(w1, None, g, first_by_id, s_max)
        d_a, d_b = from_block_world(ws, tb), from_block_world(w1r, tb1)
        for key in ("jobs", "cranes", "piles", "containers", "kpi", "ledger", "cost_episode", "event_log", "violation"):
            assert d_a[key] == d_b[key], f"세계 {i}: c_max 유무에 따라 {key} 가 다르다"


def test_two_cranes_smoke_runs_to_terminal():
    """크레인 2대 — 동등성은 조각 2 몫. 여기서는 배정 scan(K=2)·예약 거절 경로가 **완주**하는지만 본다.
    first_by_id 는 두 크레인이 같은 오더를 고르므로 DUP_JOB 거절(V_RESERVE_REJECT=16)이 날 수 있다."""
    base = fixtures.build_integrated_profile()
    prof = replace(base, block=BlockGeometry("B1", 10, 4, 4, 6.5, 2.9, 2.6, 0),
                   cranes=(replace(fixtures._spec("YC-A"), service_bay_max=10),
                           replace(fixtures._spec("YC-B"), service_bay_max=10)),
                   lane_graph=LaneGraph(("L1", "L2"), ()),
                   transfer=TransferFleetSpec("TF1", "YT", n_units=0, move_time_s=180.0))
    scn = crowded_scenario()
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    w, trace = run_jit(w0, None, Geom.from_profile(prof), first_by_id, s_max)
    assert bool(w.terminal) and int(w.overflow) == 0 and int(w.queue.overflow) == 0
    assert int(w.violation) & ~V_RESERVE_REJECT == 0, f"예약 거절 외 위반 비트 {violation_names(int(w.violation))}"
    REPORT["two-cranes-smoke"] = {"violation": int(w.violation), "steps": int(w.steps)}


# ───────────────────────────────────────────────── ⑧ 보고 (동률·WAIT·동시각·backlog 가 시험됐는지)
def test_zz_report(capsys):
    """마지막 — 무대별 스텝·결정·WAIT·find_slot 질의·정확 동률·실수 불일치. 공백이 다시 생기지 않게 단언한다."""
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    stages = {k: r for k, r in REPORT.items() if "decisions" in r}
    total_ties = sum(r["exact_ties"] for r in stages.values())
    total_waits = sum(r["waits"] for r in stages.values())
    total_same = sum(r["same_time"] for r in stages.values())
    n_backlog = sum(1 for r in stages.values() if r["backlog"] > 0)
    total_mismatch = sum(r.get("float_mismatches", 0) for r in REPORT.values())
    with capsys.disabled():
        print("\n[engine equiv report]  (maxerr 는 == 통과 뒤의 실제 차 = 0 이어야 정상)")
        for k, r in REPORT.items():
            if "decisions" not in r:
                print(f"  {k:24s} {r}")
                continue
            print(f"  {k:24s} steps={r['steps']:4d} dec={r['decisions']:3d} wait={r['waits']:2d} ev={r['events']:3d} "
                  f"same_t={r['same_time']:2d} find_slot={r['find_slot_calls']:4d} ties={r['exact_ties']:3d} "
                  f"mismatch={r.get('float_mismatches', 0)} maxerr={r['max_float_err']:.2e} backlog={r['backlog']} "
                  f"norm={r.get('log_payload_normalized', 0)} cost≠0={r['nonzero_cost']}")
        print(f"  exact ties total = {total_ties} · WAIT total = {total_waits} · same-time steps = {total_same} "
              f"· backlog>0 stages = {n_backlog} · float mismatches = {total_mismatch}")
    assert total_ties > 0, "어느 무대에서도 find_slot 정확 동률이 없었다 — tie-break 규칙이 시험되지 않았다"
    assert total_waits >= 1, "WAIT 결정이 한 번도 없었다 — yielded 경로가 시험되지 않았다"
    assert total_same >= 1, "같은 시각에 종류가 다른 사건이 한 번도 없었다 — 큐 우선순위가 엔진 수준에서 시험되지 않았다"
    assert n_backlog >= 1, "미배차 잔존(backlog>0) 무대가 없었다"
    assert total_mismatch == 0
