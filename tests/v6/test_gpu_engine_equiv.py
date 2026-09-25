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

■ 조각 2 무대 (크레인 2대 — 순차 배정·간섭·교착 탈출·장비 고장·interference/imbalance rate)
  k2-spec   조각 1 §10 무대 + YC-B(service 1..10) → 초기 위치 1+(k+0.5)·9/2 = 3.25 / 7.75 · 오더 8건(동시각 도착 짝 2 이상)
            · 주입 EQUIPMENT_DOWN(2000,'YC-B')/UP(2600) (변형: 작업 중 DOWN 310/400 · 레인 1개/2개 · gap 2/3)
  k2-fixture fixtures.build_minimal_terminal_scenario 의 본선 제거판을 크레인 2대(YC-A·YC-B 1..40) 그대로
  k2-random 무작위 야드·오더 10 시드 (정책 first/ref 번갈아 · gap 2/3 번갈아 · 정수 초 도착)
  k2-replan ★재계획이 계획을 **실제로 바꾸는** 설계 무대 (반박 검증 2026-09-25 동등성 렌즈가 지적한 시험 공백):
            gap 1 · 레인 2 · bay 4/7/9 만재 · bay 5·8 row1 = 대상+blocker · bay 6·10 row1 빈칸 · 둘 다 300초 도착 —
            YC-A 가 J-OUT-1 예약 {(5,1),(6,1)} 뒤 YC-B 의 J-OUT-2 재조작 목적지가 (6,1)→(10,1) 로 바뀌어 둘 다 SERVE.
            lockstep 이 결정마다 v5 `_plan` 을 배정 전후로 대조해 `plan_changed_pick` 을 세고 보고가 ≥1 을 단언한다.
  k2-stair  계단식·부분 겹침 담당구간 YC-A 1..6 · YC-B 5..10 (초기 위치 = service_bay_min 각각 · rail_order = (pos,id) 정렬
            · 그룹 크기 1 → 등간격 분산 없음, engine.py:105-118) — 다른 구간 크레인의 통로·유휴 점 장벽 상호작용.
  k3-random 크레인 3대 같은 구간 1..10 (초기 2.5/5.5/8.5 · 레인 3) — refresh_rates 의 sum_seq 결합 순서 보호가 K≥3 에서만
            의미가 있고, 배정 scan 의 단계 1..K−1 이 둘 이상이다.
  v5 구동은 **정본 의미** `ReferenceDispatcher.run`(dispatcher.py:19-32) — 크레인 순서대로 live 후보를 다시 뽑아
  하나씩 assign 한다 (`run_v5`; K=1 에서는 조각 1 구동과 같다). 배열은 `run_jit` 한 번 + 엔진 `step` 을 한 스텝씩
  돌리는 lockstep(`lockstep_engine`) 두 경로로 대조하고, 결정마다 (K,N) reject 코드열·물은 크레인·**열린 크레인 전원의
  live 후보 행**(dispatch.cand_live)·답·탈출 여부·결정 뒤 예약표/토큰/idle 장벽/rate 5항/yielded/down/recent_yield_count
  를 v5 와 맞춘다. 끝에는 두 경로의 최종 세계 잎이 비트 같음도 본다. v5 는 check_invariants=True 로 돌므로 v5 가
  안 죽었다 ↔ 배열 불변식 비트(16384·32768·65536) 0 이 결정·사건마다 대조된다.
  + `run_while`(학습 경로 while_loop) 세계 == `run`(scan) 세계, advance 의 unroll=1 판 == unroll=16 판 (잎 전부 비트).

■ 조각 3·4 무대 (2026-09-26 통합 — 파일 끝 '조각 3·4')
  v4-*      본선·이송 (조각 4): fixtures 전체(본선 2척·이송 2대) K=1/K=2 · stage a(양하 5·YT 1대 600초 → 버퍼 만재 STS 막힘) ·
            stage b(적하 5 를 트럭 5대와 섞어 굶김) · stage c(양하 110 사전식 해제 '-100'<'-11' · cadence 3600/27.5 · PLAN_CHANGE
            키 4종 · 선석 초과·출항 지연). 비교 = 조각 1 전 항목 + 배 15열·이송(busy_until·pending·대기 적분)·rate sts/transfer.
  p3-seq-*  PRE_ADVICE + 순차 규약(ReferenceDispatcher 의미, SERVE 만): test_gpu_wake 무대(blocked·busy-at-wake·neg-gap·
            gate-in-eta·crowded/random + ETA) — 사건열에 ETA_WAKE 포함 정확 · wake 로 열린 결정(전원 WAIT) · A 국면(wake 시각
            전진) 실제 발생.
  p3-joint-* PRE_ADVICE + 공동 규약(CentralResolver(Baseline/ServiceFirstSPT) + generate, resolver.apply): test_gpu_cands3
            무대(eta-basic·crowded-eta(prune)·dead-first-eta(탈출 REPO)·plan-failed-mandatory·eta-random·spt·block-arrival) —
            PRE_REHANDLE·REPOSITION 이 **실제로 실행**되고 결정열(크레인·종류·오더/REPO 이름)·상태 전부가 v5 와 같다.
  p3-waitall 전원 WAIT 정책 — 결정 수 유한·시각 엄격 증가 (test_yr050:147-161 규약).

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
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu import engine_step as ES                                        # noqa: E402
from yard_rl.v6.gpu import dispatch as DP                                           # noqa: E402
from yard_rl.v6.gpu.engine_step import first_by_id, run_jit, run_python             # noqa: E402
from yard_rl.v6.gpu.escape import candidate_matrices, try_escape                    # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.reserve import OK as RC_OK, REASON_TO_CODE                      # noqa: E402
from yard_rl.v6.gpu.host_convert import (event_log_from_arrays, event_stream_hash,  # noqa: E402
                                         from_block_world, repo_job_id, to_block_world)
from yard_rl.v6.gpu.state import (COST_TERMS, EMPTY_ID, MV_REHANDLE, MV_RETRIEVE, MV_STORE,   # noqa: E402
                                  PK_PRE_REHANDLE, PK_REPOSITION, PK_SERVE, PK_WAIT,
                                  V_CRANE_MIN_GAP, V_CRANE_ORDER_SWAP, V_DECISION_COVERAGE,
                                  V_LEDGER_UNREGISTERED, V_PAIRWISE_LOCK, V_RESERVE_REJECT,
                                  V_STEPS_EXHAUSTED, block_turn_time_s, censored_exposure_s,
                                  censored_turn_time_s, violation_names)
# v5 정본
from yard_rl.v6.world.contract.schema import COST_TERMS as V5_COST_TERMS, CandidateKind   # noqa: E402
from yard_rl.v6.world.contract.state import LaneGraph                               # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, InformationLevel, JobFlow, LoadStatus   # noqa: E402
from yard_rl.v6.world.domain.models import BlockGeometry, Container, Job           # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                    # noqa: E402
from yard_rl.v6.world.integrated.dispatcher import ReferenceDispatcher              # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator   # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec                   # noqa: E402
from yard_rl.v6.world.integrated.reservation import Corridor, Reservation           # noqa: E402
from yard_rl.v6.world.integrated.scenario import InjectedEvent, TerminalScenario    # noqa: E402
from yard_rl.v6.world.sim.constraints import ConstraintViolation                     # noqa: E402

FT20, FT40, FT45 = ContainerSize.FT20, ContainerSize.FT40, ContainerSize.FT45
MV_NAME = {MV_REHANDLE: "REHANDLE", MV_RETRIEVE: "RETRIEVE", MV_STORE: "STORE"}
KIND_NAME = {PK_SERVE: "SERVE", PK_PRE_REHANDLE: "PRE_REHANDLE", PK_REPOSITION: "REPOSITION", PK_WAIT: "WAIT"}
BA, PA = InformationLevel.BLOCK_ARRIVAL, InformationLevel.PRE_ADVICE

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


def random_scenario(seed: int, B: int = 10, R: int = 4, T: int = 4, *, int_arrivals: bool = False) -> TerminalScenario:
    """무작위 야드(칸 35%·단일 규격 pile)·반출 3~7·반입 1~4·장부 모드 60%·지평/drain 섞음.
    int_arrivals 면 도착을 정수 초로 (ref 정책이 대기 시간을 float32 특징으로 읽어도 순서가 v5 와 같도록)."""
    rng = random.Random(seed)

    def _arr():
        return float(rng.randint(0, 2500)) if int_arrivals else round(rng.uniform(0.0, 2500.0), 3)
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
        arr = _arr()
        jobs.append(_out(f"J-OUT-{i:02d}", tgt, max(0.0, arr - rng.uniform(0.0, 600.0)), arr, ex))
    for i in range(rng.randint(1, 4)):
        arr = _arr()
        jobs.append(_in(f"J-IN-{i:02d}", max(0.0, arr - rng.uniform(0.0, 600.0)), arr,
                        rng.choices([FT20, FT40, FT45], weights=[2, 6, 1])[0], ex))
    return TerminalScenario(scenario_id=f"random-{seed}{'-int' if int_arrivals else ''}", seed=seed,
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
# chooser(sim, crane_id, live 후보) → JobRef | None(WAIT).  v5 는 크레인마다 live 후보로 부른다 (dispatcher.py:26-31).
def chooser_first(sim, cid, cands):
    return cands[0] if cands else None


def chooser_last(sim, cid, cands):
    return cands[-1] if cands else None


def chooser_wait2(sim, cid, cands):
    """후보가 정확히 2개면 WAIT, 아니면 마지막 후보."""
    if not cands or len(cands) == 2:
        return None
    return cands[-1]


_REF = ReferenceDispatcher()


def chooser_ref(sim, cid, cands):
    """v5 ReferenceDispatcher.select — 본선 우선 → 최장 트럭대기 → job_id (dispatcher.py:14-17)."""
    return _REF.select(sim, cid, cands) if cands else None


def policy_ref(params, x, mask):
    """ReferenceDispatcher.select 의 배열판 — min by (0 if 본선 else 1, −누적대기, 오더 번호). 행마다 독립.

    누적대기(engine.py:258-265 cum_wait)는 정책 입력 특징 f0 = cum/3600 **float32** 로 읽는다 — 그래서 도착이
    정수 초인 무대(k2-spec·k2-random(int_arrivals))에서만 v5 와 같은 순서가 보장된다. params = is_vessel (N,).
    """
    ves = params
    k1 = jnp.where(ves, 0, 1).astype(jnp.int32)[None, :]
    cum = x[..., 0]
    m1 = mask & (k1 == jnp.min(jnp.where(mask, k1, 9), axis=1, keepdims=True))
    m2 = m1 & (cum == jnp.max(jnp.where(m1, cum, -jnp.inf), axis=1, keepdims=True))
    return jnp.where(jnp.any(mask, axis=1), jnp.argmax(m2, axis=1), EMPTY_ID).astype(jnp.int32)


def policy_last(params, x, mask):
    N = mask.shape[1]
    last = N - 1 - jnp.argmax(mask[:, ::-1], axis=1)
    return jnp.where(jnp.any(mask, axis=1), last, EMPTY_ID).astype(jnp.int32)


def policy_wait2(params, x, mask):
    N = mask.shape[1]
    n_c = jnp.sum(mask, axis=1)
    last = N - 1 - jnp.argmax(mask[:, ::-1], axis=1)
    return jnp.where(n_c == 2, EMPTY_ID, jnp.where(n_c > 0, last, EMPTY_ID)).astype(jnp.int32)


#: 이름 → (v5 chooser, 배열 policy_fn, params_fn(w0) → params)
POLICIES = {"first": (chooser_first, first_by_id, lambda w: None),
            "last": (chooser_last, policy_last, lambda w: None),
            "wait2": (chooser_wait2, policy_wait2, lambda w: None),
            "ref": (chooser_ref, policy_ref, lambda w: w.orders.is_vessel)}


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


def _v5_assign(sim, cid, chooser):
    """크레인 하나: live 후보(candidates_for) → chooser → assign. 반환 결정 기록 (crane, job|None, moves, dur, rehandles)."""
    ref = chooser(sim, cid, sim.candidates_for(cid))                    # dispatcher.py:26 live (앞 배정 반영)
    if ref is None:
        sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))       # 27-28행
        return (cid, None, [], None, None)
    sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=ref))   # 30-31행
    p = sim.active_plan(cid)
    return (cid, ref.job_id, _v5_moves(p), p.duration_s, p.rehandles)


def run_v5(prof, scn, chooser=chooser_first, level=BA):
    """v5 **정본 의미** 구동 = `ReferenceDispatcher.run` (dispatcher.py:19-32): 결정마다 크레인 순서(정렬됨)대로
    live 후보를 다시 뽑아 하나씩 assign 한다 — 앞 크레인의 예약이 뒤 크레인 후보에 반영된다. K=1 에서는 조각 1 의
    '후보를 모아 commit_decisions' 구동과 같은 답이다 (크레인 하나면 live == 시작 시점).
    level: 정보수준 (PRE_ADVICE 면 ETA wake 가 결정을 연다 — 순차 규약은 SERVE 후보가 없으면 WAIT).
    반환 (sim, 결정열 [(t, crane_ids, [(crane, job|None, moves, dur, rehandles)])], 동률 계수)."""
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    decisions = []
    tc = _TieCounter()
    with tc.watching(sim):
        while (dp := sim.run_until_decision()) is not None:
            rec = [_v5_assign(sim, c, chooser) for c in dp.crane_ids]
            sim.close_decision()
            decisions.append((dp.time, tuple(dp.crane_ids), rec))
    return sim, decisions, tc


# ───────────────────────────────────────────────── 배열 쪽 도구
def _caps(scn, prof):
    """명세 capacities: S_max = 8N+256 · Q = 4N(≥32) · E = S_max + N (큐 사건 + 결정마다 DISPATCH K줄)."""
    n0 = len(scn.jobs)
    n_max = max(8, 1 << (n0 - 1).bit_length())
    s_max = 8 * n_max + 256
    return dict(n_max=n_max, q_cap=max(32, 4 * n_max), log_cap=s_max + n_max), s_max


def run_array(prof, scn, *, use_jit=True, policy_fn=first_by_id, params_fn=lambda w: None, level=BA, joint=False):
    """level=PRE_ADVICE 면 step 의 pre_advice=True·horizon_s=profile.decision_horizon_s. joint 면 공동 규약
    (params_fn 은 (w0, tb, g) 를 받는다)."""
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    runner = run_jit if use_jit else run_python
    params = params_fn(w0, tb, g) if joint else params_fn(w0)
    w, trace = runner(w0, params, g, policy_fn, s_max, True, level == PA, float(prof.decision_horizon_s), joint)
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
    # ② 결정열 (시각·크레인·오더/WAIT) — 순차 규약만 (공동 규약은 compare_joint_decisions 가 종류까지 본다)
    ar_dec = _array_decisions(w, trace, tb) if v5_dec is not None else []
    if v5_dec is not None:
        v5_short = [(round(t, 6), cs, [(c, j) for (c, j, *_r) in rec]) for (t, cs, rec) in v5_dec]
        ar_short = [(round(t, 6), cs, [(c, j) for (c, j, *_r) in rec]) for (t, cs, rec) in ar_dec]
        diff = _first_diff(v5_short, ar_short)
        if diff is not None:
            i, x, y = diff
            pytest.fail(f"[{label}] ② 결정열이 {i}번째에서 갈린다: v5={x} arr={y}")
    else:
        v5_dec = []
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
        got = (a["position_bay"], a["trolley_row"], a["served_count"], a["recent_completions"], a["down"], a["down_pending"], a["yielded"], a["assigned_job"], a["recent_yield_count"])
        exp = (yc.state.position_bay, yc.state.trolley_row, yc.served_count, yc.recent_completions, yc.down, yc.down_pending, yc.yielded, yc.state.assigned_job, yc.recent_yield_count)
        assert got == exp, f"[{label}] ⑤ 크레인 {cid}: arr={got} v5={exp}"
    # ⑤-b 배·이송 (조각 4) — 배 15열 · 이송차 busy_until·pending·대기 적분 (배 없는 무대는 둘 다 빈 값)
    v5_ves = _v5_vessels_view(sim, tb.vessel_ids)
    assert d["vessels"] == v5_ves, f"[{label}] ⑤ 배\n  arr={d['vessels']}\n  v5 ={v5_ves}"
    tr = sim.transfer
    assert d["transfer"]["busy_until"] == list(tr.busy_until), f"[{label}] ⑤ 이송 busy_until arr={d['transfer']['busy_until']} v5={tr.busy_until}"
    assert d["transfer"]["pending"] == list(tr.pending), f"[{label}] ⑤ 이송 pending arr={d['transfer']['pending']} v5={tr.pending}"
    assert d["transfer"]["transfer_wait_accum_s"] == tr.transfer_wait_accum_s, f"[{label}] ⑤ 이송 대기 적분"
    # ⑥ kpi 정수
    ks = sim.kpis.snapshot()
    got6 = (d["kpi"]["rehandle_count"], d["kpi"]["completed_external"], d["kpi"]["completed_vessel"],
            d["kpi"]["pre_rehandle_count"], d["kpi"]["positioning_count"])
    exp6 = (ks.rehandle_count, ks.completed_external, ks.completed_vessel, ks.pre_rehandle_count, ks.positioning_count)
    assert got6 == exp6, f"[{label}] ⑥ kpi 정수 arr={got6} v5={exp6}"
    # ⑦ 격자
    assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}, f"[{label}] ⑦ piles"
    assert d["containers"] == {cid: (c.bay, c.row, c.tier) for cid, c in sim.stacks.containers.items()}, f"[{label}] ⑦ 컨테이너 좌표"
    # ⑧ 실격 없음·종료 · (조각 2) 레일 순서·탈출 표식·발화 횟수
    assert d["violation"] == 0 and d["overflow"] == 0 and d["terminal"], \
        f"[{label}] ⑧ violation={d['violation']} {d['violation_names']} overflow={d['overflow']} terminal={d['terminal']}"
    assert sim.terminal and d["clock"] == sim.clock and d["last_decision_at"] == sim._last_decision_at
    assert d["rail_order"] == sim._rail_order, f"[{label}] ⑧ rail_order arr={d['rail_order']} v5={sim._rail_order}"
    assert d["escape_count"] == sim.deadlock_escape_count, \
        f"[{label}] ⑧ escape_count arr={d['escape_count']} v5={sim.deadlock_escape_count}"
    assert d["escape_at"] == sim._escape_at, f"[{label}] ⑧ escape_at arr={d['escape_at']} v5={sim._escape_at}"
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


def _v5_vessels_view(sim, vessel_ids) -> dict:
    """v5 VesselProcess → host_convert(vessels_to_v5) 와 같은 모양."""
    out = {}
    for vid in vessel_ids:
        v = sim.vessels[vid]
        p = v.plan
        out[vid] = {
            "work_type": v.work_type.value, "total_moves": p.total_moves,
            "sts_move_interval_s": p.sts_move_interval_s, "quay_buffer_cap": p.quay_buffer_cap,
            "planned_start_s": p.planned_start_s, "planned_completion_s": p.planned_completion_s,
            "completion_basis": (None if p.completion_basis is None else p.completion_basis.value),
            "etd_s": p.etd_s, "started": v.started, "remaining_moves": v.remaining_moves,
            "buffer_level": v.buffer_level, "sts_blocked_since_s": v.sts_blocked_since_s,
            "sts_wait_accum_s": v.sts_wait_accum_s, "done": v.done,
            "actual_completion_s": v.truth.actual_completion_s,
        }
    return out


def _run_and_compare(prof, scn, label, *, use_jit=True, policy="first", level=BA):
    chooser, policy_fn, params_fn = POLICIES[policy]
    sim, dec, tc = run_v5(prof, scn, chooser, level)
    w, trace, tb = run_array(prof, scn, use_jit=use_jit, policy_fn=policy_fn, params_fn=params_fn, level=level)
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
        backlog=sim.unfinished_backlog(), interference=sim.cost.episode_raw()["interference"],
        imbalance=sim.cost.episode_raw()["imbalance"],
        K=len(tb.crane_ids), escapes=sim.deadlock_escape_count,
        multi_open=sum(1 for (_, cs, _) in dec if len(cs) >= 2),
        eta_wakes=sum(1 for (_, k, _) in sim.event_log if k == "ETA_WAKE"),
        advanced=int(np.asarray(trace.advanced)[:int(w.steps)].sum()),
        woke=int(np.asarray(trace.woke)[:int(w.steps)].sum()),
        vessels=len(tb.vessel_ids), sts_wait=sim.cost.episode_raw()["sts_wait"],
        transfer_wait=sim.cost.episode_raw()["transfer_wait"])
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


# ═════════════════════════════════════════════════ 조각 2 — 크레인 2대 (머리말 '조각 2 무대')
def profile_k2(lanes: int = 1, gap: float = 2.0, lo: int = 1, hi: int = 10):
    """조각 1 §10 프로파일 + YC-B(service lo..hi) — 같은 담당구간 2대 → 초기 위치 lo+(k+0.5)(hi−lo)/2 (engine.py:105-113).
    lanes=1 이면 §10 그대로(레인 하나 → 두 번째 예약은 늘 LANE_CONFLICT), lanes=2 면 L1·L2(인접 없음) 로 병행 작업이 가능."""
    base = piece1_profile()
    return replace(base,
                   cranes=(replace(fixtures._spec("YC-A"), service_bay_min=lo, service_bay_max=hi),
                           replace(fixtures._spec("YC-B"), service_bay_min=lo, service_bay_max=hi)),
                   lane_graph=LaneGraph(tuple(f"L{i + 1}" for i in range(lanes)), ()),
                   safety_gap_bay=gap)


def piece2_scenario(down_up: tuple[float, float] = (2000.0, 2600.0)) -> TerminalScenario:
    """§10 오더 5건 (동시각 짝: 300 ×2 · 1500 ×2) + 3건 (2100 ×2 짝 — YC-B 고장 중 도착 · 3000) = 8건,
    주입 EQUIPMENT_DOWN(down_up[0], 'YC-B') / UP(down_up[1]). 도착은 전부 정수 초 (ref 정책용)."""
    base = piece1_scenario()
    containers = dict(base.containers)
    containers["C4"] = _c("C4", 2, 3, 1)
    containers["C5"] = _c("C5", 9, 1, 1)
    containers["C6"] = _c("C6", 9, 1, 2)          # C5 위 blocker — 반출 시 재조작 1
    jobs = list(base.jobs) + [_out("J-OUT-4", "C4", 1600.0, 2100.0), _in("J-IN-3", 1700.0, 2100.0, FT40),
                              _out("J-OUT-5", "C5", 2500.0, 3000.0)]
    inj = [InjectedEvent(down_up[0], "EQUIPMENT_DOWN", "YC-B"), InjectedEvent(down_up[1], "EQUIPMENT_UP", "YC-B")]
    return TerminalScenario(scenario_id=f"piece2-{int(down_up[0])}", seed=0, horizon_s=7200.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=inj)


def fixture_scenario_k2() -> TerminalScenario:
    """fixtures 시나리오의 본선 제거판 — 크레인 2대(YC-A·YC-B 1..40)·주입 DOWN(2000,'YC-B')/UP(2600)·PLAN_CHANGE 는 그대로."""
    base = fixtures.build_minimal_terminal_scenario()
    return TerminalScenario(scenario_id="fixture-k2", seed=0, horizon_s=base.horizon_s,
                            drain_window_s=base.drain_window_s, containers=dict(base.containers),
                            jobs=[j for j in base.jobs if j.vessel_id is None], vessels=[],
                            injected_events=list(base.injected_events))


#: 무작위 K=2 무대 (seed, gap, 정수 초 도착, 정책) — v5 를 먼저 굴려 고른 것 (2026-09-25 탐색: gap 2/3/4 × 시드 1..24 ×
#: 정수/소수 도착 × first/ref = 288 조합 중 107 에서 DEADLOCK_ESCAPE 발화). 앞 9개는 발화 무대(1~9회), 뒤 3개는 미발화.
#: ref 정책은 소수 도착(3자리)이면 대기 특징 float32 동률 위험이 있어 정수 도착 무대에만 얹는다 (policy_ref 머리말).
K2_RANDOM = [(21, 4.0, False, "first"), (14, 4.0, True, "first"), (12, 4.0, True, "ref"), (24, 4.0, True, "ref"),
             (14, 3.0, True, "first"), (16, 3.0, False, "first"), (9, 3.0, True, "ref"), (1, 3.0, True, "ref"),
             (12, 2.0, True, "first"),
             (2, 2.0, True, "first"), (3, 2.0, True, "ref"), (5, 3.0, True, "first")]


def random_k2_name(seed, gap, ints, policy):
    return f"k2-random-s{seed}-g{int(gap)}-{'int' if ints else 'dec'}-{policy}"


def random_k2_stage(seed: int, gap: float, ints: bool, policy: str):
    """(profile, scenario, 정책 이름) — 레인 2 · 같은 구간 1..10 (초기 3.25/7.75)."""
    return profile_k2(lanes=2, gap=gap), random_scenario(seed, int_arrivals=ints), policy


def dead_first_scenario(injected=()) -> TerminalScenario:
    """(test_gpu_escape 의 설계 무대) 첫 도착(300초)이 bay 5 = 3.25/7.75 · gap 3 의 사각지대 → 결정 0회 뒤 곧바로 교착.
    600초에 bay 2·bay 9 도착 → 둘 다 정상 배정되고 크레인이 움직인 뒤 bay 5 가 풀린다."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 2, 1, 1), "C3": _c("C3", 9, 1, 1)}
    jobs = [_out("J-OUT-1", "C1", 0.0, 300.0), _out("J-OUT-2", "C2", 100.0, 600.0), _out("J-OUT-3", "C3", 100.0, 600.0)]
    return TerminalScenario(scenario_id="dead-first", seed=0, horizon_s=3600.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=list(injected))


def down_scenario(targets: tuple[str, ...]) -> TerminalScenario:
    """dead-first 에 고장(200초)·복구(1000초) 주입 — down-one: 유휴 하나만 결정 대상 · down-both: 술어 참인데 esc 비어 미발화."""
    inj = [InjectedEvent(200.0, "EQUIPMENT_DOWN", t) for t in targets] + \
          [InjectedEvent(1000.0, "EQUIPMENT_UP", t) for t in targets]
    return replace(dead_first_scenario(inj), scenario_id=f"down-{'-'.join(targets)}")


def replan_scenario() -> TerminalScenario:
    """★재계획이 뒤 크레인의 계획을 실제로 바꾸는 무대 (머리말 k2-replan; profile_k2(lanes=2, gap=1.0) 와 짝).

    bay 4 만재 · bay 5 row1 = TA(1)+BA(2), row2-4 만재 · bay 6 row1 빈칸, row2-4 만재 · bay 7 만재 · bay 8 row1 = TB(1)+BB(2),
    row2-4 만재 · bay 9 만재 · bay 10 row1 빈칸, row2-4 만재 · bay 1-3 빈 야드. J-OUT-1(TA)·J-OUT-2(TB) 둘 다 300초.
    결정 시작: YC-A 후보 {J-OUT-1} (J-OUT-2 통로 [3.25,8] 이 YC-B 점 장벽 7.75 와 겹침) · YC-B 후보 {J-OUT-1, J-OUT-2}
    (J-OUT-2 재조작 목적지 (6,1), 동률 13.0 → bay 6 < 10). YC-A 가 J-OUT-1 예약(통로 [3.25,6]·칸 {(5,1),(6,1)}·L1) →
    YC-B live: J-OUT-1 taken · J-OUT-2 **재계획** → (6,1) 제외 → 목적지 (10,1)·통로 [7.75,10]·L2 → SERVE.
    재계획이 없었다면(P0 그대로) 통로 [6,8] 이 [3.25,6]+gap 과 겹쳐 CRANE_INTERFERENCE → WAIT 였을 것."""
    cont = {}
    n = 0

    def full(bay, rows=(1, 2, 3, 4)):
        nonlocal n
        for r in rows:
            for t in (1, 2, 3, 4):
                cont[f"F{n:03d}"] = _c(f"F{n:03d}", bay, r, t)
                n += 1
    full(4); full(5, (2, 3, 4)); full(6, (2, 3, 4)); full(7); full(8, (2, 3, 4)); full(9); full(10, (2, 3, 4))
    cont["TA"] = _c("TA", 5, 1, 1); cont["BA"] = _c("BA", 5, 1, 2)
    cont["TB"] = _c("TB", 8, 1, 1); cont["BB"] = _c("BB", 8, 1, 2)
    jobs = [_out("J-OUT-1", "TA", 0.0, 300.0), _out("J-OUT-2", "TB", 0.0, 300.0)]
    return TerminalScenario(scenario_id="replan", seed=0, horizon_s=3600.0, drain_window_s=0.0,
                            containers=cont, jobs=jobs, vessels=[], injected_events=[])


def profile_k2_stair(lanes: int = 2, gap: float = 2.0):
    """계단식·부분 겹침 담당구간 — YC-A 1..6 · YC-B 5..10 (머리말 k2-stair). 그룹 크기 1 이라 초기 위치 = service_bay_min."""
    base = piece1_profile()
    return replace(base,
                   cranes=(replace(fixtures._spec("YC-A"), service_bay_min=1, service_bay_max=6),
                           replace(fixtures._spec("YC-B"), service_bay_min=5, service_bay_max=10)),
                   lane_graph=LaneGraph(tuple(f"L{i + 1}" for i in range(lanes)), ()),
                   safety_gap_bay=gap)


def profile_k3(lanes: int = 3, gap: float = 2.0):
    """크레인 3대 같은 구간 1..10 (머리말 k3-random) — 초기 위치 1+(k+0.5)·9/3 = 2.5 / 5.5 / 8.5."""
    base = piece1_profile()
    return replace(base,
                   cranes=tuple(replace(fixtures._spec(cid), service_bay_min=1, service_bay_max=10)
                                for cid in ("YC-A", "YC-B", "YC-C")),
                   lane_graph=LaneGraph(tuple(f"L{i + 1}" for i in range(lanes)), ()),
                   safety_gap_bay=gap)


def _v5_plan_key(p):
    """v5 JobPlan → 비교 튜플 (moves · corridor · slots · lane · dur · rehandles)."""
    return (_v5_moves(p), (float(p.corridor[0]), float(p.corridor[1])), frozenset(p.slots), p.lane_id,
            p.duration_s, p.rehandles)


def _arr_plan_key(P, k, n, tb):
    """배열 (K,N) PlanOut 의 (k,n) 칸 → 같은 모양의 튜플."""
    moves = []
    for m in range(int(P.n_moves[k, n])):
        moves.append((tb.cont_ids[int(P.mv_cont[k, n, m])], tuple(int(v) for v in P.mv_src[k, n, m]),
                      tuple(int(v) for v in P.mv_dst[k, n, m]), MV_NAME[int(P.mv_kind[k, n, m])]))
    lane = tb.lane_ids[int(P.lane[k, n])] if int(P.lane[k, n]) >= 0 else None
    slots = frozenset((int(b) + 1, int(r) + 1) for b, r in np.argwhere(np.asarray(P.slots[k, n])))
    return (moves, (float(P.lo[k, n]), float(P.hi[k, n])), slots, lane, float(P.dur[k, n]), int(P.rehandles[k, n]))


def v5_pair_table(sim, tb):
    """크레인×오더 전 쌍을 v5 함수로 — candidates_for(477-490행) 의 단계별 결과 (disp · feasible · reject 코드)."""
    K, n0 = len(tb.crane_ids), tb.n0
    disp = np.zeros((K, n0), bool)
    feas = np.zeros((K, n0), bool)
    code = np.full((K, n0), -1, np.int32)
    for n, jid in enumerate(tb.job_ids):
        j = sim.jobs[jid]
        taken = sim.reservations.job_taken(jid) is not None
        for k, cid in enumerate(tb.crane_ids):
            d = bool(sim._dispatchable(j, cid))
            disp[k, n] = d
            if not d or taken:
                continue
            yc, spec = sim.fleet.get(cid), sim.fleet.spec(cid)
            ref = sim._jobref(j, spec, yc)
            if ref is None:
                continue
            plan = sim._plan(cid, ref)
            if plan is None:
                continue
            feas[k, n] = True
            code[k, n] = REASON_TO_CODE[sim.reservations.reject_reason(sim._reservation(plan))]
    return disp, feas, code


def _v5_reservations(sim, tb):
    out = {}
    for cid, r in sim.reservations._by_crane.items():
        out[cid] = {"job_token": r.job_token, "corridor": (r.corridor.lo, r.corridor.hi),
                    "lane_id": r.lane_id, "release_at": r.release_at, "slots": frozenset(r.slots)}
    return out


def _leaf_diff(a, b) -> list[str]:
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(a)]
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return [names[i] for i, (x, y) in enumerate(zip(la, lb))
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]


def lockstep_engine(prof, scn, policy: str, label: str):
    """엔진 `step` 을 한 스텝씩 돌리며 v5 와 **결정마다** 나란히 대조한다 (run_jit 과 별개의 두 번째 경로).

    결정 시점마다: 시각 · 물은 크레인 · 탈출 여부 · (K,N) reject 코드열(disp/feasible/code 전 쌍, 결정 시작 시점) ·
    열린 크레인 **전원**의 live 후보 행(dispatch.cand_live — 앞 크레인 예약이 반영된 재계산) · 크레인별 답 ·
    재계획이 계획을 바꾼 결정(v5 `_plan` 배정 전후 대조 → plan_changed_*; 바뀐 계획은 배열 live 계획과 == 대조) ·
    결정 뒤 예약표/토큰 역표/idle 장벽/rate 5항(interference·imbalance 포함)/yielded/down/down_pending/
    last_decision_at/escape_count · 불변식 비트 0. 결정 세계에서는 엔진 `decide` 와 겉옷 `dispatch.decide_seq` 의 결과
    잎 전부가 비트 같은지도 본다 (같은 인자를 넘기는지). 끝에는 조각 1 `compare` 전 항목.
    반환 (sim, 최종 세계, tables, 집계).
    """
    chooser, policy_fn, params_fn = POLICIES[policy]
    caps, s_max = _caps(scn, prof)
    w, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    K = len(tb.crane_ids)
    params = params_fn(w)
    step_jit = jax.jit(partial(ES.step, params=params, g=g, policy_fn=policy_fn))
    cand_jit = jax.jit(partial(candidate_matrices, g=g))
    dec_engine = jax.jit(partial(ES.decide, params=params, g=g, policy_fn=policy_fn))
    dec_module = jax.jit(partial(DP.decide_seq, params=params, g=g, policy_fn=policy_fn))
    disp_trace = jax.jit(partial(DP.dispatch, params=params, g=g, policy_fn=policy_fn, with_trace=True))
    inv_jit = jax.jit(partial(ES.check_invariants, g=g))
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    v5_dec, traces = [], []
    st = dict(decisions=0, escapes=0, waits=0, multi_open=0, codes={}, code4_pairs=0, code3_pairs=0,
              regained=0, decide_vs_module=0, live_rows=0, plan_changed_any=0, plan_changed_pick=0,
              plan_changed_checked=0, invariant_checks=0)
    while True:
        esc_before = sim._escape_count
        dp = sim.run_until_decision()
        tr = None
        while not bool(w.terminal):                                     # 결정 스텝이 나올 때까지 사건만 처리
            w_pre = w
            w, tr = step_jit(w, None)
            traces.append(tr)
            if bool(tr.decided):
                break
        if dp is None:
            assert bool(w.terminal), f"[{label}] v5 는 끝났는데 배열은 아직 (clock={float(w.clock)})"
            break
        assert tr is not None and bool(tr.decided), f"[{label}] 배열은 끝났는데 v5 는 결정 {dp}"
        v5_esc = sim._escape_count > esc_before
        # ① 시각·탈출·물은 크레인
        assert float(w.clock) == dp.time, f"[{label}] 결정 시각 v5={dp.time!r} arr={float(w.clock)!r}"
        assert bool(tr.escaped) == v5_esc, f"[{label}] t={dp.time} 탈출 v5={v5_esc} arr={bool(tr.escaped)}"
        open_ids = tuple(tb.crane_ids[k] for k in range(K) if bool(tr.open[k]))
        assert open_ids == tuple(dp.crane_ids), f"[{label}] t={dp.time} 물은 크레인 v5={dp.crane_ids} arr={open_ids}"
        # ② reject 코드열 — 결정 시작 시점 (K,N) 전 쌍 (yielded 와 무관 · 탈출의 해제 전후 같다)
        m = cand_jit(w_pre)
        disp5, feas5, code5 = v5_pair_table(sim, tb)
        n0 = tb.n0
        dispa, feasa, codea = np.asarray(m.disp)[:, :n0], np.asarray(m.feasible)[:, :n0], np.asarray(m.code)[:, :n0]
        assert np.array_equal(dispa, disp5), f"[{label}] t={dp.time} dispatchable 다름 {np.argwhere(dispa != disp5).tolist()}"
        assert np.array_equal(feasa, feas5), f"[{label}] t={dp.time} 계획 성립 다름 {np.argwhere(feasa != feas5).tolist()}"
        both = feasa & feas5
        if not np.array_equal(np.where(both, codea, -1), np.where(both, code5, -1)):
            pairs = [(k, n, int(codea[k, n]), int(code5[k, n])) for k, n in np.argwhere(both & (codea != code5))]
            pytest.fail(f"[{label}] t={dp.time} reject 코드 다름 (k,n,arr,v5)={pairs}")
        for c in code5[feas5]:
            st["codes"][int(c)] = st["codes"].get(int(c), 0) + 1
        st["code4_pairs"] += int((code5[feas5] == 4).sum())
        st["code3_pairs"] += int((code5[feas5] == 3).sum())
        # ③ 엔진 decide == 모듈 decide_seq (잎 전부) — 탈출이면 해제 뒤 세계 + open_override
        out_e = dec_engine(w_pre)
        if v5_esc:
            e = try_escape(w_pre, m)
            w_in, out_m = e.world, dec_module(e.world, open_override=e.open)
        else:
            w_in, out_m = w_pre, dec_module(w_pre)
        bad = _leaf_diff(out_e.world, out_m.world)
        assert not bad, f"[{label}] t={dp.time} 엔진 decide 와 dispatch.decide_seq 가 다른 잎 {bad}"
        assert np.array_equal(np.asarray(out_e.pick), np.asarray(out_m.pick))
        bad = _leaf_diff(out_e.world._replace(steps=w.steps), w)         # step 은 steps 만 +1
        assert not bad, f"[{label}] t={dp.time} step 의 결정 세계 ≠ decide 단독: {bad}"
        st["decide_vs_module"] += 1
        # ④ 열린 크레인 **전원**의 live 후보 행 (앞 크레인 예약 반영) = v5 candidates_for 순차 · 크레인별 답 · 재계획 대조
        dl = disp_trace(w_in, open_override=jnp.asarray(tr.open))       # 엔진과 같은 open 으로 단계별 흔적만 뽑는다
        assert np.array_equal(np.asarray(dl.pick), np.asarray(tr.pick)), f"[{label}] t={dp.time} dispatch 흔적의 답 ≠ 엔진 답"
        plan0 = {cid: {r.job_id: _v5_plan_key(sim._plan(cid, r)) for r in sim.candidates_for(cid)} for cid in dp.crane_ids}
        rec = []
        for i, cid in enumerate(dp.crane_ids):
            k = tb.crane_index[cid]
            cands5 = sim.candidates_for(cid)
            live5 = [r.job_id for r in cands5]
            rowa = [tb.job_ids[n] for n in range(n0) if bool(dl.cand_live[k, n])]
            assert live5 == rowa, f"[{label}] t={dp.time} {cid}(#{i}) live 후보 v5={live5} arr={rowa}"
            assert not np.asarray(dl.cand_live[k])[n0:].any(), f"[{label}] t={dp.time} {cid} 빈 오더 칸이 후보로 살아 있다"
            st["live_rows"] += 1
            if i == 0:
                rowa0 = [tb.job_ids[n] for n in range(n0) if bool(feasa[k, n]) and int(codea[k, n]) == RC_OK]
                assert live5 == rowa0, f"[{label}] t={dp.time} {cid} 첫 크레인 후보 ≠ 시작 시점 행 v5={live5} arr={rowa0}"
            if v5_esc and live5:
                st["regained"] += 1
            if i >= 1:                                                   # 재계획(694행 live reserved_slots)이 계획을 바꿨나
                changed = {r.job_id for r in cands5
                           if r.job_id in plan0[cid] and _v5_plan_key(sim._plan(cid, r)) != plan0[cid][r.job_id]}
                st["plan_changed_any"] += int(bool(changed))
                for jid in changed:                                      # 바뀐 계획 = 배열 단계 k 의 live 계획 (P_live)
                    n = tb.job_index[jid]
                    k5 = _v5_plan_key(sim._plan(cid, next(r for r in cands5 if r.job_id == jid)))
                    ka = _arr_plan_key(dl.P_live, k, n, tb)
                    assert k5 == ka, f"[{label}] t={dp.time} {cid}:{jid} 재계획 결과 다름\n  v5 ={k5}\n  arr={ka}"
                    st["plan_changed_checked"] += 1
            ref_pre = chooser(sim, cid, cands5)
            if i >= 1 and ref_pre is not None and ref_pre.job_id in plan0[cid] \
                    and _v5_plan_key(sim._plan(cid, ref_pre)) != plan0[cid][ref_pre.job_id]:
                st["plan_changed_pick"] += 1
            r5 = _v5_assign(sim, cid, chooser)
            rec.append(r5)
            pick = int(tr.pick[k])
            got = tb.job(pick) if pick >= 0 else None
            assert got == r5[1], f"[{label}] t={dp.time} {cid} 답 v5={r5[1]} arr={got} (live={live5})"
            st["waits"] += int(r5[1] is None)
        sim.close_decision()
        v5_dec.append((dp.time, tuple(dp.crane_ids), rec))
        st["decisions"] += 1
        st["escapes"] += int(v5_esc)
        st["multi_open"] += int(len(open_ids) >= 2)
        # ⑤ 결정 뒤 상태 — 예약표·토큰·idle 장벽·rate 5항(==)·크레인 표식·탈출 표식
        d = from_block_world(w, tb)
        assert d["violation"] == 0, f"[{label}] t={dp.time} violation {d['violation_names']}"
        assert d["reservations"] == _v5_reservations(sim, tb), \
            f"[{label}] t={dp.time} 예약표\n  v5 ={_v5_reservations(sim, tb)}\n  arr={d['reservations']}"
        assert d["idle_positions"] == sim.reservations.idle_positions()
        owner = {tb.job_ids[n]: tb.crane_ids[int(kk)] for n, kk in enumerate(np.asarray(w.res.token_owner)[:n0]) if kk >= 0}
        assert owner == dict(sim.reservations._tokens), f"[{label}] t={dp.time} 토큰 역표 v5={sim.reservations._tokens} arr={owner}"
        for t, v in sim.cost._rate.items():
            assert d["cost_rate"][t] == v, f"[{label}] t={dp.time} rate.{t} arr={d['cost_rate'][t]!r} v5={v!r}"
        for cid in tb.crane_ids:
            yc, a = sim.fleet.get(cid), d["cranes"][cid]
            got = (a["yielded"], a["down"], a["down_pending"], a["assigned_job"], a["available_at"], a["recent_yield_count"])
            exp = (yc.yielded, yc.down, yc.down_pending, yc.state.assigned_job, yc.state.available_at, yc.recent_yield_count)
            assert got == exp, f"[{label}] t={dp.time} 크레인 {cid} arr={got} v5={exp}"
        assert d["last_decision_at"] == sim._last_decision_at and d["escape_at"] == sim._escape_at
        assert d["escape_count"] == sim._escape_count
        # ⑥ 불변식 — v5 close_decision 의 check_invariants 가 안 던졌다 ↔ 배열 비트 0 (레일 순서·간격·예약 쌍)
        assert int(inv_jit(w)) == 0, f"[{label}] t={dp.time} 불변식 비트 {violation_names(int(inv_jit(w)))}"
        st["invariant_checks"] += 1
    trace = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *traces)
    w = ES.finish(w)
    compare(sim, v5_dec, w, trace, tb, label)
    st.update(K=K, policy=policy, steps=int(w.steps), events=len(sim.event_log), backlog=sim.unfinished_backlog(),
              interference=sim.cost.episode_raw()["interference"], imbalance=sim.cost.episode_raw()["imbalance"])
    REPORT[label] = {"lockstep": st}
    return sim, w, tb, st


def _k2_stage(name: str):
    """이름 → (profile, scenario, 정책)."""
    if name.startswith("k2-spec"):
        _, _, lanes, gap, down, policy = name.split("-")
        return profile_k2(lanes=int(lanes[1:]), gap=float(gap[1:])), \
            piece2_scenario({"d2000": (2000.0, 2600.0), "d310": (310.0, 400.0)}[down]), policy
    if name == "k2-fixture":
        return fixtures.build_integrated_profile(), fixture_scenario_k2(), "first"
    if name == "k2-dead-first":
        return profile_k2(lanes=2, gap=3.0), dead_first_scenario(), "ref"
    if name == "k2-down-one":
        return profile_k2(lanes=2, gap=3.0), down_scenario(("YC-B",)), "ref"
    if name == "k2-down-both":
        return profile_k2(lanes=2, gap=3.0), down_scenario(("YC-A", "YC-B")), "ref"
    if name.startswith("k2-random-"):
        for (seed, gap, ints, policy) in K2_RANDOM:
            if random_k2_name(seed, gap, ints, policy) == name:
                return random_k2_stage(seed, gap, ints, policy)
    if name.startswith("k2-replan-"):
        return profile_k2(lanes=2, gap=1.0), replan_scenario(), name.split("-")[-1]
    if name.startswith("k2-stair-"):
        _, _, seed, policy = name.split("-")
        return profile_k2_stair(), random_scenario(int(seed[1:]), int_arrivals=True), policy
    if name.startswith("k3-random-"):
        _, _, seed, policy = name.split("-")
        return profile_k3(), random_scenario(int(seed[1:]), int_arrivals=True), policy
    raise KeyError(name)


#: 계단식(K=2)·크레인 3대 무대 — v5 를 먼저 굴려 2대 이상 동시 개방이 있는 시드를 골랐다 (2026-09-25).
K2_STAIR = ["k2-stair-s4-first", "k2-stair-s10-ref"]        # v5: 동시 개방 5·8 · 탈출 1·4
K3_STAGES = ["k3-random-s4-first", "k3-random-s5-ref"]      # v5: 동시 개방 4·6 · 탈출 0·5 · backlog 0·2
K2_STAGES = ["k2-spec-l1-g2-d2000-first", "k2-spec-l2-g2-d2000-first", "k2-spec-l2-g3-d2000-ref",
             "k2-spec-l2-g2-d310-first", "k2-spec-l1-g3-d310-ref", "k2-fixture",
             "k2-dead-first", "k2-down-one", "k2-down-both"] + \
            [random_k2_name(*r) for r in K2_RANDOM] + \
            ["k2-replan-first", "k2-replan-ref"] + K2_STAIR + K3_STAGES


@pytest.mark.parametrize("stage", K2_STAGES, ids=K2_STAGES)
def test_k2_equivalence_run_and_lockstep(stage):
    """크레인 2대 — (a) run_jit 완주 vs v5 (조각 1 compare 전 항목 + rail_order·escape) · (b) 엔진 step lockstep
    (결정마다 reject 코드열·답·상태) · (c) 두 경로의 최종 세계 잎 전부 비트 동일."""
    prof, scn, policy = _k2_stage(stage)
    sim, w_run, trace, tb = _run_and_compare(prof, scn, stage, policy=policy)
    assert len(tb.crane_ids) == (3 if stage.startswith("k3-") else 2)
    sim2, w_lock, _, st = lockstep_engine(prof, scn, policy, f"{stage}#lockstep")
    bad = _leaf_diff(w_run, w_lock)
    assert not bad, f"[{stage}] run_jit 세계와 lockstep 세계가 다른 잎 {bad}"
    assert sim.event_stream_hash() == sim2.event_stream_hash()
    REPORT[stage].update(escapes=sim.deadlock_escape_count, code4=st["code4_pairs"], code3=st["code3_pairs"],
                         imbalance=st["imbalance"])


def test_k2_spec_initial_positions_rail_order_and_paths():
    """k2-spec 의 성질 — 초기 위치 1+(k+0.5)·9/2 = 3.25/7.75 · rail_order (YC-A, YC-B) · 실제로 밟은 경로:
    YC-B 유휴 중 DOWN(2000) → down, 작업 중 DOWN(310) → down_pending → 완료 후 down · 2대 동시 결정 · imbalance>0."""
    prof, scn = profile_k2(lanes=2), piece2_scenario()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    pos = [1 + (k + 0.5) * 9 / 2 for k in range(2)]
    assert [float(b) for b in w0.cranes.bay] == pos == [sim.fleet.get(c).state.position_bay for c in tb.crane_ids] == [3.25, 7.75]
    assert from_block_world(w0, tb)["rail_order"] == sim._rail_order == ("YC-A", "YC-B")
    assert dict(from_block_world(w0, tb)["idle_positions"]) == sim.reservations.idle_positions()
    for lab in ("k2-spec-l2-g2-d2000-first", "k2-spec-l2-g2-d310-first"):
        if lab not in REPORT:
            _run_and_compare(*_k2_stage(lab)[:2], lab, policy=_k2_stage(lab)[2])
    r_idle, r_busy = REPORT["k2-spec-l2-g2-d2000-first"], REPORT["k2-spec-l2-g2-d310-first"]
    assert r_idle["multi_open"] >= 1 and r_busy["multi_open"] >= 1, "2대가 동시에 열린 결정이 없다"
    assert r_idle["imbalance"] > 0, "imbalance rate 가 0 — 두 크레인 부하 차이가 생기지 않았다"
    # 작업 중 DOWN(310/400) 이 실제로 down_pending 경로를 밟았는지 — v5 를 직접 굴려 확인
    sim_b, _, _ = run_v5(prof, piece2_scenario((310.0, 400.0)))
    log = sim_b.event_log
    t_down = next(t for (t, k, p) in log if k == "EQUIPMENT_DOWN")
    assert any(k == "DISPATCH" and p.startswith("YC-B:") and t < t_down for (t, k, p) in log), "310초 전에 YC-B 가 작업을 잡지 않았다"


def _ensure_k2(lab: str) -> dict:
    if lab not in REPORT:
        prof, scn, policy = _k2_stage(lab)
        _run_and_compare(prof, scn, lab, policy=policy)
    return REPORT[lab]


def test_k2_escape_paths_are_exercised():
    """X 국면이 실제로 밟혔는지 — v5 DEADLOCK_ESCAPE 가 발화한 무대 ≥ 5 (그 무대들의 배열 답은 compare ①·⑧ 가 이미 맞췄다),
    설계 무대: dead-first(결정 0회 뒤 300초 발화 · 둘 다 대상) · down-one(YC-A 만 대상) · down-both(술어 참·esc 비어 미발화)."""
    fired = {k: r["escapes"] for k in K2_STAGES if (r := _ensure_k2(k))["escapes"] > 0}
    assert len(fired) >= 5, f"탈출 발화 무대가 5개 미만: {fired}"
    prof, scn, _ = _k2_stage("k2-dead-first")
    sim, dec, _ = run_v5(prof, scn, chooser_ref)
    assert sim.deadlock_escape_count >= 1 and dec[0][0] == 300.0 and dec[0][1] == ("YC-A", "YC-B")
    assert [k for (_, k, _) in sim.event_log if k in ("DEADLOCK_ESCAPE", "DISPATCH")][0] == "DEADLOCK_ESCAPE"
    assert all(j is None for (_, j, *_r) in dec[0][2])            # 탈출 결정 = 전원 WAIT (SERVE 후보 없음)
    assert sim.cost.episode_raw()["interference"] > 0              # WAIT → yielded 가 rate 로 적립
    prof, scn, _ = _k2_stage("k2-down-one")
    sim1, dec1, _ = run_v5(prof, scn, chooser_ref)
    assert sim1.deadlock_escape_count >= 1 and any(cs == ("YC-A",) for (_, cs, _) in dec1)
    prof, scn, _ = _k2_stage("k2-down-both")
    sim2, dec2, _ = run_v5(prof, scn, chooser_ref)
    assert sim2.deadlock_escape_count == 0 and all(t >= 1000.0 for (t, _, _) in dec2)   # 둘 다 고장 → 복구 뒤에야 결정
    assert REPORT["k2-dead-first"]["escapes"] >= 1 and REPORT["k2-down-one"]["escapes"] >= 1 and REPORT["k2-down-both"]["escapes"] == 0


def test_k2_python_loop_matches_jit():
    """시험 4) 의 K=2 판 — jit 없이 파이썬 루프로 step 을 반복해도 잎 전부(비트) 같다."""
    prof, scn = profile_k2(lanes=2), piece2_scenario()
    w_j, _, tb = run_array(prof, scn, use_jit=True)
    w_p, tr_p, _ = run_array(prof, scn, use_jit=False)
    bad = _leaf_diff(w_j, w_p)
    assert not bad, f"jit 판과 파이썬 루프 판이 다른 잎: {bad}"


def test_k2_vmap_batch_matches_single_worlds():
    """K=2 세계 둘(spec d2000 · d310)을 vmap(run) 으로 — cond(결정/탈출/사건/종료)가 select 로 풀려도 단일 판과 잎 전부 같다."""
    prof = profile_k2(lanes=2)
    caps, s_max = dict(n_max=8, q_cap=32, log_cap=328), 320
    g = Geom.from_profile(prof)
    worlds = [to_block_world(prof, s, **caps)[0] for s in (piece2_scenario(), piece2_scenario((310.0, 400.0)))]
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


def test_k2_replan_actually_changes_plan():
    """k2-replan 무대의 설계 의도 — v5 만으로: 첫 결정에서 둘 다 열리고 둘 다 SERVE, YC-B 의 J-OUT-2 재조작 목적지가
    시작 시점 (6,1) 에서 배정 시점 (10,1) 로 바뀐다 (lockstep 의 plan_changed_pick 이 이 무대에서 ≥1)."""
    prof, scn, _ = _k2_stage("k2-replan-first")
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    dp = sim.run_until_decision()
    assert dp is not None and tuple(dp.crane_ids) == ("YC-A", "YC-B")
    refB = {r.job_id: r for r in sim.candidates_for("YC-B")}
    dst0 = tuple(sim._plan("YC-B", refB["J-OUT-2"]).moves[0].dst)
    sim.assign("YC-A", CraneAssignment("YC-A", CandidateKind.SERVE, job_ref=sim.candidates_for("YC-A")[0]))
    liveB = sim.candidates_for("YC-B")
    assert [r.job_id for r in liveB] == ["J-OUT-2"], [r.job_id for r in liveB]
    dst1 = tuple(sim._plan("YC-B", liveB[0]).moves[0].dst)
    assert dst0[:2] == (6, 1) and dst1[:2] == (10, 1), (dst0, dst1)
    for lab in ("k2-replan-first", "k2-replan-ref"):
        if f"{lab}#lockstep" not in REPORT:
            lockstep_engine(*_k2_stage(lab), f"{lab}#lockstep")
        st = REPORT[f"{lab}#lockstep"]["lockstep"]
        assert st["plan_changed_pick"] >= 1 and st["plan_changed_checked"] >= 1, st


def test_k3_and_stair_stages_exercise_multi_crane_paths():
    """K=3·계단식 무대가 실제로 다중 크레인 경로를 밟았는지 — 2대 이상 동시 개방 · K=3 imbalance>0 · 계단식 c4 거절."""
    for lab in K3_STAGES + K2_STAIR:
        _ensure_k2(lab)
        if f"{lab}#lockstep" not in REPORT:
            lockstep_engine(*_k2_stage(lab), f"{lab}#lockstep")
    k3 = [REPORT[l] for l in K3_STAGES]
    assert all(r["K"] == 3 for r in k3)
    assert sum(r["multi_open"] for r in k3) >= 1, "K=3 무대에서 2대 이상 동시 개방이 없다"
    assert any(r["imbalance"] > 0 for r in k3), "K=3 무대에서 imbalance rate 가 0 — sum_seq 결합 순서가 시험되지 않았다"
    st3 = [REPORT[f"{l}#lockstep"]["lockstep"] for l in K3_STAGES]
    assert sum(s["live_rows"] for s in st3) >= 3
    stair = [REPORT[f"{l}#lockstep"]["lockstep"] for l in K2_STAIR]
    assert sum(s["code4_pairs"] for s in stair) >= 1, "계단식 무대에서 CRANE_INTERFERENCE 가 한 번도 안 나왔다"
    assert sum(REPORT[l]["multi_open"] for l in K2_STAIR) >= 1


# ───────────────────────────────────────────────── ⑦-b 조각 2 고침 (반박 검증 2026-09-25) — 불변식 · 학습 경로 · unroll · yield_count
def test_invariant_bits_detect_rail_swap_gap_and_pairwise():
    """`check_invariants` = v5 1107-1137행: 정상 세계 0 · bay 맞바꿈 → CRANE_ORDER_SWAP(v5 도 같은 코드로 던진다) ·
    간격 < gap → CRANE_MIN_GAP · rail_order 순열을 그대로 읽는지(역순열 함정) · 활성 예약 쌍 레인/통로/토큰/칸 겹침 → PAIRWISE_LOCK."""
    prof, scn = profile_k2(lanes=2), piece2_scenario()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    chk = jax.jit(partial(ES.check_invariants, g=g))
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    sim.check_invariants()
    assert int(chk(w0)) == 0
    cr = w0.cranes
    a, b = sim.fleet.get("YC-A"), sim.fleet.get("YC-B")
    pa, pb = a.state.position_bay, b.state.position_bay
    # ① 순서 뒤집힘
    assert int(chk(w0._replace(cranes=cr._replace(bay=cr.bay[::-1])))) & V_CRANE_ORDER_SWAP
    a.state.position_bay, b.state.position_bay = pb, pa
    with pytest.raises(ConstraintViolation, match="CRANE_ORDER_SWAP"):
        sim.check_invariants()
    a.state.position_bay, b.state.position_bay = pa, pb
    # ② 최소 간격
    w_gap = w0._replace(cranes=cr._replace(bay=cr.bay.at[1].set(cr.bay[0] + g.gap / 2)))
    assert int(chk(w_gap)) == V_CRANE_MIN_GAP, violation_names(int(chk(w_gap)))
    b.state.position_bay = pa + prof.safety_gap_bay / 2
    with pytest.raises(ConstraintViolation, match="CRANE_MIN_GAP"):
        sim.check_invariants()
    b.state.position_bay = pb
    # ③ rail_order 는 '자리 i 의 크레인 번호' 순열 — 자리를 바꾸면 정상 bay 에서도 SWAP (역순열로 읽으면 K=2 에서 못 잡는다)
    w_ro = w0._replace(cranes=cr._replace(rail_order=jnp.array([1, 0], jnp.int32)))
    assert int(chk(w_ro)) & V_CRANE_ORDER_SWAP
    # ④ 활성 예약 쌍 — 레인 공유 / 통로 겹침 / 토큰 공유 / 칸 공유 각각, 그리고 겹치지 않는 쌍은 통과
    res = w0.res
    B, R, _ = w0.stacks.shape
    def mk(lo, hi, lane, tok, slot_a=None, slot_b=None):
        slots = jnp.zeros((2, B, R), bool)
        if slot_a is not None:
            slots = slots.at[0, slot_a[0], slot_a[1]].set(True)
        if slot_b is not None:
            slots = slots.at[1, slot_b[0], slot_b[1]].set(True)
        return w0._replace(res=res._replace(active=jnp.array([True, True]), token=jnp.array(tok, jnp.int32),
                                            lo=jnp.array(lo, jnp.float64), hi=jnp.array(hi, jnp.float64),
                                            lane=jnp.array(lane, jnp.int32), slots=slots))
    assert int(chk(mk([1.0, 6.0], [2.0, 8.0], [0, 1], [0, 1]))) == 0                 # 떨어진 통로·다른 레인·다른 토큰
    assert int(chk(mk([1.0, 6.0], [2.0, 8.0], [0, 0], [0, 1]))) == V_PAIRWISE_LOCK    # LANE_DOUBLE
    assert int(chk(mk([1.0, 3.5], [2.0, 8.0], [0, 1], [0, 1]))) == V_PAIRWISE_LOCK    # CORRIDOR_OVERLAP (gap 2: 2+2 > 3.5)
    assert int(chk(mk([1.0, 6.0], [2.0, 8.0], [0, 1], [0, 0]))) == V_PAIRWISE_LOCK    # TOKEN_DOUBLE
    assert int(chk(mk([1.0, 6.0], [2.0, 8.0], [0, 1], [0, 1], (0, 0), (0, 0)))) == V_PAIRWISE_LOCK   # SLOT_DOUBLE
    # v5 의 같은 검사 (_assert_pairwise_resources) 가 같은 입력에서 던진다
    sim.reservations._by_crane["YC-A"] = Reservation("YC-A", "J-OUT-1", Corridor(1.0, 2.0), frozenset(), "L1", 100.0)
    sim.reservations._by_crane["YC-B"] = Reservation("YC-B", "J-OUT-2", Corridor(6.0, 8.0), frozenset(), "L1", 100.0)
    with pytest.raises(ConstraintViolation, match="LANE_DOUBLE"):
        sim._assert_pairwise_resources()
    sim.reservations._by_crane["YC-B"] = Reservation("YC-B", "J-OUT-2", Corridor(3.5, 8.0), frozenset(), "L2", 100.0)
    with pytest.raises(ConstraintViolation, match="CORRIDOR_OVERLAP"):
        sim._assert_pairwise_resources()
    sim.reservations._by_crane["YC-B"] = Reservation("YC-B", "J-OUT-2", Corridor(6.0, 8.0), frozenset(), "L2", 100.0)
    sim._assert_pairwise_resources()
    # ⑤ 엔진 경로: 결정·사건 뒤 검사가 실제로 들어간다 — 첫 스텝부터 rail_order 를 뒤집어 넣으면 첫 결정/사건에서 비트가 켜진다
    w_bad, tr = run_jit(w_ro, None, g, first_by_id, 64)
    assert int(w_bad.violation) & V_CRANE_ORDER_SWAP
    w_off, _ = run_jit(w_ro, None, g, first_by_id, 64, False)              # check=False 면 검사하지 않는다
    assert not (int(w_off.violation) & (V_CRANE_ORDER_SWAP | V_CRANE_MIN_GAP | V_PAIRWISE_LOCK))


def test_run_while_matches_run_single_and_vmap():
    """학습 경로 `run_while`(lax.while_loop) 의 최종 세계 == `run`(lax.scan) 의 세계 — 단일 · vmap(세계 2개, 종료 시각 다름)."""
    prof = profile_k2(lanes=2)
    caps, s_max = dict(n_max=16, q_cap=64, log_cap=400), 384
    g = Geom.from_profile(prof)
    scns = [piece2_scenario(), random_scenario(14, int_arrivals=True)]           # 오더 수·종료 시점이 다른 두 세계
    c_max = max(len(s.containers) for s in scns) + caps["n_max"] + 3
    worlds = [to_block_world(prof, s, c_max=c_max, **caps)[0] for s in scns]
    singles = []
    for w0 in worlds:
        ws, _ = run_jit(w0, None, g, first_by_id, s_max)
        ww = ES.run_while_jit(w0, None, g, first_by_id, s_max)
        bad = _leaf_diff(ws, ww)
        assert not bad, f"run_while 와 run 이 다른 잎 {bad}"
        assert bool(ww.terminal) and int(ww.violation) == 0 and int(ww.steps) == int(ws.steps)
        singles.append(ws)
    assert _leaf_diff(singles[0], singles[1]), "두 세계의 궤적이 같다 — 배치 안 종료 시점 차이가 시험되지 않는다"
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    wb = jax.jit(jax.vmap(lambda w: ES.run_while(w, None, g, first_by_id, s_max)))(batched)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(worlds[0])]
    for i, ws in enumerate(singles):
        lb = [np.asarray(l)[i] for l in jax.tree_util.tree_leaves(wb)]
        ls = [np.asarray(l) for l in jax.tree_util.tree_leaves(ws)]
        bad = [names[j] for j, (a, b) in enumerate(zip(lb, ls)) if not np.array_equal(a, b, equal_nan=True)]
        assert not bad, f"세계 {i}: vmap(run_while) 판과 단일 판이 다른 잎 {bad}"


def test_advance_unroll_is_bit_identical():
    """advance 의 N·2N 단 scan 을 unroll=1(조각 1 원판)·16(기본)·전부 로 펴도 최종 세계 잎 전부 비트 동일 —
    같은 순서·같은 반올림이라는 주장을 시험으로 고정한다 (장부 모드라 terminal_walk 도 실제로 돈다)."""
    prof, scn = profile_k2(lanes=2), piece2_scenario()
    caps, s_max = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    assert bool(ES.ledger_mode(w0.orders))
    saved = ES.ADVANCE_UNROLL
    outs = {}
    try:
        for u in (1, 16, 10_000):
            ES.ADVANCE_UNROLL = u
            f = jax.jit(lambda w: ES.run(w, None, g, first_by_id, s_max))   # 새 함수 → 새 추적 (전역이 바뀌었으므로)
            outs[u], _ = f(w0)
    finally:
        ES.ADVANCE_UNROLL = saved
    assert saved == 16
    for u in (16, 10_000):
        bad = _leaf_diff(outs[1], outs[u])
        assert not bad, f"unroll={u} 판이 unroll=1 판과 다른 잎 {bad}"
    assert bool(outs[1].terminal) and int(outs[1].violation) == 0 and float(outs[1].ledger.terminal_area) > 0


def test_lost_contention_increments_yield_count():
    """assign 687-688행: WAIT + yield_reason=LOST_CONTENTION → recent_yield_count+1. 배열은 `lost` 마스크로 같은 규칙,
    정본 구동(lost=None)에서는 0 그대로 — v5 를 같은 결정에서 같은 사유로 돌려 대조."""
    prof, scn = profile_k2(lanes=2), piece2_scenario()
    caps, _ = _caps(scn, prof)
    w, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    step_jit = jax.jit(partial(ES.step, params=None, g=g, policy_fn=first_by_id))
    while True:
        w_pre = w
        w, tr = step_jit(w, None)
        if bool(tr.decided):
            break
    assert int(jnp.sum(tr.open)) == 2                                    # 300초 — 둘 다 열린다
    wait_all = lambda params, x, mask: jnp.full((mask.shape[0],), EMPTY_ID, jnp.int32)
    d0 = DP.dispatch(w_pre, None, g, wait_all)
    d1 = DP.dispatch(w_pre, None, g, wait_all, lost=jnp.array([True, False]))
    assert [int(v) for v in d0.world.cranes.yield_count] == [0, 0]
    assert [int(v) for v in d1.world.cranes.yield_count] == [1, 0]
    assert [bool(v) for v in d1.world.cranes.yielded] == [True, True]
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    dp = sim.run_until_decision()
    assert tuple(dp.crane_ids) == ("YC-A", "YC-B")
    sim.assign("YC-A", CraneAssignment("YC-A", CandidateKind.WAIT, yield_reason="LOST_CONTENTION"))
    sim.assign("YC-B", CraneAssignment("YC-B", CandidateKind.WAIT))
    sim.close_decision()
    dd = from_block_world(d1.world, tb)
    for cid in ("YC-A", "YC-B"):
        yc = sim.fleet.get(cid)
        assert (dd["cranes"][cid]["recent_yield_count"], dd["cranes"][cid]["yielded"]) == (yc.recent_yield_count, yc.yielded)


# ═════════════════════════════════════════════════ 조각 3·4 — PRE_ADVICE(wake·PRE·REPO·공동 규약) · 본선·이송
def _load_test(name: str):
    """tests/v6/<name>.py 를 **호출 시점에** 불러온다 (tests/ 에 __init__ 이 없다; 위에서 부르면 test_gpu_escape 가
    이 파일을 되불러 순환)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_{name}_for_equiv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LAZY: dict[str, object] = {}


def _lazy(name: str):
    if name not in _LAZY:
        _LAZY[name] = _load_test(name)
    return _LAZY[name]


# ── 조각 4 무대 (test_gpu_vessel 의 빌더 재사용) ──
def _v4_stage(name: str):
    V = _lazy("test_gpu_vessel")
    if name == "v4-fixture-k1":
        return V.stage_base() + ("first",)
    if name == "v4-stage-a-k1":
        return V.stage_a() + ("first",)
    if name == "v4-stage-b-k1":
        return V.stage_b() + ("first",)
    if name == "v4-stage-c-k1":
        return V.stage_c() + ("first",)
    if name == "v4-stage-b-k1-ref":
        return V.stage_b() + ("ref",)
    if name == "v4-fixture-k2":
        return fixtures.build_integrated_profile(), fixtures.build_minimal_terminal_scenario(), "first"
    if name == "v4-fixture-k2-ref":
        return fixtures.build_integrated_profile(), fixtures.build_minimal_terminal_scenario(), "ref"
    if name == "v4-stage-b-k2":
        _, scn = V.stage_b()
        return fixtures.build_integrated_profile(), scn, "first"
    raise KeyError(name)


V4_STAGES = ["v4-fixture-k1", "v4-stage-a-k1", "v4-stage-b-k1", "v4-stage-c-k1", "v4-stage-b-k1-ref",
             "v4-fixture-k2", "v4-fixture-k2-ref", "v4-stage-b-k2"]


@pytest.mark.parametrize("stage", V4_STAGES, ids=V4_STAGES)
def test_v4_vessel_equivalence(stage):
    """본선·이송 처리기가 엔진에 끼워진 채로 완주 — 조각 1 전 항목 + 배·이송 열 == v5."""
    prof, scn, policy = _v4_stage(stage)
    assert scn.vessels, "본선이 없는 무대"
    sim, w, trace, tb = _run_and_compare(prof, scn, stage, policy=policy)
    kinds = {k for (_, k, _) in sim.event_log}
    assert {"VESSEL_START", "STS_MOVE", "TRANSFER_ARRIVE", "VESSEL_RELEASED"} <= kinds, kinds
    if stage in ("v4-stage-a-k1", "v4-stage-b-k1", "v4-stage-b-k1-ref", "v4-stage-b-k2", "v4-stage-c-k1"):
        assert sim.cost.episode_raw()["sts_wait"] > 0, "무대가 STS 막힘을 만들지 못했다"
    if stage == "v4-stage-a-k1":
        assert sim.cost.episode_raw()["transfer_wait"] > 0, "무대가 이송 대기를 만들지 못했다"
    if stage == "v4-stage-c-k1":
        assert sim.cost.episode_raw()["vessel_delay"] > 0 and sim.cost.episode_raw()["depart_delay"] > 0
        assert sim.kpis.snapshot().berth_overrun_s > 0
        assert "PLAN_CHANGE" in kinds
        rel = [p for (_, k, p) in sim.event_log if k == "VESSEL_RELEASED"]
        assert "J-V-DISC-0-100" in rel and rel.index("J-V-DISC-0-100") < rel.index("J-V-DISC-0-11"), "사전식 해제 순서"


# ── 조각 3 무대 — 순차 규약 (ReferenceDispatcher 의미 · PRE_ADVICE) ──
def _p3_seq_stage(name: str):
    W = _lazy("test_gpu_wake")
    prof = W.PROF                                                         # fixtures 프로파일 (K=2 · 1..40 · 지평 1800)
    if name == "p3-seq-blocked":
        return prof, W.blocked_target_sc(2500.0, 3000.0), "ref"           # wake 700 = 2500 − 1800 (test_yr050)
    if name == "p3-seq-blocked-early":
        return prof, W.blocked_target_sc(900.0, 1500.0, sid="blocked-early"), "ref"   # wake 0 (max(0, 900−1800))
    if name == "p3-seq-busy-at-wake":
        return prof, W.busy_at_wake_sc(), "ref"
    if name == "p3-seq-neg-gap":
        return prof, W.neg_gap_sc(), "ref"
    if name == "p3-seq-gate-in-eta":
        return prof, W.gate_in_eta_sc(), "ref"
    if name == "p3-seq-crowded-eta":
        return piece1_profile(), W.with_eta(crowded_scenario(), 300.0, "crowded-eta"), "first"
    if name == "p3-seq-crowded-eta-k2":
        return profile_k2(lanes=2), W.with_eta(crowded_scenario(), 600.0, "crowded-eta-k2"), "first"
    if name.startswith("p3-seq-random-"):
        seed = int(name.split("-")[-1][1:])
        rng = random.Random(seed)
        scn = W.with_eta(random_scenario(seed, int_arrivals=True), lambda j, r: float(r.randint(-600, 900)),
                         f"random-eta-{seed}", rng=rng)
        return profile_k2(lanes=2), scn, "first"
    raise KeyError(name)


P3_SEQ_STAGES = ["p3-seq-blocked", "p3-seq-blocked-early", "p3-seq-busy-at-wake", "p3-seq-neg-gap", "p3-seq-gate-in-eta",
                 "p3-seq-crowded-eta", "p3-seq-crowded-eta-k2", "p3-seq-random-s3", "p3-seq-random-s14"]


@pytest.mark.parametrize("stage", P3_SEQ_STAGES, ids=P3_SEQ_STAGES)
def test_p3_pre_advice_sequential_equivalence(stage):
    """PRE_ADVICE + 순차 규약: ETA wake 가 결정을 열고(armed & eta_opportunity) SERVE 후보가 없으면 WAIT — 사건열(ETA_WAKE
    포함)·결정열·상태 전부 == v5. W·A 국면이 실제로 밟혔는지 집계."""
    prof, scn, policy = _p3_seq_stage(stage)
    sim, w, trace, tb = _run_and_compare(prof, scn, stage, policy=policy, level=PA)
    r = REPORT[stage]
    if stage != "p3-seq-gate-in-eta":
        assert r["eta_wakes"] >= 1, "ETA_WAKE 가 없다"
    if stage in ("p3-seq-blocked", "p3-seq-busy-at-wake"):                 # neg-gap 은 wake 0 (시계 전진 없이 소비)
        assert r["advanced"] >= 1, "wake 시각으로의 전진(A 국면)이 없었다"
    if stage == "p3-seq-gate-in-eta":
        wakes = [p for (_, k, p) in sim.event_log if k == "ETA_WAKE"]
        assert wakes == ["J-OUT-T"], wakes                              # GATE_IN 의 ETA 는 시드되지 않는다
    if stage == "p3-seq-blocked":
        t_wake = next(t for (t, k, _) in sim.event_log if k == "ETA_WAKE")
        assert t_wake == 700.0
        dec_t = [t for (t, _, rec) in _array_decisions(w, trace, tb)]
        assert 700.0 in dec_t, dec_t                                    # wake 로 열린 결정 (SERVE 없음 → WAIT)


# ── 조각 3 무대 — 공동 규약 (CentralResolver + generate · PRE/REPO 실행) ──
_RESOLVERS: dict[tuple, object] = {}


def _resolver_fn(pref: str, g, count_lost: bool):
    key = (pref, g, count_lost)
    if key not in _RESOLVERS:
        _RESOLVERS[key] = DP.make_resolver(pref, g, count_lost=count_lost)
    return _RESOLVERS[key]


def policy_waitall_joint(params, world, c3, fl, pr, open_):
    K = open_.shape[0]
    return jnp.full((K,), EMPTY_ID, jnp.int32), jnp.zeros((K,), bool), jnp.int32(0)


def run_v5_joint(prof, scn, level, pref: str, *, count_lost: bool = True, waitall: bool = False):
    """v5 공동 규약 구동 — CentralResolver(선호) + CandidateGenerator(LEGACY_DEFAULT) + resolver.apply (yield_reason 전달 →
    LOST_CONTENTION 이면 yield_count). waitall 이면 전원 WAIT commit (test_yr050:147-161).
    반환 (sim, 결정열 [(t, crane_ids, [(crane, 종류, 오더/REPO 이름|None)])], 통계)."""
    from yard_rl.v6.world.integrated.baselines import ServiceFirstSPTPreference
    from yard_rl.v6.world.integrated.candidates import CandidateGenerator
    from yard_rl.v6.world.integrated.policy_config import LEGACY_DEFAULT
    from yard_rl.v6.world.integrated.resolver import BaselinePreference, CentralResolver
    prefs = {"baseline": BaselinePreference, "sf_spt": ServiceFirstSPTPreference}
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    resolver = CentralResolver(prefs[pref]())
    decisions = []
    st = dict(lost=0, pruned=0, items=0)
    prev = None
    while (dp := sim.run_until_decision()) is not None:
        assert prev is None or dp.time > prev, f"같은 시각 {dp.time} 재결정 — wake 1회성 위반"
        prev = dp.time
        if waitall:
            assert len(decisions) < 200, "결정 폭주 — 재질문 무한루프 의심"
            sim.commit_decisions([CraneAssignment(c, CandidateKind.WAIT) for c in dp.crane_ids])
            decisions.append((dp.time, tuple(dp.crane_ids), [(c, "WAIT", None) for c in dp.crane_ids]))
            continue
        gb = {c: gen.generate(sim, c, level) for c in dp.crane_ids}
        for c in dp.crane_ids:
            raw = gen._serve(sim, c, sim.now) + gen._pre_rehandle(sim, c, sim.now, level) + gen._reposition(sim, c, sim.now, level)
            st["pruned"] += int(len(raw) + 1 > len(gb[c].items))
            st["items"] += len(gb[c].items)
        resn = resolver.resolve(sim, dp, gb)
        rec = []
        for r in resn.resolutions:
            if r.action == CandidateKind.WAIT:
                rec.append((r.crane_id, "WAIT", None))
                st["lost"] += int(r.yield_reason == "LOST_CONTENTION")
            else:
                gc = gb[r.crane_id].items[r.chosen_candidate_id]
                rec.append((r.crane_id, r.action.value, gc.job_ref.job_id))
        if count_lost:
            resolver.apply(sim, resn, gb)
        else:
            from yard_rl.v6.world.integrated.baselines import _apply
            _apply(sim, {r.crane_id: (gb[r.crane_id].items[r.chosen_candidate_id] if r.chosen_candidate_id is not None
                                      else next(g for g in gb[r.crane_id].items if g.kind == CandidateKind.WAIT))
                         for r in resn.resolutions})
        decisions.append((dp.time, tuple(dp.crane_ids), rec))
    assert sim.terminal
    return sim, decisions, st


def _array_joint_decisions(w, trace, tb):
    """StepTrace → [(t, crane_ids, [(crane, 종류 이름, 오더/REPO 이름|None)])]."""
    n_steps = int(w.steps)
    tr = jax.tree_util.tree_map(lambda a: np.asarray(a)[:n_steps], trace)
    K = len(tb.crane_ids)
    out = []
    for i in range(n_steps):
        if not tr.decided[i]:
            continue
        rec = []
        ks = [k for k in range(K) if tr.open[i][k]]
        for k in ks:
            kind = int(tr.pick_kind[i][k])
            cid = tb.crane_ids[k]
            if kind == PK_WAIT or kind < 0:
                rec.append((cid, "WAIT", None))
            elif kind == PK_REPOSITION:
                rec.append((cid, "REPOSITION", repo_job_id(cid, float(tr.pick_bay[i][k]))))
            else:
                rec.append((cid, KIND_NAME[kind], tb.job_ids[int(tr.pick[i][k])]))
        out.append((float(tr.clock[i]), tuple(tb.crane_ids[k] for k in ks), rec))
    return out


def compare_joint_decisions(v5_dec, w, trace, tb, label):
    ar = _array_joint_decisions(w, trace, tb)
    v5s = [(round(t, 6), cs, rec) for (t, cs, rec) in v5_dec]
    ars = [(round(t, 6), cs, rec) for (t, cs, rec) in ar]
    diff = _first_diff(v5s, ars)
    if diff is not None:
        i, x, y = diff
        pytest.fail(f"[{label}] ② 공동 결정열이 {i}번째에서 갈린다:\n  v5 ={x}\n  arr={y}")
    return ar


def _p3_joint_stage(name: str):
    C = _lazy("test_gpu_cands3")
    key = name[len("p3-joint-"):]
    if key == "waitall-blocked":
        W = _lazy("test_gpu_wake")
        return W.PROF, W.blocked_target_sc(2500.0, 3000.0), PA, "baseline", True
    if key == "waitall-crowded":
        return C.prof_k2(2.0), C.crowded_eta_scenario(), PA, "baseline", True
    if key == "blocked":
        W = _lazy("test_gpu_wake")
        return W.PROF, W.blocked_target_sc(2500.0, 3000.0), PA, "baseline", False
    if key == "neg-gap":
        W = _lazy("test_gpu_wake")
        return W.PROF, W.neg_gap_sc(), PA, "baseline", False
    prof, scn, level, pref = C.STAGES[key]()
    return prof, scn, level, {"baseline": "baseline", "spt": "sf_spt"}[pref], False


P3_JOINT_STAGES = ["p3-joint-blocked", "p3-joint-neg-gap", "p3-joint-eta-basic", "p3-joint-crowded-eta",
                   "p3-joint-dead-first-eta", "p3-joint-plan-failed-mandatory",
                   "p3-joint-eta-random-s1", "p3-joint-eta-random-s2", "p3-joint-eta-random-s3", "p3-joint-eta-random-s4",
                   "p3-joint-eta-random-s5", "p3-joint-eta-random-s6", "p3-joint-spt-s7", "p3-joint-spt-s8",
                   "p3-joint-block-arrival-s9", "p3-joint-block-arrival-s10",
                   "p3-joint-waitall-blocked", "p3-joint-waitall-crowded"]


@pytest.mark.parametrize("stage", P3_JOINT_STAGES, ids=P3_JOINT_STAGES)
def test_p3_joint_resolver_equivalence(stage):
    """공동 규약 — v5 CentralResolver(선호)+generate 로 완주한 것과 (a) 사건열(ETA_WAKE·DISPATCH 의 PRE/REPO payload 포함)
    (b) 결정열(크레인·종류·오더/REPO 이름) (c) 상태 전부 == . PRE_REHANDLE·REPOSITION 이 실제로 실행되는 무대를 포함한다."""
    prof, scn, level, pref, waitall = _p3_joint_stage(stage)
    sim, dec, st = run_v5_joint(prof, scn, level, pref, count_lost=True, waitall=waitall)
    caps, s_max = _caps(scn, prof)
    g = Geom.from_profile(prof)
    if waitall:
        policy_fn, params_fn = policy_waitall_joint, (lambda w0, tb, g_: None)
    else:
        policy_fn, params_fn = _resolver_fn(pref, g, True), (lambda w0, tb, g_: DP.resolver_params(tb, g_))
    w, trace, tb = run_array(prof, scn, policy_fn=policy_fn, params_fn=params_fn, level=level, joint=True)
    ar_dec = compare_joint_decisions(dec, w, trace, tb, stage)
    err = compare(sim, None, w, trace, tb, stage)
    kinds: dict[str, int] = {}
    for (_, _, rec) in dec:
        for (_, kd, _) in rec:
            kinds[kd] = kinds.get(kd, 0) + 1
    n_wait = kinds.get("WAIT", 0)
    times = [t for (t, _, _) in sim.event_log]
    same = sum(1 for t in set(times) if len({k for (tt, k, _) in sim.event_log if tt == t and k != "DISPATCH"}) >= 2)
    REPORT[stage] = dict(
        steps=int(w.steps), decisions=len(dec), waits=n_wait, events=len(sim.event_log), same_time=same,
        find_slot_calls=0, exact_ties=0, max_float_err=err,
        nonzero_cost={k: round(v, 3) for k, v in sim.cost.episode_raw().items() if v},
        backlog=sim.unfinished_backlog(), interference=sim.cost.episode_raw()["interference"],
        imbalance=sim.cost.episode_raw()["imbalance"], K=len(tb.crane_ids), escapes=sim.deadlock_escape_count,
        multi_open=sum(1 for (_, cs, _) in dec if len(cs) >= 2),
        eta_wakes=sum(1 for (_, k, _) in sim.event_log if k == "ETA_WAKE"),
        advanced=int(np.asarray(trace.advanced)[:int(w.steps)].sum()), woke=int(np.asarray(trace.woke)[:int(w.steps)].sum()),
        joint_decisions=len(dec), kinds=kinds, pre_exec=kinds.get("PRE_REHANDLE", 0), repo_exec=kinds.get("REPOSITION", 0),
        lost=st["lost"], pruned=st["pruned"], vessels=0, sts_wait=0.0, transfer_wait=0.0, pref=pref)
    if stage == "p3-joint-blocked":
        assert kinds.get("PRE_REHANDLE", 0) >= 1, kinds                 # 도착(3000) 전 700 에 선제 재조작
        t_pre = next(t for (t, _, rec) in dec for (_, kd, _) in rec if kd == "PRE_REHANDLE")
        assert t_pre == 700.0 and sim.jobs["J-OUT-T"].status.name == "DONE"
        assert sim.kpis.snapshot().rehandle_count == 1 and sim.jobs["J-OUT-T"].rehandle_count == 0   # test_yr050:69-76 규약
    if stage == "p3-joint-dead-first-eta":
        assert sim.deadlock_escape_count >= 1 and kinds.get("REPOSITION", 0) >= 1, (sim.deadlock_escape_count, kinds)
    if stage == "p3-joint-crowded-eta":
        assert st["pruned"] >= 1 and kinds.get("PRE_REHANDLE", 0) + kinds.get("REPOSITION", 0) >= 1, (st, kinds)
    if waitall:
        assert 1 <= len(dec) < 200 and all(b > a for a, b in zip([t for (t, _, _) in dec], [t for (t, _, _) in dec][1:]))
        assert all(kd == "WAIT" for (_, _, rec) in dec for (_, kd, _) in rec)


def test_p3_joint_escape_paths_and_kinds_exercised():
    """조각 3 경로 집계 — 공동 무대 전체에서 PRE_REHANDLE 실행 ≥ 3 · REPOSITION 실행 ≥ 3 · LOST_CONTENTION ≥ 1 · prune ≥ 1 ·
    ETA_WAKE ≥ 5 · A 국면 ≥ 3 · 탈출 ≥ 1 · spt 선호 무대 ≥ 2. (이름의 escape_paths 는 verify_chunked.sh 의 집계 시험 패턴 —
    조각 실행에서 빠지고 --report 에서 병합 REPORT 로 돈다; REPORT 가 비면 무대를 직접 돌린다 ≈ 90초.)"""
    for lab in P3_JOINT_STAGES:
        if lab not in REPORT:
            test_p3_joint_resolver_equivalence(lab)
    rs = [REPORT[l] for l in P3_JOINT_STAGES]
    assert sum(r["pre_exec"] for r in rs) >= 3, [r["kinds"] for r in rs]
    assert sum(r["repo_exec"] for r in rs) >= 3, [r["kinds"] for r in rs]
    assert sum(r["lost"] for r in rs) >= 1, "LOST_CONTENTION(yield_count) 경로가 안 밟혔다"
    assert sum(r["pruned"] for r in rs) >= 1
    assert sum(r["eta_wakes"] for r in rs) >= 5
    assert sum(r["advanced"] for r in rs) >= 3
    assert sum(r["escapes"] for r in rs) >= 1
    assert sum(1 for r in rs if r["pref"] == "sf_spt") >= 2


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
            if "lockstep" in r:
                continue
            if "decisions" not in r:
                print(f"  {k:24s} {r}")
                continue
            print(f"  {k:24s} steps={r['steps']:4d} dec={r['decisions']:3d} wait={r['waits']:2d} ev={r['events']:3d} "
                  f"same_t={r['same_time']:2d} find_slot={r['find_slot_calls']:4d} ties={r['exact_ties']:3d} "
                  f"mismatch={r.get('float_mismatches', 0)} maxerr={r['max_float_err']:.2e} backlog={r['backlog']} "
                  f"norm={r.get('log_payload_normalized', 0)} cost≠0={r['nonzero_cost']}")
        print(f"  exact ties total = {total_ties} · WAIT total = {total_waits} · same-time steps = {total_same} "
              f"· backlog>0 stages = {n_backlog} · float mismatches = {total_mismatch}")
        k2 = {k: r for k, r in stages.items() if r.get("K", 1) >= 2}
        lock = {k: r["lockstep"] for k, r in REPORT.items() if "lockstep" in r}
        if k2:
            codes = {}
            for r in lock.values():
                for c, n in r["codes"].items():
                    codes[c] = codes.get(c, 0) + n
            print(f"[piece 2 report]  K>=2 stages={len(k2)} (K=3: {sum(1 for r in k2.values() if r['K'] == 3)}) · escapes(v5 DEADLOCK_ESCAPE)={sum(r['escapes'] for r in k2.values())} "
                  f"in {sum(1 for r in k2.values() if r['escapes'])} stages · multi-open decisions={sum(r['multi_open'] for r in k2.values())} "
                  f"· interference>0 stages={sum(1 for r in k2.values() if r['interference'] > 0)} "
                  f"· imbalance>0 stages={sum(1 for r in k2.values() if r['imbalance'] > 0)}")
            print(f"  lockstep: decisions={sum(r['decisions'] for r in lock.values())} · decide==decide_seq checks="
                  f"{sum(r['decide_vs_module'] for r in lock.values())} · reject codes over feasible pairs (0 OK · 3 LANE · 4 INTERF)={dict(sorted(codes.items()))} "
                  f"· regained-after-escape={sum(r['regained'] for r in lock.values())} · WAIT={sum(r['waits'] for r in lock.values())}")
            print(f"  lockstep: live rows(all open cranes)={sum(r['live_rows'] for r in lock.values())} · plan_changed(any/pick/checked)="
                  f"{sum(r['plan_changed_any'] for r in lock.values())}/{sum(r['plan_changed_pick'] for r in lock.values())}/"
                  f"{sum(r['plan_changed_checked'] for r in lock.values())} · invariant checks={sum(r['invariant_checks'] for r in lock.values())}")
        p34 = {k: r for k, r in REPORT.items() if k.startswith(("v4-", "p3-"))}
        if p34:
            print(f"[piece 3/4 report]  stages={len(p34)} · vessel stages={sum(1 for r in p34.values() if r.get('vessels', 0))} "
                  f"(sts_wait>0: {sum(1 for r in p34.values() if r.get('sts_wait', 0) > 0)} · transfer_wait>0: "
                  f"{sum(1 for r in p34.values() if r.get('transfer_wait', 0) > 0)}) · ETA_WAKE total={sum(r.get('eta_wakes', 0) for r in p34.values())} "
                  f"· W steps={sum(r.get('woke', 0) for r in p34.values())} · A steps={sum(r.get('advanced', 0) for r in p34.values())} "
                  f"· PRE exec={sum(r.get('pre_exec', 0) for r in p34.values())} · REPO exec={sum(r.get('repo_exec', 0) for r in p34.values())} "
                  f"· escapes={sum(r.get('escapes', 0) for r in p34.values())} · lost={sum(r.get('lost', 0) for r in p34.values())}")
            for k, r in p34.items():
                if "joint_decisions" in r:
                    print(f"  {k:28s} dec={r['joint_decisions']} kinds={r['kinds']} wakes={r['eta_wakes']} A={r['advanced']} "
                          f"escapes={r['escapes']} lost={r['lost']} pruned={r.get('pruned', '-')}")
    assert total_ties > 0, "어느 무대에서도 find_slot 정확 동률이 없었다 — tie-break 규칙이 시험되지 않았다"
    assert total_waits >= 1, "WAIT 결정이 한 번도 없었다 — yielded 경로가 시험되지 않았다"
    assert total_same >= 1, "같은 시각에 종류가 다른 사건이 한 번도 없었다 — 큐 우선순위가 엔진 수준에서 시험되지 않았다"
    assert n_backlog >= 1, "미배차 잔존(backlog>0) 무대가 없었다"
    assert total_mismatch == 0
    if k2:
        assert sum(r["escapes"] for r in k2.values()) >= 1, "K=2 무대에서 탈출이 한 번도 발화하지 않았다"
        assert sum(r["multi_open"] for r in k2.values()) >= 3, "2대가 동시에 열린 결정이 너무 적다"
        assert codes.get(4, 0) >= 1 and codes.get(3, 0) >= 1, f"c3/c4 거절 코드가 엔진 경로에서 안 나왔다: {codes}"
        assert any(r["interference"] > 0 for r in k2.values()), "interference rate 가 어느 K=2 무대에서도 0"
        assert sum(r["plan_changed_pick"] for r in lock.values()) >= 1, "재계획이 계획을 실제로 바꾼 결정이 없다 — 재계획 분기가 차등 시험되지 않았다"
        assert sum(r["plan_changed_checked"] for r in lock.values()) >= 1
        assert sum(r["live_rows"] for r in lock.values()) >= sum(r["decisions"] for r in lock.values())
        assert any(r["K"] == 3 for r in k2.values()), "엔진 수준 K=3 무대가 없다"
        assert sum(r["invariant_checks"] for r in lock.values()) >= 1
