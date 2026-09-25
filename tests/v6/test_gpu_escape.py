"""교착 탈출(gpu/escape.py) 이 v5 `_try_escape`·`interference_deadlock_corridors` 와 **같은 답**을 내는가
([[YR-327]] 조각 2 · engine.py:384-450, 302-305).

■ 방법 — v5 를 실제로 굴리며 `_try_escape` 가 불리는 **모든 순간**을 가로챈다 (기대값 손기입 없음)
  hook (앞) : 그 순간의 v5 상태를 배열 세계로 옮기고(`world_from_sim`), v5 술어 `interference_deadlock_corridors()`,
              크레인×오더 **전 쌍**의 (_dispatchable · 계획 성립 · reject_reason 코드), 크레인별 candidates_for 를 적는다
  hook (뒤) : 원래 `_try_escape` 를 부르고 결과(None / TerminalDecision.crane_ids)·escape_at·last_decision_at·
              escape_count·yielded·event_log 를 적는다
  배열      : `candidate_matrices` → `try_escape` 를 그 세계에 적용해 위 기록과 **전부** 대조한다
              ① (K,N) dispatchable/feasible/code 행렬 · cand 행 = candidates_for   ② deadlock == (통로 비어있지 않음) ·
              통로 집합   ③ fired · open(=pending) · escape_at · last_decision_at · escape_count · yielded · 로그 전열
  v5 상태→배열 변환기는 (a) reset 직후 `to_block_world` 와 같고 (b) 결정 0회 뒤 첫 탈출까지는 조각 1 엔진(`ES.step`)이
  스스로 도달한 세계와 같음을 따로 단언한다 — 변환기가 답을 만들어내지 않는다는 근거.

■ 무대 (K=2 는 같은 담당구간 1..10 · 초기 위치 3.25 / 7.75)
  탐색   gap ∈ {3,4} × 무작위 시드 1..21 · gap 2 × 시드 1..12 — v5 ReferenceDispatcher(순차 live 재선택)로 완주해
         escape_count>0 인 무대를 **고른다** (2026-09-25 탐색: gap 3 → 12/30, gap 4 → 20/30 시드에서 발화)
  설계   dead-first  첫 도착이 사각지대(bay 5) → 결정 0회 뒤 탈출 (엔진 궤적과 변환기 대조)
         down-both   두 크레인 고장 중 사각지대 도착 → 술어 참·esc 비어 **미발화** (yielded 해제만)
         down-one    YC-B 고장 → 유휴 하나(YC-A)만 결정 대상 · ③ 은 고장 크레인 쌍도 센다
         dead-first-waitall  전원 WAIT 정책 → 정상 결정과 같은 시각의 호출이 last_decision_at 표식(394행)에 막힌다
         regain      600초까지 WAIT → 700초 고장 사건(yielded 안 지움) 시각에 발화 → 해제 뒤 YC-A 가 후보를 되찾는다
  거짓   조각 1 K=1 무대 전부(spec10·crowded·censored·random·fixture·…) + K=2 미발화 시드 → 술어 거짓 무대 ≥ 20
  (K=2 에 조각 1 wait2 정책을 얹은 무대는 '후보 정확히 2개' 가 한 번도 안 나와 ref 와 같았다 — 뺐다)

실행: WSL venv · x64 CPU.  PYTHONPATH=src JAX_PLATFORMS=cpu pytest tests/v6/test_gpu_escape.py -q -s
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu import engine_step as ES                                        # noqa: E402
from yard_rl.v6.gpu.engine_step import first_by_id                                  # noqa: E402
from yard_rl.v6.gpu.escape import (blocked_corridors, candidate_matrices, crane_bits,   # noqa: E402
                                   idle_mask, try_escape)
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import (CRANE_STATUS_NAMES, EV_NAMES, STATUS_NAMES,   # noqa: E402
                                         event_log_from_arrays, to_block_world)
from yard_rl.v6.gpu.reserve import CRANE_INTERFERENCE, REASON_TO_CODE              # noqa: E402
from yard_rl.v6.gpu.stack_ops import from_v5_stacks                                 # noqa: E402
from yard_rl.v6.gpu.state import EMPTY_ID, EMPTY_TIME, LOG_DEADLOCK_ESCAPE          # noqa: E402
# v5 정본
from yard_rl.v6.world.contract.schema import CandidateKind                          # noqa: E402
from yard_rl.v6.world.contract.state import LaneGraph                               # noqa: E402
from yard_rl.v6.world.domain.models import BlockGeometry                            # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                    # noqa: E402
from yard_rl.v6.world.integrated.dispatcher import ReferenceDispatcher              # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator   # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec                   # noqa: E402
from yard_rl.v6.world.integrated.scenario import InjectedEvent, TerminalScenario    # noqa: E402

EPS = ES.EPS
#: 시험 전체 집계 (마지막 시험이 보고)
REPORT: dict[str, dict] = {}
#: 발화 순간의 배열 세계 (jit/vmap 시험이 재사용) — label → (pre_world, tables, g)
FIRED_WORLDS: dict[str, tuple] = {}


def _load_engine_equiv_test():
    """tests/v6/test_gpu_engine_equiv.py 의 무대 빌더를 경로로 불러온다 (tests/ 에 __init__ 이 없다)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_gpu_engine_equiv.py")
    spec = importlib.util.spec_from_file_location("_tgee_for_escape", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EQ = _load_engine_equiv_test()
_caps, _c, _out, _scn = _EQ._caps, _EQ._c, _EQ._out, _EQ._scn
random_scenario, piece1_profile, piece1_scenario = _EQ.random_scenario, _EQ.piece1_profile, _EQ.piece1_scenario

#: v5 로그 payload 가 가리키는 표 (host_convert._TARGET_KIND 와 같은 뜻)
_LOG_CRANE = {"JOB_COMPLETED", "EQUIPMENT_DOWN", "EQUIPMENT_UP"}
_LOG_JOB = {"BLOCK_ARRIVAL", "JOB_RELEASED", "VESSEL_RELEASED", "ETA_UPDATED", "ETA_WAKE"}


# ───────────────────────────────────────────────── 무대 빌더 (K=2)
def prof_k2(gap: float, lo: int = 1, hi: int = 10):
    """크레인 2대(YC-A·YC-B) 같은 담당구간 [lo,hi] · 10×4×4 · 레인 2(인접 없음) · 이송 0 · safety_gap=gap."""
    base = fixtures.build_integrated_profile()
    return replace(base, block=BlockGeometry("B1", 10, 4, 4, 6.5, 2.9, 2.6, 0),
                   cranes=(replace(fixtures._spec("YC-A"), service_bay_min=lo, service_bay_max=hi),
                           replace(fixtures._spec("YC-B"), service_bay_min=lo, service_bay_max=hi)),
                   lane_graph=LaneGraph(("L1", "L2"), ()),
                   transfer=TransferFleetSpec("TF1", "YT", n_units=0, move_time_s=180.0),
                   safety_gap_bay=gap)


def dead_first_scenario(injected=()) -> TerminalScenario:
    """첫 도착(300초)이 bay 5 = 3.25/7.75 · gap 3 의 사각지대 → 결정 0회 뒤 곧바로 교착.
    600초에 bay 2·bay 9 도착 → 둘 다 정상 배정되고 크레인이 움직인 뒤 bay 5 가 풀린다."""
    containers = {"C1": _c("C1", 5, 1, 1), "C2": _c("C2", 2, 1, 1), "C3": _c("C3", 9, 1, 1)}
    jobs = [_out("J-OUT-1", "C1", 0.0, 300.0), _out("J-OUT-2", "C2", 100.0, 600.0), _out("J-OUT-3", "C3", 100.0, 600.0)]
    return TerminalScenario(scenario_id="dead-first", seed=0, horizon_s=3600.0, drain_window_s=0.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=list(injected))


def down_scenario(targets: tuple[str, ...]) -> TerminalScenario:
    """dead-first 에 고장(200초)·복구(1000초) 주입."""
    inj = [InjectedEvent(200.0, "EQUIPMENT_DOWN", t) for t in targets] + \
          [InjectedEvent(1000.0, "EQUIPMENT_UP", t) for t in targets]
    scn = dead_first_scenario(inj)
    return replace(scn, scenario_id=f"down-{'-'.join(targets)}")


# ───────────────────────────────────────────────── 정책 (v5 쪽 — 순차 live 재선택)
_REF = ReferenceDispatcher()


def chooser_ref(sim, cid, cands):
    """ReferenceDispatcher.select — 본선 우선 → 최장 트럭대기 → job_id (dispatcher.py:19-22)."""
    return _REF.select(sim, cid, cands) if cands else None


def chooser_first(sim, cid, cands):
    return cands[0] if cands else None


def chooser_wait2(sim, cid, cands):
    """후보가 정확히 2개면 WAIT, 아니면 마지막 후보 (조각 1 wait2 와 같음)."""
    return None if (not cands or len(cands) == 2) else cands[-1]


def chooser_wait_until(t_s: float):
    """clock ≤ t_s 동안은 후보가 있어도 WAIT, 그 뒤는 ReferenceDispatcher — yielded 를 일부러 남긴다."""
    def chooser(sim, cid, cands):
        return None if (sim.clock <= t_s or not cands) else _REF.select(sim, cid, cands)
    return chooser


# ───────────────────────────────────────────────── v5 진행 중 상태 → 배열 세계
def _opt(v):
    return -np.inf if v is None else float(v)


def world_from_sim(sim, w0, tb, g):
    """v5 진행 중 상태 → 배열 세계 — **술어·탈출이 읽고 쓰는 열만** 채운다 (나머지는 w0 그대로).

    채우는 것: clock·end·escape_at·last_decision_at·escape_count / orders(status·assigned_crane·waiting·block_in_s·
    service_s·done_s·rehandles) / cranes(bay·row·assigned·status·available_at·down·down_pending·yielded·served·
    completions·is_loaded·loaded_m·empty_m) / stacks·conts / res 전부 / log 전부.
    안 채우는 것: queue·plan·kpi·ledger·cost·lane·wake·decision (술어가 안 읽고, 탈출은 decision 을 덮어쓴다).
    """
    K, N = w0.k, w0.n
    B, R, _ = w0.stacks.shape
    cidx, jidx = tb.crane_index, tb.job_index
    lidx = {l: i for i, l in enumerate(tb.lane_ids)}

    # 오더 (domain/models.py:32-75 · kpis._waiting)
    o = {f: np.asarray(getattr(w0.orders, f)).copy()
         for f in ("status", "assigned_crane", "waiting", "block_in_s", "service_s", "done_s", "rehandles")}
    waiting = sim.kpis._waiting
    for jid, j in sim.jobs.items():
        n = jidx[jid]
        o["status"][n] = STATUS_NAMES.index(j.status.name)
        o["assigned_crane"][n] = cidx[j.assigned_crane] if j.assigned_crane is not None else EMPTY_ID
        o["waiting"][n] = jid in waiting
        if jid in waiting:
            o["block_in_s"][n] = waiting[jid]
        elif j.is_external_truck and j.service_start is not None:
            o["block_in_s"][n] = float(j.actual_block_arrival)
        o["service_s"][n] = EMPTY_TIME if j.service_start is None else j.service_start
        o["done_s"][n] = EMPTY_TIME if j.service_end is None else j.service_end
        o["rehandles"][n] = j.rehandle_count
    orders = w0.orders._replace(**{f: jnp.asarray(v) for f, v in o.items()})

    # 크레인 (cranes.py:14-33 · models.py:85-96)
    cr = {f: np.asarray(getattr(w0.cranes, f)).copy()
          for f in ("bay", "row", "assigned", "status", "available_at", "down", "down_pending", "yielded",
                    "served", "completions", "is_loaded", "loaded_m", "empty_m")}
    for cid in sim.fleet.ids():
        k, yc = cidx[cid], sim.fleet.get(cid)
        st = yc.state
        cr["bay"][k], cr["row"][k] = st.position_bay, st.trolley_row
        cr["assigned"][k] = jidx[st.assigned_job] if st.assigned_job is not None else EMPTY_ID
        cr["status"][k] = CRANE_STATUS_NAMES.index(st.status.name)
        cr["available_at"][k] = st.available_at
        cr["down"][k], cr["down_pending"][k], cr["yielded"][k] = yc.down, yc.down_pending, yc.yielded
        cr["served"][k], cr["completions"][k] = yc.served_count, yc.recent_completions
        cr["is_loaded"][k] = yc.is_loaded
        cr["loaded_m"][k], cr["empty_m"][k] = st.loaded_travel_m, st.empty_travel_m
    cranes = w0.cranes._replace(**{f: jnp.asarray(v) for f, v in cr.items()})

    # 격자·컨테이너 — 반입 예비칸(아직 안 놓인 것)의 규격은 w0 에서
    stacks, conts, _ = from_v5_stacks(sim.stacks, g, cont_ids=list(tb.cont_ids), n_cont=len(tb.cont_ids))
    conts = conts._replace(c_size=jnp.where(conts.c_alive, conts.c_size, w0.conts.c_size))

    # 예약표 (reservation.py:25-54) — 비활성 행은 w0 값
    rs = {f: np.asarray(getattr(w0.res, f)).copy() for f in w0.res._fields}
    rt = sim.reservations
    for cid, r in rt._by_crane.items():
        k = cidx[cid]
        rs["active"][k] = True
        rs["token"][k] = jidx[r.job_token] if r.job_token is not None else EMPTY_ID
        rs["lo"][k], rs["hi"][k] = r.corridor.lo, r.corridor.hi
        rs["lane"][k] = lidx[r.lane_id] if r.lane_id is not None else EMPTY_ID
        rs["release_at"][k] = r.release_at
        for (b, rr) in r.slots:
            rs["slots"][k, b - 1, rr - 1] = True
    rs["token_owner"][:] = EMPTY_ID
    for tok, cid in rt._tokens.items():
        rs["token_owner"][jidx[tok]] = cidx[cid]
    rs["idle_pos"][:] = EMPTY_TIME
    for cid, p in rt._idle_pos.items():
        rs["idle_pos"][cidx[cid]] = p
    res = w0.res._replace(**{f: jnp.asarray(v) for f, v in rs.items()})

    # 사건 로그 (engine.py:157) — payload → 번호 (host_convert 머리말의 역방향)
    lg = {f: np.asarray(getattr(w0.log, f)).copy() for f in ("t", "kind", "target")}
    assert len(sim.event_log) <= lg["t"].shape[0], "log_cap 부족"
    for i, (t, kname, payload) in enumerate(sim.event_log):
        lg["t"][i], lg["kind"][i] = t, EV_NAMES.index(kname)
        if kname == "DISPATCH":
            lg["target"][i] = cidx[payload.split(":")[0]]
        elif kname == "DEADLOCK_ESCAPE":
            lg["target"][i] = sum(1 << cidx[c] for c in payload.split(",") if c)
        elif kname in _LOG_CRANE:
            lg["target"][i] = cidx.get(payload, EMPTY_ID)
        elif kname in _LOG_JOB:
            lg["target"][i] = jidx.get(payload, EMPTY_ID)
        else:
            lg["target"][i] = EMPTY_ID
    log = w0.log._replace(t=jnp.asarray(lg["t"]), kind=jnp.asarray(lg["kind"]), target=jnp.asarray(lg["target"]),
                          n=jnp.asarray(len(sim.event_log), jnp.int32))

    f64 = lambda v: jnp.asarray(v, jnp.float64)
    return w0._replace(clock=f64(sim.clock), end_s=f64(sim.end),
                       escape_at=f64(_opt(sim._escape_at)), last_decision_at=f64(_opt(sim._last_decision_at)),
                       escape_count=jnp.asarray(sim._escape_count, jnp.int32),
                       orders=orders, cranes=cranes, stacks=stacks, conts=conts, res=res, log=log)


#: 변환기가 채우는 열 — 엔진 궤적과의 대조 범위
_WRITTEN = {
    "orders": ("status", "assigned_crane", "waiting", "block_in_s", "service_s", "done_s", "rehandles"),
    "cranes": ("bay", "row", "assigned", "status", "available_at", "down", "down_pending", "yielded",
               "served", "completions", "is_loaded", "loaded_m", "empty_m"),
    "stacks": ("grid", "height", "top_size"),
    "conts": ("c_bay", "c_row", "c_tier", "c_size", "c_avail", "c_alive"),
    "res": ("active", "token", "lo", "hi", "lane", "release_at", "slots", "token_owner", "idle_pos"),
    "log": ("t", "kind", "target", "n"),
}


def diff_written(a, b) -> list[str]:
    """변환기가 채우는 열에서 두 세계가 다른 잎 이름."""
    bad = []
    for name in ("clock", "end_s", "escape_at", "last_decision_at", "escape_count"):
        if not np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), equal_nan=True):
            bad.append(name)
    for grp, fields in _WRITTEN.items():
        ga, gb = getattr(a, grp), getattr(b, grp)
        for f in fields:
            if not np.array_equal(np.asarray(getattr(ga, f)), np.asarray(getattr(gb, f)), equal_nan=True):
                bad.append(f"{grp}.{f}")
    return bad


# ───────────────────────────────────────────────── v5 쪽 — 가로채기 구동
class Rec(NamedTuple):
    """`_try_escape` 한 호출의 기록."""

    pre: object            # 호출 직전 배열 세계 (world_from_sim)
    cors: tuple            # v5 interference_deadlock_corridors()
    disp: np.ndarray       # (K,n0) bool  _dispatchable
    feas: np.ndarray       # (K,n0) bool  … & ~taken & jobref & plan
    code: np.ndarray       # (K,n0) int32 reject_reason 코드 (feas 아니면 -1)
    cands: dict            # crane_id → [job_id] (candidates_for)
    n_events: int          # 호출 전까지 처리한 큐 사건 수 (로그 중 큐 종류)
    n_decisions: int       # 호출 전까지 열린 결정 수 (탈출 포함)
    post: dict             # 호출 뒤: fired·crane_ids·escape_at·last_decision_at·escape_count·yielded·log


def v5_pair_table(sim, tb):
    """크레인×오더 전 쌍을 v5 함수로 — candidates_for(477-490행)·interference_deadlock_corridors(437-450행) 의 단계."""
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


def run_v5_hooked(prof, scn, chooser, w0, tb, g):
    """v5 완주 (순차 live 재선택) — `_try_escape` 호출마다 Rec 을 남긴다. 반환 (sim, recs)."""
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    recs: list[Rec] = []
    orig = sim._try_escape
    n_dec = 0

    def hooked():
        pre = world_from_sim(sim, w0, tb, g)
        cors = sim.interference_deadlock_corridors()
        disp, feas, code = v5_pair_table(sim, tb)
        cands = {cid: [r.job_id for r in sim.candidates_for(cid)] for cid in sim.fleet.ids()}
        n_ev = sum(1 for (_, k, _) in sim.event_log if k in EV_NAMES[:12])
        out = orig()
        post = dict(fired=out is not None, crane_ids=() if out is None else tuple(out.crane_ids),
                    escape_at=sim._escape_at, last_decision_at=sim._last_decision_at,
                    escape_count=sim._escape_count,
                    yielded={cid: sim.fleet.get(cid).yielded for cid in sim.fleet.ids()},
                    log=list(sim.event_log))
        recs.append(Rec(pre, cors, disp, feas, code, cands, n_ev, n_dec, post))
        return out

    sim._try_escape = hooked
    while (dp := sim.run_until_decision()) is not None:
        n_dec += 1
        for cid in dp.crane_ids:                       # dispatcher.py:26-32 — 앞 크레인 배정 반영
            ref = chooser(sim, cid, sim.candidates_for(cid))
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=ref) if ref is not None
                       else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    return sim, recs


# ───────────────────────────────────────────────── 배열 쪽 — 대조
cand_jit = jax.jit(candidate_matrices, static_argnames=("g",))
escape_jit = jax.jit(try_escape)


def _norm_log(log, tb):
    """조각 1 compare ① 과 같은 정규화 — 없는 크레인/선박을 겨냥한 payload 는 '' 로."""
    out = []
    for (t, k, p) in log:
        if k in ("EQUIPMENT_DOWN", "EQUIPMENT_UP") and p not in tb.crane_ids:
            p = ""
        if k == "PLAN_CHANGE" and p not in tb.vessel_ids:
            p = ""
        out.append((round(t, 6), k, p))
    return out


def check_record(label, i, rec: Rec, tb, g) -> dict:
    """호출 하나를 v5 기록과 대조. 반환 = 집계용 표식."""
    K, n0 = len(tb.crane_ids), tb.n0
    pre = rec.pre
    m = cand_jit(pre, g)
    e = escape_jit(pre, m)
    tag = f"[{label} #{i} t={float(pre.clock):.3f}]"

    # ① 후보 구조 — (K,N) 전 쌍
    disp = np.asarray(m.disp)[:, :n0]
    feas = np.asarray(m.feasible)[:, :n0]
    code = np.asarray(m.code)[:, :n0]
    bad = []
    if not np.array_equal(disp, rec.disp):
        bad.append(f"dispatchable 다름 {np.argwhere(disp != rec.disp).tolist()}")
    if not np.array_equal(feas, rec.feas):
        bad.append(f"feasible(계획 성립) 다름 {np.argwhere(feas != rec.feas).tolist()}")
    both = feas & rec.feas
    if not np.array_equal(np.where(both, code, -1), np.where(both, rec.code, -1)):
        pairs = [(k, n, int(code[k, n]), int(rec.code[k, n])) for k, n in np.argwhere(both & (code != rec.code))]
        bad.append(f"reject 코드 다름 (k,n,arr,v5)={pairs}")
    cand = np.asarray(m.cand)
    for k, cid in enumerate(tb.crane_ids):
        got = [tb.job_ids[n] for n in np.nonzero(cand[k])[0]]
        if got != rec.cands[cid]:
            bad.append(f"candidates_for({cid}) arr={got} v5={rec.cands[cid]}")
    assert not bad, f"{tag} ① 후보 구조가 갈린다:\n  " + "\n  ".join(bad)
    assert np.array_equal(np.asarray(m.idle), np.asarray(idle_mask(pre.cranes)))

    # ② 술어 · 통로
    dl = bool(e.deadlock)
    assert dl == bool(rec.cors), f"{tag} ② 술어 arr={dl} v5={rec.cors}"
    if dl:
        lo, hi, has = blocked_corridors(m, e.blocked)
        got = tuple(sorted({(float(lo[n]), float(hi[n])) for n in range(n0) if bool(has[n])}))
        assert got == rec.cors, f"{tag} ② 통로 arr={got} v5={rec.cors}"
    # ③ 발화 · 결과
    post = rec.post
    assert bool(e.fired) == post["fired"], f"{tag} ③ fired arr={bool(e.fired)} v5={post['fired']} (deadlock={dl})"
    open_ids = tuple(tb.crane_ids[k] for k in range(K) if bool(e.open[k]))
    assert open_ids == post["crane_ids"], f"{tag} ③ 결정 대상 arr={open_ids} v5={post['crane_ids']}"
    w2 = e.world
    pend = tuple(tb.crane_ids[k] for k in range(K) if bool(w2.decision.pending[k]))
    assert pend == post["crane_ids"], f"{tag} ③ pending arr={pend} v5={post['crane_ids']}"
    assert not bool(jnp.any(w2.decision.answered)) or not post["fired"]
    assert float(w2.escape_at) == _opt(post["escape_at"]), f"{tag} ③ escape_at arr={float(w2.escape_at)} v5={post['escape_at']}"
    assert float(w2.last_decision_at) == _opt(post["last_decision_at"]), \
        f"{tag} ③ last_decision_at arr={float(w2.last_decision_at)} v5={post['last_decision_at']}"
    assert int(w2.escape_count) == post["escape_count"], f"{tag} ③ escape_count arr={int(w2.escape_count)} v5={post['escape_count']}"
    yl = {cid: bool(w2.cranes.yielded[k]) for k, cid in enumerate(tb.crane_ids)}
    assert yl == post["yielded"], f"{tag} ③ yielded arr={yl} v5={post['yielded']}"
    got_log = _norm_log(event_log_from_arrays(w2, tb), tb)
    exp_log = _norm_log(post["log"], tb)
    assert got_log == exp_log, f"{tag} ③ 로그 (arr {len(got_log)}건 · v5 {len(exp_log)}건)\n  arr 끝={got_log[-3:]}\n  v5 끝={exp_log[-3:]}"
    if post["fired"]:
        assert int(w2.log.kind[int(w2.log.n) - 1]) == LOG_DEADLOCK_ESCAPE
        assert int(w2.log.target[int(w2.log.n) - 1]) == int(crane_bits(e.open))
        assert int(w2.overflow) == 0
    else:
        # 미발화면 yielded 해제 외에는 세계가 그대로다 (398행이 400행 앞이라 go 면 해제만)
        assert diff_written(w2, pre) in ([], ["cranes.yielded"])

    clock = float(pre.clock)
    return dict(
        deadlock=dl, fired=post["fired"],
        escat_blocked=dl and float(pre.escape_at) == clock,
        mark_blocked=dl and float(pre.escape_at) != clock and clock <= float(pre.last_decision_at) + EPS,
        regained=post["fired"] and bool(jnp.any(e.cand)),
        go_empty=dl and float(pre.escape_at) != clock and not (clock <= float(pre.last_decision_at) + EPS)
                 and not post["fired"] and clock < float(pre.end_s) - EPS,
        yield_cleared=dl and not post["fired"] and any(np.asarray(pre.cranes.yielded)) and not any(post["yielded"].values()),
        open_partial=post["fired"] and len(post["crane_ids"]) < K,
        n_blocked=int(np.asarray(e.blocked).sum()), n_cand=int(cand.sum()),
        any_code4=bool((code == CRANE_INTERFERENCE).any()),
    )


def run_and_check(label, prof, scn, chooser):
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    sim, recs = run_v5_hooked(prof, scn, chooser, w0, tb, g)
    assert recs, f"[{label}] _try_escape 가 한 번도 안 불렸다"
    flags = [check_record(label, i, r, tb, g) for i, r in enumerate(recs)]
    n_fired = sum(f["fired"] for f in flags)
    assert n_fired == sim.deadlock_escape_count == sum(1 for (_, k, _) in sim.event_log if k == "DEADLOCK_ESCAPE")
    for i, r in enumerate(recs):
        if r.post["fired"] and label not in FIRED_WORLDS:
            FIRED_WORLDS[label] = (r.pre, tb, g, r)
    REPORT[label] = dict(
        K=len(tb.crane_ids), calls=len(recs), deadlock=sum(f["deadlock"] for f in flags), fired=n_fired,
        escat_blocked=sum(f["escat_blocked"] for f in flags), mark_blocked=sum(f["mark_blocked"] for f in flags),
        go_empty=sum(f["go_empty"] for f in flags), yield_cleared=sum(f["yield_cleared"] for f in flags),
        regained=sum(f["regained"] for f in flags), open_partial=sum(f["open_partial"] for f in flags),
        code4=sum(f["any_code4"] for f in flags), decisions=max(r.n_decisions for r in recs),
        backlog=sim.unfinished_backlog(), events=len(sim.event_log))
    return sim, recs, tb, g, w0


# ───────────────────────────────────────────────── 무대 목록
def _k2_stage(gap, seed):
    return lambda: (prof_k2(gap), random_scenario(seed), chooser_ref)


STAGES: dict[str, object] = {}
for _gap in (3.0, 4.0):
    for _seed in range(1, 22):          # 시드 21 = 탐색에서 발화가 가장 많던 무대 (gap3 8회 · gap4 9회)
        STAGES[f"k2-gap{int(_gap)}-s{_seed}"] = _k2_stage(_gap, _seed)
for _seed in range(1, 13):
    STAGES[f"k2-gap2-s{_seed}"] = _k2_stage(2.0, _seed)
STAGES["dead-first"] = lambda: (prof_k2(3.0), dead_first_scenario(), chooser_ref)
STAGES["down-both"] = lambda: (prof_k2(3.0), down_scenario(("YC-A", "YC-B")), chooser_ref)
STAGES["down-one"] = lambda: (prof_k2(3.0), down_scenario(("YC-B",)), chooser_ref)
# 항상 WAIT: 600초 결정(둘 다 후보 있음)에서 WAIT → yielded → 같은 시각 _try_escape 가 last_decision_at 표식(394행)에 막힌다
STAGES["dead-first-waitall"] = lambda: (prof_k2(3.0), dead_first_scenario(), chooser_wait_until(np.inf))
# 600초까지 WAIT → 700초 EQUIPMENT_DOWN(YC-B) 은 yielded 를 안 지운다 → 그 시각 탈출이 발화하고 YC-A 가 (해제 뒤) 후보를 되찾는다
STAGES["regain"] = lambda: (prof_k2(3.0), replace(dead_first_scenario(
    (InjectedEvent(700.0, "EQUIPMENT_DOWN", "YC-B"), InjectedEvent(1000.0, "EQUIPMENT_UP", "YC-B"))),
    scenario_id="regain"), chooser_wait_until(600.0))
# 조각 1 K=1 무대 (술어 거짓 — 크레인 하나면 간섭이 없다)
STAGES["k1-spec10"] = lambda: (piece1_profile(), piece1_scenario(), chooser_first)
STAGES["k1-crowded"] = lambda: (piece1_profile(), _EQ.crowded_scenario(), chooser_first)
STAGES["k1-censored"] = lambda: (piece1_profile(), _EQ.crowded_scenario(2000.0), chooser_first)
for _seed in range(1, 7):
    STAGES[f"k1-random-{_seed}"] = (lambda sd=_seed: (piece1_profile(), random_scenario(sd), chooser_first))
STAGES["k1-wait2-crowded"] = lambda: (piece1_profile(), _EQ.crowded_scenario(), chooser_wait2)
STAGES["k1-same-time"] = lambda: (piece1_profile(), _EQ.same_time_scenario(), chooser_first)
STAGES["k1-dup-5..5"] = lambda: (_EQ._narrow_profile(5, 5), _EQ.dup_range_cap_scenario(), chooser_first)
STAGES["k1-dup-4..6"] = lambda: (_EQ._narrow_profile(4, 6), _EQ.dup_range_cap_scenario(), chooser_first)
STAGES["k1-edge-end"] = lambda: (piece1_profile(), _EQ.edge_end_scenario(), chooser_first)
STAGES["k1-edge-drain"] = lambda: (piece1_profile(), _EQ.edge_drain_scenario(), chooser_first)
STAGES["k1-turntime"] = lambda: (piece1_profile(), _EQ.turntime_edge_scenario(), chooser_first)
STAGES["k1-vl-deadline"] = lambda: (piece1_profile(), _EQ.vessel_deadline_scenario(), chooser_first)
STAGES["k1-fixture-ignored"] = lambda: (_EQ.fixture_profile_k1(), _EQ.fixture_scenario_k1("YC-B"), chooser_first)
STAGES["k1-fixture-idle-down"] = lambda: (_EQ.fixture_profile_k1(), _EQ.fixture_scenario_k1("YC-A"), chooser_first)
STAGES["k1-fixture-busy-down"] = lambda: (_EQ.fixture_profile_k1(), _EQ.fixture_scenario_k1("YC-A", (310.0, 400.0)), chooser_first)


# ───────────────────────────────────────────────── ① 변환기 — reset 직후 to_block_world 와 같다
@pytest.mark.parametrize("stage", ["dead-first", "k2-gap3-s21", "k1-fixture-ignored"])
def test_converter_matches_reset(stage):
    prof, scn, _ = STAGES[stage]()
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, **caps)
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    w = world_from_sim(sim, w0, tb, Geom.from_profile(prof))
    assert diff_written(w, w0) == [], diff_written(w, w0)


# ───────────────────────────────────────────────── ② 모든 무대 · 모든 _try_escape 호출
@pytest.mark.parametrize("stage", list(STAGES), ids=list(STAGES))
def test_escape_equivalence_every_call(stage):
    prof, scn, chooser = STAGES[stage]()
    run_and_check(stage, prof, scn, chooser)


# ───────────────────────────────────────────────── ③ 설계 무대의 성질 (v5 가 실제로 그 경로를 밟았는지)
def test_dead_first_fires_before_any_decision_and_matches_engine_trajectory():
    """결정 0회 뒤 첫 탈출: 조각 1 엔진(`ES.step`)이 사건만 처리해 도달한 세계 == 변환기 세계, 그리고 그 세계에
    try_escape 를 적용하면 v5 와 같은 결정(둘 다 유휴 → 둘 다 결정 대상 · 로그 · count 1)."""
    prof, scn, chooser = STAGES["dead-first"]()
    sim, recs, tb, g, w0 = run_and_check("dead-first", prof, scn, chooser)
    first = next(i for i, r in enumerate(recs) if r.post["fired"])
    assert recs[first].n_decisions == 0 and float(recs[first].pre.clock) == 300.0
    w, n_ev = w0, 0
    for i in range(first + 1):
        while n_ev < recs[i].n_events:
            w, tr = ES.step(w, None, params=None, g=g, policy_fn=first_by_id)
            assert not bool(tr.decided) and int(tr.kind) >= 0
            n_ev += 1
        bad = diff_written(recs[i].pre, w)
        assert not bad, f"호출 #{i}: 변환기 세계와 엔진 세계가 다른 잎 {bad}"
    m = candidate_matrices(w, g)
    e = try_escape(w, m)
    assert bool(e.fired)
    assert [bool(x) for x in e.open] == [True, True] and int(e.world.escape_count) == 1
    assert event_log_from_arrays(e.world, tb)[-1] == recs[first].post["log"][-1] == (300.0, "DEADLOCK_ESCAPE", "YC-A,YC-B")
    assert not bool(jnp.any(e.cand))                       # SERVE 후보는 없다 — 정책은 WAIT 만 답할 수 있다
    assert sim.deadlock_escape_count >= 1 and sim.unfinished_backlog() == 0   # 600초 뒤 풀린다


def _ensure(label: str) -> dict:
    """무대를 아직 안 돌렸으면 돌리고 집계를 돌려준다 (시험 단독 실행 대비)."""
    if label not in REPORT:
        run_and_check(label, *STAGES[label]())
    return REPORT[label]


def test_down_both_predicate_true_but_no_escape():
    """두 크레인 고장: ①②③ 참(③ 은 고장 크레인 쌍) 인데 esc 가 비어 미발화 — yielded 해제만 (398행)."""
    r = _ensure("down-both")
    assert r["deadlock"] >= 1 and r["fired"] == 0 and r["go_empty"] >= 1, r


def test_down_one_opens_only_the_idle_crane():
    r = _ensure("down-one")
    assert r["fired"] >= 1 and r["open_partial"] >= 1, r
    _, tb, _, rec = FIRED_WORLDS["down-one"]
    assert rec.post["crane_ids"] == ("YC-A",) and rec.post["log"][-1][2] == "YC-A"


def test_wait_all_blocked_by_last_decision_mark():
    """정상 결정에서 전원 WAIT 한 같은 시각의 _try_escape: 술어는 참인데 394행 표식으로 미발화 — yielded 도 그대로."""
    r = _ensure("dead-first-waitall")
    assert r["mark_blocked"] >= 1 and r["yield_cleared"] == 0, r


def test_regain_candidates_after_yield_clear():
    """EQUIPMENT_DOWN 은 yielded 를 안 지우므로(865-867행) 양보 중 크레인이 남은 채 탈출이 발화하고, 해제 뒤 후보를 되찾는다
    (② 는 해제 전 값 · EscapeOut.cand 는 해제 뒤 값)."""
    _, recs, _, _, _ = run_and_check("regain", *STAGES["regain"]())
    r = REPORT["regain"]
    assert r["regained"] >= 1 and r["fired"] >= 1, r
    # 300초 탈출(둘 다 유휴·양보 없음) 뒤, 700초 고장 사건 시각의 탈출: 양보 중이던 YC-A 만 결정 대상
    rec = next(x for x in recs if x.post["fired"] and any(np.asarray(x.pre.cranes.yielded)))
    assert float(rec.pre.clock) == 700.0 and rec.post["crane_ids"] == ("YC-A",)
    assert rec.cands["YC-A"] == []                        # 술어 ② 는 해제 **전** — 양보 중이라 후보 없음
    m = candidate_matrices(rec.pre, Geom.from_profile(prof_k2(3.0)))
    e = try_escape(rec.pre, m)
    assert bool(jnp.any(e.cand[0])) and not bool(jnp.any(e.cand[1]))   # 해제 뒤 YC-A 가 후보를 되찾는다


# ───────────────────────────────────────────────── ④ jit · vmap 이 답을 바꾸지 않는다
def test_jit_and_vmap_agree_on_fired_worlds():
    for lab in ("dead-first", "down-one"):
        _ensure(lab)
    picks = [FIRED_WORLDS[lab] for lab in ("dead-first", "down-one")]
    g = picks[0][2]
    worlds = [p[0] for p in picks]
    assert len({tuple(l.shape for l in jax.tree_util.tree_leaves(w)) for w in worlds}) == 1
    outs_eager = [try_escape(w, candidate_matrices(w, g)) for w in worlds]
    outs_jit = [escape_jit(w, cand_jit(w, g)) for w in worlds]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    run_b = jax.jit(jax.vmap(lambda w: try_escape(w, candidate_matrices(w, g))))
    out_b = run_b(batched)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(outs_eager[0])]
    for i in range(len(worlds)):
        le = [np.asarray(l) for l in jax.tree_util.tree_leaves(outs_eager[i])]
        lj = [np.asarray(l) for l in jax.tree_util.tree_leaves(outs_jit[i])]
        lb = [np.asarray(l)[i] for l in jax.tree_util.tree_leaves(out_b)]
        bad_j = [names[j] for j, (a, b) in enumerate(zip(le, lj)) if not np.array_equal(a, b, equal_nan=True)]
        bad_b = [names[j] for j, (a, b) in enumerate(zip(le, lb)) if not np.array_equal(a, b, equal_nan=True)]
        assert not bad_j and not bad_b, f"세계 {i}: jit 다름 {bad_j} · vmap 다름 {bad_b}"
        assert bool(outs_eager[i].fired)


# ───────────────────────────────────────────────── ⑤ 보고 — 무엇이 실제로 시험됐는지
def test_zz_report(capsys):
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    hits = {k: r for k, r in REPORT.items() if r["fired"] > 0}
    false_stages = {k: r for k, r in REPORT.items() if r["deadlock"] == 0}
    tot = lambda key: sum(r[key] for r in REPORT.values())
    with capsys.disabled():
        print("\n[escape equiv report]  calls = _try_escape 호출 수 · deadlock = 술어 참 · fired = 발화 · "
              "escat/mark = 같은시각/결정표식으로 막힘 · go_empty = 술어 참인데 유휴 없음 · regained = 발화 후 후보 되찾음")
        for k, r in REPORT.items():
            print(f"  {k:24s} K={r['K']} calls={r['calls']:3d} deadlock={r['deadlock']:2d} fired={r['fired']:2d} "
                  f"escat={r['escat_blocked']:2d} mark={r['mark_blocked']:2d} go_empty={r['go_empty']} "
                  f"yclr={r['yield_cleared']} regained={r['regained']} partial={r['open_partial']} "
                  f"code4={r['code4']:2d} dec={r['decisions']:3d} backlog={r['backlog']}")
        print(f"  stages={len(REPORT)} · escape>0 stages={len(hits)} ({sorted(hits)}) · predicate-false stages={len(false_stages)}"
              f" · calls={tot('calls')} · fired={tot('fired')} · escat_blocked={tot('escat_blocked')} · mark_blocked={tot('mark_blocked')}"
              f" · go_empty={tot('go_empty')} · regained={tot('regained')} · open_partial={tot('open_partial')}")
    assert len(hits) >= 5, "v5 가 DEADLOCK_ESCAPE 를 내는 무대가 5개도 안 됐다"
    assert len(false_stages) >= 20, f"술어 거짓 무대가 20개 미만 ({len(false_stages)})"
    assert tot("escat_blocked") >= 1, "같은 시각 재발화 방지(escape_at == clock)가 한 번도 시험되지 않았다"
    assert tot("go_empty") >= 1, "술어 참·esc 비어 미발화 경로가 시험되지 않았다"
    assert tot("open_partial") >= 1, "유휴 일부만 결정 대상인 경우가 시험되지 않았다"
    assert tot("mark_blocked") >= 1, "last_decision_at 표식(394행)으로 막히는 호출이 시험되지 않았다"
    assert tot("regained") >= 1, "yielded 해제 뒤 후보를 되찾는 탈출이 시험되지 않았다"
    assert all(r["deadlock"] == 0 for k, r in REPORT.items() if r["K"] == 1)
