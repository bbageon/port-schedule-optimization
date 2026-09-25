"""깨우기(gpu/wake.py) 가 v5 `_consume_due_wakes`·`_next_wake_time`·`schedule_defer_wake`·`_decision_cranes` 의 armed 항·
`eta_opportunity` 와 **같은 답**을 내는가 ([[YR-327]] 조각 3 · key=wake · engine.py:159-179, 339-382, 453-464).

■ 방법 — v5 를 PRE_ADVICE 로 실제로 굴리며 `_consume_due_wakes` 가 불리는 **모든 순간**을 가로챈다 (기대값 손기입 없음)
  hook(앞): 그 순간의 v5 상태 → 배열 세계 (test_gpu_escape.world_from_sim + wake 열) · `_next_wake_time()`
  hook(뒤): 원래 함수의 반환(fired) · wake_idx · armed · defer 목록 · yielded · 로그 · `_next_wake_time()` ·
            `_decision_cranes()` · 크레인별 `eta_opportunity` · `candidates_for`
  배열   : `consume_due_wakes` → fired · wake_idx · armed · defer · yielded · **로그 전열** 대조 ·
           `next_wake_time` 앞/뒤 · `next_wake_in_window` · `eta_opportunity_mask` ·
           `open_with_armed(candidate_matrices.cand, …)` == `_decision_cranes()`
  + `seed_wakes` == v5 `_eta_wakes` == host_convert 시드 (넘침은 dropped 로 보고)
  + `schedule_defer_wake` 무작위 열(과거·창 밖·중복·동시각) == v5 insort, 그 뒤 임의 시각 소비 == v5
  + jit / vmap 이 답을 바꾸지 않는다 (발화 순간 세계들)
  + 낮은 정보수준(GATE_IN)에서 ETA wake 완전 비활성 (v5 `_next_wake_time() is None`) — pre_advice=False 가 그것
  + 전원 WAIT 정책 — 결정 유한 · 시각 엄격 증가 (test_yr050:147-161 규약) 그 사이 모든 소비 순간이 일치

■ 무대
  yr050 무대 (fixtures 프로파일 — K=2 · 1..40 · 지평 1800): blocked-target(ETA 900·도착 1500 → wake 0) ·
    busy-at-wake(wake 50 이 서비스 중 → armed 잔존, 유휴화 시점 개방) · no-eta · neg-gap(ETA 60·도착 900 → 음수 gap 기회) ·
    gate-in-eta(GATE_IN 의 ETA 는 시드 안 됨) · defer(WAIT 마다 now+600 예약 = baselines DEFER_ALL 만료) · low-info(GATE_IN)
  조각 1 §10 무대 (K=1 · 10×4×4) 에 provided_eta 부여: eta=도착 (전부 wake 0 → 한 번에 3건) · eta=도착+1800+k (도착 뒤 wake —
    PLANNED 아님 → 기회 없음) · crowded (12건, eta=도착+1500)
  조각 2 무대 (prof_k2 gap 3 · K=2): 무작위 시드 4개, eta = 도착 − 무작위 (정수 초)
  v5 정책: CentralResolver(BaselinePreference)+CandidateGenerator (PRE_REHANDLE·REPOSITION 실행, test_yr050 `_drive`) ·
           ReferenceDispatcher 규칙 (SERVE/WAIT) · 전원 WAIT · DEFER 예약 변형

■ 시험 전용 규약 — REPOSITION 중인 크레인
  v5 는 `assigned_job = "REPO:<crane>:<bay>"` 로 바쁨을 표시한다. 오더가 아니라 배열 `cranes.assigned` 에 번호가 없으므로
  변환기가 **N(오더 칸 수) 을 '오더 없는 바쁨' 표식**으로 넣는다 (assigned ≥ 0 → 유휴 아님, 어떤 오더 표도 그 번호로
  읽지 않는다). 통합판의 REPOSITION 표현은 dispatch/candidates 담당이 정한다 (open_issues).

실행: WSL venv · x64 CPU.  PYTHONPATH=src JAX_PLATFORMS=cpu pytest tests/v6/test_gpu_wake.py -q -s
"""
from __future__ import annotations

import importlib.util
import os
import random
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)   # ★float64 — engine_step.py 머리말
jnp = jax.numpy

from yard_rl.v6.gpu import wake as WK                                              # noqa: E402
from yard_rl.v6.gpu.escape import candidate_matrices                                # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import event_log_from_arrays, to_block_world       # noqa: E402
from yard_rl.v6.gpu.state import EMPTY_ID, EMPTY_TIME, LOG_DEFER_WAKE, LOG_ETA_WAKE   # noqa: E402
# v5 정본
from yard_rl.v6.world.contract.schema import CandidateKind                          # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, InformationLevel, JobFlow, LoadStatus   # noqa: E402
from yard_rl.v6.world.domain.models import Container, Job                          # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                    # noqa: E402
from yard_rl.v6.world.integrated.candidates import CandidateGenerator, eta_opportunity   # noqa: E402
from yard_rl.v6.world.integrated.dispatcher import ReferenceDispatcher              # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator   # noqa: E402
from yard_rl.v6.world.integrated.resolver import BaselinePreference, CentralResolver   # noqa: E402
from yard_rl.v6.world.integrated.scenario import TerminalScenario                   # noqa: E402

EPS = WK.EPS
PA, GI = InformationLevel.PRE_ADVICE, InformationLevel.GATE_IN
#: 시험 전체 집계 (마지막 시험이 보고)
REPORT: dict[str, dict] = {}
#: 발화 순간의 배열 세계 (jit/vmap 시험이 재사용) — label → [(pre_world, pre_advice)]
FIRED_WORLDS: dict[str, list] = {}
#: DEFER 칸 수 — 무대 전부 같은 값 (같은 가족은 모양이 같아야 vmap 으로 쌓인다)
N_DEFER = 8


def _load(name: str):
    """tests/v6/<name>.py 를 경로로 불러온다 (tests/ 에 __init__ 이 없다)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_{name}_for_wake", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ESC = _load("test_gpu_escape")             # world_from_sim · prof_k2 · (그 안에서) engine_equiv 무대
_EQ = _ESC._EQ
world_from_sim, prof_k2 = _ESC.world_from_sim, _ESC.prof_k2
_caps, random_scenario = _EQ._caps, _EQ.random_scenario
piece1_profile, piece1_scenario, crowded_scenario = _EQ.piece1_profile, _EQ.piece1_scenario, _EQ.crowded_scenario


# ───────────────────────────────────────────────── 무대 빌더
PROF = fixtures.build_integrated_profile()     # test_yr050:26 — 결정 지평 1800 · YC 2기 (1..40)


def _c(cid, bay, row, tier):
    return Container(container_id=cid, size=ContainerSize.FT40, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _out(jid, target, arrival, eta):
    """test_yr050:35-38 — 장부 없음(exit_travel None)."""
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0,
               actual_gate_in=max(0.0, arrival - 600.0), actual_block_arrival=arrival,
               provided_eta=eta, target_container=target)


def _sc(sid, jobs, containers):
    return TerminalScenario(scenario_id=sid, seed=0, horizon_s=7200.0, drain_window_s=1800.0,
                            containers=containers, jobs=jobs, vessels=[], injected_events=[])


def blocked_target_sc(eta, arrival, sid="blocked"):
    """반출 대상 C-T(bay5) 위에 blocker C-B → ETA 만 보이면 선제 재조작 기회 (test_yr050:46-49)."""
    return _sc(sid, [_out("J-OUT-T", "C-T", arrival, eta)],
               {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2)})


def busy_at_wake_sc():
    """test_yr050:81-92 — wake(50) 가 서비스 중 · 바쁜 크레인의 armed 가 유휴화 시점까지 남는다."""
    containers = {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2), "C-X": _c("C-X", 3, 2, 1)}
    return _sc("busy-at-wake", [_out("J-BUSY-X", "C-X", 0.0, None), _out("J-OUT-T", "C-T", 2100.0, 1850.0)], containers)


def neg_gap_sc():
    """test_yr050:183-191 — ETA 60 이 지난 뒤(음수 gap) 에도 선제 기회가 열린다."""
    containers = {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2), "C-D": _c("C-D", 20, 2, 1),
                  "C-X": _c("C-X", 8, 2, 1), "C-Y": _c("C-Y", 35, 3, 1)}
    jobs = [_out("J-BUSY-X", "C-X", 0.0, None), _out("J-BUSY-Y", "C-Y", 0.0, None),
            _out("J-DECOY", "C-D", 250.0, 250.0), _out("J-OUT-T", "C-T", 900.0, 60.0)]
    return _sc("neg-gap", jobs, containers)


def gate_in_eta_sc():
    """test_yr050:136-142 — GATE_IN 의 provided_eta 는 wake 로 시드되지 않는다."""
    gate_in = Job(job_id="J-IN-X", flow=JobFlow.GATE_IN, release_time=0.0, actual_gate_in=100.0,
                  actual_block_arrival=700.0, provided_eta=650.0, inbound_size=ContainerSize.FT40,
                  inbound_load=LoadStatus.FULL)
    return _sc("gate-in-eta", [_out("J-OUT-T", "C-T", 3000.0, 2500.0), gate_in],
               {"C-T": _c("C-T", 5, 1, 1), "C-B": _c("C-B", 5, 1, 2)})


def with_eta(scn: TerminalScenario, offset, sid: str, *, rng: random.Random | None = None) -> TerminalScenario:
    """외부트럭마다 provided_eta = 도착 + offset (offset 이 callable 이면 offset(job, rng))."""
    jobs = []
    for j in scn.jobs:
        if j.is_external_truck and j.actual_block_arrival is not None:
            off = offset(j, rng) if callable(offset) else offset
            jobs.append(replace(j, provided_eta=float(j.actual_block_arrival) + off))
        else:
            jobs.append(j)
    return replace(scn, scenario_id=sid, jobs=jobs)


# ───────────────────────────────────────────────── v5 구동 (정책)
_REF = ReferenceDispatcher()
_GEN = CandidateGenerator()


def drive_resolver(sim):
    """test_yr050:56-67 `_drive` — CentralResolver(BaselinePreference) · PRE_REHANDLE/REPOSITION 이 실제로 실행된다."""
    r = CentralResolver(BaselinePreference())
    while (dp := sim.run_until_decision()) is not None:
        gen_by = {c: _GEN.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        resn = r.resolve(sim, dp, gen_by)
        r.apply(sim, resn, gen_by)


def drive_ref(sim):
    """ReferenceDispatcher.run 의미 — 크레인 순 live 후보 · SERVE/WAIT 만 (dispatcher.py:19-32)."""
    while (dp := sim.run_until_decision()) is not None:
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            ref = _REF.select(sim, cid, cands) if cands else None
            sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=ref) if ref is not None
                       else CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()


def drive_waitall(sim):
    """test_yr050:147-161 — 전원 WAIT. 결정이 유한하고 시각이 엄격히 증가해야 한다 (wake 1회성)."""
    n, prev = 0, None
    while (dp := sim.run_until_decision()) is not None:
        n += 1
        assert n < 50, "결정 폭주 — 재질문 무한루프 의심"
        assert prev is None or dp.time > prev, f"같은 시각 {dp.time} 재결정 — wake 1회성 위반"
        prev = dp.time
        sim.commit_decisions([CraneAssignment(c, CandidateKind.WAIT) for c in dp.crane_ids])
    assert sim.terminal


def drive_defer(sim):
    """drive_ref + WAIT 마다 `schedule_defer_wake(now+600)` (baselines._apply 165행 DEFER_ALL 만료) · 거부 경로도 찌른다
    (과거 시각 · 창 밖 · 정확히 now — 셋 다 무시돼야 한다)."""
    while (dp := sim.run_until_decision()) is not None:
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            ref = _REF.select(sim, cid, cands) if cands else None
            if ref is None:
                sim.schedule_defer_wake(sim.clock + 600.0)
                sim.schedule_defer_wake(sim.clock)               # 무시 (t ≤ clock+EPS)
                sim.schedule_defer_wake(sim.clock - 100.0)       # 무시
                sim.schedule_defer_wake(sim.end + 10.0)          # 무시 (창 밖)
                sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))
            else:
                sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=ref))
        sim.close_decision()


# ───────────────────────────────────────────────── v5 진행 중 상태 → 배열 세계 (wake 열 포함)
def wake_from_sim(sim, w0, tb):
    """v5 `_eta_wakes`·`_wake_idx`·`_eta_armed`·`_defer_wakes` → WakeArrays (모양은 w0 의 것)."""
    W, D = w0.wake.eta_wake_s.shape[0], w0.wake.defer_wake_s.shape[0]
    jidx = tb.job_index
    assert len(sim._eta_wakes) <= W and len(sim._defer_wakes) <= D, "wake/defer 칸 부족"
    ws = np.full((W,), EMPTY_TIME, np.float64)
    wj = np.full((W,), EMPTY_ID, np.int32)
    for i, (t, jid) in enumerate(sim._eta_wakes):
        ws[i], wj[i] = t, jidx[jid]
    ds = np.full((D,), EMPTY_TIME, np.float64)
    for i, t in enumerate(sim._defer_wakes):
        ds[i] = t
    armed = np.asarray([cid in sim._eta_armed for cid in tb.crane_ids], bool)
    return w0.wake._replace(eta_wake_s=jnp.asarray(ws), eta_wake_job=jnp.asarray(wj),
                            wake_idx=jnp.asarray(sim._wake_idx, jnp.int32), eta_armed=jnp.asarray(armed),
                            defer_wake_s=jnp.asarray(ds), defer_n=jnp.asarray(len(sim._defer_wakes), jnp.int32))


def world_from_sim_pa(sim, w0, tb, g):
    """test_gpu_escape.world_from_sim + wake 열. REPOSITION 중인 크레인(assigned_job 이 오더가 아님)은 머리말 규약대로
    assigned = N 으로 둔다 (변환기가 오더 표에서 못 찾아 KeyError 를 내므로 잠시 None 으로 뺐다가 되돌린다)."""
    jidx = tb.job_index
    repo: dict[str, str] = {}
    for cid in sim.fleet.ids():
        st = sim.fleet.get(cid).state
        if st.assigned_job is not None and st.assigned_job not in jidx:
            repo[cid] = st.assigned_job
            st.assigned_job = None
    try:
        w = world_from_sim(sim, w0, tb, g)
    finally:
        for cid, jid in repo.items():
            sim.fleet.get(cid).state.assigned_job = jid
    if repo:
        cidx = tb.crane_index
        assigned = np.asarray(w.cranes.assigned).copy()
        for cid in repo:
            assigned[cidx[cid]] = w0.n
        w = w._replace(cranes=w.cranes._replace(assigned=jnp.asarray(assigned)))
    return w._replace(wake=wake_from_sim(sim, w0, tb))


def _norm_log(log):
    """DISPATCH 는 크레인 부분만 (PRE/REPO 의 payload 는 오더 표로 되돌릴 수 없다 — 이 시험의 대상이 아니다)."""
    out = []
    for (t, k, p) in log:
        if k == "DISPATCH":
            p = p.split(":")[0]
        out.append((round(t, 6), k, p))
    return out


def _opt(v):
    return EMPTY_TIME if v is None else float(v)


# ───────────────────────────────────────────────── 가로채기 구동
class Rec(NamedTuple):
    """`_consume_due_wakes` 한 호출의 기록."""

    pre: object          # 호출 직전 배열 세계 (wake 열 포함)
    nwt_pre: float       # 호출 직전 v5 _next_wake_time() (None → +inf)
    post: dict           # 호출 뒤 v5 값들
    n_decisions: int


class DeferRec(NamedTuple):
    """`schedule_defer_wake` 한 호출의 기록 — (clock, end, t, 앞 목록, 뒤 목록)."""

    clock: float
    end: float
    t: float
    before: list
    after: list


def run_v5_hooked(prof, scn, level, drive, w0, tb, g):
    """v5 완주 — `_consume_due_wakes`·`schedule_defer_wake` 호출마다 기록. 반환 (sim, recs, defer_recs, kinds)."""
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    recs: list[Rec] = []
    drecs: list[DeferRec] = []
    kinds: dict[str, int] = {}
    orig, orig_sched, orig_assign = sim._consume_due_wakes, sim.schedule_defer_wake, sim.assign
    state = {"n_dec": 0}

    def hooked():
        pre = world_from_sim_pa(sim, w0, tb, g)
        nwt_pre = _opt(sim._next_wake_time())
        fired = orig()
        post = dict(
            fired=bool(fired), wake_idx=sim._wake_idx, armed=tuple(c in sim._eta_armed for c in tb.crane_ids),
            defer=list(sim._defer_wakes), yielded=tuple(sim.fleet.get(c).yielded for c in tb.crane_ids),
            log=list(sim.event_log), nwt_post=_opt(sim._next_wake_time()),
            open=tuple(sim._decision_cranes()),
            eta_opp=tuple(bool(eta_opportunity(sim, c, sim.info_level)) for c in tb.crane_ids),
            cands={c: [r.job_id for r in sim.candidates_for(c)] for c in tb.crane_ids},
            busy=tuple(sim.fleet.get(c).state.assigned_job is not None for c in tb.crane_ids),
            neg_gap=any(j.provided_eta is not None and j.provided_eta <= sim.clock and j.status.name == "PLANNED"
                        and j.flow == JobFlow.GATE_OUT for j in sim.jobs.values()))
        recs.append(Rec(pre, nwt_pre, post, state["n_dec"]))
        return fired

    def hooked_sched(t):
        before = list(sim._defer_wakes)
        orig_sched(t)
        drecs.append(DeferRec(sim.clock, sim.end, float(t), before, list(sim._defer_wakes)))

    def hooked_assign(cid, a):
        kinds[a.action.value] = kinds.get(a.action.value, 0) + 1
        return orig_assign(cid, a)

    sim._consume_due_wakes, sim.schedule_defer_wake, sim.assign = hooked, hooked_sched, hooked_assign
    orig_run = sim.run_until_decision

    def counted_run():
        out = orig_run()
        if out is not None:
            state["n_dec"] += 1
        return out

    sim.run_until_decision = counted_run
    drive(sim)
    return sim, recs, drecs, kinds


# ───────────────────────────────────────────────── 배열 쪽 — 대조
consume_jit = jax.jit(WK.consume_due_wakes, static_argnames=("pre_advice",))
nwt_jit = jax.jit(WK.next_wake_time, static_argnames=("pre_advice",))
nwt_win_jit = jax.jit(WK.next_wake_in_window, static_argnames=("pre_advice",))
opp_jit = jax.jit(WK.eta_opportunity_mask, static_argnames=("g", "pre_advice"))
cand_jit = jax.jit(candidate_matrices, static_argnames=("g",))
sched_jit = jax.jit(WK.schedule_defer_wake)


def _defer_list(w):
    n = int(w.wake.defer_n)
    arr = np.asarray(w.wake.defer_wake_s)
    assert np.all(np.isposinf(arr[n:])), "defer 유효 칸 밖이 +inf 가 아니다"
    assert np.all(np.diff(arr[:n]) >= 0), "defer 칸이 정렬돼 있지 않다"
    return [float(x) for x in arr[:n]]


def check_record(label, i, rec: Rec, tb, g, level, horizon) -> dict:
    """호출 하나를 v5 기록과 대조. 반환 = 집계용 표식."""
    K = len(tb.crane_ids)
    pre, post = rec.pre, rec.post
    pa = level == PA
    tag = f"[{label} #{i} t={float(pre.clock):.3f}]"
    # ① 앞 — 다음 wake 시각
    assert float(nwt_jit(pre, pre_advice=pa)) == rec.nwt_pre, \
        f"{tag} ① next_wake_time(앞) arr={float(nwt_jit(pre, pre_advice=pa))} v5={rec.nwt_pre}"
    # ② 소비
    w2, fired = consume_jit(pre, pre_advice=pa)
    assert bool(fired) == post["fired"], f"{tag} ② fired arr={bool(fired)} v5={post['fired']}"
    assert int(w2.wake.wake_idx) == post["wake_idx"], f"{tag} ② wake_idx arr={int(w2.wake.wake_idx)} v5={post['wake_idx']}"
    armed = tuple(bool(x) for x in np.asarray(w2.wake.eta_armed))
    assert armed == post["armed"], f"{tag} ② armed arr={armed} v5={post['armed']}"
    assert _defer_list(w2) == post["defer"], f"{tag} ② defer arr={_defer_list(w2)} v5={post['defer']}"
    yl = tuple(bool(x) for x in np.asarray(w2.cranes.yielded))
    assert yl == post["yielded"], f"{tag} ② yielded arr={yl} v5={post['yielded']}"
    got_log, exp_log = _norm_log(event_log_from_arrays(w2, tb)), _norm_log(post["log"])
    assert got_log == exp_log, (f"{tag} ② 로그 (arr {len(got_log)}건 · v5 {len(exp_log)}건)\n"
                                f"  arr 끝={got_log[-4:]}\n  v5 끝={exp_log[-4:]}")
    assert int(w2.overflow) == 0
    n_pre = int(pre.log.n)
    new_kinds = [int(k) for k in np.asarray(w2.log.kind)[n_pre:int(w2.log.n)]]
    n_dd, n_ed = new_kinds.count(LOG_DEFER_WAKE), new_kinds.count(LOG_ETA_WAKE)
    assert new_kinds == [LOG_DEFER_WAKE] * n_dd + [LOG_ETA_WAKE] * n_ed        # DEFER 먼저 (369-370 → 375-377행)
    assert (n_dd + n_ed > 0) == post["fired"]
    # ③ 뒤 — 다음 wake 시각 · 창 규칙
    assert float(nwt_jit(w2, pre_advice=pa)) == post["nwt_post"], \
        f"{tag} ③ next_wake_time(뒤) arr={float(nwt_jit(w2, pre_advice=pa))} v5={post['nwt_post']}"
    exp_win = post["nwt_post"] if post["nwt_post"] <= float(pre.end_s) + EPS else EMPTY_TIME   # 306-308행
    assert float(nwt_win_jit(w2, pre_advice=pa)) == exp_win
    # ④ 선제 기회 · 결정 개방 (armed 항)
    opp = opp_jit(w2, g, horizon_s=horizon, pre_advice=pa)
    got_opp = tuple(bool(x) for x in np.asarray(opp))
    assert got_opp == post["eta_opp"], f"{tag} ④ eta_opportunity arr={got_opp} v5={post['eta_opp']}"
    m = cand_jit(w2, g)
    cand = np.asarray(m.cand)
    for k, cid in enumerate(tb.crane_ids):
        got = [tb.job_ids[n] for n in np.nonzero(cand[k])[0]]
        assert got == post["cands"][cid], f"{tag} ④ candidates_for({cid}) arr={got} v5={post['cands'][cid]}"
    open_ = WK.open_with_armed(m.cand, m.eligible, w2.wake.eta_armed, opp)
    open_ids = tuple(tb.crane_ids[k] for k in range(K) if bool(open_[k]))
    assert open_ids == post["open"], f"{tag} ④ _decision_cranes arr={open_ids} v5={post['open']}"
    armed_only = [k for k in range(K) if bool(open_[k]) and not cand[k].any()]
    if post["fired"]:
        FIRED_WORLDS.setdefault(label, []).append((pre, pa))
    return dict(fired=post["fired"], n_eta=n_ed, n_defer=n_dd, multi=n_ed >= 2,
                armed_only_open=bool(armed_only), armed_open_retained=bool(armed_only) and not post["fired"],
                busy_at_fire=post["fired"] and any(post["busy"]),
                opp_true=any(got_opp), opp_neg_gap=any(got_opp) and post["neg_gap"],
                yield_cleared=post["fired"] and any(np.asarray(pre.cranes.yielded)))


def check_defer_record(label, i, d: DeferRec, w0):
    """`schedule_defer_wake` 한 호출 — 앞 목록을 담은 세계에 t 를 예약해 뒤 목록과 대조."""
    D = w0.wake.defer_wake_s.shape[0]
    assert len(d.before) <= D
    ds = np.full((D,), EMPTY_TIME, np.float64)
    ds[:len(d.before)] = d.before
    w = w0._replace(clock=jnp.asarray(d.clock, jnp.float64), end_s=jnp.asarray(d.end, jnp.float64),
                    wake=w0.wake._replace(defer_wake_s=jnp.asarray(ds), defer_n=jnp.asarray(len(d.before), jnp.int32)))
    w2 = sched_jit(w, d.t)
    got = _defer_list(w2)
    assert got == d.after, f"[{label} defer #{i} clock={d.clock} t={d.t}] arr={got} v5={d.after}"
    assert int(w2.overflow) == 0
    return len(d.after) > len(d.before)


def run_and_check(label):
    prof, scn, level, drive, caps_over = STAGES[label]()
    caps, _ = _caps(scn, prof)
    caps.update(caps_over)
    w0, tb = to_block_world(prof, scn, n_wake=caps["n_max"], n_defer=N_DEFER, **caps)
    g = Geom.from_profile(prof)
    horizon = float(prof.decision_horizon_s)
    sim, recs, drecs, kinds = run_v5_hooked(prof, scn, level, drive, w0, tb, g)
    assert recs, f"[{label}] _consume_due_wakes 가 한 번도 안 불렸다"
    assert sim.terminal
    flags = [check_record(label, i, r, tb, g, level, horizon) for i, r in enumerate(recs)]
    accepted = [check_defer_record(label, i, d, w0) for i, d in enumerate(drecs)]
    n_eta_log = sum(1 for (_, k, _) in sim.event_log if k == "ETA_WAKE")
    n_def_log = sum(1 for (_, k, _) in sim.event_log if k == "DEFER_WAKE")
    assert sum(f["n_eta"] for f in flags) == n_eta_log == sim._wake_idx
    assert sum(f["n_defer"] for f in flags) == n_def_log == sum(accepted)
    if level != PA:
        assert n_eta_log == 0 and all(r.nwt_pre == EMPTY_TIME or r.post["defer"] for r in recs)
    REPORT[label] = dict(
        K=len(tb.crane_ids), level=level.value, calls=len(recs), fired=sum(f["fired"] for f in flags),
        eta=n_eta_log, defer=n_def_log, multi=sum(f["multi"] for f in flags),
        armed_open=sum(f["armed_only_open"] for f in flags), retained=sum(f["armed_open_retained"] for f in flags),
        busy_fire=sum(f["busy_at_fire"] for f in flags), opp=sum(f["opp_true"] for f in flags),
        neg_gap=sum(f["opp_neg_gap"] for f in flags), yclr=sum(f["yield_cleared"] for f in flags),
        sched=len(drecs), sched_ok=sum(accepted), decisions=max(r.n_decisions for r in recs),
        kinds=kinds, n_wakes=len(sim._eta_wakes), events=len(sim.event_log), backlog=sim.unfinished_backlog())
    return sim, recs, tb, g, w0


# ───────────────────────────────────────────────── 무대 목록 — label → () → (prof, scn, level, drive, caps 덮어쓰기)
def _eta_minus(lo, hi):
    return lambda j, rng: -float(rng.randint(lo, hi))


STAGES: dict[str, object] = {
    # yr050 가족 (K=2 · 40×4×4 · n_max 8)
    "y50-blocked-resolver": lambda: (PROF, blocked_target_sc(900.0, 1500.0), PA, drive_resolver, {}),
    "y50-blocked-ref": lambda: (PROF, blocked_target_sc(900.0, 1500.0), PA, drive_ref, {}),
    "y50-blocked-waitall": lambda: (PROF, blocked_target_sc(900.0, 1500.0), PA, drive_waitall, {}),
    "y50-blocked-defer": lambda: (PROF, blocked_target_sc(900.0, 1500.0), PA, drive_defer, {}),
    "y50-blocked-lowinfo": lambda: (PROF, blocked_target_sc(900.0, 1500.0), GI, drive_ref, {}),
    "y50-busy-at-wake": lambda: (PROF, busy_at_wake_sc(), PA, drive_resolver, {}),
    "y50-no-eta": lambda: (PROF, blocked_target_sc(None, 1500.0, "no-eta"), PA, drive_resolver, {}),
    "y50-neg-gap": lambda: (PROF, neg_gap_sc(), PA, drive_resolver, {}),
    "y50-neg-gap-ref": lambda: (PROF, neg_gap_sc(), PA, drive_ref, {}),
    "y50-gate-in-eta": lambda: (PROF, gate_in_eta_sc(), PA, drive_ref, {}),
    # 조각 1 §10 가족 (K=1 · 10×4×4)
    "p1-spec10-eta=arr": lambda: (piece1_profile(), with_eta(piece1_scenario(), 0.0, "p1-eta-arr"), PA, drive_ref, {}),
    "p1-spec10-eta=arr-resolver": lambda: (piece1_profile(), with_eta(piece1_scenario(), 0.0, "p1-eta-arr-r"), PA, drive_resolver, {}),
    "p1-spec10-eta-late": lambda: (piece1_profile(), with_eta(piece1_scenario(), lambda j, r: 1800.0 + 7.0, "p1-eta-late"), PA, drive_ref, {}),
    "p1-crowded-eta": lambda: (piece1_profile(), with_eta(crowded_scenario(), 1500.0, "p1-crowded-eta"), PA, drive_ref, {}),
}
for _seed in (1, 2, 3, 4):
    STAGES[f"k2-random-{_seed}"] = (lambda sd=_seed: (
        prof_k2(3.0), with_eta(random_scenario(sd, int_arrivals=True), _eta_minus(0, 2400), f"k2-eta-{sd}", rng=random.Random(sd)),
        PA, drive_resolver if sd % 2 else drive_ref, dict(n_max=16)))


def _ensure(label: str) -> dict:
    if label not in REPORT:
        run_and_check(label)
    return REPORT[label]


# ───────────────────────────────────────────────── ① 시드 — seed_wakes == v5 _eta_wakes == host_convert
@pytest.mark.parametrize("stage", ["y50-blocked-ref", "y50-gate-in-eta", "y50-no-eta", "p1-spec10-eta=arr",
                                   "p1-crowded-eta", "k2-random-1"])
def test_seed_wakes_matches_v5_and_host(stage):
    prof, scn, level, _, caps_over = STAGES[stage]()
    caps, _ = _caps(scn, prof)
    caps.update(caps_over)
    w0, tb = to_block_world(prof, scn, n_wake=caps["n_max"], n_defer=N_DEFER, **caps)
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
    K, horizon = len(tb.crane_ids), float(prof.decision_horizon_s)
    wk, dropped = WK.seed_wakes(w0.orders, K, horizon_s=horizon, end_s=w0.end_s, n_wake=caps["n_max"], n_defer=N_DEFER)
    jidx = tb.job_index
    exp = [(float(t), jidx[jid]) for (t, jid) in sim._eta_wakes]
    ws, wj = np.asarray(wk.eta_wake_s), np.asarray(wk.eta_wake_job)
    got = [(float(ws[i]), int(wj[i])) for i in range(len(ws)) if np.isfinite(ws[i])]
    assert got == exp, f"[{stage}] seed_wakes arr={got} v5={exp}"
    assert int(dropped) == 0 and int(WK.n_wakes(w0.orders, horizon_s=horizon, end_s=w0.end_s)) == len(exp)
    assert np.all(wj[len(exp):] == EMPTY_ID) and np.all(np.isposinf(ws[len(exp):]))
    # host_convert (to_block_world n_wake>0) 와 같은 배열
    assert np.array_equal(ws, np.asarray(w0.wake.eta_wake_s)) and np.array_equal(wj, np.asarray(w0.wake.eta_wake_job))
    assert int(wk.wake_idx) == sim._wake_idx == 0 and not np.any(np.asarray(wk.eta_armed)) and int(wk.defer_n) == 0
    if stage == "y50-gate-in-eta":
        assert exp == [(700.0, jidx["J-OUT-T"])]          # 2500−1800; GATE_IN 부재 (test_yr050:142)
    if stage == "p1-spec10-eta=arr":
        assert [t for t, _ in exp] == [0.0, 0.0, 0.0]      # 도착 300·300·1500 − 1800 → 전부 0


def test_seed_wakes_reports_overflow_instead_of_silently_truncating():
    prof, scn = piece1_profile(), with_eta(piece1_scenario(), 0.0, "p1-eta-arr")
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, n_wake=caps["n_max"], **caps)
    sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=PA)
    wk, dropped = WK.seed_wakes(w0.orders, 1, horizon_s=1800.0, end_s=w0.end_s, n_wake=1)
    assert int(dropped) == len(sim._eta_wakes) - 1 == 2
    assert (float(wk.eta_wake_s[0]), tb.job_ids[int(wk.eta_wake_job[0])]) == sim._eta_wakes[0]
    # N < W 인 경우도 모양이 맞고 뒤가 빈 칸
    wk2, d2 = WK.seed_wakes(w0.orders, 1, horizon_s=1800.0, end_s=w0.end_s, n_wake=w0.n + 5)
    assert int(d2) == 0 and wk2.eta_wake_s.shape == (w0.n + 5,) and np.all(np.isposinf(np.asarray(wk2.eta_wake_s)[3:]))


# ───────────────────────────────────────────────── ② 모든 무대 · 모든 _consume_due_wakes 호출
@pytest.mark.parametrize("stage", list(STAGES), ids=list(STAGES))
def test_wake_equivalence_every_call(stage):
    run_and_check(stage)


# ───────────────────────────────────────────────── ③ 설계 무대의 성질 (v5 가 실제로 그 경로를 밟았는지)
def test_blocked_target_pre_rehandle_before_arrival():
    """test_yr050:71-79 — 첫 도착(1500) 전 wake(0) 로 열린 결정에서 선제 재조작이 실행된다. armed 항만으로 열린 결정 ≥ 1."""
    sim, recs, tb, _, _ = run_and_check("y50-blocked-resolver")
    r = REPORT["y50-blocked-resolver"]
    assert r["eta"] == 1 and r["armed_open"] >= 1 and r["kinds"].get("PRE_REHANDLE", 0) >= 1, r
    j = sim.jobs["J-OUT-T"]
    assert j.status.name == "DONE" and j.rehandle_count == 0 and sim.kpis.rehandle_count == 1
    first = next(x for x in recs if x.post["fired"])
    assert float(first.pre.clock) == 0.0 and first.post["log"][-1] == (0.0, "ETA_WAKE", "J-OUT-T")


def test_busy_at_wake_keeps_armed_until_idle():
    """test_yr050:81-104 — wake(50) 순간 바쁜 크레인의 armed 가 유지돼 유휴화 시점에 (fired 아닌 호출에서) 개방된다."""
    r = _ensure("y50-busy-at-wake")
    assert r["busy_fire"] >= 1 and r["retained"] >= 1, r


def test_negative_gap_opens_opportunity():
    """ETA(60) 가 지난 미도착 트럭도 선제 기회다 — 명세의 '0<eta−clock' 이 아니라 v5 코드가 정본."""
    for lab in ("y50-neg-gap", "y50-neg-gap-ref"):
        r = _ensure(lab)
        assert r["neg_gap"] >= 1, r


def test_no_eta_never_fires():
    r = _ensure("y50-no-eta")
    assert r["n_wakes"] == 0 and r["fired"] == 0 and r["eta"] == 0 and r["opp"] == 0, r


def test_low_info_level_disables_eta_wakes():
    """GATE_IN: v5 `_next_wake_time() is None`·ETA_WAKE 0건 — pre_advice=False 가 그것. 같은 세계에 pre_advice=True 를
    주면 wake 가 보인다(플래그가 실제로 일하는 증거)."""
    _, recs, _, _, _ = run_and_check("y50-blocked-lowinfo")
    r = REPORT["y50-blocked-lowinfo"]
    assert r["eta"] == 0 and r["fired"] == 0 and r["n_wakes"] == 1, r
    assert all(x.nwt_pre == EMPTY_TIME for x in recs)
    w = recs[0].pre
    assert float(WK.next_wake_time(w, pre_advice=False)) == EMPTY_TIME
    assert float(WK.next_wake_time(w, pre_advice=True)) == 0.0
    _, f_true = WK.consume_due_wakes(w, pre_advice=True)
    assert bool(f_true)


def test_waitall_finite_and_strictly_increasing():
    r = _ensure("y50-blocked-waitall")
    assert r["fired"] >= 1 and r["decisions"] >= 1, r


def test_defer_wakes_scheduled_consumed_and_rejected():
    r = _ensure("y50-blocked-defer")
    assert r["sched"] >= 4 and r["sched_ok"] >= 1 and r["sched"] > r["sched_ok"], r     # 거부 경로도 찔렀다
    assert r["defer"] >= 1, r


def test_multi_wake_in_one_call():
    r = _ensure("p1-spec10-eta=arr")
    assert r["multi"] >= 1 and r["eta"] == 3, r


def test_late_eta_wake_after_arrival_gives_no_opportunity():
    r = _ensure("p1-spec10-eta-late")
    assert r["eta"] >= 1 and r["opp"] == 0, r


# ───────────────────────────────────────────────── ④ schedule_defer_wake · 소비 — v5 를 직접 불러 무작위 열 대조
def test_schedule_and_consume_defer_random_sequence():
    """v5 sim 의 clock 을 옮겨 가며 `schedule_defer_wake`·`_consume_due_wakes` 를 직접 부르고 배열과 맞춘다
    (과거 · 정확히 now · now+EPS/2 · 창 밖 · 중복 · 동시각 · 한 번에 여러 개 도래)."""
    prof, scn = PROF, blocked_target_sc(900.0, 1500.0)
    caps, _ = _caps(scn, prof)
    w0, tb = to_block_world(prof, scn, n_wake=caps["n_max"], n_defer=N_DEFER, **caps)
    for level in (PA, GI):
        sim = TerminalSimulator(prof, scn, check_invariants=True, info_level=level)
        w = w0._replace(wake=wake_from_sim(sim, w0, tb))
        rng = random.Random(7)
        n_ok = n_fire = 0
        for step in range(60):
            c = float(sim.clock)
            pick = rng.random()
            if pick < 0.15:
                t = c - rng.uniform(0.0, 300.0)                       # 과거
            elif pick < 0.25:
                t = c                                                 # 정확히 now → 무시
            elif pick < 0.3:
                t = c + 5e-10                                         # now+EPS/2 → 무시
            elif pick < 0.4:
                t = sim.end + rng.uniform(0.0, 100.0)                 # 창 밖
            elif pick < 0.5 and sim._defer_wakes:
                t = rng.choice(sim._defer_wakes)                      # 중복
            else:
                t = c + float(rng.randint(1, 900))
            if len(sim._defer_wakes) < N_DEFER:
                before = len(sim._defer_wakes)
                sim.schedule_defer_wake(t)
                w = sched_jit(w, t)
                n_ok += len(sim._defer_wakes) > before
                assert _defer_list(w) == list(sim._defer_wakes), f"step {step} t={t} clock={c}"
                assert int(w.overflow) == 0
            # 시계를 옮기고 소비 — 여러 개가 한 번에 도래하기도 한다
            if rng.random() < 0.5:
                c2 = c + float(rng.randint(0, 600))
                sim.clock = c2
                w = w._replace(clock=jnp.asarray(c2, jnp.float64))
                fired = sim._consume_due_wakes()
                w, f_arr = consume_jit(w, pre_advice=(level == PA))
                n_fire += bool(fired)
                assert bool(f_arr) == fired and _defer_list(w) == list(sim._defer_wakes)
                assert int(w.wake.wake_idx) == sim._wake_idx
                assert tuple(bool(x) for x in np.asarray(w.wake.eta_armed)) == tuple(c in sim._eta_armed for c in tb.crane_ids)
                assert _norm_log(event_log_from_arrays(w, tb)) == _norm_log(sim.event_log)
                assert float(nwt_jit(w, pre_advice=(level == PA))) == _opt(sim._next_wake_time())
        assert n_ok >= 10 and n_fire >= 5, (n_ok, n_fire)
        # 칸이 꽉 찬 뒤의 창 안 예약은 overflow 로 표시된다 (v5 는 무한 리스트 — 표현 한계를 조용히 넘기지 않는다).
        # 새 세계(clock 0 · end 9000)에서 1..D 초를 채운 뒤 100 초를 더 넣는다 — 창 안이므로 v5 라면 받았을 예약이다.
        full = w0._replace(wake=wake_from_sim(TerminalSimulator(prof, scn, check_invariants=True, info_level=level), w0, tb))
        for i in range(N_DEFER):
            full = sched_jit(full, 1.0 + i)
        assert int(full.wake.defer_n) == N_DEFER and int(full.overflow) == 0
        over = sched_jit(full, 100.0)
        assert int(over.overflow) == 1 and _defer_list(over) == _defer_list(full)
        assert int(sched_jit(full, float(full.end_s) + 1.0).overflow) == 0     # 창 밖은 무시 — 넘침 아님


def test_schedule_defer_with_zero_slots_flags_overflow():
    prof, scn = piece1_profile(), piece1_scenario()
    caps, _ = _caps(scn, prof)
    w0, _ = to_block_world(prof, scn, **caps)                        # n_defer=0 · n_wake=0 (Y01 과 같은 꼴)
    assert w0.wake.defer_wake_s.shape == (0,) and w0.wake.eta_wake_s.shape == (0,)
    w1 = WK.schedule_defer_wake(w0, 10.0)
    assert int(w1.overflow) == 1
    w2 = WK.schedule_defer_wake(w0, -5.0)                            # 창 밖은 무시 — 넘침 아님
    assert int(w2.overflow) == 0
    w3, fired = WK.consume_due_wakes(w0, pre_advice=True)
    assert not bool(fired) and int(w3.log.n) == 0
    assert float(WK.next_wake_time(w0, pre_advice=True)) == EMPTY_TIME


# ───────────────────────────────────────────────── ⑤ jit · vmap 이 답을 바꾸지 않는다
def test_jit_and_vmap_agree_on_fired_worlds():
    labs = ("y50-blocked-ref", "y50-blocked-waitall", "y50-blocked-defer")     # 같은 시나리오 → 같은 모양
    for lab in labs:
        _ensure(lab)
    worlds = [w for lab in labs for (w, pa) in FIRED_WORLDS.get(lab, []) if pa][:6]
    assert len(worlds) >= 3
    assert len({tuple(l.shape for l in jax.tree_util.tree_leaves(w)) for w in worlds}) == 1
    eager = [WK.consume_due_wakes(w, pre_advice=True) for w in worlds]
    jitted = [consume_jit(w, pre_advice=True) for w in worlds]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    run_b = jax.jit(jax.vmap(lambda w: WK.consume_due_wakes(w, pre_advice=True)))
    out_b = run_b(batched)
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(eager[0])]
    for i in range(len(worlds)):
        le = [np.asarray(l) for l in jax.tree_util.tree_leaves(eager[i])]
        lj = [np.asarray(l) for l in jax.tree_util.tree_leaves(jitted[i])]
        lb = [np.asarray(l)[i] for l in jax.tree_util.tree_leaves(out_b)]
        bad_j = [names[j] for j, (a, b) in enumerate(zip(le, lj)) if not np.array_equal(a, b, equal_nan=True)]
        bad_b = [names[j] for j, (a, b) in enumerate(zip(le, lb)) if not np.array_equal(a, b, equal_nan=True)]
        assert not bad_j and not bad_b, f"세계 {i}: jit 다름 {bad_j} · vmap 다름 {bad_b}"
        assert bool(eager[i][1])
    nwt_b = jax.jit(jax.vmap(lambda w: WK.next_wake_time(w, pre_advice=True)))(batched)
    assert [float(x) for x in nwt_b] == [float(WK.next_wake_time(w, pre_advice=True)) for w in worlds]


# ───────────────────────────────────────────────── ⑥ 보고 — 무엇이 실제로 시험됐는지
def test_zz_report(capsys):
    for lab in STAGES:
        _ensure(lab)
    tot = lambda key: sum(r[key] for r in REPORT.values())
    with capsys.disabled():
        print("\n[wake equiv report]  calls = _consume_due_wakes 호출 · fired = 발화 · eta/defer = 로그 건수 · multi = 한 호출에 ETA ≥2 · "
              "armed_open = armed 항만으로 열린 결정 · retained = 그중 발화 없는 호출(유휴화 시점) · neg_gap = 음수 gap 기회 · sched = DEFER 예약 호출(ok=수용)")
        for k, r in REPORT.items():
            print(f"  {k:28s} K={r['K']} {r['level']:12s} calls={r['calls']:3d} fired={r['fired']:2d} eta={r['eta']:2d} defer={r['defer']:2d} "
                  f"multi={r['multi']} armed_open={r['armed_open']:2d} retained={r['retained']} busy_fire={r['busy_fire']} "
                  f"opp={r['opp']:3d} neg_gap={r['neg_gap']:2d} yclr={r['yclr']} sched={r['sched']}/{r['sched_ok']} "
                  f"dec={r['decisions']:3d} kinds={r['kinds']} backlog={r['backlog']}")
        print(f"  stages={len(REPORT)} · calls={tot('calls')} · fired={tot('fired')} · eta={tot('eta')} · defer={tot('defer')} · "
              f"multi={tot('multi')} · armed_open={tot('armed_open')} · retained={tot('retained')} · neg_gap={tot('neg_gap')} · "
              f"yclr={tot('yclr')} · sched={tot('sched')}/{tot('sched_ok')}")
    assert tot("fired") >= 15 and tot("eta") >= 20, "ETA wake 발화가 충분히 시험되지 않았다"
    assert tot("multi") >= 1, "한 호출에 여러 wake 를 소비하는 경로가 시험되지 않았다"
    assert tot("armed_open") >= 3, "armed 항만으로 열린 결정이 시험되지 않았다"
    assert tot("retained") >= 1, "바쁜 크레인의 armed 잔존(유휴화 시점 개방)이 시험되지 않았다"
    assert tot("neg_gap") >= 1, "음수 gap 기회(명세와 다른 v5 게이트)가 시험되지 않았다"
    assert tot("defer") >= 1 and tot("sched") > tot("sched_ok"), "DEFER 예약·소비·거부가 시험되지 않았다"
    assert tot("yclr") >= 1, "wake 발화가 yielded 를 지우는 경로가 시험되지 않았다"
    assert sum(1 for r in REPORT.values() if r["kinds"].get("PRE_REHANDLE", 0) > 0) >= 2
    assert any(r["kinds"].get("REPOSITION", 0) > 0 for r in REPORT.values()), "REPOSITION 실행 중 상태가 시험되지 않았다"
    assert any(r["level"] != "PRE_ADVICE" for r in REPORT.values())
