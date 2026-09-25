"""배열 본선·이송(gpu/vessel.py) 이 v5 `TerminalSimulator` 와 **같은 답**을 내는가 ([[YR-327]] 조각 4 · key=vessel).

■ 방법 — v5 를 실제로 굴리며 **본선 사건마다 상태를 가로채** 배열 처리기에 같은 입력을 주고 결과를 `==` 로 댄다
  ① 처리기 lockstep: VESSEL_START·STS_MOVE·TRANSFER_ARRIVE·VESSEL_RELEASED·PLAN_CHANGE 와 JOB_COMPLETED(적하 훅)
     마다 v5 `_handle` 앞뒤 상태를 찍고, 앞 상태로 배열 세계를 만들어 처리기를 돌린 뒤 뒤 상태와 대조 —
     배별 started/remaining/buffer/blocked_since/wait_accum/done/actual_completion/계획(pc·basis·etd),
     이송차 busy_until·pending 열, **새로 push 된 사건열(시각·종류·대상, 넣은 순서)**, 오더 status·deadline,
     비용 vessel_delay/depart_delay(pending·episode)·kpi berth_overrun, yielded 해제.
  ② `_advance` 마다 STS 대기 적분·이송 대기 적분 (integrate_vessel_wait) 과 rate×dt 증분(mul_exact) 대조
  ③ `_refresh_rates` 마다 sts_wait·transfer_wait 요율 (vessel_rates) 대조
  ④ `_finalize` 의 clearout (clearout_vessels) 대조
  ⑤ ★재현(replay): 배열 상태를 **이어 가며** v5 사건 로그를 따라간다 — 배열 큐가 스스로 만든 본선 사건이
     v5 의 다음 본선 사건과 (시각·종류·대상) 같아야 하고, 끝에서 배·이송·비용 4항·berth·남은 큐가 같다.
  전부 허용오차 없음(==), 기대값 손기입 없음.

■ 무대 (fixtures.build_minimal_terminal_scenario 를 크레인 1대로 — YC-B 대상 주입은 무시됨 866/870행)
  base  양하 2·적하 2 · YT 2대 180s · PLAN_CHANGE(완료시각) — 정상 경로
  (a)   양하 5 moves · YT 1대 · 600s → 안벽 버퍼 cap 3 도달 → STS 막힘(blocked)·TRANSFER_ARRIVE 가 재개
  (b)   적하 5 moves 를 트럭 5대와 섞어 굶김 → buffer 0 으로 STS 막힘·적하 반출 완료(JOB_COMPLETED)→이송→재개
  (c)   사전식 무대: 양하 110 moves('-100' < '-11') · cadence 3600/27.5 · 선석 초과·출항 지연·PLAN_CHANGE 4종 키
        (완료 None 세팅·basis None·etd·작업마감 쌍(모르는 job 포함)) · 적하 6 moves 마감 초과
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import vessel as VS                                              # noqa: E402
from yard_rl.v6.gpu.events import EMPTY_TIME, empty_queue, next_event               # noqa: E402
from yard_rl.v6.gpu.exact import mul_exact                                           # noqa: E402
from yard_rl.v6.gpu.host_convert import EV_NAMES, STATUS_NAMES, to_block_world       # noqa: E402
from yard_rl.v6.gpu.state import (C_DEPART_DELAY, C_STS_WAIT, C_TRANSFER_WAIT, C_VESSEL_DELAY,   # noqa: E402
                                  COST_TERMS, EV_PLAN_CHANGE, EV_STS_MOVE, EV_VESSEL_RELEASED,
                                  EV_VESSEL_START, RATE_TERMS)
from yard_rl.v6.world.contract.schema import CandidateKind                           # noqa: E402
from yard_rl.v6.world.contract.vessel import CompletionBasis                         # noqa: E402
from yard_rl.v6.world.domain.enums import ContainerSize, JobFlow, LoadStatus         # noqa: E402
from yard_rl.v6.world.domain.models import Container, Job                            # noqa: E402
from yard_rl.v6.world.integrated import fixtures                                     # noqa: E402
from yard_rl.v6.world.integrated.engine import CraneAssignment, TerminalSimulator    # noqa: E402
from yard_rl.v6.world.integrated.profile import TransferFleetSpec                    # noqa: E402
from yard_rl.v6.world.integrated.scenario import InjectedEvent, TerminalScenario     # noqa: E402
from yard_rl.v6.world.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType   # noqa: E402

FT40 = ContainerSize.FT40
VESSEL_KINDS = ("VESSEL_START", "STS_MOVE", "TRANSFER_ARRIVE", "VESSEL_RELEASED", "PLAN_CHANGE")
P_CAP = 64

#: 시험 집계 (보고용)
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── 무대
def _c(cid, bay, row, tier, size=FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL, block="B1",
                     bay=bay, row=row, tier=tier)


def _disch_job(jid, vid, rel, deadline=7200.0):
    return Job(job_id=jid, flow=JobFlow.VESSEL_DISCHARGE, release_time=rel, actual_gate_in=None,
               actual_block_arrival=None, target_container=None, inbound_size=FT40,
               inbound_load=LoadStatus.FULL, deadline=deadline, priority_class=1, vessel_id=vid)


def _load_job(jid, vid, rel, target, deadline=8000.0):
    return Job(job_id=jid, flow=JobFlow.VESSEL_LOAD, release_time=rel, actual_gate_in=None,
               actual_block_arrival=None, target_container=target, deadline=deadline,
               priority_class=1, vessel_id=vid)


def _truck_out(jid, target, arrival):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0, actual_gate_in=max(0.0, arrival - 600.0),
               actual_block_arrival=arrival, target_container=target)


def _vessel(vid, work, start, pc, basis, etd, moves, cadence=144.0, cap=3):
    return VesselProcess(vid, work, VesselPlan(planned_start_s=start, planned_completion_s=pc,
                                                completion_basis=basis, etd_s=etd, total_moves=moves,
                                                sts_move_interval_s=cadence, quay_buffer_cap=cap))


def profile_k1(n_units=2, move_time=180.0):
    base = fixtures.build_integrated_profile()
    return replace(base, cranes=(fixtures._spec("YC-A"),),
                   transfer=TransferFleetSpec("TF1", "YT", n_units=n_units, move_time_s=move_time))


def stage_base():
    return profile_k1(), fixtures.build_minimal_terminal_scenario()


def stage_a():
    """양하 5 moves · YT 1대 · 600s — 버퍼 cap 3 도달로 STS 막힘."""
    base = fixtures.build_minimal_terminal_scenario()
    jobs = [j for j in base.jobs if j.vessel_id != "V-DISCH"]
    jobs += [_disch_job(f"J-VES-D{m}", "V-DISCH", 600.0 + 144.0 * m) for m in range(5)]
    vessels = [_vessel("V-DISCH", VesselWorkType.DISCHARGE, 600.0, 7200.0, CompletionBasis.PLAN_COMPUTED, 9000.0, 5),
               _vessel("V-LOAD", VesselWorkType.LOAD, 1200.0, 8000.0, None, 9500.0, 2)]
    scn = TerminalScenario(scenario_id="vessel-a", seed=0, horizon_s=7200.0, drain_window_s=3600.0,
                           containers=dict(base.containers), jobs=jobs, vessels=vessels,
                           injected_events=list(base.injected_events))
    return profile_k1(n_units=1, move_time=600.0), scn


def stage_b():
    """적하 5 moves 를 트럭 5대와 섞어 굶김 — buffer 0 으로 STS 막힘, 반출 완료→이송→재개."""
    base = fixtures.build_minimal_terminal_scenario()
    containers = dict(base.containers)
    for i, bay in enumerate((33, 34, 36)):
        containers[f"C-VL{i + 3}"] = _c(f"C-VL{i + 3}", bay, 2, 1)
    for i, bay in enumerate((10, 12, 14, 16, 18)):
        containers[f"C-T{i}"] = _c(f"C-T{i}", bay, 3, 1)
    jobs = [j for j in base.jobs if j.vessel_id != "V-LOAD"]
    targets = ["C-VL", "C-VL2", "C-VL3", "C-VL4", "C-VL5"]
    jobs += [_load_job(f"J-VES-L{m}", "V-LOAD", 1200.0 + 144.0 * m, targets[m]) for m in range(5)]
    jobs += [_truck_out(f"J-OUT-T{i}", f"C-T{i}", 1150.0 + 60.0 * i) for i in range(5)]
    vessels = [_vessel("V-DISCH", VesselWorkType.DISCHARGE, 600.0, 7200.0, CompletionBasis.PLAN_COMPUTED, 9000.0, 2),
               _vessel("V-LOAD", VesselWorkType.LOAD, 1200.0, 8000.0, None, 9500.0, 5)]
    scn = TerminalScenario(scenario_id="vessel-b", seed=0, horizon_s=7200.0, drain_window_s=3600.0,
                           containers=containers, jobs=jobs, vessels=vessels,
                           injected_events=list(base.injected_events))
    return profile_k1(), scn


def stage_c():
    """사전식 무대 — 양하 110 moves(job id '-00'..'-109' → 사전식 '-100' < '-11') · cadence 3600/27.5 ·
    선석 초과·출항 지연·PLAN_CHANGE 키 4종 · 적하 6 moves 마감 초과."""
    cad = 3600.0 / 27.5
    containers = {f"C-VL{i}": _c(f"C-VL{i}", 30 + i, 2, 1) for i in range(6)}
    containers["C-A1"] = _c("C-A1", 5, 1, 1)
    jobs = [_disch_job(f"J-V-DISC-0-{m:02d}", "V-DISC-0", 0.0 + cad * m, deadline=15000.0) for m in range(110)]
    jobs += [_load_job(f"J-V-LOAD-1-{m:02d}", "V-LOAD-1", 300.0 + cad * m, f"C-VL{m}", deadline=1500.0) for m in range(6)]
    jobs += [_truck_out("J-OUT-A", "C-A1", 400.0)]
    vessels = [_vessel("V-DISC-0", VesselWorkType.DISCHARGE, 0.0, 12000.0, CompletionBasis.TOS_TARGET, 12500.0, 110, cadence=cad),
               _vessel("V-LOAD-1", VesselWorkType.LOAD, 300.0, 900.0, CompletionBasis.PLAN_COMPUTED, 1000.0, 6, cadence=cad)]
    injected = [
        InjectedEvent(500.0, "PLAN_CHANGE", "V-LOAD-1",
                      data=(("planned_completion_s", 1200.0), ("etd_s", 1300.0),
                            ("job_deadlines", (("J-V-LOAD-1-00", 700.0), ("NOPE", 1.0), ("J-V-LOAD-1-01", None))))),
        InjectedEvent(500.0, "PLAN_CHANGE", "V-DISC-0", data=(("completion_basis", None), ("etd_s", None))),
        InjectedEvent(2000.0, "PLAN_CHANGE", "V-DISC-0", data=(("planned_completion_s", 13000.0),)),
        InjectedEvent(2500.0, "PLAN_CHANGE", "V-NOPE", data=(("planned_completion_s", 1.0),)),
        InjectedEvent(3000.0, "PLAN_CHANGE", "V-LOAD-1", data=()),
    ]
    scn = TerminalScenario(scenario_id="vessel-c", seed=0, horizon_s=20000.0, drain_window_s=3600.0,
                           containers=containers, jobs=jobs, vessels=vessels, injected_events=injected)
    return profile_k1(n_units=2, move_time=180.0), scn


STAGES = {"base": stage_base, "a": stage_a, "b": stage_b, "c": stage_c}


# ───────────────────────────────────────────────── v5 구동 + 가로채기
def _snap(sim) -> dict:
    return dict(
        clock=sim.clock, vessels=copy.deepcopy(sim.vessels), transfer=copy.deepcopy(sim.transfer),
        heap=[(e.time, e.priority, e.seq, e.kind_name, e.payload) for e in sim.queue._heap], seq=sim.queue._seq,
        jobs={jid: (j.status.name, j.deadline) for jid, j in sim.jobs.items()},
        pending=dict(sim.cost._pending), episode=dict(sim.cost._episode), rate=dict(sim.cost._rate),
        berth=sim.kpis.berth_overrun_s, yielded={c.crane_id: c.yielded for c in sim.fleet.all()})


class Recorder:
    """v5 인스턴스의 `_handle`·`_advance`·`_refresh_rates`·`_finalize` 를 감싸 앞뒤 상태를 남긴다."""

    def __init__(self, sim):
        self.sim = sim
        self.handles: list = []      # (t, kind, payload, extra, before, after)
        self.advances: list = []     # (t, clock0, ves0, tr0, rate0, epi0, ves1, tr1, epi1)
        self.refreshes: list = []    # (ves, tr, rate)
        self.finalize: tuple | None = None
        self.completed: list = []    # (t, crane, job_id, plan kind)
        oh, oa, orf, ofin = sim._handle, sim._advance, sim._refresh_rates, sim._finalize

        def handle(ev):
            k = ev.kind_name
            if k in VESSEL_KINDS or k == "JOB_COMPLETED":
                extra = None
                if k == "JOB_COMPLETED":
                    plan = sim._active_plans.get(ev.payload)
                    extra = (plan.job_id, plan.kind.name) if plan is not None else None
                    self.completed.append((ev.time, ev.payload) + (extra or (None, None)))
                before = _snap(sim)
                oh(ev)
                self.handles.append((ev.time, k, ev.payload, extra, before, _snap(sim)))
            else:
                oh(ev)

        def advance(t):
            b = (sim.clock, copy.deepcopy(sim.vessels), copy.deepcopy(sim.transfer), dict(sim.cost._rate),
                 dict(sim.cost._episode))
            oa(t)
            self.advances.append((t,) + b + (copy.deepcopy(sim.vessels), copy.deepcopy(sim.transfer),
                                             dict(sim.cost._episode)))

        def refresh():
            orf()
            self.refreshes.append((copy.deepcopy(sim.vessels), copy.deepcopy(sim.transfer), dict(sim.cost._rate)))

        def finalize():
            if sim._terminal:
                return ofin()
            b = _snap(sim)
            ofin()
            self.finalize = (b, _snap(sim))

        sim._handle, sim._advance, sim._refresh_rates, sim._finalize = handle, advance, refresh, finalize


def run_v5(prof, scn):
    """v5 를 '첫 후보' 규칙으로 끝까지 — (sim, Recorder)."""
    sim = TerminalSimulator(prof, scn, check_invariants=True)
    rec = Recorder(sim)
    while (dp := sim.run_until_decision()) is not None:
        for cid in dp.crane_ids:
            cands = sim.candidates_for(cid)
            if cands:
                sim.assign(cid, CraneAssignment(cid, CandidateKind.SERVE, job_ref=cands[0]))
            else:
                sim.assign(cid, CraneAssignment(cid, CandidateKind.WAIT))
        sim.close_decision()
    assert sim.terminal
    return sim, rec


# ───────────────────────────────────────────────── 배열 쪽 도구
def _caps(scn):
    n0 = len(scn.jobs)
    n_max = max(8, 1 << (n0 - 1).bit_length())
    return dict(n_max=n_max, q_cap=max(32, 4 * n_max), log_cap=8 * n_max + 256)


def base_world(prof, scn):
    w0, tb = to_block_world(prof, scn, **_caps(scn))
    return w0, tb


def _v5_vessels(vessels: dict, vessel_ids) -> dict:
    """v5 VesselProcess dict → vessels_to_v5 와 같은 모양."""
    out = {}
    for vid in vessel_ids:
        v = vessels[vid]
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


def _v5_transfer(tr) -> dict:
    return {"busy_until": list(tr.busy_until), "pending": list(tr.pending),
            "transfer_wait_accum_s": tr.transfer_wait_accum_s}


def _arr_pushed(q, tb, kinds=None) -> list[tuple[float, str, str]]:
    """배열 큐의 산 사건을 **넣은 순서(seq)** 로 — (시각, 종류 이름, 대상 이름)."""
    t, k, tg, sq = (np.asarray(q.time), np.asarray(q.kind), np.asarray(q.target), np.asarray(q.seq))
    out = []
    for i in np.argsort(sq, kind="stable"):
        if not np.isfinite(t[i]):
            continue
        name = EV_NAMES[int(k[i])]
        if kinds is not None and name not in kinds:
            continue
        pay = tb.vessel(int(tg[i])) if name in ("STS_MOVE", "TRANSFER_ARRIVE", "VESSEL_START", "PLAN_CHANGE") \
            else tb.job(int(tg[i]))
        out.append((float(t[i]), name, pay))
    return out


def _v5_pushed(before, after, kinds=None) -> list[tuple[float, str, str]]:
    new = sorted((e for e in after["heap"] if e[2] > before["seq"]), key=lambda e: e[2])
    return [(e[0], e[3], e[4]) for e in new if kinds is None or e[3] in kinds]


def world_from_snapshot(w0, tb, snap, pc_arrays, clock=None):
    """가로챈 v5 앞 상태 → VesselWorld (큐는 비어 있다 — 처리기가 push 한 것만 남는다)."""
    n_max = w0.orders.n
    st = np.asarray(w0.orders.status).copy()
    dl = np.asarray(w0.orders.deadline_s).copy()
    for n, jid in enumerate(tb.job_ids):
        s, d = snap["jobs"][jid]
        st[n] = STATUS_NAMES.index(s)
        dl[n] = EMPTY_TIME if d is None else float(d)
    orders = w0.orders._replace(status=jnp.asarray(st), deadline_s=jnp.asarray(dl))
    yl = np.asarray([snap["yielded"][c] for c in tb.crane_ids], bool)
    cranes = w0.cranes._replace(yielded=jnp.asarray(yl))
    cost = w0.cost._replace(
        pending=jnp.asarray([snap["pending"][t] for t in COST_TERMS], jnp.float64),
        episode=jnp.asarray([snap["episode"][t] for t in COST_TERMS], jnp.float64),
        rate=jnp.asarray([snap["rate"][t] for t in RATE_TERMS], jnp.float64))
    kpi = w0.kpi._replace(berth_overrun=jnp.asarray(float(snap["berth"]), jnp.float64))
    ves = VS.vessel_arrays_from_v5(snap["vessels"], tb.job_ids, tb.vessel_ids, n_max)
    tr = VS.transfer_arrays_from_v5(snap["transfer"], tb.vessel_ids, p_cap=P_CAP)
    w = VS.attach(w0._replace(orders=orders, cranes=cranes, cost=cost, kpi=kpi,
                              queue=empty_queue(w0.queue.capacity)), ves, tr, pc_arrays)
    return w._replace(clock=jnp.asarray(float(snap["clock"] if clock is None else clock), jnp.float64))


#: jit 한 처리기 (모양별 1회 컴파일)
J_START = jax.jit(VS.h_vessel_start)
J_STS = jax.jit(VS.h_sts_move)
J_ARRIVE = jax.jit(VS.h_transfer_arrive)
J_RELEASED = jax.jit(VS.h_vessel_released)
J_PLAN = jax.jit(VS.h_plan_change)
J_LOAD_DONE = jax.jit(VS.on_load_completed)
J_INTEGRATE = jax.jit(VS.integrate_vessel_wait)
J_RATES = jax.jit(VS.vessel_rates)
J_CLEAROUT = jax.jit(VS.clearout_vessels)


def apply_handler(w, kind: str, payload: str, tb, extra=None):
    if kind == "VESSEL_START":
        return J_START(w, tb.vessel_index[payload])
    if kind == "STS_MOVE":
        return J_STS(w, tb.vessel_index[payload])
    if kind == "TRANSFER_ARRIVE":
        return J_ARRIVE(w, tb.vessel_index[payload])
    if kind == "VESSEL_RELEASED":
        return J_RELEASED(w, tb.job_index[payload])
    if kind == "PLAN_CHANGE":
        return J_PLAN(w, tb.vessel_index.get(payload, -1))
    if kind == "JOB_COMPLETED":
        if extra is None or extra[1] != "SERVE":
            return w
        return J_LOAD_DONE(w, tb.job_index[extra[0]], True)
    raise KeyError(kind)


# ───────────────────────────────────────────────── ① 처리기 lockstep
@pytest.mark.parametrize("stage", list(STAGES))
def test_handlers_lockstep_with_v5(stage):
    prof, scn = STAGES[stage]()
    sim, rec = run_v5(prof, scn)
    w0, tb = base_world(prof, scn)
    pc_arrays = VS.plan_change_from_scenario(scn, tb.job_ids, tb.vessel_ids)
    counts: dict[str, int] = {}
    blocked_seen = resumed = 0
    for (t, kind, payload, extra, before, after) in rec.handles:
        w = world_from_snapshot(w0, tb, before, pc_arrays)
        w2 = apply_handler(w, kind, payload, tb, extra)
        pc_arrays = w2.plan_change
        tag = f"[{stage}] t={t} {kind} {payload}"
        # 배·이송
        assert VS.vessels_to_v5(w2.vessels, tb.vessel_ids) == _v5_vessels(after["vessels"], tb.vessel_ids), f"{tag} 배 상태"
        got_tr, exp_tr = VS.transfer_to_v5(w2.transfer, tb.vessel_ids), _v5_transfer(after["transfer"])
        assert got_tr["busy_until"] == exp_tr["busy_until"], f"{tag} busy_until arr={got_tr['busy_until']} v5={exp_tr['busy_until']}"
        assert got_tr["pending"] == exp_tr["pending"], f"{tag} pending arr={got_tr['pending']} v5={exp_tr['pending']}"
        assert got_tr["overflow"] == 0
        # 새로 push 된 사건열 — 넣은 순서
        kinds = VESSEL_KINDS
        assert _arr_pushed(w2.queue, tb, kinds) == _v5_pushed(before, after, kinds), \
            f"{tag} push 열 arr={_arr_pushed(w2.queue, tb, kinds)} v5={_v5_pushed(before, after, kinds)}"
        assert int(w2.queue.overflow) == 0
        # 비용·kpi
        for term, idx in (("vessel_delay", C_VESSEL_DELAY), ("depart_delay", C_DEPART_DELAY)):
            assert float(w2.cost.pending[idx]) == after["pending"][term], f"{tag} pending.{term}"
            assert float(w2.cost.episode[idx]) == after["episode"][term], f"{tag} episode.{term}"
        assert float(w2.kpi.berth_overrun) == after["berth"], f"{tag} berth_overrun"
        if kind != "JOB_COMPLETED":
            # 오더 status·deadline · yielded
            st = np.asarray(w2.orders.status); dl = np.asarray(w2.orders.deadline_s)
            for n, jid in enumerate(tb.job_ids):
                s, d = after["jobs"][jid]
                assert STATUS_NAMES[int(st[n])] == s, f"{tag} 오더 {jid} status arr={STATUS_NAMES[int(st[n])]} v5={s}"
                assert (None if not np.isfinite(dl[n]) else float(dl[n])) == d, f"{tag} 오더 {jid} deadline"
            yl = {c: bool(np.asarray(w2.cranes.yielded)[k]) for k, c in enumerate(tb.crane_ids)}
            assert yl == after["yielded"], f"{tag} yielded arr={yl} v5={after['yielded']}"
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "STS_MOVE":
            b0 = before["vessels"][payload].sts_blocked_since_s
            b1 = after["vessels"][payload].sts_blocked_since_s
            blocked_seen += int(b0 is None and b1 is not None)
        if kind == "TRANSFER_ARRIVE":
            resumed += int(before["vessels"][payload].sts_blocked_since_s is not None
                           and after["vessels"][payload].sts_blocked_since_s is None)
    assert int(np.sum(np.asarray(pc_arrays.consumed))) == sum(1 for (_, k, p, *_r) in rec.handles
                                                             if k == "PLAN_CHANGE" and p in tb.vessel_ids)
    REPORT.setdefault(stage, {}).update(handles=counts, blocked=blocked_seen, resumed=resumed,
                                        events=len(sim.event_log), backlog=sim.unfinished_backlog())
    # 무대가 의도한 경로를 실제로 밟았나
    if stage in ("a", "b", "c"):
        assert blocked_seen >= 1 and resumed >= 1, f"[{stage}] STS 막힘·재개 경로가 안 나왔다 (blocked={blocked_seen}, resumed={resumed})"
    if stage == "a":
        assert any(len(b["transfer"].pending) > 0 for (*_x, b, _a) in rec.handles), "[a] 이송 대기(pending) 가 안 생겼다"
    if stage == "c":
        assert sim.cost.episode_raw()["vessel_delay"] > 0 and sim.cost.episode_raw()["depart_delay"] > 0


# ───────────────────────────────────────────────── ② 시계 전진 적분 · ③ 요율
@pytest.mark.parametrize("stage", ["base", "a", "b"])
def test_advance_integrals_and_rates(stage):
    prof, scn = STAGES[stage]()
    sim, rec = run_v5(prof, scn)
    w0, tb = base_world(prof, scn)
    n_max = w0.orders.n
    end = float(sim.end)
    n_go = 0
    for (t, clock0, ves0, tr0, rate0, epi0, ves1, tr1, epi1) in rec.advances:
        lo, hi = min(clock0, end), min(t, end)
        go = t > clock0 and hi > lo
        if not go:
            assert _v5_vessels(ves1, tb.vessel_ids) == _v5_vessels(ves0, tb.vessel_ids)
            continue
        n_go += 1
        dt = hi - lo
        va, ta = VS.vessel_arrays_from_v5(ves0, tb.job_ids, tb.vessel_ids, n_max), VS.transfer_arrays_from_v5(tr0, tb.vessel_ids, p_cap=P_CAP)
        vb, tb_ = J_INTEGRATE(va, ta, dt)
        got = {vid: float(vb.wait_accum_s[i]) for i, vid in enumerate(tb.vessel_ids)}
        exp = {vid: ves1[vid].sts_wait_accum_s for vid in tb.vessel_ids}
        assert got == exp, f"[{stage}] t={t} sts_wait_accum arr={got} v5={exp}"
        assert float(tb_.wait_accum_s) == tr1.transfer_wait_accum_s, f"[{stage}] t={t} transfer_wait_accum"
        # rate × dt 증분 (통합자의 advance 가 mul_exact 로 더한다) — v5 `_episode += rate*dt` 와 비트 동일
        for term in ("sts_wait", "transfer_wait"):
            inc = float(mul_exact(jnp.asarray(rate0[term], jnp.float64), jnp.asarray(dt, jnp.float64)))
            assert epi0[term] + inc == epi1[term], f"[{stage}] t={t} episode.{term} 증분"
    assert n_go > 0
    n_nonzero = 0
    for (ves, tr, rate) in rec.refreshes:
        va, ta = VS.vessel_arrays_from_v5(ves, tb.job_ids, tb.vessel_ids, n_max), VS.transfer_arrays_from_v5(tr, tb.vessel_ids, p_cap=P_CAP)
        s, w = J_RATES(va, ta)
        assert (float(s), float(w)) == (rate["sts_wait"], rate["transfer_wait"]), \
            f"[{stage}] rate arr={(float(s), float(w))} v5={(rate['sts_wait'], rate['transfer_wait'])}"
        n_nonzero += int(rate["sts_wait"] > 0 or rate["transfer_wait"] > 0)
    REPORT.setdefault(stage, {}).update(advances=n_go, refreshes=len(rec.refreshes), rate_nonzero=n_nonzero)
    if stage in ("a", "b"):
        assert n_nonzero > 0, f"[{stage}] 요율이 한 번도 0 이 아니지 않았다 — 막힘 경로 미검증"
        assert sim.cost.episode_raw()["sts_wait"] > 0


# ───────────────────────────────────────────────── ④ clearout
def _stage_c_unfinished():
    """(c) 를 지평 3000 으로 자르고 양하 계획완료를 2500 으로 — 종료시점 미완 본선(양하 110 moves)에
    end > pc 가 성립해 clearout 이 **실제로 적립**한다. 2000 초 PLAN_CHANGE(완료 13000) 는 뺀다."""
    prof, scn = stage_c()
    vessels = []
    for v in scn.vessels:
        if v.vessel_id == "V-DISC-0":
            v = replace(v, plan=replace(v.plan, planned_completion_s=2500.0, etd_s=2600.0))
        vessels.append(v)
    injected = [ie for ie in scn.injected_events if ie.time <= 1000.0]
    return prof, replace(scn, scenario_id="vessel-c-cut", horizon_s=3000.0, drain_window_s=0.0,
                         vessels=vessels, injected_events=injected)


@pytest.mark.parametrize("stage", ["base", "c-cut"])
def test_finalize_clearout(stage):
    prof, scn = STAGES[stage]() if stage in STAGES else _stage_c_unfinished()
    sim, rec = run_v5(prof, scn)
    assert rec.finalize is not None
    before, after = rec.finalize
    w0, tb = base_world(prof, scn)
    pc_arrays = VS.plan_change_from_scenario(scn, tb.job_ids, tb.vessel_ids)
    w = world_from_snapshot(w0, tb, before, pc_arrays)
    w2 = J_CLEAROUT(w)
    for term, idx in (("vessel_delay", C_VESSEL_DELAY), ("depart_delay", C_DEPART_DELAY)):
        assert float(w2.cost.pending[idx]) == after["pending"][term], f"[{stage}] clearout pending.{term}"
        assert float(w2.cost.episode[idx]) == after["episode"][term], f"[{stage}] clearout episode.{term}"
    assert float(w2.kpi.berth_overrun) == after["berth"], f"[{stage}] clearout berth"
    if stage == "c-cut":
        assert after["episode"]["vessel_delay"] > before["episode"]["vessel_delay"], "clearout 이 실제로 적립하지 않았다"
        assert any(not v.done for v in sim.vessels.values())


# ───────────────────────────────────────────────── ⑤ 재현 — 배열 상태를 이어 가며 v5 사건 로그를 따라간다
def _seed_vessel_queue(w0, tb, scn):
    """VESSEL_START·PLAN_CHANGE 만 남긴 큐 (engine.py:236-242 순서·seq 그대로) — 나머지는 배열 처리기가 스스로 만든다."""
    q = empty_queue(w0.queue.capacity)
    t, k, tg, sq = (np.asarray(w0.queue.time), np.asarray(w0.queue.kind), np.asarray(w0.queue.target), np.asarray(w0.queue.seq))
    keep = np.isfinite(t) & ((k == EV_VESSEL_START) | (k == EV_PLAN_CHANGE))
    tt = np.full_like(t, EMPTY_TIME); kk = np.full_like(k, -1); gg = np.full_like(tg, -1); ss = np.full_like(sq, -1)
    tt[keep], kk[keep], gg[keep], ss[keep] = t[keep], k[keep], tg[keep], sq[keep]
    return q._replace(time=jnp.asarray(tt), kind=jnp.asarray(kk), target=jnp.asarray(gg), seq=jnp.asarray(ss),
                      counter=w0.queue.counter)


def _advance_emulated(w, t):
    """v5 `_advance` 의 본선 몫 + rate×dt (통합자 advance 의 자리) — lo/hi/dt 규칙 그대로."""
    clock, end = float(w.clock), float(w.end_s)
    lo, hi = min(clock, end), min(t, end)
    if t > clock and hi > lo:
        dt = jnp.asarray(hi - lo, jnp.float64)
        ves, tr = J_INTEGRATE(w.vessels, w.transfer, dt)
        add = mul_exact(w.cost.rate, dt)
        cost = w.cost._replace(pending=w.cost.pending.at[C_STS_WAIT].add(add[0]).at[C_TRANSFER_WAIT].add(add[1]),
                               episode=w.cost.episode.at[C_STS_WAIT].add(add[0]).at[C_TRANSFER_WAIT].add(add[1]))
        w = w._replace(vessels=ves, transfer=tr, cost=cost)
    return w._replace(clock=jnp.asarray(float(t), jnp.float64))


def _refresh_emulated(w):
    s, tw = J_RATES(w.vessels, w.transfer)
    return w._replace(cost=w.cost._replace(rate=w.cost.rate.at[0].set(s).at[1].set(tw)))


@pytest.mark.parametrize("stage", list(STAGES))
def test_replay_vessel_chain(stage):
    prof, scn = STAGES[stage]()
    sim, rec = run_v5(prof, scn)
    w0, tb = base_world(prof, scn)
    n_max = w0.orders.n
    ves = VS.vessel_arrays_from_scenario(scn, tb.job_ids, tb.vessel_ids, n_max)
    tr = VS.transfer_arrays_from_profile(prof, p_cap=P_CAP)
    pc = VS.plan_change_from_scenario(scn, tb.job_ids, tb.vessel_ids)
    w = VS.attach(w0._replace(queue=_seed_vessel_queue(w0, tb, scn)), ves, tr, pc)
    completed = list(rec.completed)
    n_vessel_ev = 0
    first_diff = None
    for i, (t, kind, payload) in enumerate(sim.event_log):
        if kind == "DISPATCH":
            continue
        w = _advance_emulated(w, t)
        if kind in VESSEL_KINDS:
            q2, tq, kq, tg, alive = next_event(w.queue)
            got = (float(tq), EV_NAMES[int(kq)] if int(kq) >= 0 else None,
                   (tb.job(int(tg)) if int(kq) == EV_VESSEL_RELEASED else tb.vessel(int(tg))) if bool(alive) else None)
            exp = (t, kind, payload if (kind != "PLAN_CHANGE" or payload in tb.vessel_ids) else None)
            if not bool(alive) or got != exp:
                first_diff = (i, exp, got)
                break
            w = apply_handler(w._replace(queue=q2), kind, payload, tb)
            n_vessel_ev += 1
        elif kind == "JOB_COMPLETED":
            tc, cid, jid, pk = completed.pop(0)
            assert (tc, cid) == (t, payload)
            w = apply_handler(w, "JOB_COMPLETED", payload, tb, (jid, pk))
        w = _refresh_emulated(w)
    assert first_diff is None, f"[{stage}] 본선 사건열이 {first_diff[0]}번째에서 갈린다: v5={first_diff[1]} arr={first_diff[2]}"
    # _finalize (1068-1084행)
    w = _advance_emulated(w, max(float(w.clock), float(w.end_s)))
    w = J_CLEAROUT(w)
    # 끝 상태
    assert VS.vessels_to_v5(w.vessels, tb.vessel_ids) == _v5_vessels(sim.vessels, tb.vessel_ids), f"[{stage}] 끝 배 상태"
    got_tr, exp_tr = VS.transfer_to_v5(w.transfer, tb.vessel_ids), _v5_transfer(sim.transfer)
    assert got_tr["busy_until"] == exp_tr["busy_until"] and got_tr["pending"] == exp_tr["pending"], f"[{stage}] 끝 이송"
    assert got_tr["transfer_wait_accum_s"] == exp_tr["transfer_wait_accum_s"], f"[{stage}] 끝 이송 대기 적분"
    ep = sim.cost.episode_raw()
    bad = []
    for term, idx in (("sts_wait", C_STS_WAIT), ("transfer_wait", C_TRANSFER_WAIT),
                      ("vessel_delay", C_VESSEL_DELAY), ("depart_delay", C_DEPART_DELAY)):
        if float(w.cost.episode[idx]) != ep[term]:
            bad.append((term, float(w.cost.episode[idx]), ep[term]))
    assert not bad, f"[{stage}] 비용이 비트 수준에서 갈린다: {bad}"
    assert float(w.kpi.berth_overrun) == sim.kpis.berth_overrun_s, f"[{stage}] berth_overrun"
    # 남은 큐 (평가창 밖 본선 사건) — v5 힙의 본선 종류와 같아야 한다
    left_v5 = sorted((e.time, e.kind_name, e.payload) for e in sim.queue._heap if e.kind_name in VESSEL_KINDS)
    left_ar = sorted(_arr_pushed(w.queue, tb, VESSEL_KINDS))
    assert left_ar == left_v5, f"[{stage}] 남은 큐 arr={left_ar} v5={left_v5}"
    assert int(w.queue.overflow) == 0 and int(w.transfer.overflow) == 0
    REPORT.setdefault(stage, {}).update(replay_vessel_events=n_vessel_ev,
                                        cost={t: ep[t] for t in ("sts_wait", "transfer_wait", "vessel_delay", "depart_delay")})
    assert n_vessel_ev == sum(1 for (_, k, _) in sim.event_log if k in VESSEL_KINDS)
    if stage == "c":
        # 사전식 함정: '-100' 이 '-11' 보다 먼저 해제됐어야 한다 (v5 sorted(job_id))
        rel = [p for (_, k, p) in sim.event_log if k == "VESSEL_RELEASED" and p.startswith("J-V-DISC")]
        assert rel.index("J-V-DISC-0-100") < rel.index("J-V-DISC-0-11")


# ───────────────────────────────────────────────── 단위 — push_if · 링버퍼 넘침 · jit==eager
def test_push_if_false_leaves_queue_untouched():
    q = empty_queue(4)
    q2 = VS.push_if(q, False, 1.0, EV_STS_MOVE, 0)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(q, q2))
    q3 = VS.push_if(q, True, 1.0, EV_STS_MOVE, 0)
    assert int(q3.counter) == 1 and float(q3.time[0]) == 1.0


def test_ring_buffer_wraps_and_flags_overflow():
    """P=2: 요청 3건이 밀리면 2건만 담고 overflow=1; 꺼내면(head) 칸이 돌아 다시 쓰인다."""
    prof, scn = stage_base()
    w0, tb = base_world(prof, scn)
    ves = VS.vessel_arrays_from_scenario(scn, tb.job_ids, tb.vessel_ids, w0.orders.n)
    tr = VS.empty_transfer(1, 2, 100.0)._replace(busy_until=jnp.asarray([1e9], jnp.float64))   # 유닛이 영원히 바쁨
    pc = VS.plan_change_from_scenario(scn, tb.job_ids, tb.vessel_ids)
    w = VS.attach(w0._replace(queue=empty_queue(16)), ves, tr, pc)._replace(clock=jnp.asarray(5.0, jnp.float64))
    for v in (0, 1, 0):
        w = VS.transfer_request(w, v, True)
    assert VS.pending_entries(w.transfer, tb.vessel_ids) == [(5.0, tb.vessel_ids[0]), (5.0, tb.vessel_ids[1])]
    assert int(w.transfer.overflow) == 1 and int(VS.pending_count(w.transfer)) == 2
    # 유닛을 풀고 재배차 → 머리(배 0) 가 나가고 TRANSFER_ARRIVE 105 가 push 된다
    w = w._replace(transfer=w.transfer._replace(busy_until=jnp.asarray([0.0], jnp.float64)))
    w = VS.dispatch_pending(w, True)
    assert VS.pending_entries(w.transfer, tb.vessel_ids) == [(5.0, tb.vessel_ids[1])]
    assert _arr_pushed(w.queue, tb) == [(105.0, "TRANSFER_ARRIVE", tb.vessel_ids[0])]
    # 다시 넣으면 tail=3 → 칸 1 (3 % 2) 에 들어간다
    w = w._replace(transfer=w.transfer._replace(busy_until=jnp.asarray([1e9], jnp.float64)))
    w = VS.transfer_request(w, 1, True)
    assert VS.pending_entries(w.transfer, tb.vessel_ids) == [(5.0, tb.vessel_ids[1]), (5.0, tb.vessel_ids[1])]


def test_jit_equals_eager_on_sts_move():
    prof, scn = stage_a()
    sim, rec = run_v5(prof, scn)
    w0, tb = base_world(prof, scn)
    pc = VS.plan_change_from_scenario(scn, tb.job_ids, tb.vessel_ids)
    # 막히는 STS_MOVE 와 정상 STS_MOVE 하나씩
    picked = [h for h in rec.handles if h[1] == "STS_MOVE"][:6]
    for (t, kind, payload, extra, before, after) in picked:
        w = world_from_snapshot(w0, tb, before, pc)
        a = VS.h_sts_move(w, tb.vessel_index[payload])
        b = J_STS(w, tb.vessel_index[payload])
        la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
        assert all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True) for x, y in zip(la, lb)), f"t={t} jit≠eager"


def test_zz_report():
    """집계 — 어떤 무대가 어떤 경로를 밟았는지 (조각 실행이면 report_dump 병합본)."""
    print("\n[vessel REPORT]")
    for k, v in REPORT.items():
        print(f"  {k}: {v}")
    assert REPORT, "REPORT 가 비었다"
