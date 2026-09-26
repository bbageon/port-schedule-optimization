"""터미널 변환기(gpu/host_terminal.py) 가 v5 다중블록 무대를 **그대로** 옮기는가 ([[YR-327]] 조각 6 · key=convert).

■ 무대 — `scripts/v6/dump_ground_truth.py run_terminal` 과 같은 구성 (실행은 안 한다 = reset 직후)
    build_diurnal(build_h21_profile(), seed 9900777, OBS_24H, load 30 · 300) → 21블록 · 트럭 30/300 · 본선 30스트림
    v5: {b: ensure_time_ledger(TerminalSimulator(prof, scn; PRE_ADVICE))} → MultiBlockTerminal(extra=admission_epochs(OBS_24H))
        + ScheduledAnnouncer(schedule, lead 1800, end=sim_end_s)
    배열: to_terminal_world(prof, built, lead_s=1800, extra_review_epochs=admission_epochs(OBS_24H))

■ 무엇을 지키나 (기대값 손기입 없음 — v5 객체를 실제로 만들어 읽는다)
  ① 블록 번호 = built["scenarios"] 삽입 순서 = mbt.blocks 순서 = 정렬 순서 · 행 번호 = sorted(namespaced id) (트럭이 앞)
  ② 격자·컨테이너·크레인(위치·레일·idle 장벽) == v5 reset 직후
  ③ 사건 큐 시드 == v5 힙 (time·priority·kind·payload·seq 순서)
  ④ 본선 작업 행 == sim.jobs (열 전부) · 트럭 행 = 없는 오더(block −1·gate_in +inf) 이면서 정적 열 == _job_from_entry 의 Job
     · 장부 모드 켜짐 == time_ledger is not None · 등록 0 == records 비어 있음
  ⑤ 배·이송·ETA wake·검토 시각(review_s == sim.review_epochs) · end/clock
  ⑥ 명단 배열 == ScheduledAnnouncer.by_epoch (에폭별 순서 포함) · SKIP_TAIL · 검토 시각 == admission_epochs
  ⑦ 원장 초기값 · from_terminal_world 초기 dict (n_jobs·빈 로그 해시 == sim.event_stream_hash()·unfinished)
  ⑧ numpy 저장·복원 왕복 (잎 전부 비트)
  ⑨ ★미리 채운 트럭 행이 블록 엔진에 **보이지 않는다** — 터미널 세계의 Y01 조각을 `run_jit` 으로 굴리면
     같은 namespaced 시나리오를 (트럭 없이·빈 장부로) 굴린 v5 와 사건 로그·해시·비용·KPI 가 같다
  + 변환 소요 시간 보고 (zz_report)

실행: WSL venv · x64 CPU. 한 부하당 ≈ 15초 (v5 구성 + 변환). ⑨ 는 load300 에서만 (≈ 12초).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import dispatch as DP                                           # noqa: E402
from yard_rl.v6.gpu import engine_step as ES                                        # noqa: E402
from yard_rl.v6.gpu import host_terminal as HT                                      # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import event_stream_hash, from_block_world, queue_entries   # noqa: E402
from yard_rl.v6.gpu.stack_ops import SIZE_INDEX                                     # noqa: E402
from yard_rl.v6.gpu.state import EMPTY_ID, FL_GATE_IN, FL_GATE_OUT, JS_PLANNED     # noqa: E402
from yard_rl.v6.world.domain.enums import InformationLevel                         # noqa: E402
from yard_rl.v6.world.integrated import engine as eng                              # noqa: E402
from yard_rl.v6.world.integrated import multiblock as mb                           # noqa: E402
from yard_rl.v6.world.integrated import profiles as pr                             # noqa: E402
from yard_rl.v6.world.integrated import terminal_stream as ts                      # noqa: E402
from yard_rl.v6.world.integrated import yard_layout as yl                          # noqa: E402
from yard_rl.v6.world.integrated.terminal_stream import _job_from_entry            # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LEAD_S = 1800.0
SEED = 9_900_777
LVL = InformationLevel.PRE_ADVICE
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── 무대 (v5 · 배열)
def _build(load: int):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, SEED, obs=ts.OBS_24H, layout=layout,
                             params=ts.TerminalStreamParams(load_4h=load), day_total=load, background_seed=SEED)
    return prof, built


def _sim(prof, scn, level=LVL):
    s = eng.TerminalSimulator(prof, scn, check_invariants=True)
    s.info_level = level
    return s


def make_v5_terminal(prof, built, *, lead_s: float = LEAD_S):
    """dump_ground_truth.run_terminal 의 구성 골격 그대로 — 실행하지 않는다 (reset 직후 상태)."""
    sims = {b: ts.ensure_time_ledger(_sim(prof, s)) for b, s in built["scenarios"].items()}
    mbt = mb.MultiBlockTerminal(sims, extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=lead_s, end_s=built["sim_end_s"])
    return mbt, ann


@pytest.fixture(scope="module", params=[30, 300], ids=["load30", "load300"])
def stage(request):
    load = request.param
    t0 = time.perf_counter()
    prof, built = _build(load)
    t1 = time.perf_counter()
    mbt, ann = make_v5_terminal(prof, built)
    t2 = time.perf_counter()
    tw, tt = HT.to_terminal_world(prof, built, lead_s=LEAD_S, extra_review_epochs=ts.admission_epochs(ts.OBS_24H))
    t3 = time.perf_counter()
    jax.block_until_ready(tw)
    t4 = time.perf_counter()
    REPORT[f"load{load}"] = dict(build_s=round(t1 - t0, 3), v5_terminal_s=round(t2 - t1, 3),
                                 convert_s=round(t3 - t2, 3), convert_ready_s=round(t4 - t2, 3),
                                 blocks=tw.b, n_max=tt.n_max, c_max=tt.c_max, v_max=tt.v_max, p_cap=tt.p_cap,
                                 n_wake=tt.n_wake, n_review=tt.n_review, q_cap=tt.q_cap, log_cap=tt.log_cap,
                                 trucks=len(tt.truck_ids),
                                 n_leaves=len(jax.tree_util.tree_leaves(tw)))
    return dict(load=load, prof=prof, built=built, mbt=mbt, ann=ann, tw=tw, tt=tt)


def _v5_vessels(sim, vessel_ids) -> dict:
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


def _cols(o) -> dict[str, np.ndarray]:
    return {f: np.asarray(getattr(o, f)) for f in o._fields}


# ───────────────────────────────────────────────── ① 번호 규약
def test_block_and_row_numbering(stage):
    built, mbt, tw, tt = stage["built"], stage["mbt"], stage["tw"], stage["tt"]
    assert tt.block_ids == tuple(built["scenarios"]) == tuple(mbt.blocks) == tuple(sorted(mbt.blocks))
    assert tw.b == len(tt.block_ids) == 21
    sched = built["schedule"]
    for b, bid in enumerate(tt.block_ids):
        sim = mbt.blocks[bid]
        tb = tt.tables[b]
        trucks = [e["job_id"] for e in sched if e["block"] == bid]
        assert tt.n_trucks[b] == len(trucks) and tt.n_vessel_jobs[b] == len(sim.jobs)
        # 행 번호 = sorted(namespaced id) 순위 (트럭 ∪ 본선 작업) — 트럭이 앞
        assert list(tb.job_ids) == sorted(list(sim.jobs) + trucks)
        assert list(tb.job_ids[:len(trucks)]) == trucks == sorted(trucks)      # 명단 순 = 사전식
        assert tb.n0 == tt.n_used[b] == len(trucks) + len(sim.jobs)
        assert all(jid.startswith(f"{bid}:") for jid in tb.job_ids)
        # 반입 예비칸 이름 = IN_{namespaced id} (v5 engine.py:604)
        for n, jid in enumerate(tb.job_ids):
            assert tb.cont_ids[tb.c0 + n] == f"IN_{jid}"
        assert tb.cont_ids[:tb.c0] == tuple(sorted(sim.stacks.containers))
        assert tb.crane_ids == sim.fleet.ids() and tb.vessel_ids == tuple(sorted(sim.vessels))
    # 명단 s → (블록, 행) 이 번호표와 맞는다
    sb, sr = np.asarray(tw.sched.block), np.asarray(tw.sched.row)
    for s, e in enumerate(sched):
        b = int(sb[s])
        assert tt.block_ids[b] == e["block"] and tt.tables[b].job_ids[int(sr[s])] == e["job_id"] == tt.truck_ids[s]
    # 잎마다 앞에 (B,) — 모양 통일
    for path, leaf in jax.tree_util.tree_leaves_with_path(tw.blocks):
        assert leaf.shape[0] == tw.b, f"{jax.tree_util.keystr(path)} 의 앞 축이 B 가 아니다: {leaf.shape}"


# ───────────────────────────────────────────────── ② 격자·크레인
def test_grid_and_cranes_match_v5_reset(stage):
    mbt, tw, tt = stage["mbt"], stage["tw"], stage["tt"]
    for b, bid in enumerate(tt.block_ids):
        sim = mbt.blocks[bid]
        w = HT.block_slice(tw, b)
        d = from_block_world(w, tt.tables[b])
        assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}, f"{bid} piles"
        assert d["containers"] == {c: (x.bay, x.row, x.tier) for c, x in sim.stacks.containers.items()}, f"{bid} 좌표"
        for cid in sim.fleet.ids():
            yc, a = sim.fleet.get(cid), d["cranes"][cid]
            sp = sim.fleet.spec(cid)
            assert (a["position_bay"], a["trolley_row"], a["available_at"], a["assigned_job"], a["status"],
                    a["service_bay_min"], a["service_bay_max"]) == \
                   (yc.state.position_bay, yc.state.trolley_row, yc.state.available_at, yc.state.assigned_job,
                    yc.state.status.name, sp.service_bay_min, sp.service_bay_max), f"{bid} 크레인 {cid}"
        assert d["rail_order"] == sim._rail_order and d["idle_positions"] == sim.reservations.idle_positions()
        assert np.all(np.asarray(w.cranes.block) == b)
        assert d["clock"] == sim.clock == 0.0 and d["end"] == sim.end and not d["terminal"]
        assert d["violation"] == 0 and d["overflow"] == 0 and d["event_log"] == []


# ───────────────────────────────────────────────── ③ 사건 큐 시드
def test_queue_seed_matches_v5_heap(stage):
    mbt, tw, tt = stage["mbt"], stage["tw"], stage["tt"]
    for b, bid in enumerate(tt.block_ids):
        sim = mbt.blocks[bid]
        got = queue_entries(HT.block_slice(tw, b).queue, tt.tables[b])
        want = sorted(sim.queue._heap, key=lambda e: (e.time, e.priority, e.seq))
        assert len(got) == len(want), f"{bid}: 큐 길이 arr={len(got)} v5={len(want)}"
        for g, e in zip(got, want):
            assert (g[0], g[1], g[3], g[4]) == (e.time, e.priority, e.kind_name, e.payload), f"{bid}: {g} vs {e}"
        assert [g[2] for g in got] == [e.seq - 1 for e in want], f"{bid}: seq 순서"     # v5 seq 는 1 부터
        assert all(":" in g[4] for g in got if g[3] in ("JOB_RELEASED", "BLOCK_ARRIVAL")), f"{bid}: payload 미namespaced"


# ───────────────────────────────────────────────── ④ 오더 열
def test_orders_match_v5_jobs_and_truck_rows_are_absent(stage):
    built, mbt, tw, tt = stage["built"], stage["mbt"], stage["tw"], stage["tt"]
    sched = built["schedule"]
    sb, sr = np.asarray(tw.sched.block), np.asarray(tw.sched.row)
    for b, bid in enumerate(tt.block_ids):
        sim = mbt.blocks[bid]
        tb = tt.tables[b]
        o = _cols(HT.block_slice(tw, b).orders)
        cont_idx, ves_idx = tb.cont_index, tb.vessel_index
        assert int((o["block"] >= 0).sum()) == len(sim.jobs), f"{bid}: 있는 오더 수"
        assert bool(ES.ledger_mode(HT.block_slice(tw, b).orders)) == (sim.time_ledger is not None)
        assert not sim.time_ledger.records and np.all(np.isinf(o["gate_in_s"])), f"{bid}: 등록 전인데 gate_in 있음"
        for jid, j in sim.jobs.items():
            n = tb.job_index[jid]
            got = (int(o["block"][n]), int(o["status"][n]), int(o["flow"][n]), bool(o["is_external"][n]),
                   bool(o["is_vessel"][n]), bool(o["is_store"][n]), int(o["target_cont"][n]),
                   int(o["inbound_size"][n]), int(o["vessel"][n]), float(o["release_s"][n]),
                   float(o["provided_eta_s"][n]), float(o["deadline_s"][n]), float(o["exit_travel_s"][n]),
                   float(o["actual_arrival_s"][n]), int(o["inbound_cont"][n]))
            exp = (b, JS_PLANNED, ["GATE_IN", "GATE_OUT", "VESSEL_LOAD", "VESSEL_DISCHARGE", "TRANSSHIPMENT",
                                   "REHANDLE"].index(j.flow.value), j.is_external_truck, j.is_vessel_linked,
                   j.inbound_size is not None,
                   cont_idx[j.target_container] if j.target_container is not None else EMPTY_ID,
                   SIZE_INDEX[j.inbound_size.value] if j.inbound_size is not None else EMPTY_ID,
                   ves_idx[j.vessel_id] if j.vessel_id is not None else EMPTY_ID, float(j.release_time),
                   np.inf if j.provided_eta is None else float(j.provided_eta),
                   np.inf if j.deadline is None else float(j.deadline),
                   -1.0 if j.exit_travel_s is None else float(j.exit_travel_s),
                   np.inf if j.actual_block_arrival is None else float(j.actual_block_arrival),
                   tb.c0 + n if j.inbound_size is not None else EMPTY_ID)
            assert got == exp, f"{bid} {jid}: arr={got} v5={exp}"
        # 트럭 행 — 없는 오더이면서 정적 열은 v5 투입이 만들 Job 그대로
        for s, e in enumerate(sched):
            if e["block"] != bid:
                continue
            n = int(sr[s]); assert int(sb[s]) == b
            j = _job_from_entry(e, e["arrival_s"])
            assert int(o["block"][n]) == -1 and np.isinf(o["gate_in_s"][n]) and int(o["status"][n]) == JS_PLANNED
            got = (int(o["flow"][n]), bool(o["is_external"][n]), bool(o["is_store"][n]), int(o["target_cont"][n]),
                   int(o["inbound_size"][n]), int(o["inbound_cont"][n]), float(o["provided_eta_s"][n]),
                   float(o["exit_travel_s"][n]), float(o["actual_arrival_s"][n]), float(o["release_s"][n]),
                   float(o["travel_s"][n]), float(o["notice_s"][n]))
            exp = (FL_GATE_OUT if e["flow"] == "GATE_OUT" else FL_GATE_IN, True, j.inbound_size is not None,
                   cont_idx[j.target_container] if j.target_container is not None else EMPTY_ID,
                   SIZE_INDEX[j.inbound_size.value] if j.inbound_size is not None else EMPTY_ID,
                   tb.c0 + n if j.inbound_size is not None else EMPTY_ID, float(j.provided_eta),
                   float(j.exit_travel_s), float(j.actual_block_arrival), 0.0, float(e["travel_s"]),
                   max(0.0, e["arrival_s"] - LEAD_S))
            assert got == exp, f"{bid} 트럭 {e['job_id']}: arr={got} v5={exp}"
            if j.inbound_size is not None:
                assert int(np.asarray(HT.block_slice(tw, b).conts.c_size)[tb.c0 + n]) == SIZE_INDEX[j.inbound_size.value]


# ───────────────────────────────────────────────── ⑤ 배·이송·wake·검토 시각
def test_vessels_transfer_wake_review_match_v5(stage):
    mbt, tw, tt = stage["mbt"], stage["tw"], stage["tt"]
    E = tt.n_review
    for b, bid in enumerate(tt.block_ids):
        sim = mbt.blocks[bid]
        w = HT.block_slice(tw, b)
        d = from_block_world(w, tt.tables[b])
        assert d["vessels"] == _v5_vessels(sim, tt.tables[b].vessel_ids), f"{bid} 배"
        assert d["transfer"]["busy_until"] == list(sim.transfer.busy_until) and d["transfer"]["pending"] == []
        assert float(w.transfer.move_time_s) == sim.transfer.move_time_s
        ws, wj = np.asarray(w.wake.eta_wake_s), np.asarray(w.wake.eta_wake_job)
        got = [(float(ws[i]), tt.tables[b].job_ids[int(wj[i])]) for i in range(ws.shape[0]) if np.isfinite(ws[i])]
        assert got == list(sim._eta_wakes), f"{bid} ETA wake"
        rv = np.asarray(w.wake.review_s)
        nv = int(np.asarray(tw.epochs.n_valid)[b])
        assert rv.shape == (E,) and list(rv[:nv]) == list(sim.review_epochs) and np.all(np.isinf(rv[nv:])), f"{bid} review"
        assert int(w.wake.review_idx) == 0 and int(w.wake.wake_idx) == sim._wake_idx == 0
        assert d["cost_episode"] == {k: 0.0 for k in d["cost_episode"]} and d["kpi"]["rehandle_count"] == 0


# ───────────────────────────────────────────────── ⑥ 명단 · 검토 시각 == ScheduledAnnouncer · admission_epochs
def test_schedule_and_epochs_match_announcer(stage):
    built, mbt, ann, tw, tt = stage["built"], stage["mbt"], stage["ann"], stage["tw"], stage["tt"]
    sched = built["schedule"]
    ep_t = [float(x) for x in np.asarray(tw.epochs.t)]
    assert ep_t == list(ts.admission_epochs(ts.OBS_24H)) == [round(t, 6) for t in ts.admission_epochs(ts.OBS_24H)]
    assert all(bool(x) for x in np.asarray(tw.epochs.on_grid)) and float(tw.epochs.period_s) == ann.period_s
    assert list(np.asarray(tw.epochs.n_valid)) == [len(mbt.blocks[b].review_epochs) for b in tt.block_ids]
    ne = np.asarray(tw.sched.notify_epoch); tail = np.asarray(tw.sched.tail_skip)
    by_epoch = {}
    for s in range(len(sched)):
        assert ne[s] >= 0, f"명단 {s} 의 통지 시각이 검토 시각이 아니다"
        by_epoch.setdefault(round(ep_t[int(ne[s])], 6), []).append(sched[s]["job_id"])
    assert set(by_epoch) == set(ann.by_epoch)
    for key, entries in ann.by_epoch.items():
        assert by_epoch[key] == [e["job_id"] for e in entries], f"에폭 {key} 의 명단 순서"
    # 그 밖의 열 — 명단 값 그대로 · SKIP_TAIL 판정 · 통지 = max(0, A−lead)
    a = np.asarray(tw.sched.arrival_s); tr = np.asarray(tw.sched.travel_s); no = np.asarray(tw.sched.notice_s)
    bi = np.asarray(tw.sched.block_in_s); eta = np.asarray(tw.sched.eta_s); ex = np.asarray(tw.sched.exit_travel_s)
    fl = np.asarray(tw.sched.flow); tg = np.asarray(tw.sched.target_cont); sz = np.asarray(tw.sched.size)
    ic = np.asarray(tw.sched.inbound_cont); sb, sr = np.asarray(tw.sched.block), np.asarray(tw.sched.row)
    for s, e in enumerate(sched):
        j = _job_from_entry(e, e["arrival_s"])
        tb = tt.tables[int(sb[s])]
        assert (float(a[s]), float(tr[s]), float(bi[s]), float(eta[s]), float(ex[s])) == \
               (e["arrival_s"], e["travel_s"], j.actual_block_arrival, j.provided_eta, e["exit_travel_s"])
        assert float(no[s]) == max(0.0, e["arrival_s"] - LEAD_S)
        assert bool(tail[s]) == (e["arrival_s"] + e["travel_s"] > built["sim_end_s"])
        if e["flow"] == "GATE_OUT":
            assert int(fl[s]) == FL_GATE_OUT and tb.cont_ids[int(tg[s])] == e["target"] and int(sz[s]) == -1 and int(ic[s]) == -1
        else:
            assert int(fl[s]) == FL_GATE_IN and int(tg[s]) == -1 and int(sz[s]) == SIZE_INDEX[j.inbound_size.value]
            assert int(ic[s]) == tb.c0 + int(sr[s]) and tb.cont_ids[int(ic[s])] == f"IN_{e['job_id']}"
    assert not tail.any(), "이 무대에서는 SKIP_TAIL 이 없어야 한다 (정답: admitted == day_total)"


# ───────────────────────────────────────────────── ⑦ 원장 초기값 · from_terminal_world
def test_per_truck_lead_matches_v3announcer():
    """★통지 리드를 **트럭마다** 주면 `to_terminal_world` 의 통지 시각·에폭 버킷이 v3 `V3Announcer` 와 같은가.

    왜: v2 `ScheduledAnnouncer` 는 전원 같은 리드라 "누구를 더 일찍 알았나" 축이 없다. v3 하루 무대
    (`stage/orders.build_stage`)는 트럭마다 `sample_lead_s` 로 리드를 뽑아 `e["lead_s"]` 에 넣는데, 배열 변환기가
    스칼라 리드만 받으면 그 축이 통째로 사라진다(통지 에폭 버킷과 notice_s 가 전부 틀어진다). 여기서는
    ① 명단의 `e["lead_s"]` 를 정본으로 쓰는지 ② (S,) 열로 직접 줘도 같은지 ③ `retarget`/`resolve_entry` 훅은
    **fail-loud 로 거절**하는지(조용히 무시하면 SKIP 사유 NO_TARGET·CONTAINER_ID_CHANGED 가 사라진다) 를 본다.
    작은 무대(Y01+Y21 · 2시간)로 — 호스트 변환만 재므로 빠르다.
    """
    import random

    from yard_rl.v6.stage.orders import V3Announcer

    obs = ts.ObservationContract(warmup_s=0.0, measure_s=7_200.0, snapshot_s=300.0)
    prof, layout = pr.build_h21_profile(), yl.terminal_layout().subset(("Y01", "Y21"))
    built = ts.build_diurnal(prof, SEED, obs=obs, layout=layout, params=ts.TerminalStreamParams(load_4h=40),
                             day_total=40, n_streams=0, drain_s=1_200.0, background_seed=SEED)
    rng = random.Random(f"v3:lead:ht:{SEED}")
    leads = []
    for e in built["schedule"]:
        e["lead_s"] = min(float(ts.sample_lead_s(rng.random())), float(e["arrival_s"]))
        leads.append(e["lead_s"])
    assert len({round(x, 6) for x in leads}) >= 10, "리드가 거의 같다 — 무대가 축을 못 만든다"
    ann = V3Announcer(built["schedule"], end_s=built["sim_end_s"], period_s=60.0)

    # ① 명단의 lead_s 가 정본 — 스칼라를 줘도 무시하고 항목별 값을 쓴다
    tw, tt = HT.to_terminal_world(prof, built, lead_s=0.0, extra_review_epochs=ts.admission_epochs(obs))
    assert [round(x, 6) for x in tt.lead_of] == [round(x, 6) for x in leads]
    ep_t = [float(x) for x in np.asarray(tw.epochs.t)]
    ne = np.asarray(tw.sched.notify_epoch); no = np.asarray(tw.sched.notice_s)
    ep_set = set(ep_t)
    by_epoch: dict[float, list[str]] = {}
    off_grid: list[str] = []
    for i, e in enumerate(built["schedule"]):
        nt = max(0.0, e["arrival_s"] - e["lead_s"])
        assert float(no[i]) == nt, f"명단 {i} 통지 시각"
        slot = round((nt // 60.0) * 60.0, 6)                    # V3Announcer 117행
        if int(ne[i]) < 0:
            # 통지 슬롯이 검토 시각 목록 밖 — v5 도 그 시각에 review 를 열지 않아 영원히 투입되지 않는다
            assert slot not in ep_set, f"명단 {i} 슬롯 {slot} 은 검토 시각인데 notify_epoch 가 -1"
            off_grid.append(e["job_id"])
            continue
        assert round(ep_t[int(ne[i])], 6) == slot, f"명단 {i} 통지 에폭 {ep_t[int(ne[i])]} ≠ 슬롯 {slot}"
        by_epoch.setdefault(slot, []).append(e["job_id"])
    exp = {k: [e["job_id"] for e in v] for k, v in ann.by_epoch.items() if k in ep_set}
    assert by_epoch == exp, f"버킷 arr={sorted(by_epoch)} v5={sorted(exp)}"
    assert len(by_epoch) >= 5, f"검토 시각에 든 버킷이 {len(by_epoch)}개뿐 — 무대가 약하다"

    # ② (S,) 열로 직접 줘도 같다 (명단에서 lead_s 를 뺀 판)
    bare = dict(built)
    bare["schedule"] = [{k: v for k, v in e.items() if k != "lead_s"} for e in built["schedule"]]
    tw2, tt2 = HT.to_terminal_world(prof, bare, lead_s=np.asarray(leads), extra_review_epochs=ts.admission_epochs(obs))
    assert np.array_equal(np.asarray(tw2.sched.notice_s), no)
    assert np.array_equal(np.asarray(tw2.sched.notify_epoch), ne)
    # 스칼라 하나면 전원 같은 리드 (v2 경로와 비트 동일)
    tw3, _ = HT.to_terminal_world(prof, bare, lead_s=LEAD_S, extra_review_epochs=ts.admission_epochs(obs))
    assert [float(x) for x in np.asarray(tw3.sched.notice_s)] == \
           [max(0.0, float(e["arrival_s"]) - LEAD_S) for e in bare["schedule"]]

    # ③ V3Announcer 전용 훅은 거절한다 (미이식 — 조용히 무시 금지)
    for kw in ({"retarget": lambda *a: None}, {"resolve_entry": lambda e: e}):
        with pytest.raises(ValueError, match="V3Announcer"):
            HT.to_terminal_world(prof, bare, lead_s=LEAD_S, extra_review_epochs=ts.admission_epochs(obs), **kw)
    REPORT["per_truck_lead"] = dict(trucks=len(leads), distinct_leads=len({round(x, 6) for x in leads}),
                                    epochs=len(ep_t), buckets=len(by_epoch))


def test_ledger_initial_and_from_terminal_world(stage):
    mbt, tw, tt = stage["mbt"], stage["tw"], stage["tt"]
    L = tw.ledger
    S = len(tt.truck_ids)
    assert not np.asarray(L.registered).any() and np.all(np.isinf(np.asarray(L.a_gate_in)))
    assert np.array_equal(np.asarray(L.owner), np.asarray(tw.sched.block)) and np.array_equal(np.asarray(L.row), np.asarray(tw.sched.row))
    assert np.array_equal(np.asarray(L.origin), np.asarray(tw.sched.block))
    assert np.asarray(L.transfer_src).shape == (S, tt.max_transfers) and int(L.txn_seq) == 0 and float(L.route_cost_s) == 0.0
    assert np.asarray(L.reserved_inbound).shape == (tw.b,) and not np.asarray(L.reserved_inbound).any()
    assert int(tw.epoch_idx) == 0
    d = HT.from_terminal_world(tw, tt)
    assert d["admitted"] == 0 == d["n_turns"] and d["turn_samples_s"] == [] and d["turn_sum_s"] == 0.0
    assert d["terminal_total"] == 0.0 and d["route_cost_s"] == 0.0 and d["end"] == max(s.end for s in mbt.blocks.values())
    assert list(d["blocks"]) == list(tt.block_ids)
    for bid, sim in mbt.blocks.items():
        blk = d["blocks"][bid]
        assert blk["n_jobs"] == len(sim.jobs) and blk["n_cranes"] == len(list(sim.fleet.all()))
        assert blk["event_hash"] == sim.event_stream_hash() and blk["n_events"] == 0 and blk["events"] == []
        assert blk["unfinished"] == sim.unfinished_backlog() and blk["deadlock_escapes"] == 0
        assert set(blk["jobs"]) == set(sim.jobs)
        assert blk["end_s"] == sim.end and blk["clock_s"] == 0.0
        assert set(blk["kpis"]) == {"queue_area_s", "tail_area_s", "loaded_gantry_m", "empty_gantry_m", "rehandle_count",
                                    "pre_rehandle_count", "completed_external", "completed_vessel", "vessel_delay_s",
                                    "positioning_count"}
        assert blk["cost_raw"] == sim.cost.episode_raw()
    # 행 → id (원장 기준) 가 정적 표와 같다 (아직 이송 없음)
    for b in range(tw.b):
        ids = HT.block_row_ids(tw, tt, b)
        assert ids[:tt.n_used[b]] == tt.tables[b].job_ids and all(x is None for x in ids[tt.n_used[b]:])
        assert HT.block_tables(tw, tt, b).job_ids == tt.tables[b].job_ids


# ───────────────────────────────────────────────── ⑧ numpy 왕복
def test_numpy_roundtrip(stage, tmp_path):
    tw = stage["tw"]
    path = tmp_path / "tw.npz"
    HT.save_terminal_world(path, tw)
    back = HT.load_terminal_world(path, tw)
    la, lb = jax.tree_util.tree_leaves(tw), jax.tree_util.tree_leaves(back)
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(tw)]
    bad = [names[i] for i, (x, y) in enumerate(zip(la, lb))
           if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]
    assert not bad, f"왕복 뒤 다른 잎 {bad}"
    with pytest.raises(ValueError):
        HT.terminal_world_from_numpy(tw, {k: v for k, v in list(HT.terminal_world_to_numpy(tw).items())[:-1]})


# ───────────────────────────────────────────────── ⑨ 미리 채운 트럭 행이 엔진에 보이지 않는다 (Y01 · load300)
def _load_dump_script():
    path = os.path.join(_ROOT, "scripts", "v6", "dump_ground_truth.py")
    spec = importlib.util.spec_from_file_location("_dgt_for_host_terminal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), (b[len(a)] if len(b) > len(a) else None)
    return None


def test_blank_truck_rows_are_invisible_to_block_engine(stage):
    """터미널 세계의 Y01 조각(트럭 행 15개가 '없는 오더'로 있음)을 블록 엔진으로 끝까지 굴리면, 같은 namespaced
    시나리오를 트럭 없이 굴린 v5(빈 장부 = ensure_time_ledger)와 사건 하나까지 같다 — 트럭 행이 후보·적분·장부에 새지 않는다."""
    if stage["load"] != 300:
        pytest.skip("load300 에서만 (Y01 = 본선 240 · 트럭 15)")
    prof, built, tw, tt = stage["prof"], stage["built"], stage["tw"], stage["tt"]
    b = tt.block_index["Y01"]
    scn = built["scenarios"]["Y01"]
    scn_ns = dataclasses.replace(scn, jobs=[dataclasses.replace(j, job_id=f"Y01:{j.job_id}") for j in scn.jobs])
    sim = ts.ensure_time_ledger(_sim(prof, scn_ns))
    D = _load_dump_script()
    pol = D.make_policy("sf_spt", LVL)
    n_dec = 0
    while (dp := sim.run_until_decision()) is not None:
        n_dec += 1
        pol(sim, dp)
    w0, tb = HT.block_slice(tw, b), tt.tables[b]
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    policy_fn, params = DP.make_resolver("sf_spt", g, count_lost=False), DP.resolver_params(tb, g)
    s_max = 8 * tt.n_max + 256
    t0 = time.perf_counter()
    w, trace = ES.run_jit(w0, params, g, policy_fn, s_max, True, True, hz, True)
    jax.block_until_ready(w)
    secs = time.perf_counter() - t0
    d = from_block_world(w, tb)
    arlog = [(round(t, 6), k, p) for (t, k, p) in d["event_log"]]
    v5log = [(round(t, 6), k, p) for (t, k, p) in sim.event_log]
    diff = _first_diff(arlog, v5log)
    if diff is not None:
        i, x, y = diff
        ctx = "\n".join(f"    #{j}: arr={arlog[j] if j < len(arlog) else '-'}  |  v5={v5log[j] if j < len(v5log) else '-'}"
                        for j in range(max(0, i - 4), min(max(len(arlog), len(v5log)), i + 4)))
        pytest.fail(f"⑨ 사건 로그가 {i}번째에서 갈린다: arr={x} v5={y}\n{ctx}\n  (arr {len(arlog)} · v5 {len(v5log)} · "
                    f"violation={d['violation_names']} overflow={d['overflow']} terminal={d['terminal']})")
    assert event_stream_hash(w, tb) == sim.event_stream_hash()
    assert d["violation"] == 0 and d["overflow"] == 0 and d["terminal"]
    assert d["cost_episode"] == sim.cost.episode_raw()
    ks = sim.kpis
    assert (d["kpi"]["rehandle_count"], d["kpi"]["completed_vessel"], d["kpi"]["completed_external"],
            d["kpi"]["queue_area_s"], d["kpi"]["vessel_delay_s"]) == \
           (ks.rehandle_count, ks.completed_vessel, ks.completed_external, ks.queue_area_s, ks.vessel_delay_s)
    assert d["ledger"]["closed_end_s"] == sim.time_ledger.closed_end_s == sim.end
    n_steps = int(w.steps)
    assert int(np.asarray(trace.decided)[:n_steps].sum()) == n_dec
    # 트럭 행은 끝까지 없는 오더로 남는다
    blk = np.asarray(w.orders.block)
    assert np.all(blk[:tt.n_trucks[b]] == -1) and np.all(np.isinf(np.asarray(w.orders.gate_in_s)[:tt.n_trucks[b]]))
    REPORT["y01_blank_rows"] = dict(events=len(arlog), decisions=n_dec, steps=n_steps, hash=sim.event_stream_hash(),
                                    run_s=round(secs, 2), device=str(jax.devices()[0].platform))


# ───────────────────────────────────────────────── ⑩ 원장 갱신 뒤 (투입·이송 흉내) 행→id 사상 · 턴타임 표본
def test_row_ids_and_turn_samples_follow_ledger(stage):
    """조정자가 원장을 갱신했을 때 from_terminal_world 가 v5 `a_to_o_samples_s`·행→id 규약대로 읽는가 — 호스트 논리만.
    트럭 s0: 최초 블록에 투입·완료(O 있음) → O−A. 트럭 s1: 투입만(O 없음) → max(0, observe−A).
    트럭 s2: 다른 블록의 여분 행으로 이송(owner·row 갱신) → 옛 행은 None, 새 행에 id·IN_ 이름."""
    prof, built = stage["prof"], stage["built"]
    tw, tt = HT.to_terminal_world(prof, built, lead_s=LEAD_S, extra_review_epochs=ts.admission_epochs(ts.OBS_24H), n_spare=2)
    assert tt.n_max == max(tt.n_used) + 2
    sb, sr, arr = np.asarray(tw.sched.block), np.asarray(tw.sched.row), np.asarray(tw.sched.arrival_s)
    s0, s1, s2 = 0, 1, 2
    b0, r0 = int(sb[s0]), int(sr[s0])
    b2_src, r2 = int(sb[s2]), int(sr[s2])
    b2_dst = (b2_src + 1) % tw.b
    spare = tt.n_used[b2_dst]                                            # 수신 블록의 첫 여분 행
    L = tw.ledger
    reg = np.asarray(L.registered).copy(); reg[[s0, s1, s2]] = True
    a = np.asarray(L.a_gate_in).copy(); a[[s0, s1, s2]] = arr[[s0, s1, s2]]
    owner = np.asarray(L.owner).copy(); row = np.asarray(L.row).copy()
    owner[s2], row[s2] = b2_dst, spare
    L2 = L._replace(registered=jnp.asarray(reg), a_gate_in=jnp.asarray(a), owner=jnp.asarray(owner), row=jnp.asarray(row),
                    transfer_count=L.transfer_count.at[s2].set(1), version=L.version.at[s2].set(1))
    o = tw.blocks.orders
    O0 = float(arr[s0]) + 1234.5
    blk = o.block.at[b0, r0].set(b0).at[b2_src, r2].set(-1).at[b2_dst, spare].set(b2_dst)
    o2 = o._replace(block=blk, gate_in_s=o.gate_in_s.at[b0, r0].set(float(arr[s0])).at[b2_dst, spare].set(float(arr[s2])),
                    gate_out_s=o.gate_out_s.at[b0, r0].set(O0))
    tw2 = tw._replace(ledger=L2, blocks=tw.blocks._replace(orders=o2))
    ids_src, ids_dst = HT.block_row_ids(tw2, tt, b2_src), HT.block_row_ids(tw2, tt, b2_dst)
    assert ids_src[r2] is None and ids_dst[spare] == tt.truck_ids[s2]
    assert ids_dst[:tt.n_used[b2_dst]] == tt.tables[b2_dst].job_ids
    tbd = HT.block_tables(tw2, tt, b2_dst)
    assert tbd.job_ids[spare] == tt.truck_ids[s2] and tbd.cont_ids[tbd.c0 + spare] == f"IN_{tt.truck_ids[s2]}"
    assert tbd.n0 == spare + 1 and tbd.job_ids[tt.n_used[b2_dst]:spare] == ()
    d = HT.from_terminal_world(tw2, tt)
    assert d["admitted"] == 3 and d["n_turns"] == 3
    want = sorted([O0 - float(arr[s0]), max(0.0, tt.observe_s - float(arr[s1])), max(0.0, tt.observe_s - float(arr[s2]))])
    assert d["turn_samples_s"] == [round(t, 6) for t in want] and d["turn_sum_s"] == round(sum(want), 6)
    assert tt.truck_ids[s0] in d["blocks"][tt.block_ids[b0]]["jobs"]
    assert tt.truck_ids[s2] in d["blocks"][tt.block_ids[b2_dst]]["jobs"] and tt.truck_ids[s2] not in d["blocks"][tt.block_ids[b2_src]]["jobs"]
    assert d["blocks"][tt.block_ids[b2_dst]]["n_jobs"] == tt.n_vessel_jobs[b2_dst] + 1


# ───────────────────────────────────────────────── 보고
def test_zz_report(capsys):
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    with capsys.disabled():
        print("\n[host_terminal report]")
        for k, r in REPORT.items():
            print(f"  {k:14s} {r}")
    for k, r in REPORT.items():
        if k.startswith("load"):
            assert r["convert_ready_s"] < 60.0, f"{k}: 변환 {r['convert_ready_s']}s — 너무 느리다"
