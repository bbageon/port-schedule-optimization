"""게이트 투입 동등성 — `gpu/admission.py` 가 v5 `ScheduledAnnouncer.review` + `admit_external_job` 과 **같은 답**을 낸다
([[YR-327]] 조각 6 · key=admit).

■ 방법 (기대값 손기입 없음 — v5 를 실제로 굴려 에폭마다 가로챈다)
  v5: `MultiBlockTerminal(ensure_time_ledger(TerminalSimulator) ×B, extra_review_epochs=admission_epochs(obs))` 를
      SF_SPT 규칙 정책으로 `run(review_fn=…)` — review_fn 을 감싸 검토 시각마다
        ① v5 블록 상태(시계·오더 status·스택·큐)를 배열 세계에 **이식**하고 (투입 이전 상태 동기화)
        ② v5 `ann.review(mbt, t)` 를 그대로 돌린 뒤
        ③ 배열판 `admit_epoch_jit(W, sched, slot)` 을 돌려
        ④ 그 에폭의 ADMIT/SKIP/SKIP_TAIL 순열(원장 행) · free_slots(전 블록, 투입 전·후) · 큐 전체(시각·종류·대상·seq) ·
           오더 열(block·gate_in·arrival·provided_eta·notice·travel·exit·flow·규격·대상) · 시간 장부 A · 전역 원장(owner·
           origin·flow·a_gate_in) 을 `==` 로 대조한다. 어긋나면 **처음 갈리는 에폭·블록·항목**을 보고한다.
  배열판 투입 결과는 에폭 사이에 **그대로 이어진다** (오더 열·큐는 v5 로 덮어쓰지 않고, 엔진이 바꾸는 열(status·스택·큐)만
  이식) — 그래서 SKIP_DUP(이미 등록) 도 배열 자신의 이전 투입에서 판정된다.

■ 무대 (명세 equivalence_test)
  build_diurnal(build_h21_profile(), seed 9_900_777, obs 2h(0/7200/300), n_streams 0, drain 1200, background_seed 동일)
    Y01 · 20대 · lead 600      Y01 · 60대 · lead 1800      Y01+Y21 · 20대 · lead 600      Y01+Y21 · 60대 · lead 1800
  + 가지 (v5 도 같은 무대를 받는다):
    cap    Y01+Y21 · 60 · 1800 · capacity_margin 을 초기 여유−3 으로 → 같은 에폭 안 4번째 반입부터 SKIP_CAP (순차 의존)
    edge   Y01+Y21 · 60 · 1800 · 명단 변형: 첫 항목 중복(SKIP_DUP) · 반출 대상 'NOPE'(SKIP_TARGET) · exit_travel None(SKIP_EXIT)
    tail   Y01 · 60 · 600 · drain 60초 → 하루 끝 도착이 sim_end 를 넘어 SKIP_TAIL
    time   tail 과 같되 announcer end_s=None → 같은 항목이 admit 층의 '투입시각 무효'(SKIP_TIME)
    leadmix ★트럭마다 다른 통지 리드 (v3 `stage/orders.V3Announcer`) · Y01+Y21 · 60 · margin 을 초기 여유−3 으로.
           v2 `ScheduledAnnouncer` 는 전원 같은 리드라 "누구를 더 일찍 알았나" 축이 없다 — 여기서는 명단의
           `e["lead_s"]` 를 트럭마다 뽑아 통지 에폭 버킷이 제각각이 되게 하고, 그 상태에서도 같은 (에폭, 블록)
           칸에 여러 대가 몰려 **뒷대가 용량으로 거절**되는지(순차 의존)까지 v5 와 맞춘다.
  한 무대 ≈ 10~20초 (컴파일 포함) — 80초 창에 두세 무대.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import admission as AD                                          # noqa: E402
from yard_rl.v6.gpu.events import EMPTY_ID, EMPTY_TIME, PRIO, empty_queue           # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import EV_NAMES, FLOW_NAMES, STATUS_NAMES, to_block_world   # noqa: E402
from yard_rl.v6.gpu.stack_ops import from_v5_stacks                                # noqa: E402
from yard_rl.v6.gpu.state import SZ_FT20, SZ_FT40                                  # noqa: E402
from yard_rl.v6.world.domain.enums import InformationLevel                         # noqa: E402
from yard_rl.v6.world.integrated import baselines as bl                            # noqa: E402
from yard_rl.v6.world.integrated import candidates as cd                           # noqa: E402
from yard_rl.v6.world.integrated import engine as eng                              # noqa: E402
from yard_rl.v6.world.integrated import multiblock as mb                           # noqa: E402
from yard_rl.v6.world.integrated import policy_config as pc                        # noqa: E402
from yard_rl.v6.world.integrated import profiles as pr                             # noqa: E402
from yard_rl.v6.world.integrated import terminal_stream as ts                      # noqa: E402
from yard_rl.v6.world.integrated import yard_layout as yl                          # noqa: E402
from yard_rl.v6.world.integrated.time_grid import on_grid                          # noqa: E402

SEED = 9_900_777
OBS = ts.ObservationContract(warmup_s=0.0, measure_s=7_200.0, snapshot_s=300.0)
LVL = InformationLevel.PRE_ADVICE
STATUS_IDX = {n: i for i, n in enumerate(STATUS_NAMES)}
SIZE_OF = {"FT20": SZ_FT20, "FT40": SZ_FT40}
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── 무대
@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    blocks: tuple[str, ...]
    load: int
    lead: float
    margin: int | None = None        # None = v5 기본 CAPACITY_MARGIN
    drain_s: float = 1_200.0
    end_none: bool = False           # announcer end_s=None
    mutate: str = ""                 # "edge" → 명단 변형
    announcer: str = "v2"            # "v2" = ScheduledAnnouncer(전원 같은 리드) · "v3" = V3Announcer(트럭별 리드)


STAGES = [
    Stage("y01-20-600", ("Y01",), 20, 600.0),
    Stage("y01-60-1800", ("Y01",), 60, 1_800.0),
    Stage("y01y21-20-600", ("Y01", "Y21"), 20, 600.0),
    Stage("y01y21-60-1800", ("Y01", "Y21"), 60, 1_800.0),
    Stage("cap", ("Y01", "Y21"), 60, 1_800.0, margin=-1),     # -1 = 초기 여유 − 3 (아래서 계산)
    Stage("edge", ("Y01", "Y21"), 60, 1_800.0, mutate="edge"),
    Stage("tail", ("Y01",), 60, 600.0, drain_s=60.0),
    Stage("time", ("Y01",), 60, 600.0, drain_s=60.0, end_none=True),
    Stage("leadmix", ("Y01", "Y21"), 60, 1_800.0, margin=-1, announcer="v3"),
]
_IDS = [s.name for s in STAGES]


def _mutate(schedule: list[dict], how: str) -> dict:
    """명단 변형 — v5 와 배열판이 **같은** 변형 명단을 받는다. 반환: 무엇을 바꿨나."""
    if how != "edge":
        return {}
    outs = [i for i, e in enumerate(schedule) if e["flow"] == "GATE_OUT"]
    ins = [i for i, e in enumerate(schedule) if e["flow"] == "GATE_IN"]
    assert outs and len(ins) >= 2
    schedule[outs[0]]["target"] = "NOPE"              # 반출 대상 부재 → SKIP_TARGET
    schedule[ins[1]]["exit_travel_s"] = None          # exit_travel 결측 → SKIP_EXIT
    schedule.append(dict(schedule[0]))                # 같은 작업 두 번 → 두 번째는 SKIP_DUP (같은 버킷 끝)
    return {"target": schedule[outs[0]]["job_id"], "exit": schedule[ins[1]]["job_id"], "dup": schedule[0]["job_id"]}


def _rule_policy():
    """SF_SPT 규칙 + LEGACY 후보 (test_world_equivalence.py:49-54 · dump_ground_truth._rule_policy)."""
    gens: dict[int, object] = {}
    pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LVL) for c in dp.crane_ids}
        bl._apply(sim, pol.decide(sim, dp, gb))
    return exec_policy


def _sim(prof, scn):
    s = eng.TerminalSimulator(prof, scn, check_invariants=True)
    s.info_level = LVL
    return ts.ensure_time_ledger(s)


def _with_truck_leads(schedule: list[dict], seed_tag: str) -> None:
    """명단에 **트럭별 통지 리드**를 심는다 — v3 `stage/orders.build_stage` 62-69행과 같은 규칙.

    리드는 시드에서 미리 뽑는다(런타임 난수를 쓰면 같은 시드가 같은 하루를 못 만든다). 통지가 창 시작보다
    앞설 수 없으므로 도착으로 눌러 담는다. v5 `V3Announcer` 와 배열판이 **같은 명단**을 받는다.
    """
    import random
    rng = random.Random(f"v3:lead:{seed_tag}")
    for e in schedule:
        e["lead_s"] = min(float(ts.sample_lead_s(rng.random())), float(e["arrival_s"]))


def _build(stage: Stage):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout().subset(stage.blocks)
    built = ts.build_diurnal(prof, SEED, obs=OBS, layout=layout, params=ts.TerminalStreamParams(load_4h=stage.load),
                             day_total=stage.load, n_streams=0, drain_s=stage.drain_s, background_seed=SEED)
    schedule = [dict(e) for e in built["schedule"]]
    changed = _mutate(schedule, stage.mutate)
    if stage.announcer == "v3":
        _with_truck_leads(schedule, f"{stage.name}:{SEED}")
    return prof, layout, built, schedule, changed


# ───────────────────────────────────────────────── v5 → 배열 이식 (검토 시각의 블록 상태)
def _table_cont_id(cid: str, bid: str, cont_index: dict, row_of_job: dict) -> str:
    """v5 컨테이너 id → 번호표 이름. 시나리오 작업의 반입칸 'IN_Y01:J-…' 는 개명 전 'IN_J-…' (multiblock._namespace_jobs);
    명단 트럭의 'IN_Y01:D-00005' 는 그대로 (명단 id 가 원래 접두 포함)."""
    if cid in cont_index:
        return cid
    pre = f"IN_{bid}:"
    if cid.startswith(pre) and f"IN_{cid[len(pre):]}" in cont_index:
        return f"IN_{cid[len(pre):]}"
    raise KeyError(f"번호표에 없는 컨테이너 {cid}")


def _row_of(jid: str, bid: str, tb, row_of_job: dict) -> int:
    if jid in row_of_job:
        return row_of_job[jid]
    p = jid[len(bid) + 1:] if jid.startswith(bid + ":") else jid
    return tb.job_index[p]


def _target_of(kind_name: str, payload: str, bid: str, tb, row_of_job: dict) -> int:
    if kind_name in ("BLOCK_ARRIVAL", "JOB_RELEASED", "VESSEL_RELEASED", "ETA_UPDATED"):
        return _row_of(payload, bid, tb, row_of_job)
    if kind_name in ("JOB_COMPLETED", "EQUIPMENT_DOWN", "EQUIPMENT_UP"):
        return tb.crane_index.get(payload, EMPTY_ID)
    if kind_name in ("TRANSFER_ARRIVE", "STS_MOVE", "VESSEL_START", "PLAN_CHANGE"):
        return tb.vessel_index.get(payload, EMPTY_ID)
    return EMPTY_ID                                                        # HORIZON


def _v5_queue_rows(sim, bid, tb, row_of_job) -> list[tuple]:
    """v5 힙 → (시각, 종류 번호, 대상 번호, seq) 정렬 목록."""
    return sorted((e.time, EV_NAMES.index(e.kind_name), _target_of(e.kind_name, e.payload, bid, tb, row_of_job), e.seq)
                  for e in sim.queue._heap)


def _array_queue_rows(q) -> list[tuple]:
    """배열 큐 → 같은 모양 (seq 는 v5 가 1 부터라 +1)."""
    t, k, tg, sq = (np.asarray(q.time), np.asarray(q.kind), np.asarray(q.target), np.asarray(q.seq))
    live = np.isfinite(t)
    return sorted((float(t[i]), int(k[i]), int(tg[i]), int(sq[i]) + 1) for i in np.nonzero(live)[0])


def _queue_from_v5(sim, bid, tb, row_of_job, capacity: int):
    """v5 힙을 배열 큐로 (칸 i 에 힙의 i 번째 항목 · seq−1 · counter = v5 _seq)."""
    q = empty_queue(capacity)
    heap = list(sim.queue._heap)
    assert len(heap) <= capacity, f"큐 칸 {capacity} < v5 힙 {len(heap)}"
    t = np.full((capacity,), EMPTY_TIME, np.float64)
    k = np.full((capacity,), EMPTY_ID, np.int32)
    tg = np.full((capacity,), EMPTY_ID, np.int32)
    sq = np.full((capacity,), EMPTY_ID, np.int32)
    for i, e in enumerate(heap):
        t[i], k[i], sq[i] = e.time, EV_NAMES.index(e.kind_name), e.seq - 1
        tg[i] = _target_of(e.kind_name, e.payload, bid, tb, row_of_job)
    return q._replace(time=jnp.asarray(t), kind=jnp.asarray(k), target=jnp.asarray(tg), seq=jnp.asarray(sq),
                      counter=jnp.int32(sim.queue._seq))


def _transplant(w, sim, bid, tb, row_of_job, g, t):
    """검토 시각 t 의 v5 블록 상태를 배열 세계 w 에 이식 — 엔진이 바꾸는 열만 (시계·오더 status·스택·컨테이너·큐).
    투입이 채우는 열(block·gate_in·…)과 큐의 투입 push 는 배열 자신의 이전 결과가 그대로 이어진다."""
    status = np.asarray(w.orders.status).copy()
    for jid, j in sim.jobs.items():
        status[_row_of(jid, bid, tb, row_of_job)] = STATUS_IDX[j.status.name]
    cont_index = tb.cont_index
    v5_names = []
    for cid in tb.cont_ids:
        if cid.startswith("IN_") and cid[3:] in tb.job_index and cid[3:] not in row_of_job:
            v5_names.append(f"IN_{bid}:{cid[3:]}")                        # 시나리오 작업 반입칸 → v5 는 접두 개명

        else:
            v5_names.append(cid)
    for cid in sim.stacks.containers:
        _table_cont_id(cid, bid, cont_index, row_of_job)                   # 번호표에 있어야 한다 (없으면 KeyError)
    stacks, conts, _ = from_v5_stacks(sim.stacks, g, cont_ids=v5_names, n_cont=len(v5_names))
    old_size = np.asarray(w.conts.c_size)
    new_size = np.asarray(conts.c_size)
    keep = np.where(np.asarray(conts.c_alive), new_size, old_size)         # 예비칸 규격은 미리 앉힌 값 유지
    conts = conts._replace(c_size=jnp.asarray(keep))
    return w._replace(clock=jnp.asarray(float(t), jnp.float64),
                      orders=w.orders._replace(status=jnp.asarray(status)),
                      stacks=stacks, conts=conts,
                      queue=_queue_from_v5(sim, bid, tb, row_of_job, w.queue.capacity))


# ───────────────────────────────────────────────── 대조
def _check_orders(W, bi, bid, sim, mbt, tb, row_of_job, sched_of_job, lead, label):
    """오더 열 · 시간 장부 · 전역 원장 == v5 (그 블록의 명단 트럭 전부)."""
    w = AD.world_at(W, bi)
    o = {f: np.asarray(getattr(w.orders, f)) for f in w.orders._fields}
    mine = {n for n in range(o["block"].shape[0]) if o["block"][n] >= 0 and tb.job_ids[n] in sched_of_job}
    v5 = {row_of_job[jid] for jid in sim.jobs if jid in row_of_job}
    assert mine == v5, f"[{label}] 등록 집합: arr-only={sorted(mine - v5)} v5-only={sorted(v5 - mine)}"
    tl = sim.time_ledger
    for jid, j in sim.jobs.items():
        if jid not in row_of_job:
            continue
        n = row_of_job[jid]
        e = sched_of_job[jid]
        got = (int(o["block"][n]), float(o["gate_in_s"][n]), float(o["actual_arrival_s"][n]),
               float(o["provided_eta_s"][n]), float(o["travel_s"][n]), float(o["exit_travel_s"][n]),
               FLOW_NAMES[int(o["flow"][n])], int(o["inbound_size"][n]),
               tb.cont_ids[int(o["target_cont"][n])] if o["target_cont"][n] >= 0 else None,
               float(o["notice_s"][n]), STATUS_NAMES[int(o["status"][n])])
        my_lead = float(e["lead_s"]) if e.get("lead_s") is not None else lead      # 트럭별 리드 (v3) 우선
        exp = (bi, j.actual_gate_in, j.actual_block_arrival, j.provided_eta, e["travel_s"],
               (-1.0 if j.exit_travel_s is None else j.exit_travel_s), j.flow.value,
               (SIZE_OF[j.inbound_size.value] if j.inbound_size is not None else -1), j.target_container,
               max(0.0, e["arrival_s"] - my_lead), j.status.name)
        assert got == exp, f"[{label}] 오더 {jid} (행 {n}): arr={got} v5={exp}"
        assert j.estimated_block_arrival == j.provided_eta and j.appointment_gate_time == j.actual_gate_in
        # ★예약 원점 열 — v5 `Job.appointment_gate_time`. 통지(notice_s)와 **다른 값**이어야 한다 (lead > 0 이면).
        assert float(o["appt_s"][n]) == float(j.appointment_gate_time), \
            f"[{label}] 오더 {jid} 예약 원점 arr={float(o['appt_s'][n])} v5={j.appointment_gate_time}"
        assert tl.records[jid].gate_in == float(o["gate_in_s"][n]), f"[{label}] 시간 장부 A {jid}"
        rec = mbt.ledger.records[jid]
        assert (rec.owner, rec.origin_block, rec.flow, rec.a_gate_in, rec.version, rec.transfer_count) == \
            (bid, bid, j.flow.value, float(o["gate_in_s"][n]), 0, 0), f"[{label}] 원장 {jid}: {rec}"
        assert bool(o["is_external"][n]) and not bool(o["is_vessel"][n]) and bool(o["is_store"][n]) == (j.inbound_size is not None)
        if j.inbound_size is not None:
            assert tb.cont_ids[int(o["inbound_cont"][n])] == f"IN_{jid}"
            assert int(np.asarray(w.conts.c_size)[int(o["inbound_cont"][n])]) == SIZE_OF[j.inbound_size.value]


def _v3_code(reason: str) -> int:
    """V3Announcer 의 skip 사유 → 배열 코드. 'TAIL' 은 자기 층에서 만든 문자열(orders.py:162), 나머지는 v5 예외 문구."""
    if reason == "TAIL":
        return AD.SKIP_TAIL
    return AD.code_of_reason(reason)


def _check_ledger_v3(v5_skips, my_rows, n_admit_delta, v5_admit_delta, label, t):
    """V3Announcer 경로 — 원장이 아니라 (skips, n_admitted) 로 대조한다 (그 스케줄러는 ADMIT/EPOCH 행을 안 남긴다)."""
    mine = [(r["job_id"], (AD.SKIP_TAIL if r["event"] == "SKIP_TAIL" else r["code"]))
            for r in my_rows if r["event"] in ("SKIP", "SKIP_TAIL")]
    v5 = [(k["job_id"], _v3_code(k["reason"])) for k in v5_skips]
    assert mine == v5, f"[{label}] t={t} SKIP 열 arr={mine} v5={v5}"
    assert n_admit_delta == v5_admit_delta, f"[{label}] t={t} 투입 수 arr={n_admit_delta} v5={v5_admit_delta}"


def _check_ledger(v5_rows, my_rows, label, t):
    assert len(v5_rows) == len(my_rows), f"[{label}] t={t} 원장 행 수 v5={len(v5_rows)} arr={len(my_rows)}\n v5={v5_rows}\n arr={my_rows}"
    for i, (a, b) in enumerate(zip(v5_rows, my_rows)):
        assert a["event"] == b["event"] and a.get("job_id") == b.get("job_id"), \
            f"[{label}] t={t} 원장 {i}번째: v5={a} arr={b}"
        if a["event"] == "ADMIT":
            assert (a["block"], a["flow"], a["arrival_s"]) == (b["block"], b["flow"], b["arrival_s"]), f"[{label}] t={t} ADMIT {a} {b}"
        elif a["event"] == "SKIP":
            assert AD.code_of_reason(a["reason"]) == b["code"], f"[{label}] t={t} SKIP 사유 v5={a['reason']!r} arr={b['reason_code']}"


# ───────────────────────────────────────────────── 본 시험
def _run_stage(stage: Stage) -> dict:
    prof, layout, built, schedule, changed = _build(stage)
    ids = list(layout.ids)
    block_index = {b: i for i, b in enumerate(ids)}
    g = Geom.from_profile(prof)
    sched_of_job = {e["job_id"]: e for e in schedule}
    epochs = ts.admission_epochs(OBS)
    lead = stage.lead

    # ── v5 ──
    sims = {b: _sim(prof, built["scenarios"][b]) for b in ids}
    margin = mb.CAPACITY_MARGIN if stage.margin is None else stage.margin
    if stage.margin is not None and stage.margin < 0:
        gb = prof.block
        free0 = gb.bay_count * gb.row_count * gb.tier_max - min(len(s.stacks.containers) for s in sims.values())
        margin = free0 - 3                                                  # 같은 에폭 안 4번째 반입부터 막힌다
    mbt = mb.MultiBlockTerminal(sims, capacity_margin=margin, extra_review_epochs=epochs)
    end5 = None if stage.end_none else built["sim_end_s"]
    if stage.announcer == "v3":
        from yard_rl.v6.stage.orders import V3Announcer                  # v5 stage 층 (트럭별 리드)
        ann = V3Announcer(schedule, end_s=end5, period_s=60.0)
    else:
        ann = ts.ScheduledAnnouncer(schedule, lead_s=lead, end_s=end5)
    end_ann = EMPTY_TIME if stage.end_none else float(built["sim_end_s"])

    # ── 배열 ──
    per_block = {b: [e for e in schedule if e["block"] == b] for b in ids}
    n_max = 8
    while n_max < max(len(built["scenarios"][b].jobs) + len(per_block[b]) for b in ids):
        n_max *= 2
    c_max = max(len(built["scenarios"][b].containers) for b in ids) + n_max
    q_cap = max(32, 4 * n_max)
    worlds, tables, row_of_job = [], {}, {}
    for b in ids:
        w0, tb = to_block_world(prof, built["scenarios"][b], n_max=n_max, q_cap=q_cap, log_cap=8 * n_max + 256, c_max=c_max)
        w1, tb2, rows = AD.attach_schedule(w0, tb, per_block[b], block_idx=block_index[b])
        worlds.append(w1)
        tables[b] = tb2
        row_of_job.update({e["job_id"]: rows[e["job_id"]] for e in per_block[b]})
    sched, info = AD.schedule_arrays(schedule, block_index=block_index, row_of_job=row_of_job, lead_s=lead,
                                     n_epochs=len(epochs))
    assert info["dropped"] == 0 and info["T"] == len(schedule)
    W = AD.stack_worlds(worlds)

    state = {"W": W, "n_epochs": 0, "codes": [], "ledger": [], "jit_checked": False, "roundtrip": 0}

    def review(mbt_, t):
        assert mbt_ is mbt
        grid = on_grid(t, 60.0)
        slot = AD.slot_of(t)
        assert grid == (slot >= 0)
        v3 = stage.announcer == "v3"
        n0 = len(ann.skips) if v3 else len(ann.ledger)
        a0 = ann.n_admitted
        # ① 이식 (투입 전 상태)
        Wt = AD.stack_worlds([_transplant(AD.world_at(state["W"], bi), sims[b], b, tables[b], row_of_job, g, t)
                              for bi, b in enumerate(ids)])
        fs_before = np.asarray(AD.free_slots_batch(Wt, g))
        v5_before = [mbt.free_slots(b) for b in ids]
        assert list(fs_before) == v5_before, f"[{stage.name}] t={t} free_slots(전) arr={list(fs_before)} v5={v5_before}"
        # ② v5
        ann.review(mbt, t)
        # ③ 배열
        W2, codes = AD.admit_epoch_jit(Wt, sched, slot, g, margin=margin, end_ann=end_ann)
        codes = np.asarray(codes)
        if grid and not state["jit_checked"] and (codes >= 0).any():
            W3, codes3 = AD.admit_epoch(Wt, sched, slot, g, margin=margin, end_ann=end_ann)
            assert np.array_equal(np.asarray(codes3), codes)
            bad = [str(p) for p, (x, y) in zip(jax.tree_util.tree_leaves_with_path(W2), zip(jax.tree_util.tree_leaves(W2), jax.tree_util.tree_leaves(W3)))
                   if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]
            assert not bad, f"[{stage.name}] jit 과 eager 가 다른 잎 {bad}"
            state["jit_checked"] = True
        # ④ 대조
        my_rows = AD.ledger_rows(codes, sched, slot, t, schedule)
        if v3:
            n_ok = sum(1 for r in my_rows if r["event"] == "ADMIT")
            _check_ledger_v3(ann.skips[n0:], my_rows, n_ok, ann.n_admitted - a0, stage.name, t)
        else:
            _check_ledger(ann.ledger[n0:], my_rows, stage.name, t)
        fs_after = np.asarray(AD.free_slots_batch(W2, g))
        v5_after = [mbt.free_slots(b) for b in ids]
        assert list(fs_after) == v5_after, f"[{stage.name}] t={t} free_slots(후) arr={list(fs_after)} v5={v5_after}"
        for bi, b in enumerate(ids):
            wq = AD.world_at(W2, bi).queue
            mine, v5q = _array_queue_rows(wq), _v5_queue_rows(sims[b], b, tables[b], row_of_job)
            assert mine == v5q, f"[{stage.name}] t={t} 블록 {b} 큐: arr={mine} v5={v5q}"
            assert int(wq.counter) == sims[b].queue._seq and int(wq.overflow) == 0
            _check_orders(W2, bi, b, sims[b], mbt, tables[b], row_of_job, sched_of_job, lead, f"{stage.name} t={t}")
        # 세션 이어 돌리기 경로 — numpy 왕복 (짝수 에폭마다)
        if state["n_epochs"] % 2 == 0:
            W2 = AD.tree_to_jax(AD.tree_to_numpy(W2))
            state["roundtrip"] += 1
        state["W"] = W2
        state["n_epochs"] += 1
        state["codes"].append(codes)
        state["ledger"].extend(my_rows)

    out = mbt.run(_rule_policy(), review_fn=review)
    mbt.check_invariants()
    all_codes = np.concatenate([c.reshape(-1) for c in state["codes"]]) if state["codes"] else np.zeros((0,), np.int32)
    n_admit = int((all_codes == AD.ADMIT).sum())
    assert n_admit == ann.n_admitted, f"[{stage.name}] 투입 수 arr={n_admit} v5={ann.n_admitted}"
    if stage.announcer == "v3":
        mine = [(r["job_id"], (AD.SKIP_TAIL if r["event"] == "SKIP_TAIL" else r["code"]))
                for r in state["ledger"] if r["event"] in ("SKIP", "SKIP_TAIL")]
        v5 = [(k["job_id"], _v3_code(k["reason"])) for k in ann.skips]
        assert mine == v5, f"[{stage.name}] SKIP 전열 arr={mine} v5={v5}"
        assert len(v5) == ann.n_skipped
    else:
        assert len(state["ledger"]) == len(ann.ledger), f"[{stage.name}] 원장 길이 arr={len(state['ledger'])} v5={len(ann.ledger)}"
        _check_ledger(ann.ledger, state["ledger"], stage.name, "전체")
    assert state["n_epochs"] >= len(epochs)
    counts = {AD.CODE_NAMES[c]: int((all_codes == c).sum()) for c in range(len(AD.CODE_NAMES))}
    n_lead = len({round(float(e["lead_s"]), 6) for e in schedule if e.get("lead_s") is not None})
    return dict(stage=stage.name, blocks=ids, load=stage.load, lead=lead, margin=margin, epochs=state["n_epochs"],
                announcer=stage.announcer, n_distinct_leads=n_lead,
                admitted=n_admit, counts=counts, M=info["M"], n_max=n_max, changed=changed,
                terminal_total=out["terminal_total"], roundtrip=state["roundtrip"])


@pytest.mark.parametrize("stage", STAGES, ids=_IDS)
def test_admission_matches_v5(stage: Stage):
    r = _run_stage(stage)
    REPORT[stage.name] = r
    c = r["counts"]
    assert r["admitted"] > 0
    if stage.name == "cap":
        assert c["SKIP_CAP"] > 0, f"용량 SKIP 이 한 번도 안 나왔다 (margin={r['margin']})"
    if stage.name == "edge":
        assert c["SKIP_DUP"] == 1 and c["SKIP_TARGET"] == 1 and c["SKIP_EXIT"] == 1, c
    if stage.name == "tail":
        assert c["SKIP_TAIL"] > 0 and c["SKIP_TIME"] == 0, c
    if stage.name == "time":
        assert c["SKIP_TIME"] > 0 and c["SKIP_TAIL"] == 0, c
    if stage.name == "leadmix":
        # ★트럭별 리드가 실제로 갈렸고(전원 같은 값이면 v2 와 다를 게 없다), 용량 거절도 실제로 났다
        assert r["n_distinct_leads"] >= 10, f"통지 리드가 거의 같다 ({r['n_distinct_leads']}가지) — 무대가 축을 못 만든다"
        assert c["SKIP_CAP"] > 0, f"용량 SKIP 이 한 번도 안 나왔다 (margin={r['margin']}) {c}"
    if stage.name in ("y01-20-600", "y01-60-1800", "y01y21-20-600", "y01y21-60-1800"):
        assert r["admitted"] == stage.load and sum(v for k, v in c.items() if k != "ADMIT") == 0, c


# ───────────────────────────────────────────────── 호스트 도구 (v5 함수와 대조)
def test_admission_epochs_and_slots_match_v5():
    """`admission_epochs_s` == v5 `admission_epochs(OBS_24H)` (1,441 격자) · `slot_of` 는 v5 by_epoch 키와 같은 버킷."""
    v5 = ts.admission_epochs(ts.OBS_24H)
    mine = AD.admission_epochs_s(ts.OBS_24H.observe_s)
    assert mine.shape[0] == len(v5) == 1441 and all(float(a) == b for a, b in zip(mine, v5))
    v5s = ts.admission_epochs(OBS)
    assert [float(x) for x in AD.admission_epochs_s(OBS.observe_s)] == list(v5s)
    for t in v5s:
        assert on_grid(t, 60.0) and AD.slot_of(t) == int(round(t / 60.0))
    for t in (0.5, 59.999, 60.0000011, 7199.0):
        assert AD.slot_of(t) == (int(round(t / 60.0)) if on_grid(t, 60.0) else -1)
    # by_epoch 키 (round((notify // 60) * 60, 6)) 와 slot 의 일치 — 실제 명단으로
    prof, layout = pr.build_h21_profile(), yl.terminal_layout().subset(("Y01",))
    built = ts.build_diurnal(prof, SEED, obs=OBS, layout=layout, params=ts.TerminalStreamParams(load_4h=20),
                             day_total=20, n_streams=0, drain_s=1200.0, background_seed=SEED)
    for lead in (600.0, 1800.0):
        ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=lead, end_s=built["sim_end_s"])
        rows = {e["job_id"]: i for i, e in enumerate(built["schedule"])}
        sched, info = AD.schedule_arrays(built["schedule"], block_index={"Y01": 0}, row_of_job=rows, lead_s=lead,
                                         n_epochs=len(ts.admission_epochs(OBS)))
        bucket = np.asarray(sched.bucket)
        for key, entries in ann.by_epoch.items():
            s = AD.slot_of(key)
            assert s >= 0 and [built["schedule"][i]["job_id"] for i in bucket[s, 0] if i >= 0] == [e["job_id"] for e in entries]
        assert int((bucket >= 0).sum()) == len(built["schedule"]) - info["dropped"] == 20


def test_terminal300_schedule_buckets_match_v5():
    """정답 무대(터미널 부하 300 · 21블록 · OBS_24H · lead 1800 — dump_ground_truth.run_terminal 과 같은 명단)의 버킷이
    v5 `ScheduledAnnouncer.by_epoch` 와 **에폭마다 같은 항목·같은 순서**이고, 1,441 격자 안에 전부 든다 (dropped 0).
    호스트 단계만 (엔진·세계 없음) — 통합 단계가 이 명단을 그대로 쓴다."""
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, SEED, obs=ts.OBS_24H, layout=layout, params=ts.TerminalStreamParams(load_4h=300),
                             day_total=300, background_seed=SEED)
    schedule = built["schedule"]
    ann = ts.ScheduledAnnouncer(schedule, lead_s=1800.0, end_s=built["sim_end_s"])
    block_index = {b: i for i, b in enumerate(layout.ids)}
    rows = {e["job_id"]: i for i, e in enumerate(schedule)}            # 행은 통합 단계가 정한다 — 여기선 자리표시
    sched, info = AD.schedule_arrays(schedule, block_index=block_index, row_of_job=rows, lead_s=1800.0,
                                     n_epochs=len(ts.admission_epochs(ts.OBS_24H)))
    assert (info["T"], info["E"], info["B"], info["dropped"]) == (300, 1441, 21, 0), info
    bucket = np.asarray(sched.bucket)
    seen = 0
    for key, entries in ann.by_epoch.items():
        s = AD.slot_of(key)
        assert s >= 0
        for b, bi in block_index.items():
            mine = [schedule[i]["job_id"] for i in bucket[s, bi] if i >= 0]
            v5 = [e["job_id"] for e in entries if e["block"] == b]
            assert mine == v5, f"슬롯 {s} 블록 {b}: arr={mine} v5={v5}"
            seen += len(v5)
    assert seen == 300 == int((bucket >= 0).sum())
    # 전 항목이 sim_end 안에 도착 → SKIP_TAIL 이 하나도 없어야 한다 (정답 admitted=300 과 같은 조건)
    assert all(e["arrival_s"] + e["travel_s"] <= built["sim_end_s"] for e in schedule)
    assert float(np.asarray(sched.epoch_s)[-1]) == ts.OBS_24H.observe_s
    REPORT["terminal300-buckets"] = dict(stage="terminal300-buckets", blocks=list(layout.ids), load=300, lead=1800.0,
                                         margin=mb.CAPACITY_MARGIN, epochs=info["E"], M=info["M"], admitted=0,
                                         counts={}, roundtrip=0)


def test_terminal300_crowded_buckets_match_v3announcer():
    """★21블록 규모에서 **한 (에폭, 블록) 칸에 여러 대**가 몰리는 명단도 v5 와 같은 버킷·같은 순서인가.

    왜 필요한가: 정답 무대(터미널 300 · lead 1800 고정)는 300칸이 전부 1건이라 "앞 투입이 뒤 투입의 free_slots 를
    줄인다" 는 에폭 안 순차 의존이 **한 번도 발동하지 않는다**(검증 지적). 여기서는 같은 명단에 트럭별 리드를
    심어 통지 시각을 흩뜨려 한 칸에 여러 대가 들어가게 만들고, 배열 버킷이 v5 `V3Announcer.by_epoch` 와
    에폭마다 같은 항목·같은 순서인지 본다. 엔진 없이 호스트 단계만 — 1초대.
    """
    import random
    from yard_rl.v6.stage.orders import V3Announcer
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, SEED, obs=ts.OBS_24H, layout=layout, params=ts.TerminalStreamParams(load_4h=300),
                             day_total=300, background_seed=SEED)
    schedule = [dict(e) for e in built["schedule"]]
    rng = random.Random(f"v3:lead:crowd:{SEED}")
    for e in schedule:
        e["lead_s"] = min(float(ts.sample_lead_s(rng.random())), float(e["arrival_s"]))
    ann = V3Announcer(schedule, end_s=built["sim_end_s"], period_s=60.0)
    block_index = {b: i for i, b in enumerate(layout.ids)}
    rows = {e["job_id"]: i for i, e in enumerate(schedule)}            # 행은 통합 단계가 정한다 — 여기선 자리표시
    sched, info = AD.schedule_arrays(schedule, block_index=block_index, row_of_job=rows, lead_s=0.0,
                                     n_epochs=len(ts.admission_epochs(ts.OBS_24H)))
    assert (info["T"], info["E"], info["B"], info["dropped"]) == (300, 1441, 21, 0), info
    bucket = np.asarray(sched.bucket)
    seen, crowded = 0, 0
    for key, entries in ann.by_epoch.items():
        s_ = AD.slot_of(key)
        assert s_ >= 0, f"통지 격자 밖 {key}"
        for b, bi in block_index.items():
            mine = [schedule[i]["job_id"] for i in bucket[s_, bi] if i >= 0]
            v5 = [e["job_id"] for e in entries if e["block"] == b]
            assert mine == v5, f"슬롯 {s_} 블록 {b}: arr={mine} v5={v5}"
            seen += len(v5)
            crowded += 1 if len(v5) >= 2 else 0
    assert seen == 300 == int((bucket >= 0).sum())
    assert crowded >= 1 and info["M"] >= 2, \
        f"한 칸에 여러 대인 경우가 없다 (혼잡 칸 {crowded} · M={info['M']}) — 순차 의존을 못 밟는 무대다"
    n_lead = len({round(float(e["lead_s"]), 6) for e in schedule})
    REPORT["terminal300-crowded"] = dict(stage="terminal300-crowded", blocks=list(layout.ids), load=300, lead=-1.0,
                                         margin=mb.CAPACITY_MARGIN, epochs=info["E"], M=info["M"], admitted=0,
                                         counts={"crowded_cells": crowded, "distinct_leads": n_lead}, roundtrip=0)


def test_code_of_reason_covers_v5_messages():
    """v5 `admit_external_job` 의 raise 문구 다섯 가지가 전부 코드로 분류된다 (소스에서 읽어 확인)."""
    import inspect
    src = inspect.getsource(mb.MultiBlockTerminal.admit_external_job)
    for key, code in AD._REASON_KEYS:
        assert key in src, f"v5 문구가 바뀌었다: {key!r}"
        assert AD.code_of_reason(f"Y01:D-00001: {key} …") == code
    assert AD.code_of_reason("낯선 문구") == -2


def test_zz_report(capsys):
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    with capsys.disabled():
        print("\n[admission report]")
        for k, r in REPORT.items():
            print(f"  {k:16s} blocks={','.join(r['blocks'])} load={r['load']} lead={r['lead']:.0f} margin={r['margin']} "
                  f"epochs={r['epochs']} M={r['M']} admitted={r['admitted']} {r['counts']} roundtrip={r['roundtrip']}")
    if os.environ.get("REPORT_LOAD"):                     # 조각 병합 뒤 — 무대 여덟이 전부 돌았어야 한다
        assert set(_IDS) <= set(REPORT), sorted(set(_IDS) - set(REPORT))
