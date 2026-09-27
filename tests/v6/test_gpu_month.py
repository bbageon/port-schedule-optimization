"""★30일 무대(gpu/month.py) 가 v5 `MonthTerminal` + `run_month` **기본 갈래**와 같은 답을 낸다
([[YR-327]] 조각 8 · key=month).

■ 사다리 — 앞이 깨지면 그 자리에서 **처음 갈리는 사건(블록·시각·종류)** 을 보고한다
  ① 범위 — `seed_data`(고정 화물 = CargoTerminal 갈래)·`targets` 를 든 본선 행을 주면 **큰 소리로 거절**한다.
  ② 무대 조립 — 오더 행 배치가 v5 `sorted(sim.jobs)` 와 같은 순서인가 (트럭 먼저 · 본선은 날짜 순),
     검토 시각이 60초 격자이고 **날 경계가 그 안에** 있는가, 컨테이너 이름이 v5 와 같은가.
  ③ 2블록 · 2일 · 부하 20 — v5 `run_month` 기본 갈래의 뼈대를 **무대 크기만 줄여** 그대로 굴리고
     (MonthTerminal · inject_vessel · V3Announcer(retarget) · prune/retire · _MonthTape · SF_SPT · 시장 없음)
     ★하루 경계(t=86,400)와 런 끝에서 대조한다:
       · 블록별 사건 전열·해시 · 비용 13항 · KPI 10항 · 장부 적분 3항
       · 오더별 (status·크레인·재조작·S·C·A·B·O) — v5 가 치운(pruned) 행은 **치운 이유까지** 확인
       · ★야드 (파일 전부 · 컨테이너 좌표) — **어제 야드 위에 오늘** 이 성립하는가
       · 본선 주입 원장 전행 (moves·start·planned_completion·why) · 배별 (remaining·유휴·done·완료시각)
       · 투입 (admitted 수 · SKIP 사유별) · 날별 Φ 네 항
  ④ 2블록 · 3일 — 경계가 두 번. 이어짐이 **누적**되는지 (사흘째 야드가 이틀치 흔적을 들고 있나).
  ⑤ prune/retire 가 **무동작**인가 — v5 가 치운 건수와 배열 감사 수치가 `==` 인지 (gpu/month 머리말의 주장).

⚠️ 21블록 30일 전체는 **통합 단계** 몫이다 (에폭 43,201개 × 21블록). 여기서는 하루 경계 논리와 본선 주입을
   작은 무대에서 비트까지 못 박는다.

⚠️ v5 `run_month` 은 `layout` 인자가 없어 늘 21블록을 쓴다. 그래서 이 파일은 `run_month` 의 **몸통을
   그대로 재구성**한다 (month_run.py:242-579 에서 시장·반사실 교사만 뺀 NO_REALLOC 배선) — v5 함수를
   실제로 부르며 기대값을 손기입하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import admission as AD                                        # noqa: E402
from yard_rl.v6.gpu import dispatch as DP                                         # noqa: E402
from yard_rl.v6.gpu import host_terminal as HT                                    # noqa: E402
from yard_rl.v6.gpu import month as MO                                            # noqa: E402
from yard_rl.v6.gpu import multiblock as MB                                       # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                              # noqa: E402
from yard_rl.v6.gpu.state import COST_TERMS                                       # noqa: E402
from yard_rl.v6.reward.phi import terminal_cost_krw as v5_phi                     # noqa: E402
from yard_rl.v6.stage import month as v5m                                         # noqa: E402
from yard_rl.v6.stage import month_engine as v5me                                 # noqa: E402
from yard_rl.v6.stage.episode import _rule_policy, _sim_from                      # noqa: E402
from yard_rl.v6.stage.month_run import SNAP_S, _MonthTape                         # noqa: E402
from yard_rl.v6.stage.orders import EPOCH_S, V3Announcer                          # noqa: E402
from yard_rl.v6.world.integrated.multiblock import TransferError                  # noqa: E402
from yard_rl.v6.world.integrated.profiles import build_h21_profile                # noqa: E402
from yard_rl.v6.world.integrated.terminal_stream import (DIURNAL_DRAIN_S, OBS_24H,  # noqa: E402
                                                          ensure_time_ledger)
from yard_rl.v6.world.integrated.yard_layout import terminal_layout               # noqa: E402

SEED = 9_900_777
KPI_KEYS = ("queue_area_s", "tail_area_s", "loaded_gantry_m", "empty_gantry_m", "rehandle_count",
            "pre_rehandle_count", "completed_external", "completed_vessel", "vessel_delay_s",
            "positioning_count")
REPORT: dict = {}


# ───────────────────────────────────────────────── 무대 (v5 · 배열) — 기대값 손기입 없음
def _plan(n_days: int, load: int):
    return [v5m.DayPlan(index=i, load=load, label="시험", seed=SEED + 1_000 * (i + 1),
                        t0=i * v5m.DAY_S, n_days=n_days) for i in range(n_days)]


def _build(blocks, n_days: int, load: int, cap_moves: int | None = None):
    """v5 `run_month` 242-250행 — 명단·본선 계획. 블록만 줄인다.

    `cap_moves` 는 본선 한 척의 **물량 상한**이다. 실제 계획은 스트림당 235~543 상자라 오더 칸 N 이
    수백~수천이 되고, 그러면 컴파일과 에폭 수 때문에 시험이 시간 안에 안 끝난다. 물량을 깎아도
    **양하/적하 두 갈래·해제 흐름·재고 소비·계획 시각 식은 그대로** 밟는다 (v5·배열 둘에 같은 행을 준다).
    상한 없는 규모는 통합 단계 몫이다 (머리말 ⚠️).
    """
    prof = build_h21_profile()
    layout = terminal_layout().subset(blocks)
    days = _plan(n_days, load)
    built = v5m.build_month(SEED, days=days, profile=prof, layout=layout, lead_mode="DIST")
    v_by_day = v5m.plan_month_vessels(days, layout, obs=OBS_24H,
                                      truck_net=v5m.truck_net_by_block(built["schedule"]))
    if cap_moves is not None:
        v_by_day = {d: [dict(r, moves=min(int(r["moves"]), int(cap_moves))) for r in rows]
                    for d, rows in v_by_day.items()}
    return prof, layout, days, built, v_by_day


def _v5_month(prof, layout, days, built, v_by_day, *, on_boundary=None):
    """v5 `run_month` 기본 갈래의 **몸통** (month_run.py:252-579, 시장·교사 제외 = NO_REALLOC).

    부르는 v5 함수는 전부 원본이다 — `MonthTerminal` · `inject_vessel` · `V3Announcer(retarget=make_retarget)`
    · `prune_completed` · `retire_done_vessels` · `_MonthTape` · `month_vessel_idle` · `_rule_policy`.
    """
    import dataclasses

    n_days = len(days)
    month_s = n_days * v5m.DAY_S
    sim_end = month_s + DIURNAL_DRAIN_S
    scns = {b: dataclasses.replace(s, jobs=[], vessels=[], horizon_s=month_s,
                                   drain_window_s=DIURNAL_DRAIN_S)
            for b, s in built["day0"]["scenarios"].items()}
    extra = tuple(i * EPOCH_S for i in range(int(month_s // EPOCH_S) + 1))
    mbt = v5me.MonthTerminal({b: ensure_time_ledger(_sim_from(s, prof)) for b, s in scns.items()},
                             extra_review_epochs=extra)
    ann = V3Announcer(built["schedule"], end_s=sim_end, retarget=v5m.make_retarget(SEED))
    meta: dict = {}
    for rows in v_by_day.values():
        meta.update(v5m.vessel_meta(rows))
    archive: dict = {}
    tape = _MonthTape(meta, archive)
    out = {"vessel_admissions": [], "pruned": [], "retired": [], "days": [], "live": []}
    state = {"day": 0, "snap": 0.0, "opened": (0, 0, 0)}

    def open_day(d):
        n = moves = skipped = 0
        for r in v_by_day.get(d.index, []):
            try:
                a = v5me.inject_vessel(mbt, r["block"], r, key=r["key"],
                                       deadline_mult=v5me.VESSEL_DEADLINE_MULT,
                                       size_seed=f"v3:month:{SEED}:{r['key']}")
            except TransferError as ex:
                skipped += 1
                out["vessel_admissions"].append({"key": r["key"], "day": d.index, "block": r["block"],
                                                 "time_s": d.t0, "asked": r["moves"], "moves": 0,
                                                 "ok": False, "why": str(ex)})
                continue
            n += 1
            moves += a.moves
            out["vessel_admissions"].append({"key": a.vessel_key, "day": d.index, "block": r["block"],
                                             "time_s": d.t0, "ok": True, "moves": a.moves,
                                             "asked": a.asked_moves, "why": a.reason})
        return n, moves, skipped

    def close_day(d, t, opened):
        v, yc, rh = tape.diff(d.t0, t)
        phi = v5_phi({k: r for k, r in records.items() if k.startswith(f"D{d.index:02d}-")},
                     end_s=t, vessel_idle=v, yc_extra_move_s=yc, rehandles=rh)
        rep = {"index": d.index, "load": d.load, "vessels": opened[0], "vessel_moves": opened[1],
               "vessel_skipped": opened[2], "phi": phi}
        rep["pruned"] = v5m.prune_completed(mbt, t)
        rep["retired"] = v5m.retire_done_vessels(mbt, archive, t=t)
        out["pruned"].append(rep["pruned"])
        out["retired"].append(rep["retired"])
        out["live"].append(rep)

    # ★기록은 v5 `orders_from_schedule` 것을 쓰고, `bridge._sync` 대신 같은 규약으로 찍는다
    #   (시장 다리는 조각 8 의 다른 담당 몫 — 여기서는 Φ 원료만 필요하다).
    from yard_rl.v6.stage.orders import orders_from_schedule
    orders, records = orders_from_schedule(built)
    from yard_rl.v6.schema import Stage

    def sync(t):
        for sim in mbt.blocks.values():
            tl = getattr(sim, "time_ledger", None)
            if tl is None:
                continue
            for jid, tt_ in tl.records.items():
                rec = records.get(jid)
                if rec is None:
                    continue
                for stage, val in ((Stage.GATE_IN, tt_.gate_in), (Stage.BLOCK_IN, tt_.block_arrival),
                                   (Stage.JOB_DONE, tt_.job_done), (Stage.GATE_OUT, tt_.gate_out)):
                    if val is None or val > t + 1e-9 or stage in rec._stamped:
                        continue
                    rec.stamp(stage, float(val))

    def review(m, t):
        ann.review(m, t)
        sync(t)
        if t >= state["snap"]:
            tape.snap(m, t)
            state["snap"] = t + SNAP_S
        while state["day"] < n_days and t >= days[state["day"]].t0 - 1e-9:
            d = days[state["day"]]
            tape.snap(m, d.t0)
            if on_boundary is not None:
                on_boundary(m, d.index, d.t0)
            if state["day"] > 0:
                close_day(days[state["day"] - 1], d.t0, state["opened"])
            state["opened"] = open_day(d)
            state["day"] += 1
            state["snap"] = t + SNAP_S

    exec_policy, exc = _rule_policy("SF_SPT", seed=SEED)
    mbt.run(exec_policy, review_fn=review)
    sync(sim_end)
    tape.snap(mbt, month_s)
    close_day(days[-1], month_s, state["opened"])
    sync(sim_end)
    for d in days:
        v, yc, rh = tape.diff(d.t0, min(d.t1, month_s))
        out["days"].append(v5_phi({k: r for k, r in records.items() if k.startswith(f"D{d.index:02d}-")},
                                  end_s=sim_end, vessel_idle=v, yc_extra_move_s=yc, rehandles=rh))
    out.update(mbt=mbt, ann=ann, archive=archive, tape=tape, records=records,
               policy_exceptions=exc["n"], sim_end=sim_end, month_s=month_s)
    return out


# ───────────────────────────────────────────────── 대조 도구
def _v5_block_snapshot(sim):
    """v5 블록 하나의 상태 — 배열판과 대조할 항목만."""
    tl = sim.time_ledger
    return {
        "events": [(round(t, 6), k, p) for (t, k, p) in sim.event_log],
        "hash": sim.event_stream_hash(),
        "cost": dict(sim.cost.episode_raw()),
        "kpis": {"queue_area_s": sim.kpis.queue_area_s, "tail_area_s": sim.kpis.tail_area_s,
                 "loaded_gantry_m": sim.kpis.loaded_gantry_m, "empty_gantry_m": sim.kpis.empty_gantry_m,
                 "rehandle_count": sim.kpis.rehandle_count,
                 "pre_rehandle_count": sim.kpis.pre_rehandle_count,
                 "completed_external": sim.kpis.completed_external,
                 "completed_vessel": sim.kpis.completed_vessel,
                 "vessel_delay_s": sim.kpis.vessel_delay_s,
                 "positioning_count": sim.kpis.positioning_count},
        "ledger": (tl.terminal_area_s, tl.block_area_s, tl.block_tail_area_s),
        "clock": sim.clock, "end": sim.end, "escapes": sim.deadlock_escape_count,
        "unfinished": sim.unfinished_backlog(),
        "jobs": {jid: (j.status.name, j.assigned_crane, j.rehandle_count, j.service_start, j.service_end,
                       (tl.records[jid].gate_in if jid in tl.records else None),
                       (tl.records[jid].block_arrival if jid in tl.records else None),
                       (tl.records[jid].gate_out if jid in tl.records else None))
                 for jid, j in sim.jobs.items()},
        "boxes": {c: (x.bay, x.row, x.tier, x.size.value) for c, x in sim.stacks.containers.items()},
        "vessels": {k: (v.remaining_moves, v.buffer_level, v.sts_wait_accum_s, v.done,
                        v.truth.actual_completion_s) for k, v in sim.vessels.items()},
    }


def _array_block_snapshot(run, tt, b: int):
    w = HT.block_slice(MB.tree_to_numpy(run.tw), b)
    tb = HT.block_tables(MB.tree_to_numpy(run.tw), tt, b)
    from yard_rl.v6.gpu.host_convert import event_stream_hash, from_block_world
    d = from_block_world(w, tb)
    blk = np.asarray(w.orders.block)
    present = [n for n in range(len(tb.job_ids)) if blk[n] >= 0]
    jobs = {}
    for n in present:
        a = d["jobs"][tb.job_ids[n]]
        jobs[tb.job_ids[n]] = (a["status"], a["assigned_crane"], a["rehandle_count"], a["service_start"],
                               a["service_end"], a["gate_in"], a["block_arrival"], a["actual_gate_out"])
    alive = np.asarray(w.conts.c_alive)
    cb, cr, ct, cs = (np.asarray(w.conts.c_bay), np.asarray(w.conts.c_row), np.asarray(w.conts.c_tier),
                      np.asarray(w.conts.c_size))
    sz = ("FT20", "FT40", "FT45")
    boxes = {tb.cont_ids[c]: (int(cb[c]), int(cr[c]), int(ct[c]), sz[int(cs[c])])
             for c in np.nonzero(alive)[0]}
    va = np.asarray(w.vessels.alive)
    vessels = {tb.vessel_ids[v]: (int(w.vessels.remaining[v]), int(w.vessels.buffer[v]),
                                  float(w.vessels.wait_accum_s[v]), bool(w.vessels.done[v]),
                                  (None if not np.isfinite(float(w.vessels.actual_completion_s[v]))
                                   else float(w.vessels.actual_completion_s[v])))
               for v in np.nonzero(va)[0]}
    return {
        "events": [(round(t, 6), k, p) for (t, k, p) in d["event_log"]],
        "hash": event_stream_hash(w, tb), "cost": d["cost_episode"],
        "kpis": {k: d["kpi"][k] for k in KPI_KEYS},
        "ledger": (d["ledger"]["terminal_area_s"], d["ledger"]["block_area_s"],
                   d["ledger"]["block_tail_area_s"]),
        "clock": d["clock"], "end": d["end"], "escapes": d["escape_count"],
        "violation": d["violation"], "violation_names": d["violation_names"], "overflow": d["overflow"],
        "jobs": jobs, "boxes": boxes, "vessels": vessels,
        "n_jobs": len(present),
    }


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), \
               (b[len(a)] if len(b) > len(a) else None)
    return None


def _cmp_block(label: str, bid: str, ar: dict, v5: dict, *, t: float, pruned_ok) -> None:
    diff = _first_diff(ar["events"], v5["events"])
    if diff is not None:
        i, x, y = diff
        ctx = "\n".join(
            f"    #{j}: arr={ar['events'][j] if j < len(ar['events']) else '-'}  |  "
            f"v5={v5['events'][j] if j < len(v5['events']) else '-'}"
            for j in range(max(0, i - 4), min(max(len(ar['events']), len(v5['events'])), i + 4)))
        pytest.fail(f"[{label}] 블록 {bid} t={t} 사건 로그가 {i}번째에서 갈린다: arr={x} v5={y}\n{ctx}\n"
                    f"  (arr {len(ar['events'])}건 · v5 {len(v5['events'])}건 · "
                    f"violation={ar['violation_names']} overflow={ar['overflow']})")
    assert ar["hash"] == v5["hash"], f"[{label}] {bid} t={t} 사건 해시"
    assert ar["violation"] == 0 and ar["overflow"] == 0, \
        f"[{label}] {bid} violation={ar['violation_names']} overflow={ar['overflow']}"
    bad = [(k, ar["cost"][k], v5["cost"][k]) for k in COST_TERMS if ar["cost"][k] != v5["cost"][k]]
    assert not bad, f"[{label}] {bid} t={t} 비용 항목 (항, arr, v5): {bad}"
    assert ar["kpis"] == v5["kpis"], f"[{label}] {bid} t={t} KPI arr={ar['kpis']} v5={v5['kpis']}"
    assert ar["ledger"] == v5["ledger"], f"[{label}] {bid} t={t} 장부 적분 arr={ar['ledger']} v5={v5['ledger']}"
    assert (ar["clock"], ar["end"], ar["escapes"]) == (v5["clock"], v5["end"], v5["escapes"]), \
        f"[{label}] {bid} t={t} 시계/끝/탈출 arr={(ar['clock'], ar['end'], ar['escapes'])} " \
        f"v5={(v5['clock'], v5['end'], v5['escapes'])}"
    # ── 오더: v5 가 치운 행은 배열판에만 있다 (머리말 ■ prune 무동작) ──
    only_arr = set(ar["jobs"]) - set(v5["jobs"])
    assert not (set(v5["jobs"]) - set(ar["jobs"])), \
        f"[{label}] {bid} t={t} v5 에만 있는 작업 {sorted(set(v5['jobs']) - set(ar['jobs']))[:5]}"
    for jid in only_arr:
        st, _, _, _, _, _, _, o = ar["jobs"][jid]
        assert pruned_ok(st, o), f"[{label}] {bid} t={t} 배열에만 있는 {jid} 가 '치운 것' 이 아니다: {ar['jobs'][jid]}"
    for jid in v5["jobs"]:
        assert ar["jobs"][jid] == v5["jobs"][jid], \
            f"[{label}] {bid} t={t} 오더 {jid}: arr={ar['jobs'][jid]} v5={v5['jobs'][jid]}"
    # ── ★야드 — 하루 경계의 핵심 (어제 야드 위에 오늘) ──
    assert set(ar["boxes"]) == set(v5["boxes"]), (
        f"[{label}] {bid} t={t} 야드 상자 집합이 다르다 — arr에만 "
        f"{sorted(set(ar['boxes']) - set(v5['boxes']))[:5]} · v5에만 "
        f"{sorted(set(v5['boxes']) - set(ar['boxes']))[:5]} (arr {len(ar['boxes'])} · v5 {len(v5['boxes'])})")
    bad = [(c, ar["boxes"][c], v5["boxes"][c]) for c in v5["boxes"] if ar["boxes"][c] != v5["boxes"][c]]
    assert not bad, f"[{label}] {bid} t={t} 상자 좌표 {bad[:5]}"
    # ── 배: v5 는 끝난 배를 치운다 (retire) — 배열판에만 남은 것은 done 이어야 한다 ──
    assert not (set(v5["vessels"]) - set(ar["vessels"])), \
        f"[{label}] {bid} t={t} v5 에만 있는 배 {set(v5['vessels']) - set(ar['vessels'])}"
    for k in set(ar["vessels"]) - set(v5["vessels"]):
        assert ar["vessels"][k][3], f"[{label}] {bid} t={t} 배열에만 남은 배 {k} 가 done 이 아니다"
    for k in v5["vessels"]:
        assert ar["vessels"][k] == v5["vessels"][k], \
            f"[{label}] {bid} t={t} 배 {k}: arr={ar['vessels'][k]} v5={v5['vessels'][k]}"


def _pruned_ok_at(t: float):
    def ok(status: str, gate_out) -> bool:
        return status in ("DONE", "CANCELLED") and (gate_out is None or gate_out <= t + 1e-9)
    return ok


def _run_ladder(blocks, n_days: int, load: int, label: str, cap: int | None):
    prof, layout, days, built, v_by_day = _build(blocks, n_days, load, cap_moves=cap)
    snap5: dict = {}
    snapA: dict = {}

    def cap5(m, day, t):
        snap5[day] = {b: _v5_block_snapshot(s) for b, s in m.blocks.items()}

    def capA(run, day, t):
        snapA[day] = {bid: _array_block_snapshot(run, tt_box["tt"], b)
                      for b, bid in enumerate(tt_box["tt"].block_ids)}

    out5 = _v5_month(prof, layout, days, built, v_by_day, on_boundary=cap5)
    tt_box: dict = {}

    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout)
    tt_box["tt"] = tt
    g = Geom.from_profile(prof)
    run0 = MO.make_month_run(tw, tt, g)
    k0 = tt.tables[0].crane_index[prof.cranes[0].crane_id]
    eng = MO.month_engine(tt, g, policy_fn=DP.make_resolver("sf_spt", g, count_lost=False),
                          horizon_s=float(prof.decision_horizon_s), k0=int(k0))
    codes: list = []
    run, res, tape = MO.run_month(run0, tt, lay, eng, seed=SEED, profile=prof, on_boundary=capA,
                                  on_epoch=lambda r, e, t, c: codes.append(np.asarray(c)))
    return dict(prof=prof, layout=layout, days=days, built=built, v_by_day=v_by_day, out5=out5,
                run=run, tt=tt, lay=lay, res=res, tape=tape, codes=np.asarray(codes),
                snap5=snap5, snapA=snapA, label=label)


_CACHE: dict = {}


def _ladder(blocks, n_days, load, label, cap):
    key = (blocks, n_days, load, cap)
    if key not in _CACHE:
        _CACHE[key] = _run_ladder(blocks, n_days, load, label, cap)
    return _CACHE[key]


def _ladder3():
    return _ladder(("Y01", "Y04"), 2, 150, "month-2b-2d", 24)


def _ladder5():
    # ★배 한 척에 스트림이 **셋 이상** 모이는 무대 (STS 4 중형선) — `sum()` 의 Neumaier 보정합이
    #   순서에 걸리는 지점을 밟는다. 둘뿐이면 덧셈이 교환법칙으로 같아 순서를 못 시험한다.
    return _ladder(("Y01", "Y04", "Y05", "Y12", "Y15", "Y16"), 2, 60, "month-6b-2d", 12)


def _ladder4():
    # ⚠️ 블록이 하나면 `plan_streams` 가 본선을 배정하지 않는다 (실측) — 본선 축이 통째로 빠지므로 둘 이상
    return _ladder(("Y01", "Y21"), 3, 120, "month-2b-3d", 16)


# ───────────────────────────────────────────────── ① 범위 — 큰 소리로 거절한다
def test_fixed_cargo_branch_is_refused():
    """`seed_data`(CargoTerminal 갈래)는 이번 범위 밖 — 조용히 다른 답을 내지 않는다."""
    prof, layout, days, built, v_by_day = _build(("Y01",), 1, 8)
    with pytest.raises(MO.MonthScopeError) as ex:
        MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout,
                          seed_data={"vessels_by_day": {}})
    assert "CargoTerminal" in str(ex.value)


def test_fixed_vessel_targets_are_refused():
    """본선 행에 `targets`(고정 명단)가 있으면 거절한다 — 그 갈래는 전역 큐·원격 인계가 붙는다."""
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04"), 2, 20)
    rows = {k: [dict(r, targets=["X"]) for r in v] for k, v in v_by_day.items()}
    assert any(rows.values()), "본선 계획이 비었다 — 거절을 시험하지 못한다"
    with pytest.raises(MO.MonthScopeError):
        MO.to_month_world(prof, built, rows, days=days, seed=SEED, layout=layout)


# ───────────────────────────────────────────────── ② 무대 조립
def test_rows_follow_v5_sorted_job_id_order():
    """★오더 행 배치 = v5 `sorted(sim.jobs)` — 트럭이 먼저, 본선은 **붙는 날짜 순**."""
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04"), 2, 20)
    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout)
    for b, bid in enumerate(tt.block_ids):
        ids = list(tt.tables[b].job_ids)
        assert ids == sorted(ids), f"{bid}: 행 배치가 사전식이 아니다 — v5 name_rank 와 갈린다"
        n_t = lay.n_trucks[b]
        # ★트럭 이름에는 블록 접두가 없다 — `stage/orders.build_stage` 65행이 떼어 낸다
        #   (엔진 id 와 docKey 를 같은 문자열로 만들려고). 본선 작업은 `_namespace_jobs` 꼴이라
        #   `"D…" < "Y01:J-…"` 로 **트럭이 여전히 앞 번호**다.
        assert all(not j.startswith(f"{bid}:J-") for j in ids[:n_t]), f"{bid}: 앞 {n_t} 행이 트럭이 아니다"
        assert all(j.startswith(f"{bid}:J-") for j in ids[n_t:]), f"{bid}: 뒤 행이 본선 작업이 아니다"
        # 날짜가 앞에 오므로 붙는 순서 = 행 순서
        for key, rows in ((k[1], v) for k, v in lay.rows.items() if k[0] == b):
            day = lay.day_of_key[key]
            for m, n in enumerate(rows):
                assert ids[n] == f"{bid}:J-{key}-{m:04d}", f"{bid}: 행 {n} 이름이 어긋난다"
            assert all(lay.day_of_key[k2] <= day for (b2, k2), v2 in lay.rows.items()
                       if b2 == b and v2[0] < rows[0]), f"{bid}: 날짜 순서가 깨졌다"


def test_review_epochs_contain_every_day_boundary():
    """검토 시각은 60초 격자이고 날 경계 `i·86400` 이 **그 안에** 있어야 한다 (경계 훅이 걸릴 자리)."""
    prof, layout, days, built, v_by_day = _build(("Y01",), 3, 12)
    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout)
    eps = np.asarray(tw.epochs.t)
    assert eps.shape[0] == int(lay.month_s // MO.EPOCH_S) + 1
    assert np.all(np.diff(eps) == MO.EPOCH_S)
    for d in lay.days:
        assert float(d.t0) in set(float(x) for x in eps), f"날 경계 {d.t0} 가 검토 시각에 없다"


def test_container_names_match_v5_for_injected_vessel_boxes():
    """양하가 낳는 상자 이름이 v5 `IN_{본선작업 id}` 와 같아야 한다 — `free_targets` 의 정렬이 여기 걸린다."""
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04"), 2, 20)
    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout)
    for b, bid in enumerate(tt.block_ids):
        tb = tt.tables[b]
        for n, jid in enumerate(tb.job_ids):
            assert tb.cont_ids[tb.c0 + n] == f"IN_{jid}"


# ───────────────────────────────────────────────── ③ 2블록 · 2일
def test_day_boundary_carries_yesterdays_yard():
    """★하루 경계에서 **세계가 이어진다** — 어제 야드 위에 오늘.

    t = 86,400 (이튿날 배를 붙이기 **직전**) 의 야드·오더·계수기·비용·사건 전열을 v5 와 대조한다.
    이것이 30일 무대의 존재 이유이고, 틀리면 이튿날부터 전부 다른 세계다.
    """
    L = _ladder3()
    assert set(L["snapA"]) == set(L["snap5"]) == {0, 1}, "경계 사진이 안 찍혔다"
    for day in (0, 1):
        t = L["lay"].days[day].t0
        for b, bid in enumerate(L["tt"].block_ids):
            _cmp_block(f"{L['label']} d{day}", bid, L["snapA"][day][bid], L["snap5"][day][bid],
                       t=t, pruned_ok=_pruned_ok_at(t))
    n0 = sum(len(L["snap5"][0][b]["boxes"]) for b in L["snap5"][0])
    n1 = sum(len(L["snap5"][1][b]["boxes"]) for b in L["snap5"][1])
    assert n1 != n0, "야드가 하루 뒤에도 똑같다 — 이어짐을 시험하는 무대가 아니다"
    REPORT["boundary"] = {"boxes_day0": n0, "boxes_day1": n1}


def test_final_world_matches_v5():
    """런 끝 — 사건 전열·해시·비용 13항·KPI·오더·야드·배 전부."""
    L = _ladder3()
    t = L["out5"]["sim_end"]
    for b, bid in enumerate(L["tt"].block_ids):
        ar = _array_block_snapshot(L["run"], L["tt"], b)
        v5 = _v5_block_snapshot(L["out5"]["mbt"].blocks[bid])
        _cmp_block(f"{L['label']} final", bid, ar, v5, t=t, pruned_ok=_pruned_ok_at(t))
    REPORT["final"] = {"blocks": len(L["tt"].block_ids), "epochs": int(L["res"].n_epochs)}


def test_vessel_injection_ledger_matches_v5():
    """본선 주입 원장 전행 — 물량·시작·계획완료·사유까지."""
    L = _ladder3()
    ar = L["res"].vessel_admissions
    v5 = L["out5"]["vessel_admissions"]
    assert len(ar) == len(v5), f"주입 줄 수 arr={len(ar)} v5={len(v5)}"
    for i, (a, b) in enumerate(zip(ar, v5)):
        assert a == b, f"주입 {i}번째: arr={a} v5={b}"
    assert any(r["ok"] for r in ar), "배가 한 척도 안 붙었다 — 무대가 본선을 시험하지 않는다"


def test_admission_matches_v5_including_no_target():
    """투입 — 들어온 수와 SKIP 사유별 (★`retarget` 의 NO_TARGET 이 배열판에도 있어야 한다)."""
    L = _ladder3()
    admitted = int(np.asarray(L["run"].tw.ledger.registered).sum())
    ann = L["out5"]["ann"]
    assert admitted == ann.n_admitted, f"투입 수 arr={admitted} v5={ann.n_admitted}"
    # 배열 SKIP = 조정자 코드(TAIL/DUP/EXIT/TIME/CAP/TARGET) + 호스트 NO_TARGET
    sched = [{"job_id": L["tt"].truck_ids[s],
              "block": L["tt"].block_ids[int(np.asarray(L["run"].tw.sched.block)[s])],
              "flow": "GATE_IN", "arrival_s": float(np.asarray(L["run"].tw.sched.arrival_s)[s])}
             for s in range(L["run"].tw.sched.s)]
    ep = np.asarray(L["run"].tw.epochs.t)
    ar_skip: list = []
    for e in range(L["codes"].shape[0]):
        t = float(ep[e])
        for r in AD.ledger_rows(L["codes"][e], L["run"].adm, AD.slot_of(t, L["tt"].period_s), t, sched):
            if r["event"] == "SKIP" and r["reason_code"] != "SKIP_DUP":
                ar_skip.append((r["job_id"], r["reason_code"]))
    ar_skip += [(s["job_id"], "NO_TARGET") for s in L["res"].truck_skips]
    v5_skip = [(s["job_id"], s["reason"]) for s in ann.skips]
    assert sorted(ar_skip) == sorted(v5_skip), (
        f"SKIP 갈림 — arr에만 {sorted(set(ar_skip) - set(v5_skip))[:5]} · "
        f"v5에만 {sorted(set(v5_skip) - set(ar_skip))[:5]}")
    assert any(r == "NO_TARGET" for _, r in v5_skip), (
        "NO_TARGET 이 한 번도 안 났다 — `retarget` 이식을 시험하지 못한다 "
        "(30일이면 반출 이름이 겹쳐 실제로 난다: 부하 150·2블록·2일에서 5건)")
    REPORT["admission"] = {"admitted": admitted, "skipped": len(v5_skip),
                           "no_target": sum(1 for _, r in v5_skip if r == "NO_TARGET")}


def test_day_phi_matches_v5():
    """날별 Φ 네 항 — 항1(그 날 트럭) · 항2·3·4(계수기 차분)."""
    L = _ladder3()
    for d, exp in zip(L["lay"].days, L["out5"]["days"]):
        got = MO.day_phi(L["run"], L["lay"], int(d.index), end_s=L["out5"]["sim_end"],
                         tape=L["tape"], t0=d.t0, t1=min(d.t1, L["lay"].month_s))
        assert (float(got.wait), float(got.move), float(got.rehandle), float(got.vessel)) == \
               (exp.wait, exp.move, exp.rehandle, exp.vessel), \
            f"day {d.index} Φ arr=({float(got.wait)}, {float(got.move)}, {float(got.rehandle)}, " \
            f"{float(got.vessel)}) v5=({exp.wait}, {exp.move}, {exp.rehandle}, {exp.vessel})"
        assert int(got.n_trucks) == exp.n_trucks and int(got.n_censored) == exp.n_censored
    # ★잠정 Φ — 하루가 끝나는 순간(그 시각까지 찍힌 기록만)의 값도 같아야 한다.
    #   여기서 `day_orders` 의 `sync_t` 규약(= v5 `bridge._sync` 의 '값 ≤ t 만 찍는다')이 걸린다.
    assert len(L["res"].live) == len(L["out5"]["live"])
    for a, b in zip(L["res"].live, L["out5"]["live"]):
        e = b["phi"]
        assert (a["phi_krw"], a["c_wait"], a["c_move"], a["c_rehandle"], a["c_vessel"],
                a["n_trucks"], a["n_censored"]) ==                (e.total, e.wait, e.move, e.rehandle, e.vessel, e.n_trucks, e.n_censored),             f"day {a['index']} 잠정 Φ arr={a} v5={e.as_dict()}"
    REPORT["phi"] = [round(x.total, 3) for x in L["out5"]["days"]]
    REPORT["phi_live"] = [round(x["phi"].total, 3) for x in L["out5"]["live"]]


def test_counter_tape_matches_v5():
    """계수기 사진첩 — YC 빈 주행·재조작·배별 유휴가 v5 `_MonthTape` 와 같은가."""
    L = _ladder3()
    for d in L["lay"].days:
        v_a, yc_a, rh_a = L["tape"].read(d.t0)
        v_5, yc_5, rh_5 = L["out5"]["tape"].read(d.t0)
        assert (yc_a, rh_a) == (yc_5, rh_5), f"t={d.t0} 계수기 arr=({yc_a},{rh_a}) v5=({yc_5},{rh_5})"
        assert v_a == v_5, f"t={d.t0} 배별 유휴 arr={v_a} v5={v_5}"


# ───────────────────────────────────────────────── ④ 2블록 · 3일 (경계 두 번)
def test_three_days_keep_carrying():
    """경계가 두 번 — 사흘째 세계가 이틀치 흔적을 들고 있는가."""
    L = _ladder4()
    assert set(L["snapA"]) == {0, 1, 2}
    for day in (0, 1, 2):
        t = L["lay"].days[day].t0
        for b, bid in enumerate(L["tt"].block_ids):
            _cmp_block(f"{L['label']} d{day}", bid, L["snapA"][day][bid], L["snap5"][day][bid],
                       t=t, pruned_ok=_pruned_ok_at(t))
    for b, bid in enumerate(L["tt"].block_ids):
        _cmp_block(f"{L['label']} final", bid, _array_block_snapshot(L["run"], L["tt"], b),
                   _v5_block_snapshot(L["out5"]["mbt"].blocks[bid]),
                   t=L["out5"]["sim_end"], pruned_ok=_pruned_ok_at(L["out5"]["sim_end"]))


# ───────────────────────────────────────────────── ⑤ prune/retire 는 무동작
def test_prune_and_retire_audits_match_v5():
    """★배열판이 아무것도 치우지 않아도 **치웠을 건수**는 v5 와 같다 (gpu/month 머리말의 주장)."""
    L = _ladder3()
    # ⚠️ v5 의 `a_sorted` 열은 배열판에 대응물이 없다 — 시간장부의 정렬 리스트를 닫힌 식으로 대체했고,
    #    그 값은 '지나간 A 시각 수' 라 나간 트럭 수와도 다르다 (실측 부하 150: v5 150 vs 149). 빼고 본다.
    keys = ("jobs", "ledger", "time_ledger")
    got = [{k: p[k] for k in keys} for p in L["res"].pruned]
    exp = [{k: p[k] for k in keys} for p in L["out5"]["pruned"]]
    assert got == exp, f"prune 감사 arr={got} v5={exp}"
    assert L["res"].retired == L["out5"]["retired"], \
        f"retire 감사 arr={L['res'].retired} v5={L['out5']['retired']}"
    assert sum(p["jobs"] for p in L["out5"]["pruned"]) > 0, "치울 것이 없는 무대 — 주장을 시험하지 못한다"


# ───────────────────────────────────────────────── ⑥ 적하 대상 — 메르센 셔플이 같은 목록을 낸다
def _v5_fresh_terminal(prof, days, built):
    """굴리지 않은 v5 `MonthTerminal` (t=0) — `free_targets`·`inject_vessel` 단독 대조용."""
    import dataclasses

    month_s = len(days) * v5m.DAY_S
    scns = {b: dataclasses.replace(s, jobs=[], vessels=[], horizon_s=month_s,
                                   drain_window_s=DIURNAL_DRAIN_S)
            for b, s in built["day0"]["scenarios"].items()}
    extra = tuple(i * EPOCH_S for i in range(int(month_s // EPOCH_S) + 1))
    return v5me.MonthTerminal({b: ensure_time_ledger(_sim_from(s, prof)) for b, s in scns.items()},
                              extra_review_epochs=extra)


def test_free_targets_matches_v5_at_t0():
    """★`free_targets` — 같은 야드·같은 시드면 **같은 목록**이어야 한다 (순서까지).

    시드 고정 메르센 셔플은 `sorted(야드 상자 이름)` 에 걸려 있다. 배열판이 상자 이름을 하나라도
    다르게 붙이면 여기서 무너진다 (그래서 양하 상자 이름을 `IN_{본선작업 id}` 로 맞춰 둔다).
    """
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04"), 2, 150, cap_moves=24)
    mbt = _v5_fresh_terminal(prof, days, built)
    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout)
    run = MO.make_month_run(tw, tt, Geom.from_profile(prof))
    for b, bid in enumerate(tt.block_ids):
        for limit, tag in ((5, "a"), (137, "b")):
            exp = v5me.free_targets(mbt.blocks[bid], limit=limit, seed=f"seed:{tag}")
            got = MO.free_targets(run, tt, b, limit=limit, seed=f"seed:{tag}")
            assert got == exp, (f"{bid} limit={limit} 대상 목록이 갈린다 — 첫 차이 "
                                f"{_first_diff(got, exp)} (arr {len(got)} · v5 {len(exp)})")
        assert sorted(MO.block_boxes(run, tt, b)) == sorted(mbt.blocks[bid].stacks.containers),             f"{bid} 야드 상자 이름 집합이 다르다"


def test_vessel_shortage_is_clipped_like_v5():
    """★재고보다 많이 실으라고 하면 **깎이고 사유가 남는다** — v5 와 같은 수·같은 사유.

    `inject_vessel` 의 적하 갈래에서만 나는 일이라 계획 그대로는 잘 안 밟힌다 (한 스트림이 235~543 상자,
    블록 재고가 648). 그래서 물량을 재고보다 크게 만들어 그 갈래를 직접 밟는다.
    """
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04"), 2, 150, cap_moves=24)
    bid = "Y01"
    rows = {0: [dict(r, work="LOAD", moves=5_000) for r in v_by_day[0] if r["block"] == bid],
            1: []}
    assert rows[0], "Y01 에 첫날 본선이 없다"
    key = rows[0][0]["key"]
    mbt = _v5_fresh_terminal(prof, days, built)
    a5 = v5me.inject_vessel(mbt, bid, rows[0][0], key=key,
                            size_seed=f"v3:month:{SEED}:{key}",
                            deadline_mult=v5me.VESSEL_DEADLINE_MULT)
    tw, tt, lay = MO.to_month_world(prof, built, rows, days=days, seed=SEED, layout=layout)
    run = MO.make_month_run(tw, tt, Geom.from_profile(prof))
    run, aa = MO.inject_vessel(run, tt, lay, key=key, seed=SEED, profile=prof,
                               deadline_mult=MO.VESSEL_DEADLINE_MULT)
    assert (aa.moves, aa.asked_moves, aa.reason, aa.start_s, aa.planned_completion_s) ==            (a5.moves, a5.asked_moves, a5.reason, a5.start_s, a5.planned_completion_s),         f"깎인 결과가 갈린다 arr={aa} v5={a5}"
    assert aa.clipped and aa.moves < 5_000
    # 붙은 오더 행이 v5 job 과 같은 대상·해제시각인가
    b = lay.block_of_key[key]
    o = MB.tree_to_numpy(run.tw.blocks.orders)
    names = tt.tables[b].cont_ids
    for m, n in enumerate(lay.rows[(b, key)][:aa.moves]):
        jid = f"{bid}:J-{key}-{m:04d}"
        j = mbt.blocks[bid].jobs[jid]
        assert names[int(o.target_cont[b, n])] == j.target_container, f"{jid} 대상"
        assert float(o.release_s[b, n]) == j.release_time, f"{jid} 해제시각"
        assert float(o.deadline_s[b, n]) == j.deadline, f"{jid} 마감"
    REPORT["clipped"] = {"asked": aa.asked_moves, "moves": aa.moves, "why": aa.reason}


def test_split_run_matches_one_shot():
    """★에폭을 둘로 나눠 이어 돌린 결과가 한 번에 돌린 것과 같다 — 30일은 세션을 나눠야 한다.

    `MonthResult.state` 가 날 번호·사진 시각·이미 센 행을 들고 있으므로 같은 `res`/`tape` 를 다시
    주면 이어진다. 잎을 **비트로** 맞춘다 (값 비교는 dtype·−0.0 을 못 잡는다).
    """
    L = _ladder3()
    prof = L["prof"]
    tw2, tt2, lay2 = MO.to_month_world(prof, L["built"], L["v_by_day"], days=L["days"], seed=SEED,
                                       layout=L["layout"])
    g = Geom.from_profile(prof)
    run2 = MO.make_month_run(tw2, tt2, g)
    k0 = tt2.tables[0].crane_index[prof.cranes[0].crane_id]
    eng = MO.month_engine(tt2, g, policy_fn=DP.make_resolver("sf_spt", g, count_lost=False),
                          horizon_s=float(prof.decision_horizon_s), k0=int(k0))
    cut = run2.n_epochs // 2
    res = MO.MonthResult()
    tape = MO.MonthTape(lay2, prof)
    run2, res, tape = MO.run_month(run2, tt2, lay2, eng, seed=SEED, profile=prof, e0=0, e1=cut,
                                   finish=False, res=res, tape=tape)
    run2, res, tape = MO.run_month(run2, tt2, lay2, eng, seed=SEED, profile=prof, e0=cut,
                                   finish=True, res=res, tape=tape)
    la = jax.tree_util.tree_leaves(MB.tree_to_numpy(L["run"]))
    lb = jax.tree_util.tree_leaves(MB.tree_to_numpy(run2))
    bad = []
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(L["run"])]
    for i, (x, y) in enumerate(zip(la, lb)):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(f"{names[i]}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        elif x.dtype.kind == "f":
            xb = x.view({4: np.uint32, 8: np.uint64}[x.dtype.itemsize])
            yb = y.view({4: np.uint32, 8: np.uint64}[y.dtype.itemsize])
            if not np.array_equal(xb, yb):
                bad.append(f"{names[i]}: float 비트 {int((xb != yb).sum())}칸")
        elif not np.array_equal(x, y):
            bad.append(f"{names[i]}: {int((x != y).sum())}칸")
    assert not bad, f"나눠 돌린 상태가 갈린다: {bad[:8]}"
    assert [{k: v for k, v in d.items()} for d in res.vessel_admissions] == L["res"].vessel_admissions
    assert res.pruned == L["res"].pruned and res.retired == L["res"].retired


# ───────────────────────────────────────────────── ⑦ 배별 유휴 — 합의 **순서**까지
def test_vessel_idle_sum_order_matches_v5_with_three_or_more_streams():
    """★한 배에 스트림이 **셋 이상** 모이는 무대에서 배별 유휴·Φ·세계가 v5 와 같다.

    왜 셋이 중요한가 — 파이썬 3.12 `sum()` 은 실수 목록을 Neumaier 보정합으로 더하므로 세 항부터
    순서가 마지막 비트를 바꿀 수 있다. v5 순서는 `archive`(치운 순) → 블록 dict 순 × `sim.vessels`
    삽입 순이고, 배열판은 그 순서를 `live_vessel_order` + `MonthTape.archive` 로 재현한다.

    ⚠️ **솔직히**: 이 무대의 유휴 값들은 순서를 뒤집어도 같은 합을 낸다 (실측 — 값이 그렇게 생겼다).
    그래서 순서 규칙 자체는 아래 `test_live_vessel_order_matches_v5` 와 **치운 순서 목록 대조**가
    지킨다. 이 시험은 "셋 이상인 무대에서도 값이 전부 같다" 를 못 박는 쪽이다.
    """
    L = _ladder5()
    streams = {}
    for rows in L["v_by_day"].values():
        for r in rows:
            streams.setdefault(r["ship"], []).append(r["key"])
    assert any(len(v) >= 3 for v in streams.values()), (
        f"스트림이 셋 이상인 배가 없다 — 합의 순서를 시험하지 못한다: "
        f"{ {k: len(v) for k, v in streams.items()} }")
    for d in L["lay"].days:
        v_a, yc_a, rh_a = L["tape"].read(d.t0)
        v_5, yc_5, rh_5 = L["out5"]["tape"].read(d.t0)
        assert (yc_a, rh_a) == (yc_5, rh_5), f"t={d.t0} 계수기 arr=({yc_a},{rh_a}) v5=({yc_5},{rh_5})"
        assert v_a == v_5, (f"t={d.t0} 배별 유휴가 갈린다 — "
                            f"{[(k, v_a.get(k), v_5.get(k)) for k in set(v_a) | set(v_5) if v_a.get(k) != v_5.get(k)]}")
    # ★치운 순서 자체를 못 박는다 — 이것이 archive 구간의 합 순서다 (v5 dict 삽입 순서)
    assert list(L["tape"].archive) == list(L["out5"]["archive"]), (
        f"치운 순서가 갈린다 arr={list(L['tape'].archive)} v5={list(L['out5']['archive'])}")
    # ★달 끝 사진 — 여기서는 archive 가 차 있어 '치운 순 → 블록 순 × 붙은 순' 이 실제로 쓰인다
    ms = L["lay"].month_s
    assert L["tape"].read(ms) == L["out5"]["tape"].read(ms), (
        f"t={ms} 사진이 갈린다 arr={L['tape'].read(ms)} v5={L['out5']['tape'].read(ms)}")
    for d, exp in zip(L["lay"].days, L["out5"]["days"]):
        got = MO.day_phi(L["run"], L["lay"], int(d.index), end_s=L["out5"]["sim_end"],
                         tape=L["tape"], t0=d.t0, t1=min(d.t1, ms))
        assert (float(got.wait), float(got.move), float(got.rehandle), float(got.vessel)) ==                (exp.wait, exp.move, exp.rehandle, exp.vessel), f"day {d.index} Φ 갈림"
    for day in (0, 1):
        t = L["lay"].days[day].t0
        for b, bid in enumerate(L["tt"].block_ids):
            _cmp_block(f"{L['label']} d{day}", bid, L["snapA"][day][bid], L["snap5"][day][bid],
                       t=t, pruned_ok=_pruned_ok_at(t))
    for b, bid in enumerate(L["tt"].block_ids):
        _cmp_block(f"{L['label']} final", bid, _array_block_snapshot(L["run"], L["tt"], b),
                   _v5_block_snapshot(L["out5"]["mbt"].blocks[bid]),
                   t=L["out5"]["sim_end"], pruned_ok=_pruned_ok_at(L["out5"]["sim_end"]))
    REPORT["six_blocks"] = {"ships": {k: len(v) for k, v in streams.items()},
                            "epochs": int(L["res"].n_epochs),
                            "retired_order": len(L["tape"].archive)}


def test_live_vessel_order_matches_v5():
    """★`live_vessel_order` = v5 `for sim in mbt.blocks.values(): for key in sim.vessels` 순서.

    이것이 배별 유휴 합의 순서를 정한다 (`month_vessel_idle`). **붙는 순서와 다르다** —
    붙는 순서는 날마다 블록을 훑지만, v5 의 순회는 블록이 겉 루프다.
    """
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04", "Y05", "Y12", "Y15", "Y16"), 2, 60,
                                                 cap_moves=12)
    mbt = _v5_fresh_terminal(prof, days, built)
    for d in days:
        for r in v_by_day.get(d.index, []):
            v5me.inject_vessel(mbt, r["block"], r, key=r["key"],
                               size_seed=f"v3:month:{SEED}:{r['key']}",
                               deadline_mult=v5me.VESSEL_DEADLINE_MULT)
    exp = [k for sim in mbt.blocks.values() for k in sim.vessels]
    tw, tt, lay = MO.to_month_world(prof, built, v_by_day, days=days, seed=SEED, layout=layout)
    got = list(MO.live_vessel_order(lay))
    assert got == exp, f"live 순서가 갈린다 arr={got} v5={exp}"
    assert got != list(MO._inject_order(lay)), (
        "붙는 순서와 같아서 순서 규칙을 시험하지 못한다 — 블록이 겉 루프인 무대를 써야 한다")


def test_two_trucks_same_epoch_same_target_lose_the_second():
    """★같은 에폭·같은 블록에 **같은 대상**의 반출 트럭이 둘 오면 뒤 트럭이 `NO_TARGET` 이다.

    v5 는 트럭마다 `sim.jobs` 를 새로 읽으므로 앞 트럭이 들어간 뒤 그 상자가 `taken` 에 들어간다.
    배열판의 `admit_block` 은 "이미 찍혔나" 를 안 보므로 그냥 두면 **둘 다 들어간다** (같은 상자를
    두 기사에게 준다). 그래서 `no_target_skips` 가 에폭 안에서 `taken` 을 키워 간다 — 여기서 못 박는다.

    30일이면 반출 이름이 날마다 겹쳐 자연히 생기는 상황이지만(YR-239), 이틀짜리 무대에서는 드물어
    명단에 **쌍둥이 한 줄을 심어** 그 갈래를 직접 밟는다 (v5·배열에 같은 명단을 준다).
    """
    prof, layout, days, built, v_by_day = _build(("Y01", "Y04"), 2, 60, cap_moves=12)
    sched = built["schedule"]
    # 첫날 이른 시각의 반출 트럭 하나를 골라 **같은 대상·같은 통지 에폭**의 쌍둥이를 만든다
    d0 = [e for e in sched if e["flow"] == "GATE_OUT" and e.get("day", 0) == 0]
    assert d0, "첫날 반출 트럭이 없다"
    src = min(d0, key=lambda e: e["arrival_s"])
    twin = dict(src)
    twin["job_id"] = src["job_id"] + "T"
    twin["con_no"] = src["con_no"]
    sched2 = sorted(sched + [twin], key=lambda e: e["arrival_s"])
    built2 = dict(built, schedule=sched2)
    stop_t = src["arrival_s"] + 600.0

    out5 = _v5_month(prof, layout, days, built2, v_by_day)
    v5_no = [x["job_id"] for x in out5["ann"].skips
             if x["reason"] == "NO_TARGET" and x["t"] <= stop_t]
    assert len(v5_no) == 1 and v5_no[0] in (src["job_id"], twin["job_id"]), (
        f"v5 가 쌍둥이 하나를 떨어뜨리지 않았다 — skips={out5['ann'].skips[:5]}")

    tw, tt, lay = MO.to_month_world(prof, built2, v_by_day, days=days, seed=SEED, layout=layout)
    g = Geom.from_profile(prof)
    run = MO.make_month_run(tw, tt, g)
    k0 = tt.tables[0].crane_index[prof.cranes[0].crane_id]
    eng = MO.month_engine(tt, g, policy_fn=DP.make_resolver("sf_spt", g, count_lost=False),
                          horizon_s=float(prof.decision_horizon_s), k0=int(k0))
    n_ep = int(stop_t // MO.EPOCH_S) + 2
    run, res, tape = MO.run_month(run, tt, lay, eng, seed=SEED, profile=prof, e0=0, e1=n_ep,
                                  finish=False)
    ar_no = [x["job_id"] for x in res.truck_skips if x["t"] <= stop_t]
    assert ar_no == v5_no, f"쌍둥이 처리가 갈린다 arr={ar_no} v5={v5_no}"
    # 그리고 **한 상자를 두 기사에게 주지 않았다** — 대상이 겹치는 붙은 행이 없다
    o = MB.tree_to_numpy(run.tw.blocks.orders)
    for b in range(lay.b):
        tgt = o.target_cont[b][(o.block[b] >= 0) & (o.target_cont[b] >= 0)]
        assert len(set(int(x) for x in tgt)) == len(tgt), f"블록 {b} 에 대상이 겹치는 오더가 있다"
    REPORT["twin_target"] = {"dropped": ar_no}


def test_report(capsys):
    if REPORT:
        with capsys.disabled():
            print("\n[REPORT] gpu/month", REPORT)
