"""★다중블록 조정자(gpu/multiblock.py) 가 v5 `MultiBlockTerminal.run` + `ScheduledAnnouncer` 와 **같은 답**을 낸다 ([[YR-327]] 조각 6 · key=integrate).

■ 사다리 (piece6_spec equivalence_test — 앞이 깨지면 그 자리에서 **처음 갈리는 사건(블록·시각·종류)** 을 보고한다)
  ① y01-20     layout Y01 · obs (0, 7200, 300) · build_diurnal(load 20 · 본선 0 · drain 1200) · 크레인 2 · lead 600
                v5 = MultiBlockTerminal(ensure_time_ledger(TerminalSimulator; PRE_ADVICE), extra=admission_epochs(obs))
                     + ScheduledAnnouncer(lead) + ResolverPolicy(SF_SPT) + CandidateGenerator(LEGACY_DEFAULT)
                배열 = to_terminal_world → make_run → run_all (scan × vmap × while_loop) → finish_run
                대조: 블록 event_log 전열 == · 해시 · 비용 13항 == · KPI == · ADMIT/SKIP 원장 순열 == · n_admitted ==
                      · a_to_o 표본 == (정렬열·합) · terminal_total/totals == · JobRecord.locked == · 오더별 상태·시각 ==
                + 세션 이어 돌리기: 에폭을 둘로 나눠 numpy 저장·복원해 이어 돌린 결과 == 한 번에 돌린 결과 (잎 전부)
  ② y01y21-40  Y01+Y21 · 40대 · lead 1800 · v5 review 훅이 'A > t+1e-6 · PLANNED · 미lock 인 첫 반입 트럭' 을 t ≥ 1800 의 첫
                검토 시각에 try_pre_gate_transfer (dst = 다른 블록), 그 뒤 t ≥ 2400 의 첫 검토 시각에 try_defer(600s) — 배열은
                같은 (t, 트럭, 목적지) 를 run_python 의 review_fn 으로 재생. 대조: 두 시각의 후보 집합 == · owner/version/
                transfer_count/transfer_history/entry_deferrals/entry_deferred_s/route_cost_s == · 끝에 ① 과 같은 항목.
  ③ terminal   **정답 JSON** outputs/reports/yr327_v6_port/ground_truth/terminal_load{30,300}_seed9900777.json
                (scripts/v6/dump_ground_truth.py run_terminal — 21블록 · OBS_24H · lead 1800): 블록별 event_hash 21/21 ·
                사건 전열 · cost_raw 13항 · KPI 10항 · n_jobs · unfinished · admitted · n_turns · turn_samples_s · turn_sum_s
                **+ (두꺼운 정답, 2026-09-26) 투입 원장 전행 · locked 전건 · 오더별 (status·크레인·재처리·S·C·A·B·O) ·
                   시간장부 적분 3항 · totals · 이연 원장** — 이 넷은 예전 정답에 없어 21블록 규모에서 한 번도 대조되지
                   않았다(검증 지적). v5 를 세션 안에서 다시 굴리지 않고도 같은 항목을 보게 정답을 두껍게 했다.
                ★한 세션(WSL 80초)에 안 든다 — 세션을 나눠 잇는다 (아래 ■ 이어 돌리기).

■ 이어 돌리기 (③) — 환경 변수
    TERMINAL_SESSION_S   이번 세션이 에폭 실행에 쓸 초 (기본 40). 넘으면 상태를 저장하고 `skip("이어 돌리기 e/E")` 로 끝난다 —
                         다시 실행하면 저장된 상태에서 이어 간다. 전부 돌면 정답과 대조하고 상태 파일을 지운다.
    TERMINAL_STATE_DIR   상태·컴파일 캐시 디렉터리 (기본 outputs/v6/verify/terminal_equiv — /tmp 는 WSL 재시작에 사라진다)
    TERMINAL_LOADS       "30,300" (기본) — 돌릴 정답 부하
  드라이버: scripts/v6/verify_chunked.sh 와 같은 방식으로 `pytest -k ladder3` 를 세션이 죽을 때마다 다시 부르면 된다.
  ★실측 (부하 300 · 1,441 에폭): GPU(RTX 5090) **92 ms/에폭** (2026-09-26 완주 · 세션별 73~95) ·
  CPU x64 최선 412 · 완주 400 → GPU 세션 4~5개(예산 30~40초) · CPU 세션 20여 개. 부하 30 은 GPU 70 ms/에폭. 
  (예전 머리말의 "GPU 41 · CPU 250" 은 어떤 로그에도 남아 있지 않아 지웠다 — 검증 지적.)
  ★v5 대비 속도: **단일 터미널은 v5 파이썬이 더 빠르다** — 같은 21블록·사건 20,720 을 16~17초에 완주한다(이 파일의
  정답을 만든 그 코드). 배열판은 GPU 약 133초 · CPU 576~673초로 8~39배 느리다. lane 이 21개뿐이라 GPU 가 노는
  탓이며, 조각 6 의 목적(동등성)에는 무관하지만 **성능 근거로 인용하면 안 된다**. 이득은 세계를 쌓을 때 난다.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import admission as AD                                          # noqa: E402
from yard_rl.v6.gpu import dispatch as DP                                           # noqa: E402
from yard_rl.v6.gpu import host_terminal as HT                                      # noqa: E402
from yard_rl.v6.gpu import multiblock as MB                                         # noqa: E402
from yard_rl.v6.gpu import transfer_txn as TX                                       # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.state import COST_TERMS, FL_GATE_IN, JS_PLANNED                # noqa: E402
from yard_rl.v6.world.domain.enums import InformationLevel, JobStatus              # noqa: E402
from yard_rl.v6.world.integrated import (baselines as bl, candidates as cd, engine as eng,   # noqa: E402
                                          multiblock as mb, policy_config as pc, profiles as pr,
                                          terminal_stream as ts, time_sell, yard_layout as yl)

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_TRUTH_DIR = os.path.join(_ROOT, "outputs", "reports", "yr327_v6_port", "ground_truth")
SEED = 9_900_777
LVL = InformationLevel.PRE_ADVICE
OBS_2H = ts.ObservationContract(warmup_s=0.0, measure_s=7_200.0, snapshot_s=300.0)
KPI_KEYS = ("queue_area_s", "tail_area_s", "loaded_gantry_m", "empty_gantry_m", "rehandle_count", "pre_rehandle_count",
            "completed_external", "completed_vessel", "vessel_delay_s", "positioning_count")
REPORT: dict[str, dict] = {}


# ───────────────────────────────────────────────── 무대 (v5 · 배열) — 기대값 손기입 없음
def _sim(prof, scn):
    s = eng.TerminalSimulator(prof, scn, check_invariants=True)
    s.info_level = LVL
    return ts.ensure_time_ledger(s)


def _rule_policy():
    """SF_SPT 규칙 + LEGACY 후보 (dump_ground_truth._rule_policy · test_world_equivalence.py:49-54)."""
    gens: dict[int, object] = {}
    pol = bl.ResolverPolicy(bl.ServiceFirstSPTPreference(), "SF")

    def exec_policy(sim, dp):
        g = gens.setdefault(id(sim), cd.CandidateGenerator(config=pc.LEGACY_DEFAULT))
        gb = {c: g.generate(sim, c, LVL) for c in dp.crane_ids}
        bl._apply(sim, pol.decide(sim, dp, gb))
    return exec_policy


def _build_small(blocks, load: int):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout().subset(blocks)
    built = ts.build_diurnal(prof, SEED, obs=OBS_2H, layout=layout, params=ts.TerminalStreamParams(load_4h=load),
                             day_total=load, n_streams=0, drain_s=1_200.0, background_seed=SEED)
    return prof, layout, built


def _build_terminal(load: int):
    prof, layout = pr.build_h21_profile(), yl.terminal_layout()
    built = ts.build_diurnal(prof, SEED, obs=ts.OBS_24H, layout=layout, params=ts.TerminalStreamParams(load_4h=load),
                             day_total=load, background_seed=SEED)
    return prof, layout, built


def _v5_terminal(prof, built, obs, lead: float):
    mbt = mb.MultiBlockTerminal({b: _sim(prof, s) for b, s in built["scenarios"].items()},
                                extra_review_epochs=ts.admission_epochs(obs))
    ann = ts.ScheduledAnnouncer(built["schedule"], lead_s=lead, end_s=built["sim_end_s"])
    return mbt, ann


def _array_setup(prof, built, obs, lead: float, *, n_spare: int = 0):
    """to_terminal_world → make_run + Engine (정답 궤적 Y01 시험과 같은 정책 조합: sf_spt · PRE_ADVICE · joint)."""
    tw, tt = HT.to_terminal_world(prof, built, lead_s=lead, extra_review_epochs=ts.admission_epochs(obs), n_spare=n_spare)
    g = Geom.from_profile(prof)
    run0 = MB.make_run(tw, tt, params=MB.terminal_resolver_params(tt, g))
    k0 = tt.tables[0].crane_index[prof.cranes[0].crane_id]
    engine = MB.Engine(g=g, policy_fn=DP.make_resolver("sf_spt", g, count_lost=False), check=True, pre_advice=True,
                       horizon_s=float(prof.decision_horizon_s), joint=True, margin=mb.CAPACITY_MARGIN,
                       end_ann=float(built["sim_end_s"]), steps_per_epoch=8 * tt.n_max + 256,
                       steps_final=8 * tt.n_max + 256, k0=int(k0))
    return tw, tt, run0, engine


def _schedule_dicts(tw, tt):
    blk = np.asarray(tw.sched.block); fl = np.asarray(tw.sched.flow); arr = np.asarray(tw.sched.arrival_s)
    return [{"job_id": tt.truck_ids[s], "block": tt.block_ids[int(blk[s])],
             "flow": "GATE_IN" if int(fl[s]) == FL_GATE_IN else "GATE_OUT", "arrival_s": float(arr[s])}
            for s in range(tw.sched.s)]


def _ledger_rows(run, tw, tt, codes) -> list[dict]:
    """에폭별 코드 (E,B,M) → v5 announcer 원장 모양 (명단 순서 · EPOCH 행)."""
    sched = _schedule_dicts(tw, tt)
    rows: list[dict] = []
    ep = np.asarray(tw.epochs.t)
    for e in range(codes.shape[0]):
        t = float(ep[e])
        rows.extend(AD.ledger_rows(np.asarray(codes[e]), run.adm, AD.slot_of(t, tt.period_s), t, sched))
    return rows


def _bit_diff(run_a, run_b):
    """두 상태의 잎을 **dtype + 비트**로 비교한다 (값 비교가 아니다).

    왜: `np.array_equal` 은 dtype 을 안 보고 −0.0 == +0.0, NaN payload 차이도 통과시킨다. 이 프로젝트는
    dispatch.py 에서 −0.0 을 일부러 +0.0 으로 접을 만큼 그 구분에 민감하고, x64 가 꺼진 세션에서 복원하면
    float64 잎이 조용히 float32 가 되는데 값 비교는 그것도 못 잡는다 (검증 지적). float 은 같은 폭 정수로
    보고(view) 비트를 맞춘다. 반환: 어긋난 잎 이름 목록.
    """
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(run_a)]
    la, lb = jax.tree_util.tree_leaves(run_a), jax.tree_util.tree_leaves(run_b)
    assert len(la) == len(lb), f"잎 수가 다르다 {len(la)} vs {len(lb)}"
    bad = []
    for i, (x, y) in enumerate(zip(la, lb)):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape:
            bad.append(f"{names[i]}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
            continue
        if x.dtype.kind == "f":
            xb = x.view({4: np.uint32, 8: np.uint64}[x.dtype.itemsize])
            yb = y.view({4: np.uint32, 8: np.uint64}[y.dtype.itemsize])
            if not np.array_equal(xb, yb):
                bad.append(f"{names[i]}: float 비트 {int((xb != yb).sum())}칸")
        elif not np.array_equal(x, y):
            bad.append(f"{names[i]}: {int((x != y).sum())}칸")
    return bad


def _replay_split(run0, engine, cuts, E, tmp_path, tag):
    """에폭을 `cuts` 로 나눠 **매 경계마다 numpy 로 저장·복원** 하며 끝까지 — 반환 (run, codes)."""
    run, parts, e = run0, [], 0
    for i, n in enumerate(cuts):
        run, c = MB.run_epochs_jit(run, e, n, engine)
        parts.append(np.asarray(c))
        e += n
        p = tmp_path / f"{tag}_{i}.pkl"
        MB.save_run(p, run)
        run = MB.load_run(p)
        assert int(run.tw.epoch_idx) == e, f"[{tag}] 복원 커서 {int(run.tw.epoch_idx)} ≠ {e}"
    assert e == E, f"[{tag}] 분할 합 {e} ≠ 에폭 수 {E}"
    return MB.finish_run_jit(run, engine), np.concatenate(parts, axis=0)


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), (b[len(a)] if len(b) > len(a) else None)
    return None


def _fail_first_event_diff(label, bid, arlog, v5log, extra=""):
    diff = _first_diff(arlog, v5log)
    if diff is None:
        return
    i, x, y = diff
    ctx = "\n".join(f"    #{j}: arr={arlog[j] if j < len(arlog) else '-'}  |  v5={v5log[j] if j < len(v5log) else '-'}"
                    for j in range(max(0, i - 4), min(max(len(arlog), len(v5log)), i + 4)))
    pytest.fail(f"[{label}] 블록 {bid} 사건 로그가 {i}번째에서 갈린다 (시각 {x[0] if x else y[0]} · 종류 {x[1] if x else y[1]}): "
                f"arr={x} v5={y}\n{ctx}\n  (arr {len(arlog)}건 · v5 {len(v5log)}건){extra}")


def _compare_with_v5(label, d, mbt, ann, tt, rows, obs_end: float, out5: dict) -> dict:
    """배열 결과 dict (from_run) ↔ 지금 굴린 v5 — ①~⑥ 전부. 반환 보고용 요약."""
    # ① 사건 로그 전열 · 해시 · 비용 · KPI · 오더
    n_events = 0
    for b, bid in enumerate(tt.block_ids):
        sim = mbt.blocks[bid]
        blk = d["blocks"][bid]
        arlog = [tuple(x) for x in blk["events"]]
        v5log = [(round(t, 6), k, p) for (t, k, p) in sim.event_log]
        _fail_first_event_diff(label, bid, arlog, v5log,
                               extra=f" violation={blk['violation_names']} overflow={blk['overflow']} terminal={blk['terminal']}")
        assert blk["event_hash"] == sim.event_stream_hash(), f"[{label}] {bid} 해시"
        assert blk["violation"] == 0 and blk["overflow"] == 0 and blk["terminal"], \
            f"[{label}] {bid} violation={blk['violation_names']} overflow={blk['overflow']} terminal={blk['terminal']}"
        bad = [(t, blk["cost_raw"][t], sim.cost.episode_raw()[t]) for t in COST_TERMS if blk["cost_raw"][t] != sim.cost.episode_raw()[t]]
        assert not bad, f"[{label}] {bid} 비용 항목 (항, arr, v5): {bad}"
        k = sim.kpis
        exp_k = {"queue_area_s": k.queue_area_s, "tail_area_s": k.tail_area_s, "loaded_gantry_m": k.loaded_gantry_m,
                 "empty_gantry_m": k.empty_gantry_m, "rehandle_count": k.rehandle_count,
                 "pre_rehandle_count": k.pre_rehandle_count, "completed_external": k.completed_external,
                 "completed_vessel": k.completed_vessel, "vessel_delay_s": k.vessel_delay_s,
                 "positioning_count": k.positioning_count}
        assert blk["kpis"] == exp_k, f"[{label}] {bid} KPI arr={blk['kpis']} v5={exp_k}"
        assert blk["n_jobs"] == len(sim.jobs) and blk["unfinished"] == sim.unfinished_backlog() and blk["end_s"] == sim.end
        assert blk["clock_s"] == sim.clock and blk["deadlock_escapes"] == sim.deadlock_escape_count
        tl = sim.time_ledger
        assert (blk["ledger"]["terminal_area_s"], blk["ledger"]["block_area_s"], blk["ledger"]["block_tail_area_s"]) == \
               (tl.terminal_area_s, tl.block_area_s, tl.block_tail_area_s), f"[{label}] {bid} 장부 적분"
        assert set(blk["jobs"]) == set(sim.jobs), f"[{label}] {bid} 작업 집합 arr-only={set(blk['jobs']) - set(sim.jobs)} v5-only={set(sim.jobs) - set(blk['jobs'])}"
        for jid, j in sim.jobs.items():
            a = blk["jobs"][jid]
            r = tl.records.get(jid)
            got = (a["status"], a["assigned_crane"], a["rehandle_count"], a["service_start"], a["service_end"],
                   a["gate_in"], a["block_arrival"], a["actual_gate_out"])
            exp = (j.status.name, j.assigned_crane, j.rehandle_count, j.service_start, j.service_end,
                   r.gate_in if r else None, r.block_arrival if r else None, r.gate_out if r else None)
            assert got == exp, f"[{label}] {bid} 오더 {jid}: arr={got} v5={exp}"
        n_events += len(arlog)
    # ② 투입 원장 순열 · n_admitted
    assert d["admitted"] == ann.n_admitted, f"[{label}] 투입 수 arr={d['admitted']} v5={ann.n_admitted}"
    assert len(rows) == len(ann.ledger), f"[{label}] 원장 길이 arr={len(rows)} v5={len(ann.ledger)}"
    for i, (a, b) in enumerate(zip(ann.ledger, rows)):
        assert a["event"] == b["event"] and a.get("job_id") == b.get("job_id"), f"[{label}] 원장 {i}번째: v5={a} arr={b}"
        if a["event"] == "ADMIT":
            assert (a["block"], a["flow"], a["arrival_s"], a["t"]) == (b["block"], b["flow"], b["arrival_s"], b["t"])
        elif a["event"] == "SKIP":
            assert AD.code_of_reason(a["reason"]) == b["code"], f"[{label}] 원장 {i} SKIP 사유 v5={a['reason']!r} arr={b['reason_code']}"
    # ③ a_to_o 표본 (정렬열 · 파이썬 sum) ④ totals ⑥ 원장 locked/owner
    turns5 = sorted(mbt.ledger.a_to_o_samples_s(obs_end))
    assert d["n_turns"] == len(turns5) and d["turn_samples_s"] == [round(t, 6) for t in turns5], \
        f"[{label}] 턴 표본 arr(n={d['n_turns']}) v5(n={len(turns5)}) 첫 차이 {_first_diff(d['turn_samples_s'], [round(t, 6) for t in turns5])}"
    assert d["turn_sum_s"] == round(sum(turns5), 6)
    assert d["terminal_total"] == out5["terminal_total"] and d["end"] == out5["end"] and d["route_cost_s"] == out5["route_cost_s"]
    assert d["totals"] == out5["totals"]
    recs = mbt.ledger.records
    lk5 = {j: r.locked for j, r in recs.items() if j in tt.truck_index}
    assert d["locked"] == lk5, f"[{label}] locked 갈림 {[(j, d['locked'].get(j), lk5.get(j)) for j in lk5 if d['locked'].get(j) != lk5[j]][:5]}"
    assert d["exhausted"] == [0] * len(tt.block_ids) and d["open_overflow"] == 0 and d["hist_overflow"] == 0
    return dict(events=n_events, admitted=d["admitted"], turns=d["n_turns"], turn_sum=d["turn_sum_s"],
                ledger_rows=len(rows), locked=sum(lk5.values()))


# ───────────────────────────────────────────────── ① Y01 · 20대 · lead 600
def test_ladder1_y01_20_matches_v5(tmp_path):
    label = "ladder1-y01-20"
    prof, layout, built = _build_small(("Y01",), 20)
    mbt, ann = _v5_terminal(prof, built, OBS_2H, 600.0)
    t0 = time.perf_counter()
    out5 = mbt.run(_rule_policy(), review_fn=ann.review)
    t_v5 = time.perf_counter() - t0
    tw, tt, run0, engine = _array_setup(prof, built, OBS_2H, 600.0)
    E = run0.n_epochs
    assert E == len(ts.admission_epochs(OBS_2H)) == 121
    t0 = time.perf_counter()
    run, codes = MB.run_all(run0, engine)
    jax.block_until_ready(run)
    t_arr = time.perf_counter() - t0
    d = MB.from_run(run, tt)
    rows = _ledger_rows(run, tw, tt, np.asarray(codes))
    summ = _compare_with_v5(label, d, mbt, ann, tt, rows, OBS_2H.observe_s, out5)
    # + 세션 이어 돌리기 — **청크 크기가 답을 바꾸지 않는다**를 분할점 셋으로 건다 (검증 지적: 예전엔 55|66 한 곳뿐이고
    #   비교도 값 비교라 dtype·−0.0·NaN payload 차이를 통과시켰다). 비교는 `_bit_diff` (dtype + float 비트).
    #   청크 크기는 jit 의 static 인자라 새 크기마다 다시 컴파일한다 — 55·66(옛 분할)·121(run_all 이 이미 굽는다)
    #   세 가지만 써서 경계는 바꾸되 컴파일은 안 늘린다.
    for tag, cuts in (("55|66", (55, 66)), ("66|55", (66, 55)), ("121", (121,))):
        rb, cb = _replay_split(run0, engine, cuts, E, tmp_path, tag.replace("|", "_"))
        bad = _bit_diff(run, rb)
        assert not bad, f"[{label}] 분할 {tag} 로 나눠 돌린 결과가 다른 잎 {bad}"
        assert np.array_equal(np.asarray(codes), cb), f"[{label}] 분할 {tag} 투입 코드"
    REPORT[label] = dict(**summ, v5_s=round(t_v5, 2), arr_s=round(t_arr, 2), epochs=E, blocks=1, device=jax.devices()[0].platform)


# ───────────────────────────────────────────────── ② Y01+Y21 · 40대 · 이송 1건 · 이연 1건
def _v5_candidates(mbt, t, *, pre_gate: bool, flow: str | None, owner: str | None = None) -> list[str]:
    """등록·PLANNED·미lock 이고 (pre_gate: A > t+1e-6) 인 작업 id 정렬열 (tests/v6/test_gpu_transfer_txn.py 와 같은 규칙)."""
    out = []
    for jid in sorted(mbt.ledger.records):
        rec = mbt.ledger.records[jid]
        if rec.locked or (owner is not None and rec.owner != owner) or (flow is not None and rec.flow != flow):
            continue
        j = mbt.blocks[rec.owner].jobs.get(jid)
        if j is None or j.status != JobStatus.PLANNED or rec.a_gate_in is None:
            continue
        if pre_gate != (rec.a_gate_in > t + 1e-6):
            continue
        out.append(jid)
    return out


def _array_candidates(run, tt, t, *, flow_in: bool, owner: int | None = None) -> list[str]:
    """같은 규칙을 배열 상태에서 — registered & ~locked & (flow==GATE_IN) & PLANNED & A > t+1e-6."""
    L = run.tw.ledger
    W = run.tw.blocks
    reg = np.asarray(L.registered); lk = np.asarray(run.locked); own = np.asarray(L.owner); row = np.asarray(L.row)
    a = np.asarray(L.a_gate_in)
    st = np.asarray(W.orders.status); fl = np.asarray(W.orders.flow)
    out = []
    for s in range(reg.shape[0]):
        if not reg[s] or lk[s] or (owner is not None and own[s] != owner):
            continue
        if flow_in and int(fl[own[s], row[s]]) != FL_GATE_IN:
            continue
        if int(st[own[s], row[s]]) != JS_PLANNED or not (a[s] > t + 1e-6):
            continue
        out.append(tt.truck_ids[s])
    return sorted(out)


def _rec_view(rec) -> dict:
    return {"owner": rec.owner, "version": rec.version, "transfer_count": rec.transfer_count,
            "transfer_history": tuple(rec.transfer_history), "entry_deferrals": rec.entry_deferrals,
            "entry_deferred_s": rec.entry_deferred_s, "a_gate_in": rec.a_gate_in, "locked": rec.locked}


def _arr_rec_view(run, tt, s: int) -> dict:
    L = run.tw.ledger
    k = int(np.asarray(L.transfer_count)[s])
    hs, hd, ht = np.asarray(L.transfer_src)[s], np.asarray(L.transfer_dst)[s], np.asarray(L.transfer_t)[s]
    return {"owner": tt.block_ids[int(np.asarray(L.owner)[s])], "version": int(np.asarray(L.version)[s]),
            "transfer_count": k,
            "transfer_history": tuple((tt.block_ids[int(hs[i])], tt.block_ids[int(hd[i])], float(ht[i])) for i in range(k)),
            "entry_deferrals": int(np.asarray(L.entry_deferrals)[s]),
            "entry_deferred_s": float(np.asarray(L.entry_deferred_s)[s]),
            "a_gate_in": float(np.asarray(L.a_gate_in)[s]), "locked": bool(np.asarray(run.locked)[s])}


def test_ladder2_y01_y21_transfer_and_defer_match_v5():
    label = "ladder2-y01y21-40"
    prof, layout, built = _build_small(("Y01", "Y21"), 40)
    ids = tuple(layout.ids)
    mbt, ann = _v5_terminal(prof, built, OBS_2H, 1800.0)
    ops: dict[str, dict | None] = {"xfer": None, "defer": None}
    cand5: dict[float, list[str]] = {}

    def hook(m, t):
        ann.review(m, t)
        if ops["xfer"] is None and t >= 1800.0:
            for src in ids:
                cands = _v5_candidates(m, t, pre_gate=True, flow="GATE_IN", owner=src)
                if cands:
                    break
            else:
                return
            cand5[t] = cands
            jid = cands[0]
            dst = [b for b in ids if b != src][0]
            travel, route = layout.gate_to_block_s(dst), layout.pre_gate_route_delta_s(src, dst)
            ok = m.try_pre_gate_transfer(jid, dst, travel_s=travel, route_delta_s=route)
            assert ok, "v5 강제 이송이 실패했다 — 무대가 명세와 다르다"
            ops["xfer"] = dict(t=t, job=jid, src=src, dst=dst, travel_s=travel, route_s=route,
                               rec=_rec_view(m.ledger.records[jid]), route_cost_s=m.route_cost_s)
        elif ops["xfer"] is not None and ops["defer"] is None and t >= 2400.0:
            cands = _v5_candidates(m, t, pre_gate=True, flow=None)
            if not cands:
                return
            cand5[t] = cands
            jid = cands[0]
            ok = m.try_defer_admitted_entry(jid, 600.0)
            assert ok, "v5 강제 이연이 실패했다"
            ops["defer"] = dict(t=t, job=jid, rec=_rec_view(m.ledger.records[jid]))

    out5 = mbt.run(_rule_policy(), review_fn=hook)
    mbt.check_invariants()
    assert ops["xfer"] is not None and ops["defer"] is not None, ops
    assert ann.n_admitted == 40

    tw, tt, run0, engine = _array_setup(prof, built, OBS_2H, 1800.0, n_spare=2)
    bidx = tt.block_index
    seen: dict[str, dict] = {}

    def review_fn(run, e, t):
        x, dfr = ops["xfer"], ops["defer"]
        if t == x["t"]:
            mine = _array_candidates(run, tt, t, flow_in=True, owner=bidx[x["src"]])
            assert mine == cand5[t], f"[{label}] t={t} 이송 후보 집합 arr={mine} v5={cand5[t]}"
            run, ok, code = MB.try_pre_gate_transfer(run, tt, engine, t, tt.truck_index[x["job"]], bidx[x["dst"]],
                                                     travel_s=x["travel_s"], route_delta_s=x["route_s"])
            assert ok, f"[{label}] 배열 이송 거절 {TX.REASON_NAMES.get(code, code)}"
            got = _arr_rec_view(run, tt, tt.truck_index[x["job"]])
            assert got == x["rec"], f"[{label}] 이송 뒤 원장 arr={got} v5={x['rec']}"
            assert float(run.tw.ledger.route_cost_s) == x["route_cost_s"] == x["route_s"]
            assert MB.check_invariants(run, tt)
            seen["xfer"] = got
            # 이송 상한(1) — 같은 트럭 재시도는 거절 (v5 431행) · 세계 불변
            run2, ok2, code2 = MB.try_pre_gate_transfer(run, tt, engine, t, tt.truck_index[x["job"]], bidx[x["src"]],
                                                        travel_s=layout.gate_to_block_s(x["src"]), route_delta_s=-x["route_s"])
            assert not ok2 and code2 == TX.R_MAX_TRANSFERS
        if t == dfr["t"]:
            mine = _array_candidates(run, tt, t, flow_in=False)
            assert mine == cand5[t], f"[{label}] t={t} 이연 후보 집합 arr={mine} v5={cand5[t]}"
            run, ok, code = MB.try_defer_admitted_entry(run, tt, t, tt.truck_index[dfr["job"]], 600.0)
            assert ok, f"[{label}] 배열 이연 거절 {TX.REASON_NAMES.get(code, code)}"
            got = _arr_rec_view(run, tt, tt.truck_index[dfr["job"]])
            assert got == dfr["rec"], f"[{label}] 이연 뒤 원장 arr={got} v5={dfr['rec']}"
            run2, ok2, code2 = MB.try_defer_admitted_entry(run, tt, t, tt.truck_index[dfr["job"]], 600.0)
            assert not ok2 and code2 == TX.R_MAX_DEFERRALS
            seen["defer"] = got
        return run

    t0 = time.perf_counter()
    run, codes = MB.run_python(run0, engine, review_fn=review_fn)
    jax.block_until_ready(run)
    t_arr = time.perf_counter() - t0
    assert "xfer" in seen and "defer" in seen
    d = MB.from_run(run, tt)
    rows = _ledger_rows(run, tw, tt, np.stack([np.asarray(c) for c in codes]))
    summ = _compare_with_v5(label, d, mbt, ann, tt, rows, OBS_2H.observe_s, out5)
    x = ops["xfer"]
    assert x["job"] in d["blocks"][x["dst"]]["jobs"] and x["job"] not in d["blocks"][x["src"]]["jobs"]
    assert d["route_cost_s"] == out5["route_cost_s"] == x["route_s"] != 0.0
    # ★이연 비용의 원점 — v5 `Job.appointment_gate_time`(예약 원점)이 배열 `orders.appt_s` 로 이식됐는지.
    #   원점을 통지 시각(max(0, 도착 − lead))으로 잘못 잡으면 기사 외부 대기가 lead 만큼 부푼다 (여기선 600s 가
    #   2,400s 로 4배). v5 감사 함수를 그대로 굴려 항목까지 맞춘다.
    exp_def = time_sell.deferral_ledger(mbt)
    assert exp_def and len(exp_def) == 1, f"[{label}] v5 이연 원장이 비었다 {exp_def}"
    assert d["deferrals"] == exp_def, f"[{label}] 이연 원장 arr={d['deferrals']} v5={exp_def}"
    w5 = exp_def[0]["driver_outside_wait_s"]
    assert w5 == 600.0, f"[{label}] v5 기사 외부 대기가 600s 가 아니다 ({w5}) — 무대가 명세와 다르다"
    REPORT[label] = dict(**summ, arr_s=round(t_arr, 2), xfer=f"{x['job']} {x['src']}→{x['dst']} @{x['t']} route {x['route_s']}",
                         defer=f"{ops['defer']['job']} @{ops['defer']['t']} 외부대기 {w5}s", blocks=2,
                         device=jax.devices()[0].platform)


# ───────────────────────────────────────────────── ③ 정답 JSON (터미널 30 · 300) — 세션 이어 돌리기
def _state_dir() -> str:
    d = os.environ.get("TERMINAL_STATE_DIR") or os.path.join(_ROOT, "outputs", "v6", "verify", "terminal_equiv")
    os.makedirs(d, exist_ok=True)
    return d


def _enable_compile_cache(d: str) -> None:
    """세션마다 다시 컴파일하지 않게 — 영속 캐시 (WSL 재시작에도 남는 /mnt/c 경로)."""
    try:
        jax.config.update("jax_compilation_cache_dir", os.path.join(d, "jax_cache"))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    except Exception:                                                    # 옛 jax — 캐시 없이 간다
        pass


def _divisor_chunk(E: int, target: int) -> int:
    return max(k for k in range(1, min(E, target) + 1) if E % k == 0)


_JOB_KEYS = ("status", "assigned_crane", "rehandle_count", "service_start", "service_end",
             "gate_in", "block_arrival", "actual_gate_out")


def _compare_truth(label, d, truth, tt, rows=None) -> dict:
    blocks = truth["blocks"]
    assert list(d["blocks"]) == list(blocks)
    n_hash = 0
    n_job_rows = 0
    for bid, tb in blocks.items():
        blk = d["blocks"][bid]
        arlog = [tuple(x) for x in blk["events"]]
        tlog = [tuple(x) for x in tb["events"]]
        _fail_first_event_diff(label, bid, arlog, tlog,
                               extra=f" violation={blk['violation_names']} overflow={blk['overflow']} terminal={blk['terminal']}")
        assert blk["event_hash"] == tb["event_hash"], f"[{label}] {bid} 해시 arr={blk['event_hash']} truth={tb['event_hash']}"
        n_hash += 1
        assert blk["violation"] == 0 and blk["overflow"] == 0 and blk["terminal"], f"[{label}] {bid} {blk['violation_names']}"
        bad = [(t, blk["cost_raw"][t], tb["cost_raw"][t]) for t in COST_TERMS if blk["cost_raw"][t] != tb["cost_raw"][t]]
        assert not bad, f"[{label}] {bid} 비용 항목 (항, arr, truth): {bad}"
        assert blk["kpis"] == tb["kpis"], f"[{label}] {bid} KPI arr={blk['kpis']} truth={tb['kpis']}"
        assert (blk["n_jobs"], blk["end_s"], blk["clock_s"], blk["n_cranes"], blk["n_events"], blk["deadlock_escapes"], blk["unfinished"]) == \
               (tb["n_jobs"], tb["end_s"], tb["clock_s"], tb["n_cranes"], tb["n_events"], tb["deadlock_escapes"], tb["unfinished"]), f"[{label}] {bid} 요약"
        # ★두꺼운 정답 — 오더 전열 · 시간장부 적분 (예전 정답에는 없던 항목; 있을 때만 본다)
        if "jobs" in tb:
            miss = sorted(set(blk["jobs"]) ^ set(tb["jobs"]))[:5]
            assert not miss, f"[{label}] {bid} 작업 집합이 다르다 {miss}"
            for jid, tj in tb["jobs"].items():
                got = tuple(blk["jobs"][jid][k] for k in _JOB_KEYS)
                exp = tuple(tj[k] for k in _JOB_KEYS)
                assert got == exp, f"[{label}] {bid} 오더 {jid}: arr={got} truth={exp}"
                n_job_rows += 1
        if tb.get("ledger"):
            lk = ("terminal_area_s", "block_area_s", "block_tail_area_s")
            got, exp = tuple(blk["ledger"][k] for k in lk), tuple(tb["ledger"][k] for k in lk)
            assert got == exp, f"[{label}] {bid} 장부 적분 arr={got} truth={exp}"
    assert d["admitted"] == truth["admitted"] and d["n_turns"] == truth["n_turns"], \
        f"[{label}] admitted/turns arr=({d['admitted']},{d['n_turns']}) truth=({truth['admitted']},{truth['n_turns']})"
    assert d["turn_samples_s"] == truth["turn_samples_s"], f"[{label}] 턴 표본 첫 차이 {_first_diff(d['turn_samples_s'], truth['turn_samples_s'])}"
    assert d["turn_sum_s"] == truth["turn_sum_s"] and d["terminal_total"] == truth["terminal_total"]
    assert d["route_cost_s"] == truth["route_cost_s"] and d["end"] == truth["end"]
    assert d["exhausted"] == [0] * len(tt.block_ids)
    # ★두꺼운 정답 — locked · totals · 이연 원장 · 투입 원장 (21블록 규모에서 처음 대조된다)
    n_ann = 0
    if "locked" in truth:
        bad = [(j, d["locked"].get(j), truth["locked"][j]) for j in truth["locked"] if d["locked"].get(j) != truth["locked"][j]]
        assert not bad, f"[{label}] locked 갈림 {bad[:5]}"
        assert set(d["locked"]) == set(truth["locked"]), f"[{label}] locked 대상 집합"
    if "totals" in truth:
        assert d["totals"] == truth["totals"], f"[{label}] totals"
    if "deferrals" in truth:
        assert d["deferrals"] == truth["deferrals"], f"[{label}] 이연 원장 arr={d['deferrals']} truth={truth['deferrals']}"
    if "ann_ledger" in truth and rows is not None:
        exp_rows = truth["ann_ledger"]
        assert len(rows) == len(exp_rows), f"[{label}] 투입 원장 길이 arr={len(rows)} truth={len(exp_rows)}"
        for i, (a, b) in enumerate(zip(exp_rows, rows)):
            assert a["event"] == b["event"] and a.get("job_id") == b.get("job_id"), f"[{label}] 투입 원장 {i}: truth={a} arr={b}"
            if a["event"] == "ADMIT":
                assert (a["block"], a["flow"], a["arrival_s"], a["t"]) == (b["block"], b["flow"], b["arrival_s"], b["t"]), f"[{label}] 투입 원장 {i} ADMIT truth={a} arr={b}"
            elif a["event"] == "SKIP":
                assert AD.code_of_reason(a["reason"]) == b["code"], f"[{label}] 투입 원장 {i} SKIP truth={a['reason']!r} arr={b['reason_code']}"
        n_ann = len(rows)
    return dict(hash_match=f"{n_hash}/{len(blocks)}", events=sum(b["n_events"] for b in blocks.values()),
                admitted=d["admitted"], turns=d["n_turns"], turn_sum=d["turn_sum_s"],
                job_rows=n_job_rows, ledger_rows=n_ann, locked=sum(1 for v in d["locked"].values() if v))


_LOADS = [int(x) for x in os.environ.get("TERMINAL_LOADS", "30,300").split(",") if x.strip()]
_enable_compile_cache(_state_dir())        # ①② 도 두 번째 세션부터는 컴파일을 건너뛴다 (GPU 는 ①이 한 세션 90초를 넘긴다)


@pytest.mark.parametrize("load", _LOADS, ids=[f"load{x}" for x in _LOADS])
def test_ladder3_terminal_ground_truth(load):
    label = f"ladder3-terminal{load}"
    path = os.path.join(_TRUTH_DIR, f"terminal_load{load}_seed{SEED}.json")
    if not os.path.exists(path):
        pytest.skip(f"정답 없음: {path} — scripts/v6/dump_ground_truth.py --mode terminal --load {load}")
    sdir = _state_dir()
    _enable_compile_cache(sdir)
    budget = float(os.environ.get("TERMINAL_SESSION_S", "40"))
    state_path = os.path.join(sdir, f"load{load}_run.pkl")
    tables_path = os.path.join(sdir, f"load{load}_tables.pkl")
    codes_path = os.path.join(sdir, f"load{load}_codes.npy")   # 세션 사이에 이어 모으는 에폭 코드 (E,B,M)
    t_start = time.perf_counter()
    prof, layout, built = _build_terminal(load)
    if os.path.exists(state_path) and os.path.exists(tables_path):
        import pickle
        with open(tables_path, "rb") as f:
            tt = pickle.load(f)
        run = MB.load_run(state_path)
        g = Geom.from_profile(prof)
        k0 = tt.tables[0].crane_index[prof.cranes[0].crane_id]
        engine = MB.Engine(g=g, policy_fn=DP.make_resolver("sf_spt", g, count_lost=False), check=True, pre_advice=True,
                           horizon_s=float(prof.decision_horizon_s), joint=True, margin=mb.CAPACITY_MARGIN,
                           end_ann=float(built["sim_end_s"]), steps_per_epoch=8 * tt.n_max + 256,
                           steps_final=8 * tt.n_max + 256, k0=int(k0))
        resumed = True
    else:
        import pickle
        tw, tt, run, engine = _array_setup(prof, built, ts.OBS_24H, 1800.0)
        with open(tables_path, "wb") as f:
            pickle.dump(tt, f)
        resumed = False
    jax.block_until_ready(run)
    E = run.n_epochs
    e = int(run.tw.epoch_idx)
    chunk = _divisor_chunk(E, 11)
    # ★투입 코드는 세션을 넘어 이어 모은다 — 그래야 마지막 세션에서 투입 원장 전행을 정답과 맞춰 볼 수 있다.
    #   (E,B,M) int32 이고 터미널 명단은 칸당 1건이라 30k 정수 = 120 KB 남짓이다.
    codes_all = np.load(codes_path) if (resumed and os.path.exists(codes_path)) else None
    if codes_all is not None:
        assert codes_all.shape[0] == e, f"모아 둔 코드 {codes_all.shape[0]} 에폭 ≠ 커서 {e}"
    t_setup = time.perf_counter() - t_start
    t0 = time.perf_counter()
    n_done = 0
    while e < E and (time.perf_counter() - t0) < budget:
        run, c = MB.run_epochs_jit(run, e, chunk, engine)
        jax.block_until_ready(run)
        c = np.asarray(c)
        codes_all = c if codes_all is None else np.concatenate([codes_all, c], axis=0)
        e += chunk
        n_done += chunk
    t_run = time.perf_counter() - t0
    assert int(run.tw.epoch_idx) == e
    if e < E:
        MB.save_run(state_path, run)
        if codes_all is not None:
            np.save(codes_path, codes_all)
        msg = (f"이어 돌리기 {e}/{E} 에폭 (이번 세션 {n_done} 에폭 {t_run:.1f}s · {1000 * t_run / max(1, n_done):.0f} ms/에폭 · "
               f"준비 {t_setup:.1f}s · {'재개' if resumed else '새로'} · {jax.devices()[0].platform}) — 다시 실행하면 이어 간다")
        REPORT[label] = dict(progress=f"{e}/{E}", ms_per_epoch=round(1000 * t_run / max(1, n_done)), device=jax.devices()[0].platform)
        pytest.skip(msg)
    run = MB.finish_run_jit(run, engine)
    jax.block_until_ready(run)
    t_fin = time.perf_counter() - t0 - t_run
    d = MB.from_run(run, tt)
    truth = json.load(open(path, encoding="utf-8"))
    rows = None
    if codes_all is not None and codes_all.shape[0] == E:
        rows = _ledger_rows(run, run.tw, tt, codes_all)
    summ = _compare_truth(label, d, truth, tt, rows=rows)
    assert MB.check_invariants(run, tt), f"[{label}] 불변식 (보존·이중소유) 위반"
    for p_ in (state_path, codes_path):
        if os.path.exists(p_):
            os.remove(p_)
    REPORT[label] = dict(**summ, epochs=E, ms_per_epoch=round(1000 * t_run / max(1, n_done)), last_session_epochs=n_done,
                         finish_s=round(t_fin, 1), device=jax.devices()[0].platform, resumed=resumed)
    with open(os.path.join(sdir, f"load{load}_done.json"), "w", encoding="utf-8") as f:
        json.dump(REPORT[label], f, ensure_ascii=False, indent=1)


# ───────────────────────────────────────────────── 보고
def test_zz_report(capsys):
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    with capsys.disabled():
        print("\n[terminal_equiv report]")
        for k, r in REPORT.items():
            print(f"  {k:24s} {r}")
