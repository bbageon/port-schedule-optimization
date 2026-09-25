"""★정답 궤적 Y01 재현 — 배열 엔진(gpu/engine_step.py, 조각 1~5 통합)이 v5 가 남긴 정답을 **그대로** 낸다 ([[YR-327]]).

정답 궤적: outputs/reports/yr327_v6_port/ground_truth/block_Y01_load30_seed9900777[_<policy>_<level>].json
  (scripts/v6/dump_ground_truth.py run_block — 블록 Y01 · 본선 2척 × 120 moves · 크레인 2(YC-L·YC-W 1..24) · 24×10×6 ·
   레인 2(인접) · YT 3대 470초 · 지평 86400 + 배수 7200). 정책·정보수준 다섯 조합:
    sf_spt / PRE_ADVICE     ★원본 (해시 6668fa4902c4efe4 · 사건 1,318 · 결정 237) — ResolverPolicy(ServiceFirstSPT) +
                            CandidateGenerator(LEGACY_DEFAULT) + baselines._apply → 배열판 **공동 규약**
                            (engine_step joint=True · cands3 후보 · dispatch.make_resolver("sf_spt", count_lost=False))
    sf_spt / BLOCK_ARRIVAL  같은 해시 (Y01 은 외부트럭 0 → ETA wake·PRE 후보 없음 — 정보수준이 답을 못 바꾼다)
    reference / BLOCK_ARRIVAL · reference / PRE_ADVICE · first / BLOCK_ARRIVAL
                            (해시 760af9a960d07bf8 · 사건 1,321 · 결정 232) — ReferenceDispatcher 의미 → **순차 규약**
                            (joint=False · dispatch.policy_reference / engine_step.first_by_id)

■ 무엇을 지키나 (앞이 깨지면 그 자리에서 **처음 갈리는 사건**을 보고한다)
  ① 사건 로그 전열 (round(t,6)·종류·payload) == 정답 JSON == 같은 정책으로 지금 굴린 v5, 해시 일치
  ② 비용 13항 == (episode_raw · 비트 동일)  ③ KPI 10항 ==  ④ 결정 수 == n_decisions
  ⑤ 지금 굴린 v5 와: 오더(status·assigned_crane·rehandles·service_start/end) · 크레인(bay·row·served·completions·assigned_job)
     · 격자(piles·컨테이너 좌표) · 배(started·remaining·buffer·blocked_since·wait_accum·done·actual_completion·계획) ·
     이송(busy_until·pending·대기 적분) · rate 5항 · 레인 혼잡 적분 · last_decision_at · terminal · violation 0 · overflow 0
  + jit `run` 의 최종 세계 == `run_while` 의 세계 (잎 전부 비트) — 공동 규약에서도 학습 경로가 같은 답.
실행: WSL venv · x64 CPU. 한 조합 ≈ 6초(컴파일 포함) — 88초 창에 다섯 조합이 든다.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from yard_rl.v6.gpu import dispatch as DP                                           # noqa: E402
from yard_rl.v6.gpu import engine_step as ES                                        # noqa: E402
from yard_rl.v6.gpu.geom import Geom                                                # noqa: E402
from yard_rl.v6.gpu.host_convert import event_stream_hash, from_block_world, to_block_world   # noqa: E402
from yard_rl.v6.gpu.state import COST_TERMS, RATE_TERMS, violation_names           # noqa: E402
from yard_rl.v6.world.domain.enums import InformationLevel                         # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_TRUTH_DIR = os.path.join(_ROOT, "outputs", "reports", "yr327_v6_port", "ground_truth")
REPORT: dict[str, dict] = {}


def _load_dump_script():
    path = os.path.join(_ROOT, "scripts", "v6", "dump_ground_truth.py")
    spec = importlib.util.spec_from_file_location("_dgt_for_y01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_dump_script()

#: (정책, 정보수준) — 정답 JSON 이 있는 조합. 원본 sf_spt/PRE_ADVICE 가 첫째.
COMBOS = [("sf_spt", "PRE_ADVICE"), ("sf_spt", "BLOCK_ARRIVAL"), ("reference", "BLOCK_ARRIVAL"),
          ("reference", "PRE_ADVICE"), ("first", "BLOCK_ARRIVAL")]
_IDS = [f"{p}-{l}" for p, l in COMBOS]


def _truth_path(policy: str, level: InformationLevel) -> str:
    return os.path.join(_TRUTH_DIR, D.block_truth_name("Y01", 30, 9900777, policy, level))


def _caps(scn):
    n0 = len(scn.jobs)
    n_max = max(8, 1 << (n0 - 1).bit_length())
    s_max = 8 * n_max + 256
    return dict(n_max=n_max, q_cap=max(32, 4 * n_max), log_cap=s_max + n_max), s_max


def _array_policy(policy: str, w0, tb, g):
    """정책 이름 → (policy_fn, params, joint)."""
    if policy == "reference":
        return DP.policy_reference, w0.orders.is_vessel, False
    if policy == "first":
        return ES.first_by_id, None, False
    if policy == "sf_spt":
        return DP.make_resolver("sf_spt", g, count_lost=False), DP.resolver_params(tb, g), True
    raise KeyError(policy)


def _run_v5(prof, scn, policy: str, level):
    sim = D._sim(prof, scn, level)
    pol = D.make_policy(policy, level)
    n_dec = 0
    while (dp := sim.run_until_decision()) is not None:
        n_dec += 1
        pol(sim, dp)
    return sim, n_dec


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


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b)] if len(a) > len(b) else None), (b[len(a)] if len(b) > len(a) else None)
    return None


def _leaf_diff(a, b) -> list[str]:
    names = [str(p) for p, _ in jax.tree_util.tree_leaves_with_path(a)]
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return [names[i] for i, (x, y) in enumerate(zip(la, lb))
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]


@pytest.mark.parametrize("policy,level_name", COMBOS, ids=_IDS)
def test_y01_ground_truth_replay(policy, level_name):
    level = InformationLevel[level_name]
    path = _truth_path(policy, level)
    if not os.path.exists(path):
        pytest.skip(f"정답 궤적 없음: {path} — scripts/v6/dump_ground_truth.py --mode block --block Y01 --policy {policy} --info-level {level_name}")
    truth = json.load(open(path, encoding="utf-8"))
    label = f"Y01-{policy}-{level_name}"
    prof, built = D._build(truth["load"], truth["seed"])
    scn = built["scenarios"][truth["block"]]
    caps, s_max = _caps(scn)
    w0, tb = to_block_world(prof, scn, **caps)
    g = Geom.from_profile(prof)
    hz = float(prof.decision_horizon_s)
    pre = level == InformationLevel.PRE_ADVICE
    policy_fn, params, joint = _array_policy(policy, w0, tb, g)
    w, trace = ES.run_jit(w0, params, g, policy_fn, s_max, True, pre, hz, joint)
    d = from_block_world(w, tb)
    # ① 사건 로그 전열 — 정답 JSON 과 (처음 갈리는 사건 보고)
    arlog = [(round(t, 6), k, p) for (t, k, p) in d["event_log"]]
    tlog = [(round(t, 6), k, p) for (t, k, p) in truth["events"]]
    diff = _first_diff(arlog, tlog)
    if diff is not None:
        i, x, y = diff
        ctx = "\n".join(f"    #{j}: arr={arlog[j] if j < len(arlog) else '-'}  |  truth={tlog[j] if j < len(tlog) else '-'}"
                        for j in range(max(0, i - 4), min(max(len(arlog), len(tlog)), i + 4)))
        pytest.fail(f"[{label}] ① 사건 로그가 {i}번째에서 갈린다: arr={x} truth={y}\n{ctx}\n"
                    f"  (arr {len(arlog)}건 · truth {len(tlog)}건 · violation={d['violation']} {d['violation_names']} "
                    f"overflow={d['overflow']} terminal={d['terminal']} steps={d['steps']})")
    h = event_stream_hash(w, tb)
    assert h == truth["event_hash"], f"[{label}] ① 해시 arr={h} truth={truth['event_hash']}"
    # ② 비용 13항 · ③ KPI · ④ 결정 수
    cr = truth["cost_raw"]
    bad = [(t, d["cost_episode"][t], cr[t]) for t in COST_TERMS if d["cost_episode"][t] != cr[t]]
    assert not bad, f"[{label}] ② 비용 항목이 갈린다 (항, arr, truth): {bad}"
    kp = truth["kpis"]
    got_k = {k: d["kpi"][k] for k in kp}
    assert got_k == kp, f"[{label}] ③ KPI arr={got_k} truth={kp}"
    n_steps = int(w.steps)
    n_dec = int(np.asarray(trace.decided)[:n_steps].sum())
    assert n_dec == truth["n_decisions"], f"[{label}] ④ 결정 수 arr={n_dec} truth={truth['n_decisions']}"
    assert d["violation"] == 0 and d["overflow"] == 0 and d["terminal"], \
        f"[{label}] violation={d['violation_names']} overflow={d['overflow']} terminal={d['terminal']}"
    assert d["clock"] == truth["clock_s"] and d["end"] == truth["end_s"]
    # ⑤ 지금 굴린 v5 (같은 정책) 와 상태 전부
    sim, n_dec5 = _run_v5(prof, scn, policy, level)
    assert sim.event_stream_hash() == truth["event_hash"], f"[{label}] v5 구동기가 정답 절차와 다르다"
    assert n_dec5 == truth["n_decisions"]
    for jid, j in sim.jobs.items():
        a = d["jobs"][jid]
        got = (a["status"], a["assigned_crane"], a["rehandle_count"], a["service_start"], a["service_end"])
        exp = (j.status.name, j.assigned_crane, j.rehandle_count, j.service_start, j.service_end)
        assert got == exp, f"[{label}] ⑤ 오더 {jid}: arr={got} v5={exp}"
    for cid in sim.fleet.ids():
        yc, a = sim.fleet.get(cid), d["cranes"][cid]
        got = (a["position_bay"], a["trolley_row"], a["served_count"], a["recent_completions"], a["assigned_job"],
               a["loaded_travel_m"], a["empty_travel_m"], a["available_at"], a["yielded"], a["down"])
        exp = (yc.state.position_bay, yc.state.trolley_row, yc.served_count, yc.recent_completions, yc.state.assigned_job,
               yc.state.loaded_travel_m, yc.state.empty_travel_m, yc.state.available_at, yc.yielded, yc.down)
        assert got == exp, f"[{label}] ⑤ 크레인 {cid}: arr={got} v5={exp}"
    assert d["piles"] == {k: v for k, v in sim.stacks._stacks.items() if v}, f"[{label}] ⑤ piles"
    assert d["containers"] == {c: (x.bay, x.row, x.tier) for c, x in sim.stacks.containers.items()}, f"[{label}] ⑤ 컨테이너 좌표"
    assert d["vessels"] == _v5_vessels(sim, tb.vessel_ids), f"[{label}] ⑤ 배\n  arr={d['vessels']}\n  v5 ={_v5_vessels(sim, tb.vessel_ids)}"
    tr = sim.transfer
    assert d["transfer"]["busy_until"] == list(tr.busy_until), f"[{label}] ⑤ 이송 busy_until arr={d['transfer']['busy_until']} v5={tr.busy_until}"
    assert d["transfer"]["pending"] == list(tr.pending)
    assert d["transfer"]["transfer_wait_accum_s"] == tr.transfer_wait_accum_s
    for t, v in sim.cost._rate.items():
        assert d["cost_rate"][t] == v, f"[{label}] ⑤ rate.{t} arr={d['cost_rate'][t]!r} v5={v!r}"
    assert d["lane_cong_area_s"] == sim.lanes.cong_area_s
    assert d["last_decision_at"] == sim._last_decision_at and sim.terminal
    ks = sim.kpis.snapshot()
    assert d["kpi"]["berth_overrun_s"] == ks.berth_overrun_s and d["kpi"]["vessel_delay_s"] == ks.vessel_delay_s
    # + 학습 경로 run_while == run (잎 전부)
    ww = ES.run_while_jit(w0, params, g, policy_fn, s_max, True, pre, hz, joint)
    bad_leaves = _leaf_diff(w, ww)
    assert not bad_leaves, f"[{label}] run_while 와 run 이 다른 잎 {bad_leaves}"
    kinds = {}
    for (_, k, _) in d["event_log"]:
        kinds[k] = kinds.get(k, 0) + 1
    repo = sum(1 for (_, k, p) in d["event_log"] if k == "DISPATCH" and ":REPO:" in p)
    REPORT[label] = dict(hash=h, events=len(arlog), decisions=n_dec, steps=n_steps, joint=joint, pre_advice=pre,
                         repo_dispatch=repo, kinds=kinds, rehandles=kp["rehandle_count"],
                         sts_wait=round(cr["sts_wait"], 3), transfer_wait=round(cr["transfer_wait"], 3))


def test_zz_report(capsys):
    """마지막 — 조합별 해시·사건·결정·REPO 실행 수. 원본 sf_spt/PRE_ADVICE 가 돌았으면 REPO 가 실제로 실행됐는지 단언."""
    assert REPORT, "앞 시험이 하나도 안 돌았다"
    with capsys.disabled():
        print("\n[Y01 replay report]")
        for k, r in REPORT.items():
            print(f"  {k:28s} hash={r['hash']} events={r['events']} dec={r['decisions']} steps={r['steps']} "
                  f"joint={r['joint']} repo={r['repo_dispatch']} rehandles={r['rehandles']} "
                  f"sts_wait={r['sts_wait']} transfer_wait={r['transfer_wait']}")
    if "Y01-sf_spt-PRE_ADVICE" in REPORT:
        r = REPORT["Y01-sf_spt-PRE_ADVICE"]
        assert r["hash"] == "6668fa4902c4efe4" and r["events"] == 1318 and r["decisions"] == 237
        assert r["repo_dispatch"] >= 1, "원본 궤적의 REPOSITION 실행이 배열 로그에 없다"
    if "Y01-reference-BLOCK_ARRIVAL" in REPORT:
        assert REPORT["Y01-reference-BLOCK_ARRIVAL"]["repo_dispatch"] == 0
