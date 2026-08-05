"""YR-149 1단계 — 결정시점 반사실 진단 (사슬 한계효과 리플레이, 동결).

■ 대상: YR-133 보정 파일럿의 확정 이송 14건 (원장·시드·이동시간 전부 보존 — 정확 재현).
■ 방법(동결): 에피소드별 확정 이송 목록 T=[d1..dn] 을 시각 순으로 두고, run_j = "앞의
  j건만 그대로 실행·이후 이송 없음" (j=0..n). **k번째 결정의 한계효과 = run_k − run_{k−1}**
  — 원 궤적의 결정시점 상태를 정확히 복제(같은 시드·같은 travel rng 키). 사후 실현정보는
  진단 라벨 전용 — 배포 견적 입력 금지.
■ 분해(동결): 실현 한계효과를 v1 채널로 분해 — 소스 파급 = Δ(소스 블록 truck 채널),
  수신 파급 = Δ(수신 블록 truck 채널), 본선 파급 = Δ(vessel 채널), 이동 = Δ(move).
  예측 대응 잔차: 소스 잔차 = Δ소스 + OutRelief · 수신 잔차 = Δ수신 − InBurden ·
  본선 잔차 = Δ본선 − 0 · 이동 잔차 = Δ이동 − route/3600. **지배 축 = |에피소드 묶음
  평균 잔차| 최대인 축** (소스/수신/본선 — 2단계는 그 한 축만 보정).
■ 통계 단위: 14건은 진단 상태 수 — 에피소드 seed 쌍으로 묶어 보고. v2 실현 총비용 병기.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean, median

from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.evalkit import CHANNELS, channel_split
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S,
                                       GATE_BLOCK_MIN_S, GATE_BLOCK_SIGMA_S, trunc_normal)
from .yr088_joint_rl import LEVEL
from .yr105_conditional_transfer import _CELLS, _sim
from .yr138_episode_pilot import _v2_hard_total

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr149_transfer_diag")
PILOT = Path("outputs/reports/yr133_sell_quote/pilot.json")
ROUTE_S = 180.0


class ReplayResolver:
    """원장 확정 이송의 앞 limit 건만 그대로 재실행 — 그 외 전부 KEEP."""

    def __init__(self, actions: list[dict], limit: int, travel_fn):
        self.actions = actions[:limit]
        self.travel_fn = travel_fn
        self.done = 0

    def review(self, mbt, t: float) -> None:
        for a in self.actions:
            if abs(a["t"] - t) <= 1e-6:
                ok = mbt.try_transfer(a["job_id"], a["dst"], route_s=ROUTE_S,
                                      travel_s=self.travel_fn(a["src"], a["job_id"]))
                assert ok, f"리플레이 이송 실패 — 재현 위반: {a}"
                self.done += 1


def run_replay(seeds: dict[str, int], tag: str, actions: list[dict], limit: int) -> dict:
    mbt = MultiBlockTerminal({"A": _sim(_CELLS["A"], seeds["A"]),
                              "B": _sim(_CELLS["B"], seeds["B"])})
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def travel_fn(src, jid):                      # 원 파일럿과 동일 키 → travel 정확 재현
        rng = random.Random(f"y133:{tag}:{src}:{jid}")
        return trunc_normal(rng, GATE_BLOCK_MEAN_S,
                            GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
                            lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)

    resolver = ReplayResolver(actions, limit, travel_fn)
    chan_by_block = {b: {k: 0.0 for k in CHANNELS} for b in ("A", "B")}
    _bid = {id(s): b for b, s in mbt.blocks.items()}

    def cost_fn(sim, t0, t1, raw):
        cb = RC.cost_for(interval_start_s=t0, interval_end_s=t1, raw=raw, risk_max=0.0)
        b = _bid.get(id(sim))
        for k, v in channel_split(cb.contributions()).items():
            if b is not None:
                chan_by_block[b][k] += v
        return cb.total_normalized

    res = mbt.run(policy, resolver.review, cost_fn)
    mbt.check_invariants()
    assert resolver.done == limit, f"리플레이 미발화: {resolver.done}/{limit}"
    route_cost = res["route_cost_s"] / 3600.0
    return {"total_v1": float(res["terminal_total"] + route_cost),
            "v2_total": sum(_v2_hard_total(s) for s in mbt.blocks.values()),
            "chan_by_block": chan_by_block, "route_cost": route_cost,
            "backlog": sum(s.unfinished_backlog() for s in mbt.blocks.values()),
            "policy_exceptions": exc["n"]}


def diagnose() -> dict:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    quote_eps = pilot["arms"]["quote"]
    per_decision, per_episode = [], []
    for i, ep in enumerate(quote_eps):
        actions = [r for r in ep["quote_ledger"] if r["decision"] == "TRANSFER"]
        actions.sort(key=lambda r: r["t"])
        if not actions:
            continue
        tag = f"p{i}"
        seeds = ep["seeds"]
        print(f"[diag] pair{i} n_transfers={len(actions)}", flush=True)
        runs = [run_replay(seeds, tag, actions, j) for j in range(len(actions) + 1)]
        ep_rows = []
        for k, a in enumerate(actions, start=1):
            prev, cur = runs[k - 1], runs[k]
            d_src = (cur["chan_by_block"][a["src"]]["truck"]
                     - prev["chan_by_block"][a["src"]]["truck"])
            d_dst = (cur["chan_by_block"][a["dst"]]["truck"]
                     - prev["chan_by_block"][a["dst"]]["truck"])
            d_vessel = sum(cur["chan_by_block"][b]["vessel"]
                           - prev["chan_by_block"][b]["vessel"] for b in ("A", "B"))
            d_move = (sum(cur["chan_by_block"][b]["move"]
                          - prev["chan_by_block"][b]["move"] for b in ("A", "B"))
                      + cur["route_cost"] - prev["route_cost"])
            row = {"pair": i, "k": k, "job_id": a["job_id"], "t": a["t"],
                   "pred_net_gain": a["net_gain"],
                   "pred_out_relief": a["out_relief"], "pred_in_burden": a["in_burden"],
                   "d_total_v1": cur["total_v1"] - prev["total_v1"],
                   "d_v2": cur["v2_total"] - prev["v2_total"],
                   "d_src_truck": d_src, "d_dst_truck": d_dst,
                   "d_vessel": d_vessel, "d_move": d_move,
                   # 예측 대응 잔차 (동결 분해)
                   "resid_source": d_src + a["out_relief"],
                   "resid_receiver": d_dst - a["in_burden"],
                   "resid_vessel": d_vessel,
                   "resid_move": d_move - ROUTE_S / 3600.0,
                   "resid_total": (cur["total_v1"] - prev["total_v1"])
                                  - (-a["net_gain"])}
            ep_rows.append(row)
            per_decision.append(row)
        per_episode.append({"pair": i, "n": len(actions),
                            **{f"mean_{key}": fmean(r[key] for r in ep_rows)
                               for key in ("resid_source", "resid_receiver",
                                           "resid_vessel", "resid_move",
                                           "resid_total", "d_total_v1", "d_v2")}})
    axes = {"source": [e["mean_resid_source"] for e in per_episode],
            "receiver": [e["mean_resid_receiver"] for e in per_episode],
            "vessel": [e["mean_resid_vessel"] for e in per_episode]}
    axis_summary = {k: {"mean": fmean(v), "median": median(v)} for k, v in axes.items()}
    dominant = max(axis_summary, key=lambda k: abs(axis_summary[k]["mean"]))
    res = {"protocol": "사슬 한계효과 run_k − run_{k−1} (동결) · 통계 단위 = 에피소드 쌍",
           "n_decisions": len(per_decision), "n_episodes": len(per_episode),
           "per_decision": per_decision, "per_episode": per_episode,
           "axis_residual_summary(에피소드 묶음)": axis_summary,
           "dominant_axis(동결 규칙)": dominant}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(json.dumps({"dominant": dominant, **axis_summary,
                      "resid_total_mean(에피소드)": fmean(e["mean_resid_total"]
                                                          for e in per_episode)},
                     ensure_ascii=False, default=str))
    return res


# ---------------------------------------------------------- 1단계 정정 재실행 (27차 동결)
# 계약 불일치 정정: ①달성 가능 본선 마감 켬 ②전 축 v2 실현비용 분해 ③이동비용 v2 총비용
# 포함(route/3600 명시 가산) ④각 런 완주·backlog·정책 예외 저장+하드 검사 ⑤구 14건은
# 개발 이력 보존 — **정렬 환경 새 표본**(신규 대역 906100+)으로 축 재선정.
# 판정(동결): 지배 축 = |에피소드 묶음 평균 v2 잔차| 최대 ∧ **leave-one-out 과반 1위**
# 일 때만 "안정 1위" → 2단계 진행. 아니면 불안정 보고·2단계 보류.
import dataclasses

from ..integrated import TerminalSimulator
from ..integrated.cost_curve_v2 import (KAPPA_V2P_PATH, KappaFit, j_truck_realized,
                                        j_vessel_realized)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from ..integrated.seedbank import assign_band, independence_report
from ..integrated.transfer_quote import TransferQuoteResolver
from ..integrated.vessel import VesselWorkType
from .yr105_conditional_transfer import vessel_slack_min_s
from .yr138_episode_pilot import SLA_ANCHOR

BAND2_PATH = OUT / "band_aligned.json"
BAND2_START, BAND2_N = 906_100, 8


def _sim_aligned(cell, seed):
    prof = build_calibrated_profile()
    params = dataclasses.replace(
        calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
        time_contract_v2=True, gate_block_contract=True,
        vessel_deadline_achievable=True)          # ★27차: 달성 가능 마감
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, params),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def _gen_aligned(key, cell, seed):
    prof = build_calibrated_profile()
    params = dataclasses.replace(
        calibrated_load_params(cell[0], vessel_deadline_mult=cell[1]),
        time_contract_v2=True, gate_block_contract=True,
        vessel_deadline_achievable=True)
    return generate_terminal_scenario(prof, seed, params)


def make_band_aligned():
    import re
    pat = re.compile(r"rz1:[0-9a-f]{16}")
    exclude: set[str] = set()
    for p in Path("outputs/reports").rglob("*.json"):
        if OUT in p.parents:
            continue
        try:
            exclude |= set(pat.findall(p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    band = assign_band(family="y149-aligned", cells=_CELLS, n=BAND2_N,
                       generate=_gen_aligned, exclude=exclude, start_seed=BAND2_START)
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    OUT.mkdir(parents=True, exist_ok=True)
    BAND2_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] aligned {sum(len(v) for v in band.seeds.values())} seeds frozen")


def _v2_split(sim) -> tuple[float, float]:
    """블록 v2 실현 hard — (트럭 항, 본선 항) 분리 (yr138._v2_hard_total 의 분해판)."""
    sla = float(sim.profile.long_wait_sla_s)
    l_t = SLA_ANCHOR[0] + sla + SLA_ANCHOR[1] + SLA_ANCHOR[2]
    truck = 0.0
    tl = getattr(sim, "time_ledger", None)
    if tl is not None:
        for r in tl.records.values():
            a = getattr(r, "gate_in", None)
            if a is None:
                continue
            o = getattr(r, "gate_out", None)
            truck += j_truck_realized(o if o is not None else float(sim.end), a, a + l_t)
    vessel = 0.0
    for v in sim.vessels.values():
        if v.work_type != VesselWorkType.LOAD:
            continue
        p = v.plan.planned_completion_s
        if p is None:
            continue
        f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
        vessel += j_vessel_realized(f if f is not None else float(sim.end), p)
    return truck, vessel


def _mbt_aligned(seeds):
    return MultiBlockTerminal({"A": _sim_aligned(_CELLS["A"], seeds["A"]),
                               "B": _sim_aligned(_CELLS["B"], seeds["B"])})


def _run_core(mbt, review_fn, tag: str) -> dict:
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    res = mbt.run(policy, review_fn, lambda sim, t0, t1, raw: 0.0)
    mbt.check_invariants()
    # ★28차 정정: 모든 replay 경로에서 backlog·정책예외를 **assert** (구판은 저장만 했다 —
    # "하드 검사 통과" 표현이 성립하지 않았음).
    bl = sum(s.unfinished_backlog() for s in mbt.blocks.values())
    assert exc["n"] == 0, f"{tag}: 정책 예외 {exc['n']}"
    assert bl == 0, f"{tag}: backlog {bl}"
    route = res["route_cost_s"] / 3600.0
    split = {b: _v2_split(s) for b, s in mbt.blocks.items()}
    v2_total = sum(t + v for t, v in split.values()) + route      # ★이동비용 v2 포함
    return {"v2_total": v2_total, "route": route,
            "v2_truck_by_block": {b: s[0] for b, s in split.items()},
            "v2_vessel_by_block": {b: s[1] for b, s in split.items()},
            "backlog": sum(s.unfinished_backlog() for s in mbt.blocks.values()),
            "policy_exceptions": exc["n"]}


def _travel_fn_for(tag):
    def travel_fn(src, jid):
        rng = random.Random(f"y149:{tag}:{src}:{jid}")
        return trunc_normal(rng, GATE_BLOCK_MEAN_S,
                            GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
                            lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)
    return travel_fn


def diagnose_aligned() -> dict:
    band = json.loads(BAND2_PATH.read_text(encoding="utf-8"))
    pairs = list(zip(band["seeds"]["A"], band["seeds"]["B"]))
    kf = KappaFit.load(KAPPA_V2P_PATH, require_contract_physics=True)
    per_decision, per_episode, dev_runs = [], [], []
    for i, (sa, sb) in enumerate(pairs):
        seeds = {"A": sa, "B": sb}
        tag = f"a{i}"
        # ① 정렬 환경 견적 파일럿(새 표본 생성 — 현행 resolver: 최대 NetGain·version 검사)
        resolver = TransferQuoteResolver(kf, travel_fn=_travel_fn_for(tag),
                                         vessel_slack_fn=vessel_slack_min_s)
        base = _run_core(_mbt_aligned(seeds), resolver.review, tag)
        assert base["policy_exceptions"] == 0, "정책 예외 — 하드 검사 위반"
        actions = sorted((r for r in resolver.ledger if r["decision"] == "TRANSFER"),
                         key=lambda r: r["t"])
        dev_runs.append({"pair": i, "n_transfers": len(actions), **base})
        if not actions:
            continue
        print(f"[diag2] pair{i} n={len(actions)}", flush=True)
        # ② 사슬 리플레이 j=0..n (v2 분해·하드 검사)
        runs = []
        for j in range(len(actions) + 1):
            rr = ReplayResolver(actions, j, _travel_fn_for(tag))
            out = _run_core(_mbt_aligned(seeds), rr.review, tag)
            assert rr.done == j and out["policy_exceptions"] == 0
            runs.append(out)
        ep_rows = []
        for k, a in enumerate(actions, start=1):
            prev, cur = runs[k - 1], runs[k]
            d_src = (cur["v2_truck_by_block"][a["src"]]
                     - prev["v2_truck_by_block"][a["src"]])
            d_dst = (cur["v2_truck_by_block"][a["dst"]]
                     - prev["v2_truck_by_block"][a["dst"]])
            d_vsl = sum(cur["v2_vessel_by_block"][b] - prev["v2_vessel_by_block"][b]
                        for b in ("A", "B"))
            d_route = cur["route"] - prev["route"]
            row = {"pair": i, "k": k, "job_id": a["job_id"], "t": a["t"],
                   "pred_net_gain": a["net_gain"], "pred_out_relief": a["out_relief"],
                   "pred_in_burden": a["in_burden"],
                   "d_v2_total": cur["v2_total"] - prev["v2_total"],
                   "resid_source": d_src + a["out_relief"],
                   "resid_receiver": d_dst - a["in_burden"],
                   "resid_vessel": d_vsl,
                   "resid_move": d_route - ROUTE_S / 3600.0,
                   "backlog_after": cur["backlog"]}
            ep_rows.append(row)
            per_decision.append(row)
        per_episode.append({"pair": i, "n": len(actions),
                            **{f"mean_{key}": fmean(r[key] for r in ep_rows)
                               for key in ("resid_source", "resid_receiver",
                                           "resid_vessel", "d_v2_total")}})
    # ★28차 집계 계약(동결): **에피소드 누적합 = 주판정**(터미널 총비용과 같은 단위),
    # 이송 1건 평균 = 보조 교정지표. 축 확정은 이 표본이 아니라 미열람 5셀 자료에서 한다.
    axes_sum = {"source": [e["mean_resid_source"] * e["n"] for e in per_episode],
                "receiver": [e["mean_resid_receiver"] * e["n"] for e in per_episode],
                "vessel": [e["mean_resid_vessel"] * e["n"] for e in per_episode]}
    axes_mean = {"source": [e["mean_resid_source"] for e in per_episode],
                 "receiver": [e["mean_resid_receiver"] for e in per_episode],
                 "vessel": [e["mean_resid_vessel"] for e in per_episode]}

    def _agg(ax):
        s = {k: {"mean": fmean(v), "median": median(v)} for k, v in ax.items()}
        dom = max(s, key=lambda k: abs(s[k]["mean"]))
        n = len(next(iter(ax.values())))
        wins = {k: 0 for k in ax}
        for drop in range(n):
            sub = {k: [v for i, v in enumerate(vals) if i != drop]
                   for k, vals in ax.items()}
            wins[max(sub, key=lambda k: abs(fmean(sub[k])))] += 1
        return s, dom, wins, wins[dom] > n / 2

    summary, dominant, loo_wins, stable = _agg(axes_sum)          # 주판정
    summary_sec, dom_sec, loo_sec, stable_sec = _agg(axes_mean)   # 보조
    n_ep = len(per_episode)
    res = {"protocol": "27차 정정 재실행 + 28차 집계 계약(누적합 주판정·1건 평균 보조)·"
                       "전 replay backlog·예외 assert. 축 확정은 미열람 5셀 자료에서 수행",
           "n_decisions": len(per_decision), "n_episodes": n_ep,
           "dev_runs": dev_runs, "per_decision": per_decision,
           "per_episode": per_episode,
           "primary(에피소드 누적합)": {"summary": summary, "loo_wins": loo_wins,
                                        "dominant": dominant, "stable": stable},
           "secondary(이송 1건 평균)": {"summary": summary_sec, "loo_wins": loo_sec,
                                        "dominant": dom_sec, "stable": stable_sec},
           "note": "이 표본은 두 집계 모두 열람됨 — 보정축은 여기서 확정하지 않는다."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag_aligned.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"주판정_누적합": {"dominant": dominant, "stable": stable,
                                        "loo": loo_wins,
                                        **{k: round(v["mean"], 4)
                                           for k, v in summary.items()}},
                      "보조_1건평균": {"dominant": dom_sec, "loo": loo_sec}},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--make-band-aligned", action="store_true")
    ap.add_argument("--diag-aligned", action="store_true")
    a = ap.parse_args()
    if a.diag:
        diagnose()
    if a.make_band_aligned:
        make_band_aligned()
    if a.diag_aligned:
        diagnose_aligned()
    print("DONE")
