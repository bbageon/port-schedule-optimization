"""대표성 격자 — "일반 항만 부하 체제"에서 JR_NEW 효과 유지 검증 (YR-071 1단계 증거).

격자: 부하 {48, 80, 112 외부트럭} × 장치율 {0.30, 0.65} × 도착 {균등, 2봉 피크} = 12 셀.
정책 고정: SF_SPT(강휴리스틱, 목적 무관) vs JR_NEW(트럭대기 1차 목적 최적화 rollout).
판정: 실제 물리지표(평균/p95 대기·완료율·backlog·swa)로 — 비용 게이밍 논점 원천 차단.
피크 도착: 생성 시나리오의 외부트럭 도착시각을 2봉 혼합분포 quantile 로 재매핑
(소스 미수정 — ETA 오차·게이트 소요 delta 는 보존, 결정론 유지).
"""
import json, math, time, dataclasses
from statistics import fmean

from yard_rl.contract.schema import COST_TERMS
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from yard_rl.integrated.cost_config import RewardCalculator, default_assumed_config
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.baselines import (
    ResolverPolicy, ServiceFirstSPTPreference, JointRolloutGreedy, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/rep_grid/rep_grid.json"
PROFILE = build_integrated_profile()
GEN = CandidateGenerator()
SEEDS = [340000, 340001, 340002]
N_VES = 3
PRIORITY = {"truck_wait": 100.0, "long_wait": 100.0, "vessel_delay": 5.0, "depart_delay": 5.0,
            "sts_wait": 3.0, "transfer_wait": 3.0, "rehandle": 1.0, "crane_travel": 1.0,
            "empty_travel": 1.0, "resequence": 1.0, "imbalance": 0.5,
            "interference": 0.1, "lane_cong": 0.1}
RC_NEW = RewardCalculator(default_assumed_config().with_weight({t: PRIORITY[t] for t in COST_TERMS}))

# ---- 2봉 피크 도착 (게이트 러시 근사): 35% N(0.22H,0.05H) + 40% N(0.62H,0.07H) + 25% 균등
_PEAKS = ((0.35, 0.22, 0.05), (0.40, 0.62, 0.07))
_W_UNIF = 0.25


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cdf(t, H):
    v = _W_UNIF * min(max(t / H, 0.0), 1.0)
    for w, mu, sg in _PEAKS:
        v += w * _phi((t - mu * H) / (sg * H))
    return v


def _quantile(u, H):
    lo, hi = 0.0, H
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if _cdf(mid, H) < u:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def peakify(scn, H):
    """외부트럭 도착을 균등 quantile → 2봉 혼합 quantile 로 재매핑 (delta 구조 보존)."""
    jobs = []
    for j in scn.jobs:
        if j.actual_block_arrival is None:          # 본선 작업 — 그대로
            jobs.append(j)
            continue
        old = j.actual_block_arrival
        u = min(max(old / H, 1e-6), 1.0 - 1e-6)
        new = _quantile(u, H)
        gate_travel = old - (j.actual_gate_in or 0.0)
        eta_delta = (j.provided_eta - old) if j.provided_eta is not None else None
        jobs.append(dataclasses.replace(
            j, actual_block_arrival=new,
            actual_gate_in=max(0.0, new - gate_travel),
            provided_eta=(max(0.0, new + eta_delta) if eta_delta is not None else None)))
    return dataclasses.replace(scn, jobs=jobs)


def make_sim(seed, n_ext, fill, peaked):
    params = TerminalGenParams(n_external=n_ext, n_vessels=N_VES, fill_ratio=fill,
                               eta_error_s=300.0)
    scn = generate_terminal_scenario(PROFILE, seed, params)
    if peaked:
        scn = peakify(scn, params.horizon_s)
    return TerminalSimulator(PROFILE, scn, check_invariants=True)


def episode(policy_fac, seed, n_ext, fill, peaked):
    sim = make_sim(seed, n_ext, fill, peaked)
    pol = policy_fac()
    r = run_joint_episode(sim, pol, RC_NEW, generator=GEN)
    return {"wait": round(r["mean_wait_min"], 3), "p95": round(r["p95_wait_min"], 3),
            "compl": round(r["completion_rate"], 3), "backlog": r["backlog"],
            "swa": round(r["action_mix"]["serve_when_available"], 3),
            "cost": round(r["total_cost"], 1), "dec": r["n_decisions"],
            "trunc": r.get("combo_truncations", 0), "rehandles": r["rehandles"]}


POLS = {
    "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
    "JR_NEW": lambda: JointRolloutGreedy(RC_NEW, horizon_s=600.0, max_combos=32, generator=GEN),
}


def main():
    t0 = time.time()
    loads = [("L48", 48), ("L80", 80), ("L112", 112)]
    fills = [("F30", 0.30), ("F65", 0.65)]
    arrs = [("U", False), ("P", True)]
    cells = [(f, l, a) for f in fills for l in loads for a in arrs]   # 싼 축(fill) 우선
    out = {"seeds": SEEDS, "n_vessels": N_VES, "priority": PRIORITY, "cells": {}}
    hdr = (f"{'cell':14s} {'pol':7s} {'wait':>6s} {'p95':>6s} {'compl':>6s} {'bklog':>5s} "
           f"{'swa':>5s} {'dec':>4s} {'reh':>4s} {'JRwin':>5s}")
    print(hdr, flush=True)
    for (fl, fv), (ll, lv), (al, pk) in cells:
        cid = f"{ll}/{fl}/{al}"
        cell = {}
        for pname, fac in POLS.items():
            rows = [episode(fac, s, lv, fv, pk) for s in SEEDS]
            agg = {k: round(fmean(r[k] for r in rows), 3) for k in
                   ("wait", "p95", "compl", "swa")}
            agg["backlog"] = round(fmean(r["backlog"] for r in rows), 1)
            agg["dec"] = round(fmean(r["dec"] for r in rows), 0)
            agg["rehandles"] = round(fmean(r["rehandles"] for r in rows), 1)
            agg["trunc"] = sum(r["trunc"] for r in rows)
            agg["per_seed"] = rows
            cell[pname] = agg
        wins = sum(1 for i in range(len(SEEDS))
                   if cell["JR_NEW"]["per_seed"][i]["wait"] < cell["SF_SPT"]["per_seed"][i]["wait"])
        cell["jr_wait_wins"] = f"{wins}/{len(SEEDS)}"
        cell["wait_gap"] = round(cell["SF_SPT"]["wait"] - cell["JR_NEW"]["wait"], 3)
        for pname in POLS:
            a = cell[pname]
            print(f"{cid:14s} {pname:7s} {a['wait']:>6.2f} {a['p95']:>6.2f} {a['compl']:>6.3f} "
                  f"{a['backlog']:>5.1f} {a['swa']:>5.2f} {a['dec']:>4.0f} {a['rehandles']:>4.1f} "
                  f"{cell['jr_wait_wins'] if pname == 'JR_NEW' else '':>5s}", flush=True)
        out["cells"][cid] = cell
    out["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  (elapsed {out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
