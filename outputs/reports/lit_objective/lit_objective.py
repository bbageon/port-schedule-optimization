"""YR-080 프로토타입 — 문헌정합 목적함수 (정규화 우선·간섭 제거·본선 병기·항 축소).

문헌 근거(2026-07-20-목적함수-문헌근거):
 ② 정규화 먼저: scale_k = 기준정책(SF_SPT) 하 실측 raw_k (각 항 baseline 에서 O(1)) — Marler&Arora
 ③ 간섭·재순서 제거: 보상 weight=0 (마스크/resolver 소관)
 ① 본선 병기: 본선정시성(sts_wait·vessel_delay·depart_delay)을 육상서비스와 공동 1차 weight
 ④ 항 축소: 1차 육상서비스 / 1차 본선 / 2차 효율·proxy 로 묶어 weight 부여
검증: OLD(assumed) vs LIT 목적에서 SF_SPT vs JR — 트럭대기·본선지연·swa·완주·간섭기여.
성질: (a) 게이밍 닫힘(건강정책 유리) (b) 트럭대기 개선 유지 (c) 본선 미방치(vessel_delay 폭발 없음).
"""
import sys, json, time
sys.path.insert(0, "/mnt/c/Users/geonu/AppData/Local/Temp/claude/"
                   "c--Users-geonu-Desktop-port-reinforcement/"
                   "adbc7e00-3805-4b61-b786-7c6475e2fff8/scratchpad")
from statistics import fmean
from rep_grid import make_sim, GEN
from yard_rl.contract.schema import COST_TERMS
from yard_rl.integrated.cost import ASSUMED_SCALE
from yard_rl.integrated.cost_config import (RewardCalculator, default_assumed_config,
                                            Provenance, ProvBasis)
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          JointRolloutGreedy, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/lit_objective/lit_objective.json"
SEEDS = [340000, 340001, 340002]
HZN = 600.0   # 프로토타입 속도용 (생산 권고는 1800s, YR-078)

# 문헌정합 weight (정규화 후 = 선호). 1차 서비스·1차 본선 = 1.0, 2차 효율/proxy = 0.3, 제거 = 0.
LIT_W = {
    "truck_wait": 1.0, "long_wait": 1.0, "transfer_wait": 1.0,        # 1차 육상 직접 서비스
    "sts_wait": 1.0, "vessel_delay": 1.0, "depart_delay": 1.0,        # 1차 본선 정시성 (KPI 위계)
    "crane_travel": 0.3, "empty_travel": 0.3, "rehandle": 0.3,        # 2차 효율
    "imbalance": 0.3, "lane_cong": 0.3,                              # 2차 균형·혼잡 proxy
    "interference": 0.0, "resequence": 0.0,                          # 제거 (마스크/흡수)
}

RC_OLD = RewardCalculator.assumed_default()


def fit_scales():
    """기준정책(SF_SPT) 실측 raw 로 scale 동결 (정규화 우선). raw_k = contrib_k × ASSUMED_SCALE_k."""
    rows = []
    for s in SEEDS:
        r = run_joint_episode(make_sim(s, 80, 0.30, False),
                              ResolverPolicy(ServiceFirstSPTPreference(), "SF"), RC_OLD, generator=GEN)
        rows.append(r["term_contrib"])
    scale = {}
    for t in COST_TERMS:
        raw = fmean(r.get(t, 0.0) for r in rows) * ASSUMED_SCALE[t]
        scale[t] = max(raw, ASSUMED_SCALE[t] * 0.05)   # 0-항 바닥: 발동 전엔 기여 ~0 유지
    return scale


def build_lit_rc(scale):
    prov = Provenance(ProvBasis.FITTED_BASELINE, "SF_SPT baseline 실측", "YR-080 프로토타입")
    cfg = default_assumed_config().with_scale(scale, prov=prov).with_weight(
        {t: LIT_W[t] for t in COST_TERMS})
    return RewardCalculator(cfg)


def episode(fac, rc, seed, n_ext, fill):
    r = run_joint_episode(make_sim(seed, n_ext, fill, False), fac(rc), rc, generator=GEN)
    tot = max(1e-9, r["total_cost"])
    return {"wait": round(r["mean_wait_min"], 3), "p95": round(r["p95_wait_min"], 3),
            "swa": round(r["action_mix"]["serve_when_available"], 3),
            "compl": round(r["completion_rate"], 3),
            "vdelay": round(r["vessel_delay_min"], 2),
            "interf_share": round(r["term_contrib"].get("interference", 0.0) / tot, 3),
            "truck_share": round(r["term_contrib"].get("truck_wait", 0.0) / tot, 3),
            "vessel_share": round(sum(r["term_contrib"].get(t, 0.0)
                                      for t in ("sts_wait", "vessel_delay", "depart_delay")) / tot, 3)}


POLS = {"SF_SPT": lambda rc: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
        "JR": lambda rc: JointRolloutGreedy(rc, horizon_s=HZN, max_combos=64, generator=GEN)}


def main():
    t0 = time.time()
    scale = fit_scales()
    RC_LIT = build_lit_rc(scale)
    print("scale 동결 (일부):", {k: round(scale[k], 1) for k in
          ("truck_wait", "interference", "lane_cong", "vessel_delay", "rehandle")}, flush=True)
    out = {"seeds": SEEDS, "horizon_s": HZN, "lit_weight": LIT_W,
           "scale_fit": {k: round(v, 2) for k, v in scale.items()}, "cells": {}}
    cells = [("L80/F30", 80, 0.30), ("L112/F65", 112, 0.65)]
    for cid, n_ext, fill in cells:
        print(f"\n===== {cid} =====", flush=True)
        cell = {}
        for obj, rc in (("OLD", RC_OLD), ("LIT", RC_LIT)):
            for p, fac in POLS.items():
                rows = [episode(fac, rc, s, n_ext, fill) for s in SEEDS]
                agg = {k: round(fmean(r[k] for r in rows), 3) for k in
                       ("wait", "p95", "swa", "compl", "vdelay", "interf_share",
                        "truck_share", "vessel_share")}
                cell[f"{obj}/{p}"] = agg
                print(f"{obj+'/'+p:10s} wait={agg['wait']:>6.2f} p95={agg['p95']:>6.2f} "
                      f"swa={agg['swa']:.2f} compl={agg['compl']:.3f} vdelay={agg['vdelay']:>5.2f} "
                      f"| 기여 간섭={agg['interf_share']:.2f} 트럭={agg['truck_share']:.2f} "
                      f"본선={agg['vessel_share']:.2f}", flush=True)
        # 성질 판정
        old_gap = round(cell["OLD/SF_SPT"]["wait"] - cell["OLD/JR"]["wait"], 3)
        lit_gap = round(cell["LIT/SF_SPT"]["wait"] - cell["LIT/JR"]["wait"], 3)
        cell["verdict"] = {"old_wait_gap": old_gap, "lit_wait_gap": lit_gap,
                           "interf_removed": cell["LIT/JR"]["interf_share"] < 0.001,
                           "vessel_visible": cell["LIT/JR"]["vessel_share"] > 0.0}
        print(f"  판정: 트럭대기 격차 OLD {old_gap:+.2f} → LIT {lit_gap:+.2f} · "
              f"간섭제거={cell['verdict']['interf_removed']} · 본선가시={cell['verdict']['vessel_visible']}",
              flush=True)
        out["cells"][cid] = cell
    out["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
