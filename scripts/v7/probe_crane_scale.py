"""v7 크레인 교사 눈금 계측 ([[YR-332-b]] 4항) — 값을 **바꾸지 않고 재기만** 한다.

v4 의 `CRANE_ADV_SCALE = 10,000원` 은 v4 무대에서 잰 라벨 격차 중앙(9,470원)을 반올림한
값이다. v7 은 본선 단가(35.88원)·리드타임(DGT)·행동 범위(선제 빈 이동 끔)가 달라
격차 분포가 달라질 수 있다. 이 스크립트는 하루마다 라벨을 모아

    부하 · 라벨 수 · 격차 중앙/평균/최대(원) · 차이0 비율 · 하한(1,000원) 걸림 · 사실불일치 · 강제실패

를 JSON 으로 남긴다. 동결은 다음 단계([[YR-332-c]])의 몫이다.

    PYTHONPATH=src python scripts/v7/probe_crane_scale.py --loads 3500,7500 --labels 24 \\
        --workers 4 --out outputs/reports/yr332_v7_scaffold/crane_scale_probe.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from yard_rl.v7 import GENERATION, V6_BASE_COMMIT
from yard_rl.v7.crane.policy import CRANE_ADV_SCALE, CRANE_SCALE_FLOOR
from yard_rl.v7.crane.train import DIAGNOSTIC_BASE
from yard_rl.v7.stage.episode import run_episode
from yard_rl.v7.stage.rollout import RolloutBudget


def probe(load: int, seed: int, labels: int, horizon_s: float, workers: int) -> dict:
    t0 = time.time()
    budget = RolloutBudget(max_labels=labels, identity_checks=0)
    ep = run_episode(load=load, seed=seed, dispatcher="RL_CRANE", crane_budget=budget,
                     horizon_s=horizon_s, workers=workers)
    ss = ep.crane_labels
    gaps = [abs(ss[i].phi - ss[i + 1].phi) for i in range(0, len(ss) - 1, 2)]
    cs = ep.crane_stats
    return {"load": load, "seed": seed, "secs": round(time.time() - t0, 1),
            "labels": len(gaps), "sampled": cs.get("sampled", 0),
            "usable": cs.get("usable", 0), "seen": cs.get("seen", 0),
            "factual_mismatch": cs.get("factual_mismatch", 0),
            "force_failed": cs.get("force_failed", 0),
            "zero_ratio": (sum(1 for g in gaps if g < 1e-6) / len(gaps)) if gaps else None,
            "gap_median_krw": st.median(gaps) if gaps else None,
            "gap_mean_krw": st.mean(gaps) if gaps else None,
            "gap_max_krw": max(gaps) if gaps else None,
            "day_scale_krw": cs.get("scale"), "floor_bound": cs.get("floor_bound"),
            "phi_krw": ep.phi_krw, "c_vessel": ep.breakdown.get("c_vessel"),
            "c_wait": ep.breakdown.get("c_wait")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="probe_crane_scale")
    ap.add_argument("--loads", default="3500,7500,15000")
    ap.add_argument("--labels", type=int, default=24)
    ap.add_argument("--seed", type=int, default=DIAGNOSTIC_BASE + 3320)
    ap.add_argument("--horizon-h", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs/reports/yr332_v7_scaffold/crane_scale_probe.json")
    a = ap.parse_args(argv)
    if a.seed < DIAGNOSTIC_BASE:
        ap.error("진단 대역(9,900,000 이상)만 쓴다")
    rows = []
    for i, load in enumerate(int(x) for x in a.loads.split(",") if x):
        r = probe(load, a.seed + 10 * i + load // 1000, a.labels, a.horizon_h * 3600.0, a.workers)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    all_gaps_median = st.median([r["gap_median_krw"] for r in rows if r["gap_median_krw"] is not None] or [0])
    out = {"generation": GENERATION, "v6_base_commit": V6_BASE_COMMIT,
           "v4_crane_adv_scale_krw": CRANE_ADV_SCALE, "scale_floor_krw": CRANE_SCALE_FLOOR,
           "horizon_h": a.horizon_h, "labels_per_day": a.labels, "rows": rows,
           "median_of_day_medians_krw": all_gaps_median,
           "note": "measurement only - CRANE_ADV_SCALE is not changed (YR-332-b)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
