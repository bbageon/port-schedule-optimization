"""크레인 라벨의 **목표 격차**를 잰다 — `CRANE_ADV_SCALE` 재동결용 ([[YR-308]]).

지금 눈금 100,000원은 재배정층 [[YR-217]] 에서 **빌려온 잠정값**이다. 크레인 결정의
격차를 실측한 적이 없다. 눈금이 어긋나면:

    너무 크면  목표가 0 근처로 뭉개져 **신호가 묻힌다**
    너무 작으면 목표가 커져 **학습이 튄다**

잰다:  |Φ_사실 − Φ_차점자| 의 중앙값 (원)  — 부하별로 따로 본다.

    PYTHONPATH=src python scripts/v4/probe_crane_scale.py --labels 24 --workers 20
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from yard_rl.v4.crane.policy import CRANE_ADV_SCALE, from_advantage
from yard_rl.v4.crane.fit import scale_health
from yard_rl.v4.stage.episode import run_episode
from yard_rl.v4.stage.rollout import RolloutBudget

DIAG = 9_900_000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="크레인 라벨 격차 실측")
    ap.add_argument("--loads", default="3500,7500,15000")
    ap.add_argument("--labels", type=int, default=24)
    ap.add_argument("--horizon-h", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=DIAG + 800)
    ap.add_argument("--out", default="outputs/v4/crane-scale")
    a = ap.parse_args(argv)

    loads = [int(x) for x in a.loads.split(",") if x]
    rows, allgaps = [], []
    for i, load in enumerate(loads):
        t0 = time.time()
        seed = a.seed + i * 10 + load // 1000
        ep = run_episode(load=load, arm="NO_REALLOC", dispatcher="RL_CRANE",
                         seed=seed, horizon_s=a.horizon_h * 3600.0,
                         crane_budget=RolloutBudget(max_labels=a.labels),
                         workers=a.workers)
        ss = ep.crane_labels
        # 표본이 (고른 것, 대안) 짝으로 들어온다 — 짝의 목표 차이가 곧 Φ 격차다.
        gaps = [abs(from_advantage(ss[k].target - ss[k + 1].target))
                for k in range(0, len(ss) - 1, 2)]
        allgaps += gaps
        cs = ep.crane_stats
        r = {"load": load, "seed": seed, "secs": round(time.time() - t0, 1),
             "phi_krw": ep.phi_krw, "n_labels": len(gaps),
             "median_gap_krw": (st.median(gaps) if gaps else 0.0),
             "mean_gap_krw": (st.fmean(gaps) if gaps else 0.0),
             "max_gap_krw": (max(gaps) if gaps else 0.0),
             "zero_ratio": (cs.get("zero", 0) / max(1, len(gaps))),
             "usable_ratio": cs.get("usable", 0) / max(1, cs.get("seen", 1)),
             **{k: cs.get(k, 0) for k in
                ("seen", "usable", "sampled", "no_alt", "no_pick",
                 "factual_mismatch", "force_failed", "labeled")}}
        rows.append(r)
        print(f"부하 {load:>6,} · {r['secs']/60:>5.1f}분 · 라벨 {r['n_labels']:>3}"
              f"/{r['sampled']:<3} (쓸수있음 {r['usable_ratio']:>5.1%}) · "
              f"격차 중앙 {r['median_gap_krw']:>12,.0f}원 "
              f"평균 {r['mean_gap_krw']:>12,.0f} 최대 {r['max_gap_krw']:>12,.0f} · "
              f"0비율 {r['zero_ratio']:>5.1%}")

    med = st.median(allgaps) if allgaps else 0.0
    print(f"\n■ 전체 {len(allgaps)}건 · 중앙 격차 {med:,.0f}원")
    print(f"  지금 눈금 {CRANE_ADV_SCALE:,.0f}원 → {scale_health(med)}")
    if med > 0:
        # 목표 중앙이 1 근처가 되는 눈금 — 재배정층 [[YR-217]] 과 같은 규약
        print(f"  ★권고 눈금: {med:,.0f}원 (지금의 {med/CRANE_ADV_SCALE:.2f}배)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale.json").write_text(
        json.dumps({"rows": rows, "median_gap_krw": med,
                    "current_scale": CRANE_ADV_SCALE,
                    "horizon_h": a.horizon_h, "labels": a.labels},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out/'scale.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
