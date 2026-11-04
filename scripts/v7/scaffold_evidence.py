"""YR-332-a 증거 — DGT 리드타임 분위 · 하루 규칙 크레인 · 3일 연속 무대(생성 입력 그대로).

    PYTHONPATH=src python scripts/v7/scaffold_evidence.py outputs/reports/yr332_v7_scaffold/scaffold_evidence.json
"""
import collections
import json
import sys
import time

from yard_rl.v7 import GENERATION, V6_BASE_COMMIT
from yard_rl.v7.reward.krw import KRW_VESSEL_GT_HOUR
from yard_rl.v7.stage.dgt_lead import no_notice_share, sample_dgt_lead_s
from yard_rl.v7.stage.episode import run_episode
from yard_rl.v7.stage.month import plan_days
from yard_rl.v7.stage.month_run import run_month
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.profiles import build_h21_profile

SEED = 9_900_332
H = 3600.0
out = {"generation": GENERATION, "v6_base_commit": V6_BASE_COMMIT,
       "krw_vessel_gt_hour": KRW_VESSEL_GT_HOUR}

out["dgt_sampler_hours"] = {f: {p: round(sample_dgt_lead_s(p, f) / H, 3) for p in (0.25, 0.5, 0.75, 0.9)}
                            | {"no_notice_share": round(no_notice_share(f), 4)}
                            for f in ("GATE_IN", "GATE_OUT")}

stage = {}
for lm in ("DGT", "DIST", "FIXED"):
    b = build_stage(load=3_500, seed=SEED, profile=build_h21_profile(), lead_mode=lm)
    row = {}
    for f in ("GATE_IN", "GATE_OUT"):
        xs = sorted(e["lead_s"] for e in b["schedule"] if e["flow"] == f)
        row[f] = {"n": len(xs), "zero_share": round(sum(x == 0 for x in xs) / len(xs), 4),
                  "median_h": round(xs[len(xs) // 2] / H, 3), "p75_h": round(xs[int(len(xs) * .75)] / H, 3)}
    stage[lm] = row
out["stage_lead_in_day_load3500"] = stage

days = {}
for load in (300, 3_500):
    for lm in ("DGT", "DIST"):
        t0 = time.time()
        r = run_episode(load=load, seed=SEED, lead_mode=lm)
        b = r.breakdown
        days[f"{load}/{lm}"] = {"secs": round(time.time() - t0, 1), "admitted": r.admitted,
                                "traded": r.traded_edges, "policy_exceptions": r.policy_exceptions,
                                "phi_krw": round(r.phi_krw), "c_wait": round(b["c_wait"]),
                                "c_vessel": round(b["c_vessel"]), "vessel_share": round(b["c_vessel"] / r.phi_krw, 4),
                                "mean_turn_min": round(b["mean_turn_time_s"] / 60, 2),
                                "p90_turn_min": round(b["p90_turn_time_s"] / 60, 2),
                                "n_trucks": b["n_trucks"], "n_censored": b["n_censored"]}
out["sf_spt_day"] = days

month = {}
for lm in ("DGT", "DIST"):
    r = run_month(seed=SEED, days=plan_days(SEED, loads=[300, 300, 300]), lead_mode=lm)
    month[lm] = {"admitted": r.admitted, "skipped": r.skipped, "traded": r.traded_edges,
                 "policy_exceptions": r.policy_exceptions,
                 "skip_reasons": dict(collections.Counter(s["reason"] for s in r.truck_skips)),
                 "trucks_by_day": [d.n_trucks for d in r.days]}
out["month_3day_generated_input_load300"] = month
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
