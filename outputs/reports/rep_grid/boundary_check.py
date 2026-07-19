"""경계 셀 원인 분리 — JR 역전(L112/F65)이 체제 본질인가, 격자 경량화 탓인가.

최악 2셀(L112/F65/U·P)에서 JR 변형 3종 재실행:
  grid(재현): combos 32·h600 / full: combos 64·h600 / deep: combos 64·h1800
full/deep 이 SF 를 회복하면 → 경량화 탓 (경계 주장 철회). 여전히 열세면 → 체제 본질
(포화에서 SPT 순서가 근사최적이라는 대기이론과 정합) → 경계를 정직하게 보고.
"""
import sys, json, time
sys.path.insert(0, "/mnt/c/Users/geonu/AppData/Local/Temp/claude/"
                   "c--Users-geonu-Desktop-port-reinforcement/"
                   "adbc7e00-3805-4b61-b786-7c6475e2fff8/scratchpad")
from statistics import fmean
from rep_grid import make_sim, RC_NEW, GEN, SEEDS
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          JointRolloutGreedy, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/rep_grid/boundary_check.json"


def run(fac, seed, peaked):
    r = run_joint_episode(make_sim(seed, 112, 0.65, peaked), fac(), RC_NEW, generator=GEN)
    return {"wait": round(r["mean_wait_min"], 2), "p95": round(r["p95_wait_min"], 2),
            "compl": round(r["completion_rate"], 3), "swa": round(r["action_mix"]["serve_when_available"], 3),
            "trunc": r.get("combo_truncations", 0)}


ARMS = {
    "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
    "JR_grid_c32_h600": lambda: JointRolloutGreedy(RC_NEW, horizon_s=600.0, max_combos=32, generator=GEN),
    "JR_full_c64_h600": lambda: JointRolloutGreedy(RC_NEW, horizon_s=600.0, max_combos=64, generator=GEN),
    "JR_deep_c64_h1800": lambda: JointRolloutGreedy(RC_NEW, horizon_s=1800.0, max_combos=64, generator=GEN),
}


def main():
    t0 = time.time()
    out = {}
    for label, peaked in (("L112/F65/U", False), ("L112/F65/P", True)):
        print(f"\n===== {label} =====", flush=True)
        cell = {}
        for name, fac in ARMS.items():
            rows = [run(fac, s, peaked) for s in SEEDS]
            agg = {k: round(fmean(r[k] for r in rows), 2) for k in ("wait", "p95", "compl", "swa")}
            agg["trunc"] = sum(r["trunc"] for r in rows)
            agg["per_seed_wait"] = [r["wait"] for r in rows]
            cell[name] = agg
            print(f"{name:18s} wait={agg['wait']:>6.2f} p95={agg['p95']:>6.2f} "
                  f"compl={agg['compl']:.3f} swa={agg['swa']:.2f} trunc={agg['trunc']}", flush=True)
        out[label] = cell
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
