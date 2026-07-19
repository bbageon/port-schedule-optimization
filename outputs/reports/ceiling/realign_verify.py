"""YR-071 ②재정렬 + ③검증 — 트럭 대기 1차 목적함수.

재정렬(weight only, 사용자 결정 '트럭 대기 1차'):
  truck_wait·long_wait ×100 (지배항) / vessel 계열 ×3~5 (부차 서비스) /
  interference·lane_cong ×0.1 (proxy 강등) / 나머지 ×1.
검증 2가지:
  A. 게이밍 닫힘? — OLD 목적에선 저-swa BEAM 이 이겼다. NEW 에선 건강한 정책(고 swa)이 이기고
     탐색기(JR/BEAM)가 NEW 를 최적화하면 swa 를 올려야(=옳은 방향) 한다.
  B. 진짜 헤드룸 — NEW 목적으로 천장(BEAM_wide) vs SF_SPT 재측정.
"""
import json, time
from statistics import fmean

from yard_rl.contract.schema import COST_TERMS
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from yard_rl.integrated.cost_config import RewardCalculator, default_assumed_config
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.baselines import (
    ResolverPolicy, ServiceFirstSPTPreference, FIFOPreference,
    JointRolloutGreedy, BeamLookahead, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/ceiling/realign_verify.json"
PROFILE = build_integrated_profile()
GEN = CandidateGenerator()
SEEDS = list(range(330000, 330008))       # 8 seeds
SEEDS_WIDE = list(range(330000, 330003))  # 3 seeds (BEAM_wide)

RC_OLD = RewardCalculator.assumed_default()

# ---- 재정렬 목적함수 (트럭 대기 1차) ----
PRIORITY = {"truck_wait": 100.0, "long_wait": 100.0,
            "vessel_delay": 5.0, "depart_delay": 5.0, "sts_wait": 3.0, "transfer_wait": 3.0,
            "rehandle": 1.0, "crane_travel": 1.0, "empty_travel": 1.0, "resequence": 1.0,
            "imbalance": 0.5, "interference": 0.1, "lane_cong": 0.1}
_base = default_assumed_config()
RC_NEW = RewardCalculator(_base.with_weight({t: PRIORITY[t] for t in COST_TERMS}))


def make_sim(seed, n_ext, n_ves):
    return TerminalSimulator(
        PROFILE, generate_terminal_scenario(
            PROFILE, seed, TerminalGenParams(n_external=n_ext, n_vessels=n_ves, eta_error_s=300.0)),
        check_invariants=True)


def run(fac, seeds, rc, n_ext, n_ves):
    rows = [run_joint_episode(make_sim(s, n_ext, n_ves), fac(rc), rc, generator=GEN) for s in seeds]
    return {"total": round(fmean(r["total_cost"] for r in rows), 2),
            "swa": round(fmean(r["action_mix"]["serve_when_available"] for r in rows), 3),
            "mean_wait": round(fmean(r["mean_wait_min"] for r in rows), 2),
            "p95_wait": round(fmean(r["p95_wait_min"] for r in rows), 2)}


# policy 팩토리 (rc 인자 — 탐색기는 rc 를 내부 최적화 목적으로 사용)
POLS = {
    "SF_SPT": lambda rc: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
    "FIFO": lambda rc: ResolverPolicy(FIFOPreference(), "FIFO"),
    "JR_600": lambda rc: JointRolloutGreedy(rc, horizon_s=600.0, generator=GEN),
    "BEAM_w3": lambda rc: BeamLookahead(rc, width=3, horizon_s=600.0, generator=GEN),
}


def block(label, rc, n_ext, n_ves, seeds):
    print(f"\n----- {label} -----")
    print(f"{'policy':8s} {'total':>8s} {'swa':>5s} {'mwait':>6s} {'p95':>6s}")
    res = {}
    for name, fac in POLS.items():
        res[name] = run(fac, seeds, rc, n_ext, n_ves)
        r = res[name]
        print(f"{name:8s} {r['total']:>8.2f} {r['swa']:>5.2f} {r['mean_wait']:>6.2f} {r['p95_wait']:>6.2f}")
    winner = min(res, key=lambda n: res[n]["total"])
    healthiest = max(res, key=lambda n: res[n]["swa"])
    print(f"  최저비용={winner}(swa {res[winner]['swa']}) / 최건강={healthiest}(swa {res[healthiest]['swa']}) "
          f"→ 게이밍 {'닫힘 ✅' if winner == healthiest else '잔존 ⚠'}")
    return res, {"winner": winner, "healthiest": healthiest, "gaming_closed": winner == healthiest}


def main():
    t0 = time.time()
    out = {"seeds": SEEDS, "priority": PRIORITY, "loads": {}}
    for load, (n_ext, n_ves) in (("MODERATE", (20, 2)), ("HIGHLOAD", (48, 3))):
        print(f"\n===== {load} (n_ext={n_ext} v={n_ves}) =====")
        old_res, old_v = block("OLD 목적 (assumed)", RC_OLD, n_ext, n_ves, SEEDS)
        new_res, new_v = block("NEW 목적 (트럭대기 1차)", RC_NEW, n_ext, n_ves, SEEDS)
        entry = {"OLD": {"res": old_res, **old_v}, "NEW": {"res": new_res, **new_v}}
        if load == "MODERATE":
            bw = run(lambda rc: BeamLookahead(rc, width=6, horizon_s=900.0, max_combos=96, generator=GEN),
                     SEEDS_WIDE, RC_NEW, n_ext, n_ves)
            sfw = run(lambda rc: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                      SEEDS_WIDE, RC_NEW, n_ext, n_ves)
            entry["NEW_ceiling"] = {"BEAM_wide": bw, "SF_SPT": sfw,
                                    "headroom": round(bw["total"] - sfw["total"], 2)}
            print(f"  [NEW 천장] BEAM_wide {bw['total']}(swa {bw['swa']}) vs SF_SPT {sfw['total']}"
                  f"(swa {sfw['swa']}) → 헤드룸 {entry['NEW_ceiling']['headroom']:+.2f}")
        out["loads"][load] = entry
    out["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  (elapsed {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
