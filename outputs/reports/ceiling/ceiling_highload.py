"""천장 진단 — 고부하 (협조 헤드룸이 숨는 곳). 중간부하판(ceiling_diagnosis.py)의 고부하 대조.

고부하 = 2 크레인 고정 + 트럭·본선 대량(경합 강). 협조 이득은 혼잡에서 나타나므로
(YR-054: interference=JR격차 85%), 여기서도 천장≈SF_SPT·완벽정보 무가치면 헤드룸 없음 확정.
"""
import json, time
from statistics import fmean

from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.baselines import (
    ResolverPolicy, ServiceFirstSPTPreference, FIFOPreference,
    JointRolloutGreedy, BeamLookahead, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/ceiling/ceiling_highload.json"
PROFILE = build_integrated_profile()
RC = RewardCalculator.assumed_default()
GEN = CandidateGenerator()
N_EXT = 48          # 중간부하판 20 대비 2.4x (2 크레인 고정 → 경합 강화)
N_VES = 3
SEEDS = list(range(331000, 331008))    # 8 seeds
SEEDS_WIDE = list(range(331000, 331003))  # 3 seeds (BEAM_wide 비쌈)


def make_sim(seed, eta_err):
    params = TerminalGenParams(n_external=N_EXT, n_vessels=N_VES, eta_error_s=eta_err)
    return TerminalSimulator(PROFILE, generate_terminal_scenario(PROFILE, seed, params),
                             check_invariants=True)


def run_policy(fac, seeds, eta_err):
    rows = [run_joint_episode(make_sim(s, eta_err), fac(), RC, generator=GEN) for s in seeds]
    ag = lambda k: round(fmean(r[k] for r in rows), 2)
    return {"n": len(rows), "total_cost": ag("total_cost"), "completion": ag("completion_rate"),
            "backlog": ag("backlog"), "mean_wait": ag("mean_wait_min"),
            "p95_wait": ag("p95_wait_min"),
            "swa": round(fmean(r["action_mix"]["serve_when_available"] for r in rows), 3)}


def show(name, r):
    print(f"{name:16s} {r['total_cost']:>7.2f} {r['completion']:>6.2f} {r['backlog']:>6.1f} "
          f"{r['mean_wait']:>6.2f} {r['p95_wait']:>6.2f} {r['swa']:>5.2f}", flush=True)


def main():
    t0 = time.time()
    hdr = f"{'policy':16s} {'cost':>7s} {'compl':>6s} {'bklog':>6s} {'mwait':>6s} {'p95':>6s} {'swa':>5s}"
    out = {"setup": {"n_external": N_EXT, "n_vessels": N_VES, "seeds": SEEDS,
                     "seeds_wide": SEEDS_WIDE}, "families": {}}
    cheap = {
        "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "FIFO": lambda: ResolverPolicy(FIFOPreference(), "FIFO"),
        "JR_600": lambda: JointRolloutGreedy(RC, horizon_s=600.0, generator=GEN),
        "BEAM_w3_600": lambda: BeamLookahead(RC, width=3, horizon_s=600.0, generator=GEN),
    }
    for fam, eta in (("NOISY_eta300", 300.0), ("PERFECT_eta0", 0.0)):
        print(f"\n===== HIGHLOAD {fam} (n_ext={N_EXT} v={N_VES}) =====\n{hdr}", flush=True)
        fr = {}
        # PERFECT 는 정보가치 재확인용 최소셋만 (SF_SPT·BEAM_w3)
        names = list(cheap) if fam == "NOISY_eta300" else ["SF_SPT", "BEAM_w3_600"]
        for name in names:
            fr[name] = run_policy(cheap[name], SEEDS, eta)
            show(name, fr[name])
        if fam == "NOISY_eta300":
            fr["BEAM_wide_w6_900"] = run_policy(
                lambda: BeamLookahead(RC, width=6, horizon_s=900.0, max_combos=96, generator=GEN),
                SEEDS_WIDE, eta)
            fr["SF_SPT_wideSeeds"] = run_policy(
                lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"), SEEDS_WIDE, eta)
            show("BEAM_wide(3s)", fr["BEAM_wide_w6_900"])
            show("SF_SPT(3s)", fr["SF_SPT_wideSeeds"])
        out["families"][fam] = fr

    noisy, perfect = out["families"]["NOISY_eta300"], out["families"]["PERFECT_eta0"]
    sf = noisy["SF_SPT"]["total_cost"]
    out["headroom"] = {
        "SF_SPT_noisy": sf,
        "best_search_vs_SF": round(min(noisy["JR_600"]["total_cost"],
                                       noisy["BEAM_w3_600"]["total_cost"]) - sf, 2),
        "BEAMwide_vs_SFwide": round(noisy["BEAM_wide_w6_900"]["total_cost"]
                                    - noisy["SF_SPT_wideSeeds"]["total_cost"], 2),
        "perfect_info_gain_SF": round(perfect["SF_SPT"]["total_cost"] - sf, 2),
        "perfect_info_gain_BEAM": round(perfect["BEAM_w3_600"]["total_cost"]
                                        - noisy["BEAM_w3_600"]["total_cost"], 2)}
    out["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n===== 고부하 헤드룸 =====")
    print(json.dumps(out["headroom"], ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT}  (elapsed {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
