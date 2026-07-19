"""YR-072 — 천장 재진단: 문헌보정 v2 프로파일 × 현실 부하 (YR-070 재실행).

가설 검정: "YR-070 의 낮은 천장(헤드룸 6~9%·완벽정보 0)은 한산한 시나리오·
비보정 프로파일 탓" — v2(SNP-ARMG-STD: ARMG 문헌속도·10열6단·gate 210s) ×
문헌 부하(mid 56·high 80, 피크 창 2배)에서 같은 측정을 반복한다.
판정 규약 (실행 전 명시, YR-070 §5 와 동일): 탐색 헤드룸 = best(JR·BEAM)−SF_SPT,
정보 상금 = PERFECT(eta0)−NOISY. 상금이 커지면 학습 트랙(YR-066) 재개 근거,
그대로 작으면 "문제 성질" 확정. swa 동반 보고 (목적 오정렬 감시 — YR-071 축).
seed 신규 대역 710000~ (기존 실험과 격리).
"""
import json
import os
import time
from statistics import fmean

from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.baselines import (
    ResolverPolicy, ServiceFirstSPTPreference, FIFOPreference,
    JointRolloutGreedy, BeamLookahead, run_joint_episode)

OUT = "outputs/reports/ceiling/ceiling_v2_calibrated.json"
PROFILE = build_calibrated_profile()
RC = RewardCalculator.assumed_default()
GEN = CandidateGenerator()
SEEDS = list(range(710000, 710008))        # 8 seeds
SEEDS_WIDE = list(range(710000, 710003))   # 3 seeds (BEAM_wide 비쌈)


def make_sim(seed, level, eta_err):
    params = calibrated_load_params(level, eta_error_s=eta_err)
    return TerminalSimulator(PROFILE, generate_terminal_scenario(PROFILE, seed, params),
                             check_invariants=True)


def run_policy(fac, seeds, level, eta_err):
    rows = [run_joint_episode(make_sim(s, level, eta_err), fac(), RC, generator=GEN)
            for s in seeds]
    ag = lambda k: round(fmean(r[k] for r in rows), 2)
    return {"n": len(rows), "total_cost": ag("total_cost"), "completion": ag("completion_rate"),
            "backlog": ag("backlog"), "mean_wait": ag("mean_wait_min"),
            "p95_wait": ag("p95_wait_min"),
            "swa": round(fmean(r["action_mix"]["serve_when_available"] for r in rows), 3)}


def show(name, r):
    print(f"{name:16s} {r['total_cost']:>8.2f} {r['completion']:>6.2f} {r['backlog']:>6.1f} "
          f"{r['mean_wait']:>6.2f} {r['p95_wait']:>7.2f} {r['swa']:>5.2f}", flush=True)


def main():
    t0 = time.time()
    hdr = f"{'policy':16s} {'cost':>8s} {'compl':>6s} {'bklog':>6s} {'mwait':>6s} {'p95':>7s} {'swa':>5s}"
    out = {"profile": PROFILE.terminal_id, "seeds": SEEDS, "seeds_wide": SEEDS_WIDE,
           "params": "calibrated_load_params(level) — 피크 2배·gate 210s", "families": {}}
    cheap = {
        "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "FIFO": lambda: ResolverPolicy(FIFOPreference(), "FIFO"),
        "JR_600": lambda: JointRolloutGreedy(RC, horizon_s=600.0, generator=GEN),
        "BEAM_w3_600": lambda: BeamLookahead(RC, width=3, horizon_s=600.0, generator=GEN),
    }
    fams = (("MID56_NOISY", "mid", 300.0, list(cheap)),
            ("HIGH80_NOISY", "high", 300.0, list(cheap)),
            ("HIGH80_PERFECT", "high", 0.0, ["SF_SPT", "BEAM_w3_600"]))
    for fam, level, eta, names in fams:
        print(f"\n===== {fam} (v2 {PROFILE.terminal_id}) =====\n{hdr}", flush=True)
        fr = {}
        for name in names:
            fr[name] = run_policy(cheap[name], SEEDS, level, eta)
            show(name, fr[name])
        if fam == "HIGH80_NOISY":
            fr["BEAM_wide_w6_900"] = run_policy(
                lambda: BeamLookahead(RC, width=6, horizon_s=900.0, max_combos=96,
                                      generator=GEN), SEEDS_WIDE, level, eta)
            fr["SF_SPT_wideSeeds"] = run_policy(cheap["SF_SPT"], SEEDS_WIDE, level, eta)
            show("BEAM_wide(3s)", fr["BEAM_wide_w6_900"])
            show("SF_SPT(3s)", fr["SF_SPT_wideSeeds"])
        out["families"][fam] = fr

    mid, hi = out["families"]["MID56_NOISY"], out["families"]["HIGH80_NOISY"]
    perfect = out["families"]["HIGH80_PERFECT"]
    out["headroom"] = {
        "MID56": {"SF_SPT": mid["SF_SPT"]["total_cost"],
                  "best_search_vs_SF": round(min(mid["JR_600"]["total_cost"],
                                                 mid["BEAM_w3_600"]["total_cost"])
                                             - mid["SF_SPT"]["total_cost"], 2)},
        "HIGH80": {"SF_SPT": hi["SF_SPT"]["total_cost"],
                   "best_search_vs_SF": round(min(hi["JR_600"]["total_cost"],
                                                  hi["BEAM_w3_600"]["total_cost"])
                                              - hi["SF_SPT"]["total_cost"], 2),
                   "BEAMwide_vs_SFwide": round(hi["BEAM_wide_w6_900"]["total_cost"]
                                               - hi["SF_SPT_wideSeeds"]["total_cost"], 2),
                   "perfect_info_gain_SF": round(perfect["SF_SPT"]["total_cost"]
                                                 - hi["SF_SPT"]["total_cost"], 2),
                   "perfect_info_gain_BEAM": round(perfect["BEAM_w3_600"]["total_cost"]
                                                   - hi["BEAM_w3_600"]["total_cost"], 2)}}
    out["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n===== v2 헤드룸 =====")
    print(json.dumps(out["headroom"], ensure_ascii=False, indent=2))
    print(f"\nDONE 저장: {OUT}  (elapsed {out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
