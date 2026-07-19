"""천장(오라클 상한) 진단 — 통합 시나리오에 RL 헤드룸이 있는가.

학습 없음. 반칙 2종으로 상한을 재고 SF_SPT(현 최강 휴리스틱)와 비교:
 (A) 정보 반칙: eta_error_s=0 (완벽 ETA — provided_eta=실제도착)
 (B) 계산 반칙: BEAM 넓은 탐색(width·horizon 확대)
핵심: 총비용은 게이밍 가능하므로 건강도(swa)·완료율·대기 동반 판정.
 - 어떤 반칙 정책도 SF_SPT 를 (건강하게) 못 이기면 → 헤드룸 없음(문제/목적 문제)
 - 이기되 불건강(저 swa)하면 → 목적함수 깨짐 (지표 게이밍)
 - 건강하게 이기면 → 진짜 헤드룸 → 학습/신용 문제 정당
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

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/ceiling/ceiling.json"
PROFILE = build_integrated_profile()
RC = RewardCalculator.assumed_default()
GEN = CandidateGenerator()
N_EXT = 20
SEEDS_CHEAP = list(range(330000, 330010))   # 10 seeds (SF_SPT·FIFO·JR·BEAM)
SEEDS_WIDE = list(range(330000, 330004))    # 4 seeds (BEAM_wide — 비쌈)


def make_sim(seed, eta_err):
    params = TerminalGenParams(n_external=N_EXT, n_vessels=2, eta_error_s=eta_err)
    return TerminalSimulator(PROFILE, generate_terminal_scenario(PROFILE, seed, params),
                             check_invariants=True)


def run_policy(policy_factory, seeds, eta_err):
    rows = []
    for s in seeds:
        r = run_joint_episode(make_sim(s, eta_err), policy_factory(), RC, generator=GEN)
        rows.append(r)
    agg = lambda k: round(fmean(r[k] for r in rows), 2)
    swa = round(fmean(r["action_mix"]["serve_when_available"] for r in rows), 3)
    return {"n": len(rows), "total_cost": agg("total_cost"),
            "completion": agg("completion_rate"), "mean_wait": agg("mean_wait_min"),
            "p95_wait": agg("p95_wait_min"), "swa": swa,
            "vessel_delay": agg("vessel_delay_min")}


def policies():
    return {
        "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "FIFO": lambda: ResolverPolicy(FIFOPreference(), "FIFO"),
        "JR_600": lambda: JointRolloutGreedy(RC, horizon_s=600.0, generator=GEN),
        "BEAM_w3_600": lambda: BeamLookahead(RC, width=3, horizon_s=600.0, generator=GEN),
    }


def main():
    t0 = time.time()
    out = {"setup": {"n_external": N_EXT, "seeds_cheap": SEEDS_CHEAP,
                     "seeds_wide": SEEDS_WIDE, "rc": "assumed_default"},
           "families": {}}
    hdr = f"{'policy':14s} {'cost':>7s} {'compl':>6s} {'mwait':>6s} {'p95':>6s} {'swa':>5s}"
    for fam, eta in (("NOISY_eta300", 300.0), ("PERFECT_eta0", 0.0)):
        print(f"\n===== {fam} (eta_error={eta}) =====\n{hdr}", flush=True)
        fam_res = {}
        for name, fac in policies().items():
            res = run_policy(fac, SEEDS_CHEAP, eta)
            fam_res[name] = res
            print(f"{name:14s} {res['total_cost']:>7.2f} {res['completion']:>6.2f} "
                  f"{res['mean_wait']:>6.2f} {res['p95_wait']:>6.2f} {res['swa']:>5.2f}", flush=True)
        # 계산 반칙: BEAM_wide (비쌈 → 적은 seed)
        bw = run_policy(lambda: BeamLookahead(RC, width=6, horizon_s=900.0,
                                              max_combos=96, generator=GEN),
                        SEEDS_WIDE, eta)
        sf_wide = run_policy(lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
                             SEEDS_WIDE, eta)   # 같은 4 seed SF_SPT (공정 비교)
        fam_res["BEAM_wide_w6_900"] = bw
        fam_res["SF_SPT_wideSeeds"] = sf_wide
        print(f"{'BEAM_wide(4s)':14s} {bw['total_cost']:>7.2f} {bw['completion']:>6.2f} "
              f"{bw['mean_wait']:>6.2f} {bw['p95_wait']:>6.2f} {bw['swa']:>5.2f}", flush=True)
        print(f"{'SF_SPT(4s)':14s} {sf_wide['total_cost']:>7.2f} {sf_wide['completion']:>6.2f} "
              f"{sf_wide['mean_wait']:>6.2f} {sf_wide['p95_wait']:>6.2f} {sf_wide['swa']:>5.2f}", flush=True)
        out["families"][fam] = fam_res

    # 헤드룸 요약
    noisy, perfect = out["families"]["NOISY_eta300"], out["families"]["PERFECT_eta0"]
    sf_noisy = noisy["SF_SPT"]["total_cost"]
    out["headroom"] = {
        "SF_SPT_noisy": sf_noisy,
        "best_search_noisy_vs_SF": round(min(noisy["JR_600"]["total_cost"],
                                             noisy["BEAM_w3_600"]["total_cost"]) - sf_noisy, 2),
        "perfect_info_gain_SF": round(perfect["SF_SPT"]["total_cost"] - sf_noisy, 2),
        "BEAMwide_perfect_vs_SFwide_perfect": round(
            perfect["BEAM_wide_w6_900"]["total_cost"] - perfect["SF_SPT_wideSeeds"]["total_cost"], 2),
        "note": "음수 = 천장이 SF_SPT 보다 낮음(=개선여지). swa 동반 확인 필수."}
    out["elapsed_s"] = round(time.time() - t0, 1)

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n===== 헤드룸 요약 =====")
    print(json.dumps(out["headroom"], ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT}  (elapsed {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
