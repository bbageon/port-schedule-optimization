"""YR-071 ①진단 — 게이밍 지렛대 특정.

건강 정책(SF_SPT, 고 swa)과 저-swa 저비용 정책(JR/BEAM)의 항목별 비용기여(term_contrib)를
비교해, 저비용 정책이 *어느 항을 깎아* 총비용을 낮추는지 분해. 그 항 = 목적 오정렬의 지렛대.
"""
import json, time
from statistics import fmean

from yard_rl.contract.schema import COST_TERMS
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.integrated.baselines import (
    ResolverPolicy, ServiceFirstSPTPreference, FIFOPreference,
    JointRolloutGreedy, BeamLookahead, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/ceiling/term_diagnosis.json"
PROFILE = build_integrated_profile()
RC = RewardCalculator.assumed_default()
GEN = CandidateGenerator()
SEEDS = list(range(330000, 330010))     # 10 seeds (천장 진단과 동일 대역)


def make_sim(seed, n_ext, n_ves):
    return TerminalSimulator(
        PROFILE, generate_terminal_scenario(
            PROFILE, seed, TerminalGenParams(n_external=n_ext, n_vessels=n_ves, eta_error_s=300.0)),
        check_invariants=True)


def run_policy(fac, n_ext, n_ves):
    rows = [run_joint_episode(make_sim(s, n_ext, n_ves), fac(), RC, generator=GEN) for s in SEEDS]
    terms = {t: round(fmean(r["term_contrib"].get(t, 0.0) for r in rows), 3) for t in COST_TERMS}
    return {"total": round(fmean(r["total_cost"] for r in rows), 2),
            "swa": round(fmean(r["action_mix"]["serve_when_available"] for r in rows), 3),
            "mean_wait": round(fmean(r["mean_wait_min"] for r in rows), 2),
            "p95_wait": round(fmean(r["p95_wait_min"] for r in rows), 2),
            "terms": terms}


def diagnose(n_ext, n_ves, label):
    pols = {
        "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "FIFO": lambda: ResolverPolicy(FIFOPreference(), "FIFO"),
        "JR_600": lambda: JointRolloutGreedy(RC, horizon_s=600.0, generator=GEN),
        "BEAM_w3": lambda: BeamLookahead(RC, width=3, horizon_s=600.0, generator=GEN),
    }
    res = {name: run_policy(fac, n_ext, n_ves) for name, fac in pols.items()}
    sf, bm = res["SF_SPT"], res["BEAM_w3"]
    gap = round(sf["total"] - bm["total"], 2)      # >0 = BEAM 이 SF 보다 쌈
    # 항별 (SF기여 − BEAM기여): 양수 = BEAM 이 그 항을 깎아 이득 (게이밍 지렛대 후보)
    lever = sorted(((t, round(sf["terms"][t] - bm["terms"][t], 3)) for t in COST_TERMS),
                   key=lambda x: -x[1])
    print(f"\n===== {label} (n_ext={n_ext} v={n_ves}) =====")
    print(f"{'policy':8s} {'total':>7s} {'swa':>5s} {'mwait':>6s} {'p95':>6s}")
    for name in ("SF_SPT", "FIFO", "JR_600", "BEAM_w3"):
        r = res[name]
        print(f"{name:8s} {r['total']:>7.2f} {r['swa']:>5.2f} {r['mean_wait']:>6.2f} {r['p95_wait']:>6.2f}")
    print(f"\nBEAM 이 SF_SPT 보다 총 {gap:+.2f} 저렴 (swa {sf['swa']}→{bm['swa']}). "
          f"이 격차를 항별로 분해 (양수=BEAM 이 그 항을 깎음):")
    print(f"{'term':16s} {'SF기여':>8s} {'BEAM기여':>9s} {'Δ(SF−BEAM)':>11s} {'격차비중%':>9s}")
    for t, d in lever:
        share = round(100 * d / gap, 1) if gap != 0 else 0.0
        print(f"{t:16s} {sf['terms'][t]:>8.3f} {bm['terms'][t]:>9.3f} {d:>11.3f} {share:>9.1f}")
    return {"label": label, "n_ext": n_ext, "res": res, "gap": gap, "lever": lever}


def main():
    t0 = time.time()
    out = {"seeds": SEEDS, "families": []}
    out["families"].append(diagnose(20, 2, "MODERATE"))
    out["families"].append(diagnose(48, 3, "HIGHLOAD"))
    out["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  (elapsed {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
