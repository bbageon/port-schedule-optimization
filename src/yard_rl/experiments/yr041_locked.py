"""YR-041 — 잠금 평가 실행기 (prereg: outputs/reports/yr041_locked/prereg.md).

동결 후보 vs SF paired — 신규 셀(마감강도 0.40/0.75)·신규 seed(+900 대역). 재선택 0.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr090_dense_vessel import _ci

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr041_locked")
CELLS41 = {"high-0.40": ("high", 0.40), "high-0.75": ("high", 0.75), "mid-0.40": ("mid", 0.40)}
BASE41 = {"high-0.40": 830_100, "high-0.75": 830_100, "mid-0.40": 830_000}
N_SEEDS = 8


def _sim41(cell: str, seed: int):
    prof = build_calibrated_profile()
    lvl, dm = CELLS41[cell]
    p = dataclasses.replace(calibrated_load_params(lvl, vessel_deadline_mult=dm),
                            time_contract_v2=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, p),
                          check_invariants=True)
    s.info_level = InformationLevel.PRE_ADVICE
    return s


def _mk_policy(arm: str):
    if arm == "SF":
        return lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    if arm == "JR1800":
        return lambda: JointRolloutGreedy(RC, horizon_s=1800.0, generator=CandidateGenerator())
    kind, ts = arm.split(":")
    import torch
    from ..integrated.encoding import StateNorm
    from ..integrated.joint_distill import JointPairNet
    if kind == "CALC":
        from .yr100_single_block import OUT as SB_OUT, CalcPolicy
        ck = torch.load(SB_OUT / f"calc_s{ts}" / "rl_net.pt", map_location="cpu")
        net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
        return lambda: CalcPolicy(net, StateNorm(refs=ck["norm_refs"]))
    if kind == "CONTROL":
        from .yr088_joint_rl import RLPolicy
        ck = torch.load(Path("outputs/reports/yr090_dense_vessel") / f"control_s{ts}" / "rl_net.pt",
                        map_location="cpu")
        net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
        return lambda: RLPolicy(net, StateNorm(refs=ck["norm_refs"]))
    raise ValueError(arm)


def _row(cell, seed, policy):
    r = run_joint_episode(_sim41(cell, seed), policy, RC, generator=CandidateGenerator())
    healthy = True
    try:
        assert_healthy_action_mix(r["_mix"], label=f"{cell}/s{seed}")
    except ActionMixError:
        healthy = False
    return {"cell": cell, "seed": seed, "total": r["total_cost"],
            "berth": r["berth_overrun_min"], "wait": r["mean_wait_min"],
            "p95": r["p95_wait_min"], "compl": r["completion_rate"], "healthy": healthy}


def run(candidate: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    mk_c, mk_s = _mk_policy(candidate), _mk_policy("SF")
    rows_c, rows_s = [], []
    for cell in CELLS41:
        for i in range(N_SEEDS):
            seed = BASE41[cell] + 900 + i
            rows_c.append(_row(cell, seed, mk_c()))
            rows_s.append(_row(cell, seed, mk_s()))
            print(f"[{cell} s{seed}] cand={rows_c[-1]['total']:.2f} sf={rows_s[-1]['total']:.2f}",
                  flush=True)
    db = [c["berth"] - s["berth"] for c, s in zip(rows_c, rows_s)]
    dp95 = [c["p95"] - s["p95"] for c, s in zip(rows_c, rows_s)]
    dt = [c["total"] - s["total"] for c, s in zip(rows_c, rows_s)]
    g_compl = min(r["compl"] for r in rows_c)
    g_healthy = all(r["healthy"] for r in rows_c)
    b_ci, p_ci, t_ci = _ci(db), _ci(dp95), _ci(dt)
    guard_ok = g_compl >= 1.0 and g_healthy
    berth_v = ("우월" if b_ci[2] < 0 else "비열등" if b_ci[2] < 5.0 else "실패")
    p95_ok = p_ci[2] < 3.0
    res = {"candidate": candidate, "d_berth_ci": b_ci, "d_p95_ci": p_ci, "d_total_ci": t_ci,
           "guard": {"compl_min": g_compl, "healthy_all": g_healthy},
           "per_cell_d_berth": {c: round(fmean(x["berth"] - y["berth"]
                                for x, y in zip(rows_c, rows_s) if x["cell"] == c), 2)
                                for c in CELLS41},
           "verdict": {"guard": guard_ok, "berth": berth_v, "p95_noninferior": p95_ok,
                       "adopt": bool(guard_ok and berth_v != "실패" and p95_ok)},
           "rows_candidate": rows_c, "rows_sf": rows_s}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\nVERDICT: guard={guard_ok} berth={berth_v} ({b_ci}) p95<{3.0}={p95_ok} "
          f"({p_ci}) total={t_ci} adopt={res['verdict']['adopt']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args()
    run(a.candidate)
    print("DONE")
