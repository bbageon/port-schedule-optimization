"""YR-100 [3] — 단일 블록 후보 확정: CALC(RL+계산비용) vs CONTROL(RL) vs JR1800 vs SF.

■ 프로토콜 (결과 미열람 동결)
- 평가대역: 4셀 × 신선 seed 6 (BASE+700..705 — 학습 tr/va/기존 eval 대역과 불겹침).
- arm: SF · JR1800(공개정보 공동 rollout 참고군) · CONTROL_s{3} · CALC_s{3}.
- 지표: 순수 numeraire_v1 total(1차) · berth · 트럭 mean/P95 · 완주 · healthy.
- 판정: guard(완주 1.0·healthy) 통과 arm 중 pooled paired Δtotal(vs SF) 최소가 후보.
  학습 arm 은 시드별 개별 보고(재현성) + 3시드 평균. YR-096 진행/폐기 근거 = JR1800 상대성적.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from .yr090_dense_vessel import BASE, CELLS, _ci, _sim

RC_EVAL = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr100_candidate_eval")
SEEDS = {c: [BASE[c] + 700 + i for i in range(6)] for c in CELLS}


def _episode_row(cell, seed, policy):
    r = run_joint_episode(_sim(cell, seed), policy, RC_EVAL, generator=CandidateGenerator())
    healthy = True
    try:
        assert_healthy_action_mix(r["_mix"], label=f"{cell}/s{seed}")
    except ActionMixError:
        healthy = False
    return {"cell": cell, "seed": seed, "total": r["total_cost"],
            "berth": r["berth_overrun_min"], "wait": r["mean_wait_min"],
            "p95": r["p95_wait_min"], "compl": r["completion_rate"], "healthy": healthy}


def _policy(arm: str):
    if arm == "SF":
        return lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    if arm == "JR1800":
        return lambda: JointRolloutGreedy(RC_EVAL, horizon_s=1800.0,
                                          generator=CandidateGenerator())
    kind, ts = arm.split(":")
    import torch
    from ..integrated.encoding import StateNorm
    from ..integrated.joint_distill import JointPairNet
    if kind == "CALC":
        from .yr100_single_block import OUT as SB_OUT, CalcPolicy
        ck = torch.load(SB_OUT / f"calc_s{ts}" / "rl_net.pt", map_location="cpu")
        net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
        norm = StateNorm(refs=ck["norm_refs"])
        return lambda: CalcPolicy(net, norm)
    if kind == "CONTROL":
        from .yr088_joint_rl import RLPolicy
        ck = torch.load(Path("outputs/reports/yr090_dense_vessel") / f"control_s{ts}" / "rl_net.pt",
                        map_location="cpu")
        net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
        norm = StateNorm(refs=ck["norm_refs"])
        return lambda: RLPolicy(net, norm)
    raise ValueError(arm)


def run_arm(arm: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"arm_{arm.replace(':', '_')}.json"
    if out.exists():
        print(f"skip (cached): {out}")
        return out
    mk = _policy(arm)
    rows = []
    for cell in CELLS:
        for seed in SEEDS[cell]:
            rows.append(_episode_row(cell, seed, mk()))
            print(f"[{arm}] {cell} s{seed} total={rows[-1]['total']:.2f} "
                  f"berth={rows[-1]['berth']:.1f}", flush=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def verdict() -> dict:
    arms = {}
    for p in sorted(OUT.glob("arm_*.json")):
        arms[p.stem[4:]] = json.loads(p.read_text(encoding="utf-8"))
    sf = {(r["cell"], r["seed"]): r for r in arms.get("SF", [])}
    res: dict = {}
    for name, rows in arms.items():
        if name == "SF" or not sf:
            continue
        d_total = [r["total"] - sf[(r["cell"], r["seed"])]["total"] for r in rows]
        d_berth = [r["berth"] - sf[(r["cell"], r["seed"])]["berth"] for r in rows]
        d_p95 = [r["p95"] - sf[(r["cell"], r["seed"])]["p95"] for r in rows]
        res[name] = {
            "d_total_ci": _ci(d_total), "d_berth_ci": _ci(d_berth), "d_p95_ci": _ci(d_p95),
            "total_mean": round(fmean(r["total"] for r in rows), 2),
            "compl_min": min(r["compl"] for r in rows),
            "healthy_all": all(r["healthy"] for r in rows),
            "per_cell_total": {c: round(fmean(r["total"] for r in rows if r["cell"] == c), 2)
                               for c in CELLS}}
    if "SF" in arms:
        res["SF"] = {"total_mean": round(fmean(r["total"] for r in arms["SF"]), 2),
                     "compl_min": min(r["compl"] for r in arms["SF"]),
                     "healthy_all": all(r["healthy"] for r in arms["SF"]),
                     "per_cell_total": {c: round(fmean(r["total"] for r in arms["SF"] if r["cell"] == c), 2)
                                        for c in CELLS}}
    (OUT / "verdict.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="SF | JR1800 | CALC:<seed> | CONTROL:<seed>")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.arm:
        run_arm(a.arm)
    if a.verdict:
        print(json.dumps(verdict(), ensure_ascii=False, indent=1))
    print("DONE")
