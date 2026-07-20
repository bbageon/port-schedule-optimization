"""YR-075-a 1a단계 — 배포형 목적지 규칙(H1)이 오라클 상금을 잡는가.

0단계: 전지적 오라클이 고혼잡(high·fill0.70)에서 총비용 −8%. 여기선 관측정보만 쓰는
H1 이 채택본(FT) 에 얹혀 그 이득을 얼마나 잡는지 측정 — 잡으면 규칙 교체로 끝(RL 불요),
부족하면 1b(K-후보 RL). paired: 같은 seed·같은 정책, 목적지 규칙만 greedy/H1/oracle.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.joint_distill import CentralJointValuePolicy, load_student
from ..integrated.profiles import build_calibrated_profile
from ..integrated.rehandle_oracle import (deployable_future_selector,
                                          oracle_slot_selector)
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr075b_deploy")
MODEL = Path("outputs/reports/yr074_finetune/student_ft.pt")
SEEDS = list(range(749000, 749020))
CELLS = [("high", 0.65), ("high", 0.70)]
SELECTORS = {"greedy": None, "H1": deployable_future_selector,
             "oracle": oracle_slot_selector}


def _episode(net, norm, slots, level, fill, seed, policy_kind, selector) -> dict:
    profile = build_calibrated_profile()
    scen = generate_terminal_scenario(profile, seed,
                                      calibrated_load_params(level, fill_ratio=fill))
    sim = TerminalSimulator(profile, scen, check_invariants=True)
    sim.info_level = LEVEL
    if selector is not None:
        sim.slot_selector = selector
    pol = (ResolverPolicy(ServiceFirstSPTPreference(), "SF") if policy_kind == "SF"
           else CentralJointValuePolicy(net, norm, CandidateGenerator(), slots))
    row = run_joint_episode(sim, pol, RC, generator=CandidateGenerator())
    return {"total": row["total_cost"], "rehandles": row["rehandles"],
            "mean_wait": row["mean_wait_min"], "completion": row["completion_rate"],
            "backlog": row["backlog"]}


def run_yr075b(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    slots = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
    res: dict = {"cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, fill in CELLS:
            data: dict = {}
            for pk in ("FT", "SF"):
                for sel_name, sel in SELECTORS.items():
                    rows = [_episode(net, norm, slots, level, fill, s, pk, sel)
                            for s in SEEDS]
                    data[(pk, sel_name)] = rows
                    for s, r in zip(SEEDS, rows):
                        f.write(json.dumps({"cell": f"{level}/f{fill}", "policy": pk,
                                            "sel": sel_name, "seed": s, **r},
                                           ensure_ascii=False) + "\n")
            cell: dict = {}
            for pk in ("FT", "SF"):
                g = data[(pk, "greedy")]
                for sel_name in ("H1", "oracle"):
                    v = data[(pk, sel_name)]
                    cell[f"{pk}_{sel_name}_vs_greedy_total"] = _paired_ci(
                        [a["total"] - b["total"] for a, b in zip(v, g)])
                    cell[f"{pk}_{sel_name}_vs_greedy_wait"] = _paired_ci(
                        [a["mean_wait"] - b["mean_wait"] for a, b in zip(v, g)])
                cell[f"{pk}_greedy_total"] = round(fmean(r["total"] for r in g), 2)
                cell[f"{pk}_H1_total"] = round(fmean(r["total"]
                                                     for r in data[(pk, "H1")]), 2)
                cell[f"{pk}_oracle_total"] = round(fmean(r["total"]
                                                         for r in data[(pk, "oracle")]), 2)
                cell[f"{pk}_greedy_wait"] = round(fmean(r["mean_wait"] for r in g), 3)
                cell[f"{pk}_H1_wait"] = round(fmean(r["mean_wait"]
                                                    for r in data[(pk, "H1")]), 3)
                cell[f"{pk}_compl_ok"] = all(r["completion"] == 1.0
                                             for arm in data.values() for r in arm)
            res["cells"][f"{level}/f{fill}"] = cell
            ft_h1 = cell["FT_H1_vs_greedy_total"]
            ft_or = cell["FT_oracle_vs_greedy_total"]
            capture = (round(ft_h1["mean"] / ft_or["mean"], 2)
                       if ft_or["mean"] < 0 else None)
            print(f"[{level}/f{fill}] FT H1 Δtot {ft_h1['mean']} [{ft_h1['lo']},"
                  f"{ft_h1['hi']}] · oracle Δtot {ft_or['mean']} · capture={capture}",
                  flush=True)
    # H1 이 채택본에서 유의 개선(양 셀 총비용 CI 상한<0)이면 규칙 교체 권고
    res["H1_helps_FT"] = all(res["cells"][c][f"FT_H1_vs_greedy_total"]["hi"] < 0.0
                             for c in res["cells"])
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr075b_report.md")
    print(f"\nH1_helps_FT={res['H1_helps_FT']}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    lines = ["# YR-075-a 1a단계 — 배포형 목적지 H1 (관측정보만) vs greedy/oracle", "",
             "> 채택본 FT·SF 고정, 목적지 규칙만 교체 · 20 seed/셀 · 고혼잡 · "
             "**문헌 보정 조건**", "",
             "| 셀 | 정책 | greedy 총비용 | H1 총비용 | oracle 총비용 | "
             "H1 Δ [CI] | oracle Δ [CI] | H1 잡음율 |", "|---|---|---|---|---|---|---|---|"]
    for k, c in res["cells"].items():
        for pk in ("FT", "SF"):
            h1, orc = c[f"{pk}_H1_vs_greedy_total"], c[f"{pk}_oracle_vs_greedy_total"]
            cap = (f"{round(100*h1['mean']/orc['mean'])}%" if orc["mean"] < 0 else "—")
            lines.append(f"| {k} | {pk} | {c[f'{pk}_greedy_total']} | "
                         f"{c[f'{pk}_H1_total']} | {c[f'{pk}_oracle_total']} | "
                         f"{h1['mean']} [{h1['lo']}, {h1['hi']}] | "
                         f"{orc['mean']} [{orc['lo']}, {orc['hi']}] | {cap} |")
    lines += ["", f"**H1 이 채택본(FT) 개선: {res['H1_helps_FT']}** — True 면 규칙 교체로 "
              "이득 확보(RL 불요), False 면 1b(K-후보 RL) 필요"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr075b()
