"""YR-075-a 0b — 강한 오라클(선제 위치선점)이 H1 을 넘는가 (prereg 2026-07-20 동결).

greedy · H1(채택) · strong_oracle 를 고혼잡 셀에서 paired. strong vs H1 이 유의
개선이면 미측정 헤드룸(위치선점 축) 실재, 아니면 H1 완전 near-optimal 확정.
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
                                          strong_oracle_slot_selector)
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr075c_strong")
MODEL = Path("outputs/reports/yr074_finetune/student_ft.pt")
SEEDS = list(range(750000, 750020))
CELLS = [("high", 0.65), ("high", 0.70)]
SELECTORS = {"greedy": None, "H1": deployable_future_selector,
             "strong": strong_oracle_slot_selector}


def _episode(net, norm, slots, level, fill, seed, pk, selector) -> dict:
    profile = build_calibrated_profile()
    scen = generate_terminal_scenario(profile, seed,
                                      calibrated_load_params(level, fill_ratio=fill))
    sim = TerminalSimulator(profile, scen, check_invariants=True)
    sim.info_level = LEVEL
    if selector is not None:
        sim.slot_selector = selector
    pol = (ResolverPolicy(ServiceFirstSPTPreference(), "SF") if pk == "SF"
           else CentralJointValuePolicy(net, norm, CandidateGenerator(), slots))
    row = run_joint_episode(sim, pol, RC, generator=CandidateGenerator())
    return {"total": row["total_cost"], "rehandles": row["rehandles"],
            "mean_wait": row["mean_wait_min"], "completion": row["completion_rate"]}


def run_yr075c(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    slots = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
    res: dict = {"cells": {}}
    headroom = False
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, fill in CELLS:
            data: dict = {}
            for pk in ("FT", "SF"):
                for sn, sel in SELECTORS.items():
                    rows = [_episode(net, norm, slots, level, fill, s, pk, sel)
                            for s in SEEDS]
                    data[(pk, sn)] = rows
                    for s, r in zip(SEEDS, rows):
                        f.write(json.dumps({"cell": f"{level}/f{fill}", "policy": pk,
                                            "sel": sn, "seed": s, **r},
                                           ensure_ascii=False) + "\n")
            cell: dict = {}
            for pk in ("FT", "SF"):
                h1, strong = data[(pk, "H1")], data[(pk, "strong")]
                d_sh = _paired_ci([a["total"] - b["total"]
                                   for a, b in zip(strong, h1)])
                cell[f"{pk}_strong_vs_H1_total"] = d_sh
                cell[f"{pk}_H1_total"] = round(fmean(r["total"] for r in h1), 2)
                cell[f"{pk}_strong_total"] = round(fmean(r["total"] for r in strong), 2)
                cell[f"{pk}_greedy_total"] = round(fmean(r["total"]
                                                        for r in data[(pk, "greedy")]), 2)
                cell[f"{pk}_compl_ok"] = all(r["completion"] == 1.0
                                             for arm in data.values() for r in arm)
                if pk == "FT" and d_sh["hi"] < 0.0:
                    headroom = True
            res["cells"][f"{level}/f{fill}"] = cell
            fh = cell["FT_strong_vs_H1_total"]
            print(f"[{level}/f{fill}] FT strong vs H1 Δtot {fh['mean']} "
                  f"[{fh['lo']},{fh['hi']}] (greedy {cell['FT_greedy_total']} → "
                  f"H1 {cell['FT_H1_total']} → strong {cell['FT_strong_total']})",
                  flush=True)
    res["HEADROOM_BEYOND_H1"] = headroom
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr075c_report.md")
    print(f"\nHEADROOM_BEYOND_H1={headroom}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    lines = ["# YR-075-a 0b — 강한 오라클(선제 위치선점) vs H1", "",
             "> greedy → H1(채택) → strong_oracle · 20 seed/셀 · 고혼잡 · "
             "**문헌 보정 조건·strong 도 이동항 근사(헤드룸 하한)**", "",
             "| 셀 | 정책 | greedy | H1 | strong | strong−H1 Δ [CI] | H1 넘음 |",
             "|---|---|---|---|---|---|---|"]
    for k, c in res["cells"].items():
        for pk in ("FT", "SF"):
            d = c[f"{pk}_strong_vs_H1_total"]
            beat = "✅" if d["hi"] < 0.0 else "—"
            lines.append(f"| {k} | {pk} | {c[f'{pk}_greedy_total']} | "
                         f"{c[f'{pk}_H1_total']} | {c[f'{pk}_strong_total']} | "
                         f"{d['mean']} [{d['lo']}, {d['hi']}] | {beat} |")
    v = ("미측정 헤드룸 실재 → 풍부한 배치 규칙(비방해물→위치선점→최소이동) 설계 정당화"
         if res["HEADROOM_BEYOND_H1"]
         else "strong 이 H1 못 넘음 → H1 이 풍부한 목적에도 near-optimal 확정·YR-075-a 완전 종료")
    lines += ["", f"**판정: {v}**"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr075c()
