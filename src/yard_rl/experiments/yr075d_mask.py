"""YR-075-a 0b-2 — 사용자 제안 "순수 비방해물 mask(딱딱)" vs H1(부드러움) vs greedy.

질문: 방해물 생성을 딱딱하게 금지하고 이동은 비용이 고르게 두면 H1 보다 나은가.
고포화 불가능 케이스에서만 갈림 — 딱딱 mask 가 먼 자리로 밀려 회귀하는지 확인.
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
from ..integrated.rehandle_oracle import deployable_future_selector, mask_only_selector
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr075d_mask")
MODEL = Path("outputs/reports/yr074_finetune/student_ft.pt")
SEEDS = list(range(751000, 751020))
CELLS = [("high", 0.65), ("high", 0.70)]
SELECTORS = {"greedy": None, "H1": deployable_future_selector, "mask": mask_only_selector}


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
    return {"total": row["total_cost"], "completion": row["completion_rate"]}


def run_yr075d(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    slots = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
    res: dict = {"cells": {}}
    mask_beats_h1 = False
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, fill in CELLS:
            data = {}
            for pk in ("FT", "SF"):
                for sn, sel in SELECTORS.items():
                    rows = [_episode(net, norm, slots, level, fill, s, pk, sel)
                            for s in SEEDS]
                    data[(pk, sn)] = rows
                    for s, r in zip(SEEDS, rows):
                        f.write(json.dumps({"cell": f"{level}/f{fill}", "policy": pk,
                                            "sel": sn, "seed": s, **r},
                                           ensure_ascii=False) + "\n")
            cell = {}
            for pk in ("FT", "SF"):
                h1, mask = data[(pk, "H1")], data[(pk, "mask")]
                d = _paired_ci([a["total"] - b["total"] for a, b in zip(mask, h1)])
                cell[f"{pk}_mask_vs_H1"] = d
                for sn in ("greedy", "H1", "mask"):
                    cell[f"{pk}_{sn}_total"] = round(fmean(r["total"]
                                                          for r in data[(pk, sn)]), 2)
                    cell[f"{pk}_{sn}_incompl"] = sum(1 for r in data[(pk, sn)]
                                                     if r["completion"] < 1.0)
                if pk == "FT" and d["hi"] < 0.0:
                    mask_beats_h1 = True
            res["cells"][f"{level}/f{fill}"] = cell
            d = cell["FT_mask_vs_H1"]
            print(f"[{level}/f{fill}] FT mask vs H1 Δ {d['mean']} [{d['lo']},{d['hi']}] "
                  f"(greedy {cell['FT_greedy_total']} H1 {cell['FT_H1_total']} "
                  f"mask {cell['FT_mask_total']} · incompl g/H1/mask "
                  f"{cell['FT_greedy_incompl']}/{cell['FT_H1_incompl']}/"
                  f"{cell['FT_mask_incompl']})", flush=True)
    res["MASK_BEATS_H1"] = mask_beats_h1
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    lines = ["# YR-075-a 0b-2 — 순수 비방해물 mask(딱딱) vs H1(부드러움)", "",
             "> 20 seed/셀 고혼잡 · **문헌 보정 조건**", "",
             "| 셀 | 정책 | greedy | H1 | mask | mask−H1 Δ [CI] | 미완주 g/H1/mask |",
             "|---|---|---|---|---|---|---|"]
    for k, c in res["cells"].items():
        for pk in ("FT", "SF"):
            d = c[f"{pk}_mask_vs_H1"]
            lines.append(f"| {k} | {pk} | {c[f'{pk}_greedy_total']} | "
                         f"{c[f'{pk}_H1_total']} | {c[f'{pk}_mask_total']} | "
                         f"{d['mean']} [{d['lo']}, {d['hi']}] | "
                         f"{c[f'{pk}_greedy_incompl']}/{c[f'{pk}_H1_incompl']}/"
                         f"{c[f'{pk}_mask_incompl']} |")
    v = ("딱딱 mask 가 H1 개선 → 채택 재검토"
         if mask_beats_h1 else "딱딱 mask 가 H1 못 넘음 → 부드러운 H1 확정")
    lines += ["", f"**판정: {v}**"]
    (out / "yr075d_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMASK_BEATS_H1={mask_beats_h1}", flush=True)
    print("DONE", flush=True)
    return res


if __name__ == "__main__":
    run_yr075d()
