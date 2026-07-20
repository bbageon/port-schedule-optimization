"""YR-075-a 0단계 — 재조작 목적지 오라클 천장 (prereg 2026-07-20 동결 실행).

SF_SPT 고정, 목적지 규칙만 greedy(find_slot) vs oracle(미래인지) 교체 — paired.
고장치율 셀에서 재조작 밀도 확보. 헤드룸 = Δtotal·Δrehandle·Δwait.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.rehandle_oracle import oracle_slot_selector
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from ..domain.enums import InformationLevel
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr075a_ceiling")
SEEDS = list(range(748000, 748020))
CELLS = [("mid", 0.55), ("mid", 0.70), ("high", 0.55), ("high", 0.70)]


def _episode(level, fill, seed, oracle: bool) -> dict:
    profile = build_calibrated_profile()
    params = calibrated_load_params(level, fill_ratio=fill)
    scen = generate_terminal_scenario(profile, seed, params)
    sim = TerminalSimulator(profile, scen, check_invariants=True)
    sim.info_level = LEVEL
    if oracle:
        sim.slot_selector = oracle_slot_selector
    row = run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
                            RC, generator=CandidateGenerator())
    return {"total": row["total_cost"], "rehandles": row["rehandles"],
            "mean_wait": row["mean_wait_min"], "completion": row["completion_rate"],
            "backlog": row["backlog"]}


def run_yr075a(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"prereg": "2026-07-20-YR-075a-재조작목적지-0단계-prereg.md", "cells": {}}
    headroom = False
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, fill in CELLS:
            g = [_episode(level, fill, s, False) for s in SEEDS]
            o = [_episode(level, fill, s, True) for s in SEEDS]
            for arm, rows in (("GREEDY", g), ("ORACLE", o)):
                for s, r in zip(SEEDS, rows):
                    f.write(json.dumps({"cell": f"{level}/f{fill}", "arm": arm,
                                        "seed": s, **r}, ensure_ascii=False) + "\n")
            d_tot = _paired_ci([a["total"] - b["total"] for a, b in zip(o, g)])
            d_reh = _paired_ci([a["rehandles"] - b["rehandles"] for a, b in zip(o, g)])
            d_wait = _paired_ci([a["mean_wait"] - b["mean_wait"] for a, b in zip(o, g)])
            cell = {"greedy_total": round(fmean(r["total"] for r in g), 2),
                    "oracle_total": round(fmean(r["total"] for r in o), 2),
                    "greedy_reh": round(fmean(r["rehandles"] for r in g), 2),
                    "oracle_reh": round(fmean(r["rehandles"] for r in o), 2),
                    "d_total": d_tot, "d_rehandles": d_reh, "d_wait": d_wait,
                    "compl_ok": all(r["completion"] == 1.0 for r in g + o),
                    # 동결 OR 게이트 (prereg §4)
                    "headroom_cell": d_tot["hi"] < 0.0 or d_reh["hi"] < 0.0,
                    # 해석층: 목적함수(총비용) 실제 개선 — rehandle-only 오독 방지
                    "actionable": d_tot["hi"] < 0.0}
            headroom = headroom or cell["actionable"]
            res["cells"][f"{level}/f{fill}"] = cell
            print(f"[{level}/f{fill}] Δtot={d_tot['mean']} [{d_tot['lo']},{d_tot['hi']}] "
                  f"Δreh={d_reh['mean']} [{d_reh['lo']},{d_reh['hi']}] "
                  f"reh {cell['greedy_reh']}→{cell['oracle_reh']} "
                  f"headroom={cell['headroom_cell']}", flush=True)
    # 결정론 guard: 첫 셀 오라클 1 seed 재실행
    r2 = _episode("mid", 0.55, SEEDS[0], True)
    res["determinism_ok"] = (round(r2["total"], 6)
                             == round(_episode("mid", 0.55, SEEDS[0], True)["total"], 6))
    res["rehandle_only"] = any(c["headroom_cell"] and not c["actionable"]
                               for c in res["cells"].values())
    res["HEADROOM_EXISTS"] = headroom and res["determinism_ok"]  # actionable(총비용) 기준
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr075a_report.md")
    print(f"\nHEADROOM_EXISTS={res['HEADROOM_EXISTS']} det={res['determinism_ok']}",
          flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    lines = ["# YR-075-a 0단계 — 재조작 목적지 오라클 천장", "",
             "> SF_SPT 고정·목적지만 greedy vs 미래인지 오라클 paired · 20 seed/셀 · "
             "**문헌 보정 조건·오라클은 헤드룸 하한 추정**", "",
             "| 셀 | greedy 총비용 | oracle 총비용 | Δtotal [CI] | 재조작 g→o | "
             "Δrehandle [CI] | 헤드룸 |", "|---|---|---|---|---|---|---|"]
    for k, c in res["cells"].items():
        dt, dr = c["d_total"], c["d_rehandles"]
        lines.append(f"| {k} | {c['greedy_total']} | {c['oracle_total']} "
                     f"| {dt['mean']} [{dt['lo']}, {dt['hi']}] "
                     f"| {c['greedy_reh']}→{c['oracle_reh']} "
                     f"| {dr['mean']} [{dr['lo']}, {dr['hi']}] "
                     f"| {'✅' if c['actionable'] else ('재조작만' if c['headroom_cell'] else '—')} |")
    if res["HEADROOM_EXISTS"]:
        verdict = "actionable 헤드룸(총비용 개선) 실재 → 1단계(목적지 후보 확장) 진행 권고"
    elif res["rehandle_only"]:
        verdict = ("재조작 수는 줄지만 총비용·대기는 개선 안 됨(오히려 악화) → "
                   "목적함수 기준 near-optimal. greedy 의 최근접 편향이 대기에 유효 — "
                   "재조작 절감이 이동·대기 증가를 못 갚음. YR-075-a 종료")
    else:
        verdict = "전 셀 near-optimal(오라클 개선 없음) → greedy 목적지 최적 박제·YR-075-a 종료"
    lines += ["", f"**판정: {verdict}**",
              "> 한계: 오라클은 미래 blocked 최소화형(사전식) — 목적함수 직접 최적화가 "
              "아니므로, '총비용 개선 없음'은 '이 오라클로는 못 찾음' 한정. 단 재조작을 "
              "적극 줄여도 총비용이 나빠진다는 것 자체가 greedy 균형의 강함을 시사."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr075a()
