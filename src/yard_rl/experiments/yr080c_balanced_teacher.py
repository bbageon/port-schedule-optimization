"""YR-080 후보 B (2부/④a) — 균형 목적 교사가 realistic 시나리오서 본선 지키며 이기나.

FT 재판정(1부): 채택본 FT 는 균형 목적서 규칙 못 이김(본선 희생). 여기선 **균형 목적
교사(JR_LIT, 1800초)** 가 본선/트럭 긴장 시나리오(busan_scenario_params)서 본선을
지키며 규칙(SF)을 이기는지 = 증류할 가치(상한)가 있는지 판정. torch 불필요(SF·JR).
④b 전체 재학습(교사→증류→FT)은 YR-080 정식 가중치 후.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..contract.schema import COST_TERMS
from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost import ASSUMED_SCALE
from ..integrated.cost_config import (Provenance, ProvBasis, RewardCalculator,
                                      default_assumed_config)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import busan_scenario_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
OUT = Path("outputs/reports/yr080c_balanced_teacher")
HZN = 1800.0                                   # 채택 교사 창 (YR-078)
FIT_SEEDS = list(range(755500, 755503))
SEEDS = list(range(755000, 755008))
CELLS = ["vessel_rush", "coincident"]
LIT_W = {"truck_wait": 1.0, "long_wait": 1.0, "transfer_wait": 1.0,
         "sts_wait": 1.0, "vessel_delay": 1.0, "depart_delay": 1.0,
         "crane_travel": 0.3, "empty_travel": 0.3, "rehandle": 0.3,
         "imbalance": 0.3, "lane_cong": 0.3, "interference": 0.0, "resequence": 0.0}
RC_OLD = RewardCalculator.assumed_default()


def _sim(kind, seed):
    p = build_calibrated_profile()
    sc = generate_terminal_scenario(p, seed, busan_scenario_params(kind))
    s = TerminalSimulator(p, sc, check_invariants=True)
    s.info_level = LEVEL
    return s


def _build_lit():
    rows = [run_joint_episode(_sim("vessel_rush", s),
                              ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                              RC_OLD, generator=CandidateGenerator())["term_contrib"]
            for s in FIT_SEEDS]
    scale = {t: max(fmean(r.get(t, 0.0) for r in rows) * ASSUMED_SCALE[t],
                    ASSUMED_SCALE[t] * 0.05) for t in COST_TERMS}
    prov = Provenance(ProvBasis.FITTED_BASELINE, "v2 SF busan_rush", "YR-080c")
    return RewardCalculator(default_assumed_config().with_scale(scale, prov=prov)
                            .with_weight({t: LIT_W[t] for t in COST_TERMS}))


def _row(kind, seed, pol_fac, rc):
    r = run_joint_episode(_sim(kind, seed), pol_fac(), rc, generator=CandidateGenerator())
    return {"lit_total": round(r["total_cost"], 2), "wait": round(r["mean_wait_min"], 3),
            "vdelay": round(r["vessel_delay_min"], 2), "compl": round(r["completion_rate"], 4)}


def run_yr080c(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    RC_LIT = _build_lit()
    arms = {
        "SF": (lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"), RC_LIT),
        "JR_OLD": (lambda: JointRolloutGreedy(RC_OLD, horizon_s=HZN, generator=CandidateGenerator()), RC_LIT),
        "JR_LIT": (lambda: JointRolloutGreedy(RC_LIT, horizon_s=HZN, generator=CandidateGenerator()), RC_LIT),
    }
    res = {"horizon_s": HZN, "cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for kind in CELLS:
            data = {}
            for name, (fac, rc) in arms.items():
                rows = [_row(kind, s, fac, rc) for s in SEEDS]
                data[name] = rows
                for s, r in zip(SEEDS, rows):
                    f.write(json.dumps({"cell": kind, "arm": name, "seed": s, **r},
                                       ensure_ascii=False) + "\n")
                a = {k: round(fmean(r[k] for r in rows), 3) for k in
                     ("lit_total", "wait", "vdelay", "compl")}
                print(f"[{kind}] {name:7s} LITtot={a['lit_total']:>7.2f} "
                      f"wait={a['wait']:>6.2f} vdelay={a['vdelay']:>6.1f} compl={a['compl']:.3f}",
                      flush=True)
            sf, jl = data["SF"], data["JR_LIT"]
            d = _paired_ci([a["lit_total"] - b["lit_total"] for a, b in zip(jl, sf)])
            res["cells"][kind] = {
                "arms": {n: {k: round(fmean(r[k] for r in data[n]), 3)
                             for k in ("lit_total", "wait", "vdelay", "compl")}
                         for n in arms},
                "d_JRLIT_vs_SF": d, "jrlit_beats_sf": d["hi"] < 0.0}
            print(f"  → JR_LIT vs SF (LIT총비용) Δ {d['mean']} [{d['lo']},{d['hi']}] "
                  f"{'교사 우위' if d['hi']<0 else '규칙 우위/동급'}", flush=True)
    res["JRLIT_BEATS_SF"] = all(res["cells"][k]["jrlit_beats_sf"] for k in CELLS)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr080c_report.md")
    print(f"\nJRLIT_BEATS_SF={res['JRLIT_BEATS_SF']}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res, path):
    lines = ["# YR-080 ④a — 균형 목적 교사(JR_LIT 1800s) vs 규칙, realistic 본선시나리오", "",
             "> LIT 목적 채점·busan 본선/트럭 긴장 셀·8 seed · **진단(비확정)**", "",
             "| 셀 | arm | LIT총비용 | 트럭대기 | 본선지연 | 완주 |", "|---|---|---|---|---|---|"]
    for kind, c in res["cells"].items():
        for n, a in c["arms"].items():
            lines.append(f"| {kind} | {n} | {a['lit_total']} | {a['wait']} "
                         f"| {a['vdelay']} | {a['compl']} |")
    lines += ["", "## 판정", ""]
    for kind, c in res["cells"].items():
        d = c["d_JRLIT_vs_SF"]
        lines.append(f"- **{kind}**: JR_LIT vs SF Δ {d['mean']} [{d['lo']}, {d['hi']}] "
                     f"{'✅ 교사 우위' if c['jrlit_beats_sf'] else '— 규칙 우위/동급'}")
    lines += ["", f"**균형 교사가 규칙 초과: {res['JRLIT_BEATS_SF']}** — True 면 증류 가치 "
              "확인(④b 재학습 정당화), False 면 균형 목적선 규칙이 강함(재학습해도 상한 낮음)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr080c()
