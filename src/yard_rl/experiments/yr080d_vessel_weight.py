"""YR-080 진단 — 본선 가중치 민감도: 얼마를 줘야 교사가 실제로 본선을 지키나.

④a 경고: 프로토타입 가중치(본선 1.0)로는 균형 교사가 총점은 이기나 본선지연이 규칙
보다 높음(보호 약함). 여기선 본선 weight 를 {1,4}로 sweep 해 (a) 교사 본선지연이
규칙 이하로 내려가는 지점 (b) 총점 승리 유지 여부 (c) 총점을 어느 항서 버는지(분해)
측정. YR-080 정식 가중치 정당화의 진단 입력 (정식 계약 동결은 병렬 세션 몫).
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
OUT = Path("outputs/reports/yr080d_vessel_weight")
HZN = 1800.0
FIT_SEEDS = list(range(756500, 756503))
SEEDS = list(range(756000, 756008))
CELLS = ["vessel_rush", "coincident"]
VESSEL_WEIGHTS = [1.0, 4.0]                    # 본선 항 가중 sweep (proto=1.0)
VESSEL_TERMS = ("sts_wait", "vessel_delay", "depart_delay")
RC_OLD = RewardCalculator.assumed_default()


def _sim(kind, seed):
    p = build_calibrated_profile()
    s = TerminalSimulator(p, generate_terminal_scenario(p, seed, busan_scenario_params(kind)),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def _base_weight():
    return {"truck_wait": 1.0, "long_wait": 1.0, "transfer_wait": 1.0,
            "crane_travel": 0.3, "empty_travel": 0.3, "rehandle": 0.3,
            "imbalance": 0.3, "lane_cong": 0.3, "interference": 0.0, "resequence": 0.0}


def _fit_scale():
    rows = [run_joint_episode(_sim("vessel_rush", s),
                              ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                              RC_OLD, generator=CandidateGenerator())["term_contrib"]
            for s in FIT_SEEDS]
    return {t: max(fmean(r.get(t, 0.0) for r in rows) * ASSUMED_SCALE[t],
                   ASSUMED_SCALE[t] * 0.05) for t in COST_TERMS}


def _build(scale, vessel_w):
    w = _base_weight()
    for t in VESSEL_TERMS:
        w[t] = vessel_w
    prov = Provenance(ProvBasis.FITTED_BASELINE, "v2 SF busan_rush", "YR-080d")
    return RewardCalculator(default_assumed_config().with_scale(scale, prov=prov)
                            .with_weight({t: w[t] for t in COST_TERMS}))


def _row(kind, seed, pol_fac, rc):
    r = run_joint_episode(_sim(kind, seed), pol_fac(), rc, generator=CandidateGenerator())
    tc = r["term_contrib"]
    tot = max(1e-9, r["total_cost"])
    return {"total": round(r["total_cost"], 3), "wait": round(r["mean_wait_min"], 3),
            "vdelay": round(r["vessel_delay_min"], 2), "compl": round(r["completion_rate"], 4),
            "vessel_share": round(sum(tc.get(t, 0.0) for t in VESSEL_TERMS) / tot, 3),
            "truck_share": round(tc.get("truck_wait", 0.0) / tot, 3)}


def run_yr080d(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    scale = _fit_scale()
    res = {"vessel_weights": VESSEL_WEIGHTS, "cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for kind in CELLS:
            cell = {}
            sf_rc = _build(scale, 1.0)          # SF 채점은 대표 가중(1.0)으로 고정 비교축
            sf_rows = [_row(kind, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                            sf_rc) for s in SEEDS]
            cell["SF"] = {k: round(fmean(r[k] for r in sf_rows), 3) for k in
                          ("total", "wait", "vdelay", "vessel_share", "compl")}
            for r in sf_rows:
                f.write(json.dumps({"cell": kind, "arm": "SF", **r}, ensure_ascii=False) + "\n")
            for vw in VESSEL_WEIGHTS:
                rc = _build(scale, vw)
                jr_rows = [_row(kind, s,
                               lambda rc=rc: JointRolloutGreedy(rc, horizon_s=HZN,
                                                                generator=CandidateGenerator()),
                               rc) for s in SEEDS]
                for r in jr_rows:
                    f.write(json.dumps({"cell": kind, "arm": f"JR_w{vw}", **r},
                                       ensure_ascii=False) + "\n")
                # 본선지연 vs SF (같은 시나리오 paired) — 가중 무관 raw 분 비교
                dvd = _paired_ci([a["vdelay"] - b["vdelay"] for a, b in zip(jr_rows, sf_rows)])
                cell[f"JR_w{vw}"] = {
                    **{k: round(fmean(r[k] for r in jr_rows), 3) for k in
                       ("total", "wait", "vdelay", "vessel_share", "compl")},
                    "d_vdelay_vs_SF": dvd, "protects_vessel": dvd["hi"] < 0.0}
                print(f"[{kind}] JR_w{vw}: vdelay={cell[f'JR_w{vw}']['vdelay']:>6.1f} "
                      f"(SF {cell['SF']['vdelay']:.1f}) wait={cell[f'JR_w{vw}']['wait']:.2f} "
                      f"vshare={cell[f'JR_w{vw}']['vessel_share']:.2f} "
                      f"본선보호={cell[f'JR_w{vw}']['protects_vessel']}", flush=True)
            res["cells"][kind] = cell
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr080d_report.md")
    print("DONE", flush=True)
    return res


def _report(res, path):
    lines = ["# YR-080 진단 — 본선 가중치 민감도 (얼마 줘야 교사가 본선 지키나)", "",
             "> JR_LIT(1800s) 본선 weight sweep vs SF · busan 긴장 셀 · 8 seed · "
             "**진단 입력(정식 계약은 병렬 YR-080)**", "",
             "| 셀 | arm | 본선지연 | vs SF [CI] | 본선보호 | 트럭대기 | 본선기여 | 완주 |",
             "|---|---|---|---|---|---|---|---|"]
    for kind, c in res["cells"].items():
        lines.append(f"| {kind} | SF | {c['SF']['vdelay']} | — | (기준) "
                     f"| {c['SF']['wait']} | {c['SF']['vessel_share']} | {c['SF']['compl']} |")
        for vw in res["vessel_weights"]:
            a = c[f"JR_w{vw}"]
            d = a["d_vdelay_vs_SF"]
            lines.append(f"| {kind} | JR_w{vw} | {a['vdelay']} | {d['mean']} "
                         f"[{d['lo']}, {d['hi']}] | {'✅' if a['protects_vessel'] else '❌'} "
                         f"| {a['wait']} | {a['vessel_share']} | {a['compl']} |")
    lines += ["", "## 읽기",
              "- **본선보호 ✅** = JR 본선지연이 SF 이하(유의). 프로토타입(w1.0)이 ❌면 "
              "가중치 부족 확증 → 정식은 본선 weight 상향 필요.",
              "- 본선기여(vessel_share) 가 weight↑에도 낮으면 정규화가 본선을 억누르는 것 "
              "→ 정규화 기준(fit 셀 본선지연) 재검토 필요.",
              "- 이 진단은 YR-080 정식 가중치 정당화의 입력 — 최종 동결은 병렬 세션."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr080d()
