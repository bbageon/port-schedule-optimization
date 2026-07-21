"""YR-041 프로토타입 — λ_vessel 동적: 마감 임박일 때만 본선 우선.

통제가능성 진단(080e): 본선우선(상시)이 본선지연 회수하나 트럭 +1분. 여기선 **마감
임박도**로 조절 — 마감 먼 본선은 트럭 뒤, 임박 본선만 우선. slack 임계 sweep:
0=규칙(SF)·∞=상시우선(VesselFirst)·중간=동적. 목표: 트럭 손해 없이 본선 회수.
진단(YR-041/080 입력). 추정 없음(실측 규칙 실행).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..contract.schema import CandidateKind
from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import ResolverPolicy, run_joint_episode
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import busan_scenario_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr041_lambda_vessel")
SEEDS = list(range(758000, 758008))
CELLS = ["vessel_rush", "coincident"]
THRESHOLDS_MIN = [0, 20, 40, 60, 1e9]          # 0=SF · 1e9=상시우선 · 중간=동적


class DeadlineAwareVessel(BaselinePreference):
    """마감 slack ≤ 임계인 본선 SERVE 만 절대우선(tier0). 그 외는 serve-최단(SF 동형)."""

    def __init__(self, slack_thr_s: float):
        self.thr = slack_thr_s

    def rank(self, sim, crane_id, gc) -> tuple:
        is_serve = gc.kind == CandidateKind.SERVE
        ref = gc.job_ref
        urgent = False
        if is_serve and ref is not None and ref.is_vessel:
            j = sim.jobs.get(ref.job_id)
            if j is not None and j.deadline is not None:
                urgent = (j.deadline - sim.now) <= self.thr
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        tier = 0 if urgent else (1 if is_serve else 2)
        return (tier, dur) + super().rank(sim, crane_id, gc)


def _row(kind, seed, thr):
    p = build_calibrated_profile()
    s = TerminalSimulator(p, generate_terminal_scenario(p, seed, busan_scenario_params(kind)),
                          check_invariants=True)
    s.info_level = LEVEL
    r = run_joint_episode(s, ResolverPolicy(DeadlineAwareVessel(thr * 60.0), f"λ{thr}"),
                          RC, generator=CandidateGenerator())
    return {"vdelay": round(r["vessel_delay_min"], 2), "wait": round(r["mean_wait_min"], 3),
            "compl": round(r["completion_rate"], 4)}


def run_yr041(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    res = {"thresholds_min": THRESHOLDS_MIN, "cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for kind in CELLS:
            data = {}
            for thr in THRESHOLDS_MIN:
                rows = [_row(kind, s, thr) for s in SEEDS]
                data[thr] = rows
                for s, r in zip(SEEDS, rows):
                    f.write(json.dumps({"cell": kind, "thr_min": thr, "seed": s, **r},
                                       ensure_ascii=False) + "\n")
                a = {k: round(fmean(r[k] for r in rows), 3) for k in ("vdelay", "wait", "compl")}
                lbl = "SF(0)" if thr == 0 else ("상시(∞)" if thr >= 1e8 else f"{thr}분")
                print(f"[{kind}] λ={lbl:8s} 본선지연={a['vdelay']:>6.1f} "
                      f"트럭대기={a['wait']:>6.2f} 완주={a['compl']:.3f}", flush=True)
            base = data[0]                          # SF (thr 0) = 트럭 기준
            cell = {}
            for thr in THRESHOLDS_MIN:
                d_w = _paired_ci([a["wait"] - b["wait"] for a, b in zip(data[thr], base)])
                d_v = _paired_ci([a["vdelay"] - b["vdelay"] for a, b in zip(data[thr], base)])
                cell[thr] = {"vdelay": round(fmean(r["vdelay"] for r in data[thr]), 3),
                             "wait": round(fmean(r["wait"] for r in data[thr]), 3),
                             "compl": round(fmean(r["compl"] for r in data[thr]), 4),
                             "d_wait_vs_SF": d_w, "d_vdelay_vs_SF": d_v}
            res["cells"][kind] = cell
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr041_report.md")
    print("DONE", flush=True)
    return res


def _lbl(thr):
    return "SF(0)" if thr == 0 else ("상시(∞)" if thr >= 1e8 else f"{thr}분")


def _report(res, path):
    lines = ["# YR-041 프로토타입 — λ_vessel 동적 (마감 임박만 본선 우선)", "",
             "> slack 임계 sweep · 8 seed · busan 긴장셀 · 기준=SF(임계0). 목표: 트럭 손해 "
             "최소로 본선지연 회수. 실측(추정 없음).", "",
             "| 셀 | λ임계 | 본선지연 | Δ본선 vs SF | 트럭대기 | Δ트럭 vs SF [CI] | 완주 |",
             "|---|---|---|---|---|---|---|"]
    for kind, c in res["cells"].items():
        for thr in res["thresholds_min"]:
            a = c[thr]
            dv, dw = a["d_vdelay_vs_SF"], a["d_wait_vs_SF"]
            lines.append(f"| {kind} | {_lbl(thr)} | {a['vdelay']} | {dv['mean']} "
                         f"| {a['wait']} | {dw['mean']} [{dw['lo']}, {dw['hi']}] | {a['compl']} |")
    lines += ["", "## 읽기",
              "- **좋은 λ임계** = 본선지연이 상시우선(∞)에 근접하면서 트럭대기 Δ(vs SF)가 "
              "유의하지 않은 지점. 그 지점이 있으면 '트럭 손해 없이 본선 회수' 성립.",
              "- 없으면(전 임계서 본선↓ = 트럭↑ 동반) → 트레이드오프 불가피 → λ 로 "
              "운영자가 균형점 선택. 어느 쪽이든 실측으로 곡선 제시.",
              "- 본선 우선을 규칙/제약 층으로 넣는 설계(080e 결론)의 트럭 대가 정량 — YR-080/041 입력."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr041()
