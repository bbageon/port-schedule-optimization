"""YR-041c — λ 긴급도 확장: deadline + STS 굶주림. 041b 의 114분 방치를 잡나.

041b 발견: deadline-slack 단독은 느슨마감+트럭혼잡서 본선 굶주림(114분)을 놓침.
확장: 본선 SERVE 가 (마감 임박) **또는** (그 본선 STS 가 지금 굶는 중, sts_blocked)
이면 우선. 검증: 041b(느슨마감 굶주림) 회수 + 041(빡빡마감) 무회귀. 실측(추정 없음).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..contract.schema import CandidateKind
from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import (busan_scenario_params, calibrated_load_params,
                                       generate_terminal_scenario)
from .yr041_lambda_vessel import DeadlineAwareVessel
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr041c_sts_aware")
SEEDS = list(range(760000, 760008))
THR_MIN = 40.0


class ScheduleBehindVessel(BaselinePreference):
    """본선 SERVE 가 (마감 slack ≤ 임계) OR (계획 뒤처짐: 남은작업시간 > 계획완료까지
    남은시간)이면 절대우선. vessel_delay 는 완료시각 지연이므로 '뒤처짐'이 옳은 신호
    (sts_wait_accum 은 결정시점 15초 미미 — 무효 확인)."""

    def __init__(self, slack_thr_s: float):
        self.thr = slack_thr_s

    def rank(self, sim, crane_id, gc) -> tuple:
        is_serve = gc.kind == CandidateKind.SERVE
        ref = gc.job_ref
        urgent = False
        if is_serve and ref is not None and ref.is_vessel:
            j = sim.jobs.get(ref.job_id)
            if j is not None and j.deadline is not None and (j.deadline - sim.now) <= self.thr:
                urgent = True
            else:
                vid = "-".join(ref.job_id.split("-")[1:-1])
                v = sim.vessels.get(vid)
                if (v is not None and v.started
                        and v.plan.planned_completion_s is not None):
                    if v.remaining_service_time_s() > (v.plan.planned_completion_s - sim.now):
                        urgent = True
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        tier = 0 if urgent else (1 if is_serve else 2)
        return (tier, dur) + super().rank(sim, crane_id, gc)


def _params(cell):
    if cell == "truck_peak":                       # 041b: 느슨마감+트럭혼잡
        return calibrated_load_params("high", n_external=112, arrival_peak_amp=2.0,
                                      arrival_peak_width_frac=0.2, vessel_moves=6,
                                      vessel_deadline_mult=2.5)
    return busan_scenario_params(cell)             # vessel_rush 등


def _row(cell, seed, pref_fac):
    p = build_calibrated_profile()
    s = TerminalSimulator(p, generate_terminal_scenario(p, seed, _params(cell)),
                          check_invariants=True)
    s.info_level = LEVEL
    r = run_joint_episode(s, ResolverPolicy(pref_fac(), "x"), RC,
                          generator=CandidateGenerator())
    return {"vdelay": round(r["vessel_delay_min"], 2), "wait": round(r["mean_wait_min"], 3),
            "compl": round(r["completion_rate"], 4)}


def run_yr041c(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    arms = {
        "SF": lambda: ServiceFirstSPTPreference(),
        "λ40_deadline": lambda: DeadlineAwareVessel(THR_MIN * 60.0),
        "λ40_behind": lambda: ScheduleBehindVessel(THR_MIN * 60.0),
        "λ∞_always": lambda: DeadlineAwareVessel(1e12),   # 통제 상한(항상 우선)
    }
    res = {"cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for cell in ("truck_peak", "vessel_rush"):
            data = {}
            for name, fac in arms.items():
                rows = [_row(cell, s, fac) for s in SEEDS]
                data[name] = rows
                for s, r in zip(SEEDS, rows):
                    f.write(json.dumps({"cell": cell, "arm": name, "seed": s, **r},
                                       ensure_ascii=False) + "\n")
                a = {k: round(fmean(r[k] for r in rows), 3) for k in ("vdelay", "wait", "compl")}
                print(f"[{cell}] {name:14s} 본선지연={a['vdelay']:>6.1f} "
                      f"트럭대기={a['wait']:>6.2f} 완주={a['compl']:.3f}", flush=True)
            sf = data["SF"]
            cell_res = {}
            for name in arms:
                dv = _paired_ci([a["vdelay"] - b["vdelay"] for a, b in zip(data[name], sf)])
                dw = _paired_ci([a["wait"] - b["wait"] for a, b in zip(data[name], sf)])
                cell_res[name] = {"vdelay": round(fmean(r["vdelay"] for r in data[name]), 3),
                                  "wait": round(fmean(r["wait"] for r in data[name]), 3),
                                  "d_vdelay_vs_SF": dv, "d_wait_vs_SF": dw}
            res["cells"][cell] = cell_res
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr041c_report.md")
    print("DONE", flush=True)
    return res


def _report(res, path):
    lines = ["# YR-041c — λ 긴급도 확장 (deadline + STS 굶주림)", "",
             "> 8 seed · SF vs λ40(deadline 단독) vs λ40(deadline+STS). 041b 굶주림 회수 검증.", "",
             "| 셀 | arm | 본선지연 | Δ본선 vs SF [CI] | 트럭대기 | Δ트럭 vs SF [CI] |",
             "|---|---|---|---|---|---|"]
    for cell, c in res["cells"].items():
        for name, a in c.items():
            dv, dw = a["d_vdelay_vs_SF"], a["d_wait_vs_SF"]
            lines.append(f"| {cell} | {name} | {a['vdelay']} | {dv['mean']} "
                         f"[{dv['lo']}, {dv['hi']}] | {a['wait']} | {dw['mean']} "
                         f"[{dw['lo']}, {dw['hi']}] |")
    tp = res["cells"]["truck_peak"]
    vr = res["cells"]["vessel_rush"]
    behind_fix = tp["λ40_behind"]["d_vdelay_vs_SF"]["hi"] < 0.0
    ctrl_ceiling = tp["λ∞_always"]["vdelay"]     # 통제 상한
    lines += ["", "## 판정",
              f"- **통제 상한(λ∞)**: truck_peak 본선지연 SF {tp['SF']['vdelay']} → "
              f"λ∞ {ctrl_ceiling}. 둘이 비슷하면 **구조적(통제 불가)**, λ∞ 가 낮으면 통제 여지.",
              f"- **뒤처짐 신호(λ40_behind)**: {tp['λ40_behind']['vdelay']} "
              f"({'✅ SF 유의 감소' if behind_fix else '❌ 미개선'}) — deadline 단독 "
              f"{tp['λ40_deadline']['vdelay']}.",
              f"- vessel_rush 무회귀 확인: λ40_behind {vr['λ40_behind']['vdelay']} vs "
              f"deadline {vr['λ40_deadline']['vdelay']} vs SF {vr['SF']['vdelay']}.",
              "- 정직: sts_wait_accum(결정시점 ~15초)은 vessel_delay(완료지연)와 무관 — "
              "STS 굶주림 신호 가설 기각. vessel_delay 는 완료시각 지연이라 '뒤처짐'이 옳은 "
              "신호. 단 λ∞ 가 SF 못 낮추면 그 셀은 구조적(release·cadence 바운드)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr041c()
