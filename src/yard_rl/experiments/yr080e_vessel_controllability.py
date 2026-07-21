"""YR-080 진단 — 본선 통제가능성: 본선지연을 얼마나 낮출 수 있나 (추정 오라클 없음).

방법 (전부 실측·계산, 추정 아님):
- **본선 절대우선 정책 실행**(VesselFirstServe): 본선 SERVE 가 열리면 무조건 즉시,
  트럭은 뒤로. 두 크레인 최대 본선 처리량 = 본선지연 **실측 하한**.
- 대조: 규칙(SF)·교사(JR 1800s) — 이미 실측.
- 판정: VesselFirst 가 SF 보다 훨씬 낮추면 통제 여지 큼(설계변경 값)·SF 근처면
  규칙이 near-optimal(변경 무의미)·VesselFirst 도 높으면 구조적(수요 초과·시나리오 과함).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..contract.schema import CandidateKind
from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import busan_scenario_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr080e_controllability")
SEEDS = list(range(757000, 757008))
CELLS = ["vessel_rush", "coincident"]
HZN = 1800.0


class VesselFirstServe(BaselinePreference):
    """본선 SERVE 절대우선 → (그다음) 트럭 SERVE 최단 → 나머지. 본선 최대보호 정책."""

    def rank(self, sim, crane_id, gc) -> tuple:
        is_serve = gc.kind == CandidateKind.SERVE
        is_ves = bool(gc.job_ref is not None and gc.job_ref.is_vessel)
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        tier = 0 if (is_serve and is_ves) else (1 if is_serve else 2)
        return (tier, dur) + super().rank(sim, crane_id, gc)


def _sim(kind, seed):
    p = build_calibrated_profile()
    s = TerminalSimulator(p, generate_terminal_scenario(p, seed, busan_scenario_params(kind)),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def _row(kind, seed, pol_fac):
    sim = _sim(kind, seed)
    nv = sum(1 for j in sim.jobs.values() if j.is_vessel_linked)
    r = run_joint_episode(sim, pol_fac(), RC, generator=CandidateGenerator())
    return {"vdelay": round(r["vessel_delay_min"], 2), "wait": round(r["mean_wait_min"], 3),
            "compl": round(r["completion_rate"], 4), "n_vessel_jobs": nv}


def run_yr080e(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    arms = {
        "SF": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
        "JR": lambda: JointRolloutGreedy(RC, horizon_s=HZN, generator=CandidateGenerator()),
        "VesselFirst": lambda: ResolverPolicy(VesselFirstServe(), "VF"),
    }
    res = {"cells": {}}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for kind in CELLS:
            data = {}
            for name, fac in arms.items():
                rows = [_row(kind, s, fac) for s in SEEDS]
                data[name] = rows
                for s, r in zip(SEEDS, rows):
                    f.write(json.dumps({"cell": kind, "arm": name, "seed": s, **r},
                                       ensure_ascii=False) + "\n")
                a = {k: round(fmean(r[k] for r in rows), 3) for k in
                     ("vdelay", "wait", "compl", "n_vessel_jobs")}
                print(f"[{kind}] {name:12s} 본선지연={a['vdelay']:>6.1f} "
                      f"트럭대기={a['wait']:>6.2f} 완주={a['compl']:.3f}", flush=True)
            sf, vf = data["SF"], data["VesselFirst"]
            d = _paired_ci([a["vdelay"] - b["vdelay"] for a, b in zip(vf, sf)])
            cell = {n: {k: round(fmean(r[k] for r in data[n]), 3)
                        for k in ("vdelay", "wait", "compl", "n_vessel_jobs")} for n in arms}
            cell["d_VF_vs_SF_vdelay"] = d
            # 통제 여지 = VesselFirst 가 SF 대비 본선지연 유의 감소
            cell["controllable"] = d["hi"] < 0.0
            cell["vf_floor_vs_sf_pct"] = (round(100 * cell["VesselFirst"]["vdelay"]
                                                / max(1e-9, cell["SF"]["vdelay"]), 1))
            res["cells"][kind] = cell
            print(f"  → VesselFirst 본선지연 {cell['VesselFirst']['vdelay']} vs SF "
                  f"{cell['SF']['vdelay']} (Δ {d['mean']} [{d['lo']},{d['hi']}]) "
                  f"트럭대가 {cell['VesselFirst']['wait']} vs {cell['SF']['wait']}", flush=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr080e_report.md")
    print("DONE", flush=True)
    return res


def _report(res, path):
    lines = ["# YR-080 진단 — 본선 통제가능성 (실측 하한, 추정 오라클 없음)", "",
             "> 본선 절대우선 정책 실행 = 본선지연 실측 하한 · SF·JR 대조 · 8 seed · "
             "busan 긴장셀. 전부 실행/계산 (프록시 오라클 없음).", "",
             "| 셀 | 정책 | 본선지연 | 트럭대기 | 완주 | 본선작업수 |",
             "|---|---|---|---|---|---|"]
    for kind, c in res["cells"].items():
        for n in ("SF", "JR", "VesselFirst"):
            a = c[n]
            lines.append(f"| {kind} | {n} | {a['vdelay']} | {a['wait']} | {a['compl']} "
                         f"| {a['n_vessel_jobs']} |")
    lines += ["", "## 판정 (셀별)", ""]
    for kind, c in res["cells"].items():
        d = c["d_VF_vs_SF_vdelay"]
        verdict = ("통제 여지 큼 → 본선 가시성 설계변경 값 있음"
                   if c["controllable"] and c["vf_floor_vs_sf_pct"] < 60
                   else "SF 가 본선 하한 근처 → 설계변경 무의미(규칙 수준 유지로 충분)"
                   if c["controllable"] else
                   "VesselFirst 도 SF 못 낮춤 → 구조적(수요 초과) 또는 SF near-optimal")
        lines.append(f"- **{kind}**: VesselFirst 본선지연 {c['VesselFirst']['vdelay']} vs "
                     f"SF {c['SF']['vdelay']} (Δ {d['mean']} [{d['lo']}, {d['hi']}]) · "
                     f"트럭 대가 {c['VesselFirst']['wait']} vs {c['SF']['wait']} → {verdict}")
    lines += ["", "## 함의",
              "- VesselFirst(실측 하한)가 SF 를 **크게** 낮추면: 본선지연은 통제 가능한데 "
              "현 정책이 놓침 → 본선 가시성/예측 설계변경(⑤·λ)이 값. 단 트럭 대가 확인.",
              "- VesselFirst ≈ SF: 규칙이 이미 본선 near-optimal → 설계변경 무의미.",
              "- VesselFirst 도 높음: 구조적(STS 수요 > 크레인 용량) → 어떤 정책도 못 고침, "
              "시나리오가 과수요 (deadline_mult·STS 완화 필요). 추정 없이 실측으로 판별."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr080e()
