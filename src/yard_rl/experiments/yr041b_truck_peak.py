"""YR-041b 진단 — 반대 극단: 본선 한가 + 트럭 혼잡에서 λ 가 트럭을 지키나.

041 은 본선 스트레스 셀 위주. 여기선 **본선 한가(느슨 마감·소물량) + 트럭 혼잡(고부하
·피크)** 셀에서 λ 동적 vs 상시우선 vs SF. 기대: 본선이 안 급하니 λ(40분)은 거의 발동
안 해 트럭 = SF, 상시우선(∞)만 트럭 손해. "본선 한가하면 트럭 먼저" 성질 박제.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import ResolverPolicy, run_joint_episode
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr041_lambda_vessel import DeadlineAwareVessel
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr041b_truck_peak")
SEEDS = list(range(759000, 759008))
THRESHOLDS_MIN = [0, 40, 1e9]                  # SF · 동적40 · 상시


def _params():
    # 본선 한가(느슨 마감 2.5·소물량 6) + 트럭 혼잡(112·피크 2배)
    return calibrated_load_params("high", n_external=112, arrival_peak_amp=2.0,
                                  arrival_peak_width_frac=0.2, vessel_moves=6,
                                  vessel_deadline_mult=2.5)


def _row(seed, thr):
    p = build_calibrated_profile()
    s = TerminalSimulator(p, generate_terminal_scenario(p, seed, _params()),
                          check_invariants=True)
    s.info_level = LEVEL
    r = run_joint_episode(s, ResolverPolicy(DeadlineAwareVessel(thr * 60.0), f"λ{thr}"),
                          RC, generator=CandidateGenerator())
    return {"vdelay": round(r["vessel_delay_min"], 2), "wait": round(r["mean_wait_min"], 3),
            "compl": round(r["completion_rate"], 4)}


def _lbl(thr):
    return "SF(0)" if thr == 0 else ("상시(∞)" if thr >= 1e8 else f"{thr}분")


def run_yr041b(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for thr in THRESHOLDS_MIN:
            rows = [_row(s, thr) for s in SEEDS]
            data[thr] = rows
            for s, r in zip(SEEDS, rows):
                f.write(json.dumps({"thr_min": thr, "seed": s, **r}, ensure_ascii=False) + "\n")
            a = {k: round(fmean(r[k] for r in rows), 3) for k in ("vdelay", "wait", "compl")}
            print(f"λ={_lbl(thr):8s} 본선지연={a['vdelay']:>6.2f} 트럭대기={a['wait']:>6.2f} "
                  f"완주={a['compl']:.3f}", flush=True)
    base = data[0]
    res = {"cell": "truck_peak(본선한가+트럭혼잡)", "arms": {}}
    for thr in THRESHOLDS_MIN:
        dw = _paired_ci([a["wait"] - b["wait"] for a, b in zip(data[thr], base)])
        res["arms"][_lbl(thr)] = {
            "vdelay": round(fmean(r["vdelay"] for r in data[thr]), 3),
            "wait": round(fmean(r["wait"] for r in data[thr]), 3),
            "compl": round(fmean(r["compl"] for r in data[thr]), 4),
            "d_wait_vs_SF": dw, "truck_hurt": dw["lo"] > 0.0}
    res["lambda40_keeps_truck"] = not res["arms"]["40분"]["truck_hurt"]
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    lines = ["# YR-041b — 본선 한가 + 트럭 혼잡: λ 가 트럭 지키나 (8 seed)", "",
             f"> 셀 {res['cell']} · 본선 느슨마감(2.5)·소물량(6) + 트럭 112·피크. 실측.", "",
             "| λ | 본선지연 | 트럭대기 | Δ트럭 vs SF [CI] | 트럭 손해 | 완주 |",
             "|---|---|---|---|---|---|"]
    for lbl, a in res["arms"].items():
        d = a["d_wait_vs_SF"]
        lines.append(f"| {lbl} | {a['vdelay']} | {a['wait']} | {d['mean']} "
                     f"[{d['lo']}, {d['hi']}] | {'❌ 손해' if a['truck_hurt'] else '없음'} "
                     f"| {a['compl']} |")
    v = ("✅ λ40 이 트럭을 SF 수준 유지 (본선 한가시 트럭 먼저 확인)"
         if res["lambda40_keeps_truck"] else "λ40 도 트럭 손해 — 재검토")
    lines += ["", f"## 판정: {v}",
              "- 본선 한가(마감 멀어 40분 임계 거의 미발동) → λ40 ≈ SF 트럭. 상시(∞)만 "
              "본선 없어도 우선 시도해 트럭 손해면 동적 설계 정당성 확증.",
              "- 041(본선 급함: 본선 우선) + 041b(본선 한가: 트럭 우선) = λ 의 상황적응 박제."]
    (out / "yr041b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlambda40_keeps_truck={res['lambda40_keeps_truck']}", flush=True)
    print("DONE", flush=True)
    return res


if __name__ == "__main__":
    run_yr041b()
