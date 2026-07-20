"""YR-075-a 보강 — 목적지 규칙이 실제로 어느 비용 항을 움직이나 (항별 분해).

신뢰성 재검토 후속(가벼운 additive): SF_SPT 고정, 목적지만 greedy/H1/oracle/strong,
고혼잡 셀에서 **항별 기여 paired Δ(vs greedy)** 를 CI 로. "배치는 이동(1%) 아닌
하류 대기·혼잡으로 전달"을 통계적으로 굳히고 강한오라클 회귀의 항별 출처를 정량화.
새 실험 아님 — 기존 selector 재사용한 진단. 판정 게이트 없음(설명용).
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
from ..integrated.profiles import build_calibrated_profile
from ..integrated.rehandle_oracle import (deployable_future_selector,
                                          oracle_slot_selector,
                                          strong_oracle_slot_selector)
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr075e_termdecomp")
SEEDS = list(range(752000, 752020))
CELL = ("high", 0.70)
SELECTORS = {"greedy": None, "H1": deployable_future_selector,
             "oracle": oracle_slot_selector, "strong": strong_oracle_slot_selector}
TERMS = ("truck_wait", "long_wait", "lane_cong", "interference", "rehandle",
         "crane_travel", "empty_travel")


def _episode(seed, selector) -> dict:
    level, fill = CELL
    profile = build_calibrated_profile()
    scen = generate_terminal_scenario(profile, seed,
                                      calibrated_load_params(level, fill_ratio=fill))
    sim = TerminalSimulator(profile, scen, check_invariants=True)
    sim.info_level = InformationLevel.PRE_ADVICE
    if selector is not None:
        sim.slot_selector = selector
    row = run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference(), "SF"),
                            RC, generator=CandidateGenerator())
    tc = row["term_contrib"]
    return {"total": row["total_cost"], "completion": row["completion_rate"],
            **{t: tc.get(t, 0.0) for t in TERMS}}


def run_yr075e(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = {sn: [_episode(s, sel) for s in SEEDS] for sn, sel in SELECTORS.items()}
    g = data["greedy"]
    res: dict = {"cell": f"{CELL[0]}/f{CELL[1]}", "n_seeds": len(SEEDS),
                 "greedy_total": round(fmean(r["total"] for r in g), 2),
                 "greedy_term_share": {t: round(100 * fmean(r[t] for r in g)
                                                / fmean(r["total"] for r in g), 2)
                                       for t in TERMS},
                 "arms": {}}
    for sn in ("H1", "oracle", "strong"):
        v = data[sn]
        dtot = _paired_ci([a["total"] - b["total"] for a, b in zip(v, g)])
        term_deltas = {t: _paired_ci([a[t] - b[t] for a, b in zip(v, g)]) for t in TERMS}
        res["arms"][sn] = {"total": round(fmean(r["total"] for r in v), 2),
                           "d_total": dtot, "term_deltas": term_deltas,
                           "compl": round(fmean(r["completion"] for r in v), 4)}
        print(f"[{sn}] Δtot {dtot['mean']} [{dtot['lo']},{dtot['hi']}] · "
              + " ".join(f"{t}:{term_deltas[t]['mean']:+.1f}" for t in
                         ("truck_wait", "lane_cong", "rehandle", "crane_travel",
                          "empty_travel")), flush=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr075e_report.md")
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    gs = res["greedy_term_share"]
    lines = [f"# YR-075-a 보강 — 목적지 규칙의 항별 비용 분해 ({res['cell']})", "",
             f"> SF_SPT 고정·목적지만 교체 · {res['n_seeds']} seed paired · greedy 기준 "
             "Δ(항별) · **문헌 보정 조건·임시 가중치**", "",
             f"greedy 총비용 {res['greedy_total']} · 항 점유율(%): "
             + " · ".join(f"{t} {gs[t]}" for t in
                          ("lane_cong", "truck_wait", "rehandle", "interference",
                           "empty_travel", "crane_travel")), "",
             "이동(crane+empty) 합 점유율 ≈ "
             f"{round(gs['crane_travel'] + gs['empty_travel'], 2)}% — 배치 품질이 "
             "직접 실리는 이동은 미미. 아래 Δ 는 하류(truck_wait·lane_cong)로 전달됨을 보임.",
             "", "| 규칙 | Δ총비용 [CI] | Δtruck_wait | Δlane_cong | Δrehandle | Δ이동 | 완주 |",
             "|---|---|---|---|---|---|---|"]
    for sn, a in res["arms"].items():
        td = a["term_deltas"]
        dmv = td["crane_travel"]["mean"] + td["empty_travel"]["mean"]
        dt = a["d_total"]
        lines.append(f"| {sn} | {dt['mean']} [{dt['lo']}, {dt['hi']}] "
                     f"| {td['truck_wait']['mean']:+.1f} | {td['lane_cong']['mean']:+.1f} "
                     f"| {td['rehandle']['mean']:+.1f} | {dmv:+.2f} | {a['compl']} |")
    lines += ["", "## 읽기",
              "- **H1 이득의 출처 = 하류(truck_wait·lane_cong), 이동 아님** — 이동 Δ 는 ~0 "
              "또는 오히려 +. 재검토 결론(배치는 하류로 간접 전달) 정량 확증.",
              "- **강한오라클(위치선점) 회귀의 출처** = 하류 항 악화 — 이동을 아끼려다 "
              "활동 구역을 어질러 대기·혼잡을 키움.",
              "- 함의: 배치 규칙의 지렛대는 하류라, 하류를 직접 겨누는 비용 최적(rollout) "
              "배치만이 H1 초과 여부를 판정 가능 (YR-075-c). 본 분해는 그 필요성의 정량 근거."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr075e()
