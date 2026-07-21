"""채택 정책 재검증 — "규칙 초과"가 우연(seed 특이)인가, 강건한가.

원 test(YR-073-b, 20 seed 744k): FT+H1 이 SF 대비 대기 mid −1.69·high −2.61 유의.
여기선 **한 번도 안 쓴 신선 대역(770k)·수준별 40 seed**로 재실행 — 효과가 재현되면
우연 아님. 채택 구성 그대로(FT + H1 목적지). paired CI·결정론 검사. torch(WSL).
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
from ..integrated.joint_distill import (CentralJointValuePolicy,
                                        adopted_slot_selector, load_student)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr073_reverify")
MODEL = Path("outputs/reports/yr074_finetune/student_ft.pt")
SEEDS = {"mid": list(range(770000, 770040)), "high": list(range(770100, 770140))}


def _sim(level, seed, h1):
    p = build_calibrated_profile()
    s = TerminalSimulator(p, generate_terminal_scenario(p, seed, calibrated_load_params(level)),
                          check_invariants=True)
    s.info_level = LEVEL
    if h1:
        s.slot_selector = adopted_slot_selector()      # 채택 구성 = FT + H1
    return s


def _episode(net, norm, slots, level, seed, arm):
    sim = _sim(level, seed, h1=(arm == "FT"))
    pol = (ResolverPolicy(ServiceFirstSPTPreference(), "SF") if arm == "SF"
           else CentralJointValuePolicy(net, norm, CandidateGenerator(), slots))
    r = run_joint_episode(sim, pol, RC, generator=CandidateGenerator())
    return {"seed": seed, "level": level, "arm": arm,
            "mean_wait": round(r["mean_wait_min"], 4), "p95_wait": round(r["p95_wait_min"], 4),
            "completion": r["completion_rate"], "backlog": r["backlog"]}


def run_reverify(out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    slots = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
    rows = []
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, seeds in SEEDS.items():
            for arm in ("SF", "FT"):
                for s in seeds:
                    r = _episode(net, norm, slots, level, s, arm)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                sel = [x for x in rows if x["level"] == level and x["arm"] == arm]
                print(f"[{level}/{arm}] wait={fmean(x['mean_wait'] for x in sel):.3f} "
                      f"p95={fmean(x['p95_wait'] for x in sel):.2f} "
                      f"compl={fmean(x['completion'] for x in sel):.3f}", flush=True)
    # 결정론: 수준별 FT 선두 2 seed 재실행 동일
    det_ok = True
    for level in SEEDS:
        for s in SEEDS[level][:2]:
            r2 = _episode(net, norm, slots, level, s, "FT")
            r1 = next(r for r in rows if r["level"] == level and r["arm"] == "FT"
                      and r["seed"] == s)
            if any(r1[k] != r2[k] for k in ("mean_wait", "p95_wait")):
                det_ok = False
    res = {"n_seeds_per_cell": {k: len(v) for k, v in SEEDS.items()},
           "seed_band": "770k (신선·미사용)", "determinism_ok": det_ok, "levels": {}}
    for level in SEEDS:
        by = lambda a: sorted((r for r in rows if r["level"] == level and r["arm"] == a),
                              key=lambda r: r["seed"])
        sf, ft = by("SF"), by("FT")
        dw = _paired_ci([a["mean_wait"] - b["mean_wait"] for a, b in zip(ft, sf)])
        dp = _paired_ci([a["p95_wait"] - b["p95_wait"] for a, b in zip(ft, sf)])
        res["levels"][level] = {
            "sf_wait": round(fmean(r["mean_wait"] for r in sf), 3),
            "ft_wait": round(fmean(r["mean_wait"] for r in ft), 3),
            "d_wait": dw, "d_p95": dp,
            "ft_compl_all1": all(r["completion"] == 1.0 for r in ft),
            "ft_backlog_all0": all(r["backlog"] == 0 for r in ft),
            "win_holds": dw["hi"] < 0.0}
    res["WIN_ROBUST"] = det_ok and all(res["levels"][lv]["win_holds"] for lv in SEEDS)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "reverify_report.md")
    print(f"\nWIN_ROBUST={res['WIN_ROBUST']} det={det_ok}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res, path):
    lines = ["# 채택 정책 재검증 — 규칙 초과가 우연인가 (신선 770k 대역)", "",
             f"> FT+H1 vs SF · 수준별 {res['n_seeds_per_cell']['mid']} seed · "
             f"미사용 대역 · 결정론 {'OK' if res['determinism_ok'] else 'FAIL'} · "
             "문헌 보정 조건", "",
             "| 수준 | SF 대기 | FT 대기 | Δ대기 [CI] | ΔP95 [CI] | 완주 | 우연? |",
             "|---|---|---|---|---|---|---|"]
    for lv, a in res["levels"].items():
        dw, dp = a["d_wait"], a["d_p95"]
        lines.append(f"| {lv} | {a['sf_wait']} | {a['ft_wait']} | {dw['mean']} "
                     f"[{dw['lo']}, {dw['hi']}] | {dp['mean']} [{dp['lo']}, {dp['hi']}] "
                     f"| {'1.0' if a['ft_compl_all1'] else '<1'} "
                     f"| {'재현(우연 아님)' if a['win_holds'] else '깨짐'} |")
    orig = "원 test(20 seed 744k): mid −1.69 [−2.34,−1.12]·high −2.61 [−3.51,−1.68]"
    lines += ["", f"**재검증: {'규칙 초과 재현 = 우연 아님' if res['WIN_ROBUST'] else '재현 실패'}**",
              f"> 대조 {orig}. 효과 크기·부호가 신선 대역서 재현되면 seed 특이 아님.",
              "> 단 FT 는 단일 학습 궤적 — 본 검증은 '고정 모델의 seed 강건성'이지 "
              "'재학습 반복 강건성'은 아님(그건 재학습 필요, 별개)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_reverify()
