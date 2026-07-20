"""YR-077 1단계 — 크레인 고장 주입 강건성 (prereg 2026-07-20 동결 실행)."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMix, ActionMixError, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.joint_distill import CentralJointValuePolicy, load_student
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci
from .yr073_joint_distill import SLOTS, LEVEL

RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr077_robust")
MODEL = Path("outputs/reports/yr074_finetune/student_ft.pt")
SEEDS = {"mid": list(range(747000, 747012)), "high": list(range(747100, 747112))}
OUTAGES = (0, 1, 2)


def _sim(level, seed, n_out):
    profile = build_calibrated_profile()
    params = calibrated_load_params(level, fault_outages=n_out)
    sim = TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                            check_invariants=True)
    sim.info_level = LEVEL
    return sim


def run_yr077(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    facs = {"SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
            "STUDENT_FT": lambda: CentralJointValuePolicy(
                net, norm, CandidateGenerator(), SLOTS, name="STUDENT_FT")}
    rows, aggs = [], {}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, seeds in SEEDS.items():
            for n_out in OUTAGES:
                for name, fac in facs.items():
                    agg = aggs.setdefault((level, n_out, name), ActionMix())
                    for s in seeds:
                        row = run_joint_episode(_sim(level, s, n_out), fac(), RC,
                                                generator=CandidateGenerator())
                        mix = row["_mix"]
                        for kind, n in mix.counts.items():
                            agg.counts[kind] = agg.counts.get(kind, 0) + n
                        agg.serve_available += mix.serve_available
                        agg.serve_taken += mix.serve_taken
                        r = {"seed": s, "level": level, "outages": n_out, "arm": name,
                             "mean_wait": round(row["mean_wait_min"], 4),
                             "p95_wait": round(row["p95_wait_min"], 4),
                             "completion": round(row["completion_rate"], 4),
                             "backlog": row["backlog"]}
                        rows.append(r)
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    sel = [x for x in rows if x["level"] == level
                           and x["outages"] == n_out and x["arm"] == name]
                    print(f"[{level}/out{n_out}/{name}] "
                          f"wait={fmean(x['mean_wait'] for x in sel):.3f} "
                          f"compl={fmean(x['completion'] for x in sel):.3f}", flush=True)
    by = lambda lv, o, a: sorted((r for r in rows if r["level"] == lv
                                  and r["outages"] == o and r["arm"] == a),
                                 key=lambda r: r["seed"])
    r0 = by("mid", 1, "STUDENT_FT")[0]
    row2 = run_joint_episode(_sim("mid", r0["seed"], 1), facs["STUDENT_FT"](), RC,
                             generator=CandidateGenerator())
    det_ok = round(row2["mean_wait_min"], 4) == r0["mean_wait"]
    res: dict = {"prereg": "2026-07-20-YR-077-돌발강건성-prereg.md",
                 "determinism_ok": det_ok, "cells": {}}
    inj_pass = []
    for level in SEEDS:
        for n_out in OUTAGES:
            ft, sf = by(level, n_out, "STUDENT_FT"), by(level, n_out, "SF_SPT")
            d = _paired_ci([a["mean_wait"] - b["mean_wait"] for a, b in zip(ft, sf)])
            agg = aggs[(level, n_out, "STUDENT_FT")]
            try:
                assert_healthy_action_mix(agg, label=f"{level}/o{n_out}")
                healthy = True
            except ActionMixError:
                healthy = False
            cell = {"ft_wait": round(fmean(r["mean_wait"] for r in ft), 3),
                    "sf_wait": round(fmean(r["mean_wait"] for r in sf), 3),
                    "ft_compl": round(fmean(r["completion"] for r in ft), 4),
                    "sf_compl": round(fmean(r["completion"] for r in sf), 4),
                    "ft_p95": round(fmean(r["p95_wait"] for r in ft), 2),
                    "sf_p95": round(fmean(r["p95_wait"] for r in sf), 2),
                    "d_wait": d, "gain_kept": d["hi"] < 0.0, "mix_healthy": healthy,
                    "compl_guard": fmean(r["completion"] for r in ft)
                    >= fmean(r["completion"] for r in sf) - 0.05}
            res["cells"][f"{level}/out{n_out}"] = cell
            if n_out > 0:
                inj_pass.append(cell["gain_kept"] and cell["compl_guard"]
                                and cell["mix_healthy"])
    res["R1_PASS"] = det_ok and all(inj_pass)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    lines = [f"# YR-077 1단계 — 고장 주입 강건성 (R1 {'통과' if res['R1_PASS'] else '기각/부분'})",
             "", "> 문헌 보정 조건 · outage=15분 크레인 정지 · paired 12 seed/셀", "",
             "| 셀 | SF 대기 | FT 대기 | Δ [CI] | FT/SF 완주 | 유지 |", "|---|---|---|---|---|---|"]
    for k, c in res["cells"].items():
        d = c["d_wait"]
        lines.append(f"| {k} | {c['sf_wait']} | {c['ft_wait']} | {d['mean']} "
                     f"[{d['lo']}, {d['hi']}] | {c['ft_compl']}/{c['sf_compl']} "
                     f"| {'✅' if c['gain_kept'] else '❌'} |")
    (out / "yr077_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nR1_PASS={res['R1_PASS']} det={det_ok}", flush=True)
    print("DONE", flush=True)
    return res


if __name__ == "__main__":
    run_yr077()
