"""YR-071 G0 — NEW 계층목적 JR_NEW vs SF-SPT 확증 (prereg 동결 실행).

prereg: .claude/docs/strategy-history/2026-07-19-YR-071-목적재정렬-G0-prereg.md
- 지형: build_calibrated_profile × calibrated_load_params(mid 56 / high 80, 피크)
- arms NOISY(eta±300): SF_SPT·JR_OLD·JR_NEW / PERFECT(eta 0): SF_SPT_P0·JR_NEW_P0
- seed: mid 720000~720019 · high 720100~720119 (신규 대역, 수준별 20)
- G0: paired Δ평균대기(JR_NEW−SF_SPT) 95% CI 상한 < 0 이 양 수준 모두.
  보고는 절대차(분)+CI. guard: 완료율·backlog·P95 유의 악화·행동분포·결정론.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import fmean

from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.objectives import hierarchy_key
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

OUT_DIR = Path("outputs/reports/yr071_realign_g0")
RC = RewardCalculator.assumed_default()
N_SEEDS = 20
SEED_BASE = {"mid": 720000, "high": 720100}
BOOT_N, BOOT_SEED = 10_000, 75_168


def _policy(arm: str):
    if arm.startswith("SF_SPT"):
        return ResolverPolicy(ServiceFirstSPTPreference(), arm)
    gen = CandidateGenerator()
    obj = hierarchy_key if "NEW" in arm else None
    pol = JointRolloutGreedy(RC, generator=gen, objective=obj)
    pol.name = arm
    return pol


def _episode(level: str, seed: int, arm: str) -> dict:
    eta = 0.0 if arm.endswith("_P0") else 300.0
    profile = build_calibrated_profile()
    params = calibrated_load_params(level, eta_error_s=eta)
    sim = TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                            check_invariants=True)
    t0 = time.perf_counter()
    row = run_joint_episode(sim, _policy(arm), RC, generator=CandidateGenerator())
    wall = round(time.perf_counter() - t0, 2)
    mix = row["action_mix"]
    healthy = True
    try:
        assert_healthy_action_mix(row["_mix"], label=f"{level}/{arm}/s{seed}")
    except ActionMixError:
        healthy = False
    return {"seed": seed, "arm": arm, "level": level, "wall_s": wall,
            "mean_wait": round(row["mean_wait_min"], 4),
            "p95_wait": round(row["p95_wait_min"], 4),
            "total_cost_OLD": round(row["total_cost"], 2),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "swa": mix["serve_when_available"],
            "serve_share": mix["shares"].get("SERVE", 0.0),
            "repo_share": mix["shares"].get("REPOSITION", 0.0),
            "pre_share": mix["shares"].get("PRE_REHANDLE", 0.0),
            "n_decisions": row["n_decisions"], "healthy_mix": healthy,
            "combo_truncations": row["combo_truncations"]}


def _paired_ci(diffs: list[float]) -> dict:
    rng = random.Random(BOOT_SEED)
    m = len(diffs)
    means = sorted(fmean(diffs[rng.randrange(m)] for _ in range(m))
                   for _ in range(BOOT_N))
    return {"mean": round(fmean(diffs), 4), "lo": round(means[int(0.025 * BOOT_N)], 4),
            "hi": round(means[int(0.975 * BOOT_N)], 4), "n": m}


def _by(rows, level, arm):
    sel = sorted((r for r in rows if r["level"] == level and r["arm"] == arm),
                 key=lambda r: r["seed"])
    return sel


def _diff(rows, level, a, b, key):
    ra, rb = _by(rows, level, a), _by(rows, level, b)
    assert [r["seed"] for r in ra] == [r["seed"] for r in rb]
    return _paired_ci([x[key] - y[key] for x, y in zip(ra, rb)])


def run_yr071(out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    rows: list[dict] = []
    arms = ("SF_SPT", "JR_OLD", "JR_NEW", "SF_SPT_P0", "JR_NEW_P0")
    with rows_path.open("w", encoding="utf-8") as f:
        for level in ("mid", "high"):
            for arm in arms:
                for i in range(N_SEEDS):
                    r = _episode(level, SEED_BASE[level] + i, arm)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                agg = _by(rows, level, arm)
                print(f"[{level}/{arm}] wait={fmean(x['mean_wait'] for x in agg):.3f} "
                      f"p95={fmean(x['p95_wait'] for x in agg):.2f} "
                      f"swa={fmean(x['swa'] for x in agg):.3f} "
                      f"repo={fmean(x['repo_share'] for x in agg):.3f} "
                      f"wall={fmean(x['wall_s'] for x in agg):.1f}s", flush=True)
    # 결정론 guard: 수준별 JR_NEW 선두 2 seed 재실행 동일
    det_ok = True
    for level in ("mid", "high"):
        for i in range(2):
            r2 = _episode(level, SEED_BASE[level] + i, "JR_NEW")
            r1 = next(r for r in rows if r["level"] == level and r["arm"] == "JR_NEW"
                      and r["seed"] == SEED_BASE[level] + i)
            if any(r1[k] != r2[k] for k in ("mean_wait", "p95_wait", "total_cost_OLD",
                                            "n_decisions")):
                det_ok = False
    res: dict = {"prereg": "2026-07-19-YR-071-목적재정렬-G0-prereg.md",
                 "n_seeds": N_SEEDS, "determinism_ok": det_ok, "levels": {}}
    for level in ("mid", "high"):
        d_wait = _diff(rows, level, "JR_NEW", "SF_SPT", "mean_wait")
        d_p95 = _diff(rows, level, "JR_NEW", "SF_SPT", "p95_wait")
        lv = {
            "arms": {arm: {
                "mean_wait": round(fmean(r["mean_wait"] for r in _by(rows, level, arm)), 4),
                "p95_wait": round(fmean(r["p95_wait"] for r in _by(rows, level, arm)), 4),
                "total_cost_OLD": round(fmean(r["total_cost_OLD"]
                                              for r in _by(rows, level, arm)), 2),
                "swa": round(fmean(r["swa"] for r in _by(rows, level, arm)), 3),
                "serve_share": round(fmean(r["serve_share"] for r in _by(rows, level, arm)), 3),
                "repo_share": round(fmean(r["repo_share"] for r in _by(rows, level, arm)), 3),
                "wall_s": round(fmean(r["wall_s"] for r in _by(rows, level, arm)), 1),
                "healthy_all": all(r["healthy_mix"] for r in _by(rows, level, arm)),
                "completion_all1": all(r["completion"] == 1.0
                                       for r in _by(rows, level, arm)),
                "backlog_all0": all(r["backlog"] == 0 for r in _by(rows, level, arm)),
            } for arm in arms},
            "d_wait_JRNEW_vs_SF": d_wait, "d_p95_JRNEW_vs_SF": d_p95,
            "d_wait_JRNEW_vs_JROLD": _diff(rows, level, "JR_NEW", "JR_OLD", "mean_wait"),
            "d_wait_perfect_SF": _diff(rows, level, "SF_SPT_P0", "SF_SPT", "mean_wait"),
            "d_wait_perfect_JRNEW": _diff(rows, level, "JR_NEW_P0", "JR_NEW", "mean_wait"),
            "g0_wait_pass": d_wait["hi"] < 0.0,
            "guard_p95_ok": not (d_p95["lo"] > 0.0),
        }
        jn = lv["arms"]["JR_NEW"]
        lv["guard_all"] = (jn["completion_all1"] and jn["backlog_all0"]
                           and jn["healthy_all"] and lv["guard_p95_ok"])
        res["levels"][level] = lv
    res["G0_PASS"] = (det_ok and all(res["levels"][lv]["g0_wait_pass"]
                                     and res["levels"][lv]["guard_all"]
                                     for lv in ("mid", "high")))
    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    _report(res, out_dir / "yr071_report.md")
    print(f"\nG0_PASS={res['G0_PASS']} determinism={det_ok}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    lines = ["# YR-071 G0 — NEW 계층목적 JR_NEW vs SF-SPT (문헌 보정 v2 × 현실 부하)", "",
             f"> prereg 동결 실행 · seed 수준별 {res['n_seeds']} (신규 대역) · "
             f"결정론 {'OK' if res['determinism_ok'] else 'FAIL'} · "
             "**문헌 보정 시뮬레이션 조건 — 실운영 주장 아님**", "",
             "| 수준 | arm | 평균대기(분) | P95(분) | OLD총비용 | swa | SERVE | REPO | wall(s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for level, lv in res["levels"].items():
        for arm, a in lv["arms"].items():
            lines.append(f"| {level} | {arm} | {a['mean_wait']} | {a['p95_wait']} "
                         f"| {a['total_cost_OLD']} | {a['swa']} | {a['serve_share']} "
                         f"| {a['repo_share']} | {a['wall_s']} |")
    lines += ["", "## paired Δ (절대차, 95% CI — 백분율 아님)", ""]
    for level, lv in res["levels"].items():
        w, p = lv["d_wait_JRNEW_vs_SF"], lv["d_p95_JRNEW_vs_SF"]
        lines += [f"- **{level} — Δ평균대기(JR_NEW−SF_SPT) {w['mean']}분 "
                  f"[{w['lo']}, {w['hi']}]** → G0 {'✅' if lv['g0_wait_pass'] else '❌'} · "
                  f"ΔP95 {p['mean']} [{p['lo']}, {p['hi']}] "
                  f"(guard {'OK' if lv['guard_p95_ok'] else 'FAIL'})",
                  f"  - vs JR_OLD: Δ대기 {lv['d_wait_JRNEW_vs_JROLD']['mean']}분 "
                  f"[{lv['d_wait_JRNEW_vs_JROLD']['lo']}, {lv['d_wait_JRNEW_vs_JROLD']['hi']}]"
                  f" · 완벽 ETA(NEW): SF {lv['d_wait_perfect_SF']['mean']} "
                  f"[{lv['d_wait_perfect_SF']['lo']}, {lv['d_wait_perfect_SF']['hi']}] / "
                  f"JR_NEW {lv['d_wait_perfect_JRNEW']['mean']} "
                  f"[{lv['d_wait_perfect_JRNEW']['lo']}, {lv['d_wait_perfect_JRNEW']['hi']}]"]
    lines += ["", f"## 판정: **G0 {'통과' if res['G0_PASS'] else '기각'}**", "",
              "- 의미 제한 (prereg §4): 트럭 대기 목적 정렬 + 현 행동공간 헤드룸 확인 — "
              "본선 보호 검증 아님 (본선지연 0 시나리오, YR-041 별도).",
              "- 원자료 rows.jsonl (seed별 전체 지표·wall-time)"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr071()
