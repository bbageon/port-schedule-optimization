"""YR-073-b — 순위 증류 확증 재판정 (prereg 2026-07-20 동결 실행).

기박제 student_v1.pt (commit 2c7410f) 를 사전 지정 상수로 적재 — 재학습·재선택
없음. 신규 seed 744k 대역에서 SF-SPT 와 paired 재평가만 수행한다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import (ActionMix, ActionMixError, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.joint_distill import CentralJointValuePolicy, load_student
from .yr071_realign_g0 import _paired_ci
from .yr073_joint_distill import SLOTS, _sim

RC = RewardCalculator.assumed_default()
OUT = Path("outputs/reports/yr073b_confirm")
MODEL = Path("outputs/reports/yr073_distill/student_v1.pt")
SEEDS = {"mid": list(range(744000, 744020)), "high": list(range(744100, 744120))}


def _row(level: str, seed: int, policy, name: str, agg: ActionMix) -> dict:
    t0 = time.perf_counter()
    row = run_joint_episode(_sim(level, seed), policy, RC,
                            generator=CandidateGenerator())
    mix = row["_mix"]
    for kind, n in mix.counts.items():
        agg.counts[kind] = agg.counts.get(kind, 0) + n
    agg.serve_available += mix.serve_available
    agg.serve_taken += mix.serve_taken
    unhealthy = False
    try:
        assert_healthy_action_mix(mix, label=f"{level}/{name}/s{seed}")
    except ActionMixError:
        unhealthy = True
    return {"seed": seed, "level": level, "arm": name,
            "wall_s": round(time.perf_counter() - t0, 2),
            "mean_wait": round(row["mean_wait_min"], 4),
            "p95_wait": round(row["p95_wait_min"], 4),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "swa": row["action_mix"]["serve_when_available"],
            "repo_share": row["action_mix"]["shares"].get("REPOSITION", 0.0),
            "unhealthy_episode": unhealthy}


def run_yr073b(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    net, norm = load_student(MODEL)
    facs = {
        "SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "STUDENT_R1": lambda: CentralJointValuePolicy(net, norm, CandidateGenerator(),
                                                      SLOTS, name="STUDENT_R1"),
    }
    rows, aggs = [], {}
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for level, seeds in SEEDS.items():
            for name, fac in facs.items():
                agg = aggs.setdefault((level, name), ActionMix())
                for s in seeds:
                    r = _row(level, s, fac(), name, agg)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                sel = [r for r in rows if r["level"] == level and r["arm"] == name]
                print(f"[{level}/{name}] wait={fmean(x['mean_wait'] for x in sel):.3f} "
                      f"repo={fmean(x['repo_share'] for x in sel):.3f} "
                      f"unhealthy={sum(x['unhealthy_episode'] for x in sel)}", flush=True)
    det_ok = True
    for level in SEEDS:
        for s in SEEDS[level][:2]:
            r2 = _row(level, s, facs["STUDENT_R1"](), "STUDENT_R1", ActionMix())
            r1 = next(r for r in rows if r["level"] == level
                      and r["arm"] == "STUDENT_R1" and r["seed"] == s)
            if any(r1[k] != r2[k] for k in ("mean_wait", "p95_wait", "swa")):
                det_ok = False
    res: dict = {"prereg": "2026-07-20-YR-073b-확증재판정-prereg.md",
                 "model": str(MODEL), "determinism_ok": det_ok, "levels": {}}
    for level in SEEDS:
        by = lambda a: sorted((r for r in rows if r["level"] == level
                               and r["arm"] == a), key=lambda r: r["seed"])
        st, sf = by("STUDENT_R1"), by("SF_SPT")
        dw = _paired_ci([a["mean_wait"] - b["mean_wait"] for a, b in zip(st, sf)])
        dp95 = _paired_ci([a["p95_wait"] - b["p95_wait"] for a, b in zip(st, sf)])
        agg = aggs[(level, "STUDENT_R1")]
        try:
            assert_healthy_action_mix(agg, label=f"{level}/policy")
            policy_healthy = True
        except ActionMixError:
            policy_healthy = False
        repo_st = fmean(r["repo_share"] for r in st)
        repo_sf = fmean(r["repo_share"] for r in sf)
        lv = {"student": {"mean_wait": round(fmean(r["mean_wait"] for r in st), 4),
                          "p95_wait": round(fmean(r["p95_wait"] for r in st), 4),
                          "repo_share": round(repo_st, 3),
                          "swa_agg": round(agg.serve_when_available(), 3),
                          "wall_s": round(fmean(r["wall_s"] for r in st), 2),
                          "unhealthy_episodes": sum(r["unhealthy_episode"] for r in st)},
              "sf": {"mean_wait": round(fmean(r["mean_wait"] for r in sf), 4),
                     "repo_share": round(repo_sf, 3)},
              "d_wait": dw, "d_p95": dp95,
              "g1_pass": dw["hi"] < 0.0,
              "guards": {"completion_all1": all(r["completion"] == 1.0 for r in st),
                         "backlog_all0": all(r["backlog"] == 0 for r in st),
                         "p95_ok": not (dp95["lo"] > 0.0),
                         "policy_mix_healthy": policy_healthy,
                         "repo_relative_ok": repo_st <= repo_sf + 0.10}}
        lv["guard_all"] = all(lv["guards"].values())
        res["levels"][level] = lv
    res["G1B_PASS"] = det_ok and all(res["levels"][lv]["g1_pass"]
                                     and res["levels"][lv]["guard_all"]
                                     for lv in SEEDS)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr073b_report.md")
    print(f"\nG1B_PASS={res['G1B_PASS']} det={det_ok}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    lines = ["# YR-073-b — 순위 증류 확증 재판정 (사전 지정 R1 × 신규 seed)", "",
             f"> 모델 {res['model']} (commit 2c7410f 박제본, 재학습·재선택 없음) · "
             f"결정론 {'OK' if res['determinism_ok'] else 'FAIL'} · "
             "**문헌 보정 시뮬레이션 조건**", ""]
    for level, lv in res["levels"].items():
        st, w, p = lv["student"], lv["d_wait"], lv["d_p95"]
        lines += [f"## {level}", "",
                  f"- 학생 평균대기 **{st['mean_wait']}분** vs SF {lv['sf']['mean_wait']}"
                  f" — Δ **{w['mean']} [{w['lo']}, {w['hi']}]** → G1′ "
                  f"{'✅' if lv['g1_pass'] else '❌'} · ΔP95 {p['mean']} "
                  f"[{p['lo']}, {p['hi']}]",
                  f"- REPO {st['repo_share']} (SF {lv['sf']['repo_share']}) · 집계 swa "
                  f"{st['swa_agg']} · 퇴화 에피소드 {st['unhealthy_episodes']}/20 (보고) "
                  f"· wall {st['wall_s']}s/ep · guards {lv['guards']}", ""]
    lines += [f"**판정: G1′ {'통과' if res['G1B_PASS'] else '기각'}** · 원자료 rows.jsonl"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_yr073b()
