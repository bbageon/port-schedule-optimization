"""YR-080 2b — 기준재 교사 증류, 학생층 판정.

2a 결론: 교사층에서 목적 재설계는 실효(본선 보호). 그러나 배포물은 rollout 없는
feed-forward 학생 NN 이고, 저장소 유일 실측 증류(yr073, 구목적)는 붕괴했다. 사용자
결정(2026-07-22): 교사 선정 기준 = **학생 재현성 우선** → 관측 가능한 짧은 창
(sts5@1800, 선행신호가 학생 입력 안) 교사를 1순위로 증류해 **학생이 살아남는지** 본다.

판정 지표(학생층): berth_overrun(선석초과)·트럭 P95(꼬리)·건전성(action mix)·완주·
backlog·top1 재현율(교사 선택 재현). yr073 붕괴 지표(healthy=false·확보율 57%)와 대조.

실행: WSL torch CPU. 수집은 병렬(교사 rollout, torch 무관), 학습은 순차. 결정론 불변.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.adapter import capture
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _feasible_joint,
                                    _rollout_cost, _wait_of, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.encoding import StateNorm, encode_observation
from ..integrated.joint_distill import (CentralJointValuePolicy, JointDecisionSample,
                                        save_student, top1_agreement, train_joint_net)
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr059_state_norm import fit_state_norm

LEVEL = InformationLevel.PRE_ADVICE
HORIZON = 1_800.0                                    # 관측가능 짧은 창 (학생 재현성 우선)
# 교사 목적: 기준재 (이동0·sts5 선행신호·ρ_vessel 33=고원, long_wait 1). 8~33 동치.
TEACHER_OVERRIDES = {"crane_travel": 0.0, "empty_travel": 0.0, "sts_wait": 5.0}
RC = RewardCalculator.numeraire(TEACHER_OVERRIDES)
SF = lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF")
CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
SLOTS = tuple(sorted(c.crane_id for c in build_calibrated_profile().cranes))
OUT = Path("outputs/reports/yr080_distill")

# seed 대역 (셀별): train·val·test 분리
TRAIN_N, VAL_N, TEST_N = 6, 3, 6
BASE = {"mid-loose": 810000, "high-loose": 810100, "mid-tight": 810200, "high-tight": 810300}


def _sim(cell: str, seed: int):
    level, dmult = CELLS[cell]
    prof = build_calibrated_profile()
    params = calibrated_load_params(level, vessel_deadline_mult=dmult)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, params),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def _p95_min(sim) -> float:
    ws = sorted(sim.kpis.wait_samples_s)
    if not ws:
        return 0.0
    return ws[min(len(ws) - 1, int(0.95 * len(ws)))] / 60.0


def collect_one(task: tuple) -> dict:
    """교사(기준재 scalar argmin) 로깅 에피소드 — 병렬 워커용(top-level·torch 무관).

    반환: 증류 표본(직렬화 tuple) + 교사 KPI(berth·P95·완주). norm 은 refs 로 재구성.
    """
    cell, seed, norm_refs = task
    norm = StateNorm(refs=norm_refs, basis="fitted_baseline_p90")
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC, horizon_s=HORIZON, generator=gen, objective=None)
    sf = SF()
    samples, k, n_dis = [], 0, 0
    dp = sim.run_until_decision()
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "collect", k, generator=gen)
        encs = {ob.crane_id: encode_observation(state, ob, norm=norm) for ob in obs}
        ca, cb = SLOTS
        pos = lambda c, a: (encs[c].candidate_ids.index(a[c].candidate_id)
                            if c in a else -1)
        combos, scores, assigns, best, best_key = [], [], [], None, None
        for combo in jr._admissible_combos(sim, dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if not _feasible_joint(sim, assign):
                continue
            cost, _ = _rollout_cost(sim, assign, RC, horizon_s=HORIZON,
                                    base_policy=sf, generator=gen)
            score = round(cost, 9)                     # 기준재 경제 스칼라
            tie = tuple((c, assign[c].candidate_id) for c in sorted(dp.crane_ids))
            if best_key is None or (score, tie) < best_key:
                best_key, best = (score, tie), len(assigns)
            combos.append((pos(ca, assign), pos(cb, assign)))
            scores.append(score)
            assigns.append(assign)
        if assigns:
            sfa = sf.decide(sim, dp, gen_by)
            sf_pair = (pos(ca, sfa), pos(cb, sfa))
            sf_pos = combos.index(sf_pair) if sf_pair in combos else None
            disagree = sf_pos is None or sf_pos != best
            n_dis += int(disagree)
            ea, eb = encs.get(ca), encs.get(cb)
            samples.append(JointDecisionSample(
                (ea or eb).g, ea.yc if ea else (), ea.queue if ea else (),
                ea.cand if ea else (), eb.yc if eb else (), eb.queue if eb else (),
                eb.cand if eb else (), tuple(combos), tuple(scores), best, sf_pos,
                disagree, cell))
            _apply(sim, assigns[best])
        else:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
        k += 1
    jobs = list(sim.jobs.values())
    return {"cell": cell, "seed": seed, "samples": samples, "n_dec": k, "n_dis": n_dis,
            "berth": round(sim.kpis.berth_overrun_s / 60.0, 3),
            "p95": round(_p95_min(sim), 3),
            "mean_wait": round(fmean(w / 60.0 for w in sim.kpis.wait_samples_s)
                               if sim.kpis.wait_samples_s else 0.0, 3),
            "completion": sum(1 for j in jobs if j.status.name == "DONE") / len(jobs)}


def eval_student(net, norm, cell: str, seed: int) -> dict:
    pol = CentralJointValuePolicy(net, norm, CandidateGenerator(), SLOTS, name="STUDENT")
    row = run_joint_episode(_sim(cell, seed), pol, RC, generator=CandidateGenerator())
    healthy = True
    try:
        assert_healthy_action_mix(row["_mix"], label=f"{cell}/STUDENT/s{seed}")
    except ActionMixError:
        healthy = False
    return {"cell": cell, "seed": seed,
            "berth": round(row["berth_overrun_min"], 3),
            "p95": round(row["p95_wait_min"], 3),
            "mean_wait": round(row["mean_wait_min"], 3),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "healthy": healthy, "repo_share": row["action_mix"]["shares"].get("REPOSITION", 0.0),
            "serve_share": row["action_mix"]["shares"].get("SERVE", 0.0)}


def run(out: Path = OUT, workers: int = 16) -> dict:
    import multiprocessing as mp
    out.mkdir(parents=True, exist_ok=True)
    prof = build_calibrated_profile()
    print("[phase] fit state_norm", flush=True)
    fit_seeds = [BASE["high-tight"] + i for i in range(5)]
    norm, _ = fit_state_norm(prof, calibrated_load_params("high", vessel_deadline_mult=0.5),
                             fit_seeds, progress=lambda *_: None)
    refs = norm.refs

    train_tasks = [(c, BASE[c] + i, refs) for c in CELLS for i in range(TRAIN_N)]
    test_tasks = [(c, BASE[c] + 100 + i, refs) for c in CELLS for i in range(TEST_N)]
    print(f"[phase] parallel collect (train {len(train_tasks)} + test {len(test_tasks)})",
          flush=True)
    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        allc = pool.map(collect_one, train_tasks + test_tasks)
    train_c = allc[:len(train_tasks)]
    test_c = allc[len(train_tasks):]
    train_samples = [s for r in train_c for s in r["samples"]]
    test_samples = [s for r in test_c for s in r["samples"]]
    print(f"[collect] train {len(train_samples)} · test {len(test_samples)} 표본 "
          f"({round(time.perf_counter() - t0)}s)", flush=True)

    def val_fn(net):
        waits = []
        for c in CELLS:
            for i in range(VAL_N):
                waits.append(eval_student(net, norm, c, BASE[c] + 50 + i)["mean_wait"])
        return round(fmean(waits), 4)

    print("[phase] train student", flush=True)
    tr = train_joint_net(train_samples, val_fn=val_fn, progress=lambda *_: None)
    save_student(out / "student_v0.pt", tr, refs)
    agree = top1_agreement(tr.net, test_samples)

    print("[phase] eval student on test", flush=True)
    st_rows = [eval_student(tr.net, norm, c, BASE[c] + 100 + i)
               for c in CELLS for i in range(TEST_N)]

    def by(rows, cell):
        return [r for r in rows if r["cell"] == cell]

    res: dict = {"horizon_s": HORIZON, "teacher": "numeraire " + str(TEACHER_OVERRIDES),
                 "agreement": agree, "best_tag": tr.best_tag, "cells": {}}
    for c in CELLS:
        s, t = by(st_rows, c), by(test_c, c)
        res["cells"][c] = {
            "student": {"berth": round(fmean(r["berth"] for r in s), 2),
                        "p95": round(fmean(r["p95"] for r in s), 2),
                        "mean_wait": round(fmean(r["mean_wait"] for r in s), 3),
                        "healthy_all": all(r["healthy"] for r in s),
                        "completion_all1": all(r["completion"] == 1.0 for r in s),
                        "backlog_all0": all(r["backlog"] == 0 for r in s),
                        "repo_share": round(fmean(r["repo_share"] for r in s), 3),
                        "serve_share": round(fmean(r["serve_share"] for r in s), 3)},
            "teacher": {"berth": round(fmean(r["berth"] for r in t), 2),
                        "p95": round(fmean(r["p95"] for r in t), 2),
                        "mean_wait": round(fmean(r["mean_wait"] for r in t), 3),
                        "completion_all1": all(r["completion"] == 1.0 for r in t)}}
    # yr073 붕괴 대조 게이트
    res["student_healthy_all"] = all(res["cells"][c]["student"]["healthy_all"] for c in CELLS)
    res["student_complete_all"] = all(res["cells"][c]["student"]["completion_all1"] for c in CELLS)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr080_2b_report.md")
    print(f"\n[verdict] student healthy_all={res['student_healthy_all']} "
          f"complete_all={res['student_complete_all']} "
          f"top1={agree['top1_all']}·disagree {agree['top1_disagree']}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    a = res["agreement"]
    lines = ["# YR-080 2b — 기준재 교사 증류 (학생층 판정)", "",
             f"> 교사={res['teacher']} @{int(res['horizon_s'])}s · 학생 재현성 우선(사용자 결정).",
             f"> top1 재현 전체 {a['top1_all']}·분기한정 {a['top1_disagree']}(n={a['n_disagree']}) "
             f"· best {res['best_tag']}. **문헌 보정 시뮬 조건.**", "",
             "| 셀 | arm | berth(분) | P95(분) | 평균대기 | 건전 | 완주 |",
             "|---|---|---|---|---|---|---|"]
    for c, cv in res["cells"].items():
        s, t = cv["student"], cv["teacher"]
        lines.append(f"| {c} | 학생 | {s['berth']} | {s['p95']} | {s['mean_wait']} "
                     f"| {'OK' if s['healthy_all'] else 'FAIL'} "
                     f"| {'OK' if s['completion_all1'] else 'FAIL'} |")
        lines.append(f"| {c} | 교사 | {t['berth']} | {t['p95']} | {t['mean_wait']} | — "
                     f"| {'OK' if t['completion_all1'] else 'FAIL'} |")
    verdict = ("생존" if res["student_healthy_all"] and res["student_complete_all"]
               else "붕괴/부분붕괴")
    lines += ["", f"## 판정: 학생 **{verdict}** "
              f"(healthy_all={res['student_healthy_all']}·complete_all={res['student_complete_all']})",
              "", "yr073(구목적) 붕괴 대조: healthy=false·확보율 57%. 여기서 학생이 건전·완주·"
              "교사 berth 재현하면 기준재 교사가 증류 가능(2a 교사층 실효가 배포로 전이).",
              "미달 시 = 증류 자체가 병목(목적과 별개) → 학생 구조/DAgger/FT 후속."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    run(Path(a.out), workers=a.workers)
