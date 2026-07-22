"""YR-080 2단계a — 교사 재판정 (신 기준재 목적 vs 구 사전식 목적).

spec 재판정 규칙: "계층 순위나 교사 행동이 실질적으로 바뀌면 현재 체크포인트는 자동
승계하지 않는다. 교사 재수집→증류/DAgger→FT 재검증을 다시 한다."

이 드라이버는 **NN 학습 0** 으로 그 게이트에 답한다 — 신/구 교사를 같은 시나리오·같은
계획기(1800s rollout)에서 굴려:
  ① KPI 분기: 트럭 평균/P95 대기 · 선석 초과(berth_overrun) · 기준재 총비용 (paired CI)
  ② 결정 일치율: 신 교사 궤적에서 구 교사가 같은 조합을 골랐을 비율 (셀당 1 seed 스팟)
을 잰다. 타이트 마감 셀(ρ_vessel 이 실제로 무는 곳)이 핵심 — 느슨 셀만으론 YR-080d 처럼
"목적 바꿔도 행동 불변" 함정에 빠진다 (spec §2).

판정: 타이트 셀에서 신 교사가 선석 초과를 유의하게 줄이거나(ΔCI<0) 결정이 갈리면
(일치율↓) → 구 체크포인트 기각·2b(증류) 진행. 전 셀에서 신≈구면 체크포인트 유지.

실행: WSL (~/.venvs/yard-rl, torch 불필요 — rollout 만). 출력 outputs/reports/yr080_readjudicate/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _feasible_joint,
                                    _wait_of, assert_healthy_action_mix, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.objectives import hierarchy_key
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from .yr071_realign_g0 import _paired_ci

LEVEL = InformationLevel.PRE_ADVICE
HORIZON = 1_800.0                                 # YR-078 채택 창 — 교사·평가 공통
RC_NUM = RewardCalculator.numeraire_v1()          # 신 목적 (기준재 경제 스칼라)
RC_LEX = RewardCalculator.assumed_default()        # 구 목적 (사전식 tier 내부 상수)
OUT = Path("outputs/reports/yr080_readjudicate")

# 셀 = (level, vessel_deadline_mult). loose=2.0(현행)·tight=0.5(선석 초과 유발).
CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
BASE_SEED = {"mid-loose": 780000, "high-loose": 780100,
             "mid-tight": 780200, "high-tight": 780300}


def _sim(cell: str, seed: int):
    level, dmult = CELLS[cell]
    profile = build_calibrated_profile()
    params = calibrated_load_params(level, vessel_deadline_mult=dmult)
    sim = TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                            check_invariants=True)
    sim.info_level = LEVEL
    return sim


def _teacher_num():
    p = JointRolloutGreedy(RC_NUM, horizon_s=HORIZON, generator=CandidateGenerator(),
                           objective=None)         # scalar argmin = 기준재 경제 스칼라
    p.name = "TEACHER_NUM"
    return p


def _teacher_lex():
    p = JointRolloutGreedy(RC_LEX, horizon_s=HORIZON, generator=CandidateGenerator(),
                           objective=hierarchy_key)   # 사전식 (구 목적)
    p.name = "TEACHER_LEX"
    return p


ARMS = {"SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF_SPT"),
        "TEACHER_LEX": _teacher_lex, "TEACHER_NUM": _teacher_num}


def _eval_row(cell: str, seed: int, factory, name: str) -> dict:
    """KPI 실행 — 로깅 RC 는 기준재 고정(총비용 arm 간 비교 가능). 정책 내부 RC 는 factory 몫."""
    t0 = time.perf_counter()
    row = run_joint_episode(_sim(cell, seed), factory(), RC_NUM,
                            generator=CandidateGenerator())
    healthy = True
    try:
        assert_healthy_action_mix(row["_mix"], label=f"{cell}/{name}/s{seed}")
    except ActionMixError:
        healthy = False
    mix = row["action_mix"]
    return {"cell": cell, "seed": seed, "arm": name,
            "wall_s": round(time.perf_counter() - t0, 2),
            "mean_wait": round(row["mean_wait_min"], 4),
            "p95_wait": round(row["p95_wait_min"], 4),
            "berth_overrun": round(row["berth_overrun_min"], 4),
            "numeraire_total": round(row["total_cost"], 4),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "repo_share": mix["shares"].get("REPOSITION", 0.0), "healthy": healthy}


def decision_agreement(cell: str, seed: int) -> dict:
    """신 교사 궤적에서 결정별로 구 교사·SF 가 같은 조합을 골랐을 비율 (스팟체크).

    두 교사 모두 같은 조합 집합·같은 rollout 창을 쓴다 — 차이는 순위 기준(스칼라 vs
    사전식)뿐. 신 교사가 실행(궤적 결정), 구 교사·SF 는 같은 상태에서 관측만.
    """
    from ..integrated.baselines import _rollout_cost
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    num = _teacher_num()
    sf = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    n = agree_lex = agree_sf = n_overrun_dec = 0
    dp = sim.run_until_decision()
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        combos = []
        for combo in num._admissible_combos(sim, dp, gen_by):
            assign = dict(zip(dp.crane_ids, combo))
            if _feasible_joint(sim, assign):
                combos.append(assign)
        if combos:
            def _key(assign, rc, obj):
                sink = {} if obj is not None else None
                cost, _ = _rollout_cost(sim, assign, rc, horizon_s=HORIZON,
                                        base_policy=sf, generator=gen, term_sink=sink)
                score = round(cost, 9) if obj is None else obj(sink)
                return (score, tuple((c, assign[c].candidate_id)
                                     for c in dp.crane_ids))
            num_pick = min(combos, key=lambda a: _key(a, RC_NUM, None))
            lex_pick = min(combos, key=lambda a: _key(a, RC_LEX, hierarchy_key))
            sfa = sf.decide(sim, dp, gen_by)
            n += 1
            same_lex = all(num_pick[c].candidate_id == lex_pick[c].candidate_id
                           for c in dp.crane_ids)
            same_sf = all(c in sfa and num_pick[c].candidate_id == sfa[c].candidate_id
                          for c in dp.crane_ids)
            agree_lex += int(same_lex)
            agree_sf += int(same_sf)
            _apply(sim, num_pick)
        else:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        dp = sim.run_until_decision()
    return {"cell": cell, "seed": seed, "n_dec": n,
            "agree_num_lex": round(agree_lex / max(1, n), 4),
            "agree_num_sf": round(agree_sf / max(1, n), 4)}


def _by(rows, cell, arm):
    return sorted((r for r in rows if r["cell"] == cell and r["arm"] == arm),
                  key=lambda r: r["seed"])


def _diff(rows, cell, a, b, key):
    ra, rb = _by(rows, cell, a), _by(rows, cell, b)
    return _paired_ci([x[key] - y[key] for x, y in zip(ra, rb)])


def run(out: Path = OUT, n_seeds: int = 10, agree_seeds: int = 1) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for cell in CELLS:
            seeds = [BASE_SEED[cell] + i for i in range(n_seeds)]
            for name, fac in ARMS.items():
                for s in seeds:
                    r = _eval_row(cell, s, fac, name)
                    rows.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                ag = _by(rows, cell, name)
                print(f"[kpi {cell}/{name}] wait={fmean(x['mean_wait'] for x in ag):.3f} "
                      f"berth={fmean(x['berth_overrun'] for x in ag):.2f} "
                      f"num={fmean(x['numeraire_total'] for x in ag):.2f} "
                      f"wall={fmean(x['wall_s'] for x in ag):.1f}s", flush=True)

    agree = []
    for cell in CELLS:
        for s in [BASE_SEED[cell] + i for i in range(agree_seeds)]:
            a = decision_agreement(cell, s)
            agree.append(a)
            print(f"[agree {cell}] n={a['n_dec']} num≈lex={a['agree_num_lex']} "
                  f"num≈sf={a['agree_num_sf']}", flush=True)

    res: dict = {"horizon_s": HORIZON, "n_seeds": n_seeds, "cells": {},
                 "agreement": agree}
    for cell in CELLS:
        arm_stats = {a: {k: round(fmean(r[k] for r in _by(rows, cell, a)), 4)
                         for k in ("mean_wait", "p95_wait", "berth_overrun",
                                   "numeraire_total", "repo_share", "wall_s")}
                     | {"completion_all1": all(r["completion"] == 1.0
                                               for r in _by(rows, cell, a)),
                        "backlog_all0": all(r["backlog"] == 0 for r in _by(rows, cell, a)),
                        "healthy_all": all(r["healthy"] for r in _by(rows, cell, a))}
                     for a in ARMS}
        d_berth = _diff(rows, cell, "TEACHER_NUM", "TEACHER_LEX", "berth_overrun")
        d_wait = _diff(rows, cell, "TEACHER_NUM", "TEACHER_LEX", "mean_wait")
        d_num = _diff(rows, cell, "TEACHER_NUM", "TEACHER_LEX", "numeraire_total")
        ag = [a for a in agree if a["cell"] == cell]
        res["cells"][cell] = {
            "arms": arm_stats,
            "d_berth_num_vs_lex": d_berth, "d_wait_num_vs_lex": d_wait,
            "d_numeraire_num_vs_lex": d_num,
            "agree_num_lex": round(fmean(a["agree_num_lex"] for a in ag), 4) if ag else None,
            # 재판정 신호: 선석초과 유의 감소(상한<0) 또는 결정 갈림(일치<0.9)
            "behavior_changed": bool(d_berth["hi"] < 0.0
                                     or (ag and fmean(a["agree_num_lex"]
                                                      for a in ag) < 0.9)),
        }
    tight = [c for c in CELLS if c.endswith("tight")]
    res["VERDICT_RECOLLECT"] = any(res["cells"][c]["behavior_changed"] for c in tight)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _report(res, out / "yr080_readjudicate_report.md")
    print(f"\nVERDICT_RECOLLECT={res['VERDICT_RECOLLECT']}", flush=True)
    print("DONE", flush=True)
    return res


def _report(res: dict, path: Path) -> None:
    lines = ["# YR-080 2단계a — 교사 재판정 (기준재 신목적 vs 사전식 구목적)", "",
             f"> 계획기 rollout {int(res['horizon_s'])}s · seed {res['n_seeds']}/셀 · "
             "NN 학습 0 (rollout 교사만) · **문헌 보정 시뮬레이션 조건**", "",
             "신/구 교사를 같은 시나리오·같은 계획기에서 굴려 행동 변화를 잰다. 타이트 셀"
             "(마감 ×0.5)이 핵심 — ρ_vessel 이 무는 곳.", "",
             "| 셀 | arm | 평균대기(분) | P95 | 선석초과(분) | 기준재총비용 | 완주 |",
             "|---|---|---|---|---|---|---|"]
    for cell, cv in res["cells"].items():
        for a, s in cv["arms"].items():
            g = "✅" if (s["completion_all1"] and s["backlog_all0"]) else "⚠"
            lines.append(f"| {cell} | {a} | {s['mean_wait']} | {s['p95_wait']} "
                         f"| {s['berth_overrun']} | {s['numeraire_total']} | {g} |")
    lines += ["", "## 재판정 (신 교사 − 구 교사, paired 95% CI)", ""]
    for cell, cv in res["cells"].items():
        b, w = cv["d_berth_num_vs_lex"], cv["d_wait_num_vs_lex"]
        nu = cv["d_numeraire_num_vs_lex"]
        lines.append(
            f"- **{cell}**: Δ선석초과 **{b['mean']}분 [{b['lo']}, {b['hi']}]** · "
            f"Δ평균대기 {w['mean']} [{w['lo']}, {w['hi']}] · Δ기준재 {nu['mean']} "
            f"[{nu['lo']}, {nu['hi']}] · 결정일치(신≈구) {cv['agree_num_lex']} → "
            f"행동변화 {'예' if cv['behavior_changed'] else '아니오'}")
    verdict = "필요 (2b 진행)" if res["VERDICT_RECOLLECT"] else "불필요 (체크포인트 유지 가능)"
    lines += ["", f"**판정: 재수집 {verdict}** — 타이트 셀 기준. 원자료 rows.jsonl."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--agree-seeds", type=int, default=1)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args()
    run(Path(a.out), n_seeds=a.seeds, agree_seeds=a.agree_seeds)
