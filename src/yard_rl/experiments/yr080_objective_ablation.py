"""YR-080 2단계a — 목적함수 ablation (교사 층위).

스모크+probe 진단: 기준재 v1 을 1800s rollout 교사에 그대로 얹으면 본선을 오히려 더
굶긴다. 원인 2건 — (1) ρ_crane=1.0 이 이동을 트럭대기와 동급으로 만들어 "일단 서비스"
규율 희석·이동시간 이중계상, (2) sts_wait=0 이 창 안 유일한 본선 선행지표(안벽크레인
유휴, 60% 창 발동)를 제거. 이 드라이버는 교정 후보를 같은 셀에서 굴려 **어떤 목적이
트럭·본선을 동시에 지키는지** 실측한다.

목적 후보:
  SF          — 기준정책 (rollout 없음)
  LEX         — 구 사전식 (트럭 절대우선; assumed_default + hierarchy_key)
  NUM_v1      — 기준재 v1 (crane=1, sts=0) = 현행 깨진 것
  NUM_t0      — 이동 이중계상 제거 (crane_travel=empty_travel=0, sts=0)
  NUM_t0_sts1 — 이동0 + 약한 본선 선행지표 (sts_wait=1.0)
  NUM_t0_sts5 — 이동0 + 강한 본선 선행지표 (sts_wait=5.0)

KPI(berth_overrun·트럭대기)는 로깅 RC 와 무관(sim.kpis 직산) — arm 간 직접 비교.
출력: --out 에 JSON 1건 (objective×cell). Workflow 병렬 팬아웃용.
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
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.objectives import hierarchy_key
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE
HORIZON = 1_800.0
RC_LOG = RewardCalculator.numeraire_v1()          # 로깅 공통 자 (총비용 비교용; KPI 는 무관)

CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
BASE_SEED = {"mid-loose": 780000, "high-loose": 780100,
             "mid-tight": 780200, "high-tight": 780300}

_T0 = {"crane_travel": 0.0, "empty_travel": 0.0}


def make_policy(obj: str):
    """objective 이름 파싱. 접미사 '@H' 로 교사 계획창(rollout horizon) 지정 가능
    (예: 'NUM_t0@5400' = 이동0 순수목적·계획창 5400s). 없으면 HORIZON(1800)."""
    name = obj
    horizon = HORIZON
    if "@" in obj:
        obj, hs = obj.split("@", 1)
        horizon = float(hs)
    if obj == "SF":
        return ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    if obj == "LEX":
        p = JointRolloutGreedy(RewardCalculator.assumed_default(), horizon_s=horizon,
                               generator=CandidateGenerator(), objective=hierarchy_key)
    elif obj == "NUM_v1":
        p = JointRolloutGreedy(RewardCalculator.numeraire(), horizon_s=horizon,
                               generator=CandidateGenerator(), objective=None)
    elif obj == "NUM_t0":
        p = JointRolloutGreedy(RewardCalculator.numeraire(dict(_T0)), horizon_s=horizon,
                               generator=CandidateGenerator(), objective=None)
    elif obj.startswith("NUM_t0_sts"):     # NUM_t0_stsN — 이동0 + sts_wait=N (N 실수 허용)
        n = float(obj[len("NUM_t0_sts"):].replace("p", "."))
        p = JointRolloutGreedy(RewardCalculator.numeraire({**_T0, "sts_wait": n}),
                               horizon_s=horizon, generator=CandidateGenerator(), objective=None)
    else:
        raise ValueError(f"미지 objective: {obj}")
    p.name = name
    return p


def _sim(cell: str, seed: int):
    level, dmult = CELLS[cell]
    prof = build_calibrated_profile()
    params = calibrated_load_params(level, vessel_deadline_mult=dmult)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, params),
                          check_invariants=True)
    s.info_level = LEVEL
    return s


def eval_one(obj: str, cell: str, seed: int) -> dict:
    t0 = time.perf_counter()
    row = run_joint_episode(_sim(cell, seed), make_policy(obj), RC_LOG,
                            generator=CandidateGenerator())
    healthy = True
    try:
        assert_healthy_action_mix(row["_mix"], label=f"{cell}/{obj}/s{seed}")
    except ActionMixError:
        healthy = False
    return {"obj": obj, "cell": cell, "seed": seed,
            "wall_s": round(time.perf_counter() - t0, 2),
            "mean_wait": round(row["mean_wait_min"], 4),
            "p95_wait": round(row["p95_wait_min"], 4),
            "berth_overrun": round(row["berth_overrun_min"], 4),
            "numeraire_total": round(row["total_cost"], 4),
            "completion": row["completion_rate"], "backlog": row["backlog"],
            "repo_share": row["action_mix"]["shares"].get("REPOSITION", 0.0),
            "healthy": healthy}


def run(obj: str, cell: str, seeds: int, out: Path) -> dict:
    rows = [eval_one(obj, cell, BASE_SEED[cell] + i) for i in range(seeds)]
    agg = {"obj": obj, "cell": cell, "n": seeds,
           "mean_wait": round(fmean(r["mean_wait"] for r in rows), 4),
           "p95_wait": round(fmean(r["p95_wait"] for r in rows), 4),
           "berth_overrun": round(fmean(r["berth_overrun"] for r in rows), 4),
           "numeraire_total": round(fmean(r["numeraire_total"] for r in rows), 4),
           "completion_all1": all(r["completion"] == 1.0 for r in rows),
           "backlog_all0": all(r["backlog"] == 0 for r in rows),
           "healthy_all": all(r["healthy"] for r in rows),
           "repo_share": round(fmean(r["repo_share"] for r in rows), 4),
           "wall_s": round(sum(r["wall_s"] for r in rows), 1), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{obj}/{cell}] wait={agg['mean_wait']} berth={agg['berth_overrun']} "
          f"num={agg['numeraire_total']} compl={agg['completion_all1']} "
          f"wall={agg['wall_s']}s -> {out}", flush=True)
    return agg


OBJECTIVES = ["SF", "LEX", "NUM_v1", "NUM_t0", "NUM_t0_sts1", "NUM_t0_sts5"]


def _eval_task(t: tuple) -> dict:
    obj, cell, seed = t
    return eval_one(obj, cell, seed)


def psweep(seeds: int, out_dir: Path, objectives: list, cells=None,
           workers: int = 16) -> dict:
    """병렬 sweep (multiprocessing) — 독립 에피소드라 near-linear. 결정론 불변
    (각 (obj,cell,seed) 는 독립 시드). 짝지은 CI(vs LEX) 까지 산출."""
    import multiprocessing as mp
    from statistics import fmean, pstdev
    cells = cells or list(CELLS)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(obj, cell, BASE_SEED[cell] + i)
             for cell in cells for obj in objectives for i in range(seeds)]
    with mp.Pool(processes=workers) as pool:
        rows = pool.map(_eval_task, tasks)
    by: dict = {}
    for r in rows:
        by.setdefault((r["cell"], r["obj"]), {})[r["seed"]] = r

    def agg(cell, obj, key):
        return round(fmean(by[(cell, obj)][s][key] for s in by[(cell, obj)]), 4)

    def paired(cell, obj, key):
        a, b = by[(cell, obj)], by.get((cell, "LEX"), {})
        d = [a[s][key] - b[s][key] for s in sorted(a) if s in b]
        if not d:
            return None
        m = fmean(d); sd = pstdev(d) if len(d) > 1 else 0.0
        se = sd / (len(d) ** 0.5)
        return {"mean": round(m, 3), "lo": round(m - 1.96 * se, 3),
                "hi": round(m + 1.96 * se, 3)}

    grid: dict = {}
    for cell in cells:
        for obj in objectives:
            k = (cell, obj)
            if k not in by:
                continue
            grid.setdefault(cell, {})[obj] = {
                "mean_wait": agg(cell, obj, "mean_wait"),
                "p95_wait": agg(cell, obj, "p95_wait"),
                "berth_overrun": agg(cell, obj, "berth_overrun"),
                "numeraire_total": agg(cell, obj, "numeraire_total"),
                "completion_all1": all(by[k][s]["completion"] == 1.0 for s in by[k]),
                "healthy_all": all(by[k][s]["healthy"] for s in by[k]),
                "repo_share": agg(cell, obj, "repo_share"),
                "d_berth_vs_LEX": paired(cell, obj, "berth_overrun"),
                "d_wait_vs_LEX": paired(cell, obj, "mean_wait")}
    res = {"horizon_s": HORIZON, "n_seeds": seeds, "objectives": objectives,
           "workers": workers, "grid": grid,
           "rows": sorted(rows, key=lambda r: (r["cell"], r["obj"], r["seed"]))}
    (out_dir / "psweep_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 짝지은 CI (vs LEX, 음수=배 덜늦음/트럭 덜기다림) ===", flush=True)
    for cell in cells:
        print(f"[{cell}]", flush=True)
        for obj in objectives:
            g = grid.get(cell, {}).get(obj)
            if not g or obj == "LEX":
                if obj == "LEX" and g:
                    print(f"    LEX(기준)   berth={g['berth_overrun']:7.2f} "
                          f"wait={g['mean_wait']:6.3f}", flush=True)
                continue
            b, w = g["d_berth_vs_LEX"], g["d_wait_vs_LEX"]
            sig = "★" if b and b["hi"] < 0 else " "
            print(f"  {sig} {obj:12s} berth={g['berth_overrun']:7.2f} "
                  f"Δ[{b['lo']:+7.2f},{b['hi']:+7.2f}] wait={g['mean_wait']:6.3f} "
                  f"Δ[{w['lo']:+5.2f},{w['hi']:+5.2f}] compl={g['completion_all1']}", flush=True)
    print("\nPSWEEP DONE", flush=True)
    return res


def sweep(seeds: int, out_dir: Path, cells=None, objectives=None) -> dict:
    """전 (목적×셀) 순차 sweep — CPU 바운드라 코어 경합 회피 위해 단일 프로세스."""
    cells = cells or list(CELLS)
    objectives = objectives or OBJECTIVES
    out_dir.mkdir(parents=True, exist_ok=True)
    grid: dict = {}
    for cell in cells:
        for obj in objectives:
            agg = run(obj, cell, seeds, out_dir / f"{cell}__{obj}.json")
            grid.setdefault(cell, {})[obj] = {k: agg[k] for k in
                ("mean_wait", "p95_wait", "berth_overrun", "numeraire_total",
                 "completion_all1", "backlog_all0", "healthy_all", "repo_share", "wall_s")}
    res = {"horizon_s": HORIZON, "n_seeds": seeds, "grid": grid}
    (out_dir / "sweep_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    # 요약: 각 셀에서 LEX 대비 berth·wait 상대
    print("\n=== 요약 (셀별, LEX 기준 대비) ===", flush=True)
    for cell, arms in grid.items():
        lex = arms.get("LEX", {})
        print(f"[{cell}] LEX berth={lex.get('berth_overrun')} wait={lex.get('mean_wait')}", flush=True)
        for obj in objectives:
            if obj in ("LEX",):
                continue
            a = arms[obj]
            db = round(a["berth_overrun"] - lex.get("berth_overrun", 0.0), 2)
            dw = round(a["mean_wait"] - lex.get("mean_wait", 0.0), 2)
            print(f"    {obj:12s} berth={a['berth_overrun']:8.2f} (Δ{db:+.2f}) "
                  f"wait={a['mean_wait']:7.3f} (Δ{dw:+.2f}) "
                  f"compl={a['completion_all1']}", flush=True)
    print("\nSWEEP DONE", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--psweep", action="store_true")
    ap.add_argument("--objectives", default="")   # 콤마구분 (psweep)
    ap.add_argument("--cells", default="")         # 콤마구분 (psweep, 생략시 전체)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--objective")
    ap.add_argument("--cell")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="outputs/reports/yr080_readjudicate/_ablation")
    a = ap.parse_args()
    if a.psweep:
        objs = [o for o in a.objectives.split(",") if o]
        cls = [c for c in a.cells.split(",") if c] or None
        psweep(a.seeds, Path(a.out), objs, cells=cls, workers=a.workers)
    elif a.sweep:
        sweep(a.seeds, Path(a.out))
    else:
        run(a.objective, a.cell, a.seeds, Path(a.out))
