"""YR-087 후속 — 선택적 rollout 배포 정책 (룩어헤드 유지, 평균 저비용).

2b/train-fit 결론: feed-forward 학생은 교사의 rollout 판별을 amortize 못함(관측별칭).
→ 배포 정책이 **직접 짧게라도 돌려봐야** 한다. 단 매 결정 rollout 은 느리다.
착안: 대부분 결정은 뻔함(대기 트럭 서비스). **본선이 급한 순간만 rollout 을 켠다**
(vessel slack < 임계 or STS 굶주림) → 평균 저비용 + 결정적 순간 판별력 회복.

측정: 셀별 berth·트럭 P95·완주 + **rollout 발동률**(비용 대리) + 결정당 벽시계.
대조: SF(rollout 0) · TEACHER(전 결정 rollout, 최상 본선·최고비용) · SELECTIVE(중간).
문헌 보정 시뮬. 결정론(트리거는 관측 slack/STS 만 사용).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean, stdev

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE
HORIZON = 1_800.0
# rollout 발동 시 쓰는 목적 = 순수 기준재(이동0, ρ_vessel 33 고원) — 본선 보호형
_ROLLOUT_OVERRIDES = {"crane_travel": 0.0, "empty_travel": 0.0}
CELLS = {"mid-loose": ("mid", 2.0), "high-loose": ("high", 2.0),
         "mid-tight": ("mid", 0.5), "high-tight": ("high", 0.5)}
BASE = {"mid-loose": 820000, "high-loose": 820100, "mid-tight": 820200, "high-tight": 820300}
RC_LOG = RewardCalculator.numeraire_v1()


class SelectiveRolloutPolicy:
    """본선 급할 때만 rollout, 평소 SF. slack < trigger_s 또는 STS 굶주림에 발동."""

    def __init__(self, trigger_s: float = 600.0, horizon: float = HORIZON):
        self.rollout = JointRolloutGreedy(RewardCalculator.numeraire(dict(_ROLLOUT_OVERRIDES)),
                                          horizon_s=horizon, generator=CandidateGenerator(),
                                          objective=None)
        self.cheap = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
        self.trigger_s = trigger_s
        self.name = f"SELECTIVE@{int(trigger_s)}"
        self.n_rollout = 0
        self.n_total = 0

    def _vessel_at_risk(self, sim) -> bool:
        now = sim.now
        for v in sim.vessels.values():
            if v.done:
                continue
            s = v.slack_s(now)                     # 계획완료 존재 시(양하 RISK·적하 deadline) 유효
            if s is None:
                continue
            if v.sts_blocked or s < self.trigger_s:
                return True
        return False

    def decide(self, sim, dp, gen_by) -> dict:
        self.n_total += 1
        if self._vessel_at_risk(sim):
            self.n_rollout += 1
            return self.rollout.decide(sim, dp, gen_by)
        return self.cheap.decide(sim, dp, gen_by)


def _sim(cell: str, seed: int):
    level, dmult = CELLS[cell]
    prof = build_calibrated_profile()
    s = TerminalSimulator(prof, generate_terminal_scenario(
        prof, seed, calibrated_load_params(level, vessel_deadline_mult=dmult)),
        check_invariants=True)
    s.info_level = LEVEL
    return s


def make(arm: str):
    if arm == "SF":
        return ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    if arm == "TEACHER":
        p = JointRolloutGreedy(RewardCalculator.numeraire(dict(_ROLLOUT_OVERRIDES)),
                               horizon_s=HORIZON, generator=CandidateGenerator(), objective=None)
        p.name = "TEACHER"
        return p
    if arm == "TEACHER_STS5":       # 강한 본선 config (sts 선행신호) @1800
        p = JointRolloutGreedy(RewardCalculator.numeraire({**_ROLLOUT_OVERRIDES, "sts_wait": 5.0}),
                               horizon_s=HORIZON, generator=CandidateGenerator(), objective=None)
        p.name = arm
        return p
    if arm == "TEACHER_H3600":      # 강한 본선 config (긴 창) @3600
        p = JointRolloutGreedy(RewardCalculator.numeraire(dict(_ROLLOUT_OVERRIDES)),
                               horizon_s=3600.0, generator=CandidateGenerator(), objective=None)
        p.name = arm
        return p
    if arm == "TEACHER_STS5_H3600":  # 두 레버 모두 (sts 선행신호 + 긴 창) @3600
        p = JointRolloutGreedy(RewardCalculator.numeraire({**_ROLLOUT_OVERRIDES, "sts_wait": 5.0}),
                               horizon_s=3600.0, generator=CandidateGenerator(), objective=None)
        p.name = arm
        return p
    if arm.startswith("SELECTIVE@"):
        return SelectiveRolloutPolicy(trigger_s=float(arm.split("@")[1]))
    raise ValueError(arm)


def eval_one(arm: str, cell: str, seed: int) -> dict:
    pol = make(arm)
    t0 = time.perf_counter()
    row = run_joint_episode(_sim(cell, seed), pol, RC_LOG, generator=CandidateGenerator())
    wall = time.perf_counter() - t0
    healthy = True
    try:
        assert_healthy_action_mix(row["_mix"], label=f"{cell}/{arm}/s{seed}")
    except ActionMixError:
        healthy = False
    frac = (pol.n_rollout / pol.n_total) if isinstance(pol, SelectiveRolloutPolicy) and pol.n_total else (
        1.0 if arm.startswith("TEACHER") else 0.0)
    return {"arm": arm, "cell": cell, "seed": seed, "wall_s": round(wall, 2),
            "rollout_frac": round(frac, 3),
            "berth": round(row["berth_overrun_min"], 3),
            "mean_wait": round(row["mean_wait_min"], 3), "p95": round(row["p95_wait_min"], 3),
            "completion": row["completion_rate"], "backlog": row["backlog"], "healthy": healthy}


def run(out: Path, arms: list, seeds: int, workers: int = 16) -> dict:
    import multiprocessing as mp
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(a, c, BASE[c] + i) for c in CELLS for a in arms for i in range(seeds)]
    with mp.Pool(processes=workers) as pool:
        rows = pool.starmap(eval_one, tasks)
    by = {}
    for r in rows:
        by.setdefault((r["cell"], r["arm"]), []).append(r)

    _TC = {4: 2.776, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
           12: 2.179, 13: 2.160, 14: 2.145, 19: 2.093}

    def agg(cell, arm, k):
        return round(fmean(r[k] for r in by[(cell, arm)]), 3)

    def paired_vs_sf(cell, arm, k):
        """짝지은 차(arm−SF) CI — 표본sd+t (seed 대응)."""
        a = {r["seed"]: r for r in by.get((cell, arm), [])}
        b = {r["seed"]: r for r in by.get((cell, "SF"), [])}
        d = [a[s][k] - b[s][k] for s in sorted(a) if s in b]
        if len(d) < 2:
            return None
        m = fmean(d); sd = stdev(d); n = len(d); se = sd / n ** 0.5
        tc = _TC.get(n - 1, 2.1)
        return {"mean": round(m, 2), "lo": round(m - tc * se, 2), "hi": round(m + tc * se, 2)}

    res = {"cells": {}, "arms": arms, "seeds": seeds, "rows": rows}
    print("\n=== 셀별 짝지은 Δ vs SF (음수=SF보다 개선. B★=배 유의개선, W★=트럭유의개선) ===", flush=True)
    for cell in CELLS:
        res["cells"][cell] = {}
        sf = {k: agg(cell, "SF", k) for k in ("berth", "mean_wait", "p95")} if (cell, "SF") in by else {}
        print(f"[{cell}] SF berth={sf.get('berth')} wait={sf.get('mean_wait')} P95={sf.get('p95')}", flush=True)
        for arm in arms:
            if (cell, arm) not in by:
                continue
            g = {k: agg(cell, arm, k) for k in ("berth", "mean_wait", "p95", "wall_s", "rollout_frac")}
            g["completion_all1"] = all(r["completion"] == 1.0 for r in by[(cell, arm)])
            g["healthy_all"] = all(r["healthy"] for r in by[(cell, arm)])
            g["d_berth"] = paired_vs_sf(cell, arm, "berth")
            g["d_wait"] = paired_vs_sf(cell, arm, "mean_wait")
            g["d_p95"] = paired_vs_sf(cell, arm, "p95")
            res["cells"][cell][arm] = g
            if arm == "SF":
                continue
            b, p = g["d_berth"], g["d_p95"]
            bsig = "B★" if b and b["hi"] < 0 else "  "
            psig = "W★" if p and p["hi"] < 0 else "  "
            print(f"  {arm:16s} Δberth {b['mean']:+7.2f}[{b['lo']:+7.2f},{b['hi']:+7.2f}]{bsig} "
                  f"ΔP95 {p['mean']:+6.2f}[{p['lo']:+6.2f},{p['hi']:+6.2f}]{psig} "
                  f"| roll={g['rollout_frac']:.2f} wall={g['wall_s']:.0f}s", flush=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDONE", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="SF,TEACHER,SELECTIVE@600")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="outputs/reports/yr087_selective")
    a = ap.parse_args()
    run(Path(a.out), [x for x in a.arms.split(",") if x], a.seeds, a.workers)
