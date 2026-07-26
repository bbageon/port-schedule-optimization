"""YR-075-c — 재조작 목적지 K후보·30분 국소 rollout 헤드룸 (H1 잔여격차 재판정).

■ 사전등록 (2026-07-26, 결과 미열람 상태에서 동결)
- 물음: 검사한 K후보·30분 지평·조건 안에서 목적지 결정에 H1 을 넘는 상금이 있는가.
- arm (중앙정책 = JointRolloutGreedy@1800·forbid_wait·numeraire_v1 동결, **목적지 규칙만** 상이):
  greedy(find_slot) · H1(deployable_future_selector, 채택 규칙) · KROLL(K=6 후보 × 30분
  국소 rollout argmin — deepcopy=완벽 미래 사용, **낙관적 국소 benchmark 전용·배포 금지**).
- 셀 3: mid(56대)·high(80대)·high-fill0.65(고적재 회귀) — 전부 time_contract_v2=True
  (감사 2차: 트럭 학습비용 = 블록 처리시간 B−C). seed = base+500+i, i<5, paired.
- **주지표: 총비용(numeraire_v1) paired CI. KROLL−H1 상한 < 0 이면 "국소 상금 존재"**
  (그때만 관측기반 규칙/학습 개발). 아니고 guard(완주 1.0·backlog 0·위반 0) 통과면
  "검사한 K=6·30분·조건에서 H1 충분" 판정. 부지표(보고만): B−C·berth·rehandles.
- 물리: YR-091(비통과)·YR-092(pile 규격) 정정 후 실행 (spec 선결 충족, `43e25da`).

■ 목적지-as-action 계약
KROLL 은 결정에서 중앙정책이 고른 공동행동의 **첫 재조작 blocker** 에 대해 K 슬롯 후보를
H1 사전식 키 순으로 뽑고, 각 후보를 "강제 selector"(해당 blocker 1회 소비, 이후 H1 폴백)로
심은 30분 rollout 비용 argmin 을 고른다. 실행은 라이브 sim 에 같은 강제를 심어 assign 의
_plan 이 소비 — 계획·실행 동일 경로(후조건은 engine seam 관습대로 selector 내부 검증).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path
from statistics import fmean, stdev

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (JointRolloutGreedy, _rollout_cost, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.profiles import build_calibrated_profile
from ..integrated.rehandle_oracle import _oft, deployable_future_selector
from ..integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario
from ..sim.constraints import ConstraintViolation
from ..sim.travel_time import gantry_m, trolley_m

RC = RewardCalculator.numeraire_v1()
K, HORIZON_S = 6, 1800.0
CELLS = {
    "mid": (lambda: calibrated_load_params("mid"), 830000),
    "high": (lambda: calibrated_load_params("high"), 830100),
    "high-fill65": (lambda: dataclasses.replace(calibrated_load_params("high"),
                                                fill_ratio=0.65), 831500),
}
N_SEEDS = 5
OUT = Path("outputs/reports/yr075c_rollout_headroom")
_TC = {4: 2.776, 14: 2.145}


def _sim(cell: str, seed: int) -> TerminalSimulator:
    prof = build_calibrated_profile()
    params = dataclasses.replace(CELLS[cell][0](), time_contract_v2=True)
    s = TerminalSimulator(prof, generate_terminal_scenario(prof, seed, params),
                          check_invariants=True)
    s.info_level = InformationLevel.PRE_ADVICE
    return s


# ---------------------------------------------------------- 목적지 후보·강제 selector
def k_dest_candidates(sim, stk, blocker, spec, exclude, k: int = K) -> list[tuple[int, int]]:
    """H1 사전식 키(future_blocked, 즉시비용, bay, row) 순 상위 K 슬롯 (find_slot 동일 유효성)."""
    times = _oft(sim)
    geom = stk.geom
    b_time = times.get(blocker.container_id, float("inf"))
    scored = []
    for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
        for row in range(1, geom.row_count + 1):
            if (bay, row) in exclude:
                continue
            top = stk.top_tier(bay, row)
            if top >= geom.tier_max or not stk.stack_size_ok(bay, row, blocker.size):
                continue
            fb = sum(1 for cid in stk.stack(bay, row)
                     if times.get(cid, float("inf")) < b_time)
            cost = (gantry_m(geom, float(blocker.bay), bay)
                    + trolley_m(geom, float(blocker.row), row) + top * geom.tier_height_m)
            scored.append((fb, cost, bay, row))
    scored.sort()
    return [(b, r) for _, _, b, r in scored[:k]]


def forced_then_h1_selector(sim, stk, blocker, spec, exclude):
    """sim._yr075c_force[blocker_id] 가 있으면 1회 소비(유효성 검증 후) — 아니면 H1."""
    force = getattr(sim, "_yr075c_force", None)
    if force and blocker.container_id in force:
        bay, row = force.pop(blocker.container_id)
        if ((bay, row) not in exclude
                and spec.service_bay_min <= bay <= spec.service_bay_max
                and stk.top_tier(bay, row) < stk.geom.tier_max
                and stk.stack_size_ok(bay, row, blocker.size)):
            return (bay, row)
    return deployable_future_selector(sim, stk, blocker, spec, exclude)


class KDestRollout:
    """중앙 JR 이 고른 공동행동의 첫 재조작 blocker 목적지를 K후보 30분 rollout argmin 으로."""

    name = "KROLL"

    def __init__(self):
        self.jr = JointRolloutGreedy(RC, horizon_s=HORIZON_S, generator=CandidateGenerator(),
                                     forbid_strategic_wait=True)
        self.n_branched = 0          # 목적지 분기가 실제 열린 결정 수 (보고 의무)
        self.n_changed = 0           # H1 기본과 다른 목적지가 선택된 횟수

    def _first_blocker(self, sim, assign):
        """공동행동에서 재조작이 있는 첫 (crane, blocker) — crane_id 정렬 결정론."""
        for cid in sorted(assign):
            a = assign[cid]
            ref = getattr(a, "job_ref", None)
            tgt = getattr(ref, "target_container", None) if ref is not None else None
            if tgt is None:
                continue
            blockers = sim.stacks.blockers_above(tgt)
            if blockers:
                return cid, sim.stacks.containers[blockers[0]], ref
        return None

    def decide(self, sim, dp, gen_by):
        assign = self.jr.decide(sim, dp, gen_by)
        hit = self._first_blocker(sim, assign)
        if hit is None:
            return assign
        cid, blocker, ref = hit
        spec = sim.fleet.spec(cid)
        excl = frozenset({(blocker.bay, blocker.row)})
        cands = k_dest_candidates(sim, sim.stacks, blocker, spec, excl)
        if len(cands) <= 1:
            return assign
        self.n_branched += 1
        best = None
        for dest in cands:                        # 각 목적지 → 30분 국소 rollout 비용
            branch = copy.deepcopy(sim)
            branch.slot_selector = forced_then_h1_selector
            branch._yr075c_force = {blocker.container_id: dest}
            try:
                cost, _ = _rollout_cost(branch, assign, RC, horizon_s=HORIZON_S,
                                        base_policy=self.jr.base_policy,
                                        generator=self.jr.generator)
            except ConstraintViolation:
                # 강제 목적지가 corridor 를 바꿔 공동행동이 비실행(예: 비통과 간섭) —
                # 그 목적지는 이 공동행동과 물리적으로 양립 불가 → 후보 제외 (YR-091 물리).
                continue
            key = (round(cost, 9), dest)
            if best is None or key < best:
                best = key
        if best is None:                          # 전 후보 비실행 → H1 기본 경로 그대로
            return assign
        chosen = best[1]
        if chosen != cands[0]:                    # cands[0] == H1 선택 (동일 사전식 키)
            self.n_changed += 1
        sim._yr075c_force = {blocker.container_id: chosen}   # 실행 경로에 1회 강제
        return assign


def _mk_policy(arm: str):
    jr = JointRolloutGreedy(RC, horizon_s=HORIZON_S, generator=CandidateGenerator(),
                            forbid_strategic_wait=True)
    return KDestRollout() if arm == "KROLL" else jr


def _install(sim, arm):
    if arm == "H1":
        sim.slot_selector = deployable_future_selector
    elif arm == "KROLL":
        sim.slot_selector = forced_then_h1_selector    # force 비면 H1 동작 (계획·실행 동일)


def ci(d):
    m, sd, n = fmean(d), stdev(d), len(d)
    se = sd / n ** 0.5
    t = _TC.get(n - 1, 2.0)
    return round(m, 3), round(m - t * se, 3), round(m + t * se, 3)


def run(n_seeds: int = N_SEEDS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = ("greedy", "H1", "KROLL")
    rows: dict = {a: {} for a in arms}
    for cell, (_, base) in CELLS.items():
        for i in range(n_seeds):
            seed = base + 500 + i
            for arm in arms:
                sim = _sim(cell, seed)
                _install(sim, arm)
                pol = _mk_policy(arm)
                t0 = time.perf_counter()
                r = run_joint_episode(sim, pol, RC, generator=CandidateGenerator())
                rows[arm][f"{cell}/{seed}"] = {
                    "total_cost": r["total_cost"], "block_turntime_mean_min":
                        r.get("block_turntime_mean_min"), "berth_overrun_min":
                        r["berth_overrun_min"], "rehandles": r["rehandles"],
                    "completion_rate": r["completion_rate"], "backlog": r["backlog"],
                    "wall_s": round(time.perf_counter() - t0, 1),
                    "n_branched": getattr(pol, "n_branched", None),
                    "n_changed": getattr(pol, "n_changed", None)}
            print(f"[{cell}/{seed}] " + " | ".join(
                f"{a} {rows[a][f'{cell}/{seed}']['total_cost']:.2f}" for a in arms), flush=True)

    def col(a, k):
        return [rows[a][kk][k] for kk in sorted(rows[a])]

    verdict: dict = {"K": K, "horizon_s": HORIZON_S, "cells": list(CELLS), "n_seeds": n_seeds}
    print("\n=== paired CI (음수=앞 arm 우세) — 주지표 총비용 ===")
    for a, b in (("KROLL", "H1"), ("KROLL", "greedy"), ("H1", "greedy")):
        d = [x - y for x, y in zip(col(a, "total_cost"), col(b, "total_cost"))]
        c = ci(d)
        verdict[f"{a}_vs_{b}_total_cost"] = c
        print(f"  {a}−{b}: {c} {'유의' if c[2] < 0 else '무의' if c[1] <= 0 else '열세'}")
    for k in ("block_turntime_mean_min", "berth_overrun_min", "rehandles"):
        d = [x - y for x, y in zip(col("KROLL", k), col("H1", k))]
        verdict[f"KROLL_vs_H1_{k}"] = ci(d)
        print(f"  KROLL−H1 {k}: {ci(d)}")
    guards = {a: {"compl_min": min(col(a, "completion_rate")),
                  "backlog_max": max(col(a, "backlog"))} for a in arms}
    verdict["guards"] = guards
    branched = sum(v["n_branched"] or 0 for v in rows["KROLL"].values())
    changed = sum(v["n_changed"] or 0 for v in rows["KROLL"].values())
    verdict["kroll_branched"] = branched
    verdict["kroll_changed"] = changed
    print(f"  guards: {guards}")
    print(f"  KROLL 분기 결정 {branched}건 중 H1 과 다른 목적지 선택 {changed}건")
    (OUT / "results.json").write_text(
        json.dumps({"verdict": verdict, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"saved {OUT/'results.json'}")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    run(a.seeds)
    print("YR075C DONE")
