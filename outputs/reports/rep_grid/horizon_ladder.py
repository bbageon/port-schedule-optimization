"""YR-078 — 내다보기 창(지평) 사다리 (탐색 진단, 사전등록 아님).

질문: 최적 창은 상수인가, 혼잡도의 함수인가. "길수록 무조건 좋은가"(계산비용·수확체감).
Arms: 고정 {600, 1200, 1800, 2400s} + 적응형(대기열 소진시간 추정) + 단순규칙(SF_SPT) 기준.
Cells: 중부하(L80/F65/U — 600s 가 이미 이김: 평평해지는 지점 확인) ·
       포화(L112/F65/U — 600s 가 짐: 어디부터 이기나) · 포화+러시아워(L112/F65/P).
지표: 실제 대기(평균/p95)·완료율·swa + 에피소드 벽시계(창의 계산비용) + 적응형의 창 선택 분포.
공통: NEW 목적·풀조합(max_combos 64)·seed 340000~2 (rep_grid 와 동일 — 직접 비교 가능).
"""
import sys, json, time
sys.path.insert(0, "/mnt/c/Users/geonu/AppData/Local/Temp/claude/"
                   "c--Users-geonu-Desktop-port-reinforcement/"
                   "adbc7e00-3805-4b61-b786-7c6475e2fff8/scratchpad")
from statistics import fmean
from rep_grid import make_sim, RC_NEW, GEN, SEEDS
from yard_rl.domain.enums import JobStatus
from yard_rl.integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                          JointRolloutGreedy, run_joint_episode)

OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/rep_grid/horizon_ladder.json"
AVG_CYCLE_S = 180.0   # 대표 사이클 (물리모델 실측 계산 ~3분)
N_CRANES = 2


class AdaptiveHorizonJR(JointRolloutGreedy):
    """결정 시점마다 창 = clip(대기열 소진시간 추정, 600~2400s).

    소진시간 ≈ (도착해 대기 중인 작업 수 × 평균 사이클) / 크레인 수 — 시뮬레이터가
    아는 값만 사용(진실 미열람). 선택된 창을 기록해 분포 보고.
    """

    name = "JR_ADAPTIVE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chosen_horizons: list[float] = []

    def decide(self, sim, dp, gen_by):
        n_wait = sum(1 for j in sim.jobs.values() if j.status == JobStatus.WAITING)
        drain = n_wait * AVG_CYCLE_S / N_CRANES
        self.horizon_s = float(min(2400.0, max(600.0, drain)))
        self.chosen_horizons.append(self.horizon_s)
        return super().decide(sim, dp, gen_by)


def arms():
    a = {"SF_SPT": lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF")}
    for h in (600, 1200, 1800, 2400):
        a[f"JR_h{h}"] = (lambda hh=h: JointRolloutGreedy(
            RC_NEW, horizon_s=float(hh), max_combos=64, generator=GEN))
    a["JR_ADAPT"] = lambda: AdaptiveHorizonJR(RC_NEW, horizon_s=600.0, max_combos=64,
                                              generator=GEN)
    return a


def episode(fac, seed, n_ext, fill, peaked):
    t0 = time.time()
    pol = fac()
    r = run_joint_episode(make_sim(seed, n_ext, fill, peaked), pol, RC_NEW, generator=GEN)
    row = {"wait": round(r["mean_wait_min"], 3), "p95": round(r["p95_wait_min"], 3),
           "compl": round(r["completion_rate"], 3),
           "swa": round(r["action_mix"]["serve_when_available"], 3),
           "ep_s": round(time.time() - t0, 1)}
    if isinstance(pol, AdaptiveHorizonJR) and pol.chosen_horizons:
        hs = pol.chosen_horizons
        row["h_mean"] = round(fmean(hs), 0)
        row["h_min"], row["h_max"] = min(hs), max(hs)
        row["h_at_cap"] = round(sum(1 for h in hs if h >= 2400.0) / len(hs), 2)
    return row


def main():
    t0 = time.time()
    cells = [("L80/F65/U", 80, 0.65, False),
             ("L112/F65/U", 112, 0.65, False),
             ("L112/F65/P", 112, 0.65, True)]
    out = {"seeds": SEEDS, "avg_cycle_s": AVG_CYCLE_S, "cells": {}}
    for cid, n_ext, fill, peaked in cells:
        print(f"\n===== {cid} =====", flush=True)
        cell = {}
        for name, fac in arms().items():
            rows = [episode(fac, s, n_ext, fill, peaked) for s in SEEDS]
            agg = {k: round(fmean(r[k] for r in rows), 3) for k in
                   ("wait", "p95", "compl", "swa", "ep_s")}
            agg["per_seed"] = rows
            if "h_mean" in rows[0]:
                agg["h_mean"] = round(fmean(r["h_mean"] for r in rows), 0)
                agg["h_at_cap"] = round(fmean(r["h_at_cap"] for r in rows), 2)
            cell[name] = agg
            extra = (f" h_mean={agg.get('h_mean', ''):>5} cap율={agg.get('h_at_cap', '')}"
                     if "h_mean" in agg else "")
            print(f"{name:10s} wait={agg['wait']:>7.2f} p95={agg['p95']:>7.2f} "
                  f"compl={agg['compl']:.3f} swa={agg['swa']:.2f} "
                  f"ep={agg['ep_s']:>6.1f}s{extra}", flush=True)
        out["cells"][cid] = cell
    out["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
