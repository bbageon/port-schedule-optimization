"""YR-118 — 학습 정책이 건전성 검사에 전부 걸린 원인 규명 (학습 트랙의 진짜 병목).

■ 왜
YR-100-[3] 에서 학습 arm **6개(CALC×3·CONTROL×3)가 전부** `assert_healthy_action_mix` 에
걸려 탈락했다. 그 결과 guard 를 통과한 arm 이 **오라클(JR1800) 하나뿐**이 되어, 사전등록
규칙이 오라클을 배포 후보로 자동 지명하는 결손까지 유발했다(YR-107).
**이게 안 풀리면 학습 arm 을 판정에 올릴 수 없다** — 지금까지의 이득이 전부 규칙에서
나온 구조적 이유다.

■ 문제: 어느 조건에 걸렸는지도 모른다
`arm_*.json` 에는 `healthy` bool 만 있고 `action_mix` 가 없다. 검사는 두 조건인데:
  ① `serve_when_available < 0.25` — 실작업이 가능한데도 거의 안 고름
  ② 단일 비-SERVE 행동이 전체의 60% 초과 — 한 행동이 장악
**둘 중 어느 쪽인지, 어떤 행동이 장악했는지 기록이 없다.** 그래서 재실행으로 남긴다.

■ 이 파일이 하는 일 (판정이 아니라 **진단**)
같은 셀·같은 시드에서 학습 arm 과 규칙 arm 을 돌려 **행동분포 전체**를 저장한다.
- 실패 조건 특정 (①인가 ②인가, ②면 어느 행동인가)
- 규칙(SF)·오라클(JR1800) 대비 분포 비교 — "학습만의 문제"인지 "하네스 공통"인지 분리
- 셀별·시드별 분해 — 전 조건에서 실패인지 특정 조건에서만인지

**주의**: 여기서 문턱을 바꾸지 않는다. 문턱 조정은 원인을 안 뒤의 별개 결정이다
(퇴화 검출기를 느슨하게 해서 통과시키는 것은 문제를 없애는 게 아니라 가리는 것이다).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..integrated.baselines import (ActionMixError, JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, assert_healthy_action_mix,
                                    is_deployable, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.repro import repro_stamp
from .yr090_dense_vessel import BASE, CELLS, _sim
from .yr100_candidate_eval import RC_EVAL, _policy

OUT = Path("outputs/reports/yr118_healthy_diagnosis")
# YR-100-[3] 이 쓴 것과 **같은 평가대역**(BASE+700..705) — 실패를 그대로 재현해야 한다.
SEEDS = {c: [BASE[c] + 700 + i for i in range(6)] for c in CELLS}
ARMS_LEARNED = ["CALC:88000", "CALC:99000", "CALC:123000",
                "CONTROL:88000", "CONTROL:99000", "CONTROL:123000"]
ARMS_REF = ["SF", "JR1800"]


def _diagnose(cell: str, seed: int, arm: str) -> dict:
    """1 에피소드 실행 → 행동분포 전체 + 실패 조건 라벨."""
    r = run_joint_episode(_sim(cell, seed), _policy(arm)(), RC_EVAL,
                          generator=CandidateGenerator())
    mix = r["_mix"]
    d = mix.as_dict()
    reason = None
    try:
        assert_healthy_action_mix(mix, label=f"{arm}/{cell}/s{seed}")
    except ActionMixError as e:
        reason = str(e)
    # 실패 조건을 **기계로** 분류한다 (문장 파싱이 아니라 값 재계산)
    swa = d["serve_when_available"]
    dominant = max(((k, v) for k, v in d["counts"].items() if k != "SERVE"),
                   key=lambda kv: kv[1], default=(None, 0))
    dom_share = dominant[1] / d["total"] if d["total"] else 0.0
    cond = []
    if swa < 0.25:
        cond.append("①SERVE선택률<0.25")
    if dom_share > 0.60:
        cond.append(f"②{dominant[0]}장악>{0.60:.0%}")
    return {"arm": arm, "cell": cell, "seed": seed,
            "healthy": reason is None, "fail_conditions": cond,
            "serve_when_available": swa,
            "dominant_nonserve": dominant[0], "dominant_share": round(dom_share, 4),
            "shares": d["shares"], "counts": d["counts"], "n_decisions": d["total"],
            "serve_available": d["serve_available"], "serve_taken": d["serve_taken"],
            "total_cost": round(r["total_cost"], 3),
            "completion_rate": r["completion_rate"], "backlog": r["backlog"],
            "deployable": is_deployable(arm), "reason": reason}


def run(arms: list[str], cells: list[str] | None = None, n_seeds: int = 6) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cells or list(CELLS)
    rows = []
    for arm in arms:
        for cell in cells:
            for seed in SEEDS[cell][:n_seeds]:
                row = _diagnose(cell, seed, arm)
                rows.append(row)
                print(f"[{arm:14s} {cell:10s} s{seed}] healthy={row['healthy']} "
                      f"SERVE선택률={row['serve_when_available']:.3f} "
                      f"최다비SERVE={row['dominant_nonserve']}({row['dominant_share']:.2f}) "
                      f"조건={row['fail_conditions']}", flush=True)
    res = {"repro": repro_stamp(
               experiment="YR-118 학습 정책 건전성 실패 진단",
               seeds={c: SEEDS[c][:n_seeds] for c in cells},
               profile_id="calibrated",
               prereg="판정 아님 — 진단. 문턱을 바꾸지 않고 실패 조건만 특정한다.",
               extra={"arms": arms, "cells": cells}),
           "rows": rows, "summary": _summarize(rows)}
    (OUT / "diagnosis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    _print_summary(res["summary"])
    return res


def _summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        n_bad = sum(1 for r in sub if not r["healthy"])
        conds: dict[str, int] = {}
        for r in sub:
            for c in r["fail_conditions"]:
                conds[c] = conds.get(c, 0) + 1
        swa = [r["serve_when_available"] for r in sub]
        doms: dict[str, int] = {}
        for r in sub:
            if r["dominant_nonserve"]:
                doms[r["dominant_nonserve"]] = doms.get(r["dominant_nonserve"], 0) + 1
        out[arm] = {"n": len(sub), "n_unhealthy": n_bad,
                    "serve_when_available": {"min": round(min(swa), 4),
                                             "mean": round(sum(swa) / len(swa), 4),
                                             "max": round(max(swa), 4)},
                    "fail_condition_counts": conds,
                    "dominant_nonserve_counts": doms,
                    "mean_total_cost": round(sum(r["total_cost"] for r in sub) / len(sub), 3),
                    "min_completion": min(r["completion_rate"] for r in sub)}
    return out


def _print_summary(s: dict) -> None:
    print("\n=== YR-118 진단 요약 ===")
    print(f"{'arm':16s} {'미건전':>7s} {'SERVE선택률(min/평균)':>22s}  실패조건 / 최다 비-SERVE")
    for arm, v in s.items():
        w = v["serve_when_available"]
        print(f"{arm:16s} {v['n_unhealthy']:3d}/{v['n']:<3d} "
              f"{w['min']:10.3f}/{w['mean']:<10.3f}  "
              f"{v['fail_condition_counts']} / {v['dominant_nonserve_counts']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all", help="all | learned | ref | 쉼표구분")
    ap.add_argument("--cells", default="", help="쉼표구분 (기본 전체)")
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    arms = {"all": ARMS_REF + ARMS_LEARNED, "learned": ARMS_LEARNED,
            "ref": ARMS_REF}.get(a.arms) or a.arms.split(",")
    run(arms, a.cells.split(",") if a.cells else None, a.seeds)
    print("DONE")
