"""YR-128 — 가치 순위 진단: Q 의 행동 순위가 반사실 rollout 비용 순위와 일치하는가 (무학습).

■ 왜 (외부 피드백 2026-07-29 — 차분 이식 전 선행 진단)
YR-127 이 남긴 정확한 결론은 "현 조합(상태·후보·귀속·TD)에서 Q 가 WAIT 를 더 싸게
평가한다"까지다 — 행동 선택에 필요한 것은 절대 비용 예측이 아니라 **행동 간 미세한
차이의 부호·순위**이고, 그 결함 여부는 아직 직접 측정된 적이 없다. 차분 신용(YR-125
2단계)이 고칠 문제가 실제로 "가치 순위 오류"인지 먼저 확인한다.

■ 방법
기존 체크포인트(BUDGET20 ×3 = 보정 최선 arm, WAITON ×3 = 대조)로 YR-125 와 같은
8 에피소드를 argmin(Q) 실행하며, 매 결정에서 **net 이 채점한 조합 목록 그대로**
(build_rows 의 assigns) 각 조합의 반사실 rollout 비용 C600 을 잰다:
`_rollout_cost(600s 창, RC_TRAIN, base=SF)` — YR-063 의 C_600s 계보.
⚠ C600 은 오라클(미래정보, YR-107)이므로 **진단 특권**이며 정책 입력·배포 불가.
⚠ C600 자체도 근사 진실이다: 600s 창 밖 효과는 못 본다 (YR-065 창 비단조 이력 고지).

■ 지표 (결과 열람 전 동결)
  R1 top-1 일치율: argmin Q == argmin C600 인 결정 비율 (조합 ≥2).
  R2 순위 상관: 결정별 Spearman ρ(Q, C600) 평균 (조합 ≥3, 퇴화 제외).
  R3 WAIT 방향 혼동표: full(전원 실작업)·wait 조합 공존 결정에서
     qw = [min Q(wait) < min Q(full)], cw = [min C600(wait) < min C600(full)] 의 2×2.
     핵심 셀 = qw ∧ ¬cw ("Q 만 wait 이 싸다고 봄" = WAIT 과소평가).
     P(cw | qw) = Q 의 wait 선호가 반사실로도 확인되는 비율.

■ 해석 규칙 (동결)
  H-순위결함: P(cw|qw) < 0.5 (wait 선호의 과반이 오평가) → 차분(부호·순위 표적) 직접 근거.
  H-실제선호: P(cw|qw) ≥ 0.7 이고 R1 이 대조 대비 높음 → 600s 창 기준 WAIT 이 실제로
     쌈 — 결함은 순위가 아니라 **목표 정의**(창·할인·shaping vs 평가) → 차분도 같은
     반사실 창을 쓰므로 착수 전 목표 회계 재검이 선행되어야 함.
  중간(0.5~0.7): 혼재 — 차분 착수는 유지하되 목표 회계 재검을 병행 조건으로.
  진단이며 판정·채택 아님. 표본 = 열람된 진단 대역 재사용 (YR-125 1단계와 같은 지위).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import torch

from ..contract.schema import CandidateKind
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _rollout_cost,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from . import yr088_joint_rl as y88
from . import yr090_dense_vessel as y90
from .yr088_joint_rl import LEVEL, RC as RC_TRAIN, build_rows
from .yr090_dense_vessel import BASE, CELLS, _sim
from .yr119_wait_retrain import _set_forbid_wait

OUT = Path("outputs/reports/yr128_rank_diagnosis")
ARMS = {"BUDGET20": "outputs/reports/yr127_training_budget/budget20_s{ts}",
        "WAITON": "outputs/reports/yr119_wait_retrain/waiton_s{ts}"}
TRAIN_SEEDS = (88_000, 99_000, 123_000)
EPS = [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)]
HORIZON_S = 600.0


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(a: list[float], b: list[float]) -> float | None:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = fmean(ra), fmean(rb)
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if da == 0 or db == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (da * db)


def _episode_ranks(net, norm, cell, seed) -> list[dict]:
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=False)
    recs = []
    dp = sim.run_until_decision()
    k = 0
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            with torch.no_grad():
                sc, _ = net(torch.tensor(rows, dtype=torch.float32))
            q = [float(x) for x in sc]
            base = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
            c600 = [
                _rollout_cost(sim, a, RC_TRAIN, horizon_s=HORIZON_S,
                              base_policy=base, generator=gen)[0]
                for a in assigns]
            wait_mask = [any(a[c].kind == CandidateKind.WAIT for c in dp.crane_ids)
                         for a in assigns]
            rec = {"n": len(assigns),
                   "top1": q.index(min(q)) == c600.index(min(c600))}
            if len(assigns) >= 3:
                rec["rho"] = _spearman(q, c600)
            full_q = [x for x, w in zip(q, wait_mask) if not w]
            wait_q = [x for x, w in zip(q, wait_mask) if w]
            full_c = [x for x, w in zip(c600, wait_mask) if not w]
            wait_c = [x for x, w in zip(c600, wait_mask) if w]
            if full_q and wait_q:
                rec["qw"] = min(wait_q) < min(full_q)
                rec["cw"] = min(wait_c) < min(full_c)
            recs.append(rec)
            _apply(sim, assigns[q.index(min(q))])
        dp = sim.run_until_decision()
        k += 1
    return recs


def _summ(recs: list[dict]) -> dict:
    rhos = [r["rho"] for r in recs if r.get("rho") is not None]
    mix = [r for r in recs if "qw" in r]
    both = sum(1 for r in mix if r["qw"] and r["cw"])
    q_only = sum(1 for r in mix if r["qw"] and not r["cw"])
    c_only = sum(1 for r in mix if not r["qw"] and r["cw"])
    neither = sum(1 for r in mix if not r["qw"] and not r["cw"])
    n_qw = both + q_only
    return {"n_decisions": len(recs),
            "R1_top1_agree": round(fmean(1.0 if r["top1"] else 0.0 for r in recs), 4),
            "R2_spearman_mean": round(fmean(rhos), 4) if rhos else None,
            "R3_confusion": {"both_wait": both, "q_only": q_only,
                             "c_only": c_only, "neither": neither},
            "R3_p_cw_given_qw": round(both / n_qw, 4) if n_qw else None,
            "R3_n_mixed": len(mix)}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    _set_forbid_wait(False)
    try:
        for arm, pat in ARMS.items():
            for ts in TRAIN_SEEDS:
                ck = torch.load(Path(pat.format(ts=ts)) / "rl_net.pt", map_location="cpu")
                net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
                norm = StateNorm(refs=ck["norm_refs"])
                recs = []
                for cell, seed in EPS:
                    recs += _episode_ranks(net, norm, cell, seed)
                summary[f"{arm}:{ts}"] = _summ(recs)
                print(f"[{arm} s{ts}] {json.dumps(summary[f'{arm}:{ts}'])}", flush=True)
    finally:
        _set_forbid_wait(True)
    res = {"repro": repro_stamp(
               experiment="YR-128 가치 순위 진단 — Q vs 반사실 C600 순위 (무학습)",
               seeds={"train": list(TRAIN_SEEDS), "eval": sorted({s for _, s in EPS})},
               profile_id="calibrated",
               prereg="지표 R1 top-1 일치·R2 Spearman·R3 WAIT 방향 혼동표(P(cw|qw)) 와 "
                      "해석 규칙(H-순위결함 <0.5 / H-실제선호 ≥0.7 / 중간 혼재) 을 결과 "
                      "열람 전 동결. C600 = 600s 오라클 반사실(YR-107 진단 특권·YR-065 "
                      "창 한계 고지). 진단 — 판정·채택 아님.",
               extra={"horizon_s": HORIZON_S, "arms": list(ARMS)}),
           "summary": summary}
    (OUT / "diagnosis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("DONE")
