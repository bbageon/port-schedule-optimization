"""YR-136 0단계 — Ô/F̂ 예측기 오차 측정 + κ 위치-척도 적합·동결 (spec v2.1 프로토콜).

프로토콜 (결과 열람 전 동결 — spec 정본):
- 훈련 대역 4셀 × 시드 4 (BASE+0..3), SF 정책 에피소드.
- 예측 시점: 트럭 = 처음 WAITING 으로 관측된 결정 시점(predict_gate_out) /
  본선 = 개시 후 첫 결정(predict_vessel_completion). 각 1회.
- 오차 = 실현(actual_gate_out · actual_completion_s) − 예측. 미완료 = 검열(집계 제외·수 보고).
- 적합: b = 평균오차(중심 보정), κ = (√3/π)·SD(오차) — 로지스틱 근사. kappa_fit.json 동결.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_curve_v2 import predict_gate_out, predict_vessel_completion
from ..integrated.repro import repro_stamp
from .yr088_joint_rl import LEVEL
from .yr090_dense_vessel import BASE, CELLS, _sim

OUT = Path("outputs/reports/yr136_softplus_contract")
FIT_EPS = [(c, BASE[c] + i) for c in CELLS for i in range(4)]


def _episode_errors(cell: str, seed: int) -> tuple[list, list, int, int]:
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    pred_t: dict[str, float] = {}
    pred_v: dict[str, float] = {}
    dp = sim.run_until_decision()
    while dp is not None:
        for jid, j in sim.jobs.items():
            if (jid not in pred_t and getattr(j, "is_external_truck", False)
                    and j.status.name == "WAITING"):
                o = predict_gate_out(sim, jid)
                if o is not None:
                    pred_t[jid] = o
        for vid, v in sim.vessels.items():
            if vid not in pred_v and getattr(v, "started", False) and not v.done:
                f = predict_vessel_completion(sim, v)
                if f is not None:
                    pred_v[vid] = f
        gb = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        _apply(sim, pol.decide(sim, dp, gb))
        dp = sim.run_until_decision()
    et, ev, cen_t, cen_v = [], [], 0, 0
    for jid, o_hat in pred_t.items():
        o = getattr(sim.jobs.get(jid), "actual_gate_out", None)
        if o is None:
            cen_t += 1
        else:
            et.append(o - o_hat)
    for vid, f_hat in pred_v.items():
        # 사후 정산 전용 truth (GROUND_TRUTH) — 예측에는 미사용, 오차 측정에만 열람
        f = getattr(getattr(sim.vessels[vid], "truth", None), "actual_completion_s", None)
        if f is None:
            cen_v += 1
        else:
            ev.append(f - f_hat)
    return et, ev, cen_t, cen_v


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    et, ev, cen_t, cen_v = [], [], 0, 0
    for cell, seed in FIT_EPS:
        a, b, ct, cv = _episode_errors(cell, seed)
        et += a; ev += b; cen_t += ct; cen_v += cv
        print(f"[{cell}:{seed}] 트럭 {len(a)}(검열 {ct}) 본선 {len(b)}(검열 {cv})",
              flush=True)
    assert et and ev, f"오차 표본 부족: 트럭 {len(et)} · 본선 {len(ev)} (검열 {cen_t}/{cen_v})"
    k = math.sqrt(3.0) / math.pi
    fit = {"kappa_t_s": round(k * pstdev(et), 1), "bias_t_s": round(fmean(et), 1),
           "kappa_v_s": round(k * pstdev(ev), 1), "bias_v_s": round(fmean(ev), 1),
           "n_truck": len(et), "n_vessel": len(ev),
           "censored_truck": cen_t, "censored_vessel": cen_v,
           "sd_truck_s": round(pstdev(et), 1), "sd_vessel_s": round(pstdev(ev), 1),
           "protocol": "SF 정책·4셀×4시드·트럭 첫 WAITING/본선 개시 후 첫 결정·"
                       "b=mean, κ=(√3/π)·SD — spec v2.1 동결"}
    res = {"repro": repro_stamp(
               experiment="YR-136 0단계 — 예측기 오차 측정 + κ 적합 (softplus v2.1)",
               seeds={c: [BASE[c] + i for i in range(4)] for c in CELLS},
               profile_id="calibrated",
               prereg="프로토콜 사전 동결(예측 시점·오차 정의·적합식). 적합값은 이 런으로 "
                      "동결 — 같은 판정런 내 κ·b 조정 금지. 진단·개발 — 판정 아님.",
               extra={"n_episodes": len(FIT_EPS)}),
           **fit}
    (OUT / "kappa_fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(fit, ensure_ascii=False))
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("DONE")
