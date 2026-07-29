"""YR-125 1단계 — Q값 진단: 학습된 Q 가 행동을 어떻게 평가하는지 **직접 측정** (무학습).

■ 왜 (3연속 기각 뒤에는 개입보다 측정)
YR-119(허용→WAIT 남용)·121(벌점 무반응)·122(γ=1 악화)가 전부 기각됐다. 다음 개입을
고르기 전에, 기존 체크포인트 12개(WAITOFF/WAITON/DURPEN/GAMMA1 × 시드 3)의 Q 를 열어
**어느 가설이 맞는지** 판별한다. 재학습 없음 — 판정이 아니라 진단이다.

■ 지표 (결과 열람 전 동결)
  M1 선호 격차: full(전원 실작업) 조합과 wait(WAIT 포함) 조합이 공존하는 결정에서
     `gap = min Q(wait) − min Q(full)` (numeraire). gap<0 = 망이 WAIT 쪽을 더 싸게 봄.
     + SERVE 실행가능인데 최저 Q 가 wait 조합인 결정의 비율.
  M2 변별 스케일: 결정별 Q 의 표준편차·범위 (numeraire) — "행동 간 차이가 목표 잡음에
     묻힌다"(YR-122 사후 가설)를 GAMMA1 vs WAITON 비교로 직접 검정.
  M3 보정(calibration): 평가 argmin 실행으로 얻은 실현 SMDP 수익
     `G_k = r_k + γ^Δt·G_{k+1}` (학습과 동일 보상: RC_TRAIN 구간비용+shaping+벌점+UNSERVED,
     arm 고유 γ·벌점) vs 망 예측 `Q̂_k = sc[pick]×SCALE`.
     bias = mean(G−Q̂) 를 **선택에 WAIT 포함/미포함으로 분리** — WAIT 쪽 bias ≫ 0 이면
     "WAIT 의 실제 비용이 Q 에 덜 꽂힌다"(신용/회계 가설).

■ 해석 규칙 (동결)
  H-A 변별 상실: |M1 gap| ≪ M2 spread · M3 MAE 큼(선택 무관) → 차분 신용(2단계) 지지.
  H-B 체계적 오평가: gap 명확히 <0 이고 WAIT 선택의 (G−Q̂) ≫ 비-WAIT → 신용/회계 결함.
  H-C 목표는 잘 맞춤: gap<0 인데 M3 bias≈0 → 학습은 자기 목표에 충실, 목표 정의가 평가와
     불일치 → 보상 회계 재검(차분보다 회계 수정 우선).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, pstdev

import torch

from ..contract.schema import CandidateKind
from ..integrated.baselines import JointRolloutGreedy, _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.encoding import StateNorm
from ..integrated.evalkit import paired
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from . import yr088_joint_rl as y88
from . import yr090_dense_vessel as y90
from .yr088_joint_rl import LEVEL, RC as RC_TRAIN, REF_S, REPO_PENALTY, UNSERVED, build_rows
from .yr090_dense_vessel import BASE, CELLS, SCALE, _sim, graduated_wait_shaping

OUT = Path("outputs/reports/yr125_qvalue_diagnosis")
ARMS = {  # name → (ckpt dir 패턴, forbid_wait, gamma, wait_time_penalty)
    "WAITOFF": ("outputs/reports/yr119_wait_retrain/waitoff_s{ts}", True, 0.99, 0.0),
    "WAITON": ("outputs/reports/yr119_wait_retrain/waiton_s{ts}", False, 0.99, 0.0),
    "DURPEN": ("outputs/reports/yr121_wait_duration_penalty/durpen_s{ts}", False, 0.99, 1.0),
    "GAMMA1": ("outputs/reports/yr122_gamma_alignment/gamma1_s{ts}", False, 1.0, 0.0),
}
TRAIN_SEEDS = (88_000, 99_000, 123_000)
EVAL_EPS = [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)]   # 4셀×2 = 8/net


def _diagnose_episode(net, norm, cell, seed, *, forbid, gamma, wait_pen):
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=forbid)
    decisions, trans = [], []           # trans: [r, gdt, q_hat, had_wait]
    pend = None
    dp = sim.run_until_decision()
    sim.cost.cut()
    last_b = sim.now
    k = 0
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        raw = sim.cost.cut()
        if pend is not None:
            pend["r"] += RC_TRAIN.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                           raw=raw, risk_max=0.0).total_normalized
            pend["r"] += graduated_wait_shaping(sim, sim.now - last_b)
        last_b = sim.now
        if pend is not None and assigns:
            gdt = gamma ** ((sim.now - pend["t_act"]) / REF_S)
            pend["r"] += wait_pen * pend["n_wait"] * (sim.now - pend["t_act"]) / 3600.0
            trans.append([pend["r"], gdt, pend["q_hat"], pend["n_wait"] > 0])
            pend = None
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            with torch.no_grad():
                sc, _ = net(torch.tensor(rows, dtype=torch.float32))
            q = [float(x) for x in sc]
            wait_mask = [any(a[c].kind == CandidateKind.WAIT for c in dp.crane_ids)
                         for a in assigns]
            serve_ok = any(g.feasible and g.kind == CandidateKind.SERVE
                           for c in dp.crane_ids for g in gen_by[c].items)
            rec = {"n": len(assigns),
                   "std": (pstdev(q) if len(q) > 1 else 0.0) * SCALE,
                   "range": (max(q) - min(q)) * SCALE, "serve_ok": serve_ok}
            full = [x for x, wm in zip(q, wait_mask) if not wm]
            wait = [x for x, wm in zip(q, wait_mask) if wm]
            if full and wait:
                rec["gap"] = (min(wait) - min(full)) * SCALE
                rec["best_is_wait"] = min(wait) < min(full)
            decisions.append(rec)
            pick = q.index(min(q))
            n_wait = sum(1 for c in dp.crane_ids
                         if assigns[pick][c].kind == CandidateKind.WAIT)
            n_repo = sum(1 for c in dp.crane_ids
                         if assigns[pick][c].kind == CandidateKind.REPOSITION)
            pend = {"t_act": sim.now, "n_wait": n_wait, "r": REPO_PENALTY * n_repo,
                    "q_hat": float(q[pick]) * SCALE}
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    jobs = list(sim.jobs.values())
    n_unserved = sum(1 for j in jobs if j.status.name != "DONE")
    if pend is not None:
        raw = sim.cost.cut()
        pend["r"] += RC_TRAIN.cost_for(interval_start_s=last_b, interval_end_s=sim.now,
                                       raw=raw, risk_max=0.0).total_normalized
        pend["r"] += graduated_wait_shaping(sim, sim.now - last_b)
        pend["r"] += UNSERVED * n_unserved
        pend["r"] += wait_pen * pend["n_wait"] * (sim.now - pend["t_act"]) / 3600.0
        trans.append([pend["r"], 1.0, pend["q_hat"], pend["n_wait"] > 0])
    # 실현 수익 G (뒤에서 앞으로) 와 보정 오차
    calib = []
    G = 0.0
    for r, gdt, q_hat, had_wait in reversed(trans):
        G = r + gdt * G
        calib.append({"G": G, "q_hat": q_hat, "had_wait": had_wait})
    return decisions, calib


def run(n_eps: int = 2) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [(c, BASE[c] + 700 + i) for c in CELLS for i in range(n_eps)]
    summary: dict = {}
    for arm, (pat, forbid, gamma, pen) in ARMS.items():
        dec_all, cal_all = [], []
        for ts in TRAIN_SEEDS:
            ck = torch.load(Path(pat.format(ts=ts)) / "rl_net.pt", map_location="cpu")
            net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
            norm = StateNorm(refs=ck["norm_refs"])
            y88.FORBID_WAIT = y90.FORBID_WAIT = forbid
            try:
                for cell, seed in eps:
                    d, c = _diagnose_episode(net, norm, cell, seed,
                                             forbid=forbid, gamma=gamma, wait_pen=pen)
                    dec_all += d
                    cal_all += c
            finally:
                y88.FORBID_WAIT = y90.FORBID_WAIT = True
            print(f"[{arm} s{ts}] 결정 {len(dec_all)} 누적", flush=True)
        gaps = [d["gap"] for d in dec_all if "gap" in d]
        biw = [d for d in dec_all if d.get("serve_ok") and "best_is_wait" in d]
        errs = [c["G"] - c["q_hat"] for c in cal_all]
        errs_w = [c["G"] - c["q_hat"] for c in cal_all if c["had_wait"]]
        errs_n = [c["G"] - c["q_hat"] for c in cal_all if not c["had_wait"]]
        summary[arm] = {
            "n_decisions": len(dec_all), "n_gap_decisions": len(gaps),
            "M1_gap": paired(gaps).as_dict() if len(gaps) >= 2 else None,
            "M1_best_is_wait_share": (round(fmean(1.0 if d["best_is_wait"] else 0.0
                                                  for d in biw), 4) if biw else None),
            "M2_q_std_mean": round(fmean(d["std"] for d in dec_all), 3),
            "M2_q_range_mean": round(fmean(d["range"] for d in dec_all), 3),
            "M3_bias_all": round(fmean(errs), 3) if errs else None,
            "M3_bias_wait_chosen": round(fmean(errs_w), 3) if errs_w else None,
            "M3_bias_nonwait": round(fmean(errs_n), 3) if errs_n else None,
            "M3_mae": round(fmean(abs(e) for e in errs), 3) if errs else None,
            "M3_n": {"wait": len(errs_w), "nonwait": len(errs_n)},
            "G_scale_mean": round(fmean(abs(c["G"]) for c in cal_all), 2)}
    res = {"repro": repro_stamp(
               experiment="YR-125 1단계 — Q값 진단 (무학습)",
               seeds={"train": list(TRAIN_SEEDS),
                      "eval": sorted({s for _, s in eps})},
               profile_id="calibrated",
               prereg="진단 — 지표 M1/M2/M3 과 해석 규칙(H-A 변별상실 / H-B 체계적 오평가 / "
                      "H-C 목표충실·목표불일치)을 결과 열람 전 동결. 판정·채택 아님.",
               extra={"arms": {k: {"forbid": v[1], "gamma": v[2], "wait_pen": v[3]}
                               for k, v in ARMS.items()}}),
           "summary": summary}
    (OUT / "diagnosis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-per-cell", type=int, default=2)
    a = ap.parse_args()
    run(a.eps_per_cell)
    print("DONE")
