"""YR-127 — 학습 예산 단일축: 경사 갱신 ×20 (YR-125 1단계의 최소 수정 검증).

■ 왜 (결과 미열람 동결)
YR-125 1단계: 12 체크포인트 전부에서 Q̂ 가 실현 비용-투-고의 1~3% (보정오차 ≈ G 전체) —
사실상 미학습. 원인 역산 = 총 경사 갱신 ~1,000회 vs 전파 지평 150+ 결정/에피소드
(수집 전이당 경사 표본 ~0.9회; 통상 DQN 관행 ≥4~8회). 이 실험은 그 원인론의 직접 검증:
"부족한 건 메커니즘이 아니라 학습량"이라면 갱신만 늘려도 보정이 살아나야 한다.

■ 설계 — 유일한 차이 하나
  · base = YR-119 WAITON recipe 전부 동일 (FORBID_WAIT=False·REPO_PENALTY 0.5·
    UNSERVED 30·γ 0.99·lr 5e-4·배치 64·에피소드 500·학습시드 {88000, 99000, 123000}·
    학습/검증 대역·신경망·best-EMA 선택 규칙).
  · 유일 차이: y90.UPDATE_MULT = 20 — 에피소드당 경사 갱신 루프 **횟수만** 배수.
    갱신 1회의 내용물(배치 64 표집·Huber·clip 10·soft 갱신 τ)은 불변.
  · 실제 수행된 경사 갱신 수를 계수해 train_meta.json 에 박제한다 (축 실증).
  · 고지: 갱신 수 증가는 rng 소비 경로를 바꾸므로 같은 시드라도 WAITON 과 표집 이력이
    다르다 — 이는 축 자체에 내재한 차이다 (별도 교란 아님).

■ 판정 (결과 열람 전 동결 — 성공 = N1 ∧ N2 ∧ N3, 통계/운영 분리 보고)
  N1 보정 회복: YR-125 와 동일 진단(같은 8 에피소드·같은 보상 정의·γ 0.99)에서
     bias_ratio = |mean(G−Q̂)| / mean(|G|) 이 **3/3 학습시드 모두 ≤ 0.5**
     (참조: YR-119 WAITON ≈ 0.99. Q̂ 가 실현수익의 절반 이상을 설명해야 "회복").
  N2 전략적 WAIT 감소: 평가대역(4셀 × 시드 6 = 24 에피소드/모델)에서 BUDGET20 의
     전략적 WAIT 선택률(SERVE 실행가능인데 WAIT) 전체 평균이 **같은 프로세스에서
     재평가한 WAITON 의 절반 이하** AND 어떤 에피소드도 WAIT 행동점유 > 0.60 없음.
  N3 비용 비악화: 같은 (학습시드, 평가에피소드) 짝의 총비용 차(BUDGET20 − WAITON)
     upper95 < +10 (δ total) AND 미건전 에피소드 수 비증가 (BUDGET20 ≤ WAITON).
  참고 지표(판정 외): SF 대비 총비용·A→O, REPOSITION 비중, WAIT 지속시간 점유,
     시드별 재현성. 진단·개발 단계 — 채택 판정 아님 (채택은 별도 사전등록 후보평가).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

import torch

from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference
from ..integrated.encoding import StateNorm
from ..integrated.evalkit import paired
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from . import yr088_joint_rl as y88
from . import yr090_dense_vessel as y90
from .yr090_dense_vessel import BASE, CELLS
from .yr119_wait_retrain import EVAL_SEEDS, _episode, _rl_policy_factory, _set_forbid_wait
from .yr125_qvalue_diagnosis import _diagnose_episode

OUT = Path("outputs/reports/yr127_training_budget")
YR119 = Path("outputs/reports/yr119_wait_retrain")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
MULT = 20
DELTA_TOTAL = 10.0
# N1 진단 대역 — YR-125 와 동일 (4셀 × 2)
DIAG_EPS = [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)]


def train(seed: int, episodes: int = 500) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    prev = (y90.OUT, y88.FORBID_WAIT, y90.FORBID_WAIT, y90.UPDATE_MULT, y90._train_step)
    cnt = {"n": 0}
    orig_step = y90._train_step

    def counting_step(*a, **k):
        cnt["n"] += 1
        return orig_step(*a, **k)

    y90.OUT = OUT
    _set_forbid_wait(False)
    y90.UPDATE_MULT = MULT
    y90._train_step = counting_step
    try:
        t0 = time.time()
        p = y90.train_one("BUDGET20", seed, episodes=episodes)
        meta = {"updates_performed": cnt["n"], "update_mult": MULT, "episodes": episodes,
                "wall_s": round(time.time() - t0, 1)}
        (p.parent / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"[train BUDGET20 s{seed}] {meta}", flush=True)
        return p
    finally:
        y90.OUT, y88.FORBID_WAIT, y90.FORBID_WAIT, y90.UPDATE_MULT, y90._train_step = prev


def _diag_ck(ck_path: Path) -> dict:
    """N1 — YR-125 와 동일한 보정 진단 (무학습, WAITON 구성: forbid=False·γ 0.99·벌점 0)."""
    ck = torch.load(ck_path, map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    norm = StateNorm(refs=ck["norm_refs"])
    _set_forbid_wait(False)
    try:
        cal = []
        for cell, seed in DIAG_EPS:
            _, c = _diagnose_episode(net, norm, cell, seed,
                                     forbid=False, gamma=0.99, wait_pen=0.0)
            cal += c
    finally:
        _set_forbid_wait(True)
    errs = [c["G"] - c["q_hat"] for c in cal]
    g = fmean(abs(c["G"]) for c in cal)
    return {"bias": round(fmean(errs), 3), "G_scale": round(g, 2),
            "bias_ratio": round(abs(fmean(errs)) / g, 4) if g else None, "n": len(cal)}


def judge() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [(c, s) for c in CELLS for s in EVAL_SEEDS[c]]
    print(f"[eval] SF 기준선 {len(eps)} 에피소드", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"))
          for c, s in eps]
    rows: dict[str, list[dict]] = {}
    for ts in TRAIN_SEEDS:
        for name, ckdir in (("WAITON", YR119 / f"waiton_s{ts}"),
                            ("BUDGET20", OUT / f"budget20_s{ts}")):
            ck = ckdir / "rl_net.pt"
            if not ck.exists():
                raise FileNotFoundError(f"체크포인트 없음: {ck}")
            print(f"[eval] {name}:{ts}", flush=True)
            mk = _rl_policy_factory(ck, False)
            rows[f"{name}:{ts}"] = [_episode(c, s, mk) for c, s in eps]
    print("[diag] N1 보정 진단", flush=True)
    diag = {ts: _diag_ck(OUT / f"budget20_s{ts}" / "rl_net.pt") for ts in TRAIN_SEEDS}

    # ---- N1
    n1 = {"per_seed": diag,
          "pass": all(d["bias_ratio"] is not None and d["bias_ratio"] <= 0.5
                      for d in diag.values())}
    # ---- N2
    b_all = [r for ts in TRAIN_SEEDS for r in rows[f"BUDGET20:{ts}"]]
    w_all = [r for ts in TRAIN_SEEDS for r in rows[f"WAITON:{ts}"]]
    swr_b = fmean(r["strategic_wait_rate"] for r in b_all)
    swr_w = fmean(r["strategic_wait_rate"] for r in w_all)
    dom = any(r["shares"].get("WAIT", 0) > 0.60 for r in b_all)
    n2 = {"strategic_wait_rate_budget20": round(swr_b, 4),
          "strategic_wait_rate_waiton": round(swr_w, 4),
          "wait_dominates_any": dom,
          "pass": bool(swr_b <= 0.5 * swr_w and not dom)}
    # ---- N3 (같은 학습시드·같은 평가에피소드 짝)
    d_tot = [a["total"] - b["total"] for ts in TRAIN_SEEDS
             for a, b in zip(rows[f"BUDGET20:{ts}"], rows[f"WAITON:{ts}"])]
    p_tot = paired(d_tot, delta_interest=DELTA_TOTAL)
    unh_b = sum(1 for r in b_all if not r["healthy"])
    unh_w = sum(1 for r in w_all if not r["healthy"])
    n3 = {"d_total_budget20_minus_waiton": p_tot.as_dict(),
          "unhealthy": {"BUDGET20": unh_b, "WAITON": unh_w},
          "pass": bool(p_tot.ci_hi < DELTA_TOTAL and unh_b <= unh_w)}
    # ---- 참고 (판정 외)
    ref = {}
    for name, allrows in (("BUDGET20", b_all), ("WAITON", w_all)):
        sf3 = sf * len(TRAIN_SEEDS)
        ref[name] = {
            "d_total_vs_sf": paired([r["total"] - s["total"] for r, s in zip(allrows, sf3)],
                                    delta_interest=DELTA_TOTAL).as_dict(),
            "repo_share_mean": round(fmean(r["repo_share"] for r in allrows), 4),
            "wait_duration_share_mean": round(fmean(r["wait_duration_share"]
                                                    for r in allrows), 4),
            "compl_min": min(r["compl"] for r in allrows),
            "backlog_max": max(r["backlog"] for r in allrows)}
    meta = {}
    for ts in TRAIN_SEEDS:
        mp = OUT / f"budget20_s{ts}" / "train_meta.json"
        meta[ts] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
    judgment = {"N1_calibration": n1, "N2_strategic_wait": n2, "N3_cost": n3,
                "success": bool(n1["pass"] and n2["pass"] and n3["pass"]),
                "train_meta": meta}
    res = {"repro": repro_stamp(
               experiment="YR-127 학습 예산 단일축 — 경사 갱신 ×20 (base = YR-119 WAITON recipe)",
               seeds={"train": list(TRAIN_SEEDS),
                      **{c: EVAL_SEEDS[c] for c in CELLS},
                      "diag": sorted({s for _, s in DIAG_EPS})},
               profile_id="calibrated",
               prereg="유일 차이 = UPDATE_MULT 20. 성공 = N1 보정회복(3/3 시드 bias_ratio ≤ 0.5) "
                      "∧ N2 전략적 WAIT ≤ 0.5×WAITON(재평가)·도미넌스 없음 ∧ N3 짝지은 총비용 "
                      "upper95 < +10·미건전 비증가. 실패 시 YR-125 2단계(차분 1-step). "
                      "진단·개발 단계 — 채택 판정 아님.",
               extra={"update_mult": MULT, "delta_total": DELTA_TOTAL}),
           "sf": sf, "arms": rows, "diagnosis": diag, "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0, help="학습시드 1개 (예: 88000)")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    if a.train:
        train(a.train, episodes=a.episodes)
    if a.judge:
        r = judge()
        print(json.dumps(r["judgment"], ensure_ascii=False, indent=1))
    print("DONE")
