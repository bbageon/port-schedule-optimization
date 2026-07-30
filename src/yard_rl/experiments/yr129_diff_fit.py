"""YR-129 — 차분 실패 판별 진단: D̂ 적합도 4축 (무학습, 외부 피드백 3차 설계 동결).

■ 왜
YR-125 2단계 기각의 원인이 ①회귀 미적합 ②일반화/상태표현 ③후보 커버리지 부족
④표적(종결비용) 결함 중 무엇인지 분리되지 않았다. 이 진단이 분리한다 — 결과 전에는
학습량 확대·종결비용 추가·YR-124 중 무엇도 시작하지 않는다 (피드백 지시).

■ 4축 측정 (결과 열람 전 동결)
  A 훈련분포 적합: 학습 대역(BASE+0/+1, 훈련 회전에 포함된 시드)에서 정책 실행 후보의
    D̂ vs 실측 D — Pearson r·부호 일치율·MAE·보정 기울기.
    ⚠ 정직 고지: 리플레이 버퍼의 문자 그대로의 표본 재현이 아니라 **분포 수준 근사**
    (훈련 중 표본은 ε-혼합 궤적에서 나왔음).
  B 신규상태 적합: 학습에 없던 대역(BASE+700/+701 — 열람된 진단 대역)에서 같은 측정.
  C 미선택 후보 적합: 각 결정에서 **열거된 모든 공동후보**에 D 라벨을 달아
    (후보별 rollout, 결정당 baseline 1회 공유) 실행/미선택 절단별 적합 + 결정 내
    Spearman 순위상관·top-1. 전원-WAIT 후보는 D≡0 항등이므로 적합 집계에서 제외하고
    D̂ 분포만 "0-앵커 검사"로 따로 보고.
  D 종결 잔여 상관: 각 rollout 종료(t0+600s) 시점 사본의 미완 작업 수(resid)를 기록,
    |D−D̂| 오차와의 상관 + resid 3분위별 오차 — 종결비용 누락 가설의 지지/기각.
  부수: 행동 유형별(REPO 포함/부분 WAIT/전원 SERVE) 오차 분해 — "REPO 에서만 오차 큼"
    검사. 라벨 절단 없음(열거기 max_combos=64 기존 계약만 적용, 발생 시 보고).

■ 판별 규칙 (동결 — 복수 발화 가능, 전부 보고)
  R-미적합:   A 축 r < 0.5 → 회귀 덜 됨 → 처방 = **고정 데이터셋 오프라인 epoch 학습**
              (표본 재생성 금지 — 이 런이 저장한 dataset_s*.pt 로 train/val 분리·
              val 적합 정체 시 종료. 생성과 갱신을 섞으면 정답 분포가 움직여 진단 흐림).
  R-일반화:   A r ≥ 0.5 이고 (B r < 0.3 또는 B r < A r − 0.2) → 상태표현/일반화 → YR-124.
  R-커버리지: 실행 후보 부호 일치 ≥ 0.7 인데 미선택 후보 결정 내 Spearman < 0.3
              → 후보 라벨 부족 → 처방 = 상태당 K-후보 차분학습 (갱신 확대 아님).
  R-표적:     A·B·C 전부 임계 이상인데 운영 실패(YR-125 P3~P5 기지) → 종결 잔여비용
              추가 또는 시간창 재검. D 축 상관 > 0.3 이면 종결 누락 가설 지지 가중.
  R-유형:     REPO 포함 후보 MAE > 전체 MAE × 2 → 행동 유형별 차분표적 검사.
  진단이며 판정·채택 아님. novel 대역은 열람된 진단 대역 재사용(YR-125 1단계 지위).
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
from .yr088_joint_rl import LEVEL, RC as RC_TRAIN, build_rows
from .yr090_dense_vessel import BASE, CELLS, SCALE, _sim
from .yr119_wait_retrain import _set_forbid_wait
from .yr125_diff_credit import HORIZON_S
from .yr128_rank_diagnosis import _spearman

OUT = Path("outputs/reports/yr129_diff_fit")
CKPT = Path("outputs/reports/yr125_diff_credit")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
BANDS = {"train": [(c, BASE[c] + i) for c in CELLS for i in range(2)],
         "novel": [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)]}


def _resid(scratch) -> int:
    return sum(1 for j in scratch.jobs.values() if j.status.name != "DONE")


def _episode_labels(net, norm, cell, seed, band, ep_id):
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=False)
    base_pol = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    out = []                                   # (row, meta)
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
            dhat = [float(x) * SCALE for x in sc]
            pick = dhat.index(min(dhat))
            full_exists = any(all(a[c].kind != CandidateKind.WAIT for c in dp.crane_ids)
                              for a in assigns)
            if full_exists:
                baseline = {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
                cb, sb = _rollout_cost(sim, baseline, RC_TRAIN, horizon_s=HORIZON_S,
                                       base_policy=base_pol, generator=gen)
                rb = _resid(sb)
                for i, a in enumerate(assigns):
                    kinds = [a[c].kind for c in dp.crane_ids]
                    is_base = all(kd == CandidateKind.WAIT for kd in kinds)
                    if is_base:
                        d, resid = 0.0, rb
                    else:
                        ca, sa = _rollout_cost(sim, a, RC_TRAIN, horizon_s=HORIZON_S,
                                               base_policy=base_pol, generator=gen)
                        d, resid = ca - cb, _resid(sa)
                    out.append((rows[i], {
                        "band": band, "dec": f"{ep_id}:{k}", "exec": i == pick,
                        "dhat": dhat[i], "d": d, "is_base": is_base, "resid": resid,
                        "has_repo": any(kd == CandidateKind.REPOSITION for kd in kinds),
                        "has_wait": any(kd == CandidateKind.WAIT for kd in kinds)}))
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    return out


def _pearson(x, y):
    mx, my = fmean(x), fmean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _slope(x, y):
    mx, my = fmean(x), fmean(y)
    vx = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / vx if vx else None


def _fit(ms):
    """meta 리스트 → 적합 지표 (전원-WAIT 항등 후보 제외 전제)."""
    if len(ms) < 3:
        return None
    x = [m["dhat"] for m in ms]; y = [m["d"] for m in ms]
    return {"n": len(ms),
            "pearson_r": round(_pearson(x, y) or 0.0, 4),
            "sign_agree": round(fmean(1.0 if (a < 0) == (b < 0) else 0.0
                                      for a, b in zip(x, y)), 4),
            "mae": round(fmean(abs(a - b) for a, b in zip(x, y)), 3),
            "slope": round(_slope(x, y) or 0.0, 4)}


def _rank_c(ms):
    """결정 내 순위 (미선택 커버리지) — 후보 ≥3 결정의 Spearman·top-1."""
    by = {}
    for m in ms:
        by.setdefault(m["dec"], []).append(m)
    rhos, top1 = [], []
    for dec, group in by.items():
        if len(group) < 2:
            continue
        dh = [g["dhat"] for g in group]; dd = [g["d"] for g in group]
        top1.append(1.0 if dh.index(min(dh)) == dd.index(min(dd)) else 0.0)
        if len(group) >= 3:
            r = _spearman(dh, dd)
            if r is not None:
                rhos.append(r)
    return {"n_decisions": len(top1),
            "top1_agree": round(fmean(top1), 4) if top1 else None,
            "spearman_mean": round(fmean(rhos), 4) if rhos else None}


def _d_axis(ms):
    errs = [abs(m["d"] - m["dhat"]) for m in ms]
    resid = [float(m["resid"]) for m in ms]
    r = _pearson(resid, errs)
    order = sorted(range(len(ms)), key=lambda i: resid[i])
    n3 = max(1, len(order) // 3)
    lo, hi = order[:n3], order[-n3:]
    return {"corr_resid_abs_err": round(r or 0.0, 4),
            "mae_low_resid": round(fmean(errs[i] for i in lo), 3),
            "mae_high_resid": round(fmean(errs[i] for i in hi), 3),
            "resid_mean": round(fmean(resid), 2)}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    _set_forbid_wait(False)
    try:
        for ts in TRAIN_SEEDS:
            ck = torch.load(CKPT / f"diff1_s{ts}" / "rl_net.pt", map_location="cpu")
            net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
            norm = StateNorm(refs=ck["norm_refs"])
            rows_x, metas = [], []
            for band, eps in BANDS.items():
                for cell, seed in eps:
                    lab = _episode_labels(net, norm, cell, seed, band, f"{cell}:{seed}")
                    rows_x += [r for r, _ in lab]
                    metas += [m for _, m in lab]
                    print(f"[s{ts} {band} {cell}:{seed}] 라벨 {len(lab)} 누적 {len(metas)}",
                          flush=True)
            torch.save({"X": torch.tensor(rows_x, dtype=torch.float32),
                        "meta": metas, "scale": SCALE},
                       OUT / f"dataset_s{ts}.pt")        # 고정 데이터셋 (오프라인 분기용)
            live = [m for m in metas if not m["is_base"]]
            anchor = [m["dhat"] for m in metas if m["is_base"]]
            s = {}
            for band in BANDS:
                bm = [m for m in live if m["band"] == band]
                s[band] = {
                    "executed_fit": _fit([m for m in bm if m["exec"]]),
                    "unchosen_fit": _fit([m for m in bm if not m["exec"]]),
                    "all_fit": _fit(bm), "rank": _rank_c(bm), "d_axis": _d_axis(bm)}
            s["zero_anchor_dhat"] = ({"n": len(anchor),
                                      "mean": round(fmean(anchor), 3),
                                      "mean_abs": round(fmean(abs(a) for a in anchor), 3)}
                                     if anchor else None)
            s["by_type_mae"] = {
                t: (round(fmean(abs(m["d"] - m["dhat"]) for m in live if m[t]), 3)
                    if any(m[t] for m in live) else None)
                for t in ("has_repo", "has_wait")}
            s["mae_all"] = round(fmean(abs(m["d"] - m["dhat"]) for m in live), 3)
            summary[ts] = s
            print(f"[s{ts}] 완료", flush=True)
    finally:
        _set_forbid_wait(True)
    res = {"repro": repro_stamp(
               experiment="YR-129 차분 실패 판별 — D̂ 적합도 4축 (무학습)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      **{b: sorted({s for _, s in e}) for b, e in BANDS.items()}},
               profile_id="calibrated",
               prereg="4축(A 훈련분포/B 신규상태/C 미선택 후보/D 종결 잔여) + 판별 규칙 "
                      "R-미적합(A r<0.5→고정 데이터셋 오프라인 epoch)·R-일반화(B r<0.3 또는 "
                      "A−0.2 초과 하락→YR-124)·R-커버리지(실행 부호≥0.7·미선택 ρ<0.3→K-후보 "
                      "학습)·R-표적(전부 양호+운영 실패→종결비용/창)·R-유형(REPO MAE>2×) 을 "
                      "결과 열람 전 동결. 진단 — 판정·채택 아님.",
               extra={"horizon_s": HORIZON_S, "bands": {b: len(e) for b, e in BANDS.items()}}),
           "summary": summary}
    (OUT / "diagnosis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("DONE")
