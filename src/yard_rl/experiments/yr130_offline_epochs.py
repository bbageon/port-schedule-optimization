"""YR-130 — 고정 데이터셋 오프라인 epoch 학습 (YR-129 R-미적합 분기의 동결 처방).

■ 왜
YR-129: 차분 회귀가 미적합(훈련 r 0.14~0.45·기울기 4~21 = 예측 압축). 처방(3차 피드백):
**표본 재생성 금지** — 저장된 고정 데이터로 train/val 분리·epoch 반복·val 정체 시 종료.
생성과 갱신을 섞으면 정답 분포가 움직여 "표본 부족 vs 갱신 부족"이 분리되지 않는다.

■ 계약 (결과 미열람 동결)
- 데이터 고정: YR-129 dataset_s{ts}.pt. train = **훈련 대역 실행 표본만**(원 학습의
  상태당-1-표본 체제와 같은 절단 — 미선택 라벨 혼입 금지: 그건 K-후보 분기의 몫),
  val = 신규 대역 실행 표본. 전원-WAIT 항등(D≡0) 제외.
- 학습: JointPairNet 신규 초기화(manual_seed=ts), Adam 5e-4·배치 64·Huber·clip 10
  (base recipe 동일), 최대 2000 epoch, val Pearson r patience 100 조기 종료,
  best-val 스냅샷 저장. 표적 스케일 D/SCALE(=20) — r 은 스케일 불변.
- 정직 고지: 이 데이터셋(시드당 실행 표본 ~0.9~1.7k)은 원 학습 81k 보다 작다 — 검증
  질문은 "갱신 확대의 충분성"이 아니라 **"고정 표본에서 epoch 만으로 적합이 사는가"**.
- 판별 (동결):
  J1 적합: best val r ≥ 0.5 (3/3 시드) → **갱신 병목 확정** (epoch 만으로 회복).
     val r 정체 < 0.5 → 표본량·표현 한계 → 분기 = K-후보 학습(미선택 라벨 기성) 또는
     YR-124 상태표현. 중간(일부 시드만)은 시드별 보고 후 다수결 아닌 **전수 명기**.
  J2 관찰(판별 아님): best-val 망으로 YR-128 동일 순위 재진단(top-1·ρ·P(cw|qw)) —
     적합 회복이 **순위 능력**으로 이어지는지. 낮게 나오면 "적합≠순위"의 재확인이며
     다음 축(K-후보/창) 설계 근거.
  진단·개발 — 판정·채택 아님.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean

import torch
from torch import nn

from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from .yr090_dense_vessel import BASE, CELLS, SCALE
from .yr119_wait_retrain import _set_forbid_wait
from .yr128_rank_diagnosis import _episode_ranks, _summ

OUT = Path("outputs/reports/yr130_offline_epochs")
DATA = Path("outputs/reports/yr129_diff_fit")
CKPT = Path("outputs/reports/yr125_diff_credit")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
MAX_EPOCHS = 2000
PATIENCE = 100
DIAG_EPS = [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)]


def _pearson_t(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _load(ts: int):
    d = torch.load(DATA / f"dataset_s{ts}.pt", map_location="cpu")
    X, meta = d["X"], d["meta"]
    def cut(band):
        idx = [i for i, m in enumerate(meta)
               if m["band"] == band and m["exec"] and not m["is_base"]]
        y = torch.tensor([meta[i]["d"] / SCALE for i in idx], dtype=torch.float32)
        return X[idx], y
    return cut("train"), cut("novel")


def train_offline(ts: int) -> dict:
    (xt, yt), (xv, yv) = _load(ts)
    torch.manual_seed(ts)
    net = JointPairNet(xt.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(ts)
    best = {"val_r": -1.0, "epoch": 0, "state": None}
    curve = []
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        for bi in torch.randperm(len(xt), generator=g).split(64):
            sc, _ = net(xt[bi])
            loss = nn.functional.smooth_l1_loss(sc, yt[bi])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv, _ = net(xv)
            pt, _ = net(xt)
        vr, tr = _pearson_t(pv, yv), _pearson_t(pt, yt)
        if ep % 10 == 0 or ep == 1:
            curve.append({"epoch": ep, "train_r": round(tr, 4), "val_r": round(vr, 4)})
        if vr > best["val_r"] + 1e-3:
            best = {"val_r": vr, "epoch": ep, "state": copy.deepcopy(net.state_dict()),
                    "train_r": tr}
        if ep - best["epoch"] >= PATIENCE:
            break
    net.load_state_dict(best["state"]); net.eval()
    out = OUT / f"offline_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT / f"diff1_s{ts}" / "rl_net.pt", map_location="cpu")
    torch.save({"state": net.state_dict(), "in_dim": net.in_dim,
                "norm_refs": ck["norm_refs"], "best_ep": best["epoch"],
                "arm": "OFFLINE", "train_seed": ts}, out / "rl_net.pt")
    res = {"n_train": len(xt), "n_val": len(xv), "stopped_epoch": curve[-1]["epoch"],
           "best_epoch": best["epoch"], "best_val_r": round(best["val_r"], 4),
           "train_r_at_best": round(best.get("train_r", 0.0), 4), "curve": curve}
    (out / "fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print(f"[offline s{ts}] train {len(xt)} / val {len(xv)} → "
          f"best val r {res['best_val_r']} @ep{best['epoch']} "
          f"(train r {res['train_r_at_best']})", flush=True)
    return res


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fit = {ts: train_offline(ts) for ts in TRAIN_SEEDS}
    # J2 — 순위 재진단 (관찰)
    rank = {}
    _set_forbid_wait(False)
    try:
        for ts in TRAIN_SEEDS:
            ck = torch.load(OUT / f"offline_s{ts}" / "rl_net.pt", map_location="cpu")
            net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
            norm = StateNorm(refs=ck["norm_refs"])
            recs = []
            for cell, seed in DIAG_EPS:
                recs += _episode_ranks(net, norm, cell, seed)
            rank[ts] = _summ(recs)
            print(f"[rank s{ts}] {json.dumps(rank[ts])}", flush=True)
    finally:
        _set_forbid_wait(True)
    j1 = all(f["best_val_r"] >= 0.5 for f in fit.values())
    judgment = {"J1_fit": {"per_seed": {ts: {"best_val_r": f["best_val_r"],
                                             "best_epoch": f["best_epoch"],
                                             "train_r": f["train_r_at_best"],
                                             "n_train": f["n_train"]}
                                        for ts, f in fit.items()},
                           "update_bottleneck_confirmed": j1},
                "J2_rank_observation": rank}
    res = {"repro": repro_stamp(
               experiment="YR-130 고정 데이터셋 오프라인 epoch (YR-129 처방)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      "diag": sorted({s for _, s in DIAG_EPS})},
               profile_id="calibrated",
               prereg="데이터 고정(YR-129 dataset, 실행 표본만·K-후보 혼입 금지)·fresh "
                      "init·base 하이퍼 동일·val r patience 100. J1: 3/3 val r≥0.5 → 갱신 "
                      "병목 확정 / 정체 <0.5 → 표본·표현 한계(K-후보 or YR-124). J2 순위 "
                      "재진단은 관찰. 진단 — 판정·채택 아님.",
               extra={"max_epochs": MAX_EPOCHS, "patience": PATIENCE}),
           "fit": fit, "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(judgment["J1_fit"], ensure_ascii=False))
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("DONE")
