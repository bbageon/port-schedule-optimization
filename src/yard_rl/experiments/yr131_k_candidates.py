"""YR-131 — K-후보 커버리지 단일축 (a: 후보 확대만 / b: 실패 시에만 순위손실).

■ 왜 (외부 5차 피드백 순서·축 분리 반영, 결과 미열람 동결)
YR-130: 상태당 1표본(실행 후보만) 체제의 일반화 상한 r≈0.3~0.4 확정(과적합 즉발)·순위
미형성. 다음 병목 후보 = 표본 구성. 단 "후보 확대"와 "손실 교체"는 별개 축이므로 분리:
  131-a — 변경 = **상태당 전 후보 라벨 사용** (YR-129 데이터셋의 미선택 라벨 기성).
          유지 = Huber 회귀·fresh init·하이퍼·val 선택지표(Pearson r — YR-130 프로토콜).
          질문: 후보를 여러 개 보여주는 것만으로 일반화·순위가 사는가?
  131-b — 131-a 가 J1 실패할 때만. 데이터·망 동일, 변경 = Huber → 결정 내 pairwise
          순위손실(margin). 선택지표는 val 결정 내 ρ (순위손실의 목적 지표 — 손실 교체에
          수반되는 프로토콜 정의로 고지). 질문: 값이 아니라 순서를 직접 가르쳐야 하는가?
기준행동 SF-SPT 교체는 이 축 뒤로 미룸 — **같은 상태의 모든 후보에서 같은 기준값을
빼므로 상태 내 순위는 불변** (기준 교체는 개입 게이트·해석용이지 순위 결함 처방이 아님).

■ 데이터·판별 (동결)
- YR-129 dataset_s*.pt 전 live 라벨(전원-WAIT 항등 제외). train = 훈련 대역 전 후보,
  held-out = 신규 대역 전 후보. 비교 다리로 실행-후보-한정 r 도 병기(YR-130 대응치).
- J1 (3/3 시드, held-out): Pearson r ≥ 0.5 ∧ 결정 내 Spearman ρ 평균 ≥ 0.30 ∧
  top-1 ≥ 0.35 (참조: YR-130 r 0.29~0.39·ρ ≈0·top-1 0.16~0.18). a 통과 → b 생략,
  사다리 5단(종결 잔여 축)으로. a 실패 → b 실행, 같은 임계.
- 관찰 부속 (판별 외): **혼잡도 구간별 신호 분석** — 결정별 resid(창 종료 미완 수) 5분위
  × {평균|D|·D 표준편차·결정 내 범위·기준 대비 최대 개선량(−min D)}. "혼잡할수록 600s
  창의 변별 신호가 0 으로 준다"(YR-129 사후 가설)의 직접 검증 — 저장 라벨만 사용.
- 진단·개발 — 판정·채택 아님. held-out 은 열람된 진단 대역 재사용.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import fmean, pstdev

import torch
from torch import nn

from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from .yr090_dense_vessel import SCALE
from .yr128_rank_diagnosis import _spearman

OUT = Path("outputs/reports/yr131_k_candidates")
DATA = Path("outputs/reports/yr129_diff_fit")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
MAX_EPOCHS = 2000
PATIENCE = 100
MARGIN = 0.01          # 순위손실 margin (D/SCALE 단위) — 동결
PAIR_EPS = 0.01        # 라벨 차이가 이보다 작은 쌍은 잡음으로 제외 — 동결


def _load(ts: int):
    d = torch.load(DATA / f"dataset_s{ts}.pt", map_location="cpu")
    X, meta = d["X"], d["meta"]
    idx = [i for i, m in enumerate(meta) if not m["is_base"]]
    def cut(band):
        ii = [i for i in idx if meta[i]["band"] == band]
        return {"X": X[ii],
                "y": torch.tensor([meta[i]["d"] / SCALE for i in ii],
                                  dtype=torch.float32),
                "dec": [meta[i]["dec"] for i in ii],
                "exec": [meta[i]["exec"] for i in ii],
                "has_wait": [meta[i]["has_wait"] for i in ii],
                "resid": [meta[i]["resid"] for i in ii]}
    return cut("train"), cut("novel")


def _rank_metrics(pred, cut):
    by = {}
    for i, dec in enumerate(cut["dec"]):
        by.setdefault(dec, []).append(i)
    top1, rhos, qw_cw = [], [], []
    for dec, ii in by.items():
        if len(ii) < 2:
            continue
        dh = [float(pred[i]) for i in ii]
        dd = [float(cut["y"][i]) for i in ii]
        top1.append(1.0 if dh.index(min(dh)) == dd.index(min(dd)) else 0.0)
        if len(ii) >= 3:
            r = _spearman(dh, dd)
            if r is not None:
                rhos.append(r)
        w = [cut["has_wait"][i] for i in ii]
        if any(w) and not all(w):
            qw = min(x for x, f in zip(dh, w) if f) < min(x for x, f in zip(dh, w) if not f)
            cw = min(x for x, f in zip(dd, w) if f) < min(x for x, f in zip(dd, w) if not f)
            qw_cw.append((qw, cw))
    n_qw = sum(1 for q, _ in qw_cw if q)
    return {"top1": round(fmean(top1), 4) if top1 else None,
            "rho": round(fmean(rhos), 4) if rhos else None,
            "p_cw_given_qw": (round(sum(1 for q, c in qw_cw if q and c) / n_qw, 4)
                              if n_qw else None),
            "n_decisions": len(top1)}


def _pearson_t(a, b):
    if a.numel() < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _congestion_annex(tr, va):
    by = {}
    for cut in (tr, va):
        for i, dec in enumerate(cut["dec"]):
            by.setdefault(dec, {"d": [], "resid": []})
            by[dec]["d"].append(float(cut["y"][i]) * SCALE)
            by[dec]["resid"].append(cut["resid"][i])
    rows = [{"resid": fmean(v["resid"]), "abs_d": fmean(abs(x) for x in v["d"]),
             "sd_d": pstdev(v["d"]) if len(v["d"]) > 1 else 0.0,
             "range_d": max(v["d"]) - min(v["d"]), "best_gain": -min(v["d"])}
            for v in by.values() if len(v["d"]) >= 2]
    rows.sort(key=lambda r: r["resid"])
    n5 = max(1, len(rows) // 5)
    annex = []
    for q in range(5):
        seg = rows[q * n5: (q + 1) * n5] if q < 4 else rows[4 * n5:]
        annex.append({"quintile": q + 1,
                      "resid_mean": round(fmean(r["resid"] for r in seg), 1),
                      "abs_d_mean": round(fmean(r["abs_d"] for r in seg), 3),
                      "sd_d_mean": round(fmean(r["sd_d"] for r in seg), 3),
                      "range_d_mean": round(fmean(r["range_d"] for r in seg), 3),
                      "best_gain_mean": round(fmean(r["best_gain"] for r in seg), 3)})
    return annex


def train_arm(ts: int, arm: str) -> dict:
    tr, va = _load(ts)
    torch.manual_seed(ts)
    net = JointPairNet(tr["X"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(ts)
    # b 용 결정 내 쌍 사전 구축 (라벨 차 > PAIR_EPS)
    pairs = []
    if arm == "b":
        by = {}
        for i, dec in enumerate(tr["dec"]):
            by.setdefault(dec, []).append(i)
        for ii in by.values():
            for a_ in ii:
                for b_ in ii:
                    if float(tr["y"][a_]) + PAIR_EPS < float(tr["y"][b_]):
                        pairs.append((a_, b_))       # a_ 가 더 좋음(비용 작음)
        pairs = torch.tensor(pairs, dtype=torch.long)
    best = {"sel": -1.0, "epoch": 0, "state": None}
    curve = []
    steps = max(1, len(tr["X"]) // 64)
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        if arm == "a":
            for bi in torch.randperm(len(tr["X"]), generator=g).split(64):
                sc, _ = net(tr["X"][bi])
                loss = nn.functional.smooth_l1_loss(sc, tr["y"][bi])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        else:
            for _ in range(steps):
                bi = pairs[torch.randint(len(pairs), (64,), generator=g)]
                sa, _ = net(tr["X"][bi[:, 0]])
                sb, _ = net(tr["X"][bi[:, 1]])
                loss = nn.functional.margin_ranking_loss(
                    sb, sa, torch.ones(len(bi)), margin=MARGIN)   # sb > sa 이도록
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv, _ = net(va["X"])
        vr = _pearson_t(pv, va["y"])
        if arm == "a":
            sel = vr
        else:
            sel = _rank_metrics(pv, va)["rho"] or -1.0
        if ep % 10 == 0 or ep == 1:
            curve.append({"epoch": ep, "val_r": round(vr, 4), "sel": round(sel, 4)})
        if sel > best["sel"] + 1e-3:
            best = {"sel": sel, "epoch": ep, "state": copy.deepcopy(net.state_dict())}
        if ep - best["epoch"] >= PATIENCE:
            break
    net.load_state_dict(best["state"]); net.eval()
    with torch.no_grad():
        pv, _ = net(va["X"])
        pt, _ = net(tr["X"])
    ex = [i for i, e in enumerate(va["exec"]) if e]
    res = {"arm": arm, "n_train": len(tr["X"]), "n_val": len(va["X"]),
           "best_epoch": best["epoch"], "stopped_epoch": curve[-1]["epoch"],
           "val_r_all": round(_pearson_t(pv, va["y"]), 4),
           "val_r_exec_only": round(_pearson_t(pv[ex], va["y"][ex]), 4),
           "train_r_all": round(_pearson_t(pt, tr["y"]), 4),
           "rank": _rank_metrics(pv, va), "curve": curve}
    out = OUT / f"{arm}_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "in_dim": net.in_dim, "arm": arm,
                "train_seed": ts}, out / "net.pt")
    (out / "fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print(f"[131-{arm} s{ts}] {json.dumps({k: res[k] for k in ('val_r_all', 'val_r_exec_only', 'rank', 'best_epoch')})}",
          flush=True)
    return res


def _j1(fit: dict) -> bool:
    return all(f["val_r_all"] >= 0.5 and (f["rank"]["rho"] or 0.0) >= 0.30
               and (f["rank"]["top1"] or 0.0) >= 0.35 for f in fit.values())


def run(mode: str = "auto") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    annex = {}
    for ts in TRAIN_SEEDS:
        tr, va = _load(ts)
        annex[ts] = _congestion_annex(tr, va)
    fit_a = {ts: train_arm(ts, "a") for ts in TRAIN_SEEDS}
    j1a = _j1(fit_a)
    fit_b = None
    if mode == "auto" and not j1a:
        print("[131] a 실패 → b(순위손실) 실행", flush=True)
        fit_b = {ts: train_arm(ts, "b") for ts in TRAIN_SEEDS}
    judgment = {"J1_a_pass": j1a,
                "J1_b_pass": _j1(fit_b) if fit_b else None,
                "thresholds": {"val_r": 0.5, "rho": 0.30, "top1": 0.35},
                "reference_yr130": {"val_r": [0.292, 0.392, 0.302],
                                    "top1": [0.16, 0.164, 0.18], "rho": "≈0"}}
    res = {"repro": repro_stamp(
               experiment="YR-131 K-후보 커버리지 단일축 (a: 후보 확대 / b: 조건부 순위손실)",
               seeds={"train_ckpt": list(TRAIN_SEEDS)},
               profile_id="calibrated",
               prereg="데이터 = YR-129 전 live 라벨(신규 rollout 0)·YR-130 과 유일 차이 = "
                      "표본 구성(a)·추가로 손실만(b, a 실패 시). J1: 3/3 held-out r≥0.5 ∧ "
                      "ρ≥0.30 ∧ top1≥0.35. 혼잡도 5분위 신호 분석은 관찰 부속. "
                      "기준행동 교체는 순위 불변이므로 이 축에서 제외(5차 피드백). "
                      "진단 — 판정·채택 아님.",
               extra={"margin": MARGIN, "pair_eps": PAIR_EPS, "patience": PATIENCE}),
           "congestion_annex": annex, "fit_a": fit_a, "fit_b": fit_b,
           "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(judgment, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto", "a", "b"])
    a = ap.parse_args()
    if a.mode in ("a", "b"):
        for ts in TRAIN_SEEDS:
            train_arm(ts, a.mode)
    else:
        run()
    print("DONE")
