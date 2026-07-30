"""YR-132 — 종결 잔여 표적 단일축: D_H = D + λ·Δ잔여 (순위손실 유지, 새 판정 대역).

■ 왜 (외부 6차 피드백 승인 조건 반영, 결과 미열람 동결)
YR-131 확정: ①순위는 순위손실로만 형성 ②혼잡 결정(~80%)은 600s 창에 변별 신호가 없음.
저장 라벨 검증: 혼잡 상위 40% 결정에서도 후보별 미완 수가 다른 결정 57~71%·평균 범위
1.0~1.7건 — 잔여항이 실제 새 신호를 준다. 이 실험은 표적에 종결 잔여를 더하는 **단일축**.

■ 표적 (동결)
  D_H(a) = D(a) + λ·(resid(a) − resid(기준행동))   [numeraire]
  · resid = 각 후보 rollout 의 창 종료(t0+600s) 미완 작업 수 (라벨에 저장돼 있음 —
    학습·선택 데이터는 신규 rollout 0).
  · **λ = 1.0 동결** — 가격 앵커(튜닝 아님): "창이 못 본 미완 1건 = 트럭 대기 1시간과
    같은 값" (기준재 정의 앵커, YR-121 벌점 앵커와 같은 방식론).
  · **미완 '개수' 잔여는 진단용 지위** (6차 피드백): 쉬운 작업 여러 개로 급한 본선 1건을
    가리는 편법이 가능하므로, 성공 시 YR-123 비용곡선 기반 **비용가중 잔여**로 승계한다.

■ 학습·판정 (동결)
  · 학습 = YR-131-b 프로토콜 그대로 (pairwise margin 0.01·잡음쌍 제외 0.01·fresh init·
    선택지표 = 기존 novel 대역 ρ(D̂, D_H)·patience 100) — 유일 차이 = 표적 D → D_H.
  · **판정 대역 = 완전히 새로운 대역** (BASE+900/+901, 미열람 — 6차 피드백 조건):
    DIFF1 궤적으로 라벨 생성(YR-129 프로토콜·resid 포함), 이 런이 dataset_new 로 저장.
  · J1 (3/3 시드, 새 대역): 결정 내 ρ(D̂, D_H) ≥ 0.30 ∧ top-1 ≥ 0.35.
  · J2 (3/3 시드, 새 대역): **선택 후 손실(regret)** — 같은 결정에서
    regret = D_H(net 의 argmin) − min D_H. mean regret(YR-132) < mean regret(YR-131-b)
    (같은 라벨·같은 결정 짝 — 잔여 표적이 실제 선택을 개선하는가).
  · 성공 = J1 ∧ J2. 관찰 부속: 혼잡 상위 40% 부분집합 ρ_H (131-b 병기)·D_H 범위 5분위
    (신호 회복 확인)·plain D 기준 지표 병기. 진단·개발 — 판정·채택 아님.
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
from .yr128_rank_diagnosis import _spearman
from .yr129_diff_fit import _episode_labels

OUT = Path("outputs/reports/yr132_terminal_residual")
DATA129 = Path("outputs/reports/yr129_diff_fit")
B131 = Path("outputs/reports/yr131_k_candidates")
DIFF1 = Path("outputs/reports/yr125_diff_credit")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
LAM = 1.0
MARGIN, PAIR_EPS, PATIENCE, MAX_EPOCHS = 0.01, 0.01, 100, 2000
NEW_BAND = [(c, BASE[c] + 900 + i) for c in CELLS for i in range(2)]


def _cuts_with_dh(meta):
    rb = {}
    for m in meta:
        if m["is_base"]:
            rb.setdefault(m["dec"], m["resid"])
    rows = []
    for i, m in enumerate(meta):
        if m["is_base"]:
            continue
        dh = m["d"] + LAM * (m["resid"] - rb.get(m["dec"], m["resid"]))
        rows.append((i, m, dh))
    return rows


def _load129(ts: int):
    d = torch.load(DATA129 / f"dataset_s{ts}.pt", map_location="cpu")
    X, rows = d["X"], _cuts_with_dh(d["meta"])
    def cut(band):
        ii = [(i, m, dh) for i, m, dh in rows if m["band"] == band]
        return {"X": X[[i for i, _, _ in ii]],
                "y": torch.tensor([dh / SCALE for _, _, dh in ii], dtype=torch.float32),
                "dec": [m["dec"] for _, m, _ in ii]}
    return cut("train"), cut("novel")


def _rank(pred, y, dec, subset=None):
    by = {}
    for i, dc in enumerate(dec):
        if subset is None or dc in subset:
            by.setdefault(dc, []).append(i)
    top1, rhos, regret = [], [], []
    for ii in by.values():
        if len(ii) < 2:
            continue
        dh = [float(pred[i]) for i in ii]
        dd = [float(y[i]) for i in ii]
        top1.append(1.0 if dh.index(min(dh)) == dd.index(min(dd)) else 0.0)
        regret.append(dd[dh.index(min(dh))] - min(dd))
        if len(ii) >= 3:
            r = _spearman(dh, dd)
            if r is not None:
                rhos.append(r)
    return {"top1": round(fmean(top1), 4) if top1 else None,
            "rho": round(fmean(rhos), 4) if rhos else None,
            "regret_mean": round(fmean(regret) * SCALE, 4) if regret else None,
            "n_decisions": len(top1)}


def train132(ts: int) -> Path:
    tr, va = _load129(ts)
    torch.manual_seed(ts)
    net = JointPairNet(tr["X"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(ts)
    by = {}
    for i, dc in enumerate(tr["dec"]):
        by.setdefault(dc, []).append(i)
    pairs = [(a, b) for ii in by.values() for a in ii for b in ii
             if float(tr["y"][a]) + PAIR_EPS < float(tr["y"][b])]
    pairs = torch.tensor(pairs, dtype=torch.long)
    steps = max(1, len(tr["X"]) // 64)
    best = {"sel": -1.0, "epoch": 0, "state": None}
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        for _ in range(steps):
            bi = pairs[torch.randint(len(pairs), (64,), generator=g)]
            sa, _ = net(tr["X"][bi[:, 0]])
            sb, _ = net(tr["X"][bi[:, 1]])
            loss = nn.functional.margin_ranking_loss(sb, sa, torch.ones(len(bi)),
                                                     margin=MARGIN)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        net.eval()
        with torch.no_grad():
            pv, _ = net(va["X"])
        sel = _rank(pv, va["y"], va["dec"])["rho"] or -1.0
        if sel > best["sel"] + 1e-3:
            best = {"sel": sel, "epoch": ep, "state": copy.deepcopy(net.state_dict())}
        if ep - best["epoch"] >= PATIENCE:
            break
    net.load_state_dict(best["state"]); net.eval()
    out = OUT / f"dh_s{ts}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "in_dim": net.in_dim, "arm": "DH",
                "train_seed": ts, "best_epoch": best["epoch"]}, out / "net.pt")
    print(f"[132 s{ts}] 학습 완료 best_ep {best['epoch']} sel(novel ρ_H) {best['sel']:.4f}",
          flush=True)
    return out / "net.pt"


def _label_new_band(ts: int) -> Path:
    dst = OUT / f"dataset_new_s{ts}.pt"
    if dst.exists():
        return dst
    ck = torch.load(DIFF1 / f"diff1_s{ts}" / "rl_net.pt", map_location="cpu")
    net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
    norm = StateNorm(refs=ck["norm_refs"])
    rows_x, metas = [], []
    _set_forbid_wait(False)
    try:
        for cell, seed in NEW_BAND:
            lab = _episode_labels(net, norm, cell, seed, "new", f"{cell}:{seed}")
            rows_x += [r for r, _ in lab]
            metas += [m for _, m in lab]
            print(f"[label s{ts} {cell}:{seed}] 라벨 {len(lab)} 누적 {len(metas)}", flush=True)
    finally:
        _set_forbid_wait(True)
    torch.save({"X": torch.tensor(rows_x, dtype=torch.float32), "meta": metas,
                "scale": SCALE}, dst)
    return dst


def _eval_new(ts: int) -> dict:
    d = torch.load(OUT / f"dataset_new_s{ts}.pt", map_location="cpu")
    X, rows = d["X"], _cuts_with_dh(d["meta"])
    ii = [i for i, _, _ in rows]
    y_h = torch.tensor([dh / SCALE for _, _, dh in rows], dtype=torch.float32)
    y_d = torch.tensor([m["d"] / SCALE for _, m, _ in rows], dtype=torch.float32)
    dec = [m["dec"] for _, m, _ in rows]
    resid_by = {}
    for _, m, _ in rows:
        resid_by.setdefault(m["dec"], []).append(m["resid"])
    order = sorted(resid_by, key=lambda dc: fmean(resid_by[dc]))
    hi = set(order[int(len(order) * 0.6):])
    out = {}
    for name, ckp in (("YR132_DH", OUT / f"dh_s{ts}" / "net.pt"),
                      ("YR131_b", B131 / f"b_s{ts}" / "net.pt")):
        ck = torch.load(ckp, map_location="cpu")
        net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
        with torch.no_grad():
            p, _ = net(X[ii])
        out[name] = {"vs_DH": _rank(p, y_h, dec), "vs_DH_hi40": _rank(p, y_h, dec, hi),
                     "vs_D": _rank(p, y_d, dec)}
    return out


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for ts in TRAIN_SEEDS:
        _label_new_band(ts)
    for ts in TRAIN_SEEDS:
        train132(ts)
    ev = {ts: _eval_new(ts) for ts in TRAIN_SEEDS}
    for ts in TRAIN_SEEDS:
        print(f"[eval s{ts}] {json.dumps(ev[ts], ensure_ascii=False)}", flush=True)
    j1 = all((e["YR132_DH"]["vs_DH"]["rho"] or 0) >= 0.30
             and (e["YR132_DH"]["vs_DH"]["top1"] or 0) >= 0.35 for e in ev.values())
    j2 = all(e["YR132_DH"]["vs_DH"]["regret_mean"] < e["YR131_b"]["vs_DH"]["regret_mean"]
             for e in ev.values())
    judgment = {"J1_rank_newband": j1, "J2_regret_vs_131b": j2,
                "success": bool(j1 and j2),
                "per_seed": ev}
    res = {"repro": repro_stamp(
               experiment="YR-132 종결 잔여 표적 단일축 (D_H = D + λ·Δ잔여, λ=1.0 동결)",
               seeds={"train_ckpt": list(TRAIN_SEEDS),
                      "new_band": sorted({s for _, s in NEW_BAND})},
               profile_id="calibrated",
               prereg="유일 차이 = 표적 D→D_H (λ=1.0 가격앵커·개수 잔여는 진단용 지위). "
                      "학습 = 131-b 프로토콜·데이터 rollout 0. 판정 = 새 대역(BASE+900/901, "
                      "미열람) 라벨: J1 3/3 ρ_H≥0.30 ∧ top1≥0.35, J2 3/3 regret(132) < "
                      "regret(131-b) 같은 결정 짝. 성공 시 YR-123 비용가중 잔여로 승계. "
                      "진단 — 판정·채택 아님.",
               extra={"lambda": LAM, "margin": MARGIN, "pair_eps": PAIR_EPS}),
           "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps({"J1": j1, "J2": j2, "success": j1 and j2}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
    print("DONE")
