"""YR-125 2단계 — 차분 신용 이식: 학습 표적 단일축 (외부 피드백 계약 7항 동결, 결과 미열람).

■ 무엇이 유일한 차이인가
base = YR-119 WAITON recipe (WAIT 학습·실행 허용, 시드 {88000,99000,123000}, 500ep,
lr 5e-4·배치 64·에피소드당 갱신 max(1,n//64)·ε 1.0→0.05·EMA 스냅샷·같은 신경망·정규화).
**유일 차이 = 학습 표적**: TD 부트스트랩 절대비용 회귀 → **1-step 반사실 차분 회귀**.
표적 교체는 정의상 shaping·UNSERVED·γ 항의 제거를 포함한다(차분 표적은 창 내 실측
비용만 계상) — 부가 축이 아니라 표적 정의의 일부임을 고지.

■ 계약 (외부 피드백 2026-07-29 — 7항 동결)
 1. 차분값: D(a) = C600(a) − C600(기준행동). D<0 = a 가 기준(대기)보다 유리.
    망은 D 를 직접 예측하고 실행은 argmin D̂ (전원 WAIT 조합의 표적은 정확히 0).
 2. 공동 기준행동: **결정에 참여한 크레인만** 전략적 WAIT, 진행 중·비참여 크레인은
    원래 상태 유지 (rollout 이 그대로 이어감 — 자동 충족).
 3. 강제 WAIT 표본 제외: 실행가능 실작업 조합이 0개인 결정(선택지 없음·경합 패배·
    안전거리 등 구조적 WAIT)은 **학습 표본에서 제외** (실행은 하되 회귀에 안 씀).
 4. 동일 반사실 조건: 같은 sim 상태 deepcopy·같은 외생 이벤트(결정적 시뮬레이터)·
    같은 후속 정책(SF)·같은 종료시각 t0+600s 에서 C600(a)·C600(기준) 계산.
 5. 창 절단: 두 rollout 이 같은 시각에 같은 규칙으로 절단되므로 차분에서 공통 잔여는
    상쇄된다. 창 민감도(300/1200s 순위 유지)는 **후속 진단**(판정 외)으로 분리.
 6. 미래정보 범위: C600 은 **학습 교사정보만**(CTDE 특권, YR-107) — 실행망 입력은
    현재 관측뿐, 배포 시 미래정보 없음.
 7. 판정(성공 = P1~P5 전부, 3/3 시드 방향 일관 포함):
    P1 순위 회복 — YR-128 동일 재진단에서 3/3 시드 P(cw|qw) ≥ 0.5 그리고
       top-1 일치 ≥ 0.35 그리고 Spearman ρ ≥ 0.30 (참조 최고: 0.33/0.27/0.16).
    P2 WAIT 건전화 — 전략적 WAIT 선택률 전체평균 < 0.479(WAITON 재평가) AND
       WAIT 행동점유 >0.60 에피소드 0.
    P3 운영 비악화 — 짝지은 총비용(DIFF1−WAITON) upper95 < +10 AND A→O upper95 < +1.0분.
    P4 풍선 없음 — REPOSITION 비중 평균 ≤ 0.15 AND REPO 점유 >0.60 에피소드 0.
    P5 guard — 완주 1.0·backlog 0·미건전 수 ≤ WAITON.
    성공해도 "차분이 해결책 증명"이 아니라 진단·개발 단계 — 채택은 별도 사전등록.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
import time
from collections import deque
from pathlib import Path
from statistics import fmean

import torch
from torch import nn

from ..contract.schema import CandidateKind
from ..integrated.baselines import (JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference, _apply, _rollout_cost,
                                    _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.evalkit import paired
from ..integrated.joint_distill import JointPairNet
from ..integrated.repro import repro_stamp
from . import yr090_dense_vessel as y90
from .yr088_joint_rl import LEVEL, RC as RC_TRAIN, build_rows
from .yr090_dense_vessel import BASE, CELLS, SCALE, _sim
from .yr119_wait_retrain import EVAL_SEEDS, _episode, _rl_policy_factory, _set_forbid_wait
from .yr128_rank_diagnosis import _episode_ranks, _summ

OUT = Path("outputs/reports/yr125_diff_credit")
YR119 = Path("outputs/reports/yr119_wait_retrain")
TRAIN_SEEDS = (88_000, 99_000, 123_000)
HORIZON_S = 600.0
DELTA = {"total": 10.0, "a2o_min": 1.0}
DIAG_EPS = [(c, BASE[c] + 700 + i) for c in CELLS for i in range(2)]


def _collect_diff(cell, seed, net, norm, epsilon, rng):
    """1 에피소드 실행 + 차분 표본 수집. 표본 = (실행 조합 row, D/SCALE)."""
    sim = _sim(cell, seed)
    gen = CandidateGenerator()
    jr = JointRolloutGreedy(RC_TRAIN, horizon_s=1800.0, generator=gen,
                            forbid_strategic_wait=False)
    base_pol = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    samples = []
    dp = sim.run_until_decision()
    k = 0
    while dp is not None:
        gen_by = {c: gen.generate(sim, c, LEVEL) for c in dp.crane_ids}
        rows, assigns = build_rows(sim, dp, gen_by, norm, jr, k)
        if not assigns:
            _apply(sim, {c: _wait_of(gen_by[c]) for c in dp.crane_ids})
        else:
            if net is None or rng.random() < epsilon:
                pick = rng.randrange(len(assigns))
            else:
                with torch.no_grad():
                    sc, _ = net(torch.tensor(rows, dtype=torch.float32))
                q = [float(x) for x in sc]
                pick = q.index(min(q))
            full_exists = any(
                all(a[c].kind != CandidateKind.WAIT for c in dp.crane_ids)
                for a in assigns)
            if full_exists:                       # 계약 3: 구조적 WAIT 결정은 표본 제외
                baseline = {c: _wait_of(gen_by[c]) for c in dp.crane_ids}
                ca, _ = _rollout_cost(sim, assigns[pick], RC_TRAIN, horizon_s=HORIZON_S,
                                      base_policy=base_pol, generator=gen)
                cb, _ = _rollout_cost(sim, baseline, RC_TRAIN, horizon_s=HORIZON_S,
                                      base_policy=base_pol, generator=gen)
                samples.append((rows[pick], (ca - cb) / SCALE))
            _apply(sim, assigns[pick])
        dp = sim.run_until_decision()
        k += 1
    return samples


def train_one_diff(seed: int, episodes: int = 500, spc: int = 16,
                   batch: int = 64, lr: float = 5e-4) -> Path:
    out = OUT / f"diff1_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    prof = y90.build_calibrated_profile()
    norm, _ = y90.fit_state_norm(
        prof, dataclasses.replace(y90.calibrated_load_params("high", vessel_deadline_mult=0.5),
                                  time_contract_v2=True),
        [BASE["high-tight"] + i for i in range(5)], progress=lambda *_: None)
    rng = random.Random(seed); torch.manual_seed(seed)
    cells = list(CELLS)
    tr = {c: [BASE[c] + i for i in range(spc)] for c in cells}
    va = {c: [BASE[c] + 50 + i for i in range(4)] for c in cells}
    replay = deque(maxlen=40_000)
    net = ema = opt = None
    best = {"val": float("inf"), "state": None, "ep": 0}
    t0 = time.time()
    n_targets = n_updates = 0
    for ep in range(1, episodes + 1):
        eps_ = max(0.05, 1.0 - ep / episodes)
        cell = cells[ep % len(cells)]
        samples = _collect_diff(cell, tr[cell][rng.randrange(spc)], net, norm,
                                eps_ if net else 1.0, rng)
        n_targets += len(samples)
        if net is None and samples:
            net = JointPairNet(len(samples[0][0]))
            ema = copy.deepcopy(net)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
        replay.extend(samples)
        if net is not None and len(replay) >= batch:
            for _ in range(max(1, len(samples) // batch)):    # base recipe 와 동일 규칙
                b = rng.sample(list(replay), batch)
                x = torch.tensor([r for r, _ in b], dtype=torch.float32)
                y = torch.tensor([d for _, d in b], dtype=torch.float32)
                sc, _ = net(x)
                loss = nn.functional.smooth_l1_loss(sc, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                y90._soft(ema, net, 0.01)
                n_updates += 1
        if net is not None and ep % 25 == 0:
            ema.eval(); rows_v = y90._val(ema, norm, cells, va); ema.train()
            w = fmean(r[0] for r in rows_v); bo = fmean(r[1] for r in rows_v)
            compl = fmean(r[2] for r in rows_v)
            hl = fmean(1.0 if r[3] else 0.0 for r in rows_v)
            score = w + 0.3 * bo + 300.0 * (1 - compl) + 100.0 * (1 - hl)
            if score < best["val"]:
                best = {"val": score, "state": copy.deepcopy(ema.state_dict()), "ep": ep}
            print(f"[DIFF1 s{seed} ep{ep}] wait={w:.2f} berth={bo:.1f} "
                  f"healthy={hl:.2f} compl={compl:.3f}", flush=True)
    ema.load_state_dict(best["state"]); ema.eval()
    torch.save({"state": ema.state_dict(), "in_dim": ema.in_dim, "norm_refs": norm.refs,
                "best_ep": best["ep"], "arm": "DIFF1", "train_seed": seed},
               out / "rl_net.pt")
    meta = {"n_targets": n_targets, "n_updates": n_updates,
            "wall_s": round(time.time() - t0, 1), "horizon_s": HORIZON_S}
    (out / "train_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[train DIFF1 s{seed}] {meta}", flush=True)
    return out / "rl_net.pt"


def train(seed: int, episodes: int = 500) -> Path:
    _set_forbid_wait(False)
    try:
        return train_one_diff(seed, episodes=episodes)
    finally:
        _set_forbid_wait(True)


def judge() -> dict:
    from ..integrated.encoding import StateNorm
    OUT.mkdir(parents=True, exist_ok=True)
    eps = [(c, s) for c in CELLS for s in EVAL_SEEDS[c]]
    print(f"[eval] SF 기준선 {len(eps)}", flush=True)
    sf = [_episode(c, s, lambda: ResolverPolicy(ServiceFirstSPTPreference(), "SF"))
          for c, s in eps]
    rows: dict[str, list[dict]] = {}
    for ts in TRAIN_SEEDS:
        for name, ckdir in (("WAITON", YR119 / f"waiton_s{ts}"),
                            ("DIFF1", OUT / f"diff1_s{ts}")):
            print(f"[eval] {name}:{ts}", flush=True)
            mk = _rl_policy_factory(ckdir / "rl_net.pt", False)
            rows[f"{name}:{ts}"] = [_episode(c, s, mk) for c, s in eps]
    print("[diag] P1 순위 재진단", flush=True)
    rank = {}
    _set_forbid_wait(False)
    try:
        for ts in TRAIN_SEEDS:
            ck = torch.load(OUT / f"diff1_s{ts}" / "rl_net.pt", map_location="cpu")
            net = JointPairNet(ck["in_dim"]); net.load_state_dict(ck["state"]); net.eval()
            norm = StateNorm(refs=ck["norm_refs"])
            recs = []
            for cell, seed in DIAG_EPS:
                recs += _episode_ranks(net, norm, cell, seed)
            rank[ts] = _summ(recs)
            print(f"[diag DIFF1 s{ts}] {json.dumps(rank[ts])}", flush=True)
    finally:
        _set_forbid_wait(True)
    d_all = [r for ts in TRAIN_SEEDS for r in rows[f"DIFF1:{ts}"]]
    w_all = [r for ts in TRAIN_SEEDS for r in rows[f"WAITON:{ts}"]]
    p1 = {"per_seed": rank,
          "pass": all(s["R3_p_cw_given_qw"] is not None and s["R3_p_cw_given_qw"] >= 0.5
                      and s["R1_top1_agree"] >= 0.35
                      and (s["R2_spearman_mean"] or 0.0) >= 0.30 for s in rank.values())}
    swr = fmean(r["strategic_wait_rate"] for r in d_all)
    dom_w = sum(1 for r in d_all if r["shares"].get("WAIT", 0) > 0.60)
    p2 = {"strategic_wait_rate": round(swr, 4), "wait_dominates": dom_w,
          "pass": bool(swr < 0.479 and dom_w == 0)}
    d_tot = [a["total"] - b["total"] for ts in TRAIN_SEEDS
             for a, b in zip(rows[f"DIFF1:{ts}"], rows[f"WAITON:{ts}"])]
    d_a2o = [a["a2o_min"] - b["a2o_min"] for ts in TRAIN_SEEDS
             for a, b in zip(rows[f"DIFF1:{ts}"], rows[f"WAITON:{ts}"])
             if a["a2o_min"] is not None and b["a2o_min"] is not None]
    pt, pa = paired(d_tot, delta_interest=DELTA["total"]), paired(d_a2o, delta_interest=DELTA["a2o_min"])
    p3 = {"d_total": pt.as_dict(), "d_a2o": pa.as_dict(),
          "pass": bool(pt.ci_hi < DELTA["total"] and pa.ci_hi < DELTA["a2o_min"])}
    repo = fmean(r["repo_share"] for r in d_all)
    dom_r = sum(1 for r in d_all if r["shares"].get("REPOSITION", 0) > 0.60)
    p4 = {"repo_share_mean": round(repo, 4), "repo_dominates": dom_r,
          "pass": bool(repo <= 0.15 and dom_r == 0)}
    unh_d = sum(1 for r in d_all if not r["healthy"])
    unh_w = sum(1 for r in w_all if not r["healthy"])
    p5 = {"compl_min": min(r["compl"] for r in d_all),
          "backlog_max": max(r["backlog"] for r in d_all),
          "unhealthy": {"DIFF1": unh_d, "WAITON": unh_w},
          "pass": bool(min(r["compl"] for r in d_all) >= 1.0
                       and max(r["backlog"] for r in d_all) == 0 and unh_d <= unh_w)}
    sf3 = sf * len(TRAIN_SEEDS)
    ref = {name: paired([r["total"] - s["total"] for r, s in zip(allr, sf3)],
                        delta_interest=DELTA["total"]).as_dict()
           for name, allr in (("DIFF1_vs_SF", d_all), ("WAITON_vs_SF", w_all))}
    meta = {ts: json.loads((OUT / f"diff1_s{ts}" / "train_meta.json").read_text(encoding="utf-8"))
            for ts in TRAIN_SEEDS if (OUT / f"diff1_s{ts}" / "train_meta.json").exists()}
    judgment = {"P1_rank": p1, "P2_wait": p2, "P3_cost": p3, "P4_balloon": p4,
                "P5_guard": p5,
                "success": bool(p1["pass"] and p2["pass"] and p3["pass"]
                                and p4["pass"] and p5["pass"]),
                "reference_vs_sf": ref, "train_meta": meta}
    res = {"repro": repro_stamp(
               experiment="YR-125 2단계 — 차분 신용 이식 (학습 표적 단일축, 계약 7항)",
               seeds={"train": list(TRAIN_SEEDS), **{c: EVAL_SEEDS[c] for c in CELLS},
                      "diag": sorted({s for _, s in DIAG_EPS})},
               profile_id="calibrated",
               prereg="유일 차이 = 학습 표적(TD 절대비용 → 1-step 차분 D). 계약 7항 및 "
                      "P1(순위: P(cw|qw)≥0.5·top1≥0.35·rho≥0.30 3/3)·P2(전략WAIT<0.479·"
                      "도미넌스 0)·P3(총비용<+10·A→O<+1분 upper95)·P4(REPO≤0.15·장악 0)·"
                      "P5(완주·backlog·미건전) 동결. 진단·개발 — 채택 아님.",
               extra={"horizon_s": HORIZON_S, "delta": DELTA}),
           "sf": sf, "arms": rows, "rank_diag": rank, "judgment": judgment}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    if a.train:
        train(a.train, episodes=a.episodes)
    if a.judge:
        r = judge()
        print(json.dumps(r["judgment"], ensure_ascii=False, indent=1))
    print("DONE")
