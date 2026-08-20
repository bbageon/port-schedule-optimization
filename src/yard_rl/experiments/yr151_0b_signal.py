"""YR-151 0B — 판매 신호 예선 (사전등록: strategy-history 2026-08-09-YR-151-0B-사전등록).

판정 2축(동결 — 둘 다 통과해야 GO):
  ①상금 존재: 10런 전부 would-commit ≥1 그리고 시드별 Σ(−ΔJ) > 0
  ②구분력: pooled Spearman(소스 혼잡 특징, −ΔJ) ≥ 0.2
셀 = w5-L100·w5-L150 × 개발 시드 5(+10M~+14M 대역·잠금 밴드와 분리).
shadow dry-run(본 실행 불변 실측 승계)·미학습 TransferHead(초기화 7,000,000)·
채택 PPO 실행·관측창 정본·오차 0. 성능 주장 없음 — 신호 존재·구분력만.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..integrated.baselines import _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.profiles import build_h21_profile
from ..integrated.repro import code_dirty
from ..integrated.sell_review import (ANNOUNCE_LEAD_S, UnifiedSellOrchestrator)
from ..integrated.terminal_stream import (ObservationContract,
                                          TerminalStreamParams,
                                          WipAdmissionController,
                                          admission_epochs, build_fixed_wip,
                                          hotspot_rotation)
from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
from ..integrated.yard_layout import terminal_layout
from .yr088_joint_rl import LEVEL
from .yr149_load_cells import _sim_from
from .yr150_h21_pilot import _git, _sha256
from .yr151_transfer_ppo import (AdoptedExecFleet, exec_config_hash,
                                 load_adopted_execution_head, load_kf)
from .yr157_band_qual import N_HOTSPOT, SEED as BAND_SEED, cell_seed

OUT = Path("outputs/reports/yr151_0b_signal")
PREREG = Path(".claude/docs/strategy-history/2026-08-09-YR-151-0B-사전등록.md")
CELLS = ((5.0, 100), (5.0, 150))          # 동결 — 안정 BUSY 무대
DEV_REPS = 5                              # 동결 — 셀당 개발 시드 5
DEV_BASE = 10_000_000                     # 동결 — 개발 대역 (+10M~+14M)
NET_INIT = 7_000_000                      # 동결 — 미학습 초기화 시드
RHO_MIN = 0.2                             # 동결 — 구분력 임계


def dev_seed(w: float, load: int, rep: int) -> int:
    return cell_seed(w, load) + DEV_BASE + rep * 1_000_000


def _spearman(xs: list[float], ys: list[float]) -> float:
    """순위상관 — 동순위 평균 처리, 순수 파이썬(결정론)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def run_idx(idx: int) -> Path:
    """개발 런 1개 — (셀, rep) = (idx // DEV_REPS, idx % DEV_REPS)."""
    w, load = CELLS[idx // DEV_REPS]
    rep = idx % DEV_REPS
    seed = dev_seed(w, load, rep)
    obs = ObservationContract()
    layout = terminal_layout()
    hs = hotspot_rotation(layout, seed, N_HOTSPOT)
    params = TerminalStreamParams(load_4h=load, hotspot_blocks=hs, hotspot_weight=w)
    built = build_fixed_wip(build_h21_profile(), seed, wip_target=load, obs=obs,
                            layout=layout, params=params,
                            background_seed=BAND_SEED + DEV_BASE + rep * 1_000_000)
    mbt = MultiBlockTerminal({b: _sim_from(s) for b, s in built["scenarios"].items()},
                             extra_review_epochs=admission_epochs(obs))
    ctrl = WipAdmissionController(built["pool"], wip_target=load,
                                  lead_s=ANNOUNCE_LEAD_S, end_s=obs.observe_s)
    torch.manual_seed(NET_INIT)
    policy = PpoSellPolicy(TransferActor(), TransferCritic(), mode="shadow",
                           sample=True, seed=NET_INIT + idx, layout=layout)
    orch = UnifiedSellOrchestrator(policy, layout, load_kf(), dry_run=True)
    exec_actor, exec_norm = load_adopted_execution_head()
    h0 = exec_config_hash(exec_actor, 221_000, ADOPTED_C0_GUARD)
    fleet = AdoptedExecFleet(exec_actor, exec_norm, config=ADOPTED_C0_GUARD)
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def exec_policy(sim, dp):
        g = gens.setdefault(
            id(sim), CandidateGenerator(config=ADOPTED_C0_GUARD))
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, fleet.get(sim).decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def review(mbt_, t):
        ctrl.review(mbt_, t)
        orch.review(mbt_, t)

    mbt.run(exec_policy, review_fn=review)
    if exec_config_hash(exec_actor, 221_000, ADOPTED_C0_GUARD) != h0:
        raise RuntimeError("실행 구성 변조 — 계약 위반")

    # 특징 join: 결정 (t, src, picked) → 선택 행의 소스 혼잡 특징(내부/10+통근중/10)
    feat = {}
    for tr in policy.trail:
        if tr["picked"] is not None:
            row = tr["rows"][tr["action"]]
            feat[(round(tr["t"], 6), tr["src"], tr["picked"])] = float(row[0] + row[1])
    records = []
    for e in orch.ledger:
        if e["decision"] == "DRY_WOULD_COMMIT":
            k = (round(e["t"], 6), e["src"], e["job_id"])
            records.append({"gain": -e["delta_j"], "feat": feat.get(k),
                            "axis": e["axis"]})
    part = {"idx": idx, "cell": f"w{w}-L{load}", "rep": rep, "seed": seed,
            "n_would": len(records),
            "sum_gain": round(sum(r["gain"] for r in records), 6),
            "n_decisions": len(policy.trail), "exec_exceptions": exc["n"],
            "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"ob_idx{idx}.json"
    p.write_text(json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: part[k] for k in
                      ("idx", "cell", "rep", "n_would", "sum_gain")},
                     ensure_ascii=False))
    return p


def merge() -> dict:
    parts = [json.loads((OUT / f"ob_idx{i}.json").read_text(encoding="utf-8"))
             for i in range(len(CELLS) * DEV_REPS)]
    axis1 = all(p["n_would"] >= 1 and p["sum_gain"] > 0 for p in parts)
    pooled = [(r["feat"], r["gain"]) for p in parts for r in p["records"]
              if r["feat"] is not None]
    rho = _spearman([x for x, _ in pooled], [y for _, y in pooled]) if pooled else 0.0
    axis2 = rho >= RHO_MIN
    verdict = {
        "signal_go": bool(axis1 and axis2),
        "axis1_prize_exists": axis1,
        "axis2_discriminability": axis2,
        "spearman_rho": round(rho, 4), "rho_min": RHO_MIN,
        "n_pooled": len(pooled),
        "per_run": [{k: p[k] for k in ("cell", "rep", "n_would", "sum_gain")}
                    for p in parts],
        "note": "0B = 신호 존재·구분력 판정. ΔJ 는 계산 견적(proxy) — 실현 개선은 "
                "본 학습·최종 비교의 몫. 규모 주장 금지·무오차 조건 한정.",
    }
    dirty = bool(code_dirty())
    res = {"task": "YR-151-0B", "runtime": {
        "commit": _git("rev-parse", "HEAD"), "git_dirty": dirty,
        "prereg_file": str(PREREG),
        "prereg_sha256": _sha256(PREREG) if PREREG.exists() else None,
        "params": {"CELLS": [list(c) for c in CELLS], "DEV_REPS": DEV_REPS,
                   "DEV_BASE": DEV_BASE, "NET_INIT": NET_INIT,
                   "RHO_MIN": RHO_MIN}},
        "verdict": verdict}
    p = OUT / "ob_signal.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "ob_signal.json.sha256").write_text(_sha256(p) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": {k: v for k, v in verdict.items()
                                  if k != "per_run"}, "dirty": dirty},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.idx is not None:
        run_idx(a.idx)
    elif a.merge:
        merge()
    print("DONE")
