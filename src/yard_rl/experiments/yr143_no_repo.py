"""YR-143 — 전략적 YC 위치조정 없음(C0) vs 결속 PREPOSITION(C1): 분산 파일럿 단계.

■ 계약 (spec YR-143 — 착수 사전등록 동결):
  C0 = SAFETY_ONLY (SERVE+PRE_REHANDLE·교착 탈출만) / C1 = 결속 PREPO(one-shot·PLANNED
  한정). **양군 공통 = R 계약**(유한 DEFER + 반사실 순위손실 — YR-147 채택판). 유일 차이
  = PREPOSITION 유무. 학습예산·PPO·상태·보상·초기 가중치(같은 ts)·시나리오 순서 동일.
  정규화 참조 = diff1_s88000 고정(norm_refs 3초기화 동일 검증 d3082288).
■ 파일럿 (판정 아님): 신규 학습초기화 4쌍(211000~214000) × 신규 평가판 16(셀별 4,
  커서 910056+). **평균은 선택에 쓰지 않고 분산만 봉인**·최종 확증에 불합산.
  표본 계산: 단측 α=0.05·검정력 90%·파일럿 sd 의 80% 상측한계(χ²_{0.20,15}=10.307).
■ Δ = C0 − C1 (음수 = C0 우월·양수 = C1 우월 — spec 방향 고정).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import fmean, stdev

import torch

from ..integrated import candidates as cand_mod
from ..integrated.encoding import StateNorm
from ..integrated.joint_distill import JointPairNet
from ..integrated.seedbank import (BandSpec, assign_band, independence_report,
                                   realization_hash)
from . import yr088_joint_rl as y88
from .yr090_dense_vessel import CELLS
from .yr136_softplus_contract import _sim_contract
from .yr141_bound_prepo import _episode
from .yr147_defer import train_one_v3

OUT = Path("outputs/reports/yr143_no_repo")
BAND_PATH = OUT / "band_pilot.json"
BAND_START, BAND_CEIL, BAND_N = 910_056, 920_000, 4
PILOT_TS = (211_000, 212_000, 213_000, 214_000)
NORM_TS = 88_000
# δ (가정값 — 실 SLA 부재 명시·민감도 ±50% 병행 보고): v2 1.0 / 본선 3.0분 / v1 2.0
DELTA = {"v2": 1.0, "berth": 3.0, "v1": 2.0}
ARM_FLAGS = {"c0": {"safety_only": True, "bound": False},
             "c1": {"safety_only": False, "bound": True}}


def _collect_hashes() -> set[str]:
    import re
    pat = re.compile(r"rz1:[0-9a-f]{16}")
    got: set[str] = set()
    for p in Path("outputs/reports").rglob("*.json"):
        if OUT in p.parents:
            continue
        try:
            got |= set(pat.findall(p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return got


def make_band():
    exclude = _collect_hashes()
    band = assign_band(family="yr143-pilot", cells={c: None for c in CELLS}, n=BAND_N,
                       generate=lambda key, _p, seed: _sim_contract(key, seed).scenario,
                       exclude=exclude, start_seed=BAND_START)
    for ss in band.seeds.values():
        for s in ss:
            assert BAND_START <= s < BAND_CEIL, s
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] {sum(len(v) for v in band.seeds.values())} pilot seeds frozen")


def _load_band():
    d = json.loads(BAND_PATH.read_text(encoding="utf-8"))
    band = BandSpec(family=d["family"], seeds=d["seeds"], hashes=d["realization_hashes"])
    for cell, ss in band.seeds.items():
        for s, h in zip(ss, band.hashes[cell]):
            assert realization_hash(_sim_contract(cell, s).scenario) == h, f"{cell}:{s}"
    rep = independence_report(band, forbidden={"past-recorded": _collect_hashes()})
    assert rep["ok"], rep
    return band


def train(ts: int, arm: str):
    f = ARM_FLAGS[arm]
    return train_one_v3(ts, "r", out_base=OUT / arm, norm_ts=NORM_TS,
                        safety_only=f["safety_only"], bound=f["bound"])


def _mk(arm: str, ts: int):
    ck = torch.load(OUT / arm / f"ppo_s{ts}" / "net.pt", map_location="cpu")
    actor = JointPairNet(250); actor.load_state_dict(ck["actor"]); actor.eval()
    ck0 = torch.load(Path("outputs/reports/yr125_diff_credit") / f"diff1_s{NORM_TS}"
                     / "rl_net.pt", map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])

    def mk():
        y88.FORBID_WAIT = True
        return y88.RLPolicy(actor, norm, name=f"{arm}:{ts}")
    return mk


def _episode_arm(cell, seed, arm, ts):
    f = ARM_FLAGS[arm]
    prev = cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY
    cand_mod.WAIT_MODE = "DEFER_ALL"
    cand_mod.SAFETY_ONLY = f["safety_only"]
    try:
        return _episode(cell, seed, _mk(arm, ts), bound=f["bound"], one_shot=True)
    finally:
        cand_mod.WAIT_MODE, cand_mod.SAFETY_ONLY = prev


def pilot() -> dict:
    band = _load_band()
    eval_eps = [(c, s) for c in CELLS for s in band.seeds[c]]
    rows = {}
    for arm in ("c0", "c1"):
        for ts in PILOT_TS:
            print(f"[pilot] {arm}:{ts}", flush=True)
            rows[f"{arm}:{ts}"] = [_episode_arm(c, s, arm, ts) for c, s in eval_eps]
    # 필수 테스트 계수 — C0 능동 위치조정 실행 0 · C1 비결속 위치조정 0 · 가드
    contract = {}
    for arm in ("c0", "c1"):
        eps = [e for ts in PILOT_TS for e in rows[f"{arm}:{ts}"]]
        contract[arm] = {
            "compl_min": min(e["compl"] for e in eps),
            "backlog_max": max(e["backlog"] for e in eps),
            "healthy_all": all(e["healthy"] for e in eps),
            "repo_share_max": max(e["shares"].get("REPOSITION", 0) for e in eps),
            "prepo_exec_total": sum(e["prepo_exec_total"] for e in eps)}
    # 분산 봉인 — 시나리오(4초기화 평균) 단위 짝 차이 Δ = C0 − C1
    metrics = {"v2": "v2_total", "berth": "berth_over_min", "v1": "v1_total"}
    seal = {}
    CHI2_20_DF15 = 10.307
    for m, key in metrics.items():
        diffs = []
        for j in range(len(eval_eps)):
            c0m = fmean(rows[f"c0:{ts}"][j][key] for ts in PILOT_TS)
            c1m = fmean(rows[f"c1:{ts}"][j][key] for ts in PILOT_TS)
            diffs.append(c0m - c1m)
        sd = stdev(diffs)
        n = len(diffs)
        sd_ucl = sd * ((n - 1) / CHI2_20_DF15) ** 0.5
        n_req = 8.57 * (sd_ucl ** 2) / (DELTA[m] ** 2)      # (1.645+1.282)^2 ≈ 8.57
        seal[m] = {"sd": sd, "sd_ucl80": sd_ucl, "delta": DELTA[m],
                   "n_required": n_req,
                   "mean_참고_선택사용금지": fmean(diffs)}
    res = {"contract_checks": contract, "variance_seal": seal,
           "n_scen_pilot": len(eval_eps), "pilot_ts": list(PILOT_TS),
           "arms_raw": rows,
           "note": "파일럿 — 평균은 선택 사용 금지·확증 불합산 (spec 검정력 계약)"}
    (OUT / "pilot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(json.dumps({m: {k: round(v, 4) for k, v in s.items()}
                      for m, s in seal.items()}, ensure_ascii=False))
    print(json.dumps(contract, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-band", action="store_true")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--arm", choices=("c0", "c1"))
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.train:
        assert a.arm
        train(a.train, a.arm)
    if a.pilot:
        pilot()
    print("DONE")
