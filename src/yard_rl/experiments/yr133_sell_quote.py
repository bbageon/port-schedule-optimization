"""YR-133 1차 — 판매 발의·수신부담 견적·결정론 확정 파일럿 (spec 1차 계약 동결).

유일 변경 = review 규칙: yr105 혼잡 격차 → TransferQuoteResolver(견적 프로토콜).
실행정책 SF·물리·러너·비용 채널은 yr105 정본 그대로. KEEP arm = gain_margin=∞
(계산 경로 동일·확정만 차단 — yr113 패턴). 파일럿 = 기능 가드 판정(효과 확증은 후속).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from statistics import fmean

from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.cost_curve_v2 import KAPPA_V2P_PATH, KappaFit
from ..integrated.evalkit import CHANNELS, channel_split
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.repro import repro_stamp
from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S,
                                       GATE_BLOCK_MIN_S, GATE_BLOCK_SIGMA_S, trunc_normal)
from ..integrated.seedbank import assign_band, independence_report
from ..integrated.transfer_quote import TransferQuoteResolver
from .yr088_joint_rl import LEVEL
from .yr105_conditional_transfer import _CELLS, _gen_for_band, _sim, vessel_slack_min_s

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr133_sell_quote")
BAND_PATH = OUT / "band_pilot.json"
BAND_START, BAND_N = 906_000, 8       # 900k 이송 계열(901~905k)·910k 판정 계열 회피
CELL_A, CELL_B = _CELLS["A"], _CELLS["B"]


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
    band = assign_band(family="y133-pilot", cells=_CELLS, n=BAND_N,
                       generate=_gen_for_band, exclude=exclude, start_seed=BAND_START)
    rep = independence_report(band, forbidden={"past-recorded": exclude})
    assert rep["ok"], rep
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    BAND_PATH.write_text(json.dumps(
        {**band.freeze_json(), "independence": rep, "n_excluded_hashes": len(exclude),
         "created_commit": head}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[band] {sum(len(v) for v in band.seeds.values())} pilot seed 쌍 frozen")


def run_arm(seeds: dict[str, int], *, arm: str, tag: str) -> dict:
    """arm ∈ {quote, keep}. keep = gain_margin=∞ (계산 동일·확정 차단)."""
    kf = KappaFit.load(KAPPA_V2P_PATH, require_contract_physics=True)
    mbt = MultiBlockTerminal({"A": _sim(CELL_A, seeds["A"]),
                              "B": _sim(CELL_B, seeds["B"])})
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    stats = {"policy_exceptions": 0, "decisions": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        stats["decisions"] += 1
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            stats["policy_exceptions"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def travel_fn(src, dst, jid):
        # 2블록 골든 재현: dst 를 **의도적으로 무시**하고 난수 키도 구판 그대로 둔다.
        # 목적지별 주행(YR-150 0단계 route matrix)은 N블록 하네스에서만 쓴다.
        rng = random.Random(f"y133:{tag}:{src}:{jid}")
        return trunc_normal(rng, GATE_BLOCK_MEAN_S,
                            GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
                            lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)

    resolver = TransferQuoteResolver(
        kf, travel_fn=travel_fn, vessel_slack_fn=vessel_slack_min_s,
        gain_margin=(float("inf") if arm == "keep" else None))   # None = κ_T 1σ 유도(정정)

    chan = {k: 0.0 for k in CHANNELS}

    def cost_fn(sim, t0, t1, raw):
        cb = RC.cost_for(interval_start_s=t0, interval_end_s=t1, raw=raw, risk_max=0.0)
        for k, v in channel_split(cb.contributions()).items():
            chan[k] += v
        return cb.total_normalized

    res = mbt.run(policy, resolver.review, cost_fn)
    mbt.check_invariants()
    route_cost = res["route_cost_s"] / 3600.0
    a2o = mbt.ledger.a_to_o_samples_s(res["end"])
    per_epoch = {}
    for r in resolver.ledger:
        if r["decision"] == "TRANSFER":
            per_epoch[r["t"]] = per_epoch.get(r["t"], 0) + 1
    max_tc = max((rec.transfer_count for rec in mbt.ledger.records.values()), default=0)
    return {"arm": arm, "seeds": seeds,
            "total": float(res["terminal_total"] + route_cost),
            "route_cost": route_cost,
            "a2o_mean_min": (fmean(a2o) / 60.0) if a2o else None,
            "backlog": sum(s.unfinished_backlog() for s in mbt.blocks.values()),
            "compl": (sum(1 for s in mbt.blocks.values()
                          for j in s.jobs.values() if j.status.name == "DONE")
                      / max(1, sum(len(s.jobs) for s in mbt.blocks.values()))),
            "policy_exceptions": stats["policy_exceptions"],
            "n_transferred": resolver.n_transferred,
            "max_epoch_transfers": max(per_epoch.values(), default=0),
            "max_transfer_count_per_job": max_tc,
            "quote_ledger": resolver.ledger,
            "chan": {k: round(v, 4) for k, v in chan.items()}}


def pilot() -> dict:
    d = json.loads(BAND_PATH.read_text(encoding="utf-8"))
    pairs = list(zip(d["seeds"]["A"], d["seeds"]["B"]))
    rows = {"quote": [], "keep": []}
    for i, (sa, sb) in enumerate(pairs):
        for arm in ("quote", "keep"):
            print(f"[pilot] {arm} pair{i}", flush=True)
            rows[arm].append(run_arm({"A": sa, "B": sb}, arm=arm, tag=f"p{i}"))
    guards = {}
    for arm in ("quote", "keep"):
        eps = rows[arm]
        guards[arm] = {
            "compl_min": min(e["compl"] for e in eps),
            "backlog_max": max(e["backlog"] for e in eps),
            "policy_exceptions": sum(e["policy_exceptions"] for e in eps),
            "max_epoch_transfers": max(e["max_epoch_transfers"] for e in eps),
            "max_transfer_count_per_job": max(e["max_transfer_count_per_job"]
                                              for e in eps),
            "n_transferred_total": sum(e["n_transferred"] for e in eps)}
    j = {"완주": all(g["backlog_max"] == 0 for g in guards.values()),
         "예외0": all(g["policy_exceptions"] == 0 for g in guards.values()),
         "epoch1건": guards["quote"]["max_epoch_transfers"] <= 1,
         "이송상한": guards["quote"]["max_transfer_count_per_job"] <= 1,
         "KEEP무이송": guards["keep"]["n_transferred_total"] == 0,
         "이송발생": guards["quote"]["n_transferred_total"] > 0}
    j["success"] = all(j.values())
    diffs = [q["total"] - k["total"] for q, k in zip(rows["quote"], rows["keep"])]
    res = {"repro": repro_stamp(
               experiment="YR-133 1차 — 판매·수용 견적 파일럿 (기능 가드 판정)",
               seeds={"A": d["seeds"]["A"], "B": d["seeds"]["B"]},
               profile_id="calibrated",
               prereg="1차 계약: top-1 OFFER·최소 InBurden·NetGain>0 결정론 확정·"
                      "epoch 1건·이송/작업 ≤1·quote epoch 전용·fail-closed·본선 가드·"
                      "κ 동결. 성공 = 기능 가드 전부(효과 확증은 후속 단일축 — 방향은 "
                      "보고 전용).",
               extra={"band_digest": d["digest"]}),
           "guards": guards, "judgment": j,
           "pair_total_diff_참고": [round(x, 3) for x in diffs],
           "arms": rows}
    (OUT / "pilot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(json.dumps({**j, **guards["quote"],
                      "diff_mean_참고": round(fmean(diffs), 3) if diffs else None},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-band", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.make_band:
        make_band()
    if a.pilot:
        pilot()
    print("DONE")
