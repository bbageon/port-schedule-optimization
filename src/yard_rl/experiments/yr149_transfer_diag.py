"""YR-149 1단계 — 결정시점 반사실 진단 (사슬 한계효과 리플레이, 동결).

■ 대상: YR-133 보정 파일럿의 확정 이송 14건 (원장·시드·이동시간 전부 보존 — 정확 재현).
■ 방법(동결): 에피소드별 확정 이송 목록 T=[d1..dn] 을 시각 순으로 두고, run_j = "앞의
  j건만 그대로 실행·이후 이송 없음" (j=0..n). **k번째 결정의 한계효과 = run_k − run_{k−1}**
  — 원 궤적의 결정시점 상태를 정확히 복제(같은 시드·같은 travel rng 키). 사후 실현정보는
  진단 라벨 전용 — 배포 견적 입력 금지.
■ 분해(동결): 실현 한계효과를 v1 채널로 분해 — 소스 파급 = Δ(소스 블록 truck 채널),
  수신 파급 = Δ(수신 블록 truck 채널), 본선 파급 = Δ(vessel 채널), 이동 = Δ(move).
  예측 대응 잔차: 소스 잔차 = Δ소스 + OutRelief · 수신 잔차 = Δ수신 − InBurden ·
  본선 잔차 = Δ본선 − 0 · 이동 잔차 = Δ이동 − route/3600. **지배 축 = |에피소드 묶음
  평균 잔차| 최대인 축** (소스/수신/본선 — 2단계는 그 한 축만 보정).
■ 통계 단위: 14건은 진단 상태 수 — 에피소드 seed 쌍으로 묶어 보고. v2 실현 총비용 병기.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean, median

from ..integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, _apply, _wait_of
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.evalkit import CHANNELS, channel_split
from ..integrated.multiblock import MultiBlockTerminal
from ..integrated.scenario_gen import (GATE_BLOCK_MAX_S, GATE_BLOCK_MEAN_S,
                                       GATE_BLOCK_MIN_S, GATE_BLOCK_SIGMA_S, trunc_normal)
from .yr088_joint_rl import LEVEL
from .yr105_conditional_transfer import _CELLS, _sim
from .yr138_episode_pilot import _v2_hard_total

RC = RewardCalculator.numeraire_v1()
OUT = Path("outputs/reports/yr149_transfer_diag")
PILOT = Path("outputs/reports/yr133_sell_quote/pilot.json")
ROUTE_S = 180.0


class ReplayResolver:
    """원장 확정 이송의 앞 limit 건만 그대로 재실행 — 그 외 전부 KEEP."""

    def __init__(self, actions: list[dict], limit: int, travel_fn):
        self.actions = actions[:limit]
        self.travel_fn = travel_fn
        self.done = 0

    def review(self, mbt, t: float) -> None:
        for a in self.actions:
            if abs(a["t"] - t) <= 1e-6:
                ok = mbt.try_transfer(a["job_id"], a["dst"], route_s=ROUTE_S,
                                      travel_s=self.travel_fn(a["src"], a["job_id"]))
                assert ok, f"리플레이 이송 실패 — 재현 위반: {a}"
                self.done += 1


def run_replay(seeds: dict[str, int], tag: str, actions: list[dict], limit: int) -> dict:
    mbt = MultiBlockTerminal({"A": _sim(_CELLS["A"], seeds["A"]),
                              "B": _sim(_CELLS["B"], seeds["B"])})
    pol = ResolverPolicy(ServiceFirstSPTPreference(), "SF")
    gens: dict[int, CandidateGenerator] = {}
    exc = {"n": 0}

    def policy(sim, dp):
        g = gens.setdefault(id(sim), CandidateGenerator())
        gb = {c: g.generate(sim, c, LEVEL) for c in dp.crane_ids}
        try:
            _apply(sim, pol.decide(sim, dp, gb))
        except Exception:
            exc["n"] += 1
            _apply(sim, {c: _wait_of(gb[c]) for c in dp.crane_ids})

    def travel_fn(src, jid):                      # 원 파일럿과 동일 키 → travel 정확 재현
        rng = random.Random(f"y133:{tag}:{src}:{jid}")
        return trunc_normal(rng, GATE_BLOCK_MEAN_S,
                            GATE_BLOCK_SIGMA_S / GATE_BLOCK_MEAN_S,
                            lo=GATE_BLOCK_MIN_S, hi=GATE_BLOCK_MAX_S)

    resolver = ReplayResolver(actions, limit, travel_fn)
    chan_by_block = {b: {k: 0.0 for k in CHANNELS} for b in ("A", "B")}
    _bid = {id(s): b for b, s in mbt.blocks.items()}

    def cost_fn(sim, t0, t1, raw):
        cb = RC.cost_for(interval_start_s=t0, interval_end_s=t1, raw=raw, risk_max=0.0)
        b = _bid.get(id(sim))
        for k, v in channel_split(cb.contributions()).items():
            if b is not None:
                chan_by_block[b][k] += v
        return cb.total_normalized

    res = mbt.run(policy, resolver.review, cost_fn)
    mbt.check_invariants()
    assert resolver.done == limit, f"리플레이 미발화: {resolver.done}/{limit}"
    route_cost = res["route_cost_s"] / 3600.0
    return {"total_v1": float(res["terminal_total"] + route_cost),
            "v2_total": sum(_v2_hard_total(s) for s in mbt.blocks.values()),
            "chan_by_block": chan_by_block, "route_cost": route_cost,
            "backlog": sum(s.unfinished_backlog() for s in mbt.blocks.values()),
            "policy_exceptions": exc["n"]}


def diagnose() -> dict:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    quote_eps = pilot["arms"]["quote"]
    per_decision, per_episode = [], []
    for i, ep in enumerate(quote_eps):
        actions = [r for r in ep["quote_ledger"] if r["decision"] == "TRANSFER"]
        actions.sort(key=lambda r: r["t"])
        if not actions:
            continue
        tag = f"p{i}"
        seeds = ep["seeds"]
        print(f"[diag] pair{i} n_transfers={len(actions)}", flush=True)
        runs = [run_replay(seeds, tag, actions, j) for j in range(len(actions) + 1)]
        ep_rows = []
        for k, a in enumerate(actions, start=1):
            prev, cur = runs[k - 1], runs[k]
            d_src = (cur["chan_by_block"][a["src"]]["truck"]
                     - prev["chan_by_block"][a["src"]]["truck"])
            d_dst = (cur["chan_by_block"][a["dst"]]["truck"]
                     - prev["chan_by_block"][a["dst"]]["truck"])
            d_vessel = sum(cur["chan_by_block"][b]["vessel"]
                           - prev["chan_by_block"][b]["vessel"] for b in ("A", "B"))
            d_move = (sum(cur["chan_by_block"][b]["move"]
                          - prev["chan_by_block"][b]["move"] for b in ("A", "B"))
                      + cur["route_cost"] - prev["route_cost"])
            row = {"pair": i, "k": k, "job_id": a["job_id"], "t": a["t"],
                   "pred_net_gain": a["net_gain"],
                   "pred_out_relief": a["out_relief"], "pred_in_burden": a["in_burden"],
                   "d_total_v1": cur["total_v1"] - prev["total_v1"],
                   "d_v2": cur["v2_total"] - prev["v2_total"],
                   "d_src_truck": d_src, "d_dst_truck": d_dst,
                   "d_vessel": d_vessel, "d_move": d_move,
                   # 예측 대응 잔차 (동결 분해)
                   "resid_source": d_src + a["out_relief"],
                   "resid_receiver": d_dst - a["in_burden"],
                   "resid_vessel": d_vessel,
                   "resid_move": d_move - ROUTE_S / 3600.0,
                   "resid_total": (cur["total_v1"] - prev["total_v1"])
                                  - (-a["net_gain"])}
            ep_rows.append(row)
            per_decision.append(row)
        per_episode.append({"pair": i, "n": len(actions),
                            **{f"mean_{key}": fmean(r[key] for r in ep_rows)
                               for key in ("resid_source", "resid_receiver",
                                           "resid_vessel", "resid_move",
                                           "resid_total", "d_total_v1", "d_v2")}})
    axes = {"source": [e["mean_resid_source"] for e in per_episode],
            "receiver": [e["mean_resid_receiver"] for e in per_episode],
            "vessel": [e["mean_resid_vessel"] for e in per_episode]}
    axis_summary = {k: {"mean": fmean(v), "median": median(v)} for k, v in axes.items()}
    dominant = max(axis_summary, key=lambda k: abs(axis_summary[k]["mean"]))
    res = {"protocol": "사슬 한계효과 run_k − run_{k−1} (동결) · 통계 단위 = 에피소드 쌍",
           "n_decisions": len(per_decision), "n_episodes": len(per_episode),
           "per_decision": per_decision, "per_episode": per_episode,
           "axis_residual_summary(에피소드 묶음)": axis_summary,
           "dominant_axis(동결 규칙)": dominant}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(json.dumps({"dominant": dominant, **axis_summary,
                      "resid_total_mean(에피소드)": fmean(e["mean_resid_total"]
                                                          for e in per_episode)},
                     ensure_ascii=False, default=str))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()
    if a.diag:
        diagnose()
    print("DONE")
