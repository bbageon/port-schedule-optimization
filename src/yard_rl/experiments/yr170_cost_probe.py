"""YR-170 진단 — **벌금 항이 실제로 켜졌는가** (사용자 질문 2026-08-11).

학습이 안 움직인 이유를 따질 때 먼저 확인할 것: 판매의 이득은 비용의 **볼록성**
(43분 초과 벌금)에서만 나온다. 벌금 항이 거의 안 켜지면 비용은 사실상 선형이고,
트럭을 옮겨봐야 총합은 그대로인데 주행비만 늘어 **원리상 손해**다.

YR-167 자격은 **SF 집행**에서 벌금 도달률 24.0% 를 봤지만, YR-170 학습은
**채택 PPO 집행 헤드**로 돈다 — 같은 무대라도 비용 곡선이 다르다. 그래서 여기서
다시 잰다. 판정 아님 — 관측이다.

분해: Φ = Σ트럭 선형(체류) + Σ트럭 벌금(43분 초과) + 본선 지연(×10) + 주행 + 기사 대기
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median

from ..integrated.cost_curve_v2 import RHO_VESSEL_V2
from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
from ..integrated.policy_config import ADOPTED_C0_GUARD
from ..integrated.vessel import VesselWorkType
from .yr151_transfer_ppo import load_kf
from .yr170_sell_ppo_diurnal import (KeepAllTrail, OUT as TRAIN_OUT,
                                     TRAIN_SEEDS, run_episode_diurnal)

OUT = Path("outputs/reports/yr170_cost_probe")


def _decompose(mbt, end_s: float, sla_s: float) -> dict:
    l_t = SLA_ANCHOR_S + sla_s                     # 2,580초 = 벌금 임계
    lin = pen = 0.0
    stays: list[float] = []
    n_pen = 0
    for sim in mbt.blocks.values():
        tl = getattr(sim, "time_ledger", None)
        if tl is None:
            continue
        for r in tl.records.values():
            a = r.gate_in
            if a is None or a > end_s:
                continue
            o = r.gate_out if (r.gate_out is not None and r.gate_out <= end_s) else end_s
            stay = o - a
            stays.append(stay)
            lin += stay / 3600.0
            over = max(0.0, o - (a + l_t))
            pen += over / 3600.0
            if over > 0:
                n_pen += 1
    ves = 0.0
    n_ves_late = 0
    for sim in mbt.blocks.values():
        for v in sim.vessels.values():
            if v.work_type != VesselWorkType.LOAD:
                continue
            p = v.plan.planned_completion_s
            if p is None:
                continue
            f = getattr(getattr(v, "truth", None), "actual_completion_s", None)
            f_eff = f if (f is not None and f <= end_s) else end_s
            d = max(0.0, f_eff - p)
            ves += RHO_VESSEL_V2 * d / 3600.0
            n_ves_late += 1 if d > 0 else 0
    route = mbt.route_cost_s / 3600.0
    stays.sort()

    def q(p):
        return stays[min(len(stays) - 1, int(p * len(stays)))] if stays else None
    return {
        "phi_parts": {"truck_linear": round(lin, 2), "truck_penalty": round(pen, 2),
                      "vessel_delay_x10": round(ves, 2), "route": round(route, 2)},
        "penalty_share_of_phi": round(pen / (lin + pen + ves + route), 4) if stays else None,
        "n_trucks": len(stays), "n_penalty_trucks": n_pen,
        "penalty_hit_rate": round(n_pen / len(stays), 4) if stays else None,
        "stay_s": {"mean": round(fmean(stays), 1) if stays else None,
                   "median": round(median(stays), 1) if stays else None,
                   "p90": round(q(0.90), 1) if stays else None,
                   "p99": round(q(0.99), 1) if stays else None,
                   "max": round(stays[-1], 1) if stays else None,
                   "threshold_s": l_t},
        "n_vessels_late": n_ves_late,
    }


def probe(seed: int, *, policy_kind: str) -> dict:
    """policy_kind: 'keep' (전건 KEEP) | 'trained' (학습본 net.pt)"""
    from ..integrated.terminal_stream import OBS_24H
    if policy_kind == "keep":
        pol = KeepAllTrail()
    else:
        import torch
        from ..v1.ppo_policy import PpoSellPolicy, TransferActor, TransferCritic
        from ..integrated.yard_layout import terminal_layout
        ck = torch.load(TRAIN_OUT / f"ppo_s{seed}" / "net.pt", map_location="cpu")
        a, c = TransferActor(), TransferCritic()
        a.load_state_dict(ck["actor"])
        c.load_state_dict(ck["critic"])
        pol = PpoSellPolicy(a, c, mode="live", sample=False, seed=seed,
                            layout=terminal_layout())
    ep = run_episode_diurnal(seed, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             _return_mbt=True)
    mbt = ep.pop("_mbt")
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    return {"seed": seed, "policy": policy_kind,
            "phi_final": round(ep["phi_final"], 2),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            **_decompose(mbt, OBS_24H.observe_s, sla)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--policy", choices=["keep", "trained"], default="keep")
    a = ap.parse_args()
    r = probe(TRAIN_SEEDS[a.seed_idx], policy_kind=a.policy)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"probe_{a.policy}_s{r['seed']}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False))
    print("DONE")
