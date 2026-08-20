"""YR-189 평가 — Q 전환이 **규칙을 넘는가**. 사전등록 §5·§6 을 그대로 집행한다.

주판정: `q − greedy` (총비용 Φ). 모드 = **argmin(결정론, ε=0)**.
독립단위 = 날(n=16), 같은 날 짝지어 비교. |t| ≥ 2.13 (df=15, 양측 95%).
CI 가 0 을 배제할 때만 방향을 선언한다 — 포함하면 "구분 불가"로 기록한다.

세 팔은 **같은 날·같은 트럭·같은 집행**을 받는다. 다른 것은 판매 정책 하나뿐이다.
  K       안 팔기 (바닥)
  greedy  규칙 판매 (YR-179 `UnifiedNetGain`, 문턱 코드 기본값 동결)
  q       학습된 Q (이 판의 산출물, 마지막 회차 가중치 고정)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr189_q")
SPEC = ".claude/docs/dashboard-task-specs/YR-189-sell-axis-q-scoring.md"
PREREG = (".claude/docs/strategy-history/"
          "2026-08-18-YR-189-Q전환-사전등록.md")
CKPT = OUT / "s8400000" / "net.pt"

EVAL_SEED0 = 9_600_000          # 새 대역 — 이 판정에만 쓰고 재사용하지 않는다
N_EVAL_DAYS = 16
ARMS = ("K", "greedy", "q")
CI95_T = 2.13                   # df=15 양측 95%


def eval_days() -> list[int]:
    return [EVAL_SEED0 + i * 1000 for i in range(N_EVAL_DAYS)]


def _worker(args) -> dict:
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.terminal_stream import OBS_24H
    from ..integrated.yard_layout import terminal_layout
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import KeepAllTrail, run_episode_diurnal
    from .yr179_greedy_baseline import GreedyOfferPolicy
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    arm, day = args
    kf, layout = load_kf(), terminal_layout()

    scorer = None
    if arm == "K":
        pol = KeepAllTrail()
    elif arm == "greedy":
        pol = GreedyOfferPolicy(kf, layout)
    else:
        from ..v2.sell_q import QCoordScorer, QSellPolicy, SellQNet
        from ..integrated.time_sell import DEFER_DELTA_S
        net = SellQNet()
        st = torch.load(CKPT, map_location="cpu", weights_only=True)
        net.load_state_dict(st["q"])
        net.eval()
        scorer = QCoordScorer(net, layout, defer_delta_s=DEFER_DELTA_S,
                              time_slots=False)
        pol = QSellPolicy(scorer, explore=0.0)      # ★사전등록 모드: 순수 argmin

    ep = run_episode_diurnal(day, pol, kf, exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False,
                             buy_net=None, q_scorer=scorer, _return_mbt=True)
    mbt = ep.pop("_mbt")
    ao: list[float] = []
    for sim in mbt.blocks.values():
        tl = getattr(sim, "time_ledger", None)
        if tl is not None:
            ao.extend(tl.terminal_turntime_samples_s())
    ao.sort()
    n = len(ao)
    offers = sum(1 for tr in pol.trail if tr.get("picked") is not None)
    return {"arm": arm, "day": day,
            "phi_final": round(ep["phi_final"], 4),
            "ao_mean_s": round(sum(ao) / n, 2) if n else None,
            "ao_p95_s": round(ao[min(n - 1, int(0.95 * n))], 1) if n else None,
            "ao_n": n, "offers": offers, "decisions": len(pol.trail),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "policy_exceptions": ep["policy_exceptions"],
            "admitted": ep["admitted"], "observe_s": OBS_24H.observe_s}


def _paired(a: dict, b: dict, days: list[int], key: str):
    """같은 날 짝지어 a − b. 독립단위 = 날 (사전등록 §5)."""
    from statistics import fmean, pstdev
    d = [a[x][key] - b[x][key] for x in days
         if a.get(x, {}).get(key) is not None and b.get(x, {}).get(key) is not None]
    if len(d) < 2:
        return None
    m = fmean(d)
    se = pstdev(d) / (len(d) - 1) ** 0.5
    hw = CI95_T * se
    return {"n_days": len(d), "mean": round(m, 2), "se": round(se, 2),
            "t": round(m / se, 2) if se else None,
            "ci95": [round(m - hw, 2), round(m + hw, 2)],
            "ci_halfwidth": round(hw, 2),
            "n_negative": sum(1 for v in d if v < 0),
            "verdict": ("BETTER" if m + hw < 0 else
                        "WORSE" if m - hw > 0 else "INCONCLUSIVE")}


def run(*, workers: int = 16) -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..integrated.terminal_stream import OBS_24H
    _mp.set_sharing_strategy("file_system")

    if not CKPT.exists():
        raise FileNotFoundError(f"학습 가중치 없음: {CKPT} — 먼저 yr189_q_train 을 돌린다")
    days = eval_days()
    jobs = [(a, d) for a in ARMS for d in days]
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_worker, jobs))

    by = {a: {} for a in ARMS}
    for r in rows:
        by[r["arm"]][r["day"]] = r
    guards = {
        "all_admitted_3600": all(r["admitted"] == 3600 for r in rows),
        "no_policy_exceptions": all(r["policy_exceptions"] == 0 for r in rows),
        "code_dirty": bool(code_dirty()),
        "all_cells_present": all(len(by[a]) == len(days) for a in ARMS),
        "n_cells": len(rows), "n_cells_expected": len(ARMS) * len(days)}

    contrasts = {
        # ★주판정 — 사전등록 §6
        "q_vs_greedy": _paired(by["q"], by["greedy"], days, "phi_final"),
        # 함께 보고 (판정 아님)
        "q_vs_K": _paired(by["q"], by["K"], days, "phi_final"),
        "greedy_vs_K": _paired(by["greedy"], by["K"], days, "phi_final"),
        "q_vs_greedy_ao_mean": _paired(by["q"], by["greedy"], days, "ao_mean_s"),
        "q_vs_greedy_ao_p95": _paired(by["q"], by["greedy"], days, "ao_p95_s"),
    }
    summary = [{
        "arm": a,
        "phi_mean": round(fmean(by[a][d]["phi_final"] for d in days), 2),
        "ao_mean_s": round(fmean(by[a][d]["ao_mean_s"] for d in days), 2),
        "ao_p95_s": round(fmean(by[a][d]["ao_p95_s"] for d in days), 1),
        "offers_mean": round(fmean(by[a][d]["offers"] for d in days), 1),
        "n_space_mean": round(fmean(by[a][d]["n_space"] for d in days), 1),
        "n_time_mean": round(fmean(by[a][d]["n_time"] for d in days), 1),
    } for a in ARMS]

    main = contrasts["q_vs_greedy"]
    verdict = {
        "BETTER": "학습이 규칙을 넘었다 — 서사 A 재검토",
        "INCONCLUSIVE": "구분 불가 — 규칙 설계 없이 학습만으로 동등 도달",
        "WORSE": "못 넘었다 — 서사 B 확정",
    }[main["verdict"]] if main else "판정 불가(표본 부족)"

    payload = {"experiment": "YR-189 Q 전환 평가", "prereg": PREREG, "spec": SPEC,
               "kind": "evaluation", "mode": "argmin(deterministic)",
               "eval_days": days, "n_days": len(days), "arms": list(ARMS),
               "checkpoint": str(CKPT),
               "guards": guards, "summary": summary, "contrasts": contrasts,
               "main_judgment": "q_vs_greedy", "verdict": verdict,
               "cells": rows,
               "stamp": repro_stamp(
                   experiment="YR-189 Q 전환 평가",
                   seeds={"eval_days": days},
                   params={"arms": list(ARMS), "time_slots": False,
                           "explore": 0.0, "day_plan_public": True,
                           "observation": OBS_24H.as_dict()},
                   prereg=PREREG)}
    p = OUT / "eval.json"
    write_result(p, payload)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    p = run(workers=a.workers)
    d = json.loads(p.read_text(encoding="utf-8"))
    print("가드:", d["guards"])
    print(f"{'arm':>7} {'Phi':>10} {'AO평균s':>9} {'AO P95s':>9} "
          f"{'제안':>7} {'공간':>7} {'시간':>7}")
    for s in d["summary"]:
        print(f"{s['arm']:>7} {s['phi_mean']:>10.1f} {s['ao_mean_s']:>9.1f} "
              f"{s['ao_p95_s']:>9.1f} {s['offers_mean']:>7.1f} "
              f"{s['n_space_mean']:>7.1f} {s['n_time_mean']:>7.1f}")
    print()
    for k, v in d["contrasts"].items():
        if v is None:
            continue
        star = " ★주판정" if k == "q_vs_greedy" else ""
        print(f"{k:>22}  {v['mean']:>+9.2f}  CI[{v['ci95'][0]:>+8.2f},"
              f"{v['ci95'][1]:>+8.2f}]  t={v['t']:>+6.2f}  {v['verdict']}{star}")
    print()
    print("판정:", d["verdict"])
    print("DONE", p)
