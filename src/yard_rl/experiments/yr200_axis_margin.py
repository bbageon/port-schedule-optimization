"""YR-200 — 축별 문턱: 비교 폭이 다른 두 축에 같은 여유를 걸었던 대가.

■ [[YR-195]] 가 남긴 문제
`m0.17 − greedy` = +39.92 (YR-189 의 +211.41 에서 81% 감소). 그러나 판매 구성이
어긋났다.

    규칙    공간 678 · 시간 281
    m0.17   공간 415 · 시간  53      ← 시간은 잘 막았고 **공간까지 39% 깎였다**

■ 왜 같은 문턱이 다르게 먹나
문턱의 적정 크기는 **"몇 개 중에서 골랐나"** 에 달려 있다(승자의 저주).

    반입(공간) : 목적지 20곳 + 시간 1 = **21개 중 최소**  → 편향 크다
    반출(시간) : 시간 **1개뿐**                        → 편향 작다

실측(실행 전 · 비판정 대역 9,900,080~ · YR-195 체크포인트):

    공간 편향 −0.0554  ·  시간 편향 −0.0456   (1.22배)

그리고 [[YR-189]] 때(−0.175/−0.162)보다 **3배 작다** — 이미 0.17 이 걸려 있어
극단적 선택이 걸러졌기 때문이다. **즉 현 문턱은 잔여 편향의 3배로 과도하다.**

■ 무엇을 바꾸나 — 한 곳
`sell_margin` → `margin_space` / `margin_time`. 여유를 KEEP 에서 내리지 않고
**좌표마다 올린다**(한 축만 있는 제안에서는 수학적으로 동일하나, 반입 제안은
공간 20개와 시간 1개가 한 목록에서 겨루므로 좌표 쪽에 걸어야 축별로 나뉜다).

**학습은 하지 않는다** — 문턱은 학습 목표에 안 들어간다. YR-195 체크포인트 재사용.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr200_axis_margin")
CKPT = Path("outputs/reports/yr195_margin/s8400000/net.pt")   # ★YR-195 가중치 재사용
SPEC = ".claude/docs/dashboard-task-specs/YR-200-per-axis-threshold.md"
PREREG = (".claude/docs/strategy-history/"
          "2026-08-19-YR-200-축별문턱-사전등록.md")

# ★동결 격자 — (m_space, m_time) · **비용시간 원단위**. 넣을 때 Q_SCALE 로 나눈다.
COMBOS = {
    "A": (0.11, 0.09),      # ★주판정 — 실측 편향비(1.22) 유지
    "B": (0.085, 0.17),     # YR-195 스윕에서 공간이 규칙과 맞던 값
    "C": (0.17, 0.34),      # 공간 현행 + 시간 강화
    "D": (0.17, 0.17),      # 대조군 — YR-195 재현
}
MAIN = "A"

EVAL_SEED0 = 9_800_000      # 새 대역 — 9,700,000~ 는 YR-195 가 썼다
N_EVAL_DAYS = 16
CI95_T = 2.13               # df=15 양측 95%


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
    from .yr189_q_train import Q_SCALE
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    arm, day = args
    kf, layout = load_kf(), terminal_layout()

    scorer, ms, mt = None, 0.0, 0.0
    if arm == "K":
        pol = KeepAllTrail()
    elif arm == "greedy":
        pol = GreedyOfferPolicy(kf, layout)
    else:
        from ..integrated.sell_q import QCoordScorer, QSellPolicy, SellQNet
        from ..integrated.time_sell import DEFER_DELTA_S
        ms_h, mt_h = COMBOS[arm]                       # 비용시간
        ms, mt = ms_h / Q_SCALE, mt_h / Q_SCALE        # ★망 출력 눈금으로 변환
        net = SellQNet()
        net.load_state_dict(torch.load(CKPT, map_location="cpu",
                                       weights_only=True)["q"])
        net.eval()
        scorer = QCoordScorer(net, layout, defer_delta_s=DEFER_DELTA_S,
                              time_slots=False)
        pol = QSellPolicy(scorer, explore=0.0, defer_decision=True)

    ep = run_episode_diurnal(day, pol, kf, exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False,
                             buy_net=None, q_scorer=scorer,
                             margin_space=ms, margin_time=mt,
                             _return_mbt=True)
    mbt = ep.pop("_mbt")
    ao: list[float] = []
    for sim in mbt.blocks.values():
        tl = getattr(sim, "time_ledger", None)
        if tl is not None:
            ao.extend(tl.terminal_turntime_samples_s())
    ao.sort()
    n = len(ao)
    return {"arm": arm, "day": day, "m_space": ms, "m_time": mt,
            "phi_final": round(ep["phi_final"], 4),
            "ao_mean_s": round(sum(ao) / n, 2) if n else None,
            "ao_p95_s": round(ao[min(n - 1, int(0.95 * n))], 1) if n else None,
            "offers": sum(1 for x in pol.trail if x.get("picked") is not None),
            "decisions": len(pol.trail),
            "n_space": ep["n_space"], "n_time": ep["n_time"],
            "policy_exceptions": ep["policy_exceptions"],
            "admitted": ep["admitted"], "observe_s": OBS_24H.observe_s}


def _paired(a: dict, b: dict, days: list[int], key: str):
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
        raise FileNotFoundError(f"YR-195 가중치 없음: {CKPT}")
    days = eval_days()
    arms = ["K", "greedy"] + list(COMBOS)
    jobs = [(a, d) for a in arms for d in days]
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_worker, jobs))

    by = {a: {} for a in arms}
    for r in rows:
        by[r["arm"]][r["day"]] = r
    guards = {
        "all_admitted_3600": all(r["admitted"] == 3600 for r in rows),
        "no_policy_exceptions": all(r["policy_exceptions"] == 0 for r in rows),
        "code_dirty": bool(code_dirty()),
        "all_cells_present": all(len(by[a]) == len(days) for a in arms),
        "n_cells": len(rows), "n_cells_expected": len(arms) * len(days)}

    contrasts = {f"{a}_vs_greedy": _paired(by[a], by["greedy"], days, "phi_final")
                 for a in arms if a != "greedy"}
    contrasts.update({f"{a}_vs_K": _paired(by[a], by["K"], days, "phi_final")
                      for a in COMBOS})
    contrasts["MAIN_vs_greedy_ao_mean"] = _paired(by[MAIN], by["greedy"], days,
                                                  "ao_mean_s")
    summary = [{
        "arm": a,
        "m_space": round(by[a][days[0]]["m_space"] * 20, 4),
        "m_time": round(by[a][days[0]]["m_time"] * 20, 4),
        "phi_mean": round(fmean(by[a][d]["phi_final"] for d in days), 2),
        "ao_mean_s": round(fmean(by[a][d]["ao_mean_s"] for d in days), 2),
        "ao_p95_s": round(fmean(by[a][d]["ao_p95_s"] for d in days), 1),
        "n_space_mean": round(fmean(by[a][d]["n_space"] for d in days), 1),
        "n_time_mean": round(fmean(by[a][d]["n_time"] for d in days), 1),
    } for a in arms]

    mj = contrasts[f"{MAIN}_vs_greedy"]
    verdict = {
        "BETTER": "축별 문턱으로 규칙을 넘었다",
        "INCONCLUSIVE": "구분 불가 — 규칙과 동급 도달",
        "WORSE": "못 넘었다 — 문턱 축은 여기까지",
    }[mj["verdict"]] if mj else "판정 불가"

    payload = {"experiment": "YR-200 축별 문턱", "prereg": PREREG, "spec": SPEC,
               "kind": "evaluation", "mode": "argmin · 축별 여유 · 학습 재사용",
               "eval_days": days, "n_days": len(days), "arms": arms,
               "combos": {k: list(v) for k, v in COMBOS.items()}, "main": MAIN,
               "main_judgment": f"{MAIN}_vs_greedy", "verdict": verdict,
               "checkpoint": str(CKPT), "guards": guards,
               "summary": summary, "contrasts": contrasts, "cells": rows,
               "stamp": repro_stamp(
                   experiment="YR-200 축별 문턱", seeds={"eval_days": days},
                   params={"arms": arms,
                           "combos": {k: list(v) for k, v in COMBOS.items()},
                           "main": MAIN, "time_slots": False, "explore": 0.0,
                           "defer_decision": True, "day_plan_public": True,
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
    print("guards:", d["guards"])
    print("{:>8} {:>8} {:>8} {:>10} {:>9} {:>8} {:>8}".format(
        "arm", "m_space", "m_time", "Phi", "AOmean", "space", "time"))
    for s in d["summary"]:
        print(f"{s['arm']:>8} {s['m_space']:>8.3f} {s['m_time']:>8.3f} "
              f"{s['phi_mean']:>10.1f} {s['ao_mean_s']:>9.1f} "
              f"{s['n_space_mean']:>8.1f} {s['n_time_mean']:>8.1f}")
    print()
    for k, v in d["contrasts"].items():
        if v is None:
            continue
        star = " <== MAIN" if k == d["main_judgment"] else ""
        print(f"{k:>20} {v['mean']:>+9.2f} CI[{v['ci95'][0]:>+8.2f},"
              f"{v['ci95'][1]:>+8.2f}] t={v['t']:>+6.2f} {v['verdict']}{star}")
    print()
    print("verdict:", d["verdict"])
    print("DONE", p)
