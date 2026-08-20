"""YR-195 — 문턱: 눈금에 오차가 있는 자를 쓰는 법.

■ [[YR-189]] 가 진 이유 (실측)
`Q < 0` 은 참값이 0 근처인 건들을 **예측 오차의 부호**로 판다. 게다가 좌표 21개
중 최소를 고르면 "오차가 낙관 쪽으로 난 좌표"가 뽑힌다(승자의 저주). 실측:

    고른 좌표의 편향(예측−라벨) = **−0.1694**     (탐색을 켜면 +0.04 로 부호 반전)
    실제 이득 평균 −0.0105  →  **편향이 진짜 이득의 16배**
    판 것의 **43%가 실제로는 손해**

그래서 규칙보다 **2.26배** 팔았고 +211.41 로 졌다.

■ 무엇을 바꾸나 — 세 곳 (전부 opt-in · 기본값은 YR-189 그대로)

  ① 1관문 문턱 제거   `QSellPolicy(defer_decision=True)`
     블록은 "내 후보 중 뭐가 제일 부담인가" **순위만** 매긴다. 판단하지 않는다.
  ② 2관문이 판단      `UnifiedSellOrchestrator(sell_margin=m)`
     KEEP 값을 `0` → **`−m`** 으로. 좌표가 KEEP 을 `m` 만큼 이겨야 판다.
     **문턱은 여기 하나만 있다.**
  ③ ~~채점 넘겨받기~~  **폐기** (2026-08-19 배선 검증)
     "배정은 목적지를 더 붐비게만 만드니 최선 좌표가 안 바뀐 제안은 argmin 도
     안 바뀐다"를 가정했으나 **배정 기록 2,477/3,797 이 어긋났다** — 신경망이
     목적지 부하에 단조가 아니다. 게다가 **속도 이득이 1.00배**였다(계산의
     대부분은 1관문에 있고 배정기 몫은 애초에 작다). 위험한 가정을 안고 갈
     이유가 없어 전량 재채점을 유지한다.

**★단위 함정** — `sell_margin` 은 **Q 망 출력 눈금**이다. 학습이 목표를
`Q_SCALE=20` 으로 나눠 쓰므로, 비용시간 여유를 그대로 넣으면 20배 과한 문턱이
된다. 검증에서 `m=0.085`(원단위) 에 공간 판매가 **0** 이 됐다.
격자는 비용시간으로 적고 `margin_to_net_units()` 로 변환해 넣는다.

■ 왜 학습은 한 번만 하나
문턱은 **학습 목표에 들어가지 않는다.** Q 망은 "이 좌표로 팔면 비용이 얼마"를
회귀할 뿐이고, `m` 은 그 예측을 **쓰는 규칙**이다. 그래서 학습 1번 + 평가 4번이면
격자 전체를 덮는다.

■ `m` 격자 (사전등록에서 동결)
`0 / 0.085 / 0.17 / 0.34` — 실측 편향 0.1694 가 중앙, 절반과 두 배를 양옆에 둔다.
`m=0` 은 **문턱만 없는 대조군**이다(①③은 켠 상태이므로 YR-189 의 정확한 재현이
아니라 "새 구조에서 문턱만 뺀 것" — 그 차이도 함께 보고한다).
"""
from __future__ import annotations

from ..integrated.terminal_stream import DIURNAL_DAY_TOTAL

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr195_margin")
SPEC = ".claude/docs/dashboard-task-specs/YR-195-decision-threshold.md"
PREREG = (".claude/docs/strategy-history/"
          "2026-08-19-YR-195-문턱-사전등록.md")

ARM = "fixed15"
TRAIN_SEED = 8_400_000
N_ITER, EPS_PER_ITER = 16, 4          # YR-189 와 동일 (바뀌는 축을 늘리지 않는다)
# ★동결 격자 — **비용시간(원단위)** 으로 적는다. 주판정 0.17 = 실측 선택 편향
#   0.1694. 배정기에 넣을 때는 `Q_SCALE` 로 나눈다(망 출력과 눈금을 맞춘다).
#   원단위를 그대로 넣으면 20배 과한 문턱이 된다 — 2026-08-19 배선 검증에서
#   실제로 잡혔다(m=0.085 에서 공간 판매가 0 이 됐다).
MARGINS = (0.0, 0.085, 0.17, 0.34)
MAIN_MARGIN = 0.17


def margin_to_net_units(m_cost_hours: float) -> float:
    """비용시간 여유 → Q 망 출력 눈금. 학습이 목표를 Q_SCALE 로 나눠 쓴다."""
    from .yr189_q_train import Q_SCALE
    return float(m_cost_hours) / Q_SCALE

EVAL_SEED0 = 9_700_000                # 새 대역 — 9,600,000~ 는 YR-189 가 썼다
N_EVAL_DAYS = 16
CI95_T = 2.13                         # df=15 양측 95%


def eval_days() -> list[int]:
    return [EVAL_SEED0 + i * 1000 for i in range(N_EVAL_DAYS)]


# ------------------------------------------------------------------ 학습 (1회)
def _train_worker(args) -> dict:
    """YR-189 와 같은 계약 — 다른 점은 ①③ 을 켠 것뿐(문턱은 학습에 안 쓴다)."""
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..integrated.sell_q import QCoordScorer, QSellPolicy, SellQNet
    from ..integrated.time_sell import DEFER_DELTA_S
    from ..integrated.yard_layout import terminal_layout
    from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import run_episode_diurnal
    from .yr174_txn_reward import TransactionLog, realized_credit
    from .yr189_q_train import EXPLORE, EXPLORE_SIGMA, Q_SCALE
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    seed, sd, pol_seed = args

    net = SellQNet()
    net.load_state_dict(sd)
    net.eval()
    layout = terminal_layout()
    scorer = QCoordScorer(net, layout, defer_delta_s=DEFER_DELTA_S,
                          time_slots=False, explore_sigma=EXPLORE_SIGMA,
                          seed=pol_seed + 1)
    pol = QSellPolicy(scorer, explore=EXPLORE, seed=pol_seed,
                      defer_decision=True)          # ★①
    ep = run_episode_diurnal(seed, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False,
                             buy_net=None, q_scorer=scorer,
                             sell_margin=0.0,        # 학습은 문턱 없음
                             reuse_handoff=False,    # ③ 폐기 (등가성 실패)
                             _return_mbt=True)
    mbt = ep.pop("_mbt")
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    txns = TransactionLog().collect(ep["sell_ledger"])
    credits = realized_credit(mbt, txns, layout, l_t=SLA_ANCHOR_S + sla)
    by_key = {(tx["t"], tx["src"]): tx["txn"] for tx in txns}
    rows, tgts, n_lab = [], [], 0
    for r in ep["q_rows"]:
        txn = by_key.get((round(r["t"], 6), r["src"]))
        d = credits.get(txn, 0.0) if txn is not None else 0.0
        if txn is not None:
            n_lab += 1
        rows.append(r["row"].detach().clone())
        tgts.append(-float(d) / Q_SCALE)
    X = torch.stack(rows) if rows else torch.empty(0)
    y = torch.tensor(tgts, dtype=torch.float32) if tgts else torch.empty(0)
    return {"X": X, "y": y, "n_rows": len(rows), "n_committed": n_lab,
            "phi_final": ep["phi_final"], "n_space": ep["n_space"],
            "n_time": ep["n_time"], "n_txn": len(txns),
            "credit_sum": round(sum(credits.values()), 4),
            "offers": sum(1 for tr in pol.trail if tr["picked"] is not None),
            "decisions": len(pol.trail)}


def train(*, workers: int = EPS_PER_ITER) -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..integrated.sell_q import Q_ROW_DIM, SellQNet
    from ..integrated.terminal_stream import OBS_24H
    from .yr189_q_train import (BUFFER_MAX, EXPLORE, EXPLORE_SIGMA, GRAD_CLIP,
                                HUBER_BETA, LR, MINIBATCH, Q_SCALE,
                                STEPS_PER_ITER, _fit)
    _mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    torch.manual_seed(TRAIN_SEED)
    out = OUT / f"s{TRAIN_SEED}"
    out.mkdir(parents=True, exist_ok=True)

    net = SellQNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf_X, buf_y = torch.empty(0, Q_ROW_DIM), torch.empty(0)
    hist: list[dict] = []
    for it in range(N_ITER):
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        jobs = [(TRAIN_SEED + it * EPS_PER_ITER + e, sd, TRAIN_SEED + it * 100 + e)
                for e in range(EPS_PER_ITER)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            eps = list(pool.map(_train_worker, jobs))
        nx = [e["X"] for e in eps if e["n_rows"]]
        ny = [e["y"] for e in eps if e["n_rows"]]
        X = torch.cat(nx) if nx else buf_X[:0]
        y = torch.cat(ny) if ny else buf_y[:0]
        buf_X = torch.cat([buf_X, X])[-BUFFER_MAX:]
        buf_y = torch.cat([buf_y, y])[-BUFFER_MAX:]
        if buf_X.shape[0] == 0:
            raise RuntimeError(f"iter {it}: 표본 0 — 조용한 무학습 금지")
        fit = _fit(net, opt, buf_X, buf_y, steps=STEPS_PER_ITER,
                   seed=TRAIN_SEED + it)
        n = len(eps)
        hist.append({"iter": it, **fit,
                     "phi_final_mean": sum(e["phi_final"] for e in eps) / n,
                     "n_space_mean": sum(e["n_space"] for e in eps) / n,
                     "n_time_mean": sum(e["n_time"] for e in eps) / n,
                     "offers_mean": sum(e["offers"] for e in eps) / n,
                     "decisions_mean": sum(e["decisions"] for e in eps) / n,
                     "rows_new": int(X.shape[0]),
                     "committed_new": sum(e["n_committed"] for e in eps),
                     "credit_sum_mean": sum(e["credit_sum"] for e in eps) / n})
        (out / "train.json").write_text(json.dumps(
            {"train_seed": TRAIN_SEED, "arm": ARM, "history": hist,
             "in_progress": it < N_ITER - 1}, ensure_ascii=False, indent=1,
            default=str), encoding="utf-8")
        torch.save({"q": net.state_dict()}, out / "net.pt")

    payload = {"experiment": "YR-195 문턱 — 학습(1회)", "prereg": PREREG,
               "spec": SPEC, "kind": "training", "arm": ARM,
               "train_seed": TRAIN_SEED, "history": hist,
               "changed_axis": "①1관문 문턱 제거 (문턱 m 은 평가에서만. ③ 채점 넘겨받기는 폐기)",
               "code_dirty": bool(code_dirty()),
               "stamp": repro_stamp(
                   experiment="YR-195 문턱 학습", seeds={"train": [TRAIN_SEED]},
                   params={"arm": ARM, "N_ITER": N_ITER,
                           "EPS_PER_ITER": EPS_PER_ITER, "LR": LR,
                           "EXPLORE": EXPLORE, "EXPLORE_SIGMA": EXPLORE_SIGMA,
                           "BUFFER_MAX": BUFFER_MAX,
                           "STEPS_PER_ITER": STEPS_PER_ITER, "Q_SCALE": Q_SCALE,
                           "MINIBATCH": MINIBATCH, "HUBER_BETA": HUBER_BETA,
                           "GRAD_CLIP": GRAD_CLIP,
                           "train_sell_margin": 0.0,
                           "defer_decision": True, "reuse_handoff": False,
                           "observation": OBS_24H.as_dict()},
                   prereg=PREREG)}
    p = out / "train_all.json"
    write_result(p, payload)
    return p


# ------------------------------------------------------------------ 평가 (격자 4값)
def _eval_worker(args) -> dict:
    """팔 하나 × 날 하나. 팔 = `K` · `greedy` · `m<값>`."""
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

    scorer, margin, reuse = None, 0.0, False
    if arm == "K":
        pol = KeepAllTrail()
    elif arm == "greedy":
        pol = GreedyOfferPolicy(kf, layout)
    else:
        from ..integrated.sell_q import QCoordScorer, QSellPolicy, SellQNet
        from ..integrated.time_sell import DEFER_DELTA_S
        margin = float(arm[1:])                # 표시는 비용시간
        net = SellQNet()
        ck = OUT / f"s{TRAIN_SEED}" / "net.pt"
        net.load_state_dict(torch.load(ck, map_location="cpu",
                                       weights_only=True)["q"])
        net.eval()
        scorer = QCoordScorer(net, layout, defer_delta_s=DEFER_DELTA_S,
                              time_slots=False)      # 탐색 0 — 사전등록 모드
        pol = QSellPolicy(scorer, explore=0.0, defer_decision=True)
        # ③(채점 넘겨받기)는 **폐기** — 등가성 실패 · 속도 이득 0 (2026-08-19)
        reuse = False

    ep = run_episode_diurnal(day, pol, kf, exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False,
                             buy_net=None, q_scorer=scorer,
                             sell_margin=margin_to_net_units(margin),  # ★눈금 변환
                             reuse_handoff=reuse,
                             _return_mbt=True)
    mbt = ep.pop("_mbt")
    ao: list[float] = []
    for sim in mbt.blocks.values():
        tl = getattr(sim, "time_ledger", None)
        if tl is not None:
            ao.extend(tl.terminal_turntime_samples_s())
    ao.sort()
    n = len(ao)
    return {"arm": arm, "day": day, "margin": margin,
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


def evaluate(*, workers: int = 16) -> Path:
    from concurrent.futures import ProcessPoolExecutor
    from statistics import fmean

    import torch.multiprocessing as _mp
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..integrated.terminal_stream import OBS_24H
    _mp.set_sharing_strategy("file_system")

    ck = OUT / f"s{TRAIN_SEED}" / "net.pt"
    if not ck.exists():
        raise FileNotFoundError(f"학습 가중치 없음: {ck}")
    days = eval_days()
    arms = ["K", "greedy"] + [f"m{m}" for m in MARGINS]
    jobs = [(a, d) for a in arms for d in days]
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_eval_worker, jobs))

    by = {a: {} for a in arms}
    for r in rows:
        by[r["arm"]][r["day"]] = r
    guards = {
        "all_admitted_3600": all(r["admitted"] == DIURNAL_DAY_TOTAL for r in rows),
        "no_policy_exceptions": all(r["policy_exceptions"] == 0 for r in rows),
        "code_dirty": bool(code_dirty()),
        "all_cells_present": all(len(by[a]) == len(days) for a in arms),
        "n_cells": len(rows), "n_cells_expected": len(arms) * len(days)}

    main = f"m{MAIN_MARGIN}"
    contrasts = {f"{a}_vs_greedy": _paired(by[a], by["greedy"], days, "phi_final")
                 for a in arms if a != "greedy"}
    contrasts.update({f"{a}_vs_K": _paired(by[a], by["K"], days, "phi_final")
                      for a in arms if a != "K"})
    contrasts["MAIN_vs_greedy_ao_mean"] = _paired(by[main], by["greedy"], days,
                                                  "ao_mean_s")
    summary = [{
        "arm": a, "margin": by[a][days[0]]["margin"],
        "phi_mean": round(fmean(by[a][d]["phi_final"] for d in days), 2),
        "ao_mean_s": round(fmean(by[a][d]["ao_mean_s"] for d in days), 2),
        "ao_p95_s": round(fmean(by[a][d]["ao_p95_s"] for d in days), 1),
        "offers_mean": round(fmean(by[a][d]["offers"] for d in days), 1),
        "n_space_mean": round(fmean(by[a][d]["n_space"] for d in days), 1),
        "n_time_mean": round(fmean(by[a][d]["n_time"] for d in days), 1),
    } for a in arms]

    mj = contrasts[f"{main}_vs_greedy"]
    verdict = {
        "BETTER": "문턱을 넣으니 규칙을 넘었다",
        "INCONCLUSIVE": "규칙과 구분 불가 — 문턱이 격차를 없앴다",
        "WORSE": "문턱을 넣어도 못 넘었다",
    }[mj["verdict"]] if mj else "판정 불가"

    payload = {"experiment": "YR-195 문턱 격자 평가", "prereg": PREREG,
               "spec": SPEC, "kind": "evaluation",
               "mode": "argmin(deterministic) · 문턱은 배정기 한 곳",
               "eval_days": days, "n_days": len(days), "arms": arms,
               "margins": list(MARGINS), "main_margin": MAIN_MARGIN,
               "main_judgment": f"{main}_vs_greedy", "verdict": verdict,
               "checkpoint": str(ck), "guards": guards,
               "summary": summary, "contrasts": contrasts, "cells": rows,
               "stamp": repro_stamp(
                   experiment="YR-195 문턱 격자 평가",
                   seeds={"eval_days": days},
                   params={"arms": arms, "margins": list(MARGINS),
                           "main_margin": MAIN_MARGIN, "time_slots": False,
                           "explore": 0.0, "defer_decision": True,
                           "reuse_handoff": False, "day_plan_public": True,
                           "observation": OBS_24H.as_dict()},
                   prereg=PREREG)}
    p = OUT / "eval.json"
    write_result(p, payload)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "eval"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "train":
        p = train(workers=min(a.workers, EPS_PER_ITER))
        d = json.loads(p.read_text(encoding="utf-8"))
        print("{:>3} {:>9} {:>8} {:>7} {:>6} {:>6} {:>8}".format(
            "it", "Phi", "credit", "offer", "space", "time", "RMSE"))
        for h in d["history"]:
            print(f"{h['iter']:>3} {h['phi_final_mean']:>9.1f} "
                  f"{h['credit_sum_mean']:>8.0f} {h['offers_mean']:>7.0f} "
                  f"{h['n_space_mean']:>6.0f} {h['n_time_mean']:>6.0f} "
                  f"{h['rmse']:>8.4f}")
    else:
        p = evaluate(workers=a.workers)
        d = json.loads(p.read_text(encoding="utf-8"))
        print("guards:", d["guards"])
        print("{:>8} {:>6} {:>10} {:>9} {:>8} {:>8} {:>8}".format(
            "arm", "m", "Phi", "AOmean", "offer", "space", "time"))
        for s in d["summary"]:
            print(f"{s['arm']:>8} {s['margin']:>6.3f} {s['phi_mean']:>10.1f} "
                  f"{s['ao_mean_s']:>9.1f} {s['offers_mean']:>8.1f} "
                  f"{s['n_space_mean']:>8.1f} {s['n_time_mean']:>8.1f}")
        print()
        for k, v in d["contrasts"].items():
            if v is None:
                continue
            star = " <== MAIN" if k == d["main_judgment"] else ""
            print(f"{k:>24} {v['mean']:>+9.2f} CI[{v['ci95'][0]:>+8.2f},"
                  f"{v['ci95'][1]:>+8.2f}] t={v['t']:>+6.2f} {v['verdict']}{star}")
        print()
        print("verdict:", d["verdict"])
    print("DONE", p)
