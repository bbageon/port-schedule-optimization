"""YR-189 — 판매 Q 망 학습 (C안). PPO 를 버리고 **비용 회귀**로 간다.

■ 바뀌는 축은 **하나**다
무대·보상 원료(`realized_credit`)·집행(크레인 동결)·팔(`fixed15` 고정 +15분)은
[[YR-185]]·[[YR-183]] 과 같다. 바뀌는 것은 **학습 방식**뿐이다.

    구: 정책망 + critic → PPO clip · 이점 정규화 · on-policy
    신: Q 망 하나      → 후버 회귀 · argmin · **off-policy 재생**

■ 왜 off-policy 가 편향 없이 성립하나
목표가 부트스트랩이 아니라 **직접 관측된 라벨**(−D_i)이기 때문이다. TD 목표라면
행동분포가 바뀔 때 보정이 필요하지만, 여기서는 "그 좌표로 팔았더니 실제로 얼마
들었나"를 그대로 회귀한다 — 낡은 표본도 여전히 참이다.
→ 40 에피소드를 쓰고 버리던 구조가 사라진다(실행비용이 지배적인 이 문제의 핵심 이득).

■ 표본이 되는 것 / 안 되는 것
  · 표본 = **배정기가 실제로 고른 (작업, 좌표) 행**. 확정 실패도 표본이다(라벨 0 =
    "아무 이득 없었다" — 구 PPO 는 이걸 버렸다).
  · KEEP 은 표본이 **아니다**. 비용 0 은 정의이지 데이터가 아니다.
    → 구 학습에서 91% 를 차지하던 0 보상 표본이 구조적으로 사라진다.

■ 탐색
argmin 만 쓰면 고른 좌표의 라벨만 모여 나머지를 영원히 못 배운다(문헌이 지적하는
off-policy 커버리지 문제). 학습 중에만 ε 로 다른 제안을 내본다. **평가는 순수
argmin(결정론)** — 사전등록 모드다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("outputs/reports/yr189_q")
PREREG = (".claude/docs/strategy-history/"
          "2026-08-18-YR-189-Q전환-사전등록.md")
SPEC = ".claude/docs/dashboard-task-specs/YR-189-sell-axis-q-scoring.md"

ARM = "fixed15"                  # 48칸·견적망은 이 판에 올리지 않는다(단일축)
N_ITER, EPS_PER_ITER = 16, 4     # YR-183 실측: 곡선은 iter 11~13 에서 평평해졌다
LR = 3e-4                        # yr139 앵커 승계 (구 PPO 와 동일 — 축을 늘리지 않는다)
EXPLORE = 0.15                   # 제안 탐색(어느 작업을 낼까). 학습 중에만
EXPLORE_SIGMA = 0.20             # 좌표 탐색(어디로 팔까) — 점수에 얹는 잡음
BUFFER_MAX = 150_000
STEPS_PER_ITER = 600
# ★눈금 (판정 전 스모크에서 실측 — 평가 대역과 무관한 시드 9,900,002)
#   목표 −D_i 는 평균 +10.8 · 범위 −45~+127 이었다. 그대로 두면 Adam 스텝이 회당
#   ~3e-4 라 4,800 스텝으로는 10 단위 거리를 못 간다(YR-170 의 "critic 이 1/1,200
#   지점에서 멈춤"과 같은 유형의 실패). 고정 상수로 나눠 목표를 O(1) 로 만든다.
#   **판정에 영향 없음**: 결정 규칙이 `Q < KEEP_Q = 0` 이고 0/s = 0 이라 부호와
#   순서가 보존된다. 전역 매칭도 같은 상수로 나뉘므로 비교가 그대로다.
Q_SCALE = 20.0
MINIBATCH = 256
HUBER_BETA = 1.0
GRAD_CLIP = 1.0


# ------------------------------------------------------------------ 에피소드 1개
def _worker(args) -> dict:
    """(행, 목표) 표본을 만든다. 프로세스풀에서 돈다."""
    import torch
    import torch.multiprocessing as _mp
    from ..integrated.policy_config import ADOPTED_C0_GUARD
    from ..v2.sell_q import QCoordScorer, QSellPolicy, SellQNet
    from ..integrated.time_sell import DEFER_DELTA_S
    from ..integrated.yard_layout import terminal_layout
    from .yr139_blockq_v4_ppo import SLA_ANCHOR_S
    from .yr151_transfer_ppo import load_kf
    from .yr170_sell_ppo_diurnal import run_episode_diurnal
    from .yr174_txn_reward import TransactionLog, realized_credit
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")
    seed, sd, pol_seed, explore = args

    net = SellQNet()
    net.load_state_dict(sd)
    net.eval()
    layout = terminal_layout()
    scorer = QCoordScorer(net, layout, defer_delta_s=DEFER_DELTA_S,
                          time_slots=False,
                          explore_sigma=(EXPLORE_SIGMA if explore > 0 else 0.0),
                          seed=pol_seed + 1)
    pol = QSellPolicy(scorer, explore=explore, seed=pol_seed)
    ep = run_episode_diurnal(seed, pol, load_kf(), exec_config=ADOPTED_C0_GUARD,
                             day_plan_public=True, time_slots=False,
                             buy_net=None, q_scorer=scorer, _return_mbt=True)
    mbt = ep.pop("_mbt")
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    txns = TransactionLog().collect(ep["sell_ledger"])
    credits = realized_credit(mbt, txns, layout, l_t=SLA_ANCHOR_S + sla)
    # (t, src) 로 조인 — 한 블록은 한 epoch 에 제안을 하나만 낸다(유일 키).
    by_key = {(tx["t"], tx["src"]): tx["txn"] for tx in txns}
    rows, tgts, n_lab = [], [], 0
    for r in ep["q_rows"]:
        txn = by_key.get((round(r["t"], 6), r["src"]))
        d = credits.get(txn, 0.0) if txn is not None else 0.0
        if txn is not None:
            n_lab += 1
        rows.append(r["row"].detach().clone())
        tgts.append(-float(d) / Q_SCALE)  # ★부호 반전(이득 D → 비용 −D) + 눈금
    X = torch.stack(rows) if rows else torch.empty(0)
    y = torch.tensor(tgts, dtype=torch.float32) if tgts else torch.empty(0)
    return {"X": X, "y": y, "n_rows": len(rows), "n_committed": n_lab,
            "phi_final": ep["phi_final"], "n_space": ep["n_space"],
            "n_time": ep["n_time"], "n_txn": len(txns),
            "credit_sum": round(sum(credits.values()), 4),
            "offers": sum(1 for tr in pol.trail if tr["picked"] is not None),
            "decisions": len(pol.trail)}


# ------------------------------------------------------------------ 학습
def _fit(net, opt, buf_X, buf_y, *, steps: int, seed: int) -> dict:
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    n = buf_X.shape[0]
    net.train()
    losses = []
    for _ in range(steps):
        idx = torch.randint(n, (min(MINIBATCH, n),), generator=g)
        pred = net(buf_X[idx])
        loss = F.smooth_l1_loss(pred, buf_y[idx], beta=HUBER_BETA)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        losses.append(float(loss.item()))
    net.eval()
    with torch.no_grad():
        pred = net(buf_X)
        resid = float(((pred - buf_y) ** 2).mean().item()) ** 0.5
        base = float((buf_y ** 2).mean().item()) ** 0.5
    return {"loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
            "rmse": round(resid, 6), "target_rms": round(base, 6),
            "explained": round(1.0 - (resid / base) ** 2, 4) if base > 1e-9 else None,
            "pred_rms": round(float((pred ** 2).mean().item()) ** 0.5, 6),
            "pred_mean": round(float(pred.mean().item()), 6),
            "target_mean": round(float(buf_y.mean().item()), 6),
            "buffer": n}


def train(ts: int, *, n_iter: int = N_ITER, eps_per_iter: int = EPS_PER_ITER,
          workers: int = EPS_PER_ITER, out_root: Path = OUT) -> Path:
    import torch
    import torch.multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor
    from ..integrated.repro import code_dirty, repro_stamp, write_result
    from ..v2.sell_q import Q_ROW_DIM, SellQNet
    from ..integrated.terminal_stream import OBS_24H
    _mp.set_sharing_strategy("file_system")
    torch.set_num_threads(1)
    torch.manual_seed(ts)
    out = out_root / f"s{ts}"
    out.mkdir(parents=True, exist_ok=True)

    net = SellQNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    buf_X = torch.empty(0, Q_ROW_DIM)
    buf_y = torch.empty(0)
    hist: list[dict] = []
    for it in range(n_iter):
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        jobs = [(ts + it * eps_per_iter + e, sd, ts + it * 100 + e, EXPLORE)
                for e in range(eps_per_iter)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            eps = list(pool.map(_worker, jobs))
        new_X = [e["X"] for e in eps if e["n_rows"]]
        new_y = [e["y"] for e in eps if e["n_rows"]]
        X = torch.cat(new_X) if new_X else buf_X[:0]
        y = torch.cat(new_y) if new_y else buf_y[:0]
        buf_X = torch.cat([buf_X, X])[-BUFFER_MAX:]
        buf_y = torch.cat([buf_y, y])[-BUFFER_MAX:]
        if buf_X.shape[0] == 0:
            raise RuntimeError(f"iter {it}: 표본이 0 — 제안이 한 건도 확정되지 않았다"
                               " (조용한 무학습 금지)")
        fit = _fit(net, opt, buf_X, buf_y, steps=STEPS_PER_ITER, seed=ts + it)
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
            {"train_seed": ts, "arm": ARM, "history": hist,
             "in_progress": it < n_iter - 1}, ensure_ascii=False, indent=1,
            default=str), encoding="utf-8")
        torch.save({"q": net.state_dict()}, out / "net.pt")

    payload = {"experiment": "YR-189 판매 Q 전환 학습", "prereg": PREREG,
               "spec": SPEC, "kind": "training", "arm": ARM,
               "train_seed": ts, "history": hist,
               "changed_axis": "학습 방식 PPO→Q 회귀 (무대·보상·집행·팔 동일)",
               "code_dirty": bool(code_dirty()),
               "stamp": repro_stamp(
                   experiment="YR-189 Q 전환 학습", seeds={"train": [ts]},
                   params={"arm": ARM, "N_ITER": n_iter,
                           "EPS_PER_ITER": eps_per_iter, "LR": LR,
                           "EXPLORE": EXPLORE,
                           "EXPLORE_SIGMA": EXPLORE_SIGMA,
                           "BUFFER_MAX": BUFFER_MAX,
                           "STEPS_PER_ITER": STEPS_PER_ITER,
                           "Q_SCALE": Q_SCALE,
                           "MINIBATCH": MINIBATCH, "HUBER_BETA": HUBER_BETA,
                           "GRAD_CLIP": GRAD_CLIP,
                           "observation": OBS_24H.as_dict()},
                   prereg=PREREG)}
    p = out / "train_all.json"
    write_result(p, payload)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8_400_000)
    ap.add_argument("--iters", type=int, default=N_ITER)
    ap.add_argument("--eps", type=int, default=EPS_PER_ITER)
    ap.add_argument("--workers", type=int, default=EPS_PER_ITER)
    a = ap.parse_args()
    p = train(a.seed, n_iter=a.iters, eps_per_iter=a.eps, workers=a.workers)
    d = json.loads(p.read_text(encoding="utf-8"))
    hdr = ("it", "Phi", "offer", "space", "time", "rows", "RMSE", "R2")
    print("{:>3} {:>9} {:>6} {:>6} {:>6} {:>7} {:>8} {:>7}".format(*hdr))
    for h in d["history"]:
        ex = h["explained"] if h["explained"] is not None else float("nan")
        print(f"{h['iter']:>3} {h['phi_final_mean']:>9.1f} {h['offers_mean']:>6.0f} "
              f"{h['n_space_mean']:>6.0f} {h['n_time_mean']:>6.0f} "
              f"{h['buffer']:>7} {h['rmse']:>8.4f} {ex:>7.3f}")
    print("DONE", p)
