"""불확실성 분해 — 예측 오차 중 **못 줄이는 몫**은 얼마인가.

■ 질문
Q 망의 오차가 신호의 82% 다(RMSE 0.019 / 목표RMS 0.023). 이 중
  · 더 배우면 줄어드는 것 (정보 부족·학습 부족)
  · 원래 못 줄이는 것 (라벨 근사·가려진 미래)
이 각각 얼마인가. 앞이 크면 정보를 더 주는 게 답이고, 뒤가 크면 **문턱밖에
없다**.

■ 재는 법 (1-최근접이웃 잔차 분산 · Devroye–Györfi 계열)
행이 거의 같은 두 표본은 **어떤 모델도 같은 값을 예측할 수밖에 없다.**
그런데 라벨이 다르면 그 차이는 구조적으로 설명 불가다.

    sigma^2_noise  ≈  (1/2n) * Σ (y_i − y_NN(i))^2

★중요 — 이 하한은 **지금 특징 표현에 상대적**이다. 특징을 더 주면(예: 크레인의
예상 서비스 시각) 하한이 내려갈 수 있다. 그래서 이 값은 "**지금 주는 정보로는**
여기까지"라는 뜻이고, 정보 추가가 값어치 있는지를 그대로 말해준다.
"""
from concurrent.futures import ProcessPoolExecutor

CKPT = "outputs/reports/yr189_q/s8400000/net.pt"
SEEDS = [9_900_070, 9_900_071, 9_900_072, 9_900_073, 9_900_074, 9_900_075]


def one(seed):
    """학습과 같은 분포(탐색 켠 상태)에서 (행, 라벨) 수집."""
    import torch
    import torch.multiprocessing as _mp
    from yard_rl.integrated.policy_config import ADOPTED_C0_GUARD
    from yard_rl.integrated.sell_q import QCoordScorer, QSellPolicy, SellQNet
    from yard_rl.integrated.terminal_stream import OBS_24H
    from yard_rl.integrated.time_sell import DEFER_DELTA_S
    from yard_rl.integrated.yard_layout import terminal_layout
    from yard_rl.experiments.yr139_blockq_v4_ppo import SLA_ANCHOR_S
    from yard_rl.experiments.yr151_transfer_ppo import load_kf
    from yard_rl.experiments.yr170_sell_ppo_diurnal import run_episode_diurnal
    from yard_rl.experiments.yr174_txn_reward import (TransactionLog,
                                                      realized_credit)
    from yard_rl.experiments.yr189_q_train import EXPLORE, EXPLORE_SIGMA, Q_SCALE
    torch.set_num_threads(1)
    _mp.set_sharing_strategy("file_system")

    layout, kf = terminal_layout(), load_kf()
    net = SellQNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["q"])
    net.eval()
    sc = QCoordScorer(net, layout, defer_delta_s=DEFER_DELTA_S, time_slots=False,
                      explore_sigma=EXPLORE_SIGMA, seed=seed + 1)
    pol = QSellPolicy(sc, explore=EXPLORE, seed=seed)
    ep = run_episode_diurnal(seed, pol, kf, obs=OBS_24H,
                             exec_config=ADOPTED_C0_GUARD, day_plan_public=True,
                             time_slots=False, buy_net=None, q_scorer=sc,
                             _return_mbt=True)
    mbt = ep.pop("_mbt")
    sla = next(iter(mbt.blocks.values())).profile.long_wait_sla_s
    txns = TransactionLog().collect(ep["sell_ledger"])
    cr = realized_credit(mbt, txns, layout, l_t=SLA_ANCHOR_S + sla)
    key = {(t["t"], t["src"]): t["txn"] for t in txns}
    X, y, ax = [], [], []
    with torch.no_grad():
        for r in ep["q_rows"]:
            tx = key.get((round(r["t"], 6), r["src"]))
            d = cr.get(tx, 0.0) if tx is not None else 0.0
            X.append(r["row"].detach().clone())
            y.append(-float(d) / Q_SCALE)
            ax.append(1 if str(r["coord"]).startswith("TIME") else 0)
    return (torch.stack(X), torch.tensor(y), torch.tensor(ax))


def nn_noise(X, y, chunk=2048):
    """1-최근접이웃 잔차로 잡음 표준편차 추정."""
    import torch
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-8)
    Z = (X - mu) / sd                      # 차원별 눈금 통일
    n = Z.shape[0]
    best = torch.empty(n, dtype=torch.long)
    for i in range(0, n, chunk):
        d = torch.cdist(Z[i:i + chunk], Z)
        d[torch.arange(d.shape[0]), torch.arange(i, min(i + chunk, n))] = float("inf")
        best[i:i + chunk] = d.argmin(1)
    diff = y - y[best]
    return float((diff.pow(2).mean() / 2).sqrt()), best


if __name__ == "__main__":
    import torch
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        res = list(pool.map(one, SEEDS))
    X = torch.cat([r[0] for r in res])
    y = torch.cat([r[1] for r in res])
    ax = torch.cat([r[2] for r in res])
    print(f"표본 {X.shape[0]:,} (공간 {int((ax==0).sum()):,} · 시간 {int((ax==1).sum()):,})")

    net_rmse = None
    from yard_rl.integrated.sell_q import SellQNet
    net = SellQNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["q"])
    net.eval()
    with torch.no_grad():
        pred = net(X)
    net_rmse = float((pred - y).pow(2).mean().sqrt())

    print()
    print(f"{'구분':>8} {'n':>8} {'신호RMS':>9} {'모델RMSE':>10} "
          f"{'잡음하한':>9} {'설명가능 여지':>12} {'잡음/신호':>9}")
    for name, m in (("전체", torch.ones_like(ax, dtype=torch.bool)),
                    ("공간", ax == 0), ("시간", ax == 1)):
        Xi, yi = X[m], y[m]
        if Xi.shape[0] < 100:
            continue
        sig = float(yi.pow(2).mean().sqrt())
        with torch.no_grad():
            r = float((net(Xi) - yi).pow(2).mean().sqrt())
        noise, _ = nn_noise(Xi, yi)
        # 모델 오차^2 = 잡음^2 + 설명가능하나 못 배운 몫^2 (근사 분해)
        gap = max(0.0, r * r - noise * noise) ** 0.5
        print(f"{name:>8} {Xi.shape[0]:>8,} {sig:>9.5f} {r:>10.5f} "
              f"{noise:>9.5f} {gap:>12.5f} {noise/max(sig,1e-9):>9.1%}")

    print()
    print("=== 0 라벨(미측정·미확정)을 뺀 경우 ===")
    nz = y.abs() > 1e-12
    Xi, yi = X[nz], y[nz]
    sig = float(yi.pow(2).mean().sqrt())
    with torch.no_grad():
        r = float((net(Xi) - yi).pow(2).mean().sqrt())
    noise, _ = nn_noise(Xi, yi)
    gap = max(0.0, r * r - noise * noise) ** 0.5
    print(f"  n {Xi.shape[0]:,} · 신호RMS {sig:.5f} · 모델RMSE {r:.5f} · "
          f"잡음하한 {noise:.5f} · 남은 여지 {gap:.5f}")
    print(f"  0 라벨 비율 {(~nz).float().mean():.1%}")

    print()
    print("※ 잡음 하한은 **지금 특징 표현에 상대적**이다. 특징을 더 주면 내려갈 수 있다.")
    print("※ 원단위 환산은 × 20 (Q_SCALE).")
