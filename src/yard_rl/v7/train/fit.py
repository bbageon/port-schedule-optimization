"""두 학생을 **같은 눈금으로 동시에** 학습한다.

설계 정본: `.claude/docs/architecture/06-학습과-판정.md` §1 · §1-1

■ 회귀다 — PPO 도 critic 도 아니다
  선택지가 이산이라 각 세계의 Φ 를 배우고 argmin 으로 고른다. 라벨이 **그 시점
  실현 손익**이라 부트스트랩이 없다. 논문에 "Q 학습" 이라 쓰지 않는다.

■ 두 망의 눈금을 맞춘다
  각자 학습하면 편향이 서로 안 맞는다. **같은 `PHI_SCALE`** 로 동시에 갱신한다.

■ v2 의 알려진 설정 결함을 고친다 (06 §1)
  · σ 를 절대값으로 고정 → **목표 표준편차 대비 상대값**
  · 버퍼 FIFO 축출 → 반사실 라벨은 **회차마다 새로 만들므로** 버퍼 자체가 없다
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

LR = 3e-4                 # yr139 앵커 승계
MINIBATCH = 256
HUBER_BETA = 1.0
GRAD_CLIP = 1.0
STEPS_PER_ITER = 600

#: 탐색 강도 — **목표 표준편차 대비 상대값**(v2 는 절대값이라 눈금이 줄면 8.7배가 됐다)
EXPLORE_REL_SIGMA = 0.20
EXPLORE_EPS = 0.15


@dataclass
class FitReport:
    seller_loss: float = 0.0
    buyer_loss: float = 0.0
    n_seller: int = 0
    n_buyer: int = 0
    steps: int = 0

    def as_dict(self) -> dict:
        return {"seller_loss": self.seller_loss, "buyer_loss": self.buyer_loss,
                "n_seller": self.n_seller, "n_buyer": self.n_buyer,
                "steps": self.steps}


class StudentTrainer:
    """Seller 망과 Buyer 망을 한 옵티마이저 스텝에서 같이 민다."""

    def __init__(self, seller_net: nn.Module, buyer_net: nn.Module, *,
                 lr: float = LR, minibatch: int = MINIBATCH,
                 steps_per_iter: int = STEPS_PER_ITER):
        self.seller_net = seller_net
        self.buyer_net = buyer_net
        self.opt = torch.optim.Adam(
            list(seller_net.parameters()) + list(buyer_net.parameters()), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(beta=HUBER_BETA)
        self.minibatch = int(minibatch)
        self.steps_per_iter = int(steps_per_iter)

    @staticmethod
    def _sample(x: torch.Tensor, y: torch.Tensor, n: int, g: torch.Generator):
        if x.numel() == 0:
            return None, None
        idx = torch.randint(0, x.shape[0], (min(n, x.shape[0]),), generator=g)
        return x[idx], y[idx]

    def evaluate(self, labels) -> tuple[float, float]:
        """**학습에 안 쓴 표본**의 손실 — 진짜 오차는 이쪽이다.

        학습 손실은 망이 이미 본 점에 대한 오차라, 파라미터가 표본보다 많으면
        (5,761 vs 수십) 얼마든지 작아진다. argmin 이 실전에서 마주하는 것은
        **처음 보는 상황**이므로 그쪽 오차를 봐야 한다.
        """
        out = []
        for which, net in (("SELLER", self.seller_net), ("BUYER", self.buyer_net)):
            x, y = labels.tensors(which)
            if x.numel() == 0:
                out.append(0.0)
                continue
            with torch.no_grad():
                out.append(float(self.loss_fn(net(x), y).item()))
        return out[0], out[1]

    def fit(self, labels, *, seed: int = 0) -> FitReport:
        """한 회차 분 라벨로 두 망을 갱신한다.

        **표본 0 이면 즉시 멈춘다** — 06 하드가드. 독립 행위자라 Buyer 가 전량
        거절하면 거래가 0 이 되고 학습 신호가 사라진다. 그건 "학습이 안 된 것" 과
        다른 실패이므로 구분해 보고해야 한다.
        """
        xs, ys = labels.tensors("SELLER")
        xb, yb = labels.tensors("BUYER")
        rep = FitReport(n_seller=int(xs.shape[0]) if xs.numel() else 0,
                        n_buyer=int(xb.shape[0]) if xb.numel() else 0)
        if rep.n_seller == 0:
            raise RuntimeError("Seller 표본 0 — 학습 신호가 없다(즉시 중단)")

        g = torch.Generator().manual_seed(int(seed))
        s_loss = b_loss = 0.0
        for _ in range(self.steps_per_iter):
            self.opt.zero_grad(set_to_none=True)
            total = torch.zeros((), dtype=torch.float32)

            bx, by = self._sample(xs, ys, self.minibatch, g)
            if bx is not None:
                ls = self.loss_fn(self.seller_net(bx), by)
                total = total + ls
                s_loss = float(ls.item())

            bx, by = self._sample(xb, yb, self.minibatch, g)
            if bx is not None:
                lb = self.loss_fn(self.buyer_net(bx), by)
                total = total + lb
                b_loss = float(lb.item())

            total.backward()
            nn.utils.clip_grad_norm_(
                list(self.seller_net.parameters()) + list(self.buyer_net.parameters()),
                GRAD_CLIP)
            self.opt.step()
            rep.steps += 1

        rep.seller_loss, rep.buyer_loss = s_loss, b_loss
        return rep


def explore_sigma(targets: torch.Tensor) -> float:
    """탐색 σ 를 **목표 표준편차 대비 상대값**으로 만든다.

    v2 는 절대값 0.20 을 고정해, 목표 눈금이 줄자 상대 강도가 1.8배 → 8.7배가 됐다
    ([[YR-193]]). 여기서는 매 회차 목표에서 다시 뽑는다.
    """
    if targets.numel() < 2:
        return 0.0
    return float(targets.std().item()) * EXPLORE_REL_SIGMA
