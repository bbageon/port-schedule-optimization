"""크레인 망 학습 ([[YR-305]]).

재배정층 `train/fit.py` 와 **같은 설정**을 쓴다 — 크레인 축의 효과를 보려는 것이지
학습 설정의 효과를 보려는 게 아니다. 설정이 다르면 비교가 흐려진다.

■ ⚠️ 눈금 규약 ([[YR-304]] 의 교훈)
  `CRANE_ADV_SCALE` 은 아직 **실측 없이 정한 잠정값**이다. 재배정층은 [[YR-217]] 에서
  라벨 중앙 격차를 재고 10만원으로 동결했는데, 크레인 결정은 그런 실측이 없다.
  그래서 이 학습기는 **매 회차 목표의 중앙 격차를 기록한다** — 그 값이 모이면
  눈금을 다시 못박을 수 있다. 기록이 없으면 눈금 문제를 검증조차 못 한다
  ([[YR-299]] B 가 그래서 못 가려졌다).
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass

import torch
from torch import nn

from ..train.fit import GRAD_CLIP, HUBER_BETA, LR, MINIBATCH
from .policy import CRANE_ADV_SCALE, CraneNet, from_advantage


@dataclass
class CraneFitReport:
    n: int = 0                    # 학습 표본 수
    n_val: int = 0                # 검증 표본 수
    loss: float = 0.0             # 학습 손실 (마지막 미니배치)
    val_loss: float = 0.0         # 검증 손실 ★진짜 지표
    steps: int = 0
    #: ★목표 중앙 격차(원) — 눈금이 맞는지 보는 값. `CRANE_ADV_SCALE` 근처여야 한다.
    median_gap_krw: float = 0.0
    zero_ratio: float = 0.0       # 차이가 0 이던 결정 비율


class CraneTrainer:
    """표본을 받아 망을 갱신한다. 재배정층 `StudentTrainer` 와 같은 골격."""

    def __init__(self, net: CraneNet, *, lr: float = LR,
                 minibatch: int = MINIBATCH):
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(beta=HUBER_BETA)
        self.minibatch = int(minibatch)

    def fit(self, labels, *, seed: int = 0, val_frac: float = 0.2) -> CraneFitReport:
        """★검증분은 **결정 단위**로 가른다.

        한 결정이 표본 둘(고른 것·대안)을 만드는데, 그 둘이 학습·검증으로 갈리면
        **같은 결정을 보고 채점**하는 셈이라 검증이 낙관적으로 나온다.
        """
        rep = CraneFitReport()
        samples = labels.samples
        if len(samples) < 4:
            return rep

        # 목표 격차를 먼저 기록한다 — 학습이 실패해도 이 값은 남아야 한다
        gaps = [abs(from_advantage(samples[i].target - samples[i + 1].target))
                for i in range(0, len(samples) - 1, 2)]
        rep.median_gap_krw = st.median(gaps) if gaps else 0.0
        rep.zero_ratio = (labels.zero / max(1, len(gaps)))

        # 결정 단위 분할 — 표본이 (고른 것, 대안) 짝으로 들어온다
        n_dec = len(samples) // 2
        rng = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(n_dec, generator=rng).tolist()
        n_val_dec = max(1, int(n_dec * val_frac))
        val_dec = set(order[:n_val_dec])
        tr_idx = [i for d in range(n_dec) if d not in val_dec
                  for i in (2 * d, 2 * d + 1)]
        va_idx = [i for d in range(n_dec) if d in val_dec
                  for i in (2 * d, 2 * d + 1)]
        if not tr_idx or not va_idx:
            return rep

        pack = lambda idx: (
            torch.tensor([samples[i].row for i in idx], dtype=torch.float32),
            torch.tensor([samples[i].target for i in idx], dtype=torch.float32))
        xt, yt = pack(tr_idx)
        xv, yv = pack(va_idx)
        rep.n, rep.n_val = len(tr_idx), len(va_idx)

        self.net.train()
        steps = max(50, min(600, 4 * len(tr_idx)))
        for _ in range(steps):
            b = torch.randint(0, len(tr_idx), (min(self.minibatch, len(tr_idx)),),
                              generator=rng)
            self.opt.zero_grad()
            loss = self.loss_fn(self.net(xt[b]), yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
            self.opt.step()
            rep.loss = float(loss.detach())
        rep.steps = steps

        self.net.eval()
        with torch.no_grad():
            rep.val_loss = float(self.loss_fn(self.net(xv), yv))
        return rep


def scale_health(median_gap_krw: float) -> str:
    """눈금이 맞는지 한 줄로 — 목표가 1 근처여야 학습이 안정적이다."""
    if median_gap_krw <= 0:
        return "잴 수 없음 (표본 없음)"
    ratio = median_gap_krw / CRANE_ADV_SCALE
    if 0.3 <= ratio <= 3.0:
        return f"적정 (목표 중앙 {ratio:.2f})"
    if ratio < 0.3:
        return f"★눈금이 크다 — 목표가 {ratio:.3f} 로 작아 신호가 묻힌다"
    return f"★눈금이 작다 — 목표가 {ratio:.1f} 로 커 학습이 튄다"
