"""Seller · Buyer 망 — **비용을 낸다**(확률이 아니다).

설계 정본: `.claude/docs/architecture/03-결정층.md` §3-1

선택지가 **이산**이라 각 세계의 Φ 를 회귀로 배우고 **argmin** 으로 고른다.
PPO 도 critic 도 안 들인다 — v1 이 무너진 자리다([[YR-180]] 보상 91%가 0).

■ 두 망을 같은 눈금으로 둔다
  각자 학습하면 편향이 서로 안 맞는다. **같은 `PHI_SCALE` 로 동시 학습**하고,
  절대량(차가 아니라)을 배우므로 눈금은 **비판정 시드에서 재실측 후 동결**한다.

■ 가중치는 1벌씩이다
  21블록이 각자 자기 상태로 같은 Seller 망을 실행한다(중앙학습·분산실행).
  Buyer 도 마찬가지 — 블록마다 다른 망을 두지 않는다.
"""
from __future__ import annotations

import torch
from torch import nn

from ..features.candidate import BUYER_ROW_DIM, SELLER_ROW_DIM

HID = 64

#: 목표 눈금 — 원화 Φ 를 O(1) 로 만든다. **비판정 시드에서 실측 후 동결**.
#: v2 의 `Q_SCALE = 20.0` 은 비용시간 단위였다. v3 는 원화라 눈금이 완전히 다르다.
PHI_SCALE = 1_000_000.0


class _CostHead(nn.Module):
    """행 하나당 비용 하나. 낮을수록 싸다."""

    def __init__(self, in_dim: int, hid: int = HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows).squeeze(-1)


class SellerNet(_CostHead):
    """`Φ_H(KEEP)` · `Φ_H(SELL(공간 b))` · `Φ_H(SELL(시간 슬롯 k))` 를 예측한다."""

    def __init__(self, hid: int = HID):
        super().__init__(SELLER_ROW_DIM, hid)


class BuyerNet(_CostHead):
    """`Φ_H(REJECT)` · `Φ_H(BUY)` 를 예측한다 — **실제로 온 offer 를 조건으로**."""

    def __init__(self, hid: int = HID):
        super().__init__(BUYER_ROW_DIM, hid)


def to_scaled(phi_krw: float) -> float:
    """원화 Φ → 학습 목표 눈금."""
    return float(phi_krw) / PHI_SCALE


def from_scaled(y: float) -> float:
    """학습 목표 눈금 → 원화 Φ."""
    return float(y) * PHI_SCALE
