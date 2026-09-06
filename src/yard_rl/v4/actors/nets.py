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


#: ★행동 눈금 (원) — **학습 목표 전용** ([[YR-220]]).
#:
#: [[YR-218]] 은 누적 Φ 를 목표로 줬다가 실패했다. 두 세계가 분기 전 과거를 공유해
#: 목표의 99.9% 가 공통분이었고, 망 잔차가 행동 효과의 **43~94배**였다 —
#: argmin 이 행동이 아니라 잡음을 골랐다.
#:
#: v3 는 **그 결정의 평균을 빼고** 가르친다. 그러면 목표가 곧 행동 효과라
#: 눈금도 행동 효과 크기로 잡아야 O(1) 이 된다.
#: 값의 근거는 [[YR-217]] 실측 — 성사된 거래 1건의 라벨 중앙 격차
#: **103,418원**(부하 3,500) · **112,367원**(5,000). 10만원으로 동결한다.
#:
#: ⚠️ 눈금이 바뀌면 재실측 후 동결한다(06 §1 · v2 `Q_SCALE` 과 같은 규약).
ADV_SCALE = 100_000.0


def to_advantage(phi: float, base: float) -> float:
    """`(Φ − 그 결정의 기준선) / ADV_SCALE` — **학습 목표는 이것**이다.

    기준선은 그 결정에서 굴린 세계들의 평균이다. 양쪽에서 같은 값을 빼므로
    **argmin 의 순서는 안 바뀐다**(상수 차감).
    """
    return (float(phi) - float(base)) / ADV_SCALE


def from_advantage(x: float) -> float:
    """망 출력 → 원화 (기준선 대비). 보고·진단용."""
    return float(x) * ADV_SCALE


def to_scaled(phi_krw: float) -> float:
    """원화 Φ → 학습 목표 눈금."""
    return float(phi_krw) / PHI_SCALE


def from_scaled(y: float) -> float:
    """학습 목표 눈금 → 원화 Φ."""
    return float(y) * PHI_SCALE
