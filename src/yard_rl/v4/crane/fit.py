"""크레인 망 학습 ([[YR-305]]).

재배정층 `train/fit.py` 와 **같은 설정**을 쓴다 — 크레인 축의 효과를 보려는 것이지
학습 설정의 효과를 보려는 게 아니다. 설정이 다르면 비교가 흐려진다.

■ ⚠️ 눈금 규약 ([[YR-304]] 의 교훈 · [[YR-309]] 로 갱신)
  **눈금은 그날 라벨의 중앙 격차다** — 전 부하 공통 상수가 아니다. 상수를 쓰면
  부하 12,500 인 날의 목표가 3.99, 부하 3,500 인 날이 0.30 으로 13배 벌어져
  학습률이 고정인데 미는 세기가 13배 달라진다 ([[YR-308]] 실측).

  그래도 **매 회차 격차 중앙과 실제 쓴 눈금을 함께 기록한다** — 기록이 없으면
  눈금 문제를 검증조차 못 한다 ([[YR-299]] B 가 그래서 못 가려졌다).
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass

import torch
from torch import nn

from ..train.fit import GRAD_CLIP, HUBER_BETA, LR, MINIBATCH
from .policy import CRANE_ADV_SCALE, CraneNet


@dataclass
class CraneFitReport:
    n: int = 0                    # 학습 표본 수
    n_val: int = 0                # 검증 표본 수
    loss: float = 0.0             # 학습 손실 (마지막 미니배치)
    val_loss: float = 0.0         # 검증 손실 ★진짜 지표
    steps: int = 0
    #: ★목표 중앙 격차(원) — 그날 라벨이 얼마짜리 결정이었나
    median_gap_krw: float = 0.0
    #: ★실제로 쓴 눈금(원). [[YR-309]] 이후로는 `median_gap_krw` 와 같다 —
    #:  단 하한(`CRANE_SCALE_FLOOR`)이 걸린 날만 다르다.
    scale_krw: float = CRANE_ADV_SCALE
    floor_bound: bool = False     # 하한이 걸렸나 = 그날은 비용 중립에 가까웠다
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

        # 목표 격차를 먼저 기록한다 — 학습이 실패해도 이 값은 남아야 한다.
        # ★원화 원본에서 바로 잰다([[YR-309]]) — 목표를 눈금으로 되돌리면 눈금이
        #   틀렸을 때 그 오류가 진단값에도 그대로 실려 **자기 자신을 검증하게** 된다.
        gaps = [abs(samples[i].phi - samples[i + 1].phi)
                for i in range(0, len(samples) - 1, 2)]
        rep.median_gap_krw = st.median(gaps) if gaps else 0.0
        rep.scale_krw = float(getattr(labels, "scale", CRANE_ADV_SCALE))
        rep.floor_bound = bool(getattr(labels, "floor_bound", False))
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


def scale_health(median_gap_krw: float, scale: float | None = None) -> str:
    """눈금이 맞는지 한 줄로 — 목표가 1 근처여야 학습이 안정적이다.

    `scale` 을 주면 **그 눈금 기준**으로 본다. [[YR-309]] 이후 학습 경로는 눈금을
    그날 격차로 잡으므로 이 값이 항상 1.00 근처여야 한다 — **불변식 검사**가 된다.
    (하한이 걸린 날만 1 미만으로 나온다.)
    """
    if median_gap_krw <= 0:
        return "잴 수 없음 (표본 없음)"
    ratio = median_gap_krw / float(scale or CRANE_ADV_SCALE)
    if 0.3 <= ratio <= 3.0:
        return f"적정 (목표 중앙 {ratio:.2f})"
    if ratio < 0.3:
        return f"★눈금이 크다 — 목표가 {ratio:.3f} 로 작아 신호가 묻힌다"
    return f"★눈금이 작다 — 목표가 {ratio:.1f} 로 커 학습이 튄다"
