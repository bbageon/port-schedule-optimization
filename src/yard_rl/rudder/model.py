"""소형 LSTM — **지금까지 본 것만으로** 창의 최종 차이를 예측한다 ([[YR-223]] 2단계).

■ 왜 LSTM 인가 (Transformer 가 아니라)
  원 논문이 LSTM 이고, 무엇보다 **데이터가 적다.** 궤적 200개 × 180토큰 = 36,000
  토큰이다. v3 의 5,761 파라미터 MLP 가 표본 70행에 과적합하는 걸 방금 봤다
  ([[YR-222]]). Transformer 는 그보다 수십 배 크다.

      은닉 32 → 파라미터 약 5,900
      은닉 64 → 약 20,000

  **은닉 32 로 시작**하고, 관문 A(예측)를 통과한 뒤에만 키운다.

■ ★인과성은 **구조가** 보장한다
  단방향 LSTM 은 t 시점 출력이 x₁..x_t 만 본다. mask 를 따로 안 걸어도 미래가
  못 샌다. 양방향으로 바꾸는 순간 라벨이 정답지가 되므로 **금지**한다.

■ 손실 세 갈래 (RUDDER 원 논문 구성)
      주손실   (g_T − Y)²              창 끝에서 맞혀라
      보조     mean_t (g_t − Y)²        ★**일찍부터** 맞혀라 — 이게 기여도를 앞으로 민다
      정규화   mean_t g_t²  (작게)      예측이 제멋대로 튀지 않게

  보조 손실이 핵심이다. 이게 없으면 모델이 마지막 토큰에서만 답을 맞히고
  중간 예측이 아무 의미가 없어져 **기여도가 전부 마지막에 몰린다.**

■ 눈금
  Y 는 원화라 수백만 단위다. 그대로 넣으면 손실이 폭발한다. 학습 집합의 표준편차로
  나눠 O(1) 로 만들고, 그 값을 **체크포인트에 박아** 나중에 원화로 되돌린다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .tape import ACTION_IDX, DIM

#: 사전등록 기본값 — 바꾸면 재실측 후 동결한다.
HIDDEN = 32
W_AUX = 0.5
W_REG = 0.01


@dataclass
class Norm:
    """토큰 칸별 z 정규화 + Y 눈금. **학습 집합에서만** 구한다."""

    mean: list = field(default_factory=lambda: [0.0] * DIM)
    std: list = field(default_factory=lambda: [1.0] * DIM)
    y_scale: float = 1.0

    @classmethod
    def fit(cls, windows) -> "Norm":
        cols = [[] for _ in range(DIM)]
        for w in windows:
            for tk in w.tokens:
                for j, v in enumerate(tk):
                    cols[j].append(float(v))
        mean, std = [], []
        for c in cols:
            n = max(1, len(c))
            m = sum(c) / n
            var = sum((v - m) ** 2 for v in c) / max(1, n - 1)
            mean.append(m)
            std.append(max(1e-6, var ** 0.5))
        ys = [float(w.y_krw) for w in windows]
        ym = sum(ys) / max(1, len(ys))
        yv = sum((v - ym) ** 2 for v in ys) / max(1, len(ys) - 1)
        return cls(mean=mean, std=std, y_scale=max(1.0, yv ** 0.5))

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "y_scale": self.y_scale}


class RudderNet(nn.Module):
    """x(T×DIM) → g(T) — 각 시점의 **최종 Y 예측**(눈금 단위)."""

    def __init__(self, dim: int = DIM, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, num_layers=1, batch_first=True,
                            bidirectional=False)     # ★단방향 = 인과
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                             # x: (B, T, DIM)
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)               # (B, T)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rudder_loss(g, y, *, mask=None, w_aux: float = W_AUX, w_reg: float = W_REG):
    """`g`(B,T) · `y`(B,) — 주손실 + 보조 + 정규화.

    `mask`(B,T) 는 창마다 토큰 수가 달라 채운 자리를 빼는 데 쓴다.
    """
    if mask is None:
        mask = torch.ones_like(g)
    last = mask.sum(dim=1).long() - 1                 # 창별 마지막 실제 토큰
    g_last = g[torch.arange(g.shape[0]), last]
    yy = y.unsqueeze(1)
    n = mask.sum().clamp(min=1.0)
    main = ((g_last - y) ** 2).mean()
    aux = (((g - yy) ** 2) * mask).sum() / n
    reg = ((g ** 2) * mask).sum() / n
    return main + w_aux * aux + w_reg * reg, {"main": float(main.detach()),
                                              "aux": float(aux.detach())}


def to_tensor(windows, norm: Norm, *, ablate_actions: bool = False,
              shuffle_order: bool = False, gen: torch.Generator | None = None):
    """창 목록 → (x, y, mask). 관문 B·D 가 여기서 재료를 망가뜨린다.

    `ablate_actions` — ★관문 B: 행동 칸을 0 으로. 이래도 예측이 안 나빠지면
    모델은 "몇 시냐" 만 외운 것이다.
    `shuffle_order` — ★관문 D: 토큰 **순서**를 섞는다. 시간 구조가 정말 쓰이면 나빠진다.
    """
    T = max((len(w.tokens) for w in windows), default=1)
    B = len(windows)
    x = torch.zeros(B, T, DIM)
    mask = torch.zeros(B, T)
    y = torch.zeros(B)
    for i, w in enumerate(windows):
        rows = [list(r) for r in w.tokens]
        if ablate_actions:
            for r in rows:
                for j in ACTION_IDX:
                    r[j] = 0.0
        if shuffle_order and len(rows) > 1:
            idx = torch.randperm(len(rows), generator=gen).tolist()
            rows = [rows[k] for k in idx]
        for t, r in enumerate(rows):
            for j, v in enumerate(r):
                x[i, t, j] = (float(v) - norm.mean[j]) / norm.std[j]
            mask[i, t] = 1.0
        y[i] = float(w.y_krw) / norm.y_scale
    return x, y, mask
