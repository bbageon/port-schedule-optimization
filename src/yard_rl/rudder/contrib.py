"""기여도 = **예측이 얼마나 움직였나** ([[YR-223]] 3단계).

■ 한 줄 정의
      c_t = g_t − g_{t−1}          (g_0 = 0)
      Σ_t c_t = g_T ≈ Y            ← 관문 E 가 이 등식을 검사한다

  "그 epoch 을 보고 나서 최종 차이 예측이 얼마나 바뀌었나" 다. 예측이 안 움직였으면
  그 epoch 은 아무것도 안 한 것이고, 크게 움직였으면 거기서 뭔가 일어난 것이다.

■ ⚠️ 이건 **epoch 몫**이지 개별 행동 몫이 아니다
  한 epoch 에 판매·구매 결정이 여러 건 들어간다. 그중 누구 몫인지는 이 모델이
  답하지 못한다 — 그건 반사실이 답한다(분업, [[YR-223]] §3-⑤).

■ ⚠️ 그리고 이건 **상관이지 인과가 아니다**
  모델은 "이런 epoch 뒤엔 이런 Y 가 오더라" 를 배웠을 뿐이다. 그게 정말 인과인지는
  **관문 F** 가 판정한다 — 그 epoch 에 실제로 개입해 부호가 맞는지 본다.
  우리에겐 반사실 시뮬레이터가 있어서 할 수 있는 검증이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Norm, to_tensor


@dataclass
class EpochContrib:
    t: float
    krw: float          # 기여도 (원화)
    rank: int = 0       # |krw| 내림차순 순위 (0 이 가장 큼)

    def as_dict(self) -> dict:
        return {"t": self.t, "krw": self.krw, "rank": self.rank}


@torch.no_grad()
def contributions(model, norm: Norm, window, **kw) -> list:
    """창 하나의 epoch 별 기여도(원화). `window.ts` 와 같은 길이로 나온다."""
    model.eval()
    x, _y, mask = to_tensor([window], norm, **kw)
    g = model(x)[0]                       # (T,)
    n = int(mask[0].sum())
    prev = 0.0
    out = []
    for i in range(n):
        cur = float(g[i]) * norm.y_scale
        out.append(EpochContrib(t=float(window.ts[i]), krw=cur - prev))
        prev = cur
    order = sorted(range(len(out)), key=lambda i: -abs(out[i].krw))
    for r, i in enumerate(order):
        out[i].rank = r
    return out


def top_epochs(contribs, k: int = 20) -> list:
    """|기여도| 가 큰 순으로 k 개. 관문 C·F 와 **표본 추출기**가 쓴다."""
    return sorted(contribs, key=lambda c: -abs(c.krw))[:k]


def conservation(contribs, y_krw: float) -> dict:
    """★관문 E — Σ기여도 가 실제 Y 를 되찾는가."""
    s = sum(c.krw for c in contribs)
    denom = max(1.0, abs(float(y_krw)))
    return {"sum_krw": s, "y_krw": float(y_krw), "abs_err": abs(s - float(y_krw)),
            "rel_err": abs(s - float(y_krw)) / denom}


def rank_agreement(a, b, k: int = 20) -> float:
    """두 기여도 목록의 상위 k epoch 이 얼마나 겹치나 (0~1). 관문 C."""
    sa = {round(c.t, 3) for c in top_epochs(a, k)}
    sb = {round(c.t, 3) for c in top_epochs(b, k)}
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def spearman(a, b) -> float:
    """순위 상관 — 같은 시각들의 기여도 순서가 얼마나 닮았나. 관문 C."""
    from .train import pearson
    ta = {round(c.t, 3): c.krw for c in a}
    tb = {round(c.t, 3): c.krw for c in b}
    keys = sorted(set(ta) & set(tb))
    if len(keys) < 3:
        return 0.0

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0] * len(vals)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    va, vb = [ta[k] for k in keys], [tb[k] for k in keys]
    return pearson(ranks(va), ranks(vb))
