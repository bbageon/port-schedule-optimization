"""RUDDER 모델 학습 — held-out 은 **날 단위로** 가른다 ([[YR-223]] 2단계).

■ ★왜 창 단위로 가르면 안 되나
  한 날에서 뽑은 창들은 서로 **겹친다**(3시간 창을 63분 간격으로 뽑으므로 한 토큰이
  평균 3개 창에 들어간다). 창 단위로 무작위 분할하면 검증 창의 토큰 대부분이 학습
  창에도 있어, held-out 이 held-out 이 아니게 된다 — 과적합을 **못 잡는다.**

  → 날(시드) 단위로 가른다. 검증 날의 토큰은 학습에 한 번도 안 들어간다.

■ 학습 손실이 아니라 **검증 손실**을 본다
  [[YR-222]] 가 지금 이 질문을 놓고 도는 중이다: 학습 손실이 내려가도 실제로
  배운 게 아닐 수 있다. 같은 실수를 여기서 반복하지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .model import Norm, RudderNet, rudder_loss, to_tensor


@dataclass
class FitReport:
    n_train: int = 0
    n_val: int = 0
    n_params: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    #: ★관문 A — 검증 창에서 예측 g_T 와 실제 Y 의 상관
    val_corr: float = 0.0
    val_mae_krw: float = 0.0
    y_scale: float = 1.0
    epochs: int = 0
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "history"}


def split_by_seed(windows, val_frac: float = 0.2):
    """★날 단위 분할. 시드를 정렬해 뒤쪽 `val_frac` 을 검증으로 — 난수를 안 쓴다."""
    seeds = sorted({w.seed for w in windows})
    n_val = max(1, int(round(len(seeds) * val_frac))) if len(seeds) > 1 else 0
    val_seeds = set(seeds[len(seeds) - n_val:]) if n_val else set()
    tr = [w for w in windows if w.seed not in val_seeds]
    va = [w for w in windows if w.seed in val_seeds]
    return tr, va


def pearson(a, b) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def _last_pred(g, mask):
    last = mask.sum(dim=1).long() - 1
    return g[torch.arange(g.shape[0]), last]


@torch.no_grad()
def evaluate(model, norm: Norm, windows, **kw) -> dict:
    if not windows:
        return {"loss": 0.0, "corr": 0.0, "mae_krw": 0.0}
    model.eval()
    x, y, mask = to_tensor(windows, norm, **kw)
    g = model(x)
    loss, _ = rudder_loss(g, y, mask=mask)
    pred = _last_pred(g, mask)
    corr = pearson(pred.tolist(), y.tolist())
    mae = float((pred - y).abs().mean()) * norm.y_scale
    return {"loss": float(loss), "corr": corr, "mae_krw": mae}


def fit(windows, *, val_frac: float = 0.2, hidden: int | None = None,
        epochs: int = 400, lr: float = 3e-3, seed: int = 0,
        ablate_actions: bool = False, shuffle_order: bool = False,
        log_every: int = 0, on_log=None):
    """학습하고 **검증 지표**를 함께 돌려준다.

    `ablate_actions` / `shuffle_order` 는 관문 B·D 가 쓰는 재료 훼손 스위치다.
    학습·검증 **양쪽에 똑같이** 건다 — 한쪽만 걸면 비교가 성립하지 않는다.
    """
    from .model import HIDDEN
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed) + 1)
    tr, va = split_by_seed(windows, val_frac)
    norm = Norm.fit(tr or windows)              # ★눈금은 학습 집합에서만
    kw = {"ablate_actions": ablate_actions, "shuffle_order": shuffle_order,
          "gen": gen}
    x, y, mask = to_tensor(tr or windows, norm, **kw)

    model = RudderNet(hidden=(hidden or HIDDEN))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rep = FitReport(n_train=len(tr or windows), n_val=len(va),
                    n_params=model.n_params, y_scale=norm.y_scale, epochs=epochs)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        g = model(x)
        loss, parts = rudder_loss(g, y, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            ev = evaluate(model, norm, va, **kw) if va else {}
            row = {"ep": ep, "train": float(loss), **{f"val_{k}": v
                                                      for k, v in ev.items()}}
            rep.history.append(row)
            if on_log is not None:
                on_log(row)
    rep.train_loss = float(loss)
    ev = evaluate(model, norm, va, **kw) if va else {"loss": 0.0, "corr": 0.0,
                                                     "mae_krw": 0.0}
    rep.val_loss, rep.val_corr, rep.val_mae_krw = (ev["loss"], ev["corr"],
                                                   ev["mae_krw"])
    return model, norm, rep
