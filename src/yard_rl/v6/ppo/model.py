"""One shared trunk, an action-score head and a state-value head (CPU prototype)."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

RAW_DIM = 32
ROLES = ("seller", "buyer", "crane", "state")
INPUT_DIM = RAW_DIM + len(ROLES) + 1


def encode(rows, role: str) -> torch.Tensor:
    x = torch.as_tensor(rows, dtype=torch.float32)
    if x.ndim != 2 or not len(x) or x.shape[1] > RAW_DIM:
        raise ValueError("Expected a nonempty action/state matrix of at most 32 features")
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite observations")
    out = torch.zeros((len(x), INPUT_DIM), dtype=torch.float32)
    out[:, :x.shape[1]] = x
    out[:, RAW_DIM + ROLES.index(role)] = 1
    # BUY/REJECT must stay distinguishable even when every offer feature is zero.
    if role == "buyer":
        if len(x) != 2:
            raise ValueError("Buyer expects BUY, REJECT rows")
        out[0, -1] = 1
    return out


class BlockPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.trunk = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)

    def distribution(self, rows: torch.Tensor, mask=None) -> Categorical:
        logits = self.actor(self.trunk(rows)).squeeze(-1)
        mask = (torch.ones_like(logits, dtype=torch.bool) if mask is None
                else torch.as_tensor(mask, dtype=torch.bool))
        if mask.shape != logits.shape or not mask.any():
            raise ValueError("Action mask must match candidates and allow at least one")
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(states)).squeeze(-1)
