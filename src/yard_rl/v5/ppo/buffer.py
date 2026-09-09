"""Block joint actions per synchronization interval; physical-time GAE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Choice:
    role: str
    time_s: float
    rows: torch.Tensor
    mask: torch.Tensor
    action: int
    log_prob: float


@dataclass
class Interval:
    start_s: float
    end_s: float
    states: torch.Tensor
    values: np.ndarray
    choices: list[list[Choice]]
    reward: float  # Team reward, NOT the sum of 21 copies.
    terminated: bool = False


def gae(intervals, bootstrap, *, gamma: float, lam: float, time_unit_s: float):
    """gamma/lam are per time unit, not per seller/buyer/crane callback."""
    if not intervals or time_unit_s <= 0 or not (0 < gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("Invalid rollout or discount configuration")
    adv = np.zeros((len(intervals), len(bootstrap)), dtype=np.float64)
    nxt = np.asarray(bootstrap, dtype=np.float64)
    trace = np.zeros_like(nxt)
    for i in range(len(intervals) - 1, -1, -1):
        row = intervals[i]
        if row.end_s <= row.start_s:
            raise ValueError("Synchronization intervals must advance physical time")
        dt = (row.end_s - row.start_s) / time_unit_s
        continuation = 0.0 if row.terminated else 1.0
        discount = gamma ** dt * continuation
        delta = row.reward + discount * nxt - row.values
        trace = delta + discount * lam ** dt * trace
        adv[i] = trace
        nxt = row.values
    values = np.stack([r.values for r in intervals])
    return adv, adv + values
