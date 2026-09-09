"""Block joint actions per synchronization interval; physical-time GAE."""
from __future__ import annotations

from dataclasses import dataclass
import math

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
    if (not intervals or not math.isfinite(time_unit_s) or time_unit_s <= 0
            or not (0 < gamma <= 1 and 0 <= lam <= 1)):
        raise ValueError("Invalid rollout or discount configuration")
    nxt = np.asarray(bootstrap, dtype=np.float64)
    if nxt.ndim != 1 or not len(nxt) or not np.isfinite(nxt).all():
        raise ValueError("Bootstrap must be a finite vector with one value per block")
    previous = None
    for row in intervals:
        if (not math.isfinite(row.start_s) or not math.isfinite(row.end_s)
                or row.start_s < 0 or row.end_s <= row.start_s):
            raise ValueError("Synchronization intervals must advance finite physical time")
        if previous is not None and not previous.terminated and not math.isclose(
                previous.end_s, row.start_s, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Nonterminal collection intervals must be contiguous")
        if (np.shape(row.values) != nxt.shape or len(row.choices) != len(nxt)
                or not np.isfinite(row.values).all() or not math.isfinite(row.reward)):
            raise ValueError("Rollout must have finite rewards/values and matching block dimensions")
        previous = row
    adv = np.zeros((len(intervals), len(nxt)), dtype=np.float64)
    trace = np.zeros_like(nxt)
    for i in range(len(intervals) - 1, -1, -1):
        row = intervals[i]
        dt = (row.end_s - row.start_s) / time_unit_s
        continuation = 0.0 if row.terminated else 1.0
        discount = gamma ** dt * continuation
        delta = row.reward + discount * nxt - row.values
        trace = delta + discount * lam ** dt * trace
        adv[i] = trace
        nxt = row.values
    values = np.stack([r.values for r in intervals])
    return adv, adv + values
