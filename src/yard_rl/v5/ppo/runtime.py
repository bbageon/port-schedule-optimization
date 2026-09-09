"""Fixed-policy collection, shared team rewards, and in-place on-policy updates."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from ..features.block import block_features
from ..reward.phi import terminal_cost_krw
from ..stage.episode import rehandles_of, yc_empty_travel_s
from ..stage.month import month_vessel_idle
from .buffer import Choice, Interval
from .crane import CraneActor
from .model import BlockPolicy, encode
from .update import update


class DebugStop(Exception):
    """Requested short-run limit at a synchronized boundary, not an engine failure."""


@dataclass(frozen=True)
class PPOConfig:
    rollout_intervals: int = 60
    epochs: int = 2
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    time_unit_s: float = 60.0
    reward_scale_krw: float = 1_000_000.0
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    def __post_init__(self):
        counts = (self.rollout_intervals, self.epochs, self.minibatch_size)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in counts):
            raise ValueError("Batch sizes and epochs must be positive integers")
        if any(not math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("PPO configuration must be finite")
        if min(self.rollout_intervals, self.epochs, self.minibatch_size) < 1:
            raise ValueError("Batch sizes and epochs must be positive")
        if min(self.learning_rate, self.time_unit_s, self.reward_scale_krw,
               self.max_grad_norm, self.target_kl) <= 0:
            raise ValueError("PPO scales must be positive")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.clip < 1):
            raise ValueError("Invalid discount / clip")
        if min(self.value_coef, self.entropy_coef) < 0:
            raise ValueError("Loss coefficients cannot be negative")


class PPORuntime:
    def __init__(self, policy: BlockPolicy, *, config=None, seed=302,
                 training=True, stop_s=None, on_update=None):
        if stop_s is not None and (not math.isfinite(stop_s) or stop_s <= 0):
            raise ValueError("stop_s must be finite and positive")
        self.policy, self.config = policy, config or PPOConfig()
        self.training, self.stop_s, self.on_update = bool(training), stop_s, on_update
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        self.rng = np.random.default_rng(seed)
        self.action_rng = torch.Generator(device="cpu").manual_seed(seed)
        self.buffer, self.updates = [], []
        self.role_counts, self.crane_actions = Counter(), Counter()
        self.time_s, self.initial_cost, self.cost_krw = None, None, 0.0
        self.total_reward, self.intervals = 0.0, 0
        self.execute = CraneActor(self)
        self.truncated = False
        self.bound = False

    def bind(self, mbt, bridge, meta, archive):
        if self.bound:
            raise RuntimeError("A runtime belongs to exactly one continuous world")
        if bridge.on_decision is not None:
            raise ValueError("Counterfactual teacher must be disabled")
        self.mbt, self.bridge, self.meta, self.archive = mbt, bridge, meta, archive
        self.bids = sorted(mbt.blocks)
        self.index = {b: i for i, b in enumerate(self.bids)}
        self.block_of = {id(mbt.blocks[b]): b for b in self.bids}
        self.bound = True

    def block_state(self, bid, t):
        return block_features(self.mbt, bid, t, n_cands=None,
                              records=self.bridge.records, orders=self.bridge.orders,
                              end_s=self.bridge.end_s)

    def states_at(self, t):
        return encode([self.block_state(b, t) for b in self.bids], "state")

    def read_cost(self, t):
        phi = terminal_cost_krw(self.bridge.records, end_s=t,
                               vessel_idle=month_vessel_idle(self.mbt, self.meta, self.archive),
                               yc_extra_move_s=yc_empty_travel_s(self.mbt),
                               rehandles=rehandles_of(self.mbt))
        self.cost_breakdown = phi.as_dict()
        return float(phi.total)

    def select(self, role, bid, t, rows, mask=None):
        if not math.isfinite(t) or t < 0:
            raise ValueError("Decision time must be finite and nonnegative")
        if self.time_s is None:
            raise RuntimeError("A synchronized initial boundary must precede decisions")
        if t < self.time_s - 1e-6:
            raise RuntimeError("Decision timestamp precedes the collection interval")
        x = encode(rows, role)
        mask = torch.ones(len(x), dtype=torch.bool) if mask is None else mask.clone().bool()
        with torch.no_grad():
            dist = self.policy.distribution(x, mask)
            action = (int(torch.multinomial(dist.probs, 1, generator=self.action_rng))
                      if self.training else int(dist.probs.argmax()))
            logp = float(dist.log_prob(torch.tensor(action)))
        if self.training:
            self.pending[self.index[bid]].append(Choice(role, t, x, mask, action, logp))
        self.role_counts[role] += 1
        return action

    def _update(self, bootstrap):
        if not self.training or not self.buffer:
            return
        rep = update(self.policy, self.optimizer, self.buffer, bootstrap, self.config, self.rng)
        rep.update(index=len(self.updates) + 1, time_s=self.time_s, cost_krw=self.cost_krw)
        self.updates.append(rep)
        self.buffer.clear()
        if self.on_update is not None:
            self.on_update(rep)

    def boundary(self, t, *, terminated=False, final=False):
        if not math.isfinite(t) or t < 0:
            raise ValueError("Review time must be finite and nonnegative")
        if self.time_s is not None and t < self.time_s - 1e-6:
            raise RuntimeError("Review clock went backwards")
        states = self.states_at(t)
        with torch.no_grad():
            bootstrap = self.policy.value(states).numpy().copy()
        cost = self.read_cost(t)
        if not math.isfinite(cost):
            raise FloatingPointError("Non-finite environment cost")
        if self.initial_cost is None:
            self.initial_cost = cost
        elif t > self.time_s + 1e-6:
            delta = cost - self.cost_krw
            if delta < -1e-5:
                raise RuntimeError("Cumulative cost fell: lost/pruned accounting data")
            reward = -delta / self.config.reward_scale_krw
            self.total_reward += reward
            self.intervals += 1
            if self.training:
                self.buffer.append(Interval(self.time_s, t, self.states, self.values,
                                             self.pending, reward, terminated))
        elif not final:
            return  # Repeated reviews must not erase decisions or charge cost twice.
        self.time_s, self.cost_krw = float(t), cost
        should_stop = self.stop_s is not None and t >= self.stop_s - 1e-6
        if len(self.buffer) >= self.config.rollout_intervals or final or should_stop:
            self._update(np.zeros_like(bootstrap) if terminated else bootstrap)
        # Values MUST be recollected after an update, for the next on-policy batch.
        self.states = states
        with torch.no_grad():
            self.values = self.policy.value(states).numpy().copy()
        self.pending = [[] for _ in self.bids]
        if should_stop and not final:
            self.truncated = True
            raise DebugStop(f"Debug time limit {t:g}s; unfinished work retained")

    def finish(self, t, *, terminated=False):
        self.boundary(t, terminated=terminated, final=True)
        self.truncated = not terminated

    def report(self):
        return {"generation": "v5", "algorithm": "shared-block-PPO",
                "time_s": self.time_s, "blocks": len(self.bids),
                "intervals": self.intervals, "roles": dict(self.role_counts),
                "crane_actions": dict(self.crane_actions), "cost_krw": self.cost_krw,
                "initial_cost_krw": self.initial_cost, "team_reward": self.total_reward,
                "truncated": self.truncated, "updates": self.updates,
                "traded_edges": self.bridge.traded_edges, "txn_failed": self.bridge.txn_failed,
                "n_space": self.bridge.n_space, "n_time": self.bridge.n_time,
                "cost_breakdown": self.cost_breakdown,
                "config": asdict(self.config)}
