"""Evaluation-only original-assignment control, without changing crane dispatch."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import math

import torch

from ..features.candidate import BUYER_OFFER_DIM
from .runtime import PPORuntime


class OriginalComparisonRuntime(PPORuntime):
    """Same frozen crane policy in both arms; only market permissions differ.

    Role-local streams stop seller/buyer draws from advancing the crane stream.
    Both arms use this coupling, including the common no-transfer warm-up.
    This class intentionally cannot train or select an alternative dispatcher.
    """

    def __init__(self, policy, *, reallocate, seed=302, intervention_s=86400.,
                 config=None, on_boundary=None, stop_s=None):
        if not math.isfinite(intervention_s) or intervention_s < 0:
            raise ValueError('Intervention time must be finite and nonnegative')
        super().__init__(policy, config=config, seed=seed, training=False,
                         sample_actions=True, on_boundary=on_boundary, stop_s=stop_s)
        self.reallocate, self.intervention_s = bool(reallocate), float(intervention_s)
        self.role_seeds = {role: int.from_bytes(hashlib.sha256(
            f'v6-original-evaluation-v1:{seed}:{role}'.encode()).digest()[:8], 'big') % (2**63)
            for role in ('seller', 'buyer', 'crane')}
        self.role_rng = {role: torch.Generator(device='cpu').manual_seed(value)
                         for role, value in self.role_seeds.items()}
        self.forced_market = Counter()

    def select(self, role, bid, t, rows, mask=None):
        if role not in self.role_rng:
            raise ValueError(f'Unknown decision role: {role}')
        if not math.isfinite(t) or t < 0:
            raise ValueError('Decision time must be finite and nonnegative')
        if self.time_s is None or t < self.time_s - 1e-6:
            raise RuntimeError('Decision requires a synchronized boundary')
        if role != 'crane' and (not self.reallocate or t < self.intervention_s):
            x = torch.as_tensor(rows)
            if role == 'seller':
                if x.ndim != 2 or x.shape[1] != 21 or len(x) == 0 or float(x[0, 12]) != 1.:
                    raise RuntimeError('KEEP encoding changed; refusing a false original control')
                choice = 0
            else:
                if x.ndim != 2 or len(x) != 2 or not torch.all(x[1, -BUYER_OFFER_DIM:] == 0):
                    raise RuntimeError('REJECT encoding changed')
                choice = 1
            if mask is not None and not bool(mask[choice]):
                raise RuntimeError('Original-assignment action is unavailable')
            self.role_counts[role] += 1
            self.forced_market[role] += 1
            return choice
        previous = self.action_rng
        self.action_rng = self.role_rng[role]
        try:
            return super().select(role, bid, t, rows, mask)
        finally:
            self.action_rng = previous


def assignment_audit(original_orders, realized_orders):
    """Compare every original public field, not just a zero transaction counter."""
    expected = {row['doc_key']: row for row in original_orders}
    if len(expected) != len(original_orders):
        raise ValueError('Duplicate original order identity')
    actual = {key: asdict(value) for key, value in realized_orders.items()}
    if expected.keys() != actual.keys():
        raise RuntimeError('Original order membership changed')
    counts, examples = Counter(), []
    for key, before in expected.items():
        if before.keys() != actual[key].keys():
            raise RuntimeError('Original order schema changed')
        changes = {field: [value, actual[key][field]] for field, value in before.items()
                   if value != actual[key][field]}
        counts.update(changes.keys())
        if changes and len(examples) < 8:
            examples.append(dict(doc_key=key, changes=changes))
    return dict(orders=len(expected), changed_fields=dict(counts),
                original_fields_preserved=not counts, examples=examples)
