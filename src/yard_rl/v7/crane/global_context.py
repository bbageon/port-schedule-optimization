"""One critic-only feature from a synchronized, immutable terminal observation."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QueueSnapshot:
    time: float
    blocks: tuple[tuple[str, int], ...]
    total: int

    def __post_init__(self):
        if not math.isfinite(self.time) or self.time < 0 or not self.blocks:
            raise ValueError('Invalid snapshot time or missing blocks')
        if len({b for b, _ in self.blocks}) != len(self.blocks):
            raise ValueError('Duplicate snapshot block')
        if any(type(n) is not int or n < 0 for _, n in self.blocks):
            raise ValueError('Invalid block waiting count')
        if type(self.total) is not int or self.total != sum(n for _, n in self.blocks):
            raise ValueError('Global waiting sum mismatch')

    @property
    def feature(self):
        return math.log1p(self.total) / math.log1p(21 * 10)

    def observation(self, decision_time, block_id):
        if not math.isfinite(decision_time) or self.time > decision_time + 1e-6:
            raise ValueError('Global observation contains future information')
        if block_id not in dict(self.blocks):
            raise ValueError('Decision block missing from global observation')
        return dict(snapshot_time=self.time, blocks=dict(self.blocks), total=self.total,
                    feature=self.feature, decision_time=decision_time, decision_block=block_id,
                    age_s=max(0., decision_time - self.time))


def snapshot_terminal(terminal, t):
    terminal_blocks = getattr(terminal, '_terminal', set())
    rows = []
    for bid, sim in sorted(terminal.blocks.items()):
        if not math.isfinite(sim.clock) or sim.clock > t + 1e-6:
            raise ValueError('Global snapshot includes a future block clock')
        if bid not in terminal_blocks and abs(sim.clock - t) > 1e-6:
            raise ValueError('Global snapshot requires synchronized active block clocks')
        rows.append((bid, sim.kpis.waiting_count()))
    return QueueSnapshot(float(t), tuple(rows), sum(n for _, n in rows))


def validate_observation(obj):
    snapshot = QueueSnapshot(float(obj['snapshot_time']), tuple(sorted(obj['blocks'].items())), obj['total'])
    expected = snapshot.observation(obj['decision_time'], obj['decision_block'])
    if expected != obj:
        raise ValueError('Stored global observation is inconsistent')
    return snapshot


def expand_value_net(net):
    """Copy every old weight; the single new input column starts at zero."""
    import torch
    from .value_normalization import NormalizedValueNet
    if net.global_value_context:
        raise ValueError('Model already has global context')
    result = NormalizedValueNet(**net.config, global_value_context=True)
    state = {k: v.clone() for k, v in net.state_dict().items()}
    w = state['critic.0.weight']
    state['critic.0.weight'] = torch.cat((w, w.new_zeros(w.shape[0], 1)), dim=1)
    result.load_state_dict(state)
    return result.eval()
