"""Small ReSched-inspired two-branch encoder; online v7 state, not FJSP reproduction.

Independently implemented from the published architecture: operation self-attention,
edge-conditioned machine attention with a machine self connection, pair decoder.
No RoPE: the existing ten features do not contain within-job precedence positions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class GraphState:
    features: torch.Tensor
    operations: torch.Tensor
    cranes: torch.Tensor
    time_fraction: float
    global_waiting: float | None = None
    vessel_work: float | None = None    # critic-only: log(1 + hours of quay work left), see YR-332-bo


def pack_states(states):
    """Pad complete candidate sets, never truncate them or mix episodes."""
    b, n = len(states), max(len(s.features) for s in states)
    x = torch.zeros(b, n, states[0].features.shape[1])
    op = torch.zeros(b, n, dtype=torch.long)
    cr = torch.zeros_like(op)
    valid = torch.zeros(b, n, dtype=torch.bool)
    for i, s in enumerate(states):
        k = len(s.features)
        x[i, :k], op[i, :k], cr[i, :k], valid[i, :k] = s.features, s.operations, s.cranes, True
    result = (x, op, cr, valid, torch.tensor([s.time_fraction for s in states]).unsqueeze(1))
    present = [s.global_waiting is not None for s in states]
    if any(present):
        if not all(present):
            raise ValueError('Cannot mix legacy states and global-context states')
        context = torch.tensor([s.global_waiting for s in states]).unsqueeze(1)
        if not torch.isfinite(context).all() or (context < 0).any():
            raise ValueError('Invalid global waiting context')
        result += (context,)
    work = [s.vessel_work is not None for s in states]
    if any(work):
        if not all(work):
            raise ValueError('Cannot mix states with and without quay work')
        if not all(present):
            raise ValueError('Quay work is only defined beside the global context column')
        column = torch.tensor([s.vessel_work for s in states]).unsqueeze(1)
        if not torch.isfinite(column).all() or (column < 0).any():
            raise ValueError('Invalid quay work input')
        result += (column,)
    return result


def node_mean(edges, ids, valid):
    size = int(ids.max()) + 1
    index = ids.unsqueeze(-1).expand_as(edges)
    sums = edges.new_zeros(edges.shape[0], size, edges.shape[-1])
    sums.scatter_add_(1, index, edges * valid.unsqueeze(-1))
    counts = edges.new_zeros(edges.shape[0], size, 1)
    counts.scatter_add_(1, ids.unsqueeze(-1), valid.unsqueeze(-1).to(edges.dtype))
    return sums / counts.clamp_min(1), counts.squeeze(-1) > 0


def gather(nodes, ids):
    return nodes.gather(1, ids.unsqueeze(-1).expand(-1, -1, nodes.shape[-1]))


def masked_mean(x, valid):
    return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


class TwoBranchBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.edge = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(),
                                               nn.Linear(2 * width, width)) for _ in range(2)])

    def forward(self, operations, machines, edges, oi, ci, valid, ov, cv):
        # Sparse pair attention avoids a dense machine-by-operation compatibility matrix.
        b, p, d = edges.shape
        h, k, c = self.heads, self.head_width, machines.shape[1]
        q, key, value = self.q(machines), self.k(operations), self.v(operations)
        e = self.edge(edges)
        scores = ((gather(q, ci) + e).reshape(b, p, h, k) *
                  (gather(key, oi) + e).reshape(b, p, h, k)).sum(-1) / math.sqrt(k)
        messages = (gather(value, oi) + e).reshape(b, p, h, k)
        # Machine self edge participates in the SAME softmax denominator as operation edges.
        self_scores = (q.reshape(b, c, h, k) * self.k(machines).reshape(b, c, h, k)).sum(-1) / math.sqrt(k)
        scores = torch.cat((scores.masked_fill(~valid.unsqueeze(-1), -torch.inf), self_scores), dim=1)
        messages = torch.cat((messages, self.v(machines).reshape(b, c, h, k)), dim=1)
        ids = torch.cat((ci, torch.arange(c).expand(b, c)), dim=1)
        ix = ids.unsqueeze(-1).expand(-1, -1, h)
        maximum = edges.new_full((b, c, h), -torch.inf)
        maximum.scatter_reduce_(1, ix, scores, reduce='amax', include_self=True)
        weight = torch.exp(scores - maximum.gather(1, ix))
        denom = edges.new_zeros(b, c, h).scatter_add_(1, ix, weight)
        weight = weight / denom.gather(1, ix).clamp_min(1e-12)
        mixed = edges.new_zeros(b, c, h, k)
        mixed.scatter_add_(1, ids[:, :, None, None].expand(-1, -1, h, k),
                           weight.unsqueeze(-1) * messages)
        mac = self.norms[2](machines + self.out(mixed.reshape(b, c, d)))
        mac = self.norms[3](mac + self.ff[1](mac)) * cv.unsqueeze(-1)
        att, _ = self.self_attention(operations, operations, operations,
                                     key_padding_mask=~ov, need_weights=False)
        ops = self.norms[0](operations + att)
        ops = self.norms[1](ops + self.ff[0](ops)) * ov.unsqueeze(-1)
        return ops, mac


class SchedulingActorCritic(nn.Module):
    def __init__(self, kind='transformer', width=32, heads=4, layers=2, global_value_context=False,
                 vessel_work_context=False, planned_times=None):
        super().__init__()
        if kind not in ('mlp', 'transformer'):
            raise ValueError(kind)
        self.config = dict(kind=kind, width=width, heads=heads, layers=layers)
        self.planned_times = planned_times
        feature_count = 10
        if planned_times:
            from .planned_times import ARMS
            if kind != 'transformer':
                raise ValueError('Planned time experiment requires the registered Transformer')
            feature_count += 2 * len(ARMS[planned_times])
            self.config['planned_times'] = planned_times
        self.global_value_context = bool(global_value_context)
        if self.global_value_context:
            self.config['global_value_context'] = True
        self.vessel_work_context = bool(vessel_work_context)
        if self.vessel_work_context:
            if not self.global_value_context:
                raise ValueError('Quay work input requires the global context input')
            self.config['vessel_work_context'] = True
        extra = int(self.global_value_context) + int(self.vessel_work_context)
        self.kind = kind
        if kind == 'mlp':
            self.actor = nn.Sequential(nn.Linear(10, 64), nn.Tanh(), nn.Linear(64, 64),
                                       nn.Tanh(), nn.Linear(64, 1))
            self.critic = nn.Sequential(nn.Linear(11 + extra, 64), nn.Tanh(), nn.Linear(64, 1))
        else:
            self.embedding = nn.Linear(feature_count, width)
            self.blocks = nn.ModuleList([TwoBranchBlock(width, heads) for _ in range(layers)])
            self.actor = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(), nn.Linear(width, 1))
            self.critic = nn.Sequential(nn.Linear(2 * width + 1 + extra, 64), nn.Tanh(), nn.Linear(64, 1))

    def actor_and_critic_input(self, batch):
        expected = 5 + int(self.global_value_context) + int(self.vessel_work_context)
        if len(batch) != expected:
            raise ValueError('Global context or quay work missing or supplied to an incompatible model')
        x, oi, ci, valid, clock = batch[:5]
        if self.kind == 'mlp':
            logits = self.actor(x).squeeze(-1)
            context = torch.cat((masked_mean(x, valid), clock), dim=-1)
        else:
            edge = self.embedding(x)
            ops, ov = node_mean(edge, oi, valid)
            mac, cv = node_mean(edge, ci, valid)
            for block in self.blocks:
                ops, mac = block(ops, mac, edge, oi, ci, valid, ov, cv)
            logits = self.actor(torch.cat((gather(ops, oi), gather(mac, ci), edge), -1)).squeeze(-1)
            context = torch.cat((masked_mean(ops, ov), masked_mean(mac, cv), clock), -1)
        if self.global_value_context:
            context = torch.cat((context, batch[5]), -1)
        if self.vessel_work_context:
            context = torch.cat((context, batch[6]), -1)
        return logits.masked_fill(~valid, -torch.inf), context

    def forward(self, batch):
        logits, context = self.actor_and_critic_input(batch)
        return logits, self.critic(context).squeeze(-1)

    def save(self, path, **metadata):
        torch.save(dict(model_config=self.config, state_dict=self.state_dict(), **metadata), path)

    @classmethod
    def load(cls, path):
        obj = torch.load(path, map_location='cpu', weights_only=False)
        net = cls(**obj['model_config'])
        net.load_state_dict(obj['state_dict'])
        return net
