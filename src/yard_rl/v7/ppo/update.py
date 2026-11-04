"""Clipped PPO over each block's joint interval action; fixed on-policy batches."""
from __future__ import annotations

import numpy as np
import torch

from .buffer import gae


def clipped_surrogate(log_ratio, advantage, clip: float):
    ratio = log_ratio.exp()
    return torch.minimum(ratio * advantage,
                         ratio.clamp(1 - clip, 1 + clip) * advantage)


def update(policy, optimizer, intervals, bootstrap, config, rng):
    adv, returns = gae(intervals, bootstrap, gamma=config.gamma,
                       lam=config.gae_lambda, time_unit_s=config.time_unit_s)
    entries = [(i, b) for i, row in enumerate(intervals)
               for b in range(len(row.values))]
    active = [(i, b) for i, b in entries if intervals[i].choices[b]]
    # Only actor examples enter actor normalization; empty blocks still train values.
    if len(active) > 1:
        a = np.array([adv[i, b] for i, b in active])
        scale = a.std()
        if scale > 1e-8:
            adv = (adv - a.mean()) / scale
    losses, kls, gradients = [], [], []
    n_minibatches = 0
    early_stopped = False
    for _ in range(config.epochs):
        order = rng.permutation(len(entries))
        stop = False
        for offset in range(0, len(order), config.minibatch_size):
            indices = [entries[k] for k in order[offset:offset + config.minibatch_size]]
            states = torch.stack([intervals[i].states[b] for i, b in indices])
            targets = torch.tensor([returns[i, b] for i, b in indices], dtype=torch.float32)
            value_loss = (policy.value(states) - targets).square().mean()
            objectives, entropies, approx_kls = [], [], []
            for i, b in indices:
                choices = intervals[i].choices[b]
                if not choices:
                    continue
                new_logp, old_logp, entropy = [], 0.0, []
                for c in choices:
                    dist = policy.distribution(c.rows, c.mask)
                    new_logp.append(dist.log_prob(torch.tensor(c.action)))
                    old_logp += c.log_prob
                    entropy.append(dist.entropy())
                log_ratio = torch.stack(new_logp).sum() - old_logp
                objectives.append(clipped_surrogate(log_ratio, float(adv[i, b]), config.clip))
                entropies.append(torch.stack(entropy).sum())
                approx_kls.append((log_ratio.exp() - 1 - log_ratio).detach())
            actor_loss = -torch.stack(objectives).mean() if objectives else value_loss * 0
            entropy = torch.stack(entropies).mean() if entropies else value_loss * 0
            kl = float(torch.stack(approx_kls).mean()) if approx_kls else 0.0
            if not np.isfinite(kl):
                raise FloatingPointError("Non-finite PPO policy divergence")
            kls.append(kl)  # Include the batch that STOPPED training in the report.
            if kl > config.target_kl:
                early_stopped = True
                stop = True
                break
            loss = actor_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm,
                                                   error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(float(norm))
            n_minibatches += 1
        if stop:
            break
    return {"loss": float(np.mean(losses)) if losses else 0.0,
            "early_stopped": early_stopped,
            "max_kl": max(kls, default=0.0), "max_grad_norm_before_clip": max(gradients, default=0.0),
            "minibatches": n_minibatches, "intervals": len(intervals),
            "block_samples": len(entries), "active_block_samples": len(active),
            "micro_actions": sum(len(c) for r in intervals for c in r.choices)}
