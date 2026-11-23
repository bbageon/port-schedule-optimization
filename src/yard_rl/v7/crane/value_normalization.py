"""Output-preserving global value-target normalization (YR-332-ay).

PopArt head transformation: van Hasselt et al., arXiv:1602.07714, Prop. 1.
Statistics are cumulative over new selected training returns, once per update.
Adam moments are retained unchanged: function preservation is NOT optimizer
trajectory invariance. Actor advantages remain in the original reward units.
"""
from __future__ import annotations

import torch
from torch import nn

from .actor_critic import SchedulingActorCritic, pack_states
from .on_policy import action_statistics, ppo_update


class NormalizedValueNet(SchedulingActorCritic):
    def __init__(self, **config):
        super().__init__(**config)
        for name, value in [('target_mean', 0.), ('target_scale', 1.),
                            ('target_count', 0.), ('target_m2', 0.)]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))

    def normalized_forward(self, batch):
        return super().forward(batch)

    def forward(self, batch):
        logits, normalized = self.normalized_forward(batch)
        return logits, normalized * self.target_scale.float() + self.target_mean.float()

    @torch.no_grad()
    def update_statistics(self, targets):
        y = torch.as_tensor(targets, dtype=torch.float64).flatten()
        if not len(y) or not torch.isfinite(y).all():
            raise ValueError('Finite nonempty new training targets required')
        old_mean, old_scale = self.target_mean.item(), self.target_scale.item()
        n, count = len(y), self.target_count.item()
        batch_mean = y.mean().item()
        delta = batch_mean - old_mean
        total = count + n
        mean = old_mean + delta * n / total
        m2 = self.target_m2.item() + (y - batch_mean).square().sum().item()
        m2 += delta * delta * count * n / total
        unclamped_scale = (max(0., m2 / total)) ** .5
        scale = max(1e-4, unclamped_scale)
        head = self.critic[-1]
        head.weight.mul_(old_scale / scale)
        head.bias.mul_(old_scale).add_(old_mean - mean).div_(scale)
        self.target_mean.fill_(mean)
        self.target_scale.fill_(scale)
        self.target_count.fill_(total)
        self.target_m2.fill_(m2)
        return dict(count=int(total), mean=mean, scale=scale,
                    scale_floor_hit=unclamped_scale < 1e-4,
                    old_mean=old_mean, old_scale=old_scale,
                    adam_moments='retained unchanged; no optimizer invariance claim')


def normalized_ppo_update(net, optimizer, samples, *, seed, max_samples=2048,
                          epochs=4, batch_size=64):
    """Same PPO estimator, budget and coefficients; only the critic is normalized."""
    rng = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(samples), generator=rng)[:max_samples].tolist()
    corpus = [samples[i] for i in selected]
    if not corpus:
        raise ValueError('No decisions with alternative actions collected')
    target = torch.tensor([s.target for s in corpus])
    old = torch.tensor([s.old_log_probability for s in corpus])
    advantage = target - torch.tensor([s.old_value for s in corpus])
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    # Cache every selected prediction BEFORE the statistical head transformation.
    with torch.no_grad():
        prior = [net(pack_states([s.state for s in corpus[b:b+batch_size]]))
                 for b in range(0, len(corpus), batch_size)]
    normalization = net.update_statistics(target)
    value_error = logit_error = replay_error = 0.
    with torch.no_grad():
        for j, begin in enumerate(range(0, len(corpus), batch_size)):
            batch = corpus[begin:begin+batch_size]
            ll, value = net(pack_states([s.state for s in batch]))
            valid = torch.isfinite(ll)
            logit_error = max(logit_error, float((ll[valid] - prior[j][0][valid]).abs().max()))
            value_error = max(value_error, float((value - prior[j][1]).abs().max()))
            logp, _ = action_statistics(ll, batch)
            replay_error = max(replay_error, float((logp - old[begin:begin+len(batch)]).abs().max()))
    if replay_error > 2e-5 or value_error > 2e-5 or logit_error != 0.:
        raise ValueError(('Normalization changed behavior/prediction', replay_error, value_error, logit_error))
    normalization.update(value_preservation_max_error=value_error, logit_preservation_max_error=logit_error)
    normalized_target = (target - net.target_mean.float()) / net.target_scale.float()
    net.train()
    statistics = []
    for _ in range(epochs):
        order = torch.randperm(len(corpus), generator=rng)
        for begin in range(0, len(corpus), batch_size):
            ix = order[begin:begin+batch_size]
            batch = [corpus[i] for i in ix]
            ll, value = net.normalized_forward(pack_states([s.state for s in batch]))
            logp, entropy = action_statistics(ll, batch)
            ratio = (logp - old[ix]).exp()
            actor = -torch.minimum(ratio * advantage[ix], ratio.clamp(.8, 1.2) * advantage[ix]).mean()
            critic = .5 * (value - normalized_target[ix]).square().mean()
            loss = actor + .5 * critic - .01 * entropy.mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite PPO gradient')
            optimizer.step()
            statistics.append((float(loss.detach()), float(critic.detach()), float(entropy.mean().detach()),
                               float(((ratio-1)-ratio.log()).mean().detach()),
                               float(((ratio-1).abs()>.2).float().mean().detach())))
    net.eval()
    return dict(available=len(samples), selected=len(corpus), optimizer_steps=len(statistics), epochs=epochs,
                replay_logp_max_error=replay_error, normalization=normalization,
                means=dict(zip(('loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction'),
                               map(lambda xs: sum(xs)/len(xs), zip(*statistics)))))


@torch.no_grad()
def restart_value_head(net, *, seed):
    """Forget predictions made in another reward unit. Action logits are untouched."""
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    for layer in net.critic:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
    torch.set_rng_state(state)
    for name, value in [('target_mean', 0.), ('target_scale', 1.), ('target_count', 0.), ('target_m2', 0.)]:
        getattr(net, name).fill_(value)


def fit_value_head(net, samples, *, seed, epochs=30, batch_size=256, lr=1e-3):
    """Regress only the value head on observed returns, with the shared encoder held fixed.

    The policy that produced `samples` is therefore still the policy after the fit.
    """
    if not samples:
        raise ValueError('No decisions to fit the value head on')
    target = torch.tensor([s.target for s in samples])
    with torch.no_grad():
        context = torch.cat([net.actor_and_critic_input(pack_states([s.state for s in samples[b:b+64]]))[1]
                             for b in range(0, len(samples), 64)])
    normalization = net.update_statistics(target)
    y = (target - net.target_mean.float()) / net.target_scale.float()

    def rmse():
        with torch.no_grad():
            return float((net.critic(context).squeeze(-1) - y).square().mean().sqrt())

    before, steps = rmse(), 0
    optimizer = torch.optim.Adam(net.critic.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(samples), generator=rng)
        for begin in range(0, len(order), batch_size):
            ix = order[begin:begin+batch_size]
            loss = .5 * (net.critic(context[ix]).squeeze(-1) - y[ix]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite value loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
    return dict(samples=len(samples), epochs=epochs, optimizer_steps=steps, learning_rate=lr,
                normalized_rmse_before=before, normalized_rmse_after=rmse(), normalization=normalization)


def fit(net, optimizer, samples, *, arm, **kwargs):
    if arm == 'control':
        assert net.target_count.item() == 0 and net.target_scale.item() == 1
        return ppo_update(net, optimizer, samples, **kwargs)
    if arm != 'popart':
        raise ValueError(arm)
    return normalized_ppo_update(net, optimizer, samples, **kwargs)
