"""Probability-preserving legal joint actions and finite-horizon PPO returns."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time

import torch
from torch import nn

from .actor_critic import GraphState, pack_states
from .features import crane_features, decision_common
from ..world.contract.schema import CandidateKind
from ..world.integrated.resolver import CentralResolver
from ..reward.phi import terminal_cost_krw


@dataclass
class DecisionSample:
    state: GraphState
    masks: list[list[int]]
    choices: list[int]
    old_log_probability: float
    old_value: float
    past_cost: float
    target: float = 0.
    global_observation: dict | None = None


def action_statistics(logits, samples):
    """Likelihood of the ordered joint construction that was actually executed."""
    logps, entropies = [], []
    for row, sample in zip(logits, samples):
        lp, ent = row[0] * 0., row[0] * 0.
        for allowed, chosen in zip(sample.masks, sample.choices):
            dist = torch.distributions.Categorical(logits=row[allowed])
            lp = lp + dist.log_prob(torch.tensor(allowed.index(chosen)))
            ent = ent + dist.entropy()
        logps.append(lp)
        entropies.append(ent)
    return torch.stack(logps), torch.stack(entropies)


def tier(candidate):
    return (0 if candidate.mandatory else 1,
            2 if candidate.job_ref is None else 0 if candidate.kind == CandidateKind.SERVE else 1)


class OnPolicyResolver(CentralResolver):
    def __init__(self, net, *, stochastic=False, seed=0, collect=False, horizon=86400.,
                 global_context_mode='off', collect_until=None):
        super().__init__()
        self.net, self.stochastic, self.collect, self.horizon = net, stochastic, collect, horizon
        self.collect_until = horizon if collect_until is None else collect_until
        self.rng = torch.Generator().manual_seed(seed)
        self.samples, self.costs, self.latencies = [], [(0., 0.)], []
        self.decisions = self.nontrivial = self.max_candidates = self.oracle_calls = 0
        self.last_assignment = None
        if global_context_mode not in ('off', 'record', 'zero', 'global'):
            raise ValueError(global_context_mode)
        if bool(getattr(net, 'global_value_context', False)) != (global_context_mode in ('zero', 'global')):
            raise ValueError('Resolver/model global-context contracts differ')
        if (bool(getattr(net, 'vessel_work_context', False))
                and type(self).critic_vessel_work is OnPolicyResolver.critic_vessel_work):
            raise ValueError('Model expects quay work but this resolver provides none')
        self.global_context_mode = global_context_mode
        self.global_snapshots, self.global_decisions = [], []
        self.global_block_ids = {}
        self.public_schedule = None
        if getattr(net, 'planned_times', None):
            from .planned_times import PublicSchedule
            self.public_schedule = PublicSchedule(net.planned_times)

    def observe_global_epoch(self, terminal, t):
        if self.global_context_mode == 'off':
            return
        from .global_context import snapshot_terminal
        snapshot = snapshot_terminal(terminal, t)
        self.global_block_ids = {id(sim): bid for bid, sim in terminal.blocks.items()}
        if self.global_snapshots and snapshot.time < self.global_snapshots[-1].time:
            raise ValueError('Global observation clock regressed')
        self.global_snapshots.append(snapshot)

    def global_state_context(self, sim):
        if self.global_context_mode == 'off':
            return None, None
        if not self.global_snapshots:
            raise ValueError('Missing synchronized global observation')
        obs = self.global_snapshots[-1].observation(sim.clock, self.global_block_ids[id(sim)])
        value = (obs['feature'] if self.global_context_mode == 'global' else
                 0. if self.global_context_mode == 'zero' else None)
        self.global_decisions.append(obs)
        return value, obs

    def critic_vessel_work(self, sim):
        """Critic-only quay work input. None here; a resolver that tracks quay progress overrides it."""
        return None

    def resolve(self, sim, decision, gen_by_crane):
        started = time.perf_counter()
        pairs = [(c, gc) for c in decision.crane_ids for gc in gen_by_crane[c].items if gc.feasible]
        self.decisions += 1
        self.max_candidates = max(self.max_candidates, len(pairs))
        chosen, taken, masks, picks = {}, set(), [], []
        rejects, contested, keys = defaultdict(list), defaultdict(set), {}
        for c, gc in pairs:
            keys[c, gc.candidate_id] = tier(gc) + (c, gc.candidate_id)
            if gc.job_ref is not None and gc.job_ref.token is not None:
                contested[gc.job_ref.token].add(c)
        logits, value, state = None, 0., None
        global_observation = None
        lp = 0.
        while len(chosen) < len(decision.crane_ids):
            legal = []
            for i, (c, gc) in enumerate(pairs):
                if c in chosen:
                    continue
                if gc.job_ref is not None and gc.job_ref.token is not None and gc.job_ref.token in taken:
                    continue
                trial = {**chosen, c: gc}
                # Generator already establishes individual feasibility; joint oracle is
                # required once other cranes have been assigned.
                if chosen:
                    self.oracle_calls += 1
                    proj = sim.dry_run_commit({x: g.job_ref for x, g in trial.items()})
                    if set(proj.plans) != {x for x, g in trial.items() if g.job_ref is not None}:
                        continue
                legal.append(i)
            if not legal:
                raise RuntimeError('No legal completion of the joint action, including WAIT')
            level = min(tier(pairs[i][1]) for i in legal)
            legal = [i for i in legal if tier(pairs[i][1]) == level]
            if level[1] == 2:
                legal = legal[:1]  # Ordering two WAITs has no physical effect to learn.
            if len(legal) == 1:
                pick = legal[0]
            else:
                if logits is None:
                    common = decision_common(sim)
                    rows = [crane_features(sim, c, gc, common=common, feature_mode='feasible',
                                           include_flow=True) for c, gc in pairs]
                    if self.public_schedule is not None:
                        for row, times in zip(rows, self.public_schedule.candidates(sim, pairs)):
                            row.extend(times)
                    operations, cranes = {}, {c: j for j, c in enumerate(decision.crane_ids)}
                    oi = []
                    for c, gc in pairs:
                        key = (gc.job_ref.token if gc.job_ref and gc.job_ref.token is not None
                               else ('NON_JOB', c, gc.candidate_id))
                        oi.append(operations.setdefault(key, len(operations)))
                    global_value, global_observation = self.global_state_context(sim)
                    state = GraphState(torch.tensor(rows), torch.tensor(oi),
                                       torch.tensor([cranes[c] for c, _ in pairs]), sim.clock/self.horizon,
                                       global_value, self.critic_vessel_work(sim))
                    with torch.no_grad():
                        ll, vv = self.net(pack_states([state]))
                    logits, value = ll[0], float(vv[0])
                distribution = torch.distributions.Categorical(logits=logits[legal])
                j = int(torch.multinomial(distribution.probs, 1, generator=self.rng)) if self.stochastic else int(logits[legal].argmax())
                pick = legal[j]
                lp += float(distribution.log_prob(torch.tensor(j)))
            masks.append(legal)
            picks.append(pick)
            c, gc = pairs[pick]
            chosen[c] = gc
            if gc.job_ref is not None and gc.job_ref.token is not None:
                taken.add(gc.job_ref.token)
        # Validate the SAME action for which the probability was computed, without repair.
        projection = sim.dry_run_commit({c: g.job_ref for c, g in chosen.items()})
        self.oracle_calls += 1
        if set(projection.plans) != {c for c, g in chosen.items() if g.job_ref is not None}:
            raise RuntimeError('Selected action failed final physical feasibility')
        self.last_assignment = {c: g.candidate_id for c, g in chosen.items()}
        if state is not None:
            self.nontrivial += 1
            if self.collect and sim.clock < self.collect_until:
                if self.costs[-1][0] > sim.clock + 1e-6:
                    raise ValueError('Past-cost baseline contains future information')
                self.samples.append(DecisionSample(state, masks, picks, lp, value, self.costs[-1][1],
                                                   global_observation=global_observation))
        self.latencies.append(time.perf_counter()-started)
        return self._finalize(sim, decision, gen_by_crane, chosen, rejects, contested, keys)

    def observe_epoch(self, t, records, counters):
        if not self.collect:
            return
        vessel, move, rehandles = counters
        cost = terminal_cost_krw(records, end_s=t, vessel_idle=vessel,
                                 yc_extra_move_s=move, rehandles=rehandles).total
        self.observe_cost(t, cost)

    def observe_cost(self, t, cost):
        if cost < self.costs[-1][1] - 1e-6:
            raise ValueError('Cumulative synchronized objective decreased')
        self.costs.append((t, cost))

    def finish_turnaround(self, clock, scale):
        """Attach remaining weighted minutes after the last truck and port call finish."""
        if self.costs != [(0., 0.)]:
            raise ValueError("Turnaround targets require a separate clock from KRW costs")
        for sample in self.samples:
            t = sample.state.time_fraction * self.horizon
            sample.past_cost = clock.minutes_at(t)
            self.observe_cost(t, sample.past_cost)
        last = clock.times[-1] if clock.times else 0.0
        self.observe_cost(max(last, self.costs[-1][0]), clock.total_minutes)
        return self.finish(clock.total_minutes, scale)

    def finish(self, final_cost, scale):
        if not self.collect:
            return {}
        if scale <= 0 or not math.isfinite(scale):
            raise ValueError('A fixed positive training-only scale is required')
        if abs(self.costs[-1][1]-final_cost) > 1e-6:
            raise ValueError('Synchronized cost differs from reported terminal objective')
        reward = sum(-(b[1]-a[1])/scale for a, b in zip(self.costs, self.costs[1:]))
        if not math.isclose(reward, -final_cost/scale, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Reward increments do not telescope to terminal cost')
        for sample in self.samples:
            sample.target = -(final_cost-sample.past_cost)/scale
        return dict(samples=len(self.samples), decisions=self.decisions,
                    nontrivial=self.nontrivial, reward_sum=reward, final_cost=final_cost,
                    scale=scale, max_candidates=self.max_candidates, oracle_calls=self.oracle_calls)


def ppo_update(net, optimizer, samples, *, seed, max_samples=2048, epochs=4, batch_size=64):
    rng = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(samples), generator=rng)[:max_samples].tolist()
    corpus = [samples[i] for i in selected]
    target = torch.tensor([s.target for s in corpus])
    old = torch.tensor([s.old_log_probability for s in corpus])
    advantage = target - torch.tensor([s.old_value for s in corpus])
    advantage = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
    if not corpus:
        raise ValueError('No decisions with alternative actions collected')
    net.train()
    statistics = []
    replay_error = 0.
    # Check ALL sampled stored log-probabilities before the first update.
    with torch.no_grad():
        for begin in range(0, len(corpus), batch_size):
            batch = corpus[begin:begin+batch_size]
            ll, _ = net(pack_states([s.state for s in batch]))
            logp, _ = action_statistics(ll, batch)
            replay_error = max(replay_error, float((logp-old[begin:begin+len(batch)]).abs().max()))
    if replay_error > 2e-5:
        raise ValueError(f'Executed action probability mismatch: {replay_error}')
    for _ in range(epochs):
        order = torch.randperm(len(corpus), generator=rng)
        for begin in range(0, len(corpus), batch_size):
            ix = order[begin:begin+batch_size]
            batch = [corpus[i] for i in ix]
            ll, value = net(pack_states([s.state for s in batch]))
            logp, entropy = action_statistics(ll, batch)
            ratio = (logp-old[ix]).exp()
            clipped = ratio.clamp(.8, 1.2)
            actor = -torch.minimum(ratio*advantage[ix], clipped*advantage[ix]).mean()
            critic = .5*(value-target[ix]).square().mean()
            loss = actor + .5*critic - .01*entropy.mean()
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
    return dict(available=len(samples), selected=len(corpus), optimizer_steps=len(statistics),
                epochs=epochs, replay_logp_max_error=replay_error,
                means=dict(zip(('loss','critic_loss','entropy','approx_kl','clip_fraction'),
                               map(lambda xs: sum(xs)/len(xs), zip(*statistics)))))
