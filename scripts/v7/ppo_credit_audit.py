"""Read immutable PPO trajectories and checkpoints; never step an optimizer or world."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from ppo_credit_metrics import (VERSIONS, first_choice, normalized_advantages, choice_metrics,
                                mask_description, between_group_share)
from yard_rl.v7.crane.actor_critic import GraphState, SchedulingActorCritic, pack_states
from yard_rl.v7.crane.on_policy import DecisionSample, action_statistics

SOURCE = Path('outputs/reports/yr332_resched_ppo')
OUT = Path('outputs/reports/yr332_ppo_credit')
PLAN = Path('docs/research/v7-discharge/ppo-credit-analysis-plan.md')
KINDS = ('mlp', 'transformer')
SEEDS = (9951101, 9951102, 9951103)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz')
          else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def decode(obj):
    d = dict(obj)
    state = GraphState(torch.tensor(d.pop('features')), torch.tensor(d.pop('operations')),
                       torch.tensor(d.pop('cranes')), d.pop('time_fraction'))
    return DecisionSample(state=state, **d)


def prepare():
    campaign = read(SOURCE/'campaign.json')
    inputs = dict(campaign['sources'])
    for kind in KINDS:
        for seed in SEEDS:
            root = SOURCE/f'{kind}-{seed}'
            for name in ('training.json', 'initial.pt', 'final.pt',
                         *[f'update-{i}.pt' for i in range(6)],
                         *[f'samples-{i}.json.gz' for i in range(6)]):
                p = root/name
                inputs[p.as_posix()] = sha(p)
    for name in ('campaign.json', 'summary.json', 'verification.json', 'current-gates.json'):
        p = SOURCE/name
        inputs[p.as_posix()] = sha(p)
    for p, digest in inputs.items():
        assert sha(p) == digest, p
    analyzers = {p.as_posix(): sha(p) for p in [PLAN, *Path('scripts/v7').glob('ppo_credit_*.py')]}
    manifest = dict(scope='post-hoc saved-experience diagnostic, no new simulation or fitting',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        input_hashes=inputs, analyzer_hashes=analyzers, torch_version=torch.__version__,
        configuration=dict(kinds=KINDS, seeds=SEEDS, updates=6, samples_per_update=2048,
                           time_bins_hours=[0, 6, 12, 18, 24], inference_batch=16,
                           return_scale=campaign['configuration']['return_scale'],
                           gradient_batches='first actual minibatch of each of 36 updates'))
    assert not (OUT/'manifest.json').exists(), 'Do not overwrite a previous diagnosis'
    save(OUT/'manifest.json', manifest)
    save(OUT/'authorization.json', dict(task='YR-332-ax', tool_allowed=False,
        tool_reason='선결 게이트 scenario_validity부터 해결해야 함 (요청: performance)',
        observed_exit_code=1, gate_file=(SOURCE/'current-gates.json').as_posix(),
        gate_sha256=sha(SOURCE/'current-gates.json'),
        gate_commit=manifest['source_commit'],
        authorization='2026-11-17 user explicitly requested these saved-experience diagnostics',
        limits='Read-only source analysis; no training, simulator execution, policy/reward change or gate promotion'))
    print(json.dumps(dict(prepared=True, input_files=len(inputs))))


@torch.no_grad()
def infer(net, samples, batch_size=16):
    logits, values, logps = [None]*len(samples), [None]*len(samples), [None]*len(samples)
    order = sorted(range(len(samples)), key=lambda i: len(samples[i].state.features))
    for begin in range(0, len(order), batch_size):
        indices = order[begin:begin+batch_size]
        batch = [samples[i] for i in indices]
        ll, vv = net(pack_states([s.state for s in batch]))
        lp, _ = action_statistics(ll, batch)
        for j, i in enumerate(indices):
            logits[i], values[i], logps[i] = ll[j, :len(samples[i].state.features)].clone(), float(vv[j]), float(lp[j])
    return logits, values, logps


def gradients(net, samples, seed):
    advantages, permutation, rng = normalized_advantages(samples, seed)
    positions = torch.randperm(len(samples), generator=rng)[:64]
    indices = permutation[positions].tolist()
    batch = [samples[i] for i in indices]
    ll, vv = net(pack_states([s.state for s in batch]))
    lp, entropy = action_statistics(ll, batch)
    old = torch.tensor([s.old_log_probability for s in batch])
    target = torch.tensor([s.target for s in batch])
    adv = torch.tensor([advantages[i] for i in indices])
    ratio = (lp-old).exp()
    losses = (-torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean(),
              .25*(vv-target).square().mean(), -.01*entropy.mean())
    named = list(net.named_parameters())
    vectors = []
    for loss in losses:
        gs = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        vectors.append([torch.zeros_like(p).flatten() if g is None else g.flatten() for (_, p), g in zip(named, gs)])
    def summarize(parts):
        ga, gc, ge = [torch.cat(v) for v in parts]
        norms = [float(g.norm()) for g in (ga, gc, ge, ga+gc+ge)]
        return dict(actor=norms[0], weighted_critic=norms[1], weighted_entropy=norms[2], total=norms[3],
                    actor_critic_cosine=float(torch.dot(ga, gc)/(ga.norm()*gc.norm()).clamp_min(1e-12)),
                    clip_factor=min(1., 1./(norms[3]+1e-6)))
    result = dict(all=summarize(vectors), source_sample_indices=indices,
                  logp_replay_error=float((lp-old).abs().max().detach()))
    shared = [i for i, (name, _) in enumerate(named) if name.startswith(('embedding.', 'blocks.'))]
    if shared:
        result['shared'] = summarize([[v[i] for i in shared] for v in vectors])
    return result


def run(kind, seed):
    torch.set_num_threads(1)
    started = time.perf_counter()
    manifest = read(OUT/'manifest.json')
    for p, digest in manifest['analyzer_hashes'].items():
        assert sha(p) == digest, p
    root = SOURCE/f'{kind}-{seed}'
    history = read(root/'training.json')['history']
    initial, final = [SchedulingActorCritic.load(root/f'{stage}.pt').eval() for stage in ('initial', 'final')]
    records, diagnostics = [], []
    for iteration, step in enumerate(history):
        source = root/f'samples-{iteration}.json.gz'
        corpus = read(source)
        samples = [decode(s) for s in corpus['samples']]
        assert len(samples) == 2048
        boundary = step['days'][0]['accounting']['samples']
        assert sum(d['accounting']['samples'] for d in step['days']) == corpus['source_decisions']
        pre_path = root/('initial.pt' if iteration == 0 else f'update-{iteration-1}.pt')
        pre, post = SchedulingActorCritic.load(pre_path).eval(), SchedulingActorCritic.load(root/f'update-{iteration}.pt').eval()
        predictions = {name: infer(net, samples) for name, net in zip(VERSIONS, (pre, post, initial, final))}
        advantage, _, _ = normalized_advantages(samples, seed+3000+iteration)
        max_lp = max(abs(v-s.old_log_probability) for v, s in zip(predictions['pre'][2], samples))
        max_v = max(abs(v-s.old_value) for v, s in zip(predictions['pre'][1], samples))
        assert max_lp < 2e-5 and max_v < 2e-5, (max_lp, max_v)
        update_rows, target_error = [], 0.
        for i, sample in enumerate(samples):
            day_slot = int(corpus['indices'][i] >= boundary)
            day = step['days'][day_slot]
            target = -(day['result']['phi_krw']-sample.past_cost)/manifest['configuration']['return_scale']
            target_error = max(target_error, abs(target-sample.target))
            stage, mask, chosen = first_choice(sample)
            row = dict(kind=kind, seed=seed, iteration=iteration, sample_index=i,
                source_index=corpus['indices'][i], day_index=day['case']['index'], load=day['case']['load'],
                hour=sample.state.time_fraction*24., time_bin=min(int(sample.state.time_fraction*4), 3),
                target=sample.target, old_value=sample.old_value,
                pre_value=predictions['pre'][1][i], post_value=predictions['post'][1][i],
                raw_advantage=sample.target-sample.old_value, advantage=advantage[i],
                pre_logp=predictions['pre'][2][i], post_logp=predictions['post'][2][i],
                stage=stage, **mask_description(sample, mask, chosen))
            probabilities = {}
            for name in VERSIONS:
                row[name], probabilities[name] = choice_metrics(predictions[name][0][i], mask, chosen, sample.state.features)
            for a, b in (('pre', 'post'), ('initial', 'final')):
                row[a+'_'+b+'_tv'] = float(abs(probabilities[a]-probabilities[b]).sum()/2)
            update_rows.append(row)
        assert target_error < 1e-10
        diagnostic = dict(kind=kind, seed=seed, iteration=iteration, samples=len(samples),
            logp_replay_max_error=max_lp, value_replay_max_error=max_v, target_max_error=target_error,
            day_separation=between_group_share(update_rows),
            day_separation_within_time_bin=between_group_share(update_rows, strata='time_bin'),
            gradient=gradients(pre, samples, seed+3000+iteration),
            source=source.as_posix(), source_sha256=sha(source))
        records.extend(update_rows)
        diagnostics.append(diagnostic)
        print(json.dumps(dict(kind=kind, seed=seed, iteration=iteration, states=len(records),
                              seconds=round(time.perf_counter()-started, 1))), flush=True)
    output = OUT/f'{kind}-{seed}.jsonl.gz'
    with output.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            for row in records:
                gz.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())
    save(OUT/f'{kind}-{seed}-checks.json', dict(updates=diagnostics, output_sha256=sha(output),
                                              rows=len(records), seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        assert args.kind is not None and args.seed is not None
        run(args.kind, args.seed)
