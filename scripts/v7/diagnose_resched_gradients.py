"""Read-only post-hoc measurement of the FIRST minibatch of each saved PPO update.

No new fitting, no checkpoint selection, no evaluation-period data.
Component gradient magnitudes describe optimization; they do not prove why a policy wins/loses.
"""
import argparse
import json
import math

import torch

from resched_ppo import OUT, KINDS, SEEDS, folder, decode_sample, save, check
from six_step_learning import gz_read, sha
from yard_rl.v7.crane.actor_critic import SchedulingActorCritic,pack_states
from yard_rl.v7.crane.on_policy import action_statistics


def gradient_vector(loss, parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p,g in zip(parameters,values)])


def main(available_only=False):
    torch.set_num_threads(1);cfg=check();rows=[]
    for kind in KINDS:
        seed=SEEDS[0];root=folder(kind,seed)
        for it in range(6):
            if available_only and not (root/f'update-{it}.pt').exists():continue
            path=root/('initial.pt' if it==0 else f'update-{it-1}.pt')
            net=SchedulingActorCritic.load(path).train()
            source=root/f'samples-{it}.json.gz'
            samples=[decode_sample(s) for s in gz_read(source)['samples']]
            rng=torch.Generator().manual_seed(seed+3000+it)
            indices=torch.randperm(len(samples),generator=rng)[:cfg['max_samples_per_update']].tolist()
            corpus=[samples[i] for i in indices]
            target=torch.tensor([s.target for s in corpus])
            old=torch.tensor([s.old_log_probability for s in corpus])
            adv=target-torch.tensor([s.old_value for s in corpus])
            adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
            ix=torch.randperm(len(corpus),generator=rng)[:cfg['batch_size']]
            batch=[corpus[i] for i in ix]
            ll,value=net(pack_states([s.state for s in batch]))
            logp,entropy=action_statistics(ll,batch)
            error=float((logp-old[ix]).abs().max().detach());assert error<2e-5
            ratio=(logp-old[ix]).exp()
            actor=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
            critic=.25*(value-target[ix]).square().mean()
            ent=-.01*entropy.mean()
            parameters=list(net.parameters())
            ga,gc,ge=[gradient_vector(loss,parameters) for loss in (actor,critic,ent)]
            norms=[float(torch.linalg.vector_norm(g)) for g in (ga,gc,ge,ga+gc+ge)]
            rows.append(dict(kind=kind,seed=seed,iteration=it,actor_grad_norm=norms[0],
                weighted_critic_grad_norm=norms[1],weighted_entropy_grad_norm=norms[2],total_grad_norm=norms[3],
                applied_global_clip_factor=min(1.,1./(norms[3]+1e-6)),
                actor_critic_cosine=float(torch.dot(ga,gc)/(torch.linalg.vector_norm(ga)*torch.linalg.vector_norm(gc)).clamp_min(1e-12)),
                probability_replay_error=error,return_std=float(target.std(unbiased=False)),
                checkpoint_sha256=sha(path),samples_sha256=sha(source)))
    save(OUT/('gradient-progress.json' if available_only else 'gradient-diagnostic.json'),dict(rows=rows,
        scope='post-hoc, first minibatch per update, first model seed of each architecture; descriptive, not a causal ablation',
        settings_changed=False,new_fitting=False))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--available-only',action='store_true')
    main(p.parse_args().available_only)
