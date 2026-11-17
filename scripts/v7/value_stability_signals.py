"""Same-state choices and first-minibatch gradients, without changing any policy."""
from collections import defaultdict
import math

import torch

from ppo_credit_metrics import first_choice, flow_id, normalized_advantages
from resched_ppo import decode_sample
from six_step_learning import gz_read, gz_save, sha
from value_stability import check, folder, restore
from value_stability_eval import cases
from value_stability_inputs import OUT, ARMS, SEEDS, LOADS, save
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.on_policy import action_statistics


def length(grads):
    return math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None))


def gradients():
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            root=folder(arm,seed)
            for it in range(6):
                net,_,_=restore(root/('initial.pt' if it==0 else f'update-{it-1}.pt'))
                samples=[decode_sample(s) for s in gz_read(root/f'samples-{it}.json.gz')['samples']]
                _,permutation,rng=normalized_advantages(samples,seed+3000+it)
                corpus=[samples[i] for i in permutation]
                target=torch.tensor([s.target for s in corpus])
                adv=target-torch.tensor([s.old_value for s in corpus])
                adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
                if arm=='popart':net.update_statistics(target)
                ix=torch.randperm(len(corpus),generator=rng)[:64]
                batch=[corpus[i] for i in ix]
                if arm=='popart':
                    logits,value=net.normalized_forward(pack_states([s.state for s in batch]))
                    wanted=(target[ix]-net.target_mean.float())/net.target_scale.float()
                else:
                    logits,value=net(pack_states([s.state for s in batch]));wanted=target[ix]
                lp,entropy=action_statistics(logits,batch)
                ratio=(lp-torch.tensor([s.old_log_probability for s in batch])).exp()
                actor=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
                critic=.5*.5*(value-wanted).square().mean()
                params=list(net.parameters())
                ga=torch.autograd.grad(actor,params,allow_unused=True,retain_graph=True)
                gc=torch.autograd.grad(critic,params,allow_unused=True,retain_graph=True)
                gt=torch.autograd.grad(actor+critic-.01*entropy.mean(),params,allow_unused=True)
                a,c,t=length(ga),length(gc),length(gt)
                rows.append(dict(arm=arm,seed=seed,iteration=it,actor_norm=a,weighted_critic_norm=c,
                    critic_actor_ratio=c/a if a else None,total_norm=t,global_clip_factor=min(1.,1./(t+1e-6)),
                    scale=net.target_scale.item()))
    save(OUT/'gradient-diagnostic.json',dict(rows=rows,
        scope='first minibatch of each update, before clipping and Adam; not full training gradient attribution'))


def choices():
    summary=[];sources={}
    for seed in SEEDS:
        nets=[restore(folder('control',seed)/'initial.pt')[0]]
        nets.extend(restore(folder(a,seed)/'final.pt')[0] for a in ARMS)
        rows=[]
        for source_arm in ARMS:
            for load in LOADS:
                i=next(i for i,c in enumerate(cases()) if c['arm']==source_arm and c['seed']==seed
                       and c['mode']=='stochastic' and c['load']==load)
                path=OUT/'validation-samples'/f'{i:03d}.json.gz';sources[path.as_posix()]=sha(path)
                corpus=[decode_sample(s) for s in gz_read(path)['samples']]
                for begin in range(0,len(corpus),64):
                    batch=corpus[begin:begin+64]
                    with torch.no_grad():predictions=[n(pack_states([s.state for s in batch]))[0] for n in nets]
                    for j,s in enumerate(batch):
                        _,mask,_=first_choice(s)
                        groups=torch.tensor([flow_id(s.state.features[k]) for k in mask])
                        single=len({int(s.state.cranes[k]) for k in mask})==1
                        mixed=bool((groups==0).any() and (groups==1).any())
                        detail=dict(seed=seed,source_arm=source_arm,load=load,sample=begin+j,
                                    single_crane=single,mixed_truck=mixed)
                        ps=[]
                        for label,prediction in zip(('initial','control','popart'),predictions):
                            ll=prediction[j,mask].double();p=torch.softmax(ll,0);ps.append(p)
                            top=int(p.argmax());truck=groups<2
                            detail[label]=dict(top_candidate=mask[top],top_flow=int(groups[top]),
                                normalized_entropy=float(-(p*p.clamp_min(1e-300).log()).sum()/math.log(len(mask))),
                                inbound_conditional=float(p[groups==0].sum()/p[truck].sum()) if mixed else None)
                        detail['control_popart_total_variation']=float(.5*(ps[1]-ps[2]).abs().sum())
                        rows.append(detail)
        gz_save(OUT/f'choice-states-{seed}.json.gz',rows)
        for load in (0,*LOADS):
            group=[r for r in rows if load==0 or r['load']==load]
            mixed=[r for r in group if r['single_crane'] and r['mixed_truck']]
            item=dict(seed=seed,load=load,n=len(group),mixed_single_crane_n=len(mixed),
                mean_total_variation=sum(r['control_popart_total_variation'] for r in group)/len(group),
                argmax_disagreement=sum(r['control']['top_candidate']!=r['popart']['top_candidate'] for r in group)/len(group))
            for label in ('initial','control','popart'):
                item[label]=dict(mean_normalized_entropy=sum(r[label]['normalized_entropy'] for r in group)/len(group),
                    inbound_conditional=sum(r[label]['inbound_conditional'] for r in mixed)/len(mixed) if mixed else None,
                    inbound_top_fraction=sum(r[label]['top_flow']==0 for r in mixed)/len(mixed) if mixed else None)
            summary.append(item)
    save(OUT/'choice-diagnostic.json',dict(rows=summary,sources=sources,
        scope='Same stored legal masks from union of both policies; descriptive ranks, not counterfactual outcomes'))


if __name__=='__main__':
    torch.set_num_threads(1);check();gradients();choices()
