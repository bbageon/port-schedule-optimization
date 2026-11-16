"""Optional O(n) candidate/resource-star networks; identical ten node features.

This models shared-crane competition, not truck-visit precedence. Mean pooling is
the same-information control for learned hub attention. Groups are decision-local.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import crane_features, decision_common


class ContextCraneNet(nn.Module):
    requires_candidate_context = True

    def __init__(self, architecture='mean_context', dim=10, hidden=32,
                 feature_mode='feasible', include_flow=True, count_transform='clipped'):
        super().__init__()
        if architecture not in ('mean_context', 'star_attention') or dim != 10 or not include_flow:
            raise ValueError('context comparison requires the same ten-feature contract')
        self.architecture=architecture;self.feature_mode=feature_mode
        self.include_flow=include_flow;self.count_transform=count_transform
        # The name/net[0] contract keeps existing checkpoint metadata tooling usable.
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU())
        self.update=nn.Sequential(nn.Linear(hidden*2,64),nn.ReLU(),nn.Linear(64,1))
        if architecture=='star_attention':
            self.attention=nn.Linear(hidden*2,1,bias=False)

    def forward(self,x,groups=None):
        if groups is None:
            raise ValueError('explicit decision groups required; singleton fallback changes the model')
        if x.ndim!=2 or groups.ndim!=1 or len(groups)!=len(x):
            raise ValueError('one group id per candidate required')
        if not len(x): return x.new_empty(0)
        _,g=torch.unique(groups,sorted=True,return_inverse=True)
        h=self.net(x);n=int(g.max())+1
        counts=torch.bincount(g,minlength=n).to(h.dtype).unsqueeze(1)
        mean=h.new_zeros((n,h.shape[1])).index_add(0,g,h)/counts
        if self.architecture=='mean_context':
            pooled=mean
        else:
            logits=torch.nn.functional.leaky_relu(self.attention(torch.cat((h,mean[g]),dim=1)).squeeze(1),.2)
            maxima=logits.new_full((n,),-torch.inf).scatter_reduce(0,g,logits,reduce='amax',include_self=True)
            weights=(logits-maxima[g]).exp()
            weights=weights/weights.new_zeros(n).index_add(0,g,weights)[g]
            pooled=h.new_zeros((n,h.shape[1])).index_add(0,g,h*weights.unsqueeze(1))
        return self.update(torch.cat((h,pooled[g]),dim=1)).squeeze(1)

    def model_config(self):
        return dict(architecture=self.architecture,dim=self.net[0].in_features,hidden=self.net[0].out_features,
                    feature_mode=self.feature_mode,include_flow=self.include_flow,count_transform=self.count_transform)


def context_ranks(policy,sim,pairs):
    from ..world.contract.schema import CandidateKind
    if policy.force_once or policy.score_once:
        raise ValueError('counterfactual interventions for contextual policies require a separate contract')
    rows=[];groups=[];positions=[];group_ids={};keys=[(2,0.,'') for _ in pairs]
    common=decision_common(sim)
    for i,(cid,gc) in enumerate(pairs):
        if gc.job_ref is None: continue
        rows.append(crane_features(sim,cid,gc,common=common,feature_mode=policy.net.feature_mode,
                    include_flow=policy.net.include_flow,count_transform=policy.net.count_transform))
        groups.append(group_ids.setdefault(cid,len(group_ids)));positions.append(i)
    if rows:
        with torch.no_grad():
            scores=policy.net(torch.tensor(rows,dtype=torch.float32),torch.tensor(groups,dtype=torch.long)).tolist()
        for i,score in zip(positions,scores):
            gc=pairs[i][1]
            keys[i]=(0 if not policy.serve_first or gc.kind==CandidateKind.SERVE else 1,score,gc.job_ref.job_id)
    return keys
