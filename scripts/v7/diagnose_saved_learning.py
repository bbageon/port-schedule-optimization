"""Fixed splits, initialization and budget: replay and optimizer-step diagnostics."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn

from completion_learning import SEEDS
from six_step_learning import gz_read, labelset, load_net, checkpoint, sha
from yard_rl.integrated.repro import write_result, repro_stamp
from yard_rl.v7.crane.fit import CraneTrainer
from yard_rl.v7.train.fit import LR, GRAD_CLIP

OUT=Path('outputs/reports/yr323_learning_diagnosis')
OLD=Path('outputs/reports/yr332_linked_retraining')
PREREG=Path('docs/research/v7-discharge/synthetic-four-step-prereg.md')


def partition(data):
    samples=data['samples'];n=len(samples)//2;seed=data['input_seed']
    rng=torch.Generator().manual_seed(seed)
    order=torch.randperm(n,generator=rng).tolist();val=set(order[:max(1,int(n*.2))])
    x=torch.tensor([s['row'] for s in samples],dtype=torch.float32).reshape(n,2,-1)
    y=torch.tensor([s['target'] for s in samples],dtype=torch.float32).reshape(n,2)
    tr=[i for i in range(n) if i not in val];va=sorted(val)
    return (x[tr],y[tr]),(x[va],y[va]),rng


def metrics(net,xy):
    x,y=xy
    with torch.no_grad():
        p=net(x.flatten(0,1)).reshape(-1,2);pg=p[:,0]-p[:,1];yg=y[:,0]-y[:,1];nz=yg.abs()>1e-9
        correct=(pg[nz].sign()==yg[nz].sign())
        return dict(pairs=len(y),nonzero=int(nz.sum()),correct=int(correct.sum()),
                    accuracy=float(correct.float().mean()) if nz.any() else None,
                    loss=float(nn.functional.smooth_l1_loss(pg,yg)))


def main():
    torch.set_num_threads(1);started=time.perf_counter();rows=[];artifacts={}
    for seed in SEEDS:
        corpora=[gz_read(OLD/f'model-{seed}'/f'labels-{i}.json.gz') for i in range(6)]
        splits=[partition(c) for c in corpora]
        valid=(torch.cat([s[1][0] for s in splits]),torch.cat([s[1][1] for s in splits]))
        alltrain=(torch.cat([s[0][0] for s in splits]),torch.cat([s[0][1] for s in splits]))
        for arm in ('current50','replay50','current200'):
            net=load_net(OLD/f'model-{seed}'/'initial.pt')
            opt=torch.optim.Adam(net.parameters(),lr=LR);history=[];seenx=[];seeny=[]
            for it,(train,val,_) in enumerate(splits):
                x,y=train;seenx.append(x);seeny.append(y)
                # Recreate each original split and generator state before minibatch sampling.
                _,_,rng=partition(corpora[it])
                if arm=='replay50':x,y=torch.cat(seenx),torch.cat(seeny)
                steps=200 if arm=='current200' else 50
                net.train()
                for _ in range(steps):
                    b=torch.randint(0,len(y),(min(128,len(y)),),generator=rng)
                    p=net(x[b].flatten(0,1)).reshape(-1,2)
                    loss=nn.functional.smooth_l1_loss(p[:,0]-p[:,1],y[b,0]-y[b,1])
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),GRAD_CLIP);opt.step()
                net.eval()
                if arm=='current50':
                    expected=load_net(OLD/f'model-{seed}'/f'final-{it}.pt').state_dict()
                    assert all(torch.equal(v,expected[k]) for k,v in net.state_dict().items()),('original replay mismatch',seed,it)
                history.append(dict(iteration=it,train=metrics(net,train),validation=metrics(net,val),
                    previous_train=metrics(net,(torch.cat(seenx),torch.cat(seeny))),steps=steps))
            path=OUT/f'model-{seed}'/f'{arm}.pt';assert not path.exists()
            checkpoint(path,net,diagnostic=arm,model_seed=seed)
            artifacts[path.as_posix()]=sha(path)
            rows.append(dict(seed=seed,arm=arm,train=metrics(net,alltrain),validation=metrics(net,valid),history=history))
            print(seed,arm,rows[-1]['validation'],flush=True)
    result=dict(rows=rows,artifacts=artifacts,original18_checkpoints_exact=True,seconds=time.perf_counter()-started,
        scope='existing corpus fixed-split diagnostic; validation was previously seen, not final generalization',
        repro=repro_stamp(experiment='YR-323-d',seeds=dict(models=list(SEEDS)),
            params=dict(arms=['current50','replay50','current200'],steps=[50,50,200],updates=6,lr=LR),prereg=PREREG.as_posix()))
    write_result(OUT/'saved-diagnosis.json',result)


if __name__=='__main__': main()
