"""Independently recompute final Huber losses and verify finite, reproducible weights."""
from pathlib import Path
import math

import numpy as np
import torch

from context_learning import OUT,ARMS,SEEDS,check,examples,pack
from audit_day_shapes import read
from six_step_learning import gz_read,load_net,sha
from yard_rl.integrated.repro import write_result


def main():
    torch.set_num_threads(1);check()
    corpora=[gz_read(OUT/'corpora'/f'{i:02d}.json.gz') for i in range(9)]
    splits=dict(train=[e for c in corpora[:6] for e in examples(c)],
                validation=[e for c in corpora[6:] for e in examples(c)])
    rows=[]
    for seed in SEEDS:
        report=read(OUT/f'model-{seed}'/'training.json');assert len(report['rows'])==6
        for arm in ARMS:
            row=next(r for r in report['rows'] if r['arm']==arm)
            path=OUT/f'model-{seed}'/f'{arm}.pt';assert sha(path)==row['checkpoint_sha256']
            net=load_net(path);assert all(torch.isfinite(v).all() for v in net.state_dict().values())
            assert len(row['history'])==6
            assert all(h['steps']==(200 if arm=='point_steps' else 50) for h in row['history'])
            checked={}
            for split,items in splits.items():
                x,g,p,y=pack(items,row['scale'])
                with torch.no_grad():scores=net(x,g) if getattr(net,'requires_candidate_context',False) else net(x)
                residual=(scores[p[:,0]]-scores[p[:,1]]-y).numpy();absolute=np.abs(residual)
                loss=float(np.where(absolute<1,.5*residual**2,absolute-.5).mean())
                assert math.isclose(loss,row[split]['loss'],rel_tol=2e-6,abs_tol=2e-6),(seed,arm,split,loss,row[split])
                nz=np.abs(y.numpy())>1e-9
                pred=(scores[p[:,0]]-scores[p[:,1]]).numpy()
                correct=int((np.sign(pred[nz])==np.sign(y.numpy()[nz])).sum())
                assert correct==row[split]['correct'] and int(nz.sum())==row[split]['nonzero']
                checked[split]=dict(loss=loss,correct=correct,nonzero=int(nz.sum()))
            if seed==SEEDS[0]:
                repeated=load_net(OUT/f'model-{seed}-repeat'/f'{arm}.pt').state_dict()
                assert repeated.keys()==net.state_dict().keys()
                assert all(torch.equal(v,repeated[k]) for k,v in net.state_dict().items())
            rows.append(dict(seed=seed,arm=arm,checked=checked,checkpoint_sha256=sha(path)))
    write_result(OUT/'training-audit.json',dict(rows=rows,finite_checkpoints=18,
        independently_recomputed_fitting_tables=36,first_seed_six_retraining_weights_exact=True))
    print('18 finite models; 36 fitting tables independently recomputed; 6 retraining weights exact')


if __name__=='__main__':main()
