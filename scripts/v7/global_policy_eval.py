"""Fixed argmax operations and separate stochastic value diagnostics for bc."""
import argparse
from functools import partial

from global_policy_inputs import OUT,SEEDS,ARMS,LOADS,save,rollout
from global_policy_plan import check,model_path,eval_cases,repeat_indices
from global_policy_train import parallel
from global_value_samples import encode_sample
from six_step_learning import sha,gz_read,gz_save
from value_stability import restore,value_diagnostic


def evaluate(i,repeat=False):
    import torch
    torch.set_num_threads(1);cfg=check();c=cfg['evaluation_cases'][i]
    assert c==eval_cases()[i]
    if c['split']=='holdout':
        assert all((model_path(a,s)/'final.pt').exists() for a in ARMS for s in SEEDS)
    name=f'{i:03d}'+('-repeat' if repeat else '')
    path=OUT/'evaluation'/f'{name}.json.gz';assert not path.exists(),path
    net=None;cp=None
    if c['arm'] in ARMS:
        cp=model_path(c['arm'],c['seed'])/f'{c["stage"]}.pt';net,_,_=restore(cp)
    stochastic=c['mode']=='stochastic'
    action_seed=(cfg['seed_bases']['stochastic_evaluation']+100*SEEDS.index(c['seed'])+
                 10*c['index']+LOADS.index(c['load'])) if net is not None else 0
    data,orders,flows,r=rollout(c,net,mode=c['arm'] if net is not None else 'off',
        stochastic=stochastic,collect=stochastic,action_seed=action_seed)
    data['checkpoint_sha256']=sha(cp) if cp else None
    if stochastic:
        rng=torch.Generator().manual_seed(action_seed+1000)
        indices=torch.randperm(len(r.samples),generator=rng)[:2048].tolist()
        selected=[r.samples[j] for j in indices];assert len(selected)==2048
        diagnostic=value_diagnostic(net,selected,[c]*len(selected))
        data['value_diagnostic']={k:v for k,v in diagnostic.items() if k not in ('values','action_logps')}
        gz_save(OUT/'validation-samples'/f'{name}.json.gz',dict(indices=indices,
            samples=[encode_sample(s) for s in selected],diagnostic=diagnostic,source_decisions=len(r.samples)))
    for sub,obj in [('evaluation',data),('orders',orders),('flow-orders',flows)]:
        gz_save(OUT/sub/f'{name}.json.gz',obj)
    if r:gz_save(OUT/'observations'/f'{name}.json.gz',r.global_decisions)
    if repeat:
        prior=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz')
        for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
            assert data['probe'][key]==prior['probe'][key],key
        assert data['orders_sha256']==prior['orders_sha256']
        assert data['global_audit']['observation_digest']==prior['global_audit']['observation_digest']
    print(f'eval {name} {c["arm"]}/{c["seed"]} {c["stage"]} {c["split"]}:{c["index"]} {c["load"]} {c["mode"]}: '
          f'{data["result"]["mean_gate_to_job_done_s"]/60:.3f}min, {data["seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('evaluate','repeat'));p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();assert 1<=a.workers<=6
    parallel(partial(evaluate,repeat=a.mode=='repeat'),repeat_indices() if a.mode=='repeat' else list(range(len(eval_cases()))),a.workers)
