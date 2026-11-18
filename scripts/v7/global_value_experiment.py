"""Run preregistered common-experience prediction first; PPO is conditional."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
import hashlib
from pathlib import Path
import subprocess
import time

from global_value_inputs import (OUT, PREREG, FIT, OLD_BANK, MIX, MINUTES, BASE, INFO,
    AY, SEEDS, ARMS, SCALE, RUNTIME, read, save, checkpoint_path, training_cases,
    validation_cases, holdout_cases, input_case, case_name)
from global_value_samples import encode_sample, decode_sample, condition_sample
from global_value_rollout import rollout
from six_step_learning import sha, gz_save, gz_read
from value_stability import restore, checkpoint, assert_nested_equal


def model_path(arm, seed):
    return OUT/'policy'/f'{arm}-{seed}'


def eval_cases():
    rows = []
    for seed in SEEDS:
        for c in validation_cases():
            rows.append(dict(**c, arm='zero', seed=seed, stage='initial'))
    for arm in ARMS:
        for seed in SEEDS:
            for c in validation_cases()+holdout_cases():
                rows.append(dict(**c, arm=arm, seed=seed, stage='final'))
    for c in holdout_cases():
        rows.append(dict(**c, arm='SF_SPT', seed=0, stage='rule'))
    return rows


def prepare():
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no']).strip()
    assert not (OUT/'campaign.json').exists()
    assert read(OUT/'input-preflight.json')['passed'] and read(OUT/'smoke.json')['passed']
    paths = [PREREG, FIT, OLD_BANK, MIX, MINUTES, BASE, INFO, AY/'bank.json', OUT/'bank.json',
             *[checkpoint_path(s) for s in SEEDS], *Path('src/yard_rl/v7').rglob('*.py'),
             *Path('scripts/v7').glob('global_value_*.py'),
             *[Path('scripts/v7')/n for n in ('value_stability.py','value_stability_inputs.py',
                 'resched_ppo.py','six_step_learning.py','horizon_metrics.py','audit_admission_ledger.py',
                 'verify_job_completion.py','completion_learning.py','audit_day_shapes.py',
                 'evaluate_visit_links.py','benchmark_default_eval.py')]]
    cfg = dict(arms=ARMS, seeds=SEEDS, train=training_cases(), validation=validation_cases(),
        holdout=holdout_cases(), eval_cases=eval_cases(), fixed_behavior_seed=SEEDS[0],
        offline_train_samples_per_day=1024, offline_validation_samples_per_day=2048,
        offline_epochs=200, offline_batch=128, lr=.0003,
        prediction_gate=dict(overall_rmse_reduction_min=.05, all_three_pairs_improve=True,
                             max_load_increase=.02, max_time_increase=.05),
        updates=6, days_per_update=2, max_samples=2048, epochs=4, batch_size=64,
        clip=.2, value_coefficient=.5, entropy_coefficient=.01, gradient_clip=1.,
        return_scale=SCALE, discount=1., observation_s=86400, engine_s=93600, runtime=RUNTIME,
        initialization='AY final PopArt weights and moments; new Adam in both arms, not full continuation',
        normalization='same cumulative PopArt in both arms; no actor context; zero new critic column',
        gate='offline final epoch 200 only, then PPO final update 5 only; no result-dependent budget',
        synthetic_bank_seed=9964000, workers=6, torch_threads=1,
        authorization='2026-11-18 explicit two-stage synthetic diagnostic; realism rejection retained')
    save(OUT/'campaign.json', dict(configuration=cfg, sources={p.as_posix():sha(p) for p in paths},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), clean_tracked_tree=True))


def check():
    obj=read(OUT/'campaign.json')
    for p,h in obj['sources'].items():
        assert sha(p)==h, ('Frozen source changed',p)
    return obj['configuration']


def frozen_weights(net):
    h=hashlib.sha256()
    for name, p in net.named_parameters():
        if not name.startswith('critic.'):
            h.update(name.encode());h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def smoke():
    import torch
    from yard_rl.v7.crane.global_context import expand_value_net, validate_observation
    from yard_rl.v7.crane.actor_critic import pack_states
    from yard_rl.v7.crane.value_normalization import normalized_ppo_update
    torch.set_num_threads(1)
    c=input_case('train',12,3500); old,_,_=restore(checkpoint_path(SEEDS[0]))
    outputs=[]
    for mode in ('off','record'):
        data,orders,flows,r=rollout(c,old,mode=mode,stochastic=True,collect=True,action_seed=9969212)
        outputs.append((data,r));gz_save(OUT/f'smoke-{mode}.json.gz',data)
    a,b=outputs
    keys=('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256')
    assert all(a[0]['probe'][k]==b[0]['probe'][k] for k in keys)
    assert a[0]['orders_sha256']==b[0]['orders_sha256']
    assert torch.equal(a[1].rng.get_state(),b[1].rng.get_state())
    samples=b[1].samples[:256]
    assert samples and all(s.global_observation['age_s']<=60.+1e-6 for s in samples)
    all_observations=b[1].global_decisions
    for s in samples:
        validate_observation(s.global_observation)
        assert encode_sample(decode_sample(encode_sample(s)))==encode_sample(s)
    rows=[]
    for mode in ARMS:
        net=expand_value_net(old); corpus=[condition_sample(s,mode) for s in samples]
        with torch.no_grad():
            for s,cs in zip(samples,corpus):
                lo,vo=old(pack_states([s.state]));ln,vn=net(pack_states([cs.state]))
                assert torch.equal(lo,ln) and torch.allclose(vo,vn,atol=2e-5)
        result=normalized_ppo_update(net,torch.optim.Adam(net.parameters(),lr=.0003),corpus,
            seed=9969213,max_samples=128,epochs=1,batch_size=64)
        rows.append(dict(mode=mode,result=result))
    save(OUT/'smoke.json',dict(passed=True,case=c,record_only_exact=True,
        sample_roundtrip=len(samples),updates=rows,observations=len(all_observations),
        behavior_checkpoint=checkpoint_path(SEEDS[0]).as_posix(),
        before24_max_age=max(o['age_s'] for o in all_observations if o['decision_time']<86400),
        after24_max_age=max((o['age_s'] for o in all_observations if o['decision_time']>=86400),default=0)))
    print('smoke passed: exact behavior, snapshot timing, serialization, normalized updates',flush=True)


def collect(c, repeat=False):
    import torch
    torch.set_num_threads(1);cfg=check()
    name=case_name(c)+('-repeat' if repeat else '')
    path=OUT/'common'/f'{name}.json.gz'
    assert not path.exists(),path
    net,_,_=restore(checkpoint_path(cfg['fixed_behavior_seed']))
    data,orders,flows,r=rollout(c,net,mode='record',stochastic=True,collect=True,
                               action_seed=9965000+c['input_seed'])
    rng=torch.Generator().manual_seed(9966000+c['input_seed'])
    limit=cfg['offline_train_samples_per_day'] if c['split']=='train' else cfg['offline_validation_samples_per_day']
    indices=torch.randperm(len(r.samples),generator=rng)[:limit].tolist()
    assert indices
    selected=[r.samples[i] for i in indices]
    save_data=dict(case=c,indices=indices,samples=[encode_sample(s) for s in selected],
                   available=len(r.samples),data=data)
    for sub,obj in [('common',save_data),('common-orders',orders),('common-flow-orders',flows),
                    ('common-observations',r.global_decisions)]:
        gz_save(OUT/sub/f'{name}.json.gz',obj)
    if repeat:
        prior=gz_read(OUT/'common'/f'{case_name(c)}.json.gz')
        assert prior['samples']==save_data['samples'] and prior['indices']==indices
        for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
            assert prior['data']['probe'][key]==data['probe'][key],key
    print(f'common {name}: {len(selected)}/{len(r.samples)} samples, {data["seconds"]:.1f}s',flush=True)


def embed(seed):
    import torch
    from yard_rl.v7.crane.global_context import expand_value_net
    from yard_rl.v7.crane.actor_critic import pack_states
    torch.set_num_threads(1);cfg=check()
    old,_,_=restore(checkpoint_path(seed));net=expand_value_net(old)
    x=[];targets=[];metadata=[]
    for c in cfg['train']+cfg['validation']:
        raw=gz_read(OUT/'common'/f'{case_name(c)}.json.gz')
        samples=[condition_sample(decode_sample(d),'global') for d in raw['samples']]
        with torch.no_grad():
            for begin in range(0,len(samples),16):
                chunk=samples[begin:begin+16]
                _,features=net.actor_and_critic_input(pack_states([s.state for s in chunk]));x.append(features)
        targets.extend(s.target for s in samples)
        metadata.extend(dict(case=case_name(c),split=c['split'],load=c['load'],
            hour_bin=min(3,int(s.global_observation['decision_time']//21600))) for s in samples)
    path=OUT/'features'/f'{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(x=torch.cat(x),y=torch.tensor(targets),metadata=metadata,
                    frozen_weights_sha256=frozen_weights(net),source_sha256=sha(checkpoint_path(seed))),path)
    print(f'embedded {seed}: {len(targets)} common decisions',flush=True)


def error_metrics(prediction, target, metadata):
    import math
    import statistics as st
    import torch
    residual=(prediction-target).double()
    def group(indices):
        per_day={}
        for i in indices:per_day.setdefault(metadata[i]['case'],[]).append(i)
        days=[dict(case=k,n=len(ix),mse=float(residual[ix].square().mean()),bias=float(residual[ix].mean()))
              for k,ix in sorted(per_day.items())]
        return dict(rmse=math.sqrt(st.mean(d['mse'] for d in days)),
                    bias=st.mean(d['bias'] for d in days),samples=len(indices),days=days)
    result=group(list(range(len(metadata))))
    result['by_load']={str(load):group([i for i,m in enumerate(metadata) if m['load']==load])
                       for load in sorted({m['load'] for m in metadata})}
    result['by_time']={str(hour):group([i for i,m in enumerate(metadata) if m['hour_bin']==hour])
                       for hour in sorted({m['hour_bin'] for m in metadata})}
    return result


def fit_prediction(task, replay=False):
    import torch
    from yard_rl.v7.crane.global_context import expand_value_net
    torch.set_num_threads(1);cfg=check();arm,seed=task
    root=OUT/'prediction'/f'{arm}-{seed}';root.mkdir(parents=True,exist_ok=True)
    old,_,_=restore(checkpoint_path(seed));net=expand_value_net(old)
    data=torch.load(OUT/'features'/f'{seed}.pt',map_location='cpu',weights_only=False)
    x=data['x'].clone();y=data['y'];meta=data['metadata']
    if arm=='zero':x[:,-1]=0.
    train=torch.tensor([i for i,m in enumerate(meta) if m['split']=='train'])
    valid=torch.tensor([i for i,m in enumerate(meta) if m['split']=='selection'])
    for name,p in net.named_parameters():p.requires_grad_(name.startswith('critic.'))
    frozen_before=frozen_weights(net)
    assert frozen_before==data['frozen_weights_sha256']
    with torch.no_grad():before=net.critic(x).squeeze(-1)*net.target_scale.float()+net.target_mean.float()
    norm=net.update_statistics(y[train])
    normalized_y=(y-net.target_mean.float())/net.target_scale.float()
    with torch.no_grad():after=net.critic(x).squeeze(-1)*net.target_scale.float()+net.target_mean.float()
    preserve=float((before-after).abs().max());assert preserve<2e-5
    opt=torch.optim.Adam(net.critic.parameters(),lr=cfg['lr'])
    rng=torch.Generator().manual_seed(seed+9967000)
    curves=[];steps=0
    def measure(epoch):
        with torch.no_grad():pred=net.critic(x).squeeze(-1)*net.target_scale.float()+net.target_mean.float()
        curves.append(dict(epoch=epoch,train=error_metrics(pred[train],y[train],[meta[i] for i in train]),
                           validation=error_metrics(pred[valid],y[valid],[meta[i] for i in valid])))
        return pred
    measure(0)
    for epoch in range(1,cfg['offline_epochs']+1):
        order=train[torch.randperm(len(train),generator=rng)]
        for begin in range(0,len(order),cfg['offline_batch']):
            ix=order[begin:begin+cfg['offline_batch']]
            pred=net.critic(x[ix]).squeeze(-1)
            loss=.5*(pred-normalized_y[ix]).square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward()
            g=torch.nn.utils.clip_grad_norm_(net.critic.parameters(),1.);assert torch.isfinite(g)
            opt.step();steps+=1
        if epoch in (20,50,100,200):measure(epoch)
    assert frozen_weights(net)==frozen_before
    if replay:
        saved=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
        assert_nested_equal(saved['state_dict'],net.state_dict());assert_nested_equal(saved['optimizer'],opt.state_dict())
        save(root/'replay.json',dict(passed=True,optimizer_steps=steps));return
    checkpoint(root/'final.pt',net,opt,arm=arm,seed=seed,iteration=200)
    with torch.no_grad():pred=net.critic(x).squeeze(-1)*net.target_scale.float()+net.target_mean.float()
    gz_save(root/'predictions.json.gz',dict(prediction=pred.tolist(),target=y.tolist(),metadata=meta))
    save(root/'result.json',dict(arm=arm,seed=seed,normalization=norm,preservation_error=preserve,
        frozen_weights_sha256=frozen_before,actor_frozen_verified=True,optimizer_steps=steps,curves=curves,
        final=curves[-1],feature_sha256=sha(OUT/'features'/f'{seed}.pt'),checkpoint_sha256=sha(root/'final.pt')))
    print(f'prediction {arm}/{seed}: RMSE {curves[-1]["validation"]["rmse"]:.6f}, {steps} updates',flush=True)


def judge_prediction():
    import math
    import statistics as st
    cfg=check();rows={}
    for arm in ARMS:
        rows[arm]=[read(OUT/'prediction'/f'{arm}-{s}'/'result.json') for s in SEEDS]
    def pool(arm,key=None,cell=None):
        values=[r['final']['validation'] for r in rows[arm]]
        if key:values=[v[key][cell] for v in values]
        return math.sqrt(st.mean(v['rmse']**2 for v in values))
    groups={a:dict(rmse=pool(a),by_load={str(l):pool(a,'by_load',str(l)) for l in (3500,7500,15000)},
                   by_time={str(h):pool(a,'by_time',str(h)) for h in range(4)}) for a in ARMS}
    ratio=groups['global']['rmse']/groups['zero']['rmse']-1
    by_seed={str(s):rows['global'][i]['final']['validation']['rmse']/rows['zero'][i]['final']['validation']['rmse']-1
             for i,s in enumerate(SEEDS)}
    changes={k:{c:groups['global'][k][c]/groups['zero'][k][c]-1 for c in groups['zero'][k]}
             for k in ('by_load','by_time')}
    conditions=dict(overall_reduction=ratio<=-cfg['prediction_gate']['overall_rmse_reduction_min'],
        all_seed_pairs_improve=all(x<0 for x in by_seed.values()),
        load_guard=max(changes['by_load'].values())<=cfg['prediction_gate']['max_load_increase'],
        time_guard=max(changes['by_time'].values())<=cfg['prediction_gate']['max_time_increase'],
        actor_frozen=all(r['actor_frozen_verified'] for rs in rows.values() for r in rs),
        observation_contract=read(OUT/'smoke.json')['passed'])
    result=dict(passed=all(conditions.values()),conditions=conditions,groups=groups,
        rmse_change_pct=100*ratio,by_seed_change_pct={k:100*v for k,v in by_seed.items()},
        by_load_change_pct={k:100*v for k,v in changes['by_load'].items()},
        by_time_change_pct={k:100*v for k,v in changes['by_time'].items()},
        scope='Common-policy prediction diagnostic; no scheduling-performance claim',
        final_epoch=200,model_pairs=3,validation_inputs=9)
    save(OUT/'prediction-summary.json',result)
    print(f'prediction gate: {result}',flush=True)
    return result


def train_policy(task, replay=False):
    import torch
    from yard_rl.v7.crane.global_context import expand_value_net
    from yard_rl.v7.crane.value_normalization import normalized_ppo_update
    torch.set_num_threads(1);cfg=check();assert read(OUT/'prediction-summary.json')['passed']
    arm,seed=task;root=model_path(arm,seed);root.mkdir(parents=True,exist_ok=True)
    old,_,_=restore(checkpoint_path(seed));net=expand_value_net(old)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['lr']);history=[]
    if not replay:
        assert not (root/'initial.pt').exists()
        checkpoint(root/'initial.pt',net,opt,arm=arm,seed=seed,iteration=-1)
    for it in range(cfg['updates']):
        selected=[];meta=[];days=[]
        if replay:
            obj=gz_read(root/f'samples-{it}.json.gz')
            selected=[decode_sample(d) for d in obj['samples']]
        else:
            all_samples=[];all_meta=[]
            for j in (2*it,2*it+1):
                c=cfg['train'][j]
                data,orders,flows,r=rollout(c,net,mode=arm,stochastic=True,collect=True,action_seed=seed+1000+j)
                all_samples.extend(r.samples);all_meta.extend([c]*len(r.samples))
                days.append(dict(case=c,data=data));gz_save(root/f'day-{j}-observations.json.gz',r.global_decisions)
            rng=torch.Generator().manual_seed(seed+2000+it)
            indices=torch.randperm(len(all_samples),generator=rng)[:cfg['max_samples']].tolist()
            selected=[all_samples[i] for i in indices];meta=[all_meta[i] for i in indices]
            gz_save(root/f'samples-{it}.json.gz',dict(indices=indices,samples=[encode_sample(s) for s in selected],
                metadata=meta,source_decisions=len(all_samples),days=days))
        result=normalized_ppo_update(net,opt,selected,seed=seed+3000+it,max_samples=len(selected),
                                    epochs=cfg['epochs'],batch_size=cfg['batch_size'])
        if replay:
            saved=torch.load(root/f'update-{it}.pt',map_location='cpu',weights_only=False)
            assert_nested_equal(saved['state_dict'],net.state_dict());assert_nested_equal(saved['optimizer'],opt.state_dict())
        else:
            checkpoint(root/f'update-{it}.pt',net,opt,arm=arm,seed=seed,iteration=it)
            history.append(dict(iteration=it,optimization=result,days=days,samples_sha256=sha(root/f'samples-{it}.json.gz'),
                                checkpoint_sha256=sha(root/f'update-{it}.pt')))
            save(root/'training.json',dict(arm=arm,seed=seed,history=history))
        print(f'policy {arm}/{seed} update {it+1}/6, replay={replay}',flush=True)
    if replay:save(root/'replay.json',dict(passed=True,updates=6))
    else:checkpoint(root/'final.pt',net,opt,arm=arm,seed=seed,iteration=5)


def evaluate(i, repeat=False):
    import torch
    torch.set_num_threads(1);check();assert read(OUT/'prediction-summary.json')['passed']
    c=eval_cases()[i];name=f'{i:03d}'+('-repeat' if repeat else '')
    path=OUT/'evaluation'/f'{name}.json.gz';assert not path.exists(),path
    net=None;cp=None
    if c['arm'] in ARMS:
        cp=model_path(c['arm'],c['seed'])/f'{c["stage"]}.pt';net,_,_=restore(cp)
    data,orders,flows,r=rollout(c,net,mode=c['arm'] if net is not None else 'off')
    data['checkpoint_sha256']=sha(cp) if cp else None
    for sub,obj in [('evaluation',data),('orders',orders),('flow-orders',flows)]:gz_save(OUT/sub/f'{name}.json.gz',obj)
    if r:gz_save(OUT/'observations'/f'{name}.json.gz',r.global_decisions)
    if repeat:
        prior=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz')
        for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
            assert prior['probe'][key]==data['probe'][key],key
        assert prior['orders_sha256']==data['orders_sha256']
    print(f'eval {name} {c["arm"]}/{c["seed"]} {case_name(c)}, {data["seconds"]:.1f}s',flush=True)


def parallel(fn, tasks, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,t):t for t in tasks}
        for f in as_completed(futures):f.result()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','prepare','collect','embed','predict','judge',
        'train','evaluate','prediction-replay','policy-replay','common-repeat','evaluation-repeat'])
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    pairs=[(a,s) for a in ARMS for s in SEEDS]
    if args.mode=='smoke':smoke()
    elif args.mode=='prepare':prepare()
    elif args.mode=='collect':parallel(collect,training_cases()+validation_cases(),args.workers)
    elif args.mode=='embed':parallel(embed,SEEDS,min(3,args.workers))
    elif args.mode=='predict':parallel(fit_prediction,pairs,args.workers)
    elif args.mode=='judge':judge_prediction()
    elif args.mode=='train':parallel(train_policy,pairs,args.workers)
    elif args.mode=='evaluate':parallel(evaluate,range(len(eval_cases())),args.workers)
    elif args.mode=='prediction-replay':
        parallel(partial(fit_prediction,replay=True),pairs,args.workers)
    elif args.mode=='policy-replay':
        parallel(partial(train_policy,replay=True),pairs,args.workers)
    elif args.mode=='common-repeat':collect(validation_cases()[4],repeat=True)
    elif args.mode=='evaluation-repeat':
        for arm in ARMS:
            i=next(i for i,c in enumerate(eval_cases()) if c['arm']==arm and c['split']=='holdout' and c['load']==7500)
            evaluate(i,repeat=True)


if __name__=='__main__':main()
