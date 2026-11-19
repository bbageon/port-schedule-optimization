"""Fixed-budget policy training; prediction errors never decide whether to proceed."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import time

from global_policy_inputs import OUT,SEEDS,ARMS,SCALE,read,save,checkpoint_path,input_case,rollout
from global_policy_plan import check,model_path
from global_value_samples import encode_sample,decode_sample,condition_sample
from six_step_learning import sha,gz_save,gz_read
from value_stability import restore,checkpoint,assert_nested_equal,value_diagnostic


def fresh(seed):
    import torch
    from yard_rl.v7.crane.global_context import expand_value_net
    old,_,_=restore(checkpoint_path(seed))
    net=expand_value_net(old)
    return net,torch.optim.Adam(net.parameters(),lr=.0003)


def fit_once(net,opt,samples,metadata,*,seed,epochs=4,batch_size=64):
    from yard_rl.v7.crane.value_normalization import normalized_ppo_update
    before=value_diagnostic(net,samples,metadata)
    result=normalized_ppo_update(net,opt,samples,seed=seed,max_samples=len(samples),epochs=epochs,batch_size=batch_size)
    after=value_diagnostic(net,samples,metadata)
    return result,dict(before=before,after=after)


def smoke():
    import torch
    from yard_rl.v7.crane.global_context import expand_value_net
    from yard_rl.v7.crane.actor_critic import pack_states
    from audit_global_value import verify_ledger,verify_snapshots
    torch.set_num_threads(1)
    assert not (OUT/'smoke.json').exists()
    c=input_case('train',12,3500);old,_,_=restore(checkpoint_path(SEEDS[0]))
    reference=None;results=[];reference_samples=None
    for mode in ('off','zero','global'):
        net=old if mode=='off' else expand_value_net(old)
        data,orders,flows,r=rollout(c,net,mode=mode,stochastic=True,collect=True,action_seed=10089000)
        verify_ledger(data,orders,flows)
        gz_save(OUT/f'smoke-{mode}.json.gz',dict(data=data,orders=orders,flows=flows,observations=r.global_decisions))
        keys=('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256')
        signatures=[data['probe'][k] for k in keys]+[data['orders_sha256'],data['global_audit']['rng_state_sha256']]
        if reference is None:reference=signatures
        else:assert reference==signatures
        if mode=='off':continue
        audit=verify_snapshots(data,flows,r.global_decisions)
        corpus=r.samples[:256];assert len(corpus)==256
        for s in corpus:assert encode_sample(decode_sample(encode_sample(s)))==encode_sample(s)
        if reference_samples is None:reference_samples=corpus
        else:
            assert [encode_sample(condition_sample(s,'zero')) for s in corpus]==[encode_sample(s) for s in reference_samples]
        with torch.no_grad():
            for s in corpus:
                from dataclasses import replace
                a,b=old(pack_states([replace(s.state,global_waiting=None)])),net(pack_states([s.state]))
                assert torch.equal(a[0],b[0]) and torch.allclose(a[1],b[1],atol=2e-5)
        optimization,diagnostic=fit_once(net,torch.optim.Adam(net.parameters(),lr=.0003),corpus,
            [c]*len(corpus),seed=10089001,epochs=1)
        results.append(dict(mode=mode,seconds=data['seconds'],audit=audit,optimization=optimization,
            rmse_before=diagnostic['before']['rmse'],rmse_after=diagnostic['after']['rmse']))
    save(OUT/'smoke.json',dict(passed=True,case=c,three_initial_behaviors_exact=True,
        source_sample_roundtrip=512,updates=results,excluded_from_training=True))
    print('smoke: original/zero/global behavior exact; snapshots, ledger and both updates passed',flush=True)


def train(task,replay=False):
    import torch
    torch.set_num_threads(1);cfg=check();arm,seed=task;k=SEEDS.index(seed)
    root=model_path(arm,seed);root.mkdir(parents=True,exist_ok=True)
    if replay:net,opt,_=restore(root/'initial.pt')
    else:
        assert not (root/'initial.pt').exists()
        net,opt=fresh(seed)
        checkpoint(root/'initial.pt',net,opt,arm=arm,seed=seed,iteration=-1)
    history=[];started=time.perf_counter()
    for it in range(cfg['updates']):
        begin=time.perf_counter();days=[]
        if replay:
            corpus=gz_read(root/f'samples-{it}.json.gz')
            samples=[decode_sample(s) for s in corpus['samples']];metadata=corpus['metadata']
        else:
            all_samples=[];all_meta=[]
            for j in (2*it,2*it+1):
                c=cfg['train'][j]
                data,orders,flows,r=rollout(c,net,mode=arm,stochastic=True,collect=True,
                    action_seed=cfg['seed_bases']['action']+100*k+j)
                path=root/f'day-{j}.json.gz'
                gz_save(path,dict(data=data,orders=orders,flows=flows,observations=r.global_decisions))
                all_samples.extend(r.samples);all_meta.extend([c]*len(r.samples))
                days.append(dict(case=c,path=path.as_posix(),sha256=sha(path),samples=len(r.samples),
                    final_cost=data['result']['phi_krw'],seconds=data['seconds']))
                print(f'train {arm}/{seed} day {j+1}/12: {len(r.samples)} decisions, {data["seconds"]:.1f}s',flush=True)
            rng=torch.Generator().manual_seed(cfg['seed_bases']['sample']+100*k+it)
            indices=torch.randperm(len(all_samples),generator=rng)[:cfg['max_samples']].tolist()
            samples=[all_samples[i] for i in indices];metadata=[all_meta[i] for i in indices]
            assert len(samples)==2048
            gz_save(root/f'samples-{it}.json.gz',dict(samples=[encode_sample(s) for s in samples],metadata=metadata,
                indices=indices,source_decisions=len(all_samples),days=days))
        result,diagnostic=fit_once(net,opt,samples,metadata,seed=cfg['seed_bases']['update']+100*k+it,
                                   epochs=cfg['epochs'],batch_size=cfg['batch_size'])
        assert result['selected']==2048 and result['optimizer_steps']==128
        if replay:
            expected,expected_opt,_=restore(root/f'update-{it}.pt')
            assert_nested_equal(net.state_dict(),expected.state_dict());assert_nested_equal(opt.state_dict(),expected_opt.state_dict())
            assert diagnostic==gz_read(root/f'fit-diagnostic-{it}.json.gz')
        else:
            checkpoint(root/f'update-{it}.pt',net,opt,arm=arm,seed=seed,iteration=it)
            gz_save(root/f'fit-diagnostic-{it}.json.gz',diagnostic)
            history.append(dict(iteration=it,days=days,optimization=result,
                samples_sha256=sha(root/f'samples-{it}.json.gz'),checkpoint_sha256=sha(root/f'update-{it}.pt'),
                rmse_before=diagnostic['before']['rmse'],rmse_after=diagnostic['after']['rmse'],seconds=time.perf_counter()-begin))
            save(root/'training.json',dict(arm=arm,seed=seed,history=history,seconds=time.perf_counter()-started))
        print(f'fit {arm}/{seed} {it+1}/6: prediction {diagnostic["before"]["rmse"]:.4f}->{diagnostic["after"]["rmse"]:.4f}, replay={replay}',flush=True)
    if replay:save(root/'replay.json',dict(passed=True,updates=6,weights_exact=True,optimizer_exact=True,diagnostics_exact=True))
    else:checkpoint(root/'final.pt',net,opt,arm=arm,seed=seed,iteration=5)


def parallel(fn,tasks,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):f.result();print(f'completed {i}/{len(futures)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','train','replay'));p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();assert 1<=a.workers<=6
    if a.mode=='smoke':smoke()
    else:parallel(partial(train,replay=a.mode=='replay'),[(x,s) for x in ARMS for s in SEEDS],a.workers)
