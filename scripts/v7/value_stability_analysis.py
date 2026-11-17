"""Read-only independent ledger/target audit and fixed paired aggregation."""
from collections import defaultdict
import math
from pathlib import Path
import statistics as st

from resched_ppo import decode_sample
from report_synthetic_pipeline import audit, FLOW_NAMES
from six_step_learning import gz_read, sha
from value_stability import check, folder, restore, assert_nested_equal
from value_stability_eval import cases
from value_stability_inputs import OUT, ARMS, SEEDS, LOADS, SCALE, read, save


def combined(rows):
    assert rows
    flows={}
    for flow in FLOW_NAMES:
        groups=[r['completion']['at24'][flow] for r in rows]
        n=sum(g['n'] for g in groups)
        flows[flow]=dict(mean_min=st.mean(g['mean_s']/60 for g in groups),
            order_weighted_min=sum(g['total_s'] for g in groups)/n/60,
            unfinished24_mean=st.mean(g['unfinished'] for g in groups),
            unfinished26_mean=st.mean(r['completion']['at26'][flow]['unfinished'] for r in rows),
            orders=n)
    return dict(cases=len(rows),cost=st.mean(r['result']['phi_krw'] for r in rows),
        components={k:st.mean(r['result'][k] for r in rows) for k in ('c_wait','c_vessel','c_move','c_rehandle')},
        flows=flows,unfinished26_mean=st.mean(r['completion']['at26']['ALL_ORDERS']['unfinished'] for r in rows))


def change(a,b):
    return dict(cost_pct=100*(b['cost']/a['cost']-1),
        flow_pct={f:100*(b['flows'][f]['mean_min']/a['flows'][f]['mean_min']-1) for f in FLOW_NAMES},
        unfinished26_delta=b['unfinished26_mean']-a['unfinished26_mean'],
        vessel_cost_pct=100*(b['components']['c_vessel']/a['components']['c_vessel']-1))


def training_audit():
    import torch
    from ppo_credit_metrics import normalized_advantages
    rows=[];artifacts={};cells=defaultdict(lambda:dict(n=0,positive=0,raw_positive=0,sq_before=0.,sq_after=0.))
    initial_states={};first_corpora={}
    for arm in ARMS:
        for seed in SEEDS:
            root=folder(arm,seed);history=read(root/'training.json')['history']
            assert len(history)==6
            initial,_,_=restore(root/'initial.pt')
            initial_states[arm,seed]=initial.state_dict()
            seen=[]
            for h in history:
                it=h['iteration'];path=root/f'samples-{it}.json.gz';corpus=gz_read(path)
                assert sha(path)==h['samples_sha256']
                assert sha(root/f'update-{it}.pt')==h['checkpoint_sha256']
                assert h['optimization']['selected']==2048 and h['optimization']['optimizer_steps']==128
                assert len(set(corpus['indices']))==2048
                assert corpus['source_decisions']==sum(d['accounting']['samples'] for d in h['days'])
                if it==0:first_corpora[arm,seed]=corpus
                diagnostic=gz_read(root/f'fit-diagnostic-{it}.json.gz')
                samples=[decode_sample(s) for s in corpus['samples']]
                advantage,_,_=normalized_advantages(samples,seed+3000+it)
                boundary=h['days'][0]['accounting']['samples']
                for i,(ix,s,m) in enumerate(zip(corpus['indices'],samples,corpus['metadata'])):
                    day=h['days'][int(ix>=boundary)]
                    assert m==day['case']
                    expected=-(day['result']['phi_krw']-s.past_cost)/SCALE
                    assert s.target==expected
                    eligible=[t for t,c in day['cost_epochs'] if c==s.past_cost]
                    assert eligible and min(eligible)<=s.state.time_fraction*86400+1e-6
                    cell=cells[arm,m['load'],int(s.state.time_fraction*24)//6]
                    cell['n']+=1;cell['positive']+=int(advantage[i]>0)
                    cell['raw_positive']+=int(s.target-s.old_value>0)
                    for version in ('before','after'):
                        cell['sq_'+version]+=(s.target-diagnostic[version]['values'][i])**2
                seen.extend(torch.tensor([s.target for s in samples]).double().tolist())
                net,_,obj=restore(root/f'update-{it}.pt')
                assert obj['iteration']==it and obj['return_scale']==SCALE
                if arm=='popart':
                    expected=torch.tensor(seen,dtype=torch.float64)
                    assert net.target_count.item()==len(seen)
                    assert abs(net.target_mean.item()-expected.mean().item())<1e-10
                    assert abs(net.target_scale.item()-expected.std(unbiased=False).clamp_min(1e-4).item())<1e-10
                else:assert net.target_count.item()==0 and net.target_scale.item()==1
                rows.append(dict(arm=arm,seed=seed,iteration=it,before_rmse=diagnostic['before']['rmse'],
                    after_rmse=diagnostic['after']['rmse'],optimization=h['optimization'],
                    collected=corpus['source_decisions'],seconds=h['seconds']))
            replay=read(root/'replay.json');assert replay['passed'] and replay['optimizer_exact'] and replay['updates']==6
            last,last_opt,_=restore(root/'update-5.pt')
            final,final_opt,final_obj=restore(root/'final.pt')
            assert final_obj['iteration']==5
            assert_nested_equal(last.state_dict(),final.state_dict())
            assert_nested_equal(last_opt.state_dict(),final_opt.state_dict())
            for p in root.iterdir():artifacts[p.as_posix()]=sha(p)
    for seed in SEEDS:
        assert first_corpora['control',seed]==first_corpora['popart',seed]
        assert all(torch.equal(v,initial_states['popart',seed][k]) for k,v in initial_states['control',seed].items())
    cells=[dict(arm=a,load=l,hour_bin=h,**v,rmse_before=math.sqrt(v['sq_before']/v['n']),
                rmse_after=math.sqrt(v['sq_after']/v['n'])) for (a,l,h),v in sorted(cells.items())]
    return dict(rows=rows,signal_cells=cells,paired_initial_and_first_experience_exact=True,
        sample_targets_checked=6*6*2048,normalization_training_only_verified=True,
        full_checkpoint_replay_updates=36),artifacts


def main():
    import torch
    torch.set_num_threads(1);cfg=check();rows,artifacts=audit(OUT)
    assert len(rows)==126 and [r['case'] for r in rows]==cases()
    assert len(list((OUT/'evaluation').glob('*repeat.json.gz')))==2
    inputs=defaultdict(set);cohorts=defaultdict(set)
    for r in rows:
        c=r['case'];key=(c['split'],c['index'],c['load'],c['input_seed'])
        inputs[key].add(r['probe']['metadata']['input_sha256'])
        cohorts[key].add(r['probe']['flow_timing']['cohort_sha256'])
        if c['arm'] in ARMS:
            cp=folder(c['arm'],c['seed'])/f'{c["stage"]}.pt'
            assert r['checkpoint_sha256']==sha(cp)
        assert r['visits']['late_releases']==r['visits']['late_exits']==0
    assert all(len(x)==1 for x in inputs.values()) and all(len(x)==1 for x in cohorts.values())
    training,train_artifacts=training_audit();artifacts.update(train_artifacts)
    test=[r for r in rows if r['case']['split']=='holdout']
    groups={a:combined([r for r in test if r['case']['arm']==a]) for a in (*ARMS,'SF_SPT')}
    per_seed={}
    for seed in SEEDS:
        g={a:combined([r for r in test if r['case']['arm']==a and r['case']['seed']==seed]) for a in ARMS}
        per_seed[str(seed)]=dict(groups=g,change=change(g['control'],g['popart']))
    per_load={}
    for load in LOADS:
        g={a:combined([r for r in test if r['case']['arm']==a and r['case']['load']==load]) for a in (*ARMS,'SF_SPT')}
        per_load[str(load)]=dict(groups=g,change=change(g['control'],g['popart']))
    validation=[]
    for arm in ARMS:
        for stage in ('initial','update-2','final'):
            selected=[r for r in rows if r['case']['split']=='selection' and r['case']['arm']==arm and
                      r['case']['stage']==stage and r['case']['mode']=='argmax']
            if selected:validation.append(dict(arm=arm,stage=stage,**combined(selected)))
    value_validation=[]
    for arm in ARMS:
        for load in LOADS:
            v=[r['value_diagnostic'] for r in rows if r['case']['arm']==arm and
                r['case']['load']==load and r['case']['mode']=='stochastic']
            assert len(v)==3
            value_validation.append(dict(arm=arm,load=load,rmse=math.sqrt(st.mean(d['rmse']**2 for d in v)),
                by_seed=[dict(seed=s,**d) for s,d in zip(SEEDS,v)]))
    overall=change(groups['control'],groups['popart'])
    consistent=(all(r['change']['cost_pct']<0 for r in per_seed.values()) and
        overall['flow_pct']['TRUCK']<=0 and overall['unfinished26_delta']<=0 and overall['vessel_cost_pct']<=0)
    rule_delta=change(groups['SF_SPT'],groups['popart'])
    summary=dict(experiment='YR-332-ay',source_commit=read(OUT/'campaign.json')['commit'],
        evaluation_cases=126,repeats=2,test_input_cases=9,model_pairs=3,groups=groups,
        paired_change=overall,vs_rule=rule_delta,by_seed=per_seed,by_load=per_load,
        validation_curve=validation,value_validation=value_validation,training=training,
        same_input_and_cohort=True,consistent_improvement_candidate=consistent,
        adoption_performance_pass=consistent and rule_delta['cost_pct']<0 and
            rule_delta['flow_pct']['TRUCK']<0 and groups['popart']['unfinished26_mean']==0,
        reliability_pass=True,realism_pass=False,default_policy_changed=False,
        averaging='equal weight per input day and model; order-weighted secondary also reported',
        artifact_hashes=artifacts,scope='bounded synthetic diagnostic; no real-terminal or statistical confirmation')
    save(OUT/'summary.json',summary)
    print({k:summary[k] for k in ('paired_change','vs_rule','consistent_improvement_candidate','adoption_performance_pass')},flush=True)


if __name__=='__main__':main()
