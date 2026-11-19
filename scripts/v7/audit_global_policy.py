"""Independent full-ledger and stored-experience audit for the bc campaign."""
from collections import defaultdict
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from audit_global_value import verify_ledger,verify_snapshots
from global_policy_inputs import OUT,SEEDS,ARMS,SCALE,read,save,checkpoint_path
from global_policy_plan import check,model_path,eval_cases,repeat_indices
from global_value_samples import decode_sample,encode_sample,condition_sample
from six_step_learning import sha,gz_read
from value_stability import restore,assert_nested_equal
from yard_rl.v7.evaluation_probe import digest


def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),(a,b)


def sample_targets(samples,data):
    costs=defaultdict(list)
    for t,cost in data['cost_epochs']:costs[cost].append(t)
    assert all(a[0]<=b[0] and a[1]<=b[1]+1e-6 for a,b in zip(data['cost_epochs'],data['cost_epochs'][1:]))
    close(data['cost_epochs'][-1][1],data['result']['phi_krw'])
    for s in samples:
        t=s.state.time_fraction*86400.
        assert t<86400 and s.past_cost in costs and min(costs[s.past_cost])<=t+1e-6
        assert s.target==-(data['result']['phi_krw']-s.past_cost)/SCALE
        assert s.global_observation['decision_time']==t or abs(s.global_observation['decision_time']-t)<1e-6


def audit_training(cfg):
    import torch
    rows=[];first={};initials={};snapshots=observations=orders=0;signals=[];weights=[]
    for arm in ARMS:
        for seed in SEEDS:
            root=model_path(arm,seed);h=read(root/'training.json')['history'];assert len(h)==6
            net,initial_opt,obj=restore(root/'initial.pt');initials[arm,seed]=net.state_dict()
            old,_,_=restore(checkpoint_path(seed))
            for name,value in old.state_dict().items():
                expected=net.state_dict()[name]
                if name=='critic.0.weight':
                    assert torch.count_nonzero(expected[:,-1])==0;expected=expected[:,:-1]
                assert torch.equal(value,expected),name
            assert initial_opt.state_dict()['state']=={}
            count,mean,m2=net.target_count.item(),net.target_mean.item(),net.target_m2.item()
            for it,entry in enumerate(h):
                assert entry['iteration']==it
                corpus=gz_read(root/f'samples-{it}.json.gz')
                assert sha(root/f'samples-{it}.json.gz')==entry['samples_sha256']
                assert sha(root/f'update-{it}.pt')==entry['checkpoint_sha256']
                samples=[decode_sample(s) for s in corpus['samples']]
                assert len(samples)==len(set(corpus['indices']))==2048
                assert entry['optimization']['selected']==2048 and entry['optimization']['optimizer_steps']==128
                assert corpus['days']==entry['days']
                days=[]
                for j,source in enumerate(entry['days']):
                    assert source['case']==cfg['train'][2*it+j] and sha(source['path'])==source['sha256']
                    raw=gz_read(source['path']);data=raw['data'];verify_ledger(data,raw['orders'],raw['flows'])
                    counts=verify_snapshots(data,raw['flows'],raw['observations'])
                    snapshots+=counts['block_snapshots_checked'];observations+=counts['observations'];orders+=len(raw['orders'])
                    assert data['accounting']['samples']==source['samples']
                    assert source['final_cost']==data['result']['phi_krw']
                    days.append(raw)
                boundary=entry['days'][0]['samples']
                assert corpus['source_decisions']==sum(d['samples'] for d in entry['days'])
                for j in (0,1):
                    selected=[s for s,ix in zip(samples,corpus['indices']) if int(ix>=boundary)==j]
                    sample_targets(selected,days[j]['data'])
                for s,ix,meta in zip(samples,corpus['indices'],corpus['metadata']):
                    assert 0<=ix<corpus['source_decisions'] and meta==entry['days'][int(ix>=boundary)]['case']
                    assert s.state.global_waiting==(0. if arm=='zero' else s.global_observation['feature'])
                diagnostic=gz_read(root/f'fit-diagnostic-{it}.json.gz')
                for version in ('before','after'):
                    values=diagnostic[version]['values']
                    close(math.sqrt(math.fsum((s.target-v)**2 for s,v in zip(samples,values))/len(samples)),diagnostic[version]['rmse'])
                    for cell in diagnostic[version]['by_load_time']:
                        ix=[i for i,(s,m) in enumerate(zip(samples,corpus['metadata']))
                            if m['load']==cell['load'] and int(s.state.time_fraction*24)//6==cell['hour_bin']]
                        close(math.sqrt(math.fsum((samples[i].target-values[i])**2 for i in ix)/len(ix)),cell['rmse'])
                        signals.append(dict(arm=arm,seed=seed,iteration=it,version=version,**cell))
                assert max(abs(s.old_log_probability-v) for s,v in zip(samples,diagnostic['before']['action_logps']))<2e-5
                y=torch.tensor([s.target for s in samples]).double();n=len(y);delta=y.mean().item()-mean
                total=count+n;m2+=(y-y.mean()).square().sum().item()+delta*delta*count*n/total
                mean+=delta*n/total;count=total
                net,_,_=restore(root/f'update-{it}.pt')
                close(net.target_count.item(),count);close(net.target_mean.item(),mean)
                close(net.target_m2.item(),m2);close(net.target_scale.item(),max(1e-4,math.sqrt(m2/count)))
                if it==0:first[arm,seed]=corpus
                rows.append(dict(arm=arm,seed=seed,iteration=it,rmse_before=entry['rmse_before'],rmse_after=entry['rmse_after'],
                    seconds=entry['seconds'],source_decisions=corpus['source_decisions']))
            last,last_opt,_=restore(root/'update-5.pt');final,final_opt,obj=restore(root/'final.pt')
            assert obj['iteration']==5
            assert_nested_equal(last.state_dict(),final.state_dict());assert_nested_equal(last_opt.state_dict(),final_opt.state_dict())
            column=final.state_dict()['critic.0.weight'][:,-1]
            if arm=='zero':assert torch.count_nonzero(column)==0
            weights.append(dict(arm=arm,seed=seed,context_weight_norm=float(column.norm()),
                context_nonzero_weights=int(torch.count_nonzero(column))))
            replay=read(root/'replay.json');assert replay['passed'] and replay['updates']==6
    for seed in SEEDS:
        assert_nested_equal(initials['zero',seed],initials['global',seed])
        a,b=first['zero',seed],first['global',seed]
        assert a['indices']==b['indices'] and a['source_decisions']==b['source_decisions']
        assert a['samples']==[encode_sample(condition_sample(decode_sample(s),'zero')) for s in b['samples']]
        for da,db in zip(a['days'],b['days']):
            aa,bb=(gz_read(d['path'])['data'] for d in (da,db))
            for key in ('choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
                assert aa['probe'][key]==bb['probe'][key]
    return dict(passed=True,updates=len(rows),targets=73728,days=72,orders=orders,
        block_snapshots=snapshots,observations=observations,rows=rows,signal_cells=signals,
        paired_initial_and_first_two_days_exact=True,weights_optimizer_replay_updates=36,context_weights=weights)


def audit_evaluations(cfg):
    records=[];inputs=defaultdict(set);cohorts=defaultdict(set);orders=snapshots=observations=0
    paths=sorted((OUT/'evaluation').glob('*.json.gz'));assert len(paths)==182
    for path in paths:
        d=gz_read(path);i=int(path.name.split('.')[0].split('-')[0]);c=cfg['evaluation_cases'][i]
        assert d['case']==c
        truck=gz_read(OUT/'orders'/path.name);flows=gz_read(OUT/'flow-orders'/path.name)
        verify_ledger(d,truck,flows);orders+=len(truck)
        if c['arm'] in ARMS:
            assert d['checkpoint_sha256']==sha(model_path(c['arm'],c['seed'])/f'{c["stage"]}.pt')
            obs=gz_read(OUT/'observations'/path.name);z=verify_snapshots(d,flows,obs)
            snapshots+=z['block_snapshots_checked'];observations+=z['observations']
        if c['mode']=='stochastic':
            corpus=gz_read(OUT/'validation-samples'/path.name);samples=[decode_sample(s) for s in corpus['samples']]
            assert len(samples)==2048;sample_targets(samples,d)
            diag=corpus['diagnostic'];close(math.sqrt(math.fsum((s.target-v)**2 for s,v in zip(samples,diag['values']))/len(samples)),diag['rmse'])
            assert {k:v for k,v in diag.items() if k not in ('values','action_logps')}==d['value_diagnostic']
        key=(c['split'],c['index'],c['load'],c['input_seed'])
        inputs[key].add(d['probe']['metadata']['input_sha256']);cohorts[key].add(d['probe']['flow_timing']['cohort_sha256'])
        assert d['visits']['late_releases']==d['visits']['late_exits']==0
        if '-repeat' in path.name:
            assert i in cfg['repeat_indices'];prior=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz')
            for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
                assert d['probe'][key]==prior['probe'][key]
            assert d['orders_sha256']==prior['orders_sha256']
        else:records.append(d)
    assert [r['case'] for r in records]==eval_cases() and all(len(v)==1 for v in inputs.values()) and all(len(v)==1 for v in cohorts.values())
    return records,dict(passed=True,cases=180,repeats=2,orders=orders,block_snapshots=snapshots,observations=observations,
        matched_inputs=len(inputs),same_inputs_and_cohorts=True,stochastic_diagnostic_cases=27)


def main():
    import torch
    torch.set_num_threads(1);cfg=check();training=audit_training(cfg);records,evaluation=audit_evaluations(cfg)
    suites=ET.parse(OUT/'tests-initial.xml').getroot().findall('testsuite')
    assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
    tests=sum(int(s.get('tests')) for s in suites)
    save(OUT/'audit.json',dict(passed=True,training=training,evaluation=evaluation,tests=tests,
        source_files=len(read(OUT/'campaign.json')['sources']),scope='Full saved ledgers, source input pairing and exact stored-experience optimizer replay'))
    print('audit passed:',training['orders'],evaluation['orders'],'training/evaluation order rows',flush=True)


if __name__=='__main__':main()
