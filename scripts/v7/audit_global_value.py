"""Independently recount elapsed times, snapshot queues and paired prediction errors."""
from bisect import bisect_left, bisect_right
from collections import defaultdict, Counter
import hashlib
import math
from pathlib import Path
import statistics as st
import subprocess

from global_value_inputs import OUT, SEEDS, ARMS, SCALE, read, save, case_name
from global_value_experiment import check, eval_cases
from global_value_samples import decode_sample
from six_step_learning import sha, gz_read
from horizon_metrics import aggregate
from verify_job_completion import raw_metrics, assert_metrics
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.crane.global_context import QueueSnapshot, validate_observation


def verify_ledger(data, orders, flows):
    assert digest(orders)==data['orders_sha256']
    metrics=raw_metrics(orders);assert metrics==data['metrics'];assert_metrics(metrics,data['result'])
    assert all(aggregate(flows,h*3600.)==data['completion'][f'at{h}'] for h in (24,26))
    assert data['result']['admitted']==data['case']['load']
    assert len({r['job_id'] for r in flows})==len(flows)
    assert data['result']['policy_exceptions']==data['result']['traded_edges']==data['result']['rollout_worlds']==0
    for r in flows:
        values=[r[k] for k in ('origin_s','ready_s','service_start_s','yard_done_s') if r[k] is not None]
        assert all(a<=b+1e-6 for a,b in zip(values,values[1:]))


def verify_snapshots(data, flows, observations):
    by_block=defaultdict(list)
    for r in flows:
        if r['flow'].startswith('GATE_'):by_block[r['block']].append(r)
    arrivals={b:sorted(r['ready_s'] for r in rows if r['ready_s'] is not None) for b,rows in by_block.items()}
    starts={b:sorted(r['service_start_s'] for r in rows if r['service_start_s'] is not None) for b,rows in by_block.items()}
    immediate={b:Counter(r['ready_s'] for r in rows if r['ready_s'] is not None and r['ready_s']==r['service_start_s'])
               for b,rows in by_block.items()}
    snapshot_keys=set();times=[];checked=ties=0
    for o in data['global_audit']['snapshots']:
        s=QueueSnapshot(o['time'],tuple(tuple(x) for x in o['blocks']),o['total'])
        assert len(s.blocks)==21
        times.append(s.time);snapshot_keys.add((s.time,s.blocks,s.total))
        for b,count in s.blocks:
            a,z=arrivals.get(b,[]),starts.get(b,[]);t=s.time
            # Review and local events can share a timestamp. Count the possible
            # pre/post-event range, retaining the engine's actual tie order.
            lower=bisect_left(a,t)-bisect_right(z,t)+immediate.get(b,{}).get(t,0)
            upper=bisect_right(a,t)-bisect_left(z,t)
            assert lower<=count<=upper,(data['case'],b,t,count,lower,upper)
            checked+=1;ties+=int(lower!=upper)
    assert times==sorted(times) and times[0]==0 and times[-1]==86400.
    for o in observations:
        s=validate_observation(o)
        assert (s.time,s.blocks,s.total) in snapshot_keys
        assert s.time==times[bisect_right(times,o['decision_time'])-1]
        if o['decision_time']<86400:assert o['age_s']<=60.+1e-6
    assert digest(observations)==data['global_audit']['observation_digest']
    return dict(block_snapshots_checked=checked,timestamp_tie_cases=ties,observations=len(observations))


def independent_rmse(predictions):
    cells=defaultdict(list)
    for p,y,m in zip(predictions['prediction'],predictions['target'],predictions['metadata']):
        if m['split']=='selection':cells[m['case']].append((p-y)**2)
    return math.sqrt(st.mean(math.fsum(v)/len(v) for v in cells.values()))


def combined(rows):
    flows={}
    for f in ('TRUCK','VESSEL','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'):
        cells=[r['completion']['at24'][f] for r in rows]
        flows[f]=dict(mean_min=st.mean(c['mean_s']/60 for c in cells),
            order_weighted_min=sum(c['total_s'] for c in cells)/sum(c['n'] for c in cells)/60,
            unfinished24_mean=st.mean(c['unfinished'] for c in cells),
            unfinished26_mean=st.mean(r['completion']['at26'][f]['unfinished'] for r in rows))
    return dict(cases=len(rows),cost=st.mean(r['result']['phi_krw'] for r in rows),flows=flows,
        vessel_cost=st.mean(r['result']['c_vessel'] for r in rows),
        unfinished26_mean=st.mean(r['completion']['at26']['ALL_ORDERS']['unfinished'] for r in rows))


def change(a,b):
    return dict(cost_pct=100*(b['cost']/a['cost']-1),vessel_cost_pct=100*(b['vessel_cost']/a['vessel_cost']-1),
        flow_pct={f:100*(b['flows'][f]['mean_min']/a['flows'][f]['mean_min']-1) for f in a['flows']},
        unfinished26_delta=b['unfinished26_mean']-a['unfinished26_mean'])


def main():
    import torch
    import xml.etree.ElementTree as ET
    cfg=check();prediction=read(OUT/'prediction-summary.json')
    tests=ET.parse(OUT/'unit-tests.xml').getroot().find('testsuite')
    assert int(tests.attrib['failures'])==int(tests.attrib['errors'])==0
    counts=Counter();ages=[];samples=0;orders_n=0
    common_cases=cfg['train']+cfg['validation']
    for c in common_cases:
        n=case_name(c);obj=gz_read(OUT/'common'/f'{n}.json.gz');data=obj['data']
        orders=gz_read(OUT/'common-orders'/f'{n}.json.gz');flows=gz_read(OUT/'common-flow-orders'/f'{n}.json.gz')
        observations=gz_read(OUT/'common-observations'/f'{n}.json.gz')
        verify_ledger(data,orders,flows);counts.update(verify_snapshots(data,flows,observations))
        assert len(obj['indices'])==len(set(obj['indices']))==len(obj['samples'])
        assert all(0<=i<obj['available'] for i in obj['indices'])
        for raw in obj['samples']:
            s=decode_sample(raw)
            assert s.state.global_waiting is None
            assert math.isclose(s.target,-(data['result']['phi_krw']-s.past_cost)/SCALE,rel_tol=1e-12,abs_tol=1e-12)
        samples+=len(obj['samples']);orders_n+=len(orders)
        ages.extend((o['decision_time'],o['age_s'],o['total'],o['feature']) for o in observations)
    repeat=case_name(cfg['validation'][4])
    a,b=[gz_read(OUT/'common'/f'{repeat}{suffix}.json.gz') for suffix in ('','-repeat')]
    assert a['samples']==b['samples'] and a['indices']==b['indices']
    assert all(a['data']['probe'][k]==b['data']['probe'][k] for k in
               ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'))
    rmse={a:[] for a in ARMS};max_error=0.
    for arm in ARMS:
        for seed in SEEDS:
            root=OUT/'prediction'/f'{arm}-{seed}';r=read(root/'result.json')
            assert read(root/'replay.json')['passed']
            assert sha(root/'final.pt')==r['checkpoint_sha256']
            p=gz_read(root/'predictions.json.gz');value=independent_rmse(p)
            reported=r['final']['validation']['rmse']
            assert math.isclose(value,reported,rel_tol=1e-6,abs_tol=1e-8),(arm,seed,value,reported)
            max_error=max(max_error,abs(value-reported));rmse[arm].append(value)
            zero=gz_read(OUT/'prediction'/f'zero-{seed}'/'predictions.json.gz')
            assert zero['target']==p['target'] and zero['metadata']==p['metadata']
    independently_pooled={a:math.sqrt(st.mean(x*x for x in v)) for a,v in rmse.items()}
    for a,v in independently_pooled.items():assert math.isclose(v,prediction['groups'][a]['rmse'],rel_tol=1e-6)
    policy=None
    if prediction['passed']:
        from report_synthetic_pipeline import audit
        rows,artifacts=audit(OUT);assert len(rows)==len(eval_cases())
        assert [r['case'] for r in rows]==eval_cases()
        for r in rows:
            if r['case']['arm'] in ARMS:
                i=eval_cases().index(r['case']);name=f'{i:03d}.json.gz'
                flows=gz_read(OUT/'flow-orders'/name);obs=gz_read(OUT/'observations'/name)
                counts.update(verify_snapshots(r,flows,obs))
        hold=[r for r in rows if r['case']['split']=='holdout']
        groups={a:combined([r for r in hold if r['case']['arm']==a]) for a in (*ARMS,'SF_SPT')}
        per_seed={str(seed):change(*[combined([r for r in hold if r['case']['arm']==a and r['case']['seed']==seed])
                                  for a in ARMS]) for seed in SEEDS}
        paired=change(groups['zero'],groups['global']);vs_rule=change(groups['SF_SPT'],groups['global'])
        conditions=dict(all_seed_costs_lower=all(x['cost_pct']<0 for x in per_seed.values()),
            truck=paired['flow_pct']['TRUCK']<=0,outbound=paired['flow_pct']['GATE_OUT']<=0,
            vessel_mean=paired['flow_pct']['VESSEL']<=0,vessel_cost=paired['vessel_cost_pct']<=0,
            unfinished=paired['unfinished26_delta']<=0)
        for a in ARMS:
            for seed in SEEDS:assert read(OUT/'policy'/f'{a}-{seed}'/'replay.json')['passed']
        repeats=list((OUT/'evaluation').glob('*repeat.json.gz'));assert len(repeats)==2
        for path in repeats:
            a=gz_read(path);b=gz_read(path.with_name(path.name.replace('-repeat','')))
            for k in ('choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
                assert a['probe'][k]==b['probe'][k]
        policy=dict(groups=groups,paired_change=paired,vs_rule=vs_rule,by_seed=per_seed,
            conditions=conditions,consistent_candidate=all(conditions.values()),evaluations=len(rows),repeats=2,
            adoption_numerical_pass=all(conditions.values()) and vs_rule['cost_pct']<0 and
               vs_rule['flow_pct']['TRUCK']<0 and vs_rule['flow_pct']['VESSEL']<0 and groups['global']['unfinished26_mean']==0,
            validation=[r for r in rows if r['case']['split']=='selection'])
    else:
        assert not (OUT/'policy').exists() and not (OUT/'evaluation').exists()
    before=[a[1] for a in ages if a[0]<86400];after=[a[1] for a in ages if a[0]>=86400]
    def dist(values):
        v=sorted(values)
        return dict(n=len(v),mean=st.mean(v),p50=v[int((len(v)-1)*.5)],p95=v[int((len(v)-1)*.95)],max=max(v)) if v else dict(n=0)
    observation=dict(before24_age_s=dist(before),after24_age_s=dist(after),
        waiting_orders_range=[min(a[2] for a in ages),max(a[2] for a in ages)],
        feature_range=[min(a[3] for a in ages),max(a[3] for a in ages)])
    summary=dict(task='YR-332-az',source_commit=read(OUT/'campaign.json')['commit'],prediction=prediction,
        policy=policy,common_rollouts=len(common_cases),selected_common_samples=samples,
        common_truck_orders=orders_n,observation=observation,default_policy_changed=False,
        interpretation='Prediction on shared fixed-policy experience; PPO only if the preregistered prediction gate passes',
        reliability_pass=True,realism_pass=False)
    save(OUT/'summary.json',summary)
    artifacts={p.as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.log',)
               and p.name not in ('verification.json','publication.json','current-gates.json','dashboard-audit.json')}
    save(OUT/'verification.json',dict(passed=True,counts=dict(counts),common_samples_checked=samples,
        common_orders_checked=orders_n,independent_rmse=independently_pooled,max_rmse_recalculation_error=max_error,
        prediction_replays=6,unit_tests=int(tests.attrib['tests']),artifact_hashes=artifacts,
        code_sha256=sha(Path(__file__)),source_checks=len(read(OUT/'campaign.json')['sources'])))
    print(dict(prediction_passed=prediction['passed'],policy_evaluated=policy is not None,verified=True,counts=dict(counts)),flush=True)


if __name__=='__main__':main()
