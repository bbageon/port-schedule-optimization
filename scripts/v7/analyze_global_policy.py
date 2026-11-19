"""Fixed day/model-weighted comparisons; no checkpoint or budget selection."""
from collections import defaultdict
import math
import statistics as st

import numpy as np

from audit_global_value import combined
from global_policy_inputs import OUT,SEEDS,ARMS,LOADS,read,save
from global_policy_plan import check
from six_step_learning import gz_read


def pct(a,b):return 100*(b/a-1) if a else (0. if not b else None)


def change(a,b):
    return dict(cost_pct=pct(a['cost'],b['cost']),vessel_cost_pct=pct(a['vessel_cost'],b['vessel_cost']),
        flow_pct={f:pct(a['flows'][f]['mean_min'],b['flows'][f]['mean_min']) for f in a['flows']},
        unfinished26_delta=b['unfinished26_mean']-a['unfinished26_mean'])


def group(rows,label):
    arm,stage=('zero','initial') if label=='initial' else (label,'rule' if label=='SF_SPT' else 'final')
    return [r for r in rows if r['case']['arm']==arm and r['case']['stage']==stage and r['case']['mode']=='argmax']


def uncertainty(rows,baseline,candidate):
    def grid(label):
        selected=group(rows,label)
        return np.array([[st.mean(r['result']['phi_krw'] for r in selected if r['case']['index']==day and
                         (label=='SF_SPT' or r['case']['seed']==seed)) for seed in SEEDS] for day in range(3)])
    a,b=grid(baseline),grid(candidate);rng=np.random.default_rng(10089900)
    diffs=[];ratios=[]
    for _ in range(10000):
        days=rng.integers(0,3,3);models=rng.integers(0,3,3)
        x,y=a[np.ix_(days,models)].mean(),b[np.ix_(days,models)].mean()
        diffs.append(float(y-x));ratios.append(float(100*(y/x-1)))
    return dict(baseline=baseline,candidate=candidate,paired_cost_matrix=(b-a).tolist(),
        cost_difference=float((b-a).mean()),cost_difference_ci95=np.quantile(diffs,[.025,.975]).tolist(),
        change_pct=pct(float(a.mean()),float(b.mean())),change_pct_ci95=np.quantile(ratios,[.025,.975]).tolist(),
        repeats=10000,seed=10089900,method='crossed resampling of three synthetic curves and three model seeds; loads averaged first')


def prediction_groups(rows):
    output=[]
    for label in ('initial','zero','global'):
        selected=[r for r in rows if r['case']['mode']=='stochastic' and r['case']['arm']==('zero' if label=='initial' else label)
                  and r['case']['stage']==('initial' if label=='initial' else 'final')]
        assert len(selected)==9
        groups={}
        for key,condition in [('all',lambda r:True),*[(str(l),lambda r,l=l:r['case']['load']==l) for l in LOADS]]:
            cells=[r for r in selected if condition(r)]
            by_time={}
            for hour in range(4):
                errors=[c['rmse'] for r in cells for c in r['value_diagnostic']['by_load_time'] if c['hour_bin']==hour]
                by_time[str(hour)]=math.sqrt(st.mean(x*x for x in errors)) if errors else None
            groups[key]=dict(rmse=math.sqrt(st.mean(r['value_diagnostic']['rmse']**2 for r in cells)),
                by_time=by_time,samples=sum(len(gz_read(OUT/'validation-samples'/f'{r["evaluation_index"]:03d}.json.gz')['samples']) for r in cells))
        output.append(dict(label=label,groups=groups))
    return output


def main():
    cfg=check();audit=read(OUT/'audit.json');assert audit['passed']
    rows=[]
    for i,c in enumerate(cfg['evaluation_cases']):
        r=gz_read(OUT/'evaluation'/f'{i:03d}.json.gz');assert r['case']==c;r['evaluation_index']=i;rows.append(r)
    test=[r for r in rows if r['case']['split']=='holdout']
    labels=('initial','zero','global','SF_SPT')
    groups={a:combined(group(test,a)) for a in labels}
    by_seed={}
    for seed in SEEDS:
        chosen=[r for r in test if r['case']['seed'] in (0,seed)]
        g={a:combined(group(chosen,a)) for a in labels}
        by_seed[str(seed)]=dict(groups=g,paired_change=change(g['zero'],g['global']))
    by_load={}
    for load in LOADS:
        chosen=[r for r in test if r['case']['load']==load]
        g={a:combined(group(chosen,a)) for a in labels}
        by_load[str(load)]=dict(groups=g,paired_change=change(g['zero'],g['global']))
    development=[]
    for arm in ARMS:
        for stage in ('initial','update-1','update-3','final'):
            chosen=[r for r in rows if r['case']['split']=='selection' and r['case']['mode']=='argmax' and
                    r['case']['arm']==('zero' if stage=='initial' else arm) and r['case']['stage']==stage]
            assert len(chosen)==9
            development.append(dict(arm=arm,stage=stage,**combined(chosen)))
    paired=change(groups['zero'],groups['global']);rule=change(groups['SF_SPT'],groups['global'])
    conditions=dict(all_model_pairs_lower_cost=all(v['paired_change']['cost_pct']<0 for v in by_seed.values()),
        truck_no_worse_than_zero=paired['flow_pct']['TRUCK']<=0,
        outbound_no_worse_than_zero=paired['flow_pct']['GATE_OUT']<=0,
        vessel_time_no_worse_than_zero=paired['flow_pct']['VESSEL']<=0,
        vessel_no_worse_than_zero=paired['vessel_cost_pct'] is not None and paired['vessel_cost_pct']<=0,
        unfinished_no_worse_than_zero=paired['unfinished26_delta']<=0,
        lower_cost_than_rule=rule['cost_pct']<0,lower_truck_time_than_rule=rule['flow_pct']['TRUCK']<0,
        lower_vessel_time_than_rule=rule['flow_pct']['VESSEL']<0,
        all_final_jobs_complete=all(r['completion']['at26']['ALL_ORDERS']['unfinished']==0 for r in group(test,'global')))
    consistent=all(v for k,v in conditions.items() if not k.endswith('_than_rule') and k!='all_final_jobs_complete')
    save(OUT/'summary.json',dict(task='YR-332-bc',source_commit=read(OUT/'campaign.json')['commit'],
        evaluations=len(rows),repeats=2,model_pairs=3,test_inputs=9,groups=groups,paired_change=paired,vs_rule=rule,
        vs_initial={a:change(groups['initial'],groups[a]) for a in ARMS},by_seed=by_seed,by_load=by_load,
        development_curve=development,prediction=prediction_groups(rows),
        uncertainty={name:uncertainty(test,a,b) for name,a,b in [('global_vs_zero','zero','global'),
            ('global_vs_rule','SF_SPT','global'),('zero_vs_rule','SF_SPT','zero')]},
        conditions=conditions,consistent_improvement_candidate=consistent,adoption_numerical_pass=all(conditions.values()),
        source_calibration='bb source-confirmed common demand and reservation information',
        default_policy_changed=False,realism_pass=False,prediction_error_mode='report_only',
        aggregation='equal synthetic curve, load and model weights; stochastic diagnostics separate',
        limitations=['three synthetic test curves and three starting models','shared thirteen-date calibration',
                    'external transport/queue/high-volume evidence unresolved','training optimizer replay uses stored experience']))
    print(dict(paired=paired,vs_rule=rule,conditions=conditions),flush=True)


if __name__=='__main__':main()
