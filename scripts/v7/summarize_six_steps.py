"""Recompute diagnostic summaries from hashed, retained experiment evidence."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st

from benchmark_default_eval import save


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            return json.load(f)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified(arm):
    path=Path(arm['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==arm['sha256'],path
    return read(path)


def candidate_summary(folder):
    stamp=read(Path(folder)/'campaign.json')
    stage=stamp['stage']
    mechanism=stamp.get('mechanism','slots')
    names=(('legacy','existence' if mechanism=='existence' else 'shared')
           if stage==1 else ('cap12','all'))
    pairs=[]
    seen={}
    repeat_checks=[]
    for row in stamp['rows']:
        a,b=[verified(row['arms'][name]) for name in names]
        for name,d in zip(names,(a,b)):
            key=(row['load'],row['seed'],name)
            hashes={k:v for k,v in d['probe'].items() if k.endswith('_sha256')}
            hashes['input']=d['probe']['metadata']['input_sha256']
            hashes['model']=d['model_sha256']
            if key in seen:
                assert hashes==seen[key],('repeat changed results',key)
                repeat_checks.append(dict(load=row['load'],seed=row['seed'],arm=name,equal=True))
            else:
                seen[key]=hashes
        assert row['equal']['input'] and row['equal']['model']
        if stage==1:
            assert all(row['equal'].values())
        p=dict(load=row['load'],seed=row['seed'],repeat=row['repeat'],equal=row['equal'],arms={})
        for name,d in zip(names,(a,b)):
            probe=d['probe']
            p['arms'][name]=dict(wall_seconds=d['wall_seconds'],run_seconds=probe['run_seconds'],
                generation_seconds=probe['generation']['total_s'],selection_seconds=probe['selection']['total_s'],
                p95_ms=probe['decision']['p95_ms'],max_ms=probe['decision']['max_ms'],
                peak_rss_bytes=d['memory_after']['peak_rss_bytes'],max_candidates=probe['max_candidates_per_crane'],
                outcomes=d['result'],flow_status=probe['flow_status_counts'],metadata=probe['metadata'])
            if 'recall' in probe:
                p['arms'][name]['recall']=probe['recall']
        p['wall_change_pct']=100*(b['wall_seconds']/a['wall_seconds']-1)
        pairs.append(p)
    loads={}
    for load in sorted({p['load'] for p in pairs}):
        primary=[p for p in pairs if p['load']==load and p['repeat']==0]
        out={}
        for name in names:
            vals=[p['arms'][name] for p in primary]
            out[name]={k:st.mean(v[k] for v in vals) for k in
                ('wall_seconds','run_seconds','generation_seconds','selection_seconds','p95_ms','peak_rss_bytes')}
            out[name]['max_ms']=max(v['max_ms'] for v in vals)
            out[name]['max_candidates']=max(v['max_candidates'] for v in vals)
        out['wall_change_pct']=100*(out[names[1]]['wall_seconds']/out[names[0]]['wall_seconds']-1)
        out['all_repeat_changes_pct']=[p['wall_change_pct'] for p in pairs if p['load']==load]
        loads[str(load)]=out
    result=dict(stage=stage,mechanism=mechanism,commit=stamp['commit'],n_pairs=len(pairs),
                loads=loads,pairs=pairs,repeat_checks=repeat_checks,
                interpretation='timing and candidate diagnostics; no trained-policy performance claim')
    save(Path(folder)/'summary.json',result)
    print(json.dumps(dict(stage=stage,mechanism=mechanism,n_pairs=len(pairs),loads=loads),indent=2))


def learning_summary(folder):
    root=Path(folder)
    data=[read(p) for p in sorted((root/'evaluation').glob('*.json.gz'))]
    by_key={(d['arm'],d['model_seed'],d['load']):d for d in data}
    comparisons=[]
    for d in data:
        if d['arm'].startswith('rule'):
            continue
        base=by_key.get(('rule_all',None,d['load']))
        if base is None:
            continue
        a,b=base['result'],d['result']
        vessel_done=lambda x:sum(n for k,n in x['probe']['flow_status_counts'].items()
                                if k.startswith('VESSEL_') and k.endswith('/DONE'))
        comparisons.append(dict(arm=d['arm'],model_seed=d['model_seed'],load=d['load'],seed=d['seed'],
            cost_change_pct=100*(b['phi_krw']/a['phi_krw']-1),
            turn_change_pct=100*(b['mean_turn_time_s']/a['mean_turn_time_s']-1),
            vessel_cost_change=b['c_vessel']-a['c_vessel'],
            admitted_change=b['admitted']-a['admitted'],censored_24h_change=b['n_censored']-a['n_censored'],
            completed_24h_change=(b['n_trucks']-b['n_censored'])-(a['n_trucks']-a['n_censored']),
            vessel_done_26h_change=vessel_done(d)-vessel_done(base)))
    stages=[]
    for seed_dir in sorted(root.glob('model-*')):
        for p in sorted(seed_dir.glob('stage-*.json')):
            run=read(p)
            corpora=[read(r['corpus']) for r in run['rows']]
            samples=[s for c in corpora for s in c['samples']]
            flows=Counter(s['flow'] for s in samples)
            cranes=Counter(f'{s["decision_key"][1]}/{s["crane"]}' for s in samples[::2])
            pairs=Counter(f'{a["flow"]}/{b["flow"]}' for c in corpora
                          for a,b in zip(c['samples'][::2],c['samples'][1::2]))
            stages.append(dict(stage=run['stage'],model_seed=run['model_seed'],
                pairs=len(samples)//2,unique_cranes=len(cranes),flow_samples=dict(flows),
                comparison_types=dict(pairs),
                zero_pairs=sum(r['stats']['zero'] for r in run['rows']),
                factual_mismatch=sum(r['stats']['factual_mismatch'] for r in run['rows']),
                forced_failure=sum(r['stats']['force_failed'] for r in run['rows']),
                collection_seconds=sum(r['collection_seconds'] for r in run['rows']),
                worlds=0 if run['stage']==6 else sum(r['stats']['worlds'] for r in run['rows'])))
    averages={}
    for arm in sorted({r['arm'] for r in comparisons}):
        rows=[r for r in comparisons if r['arm']==arm]
        averages[arm]={k:st.mean(r[k] for r in rows) for k in
            ('cost_change_pct','turn_change_pct','admitted_change','censored_24h_change','vessel_done_26h_change')}
        means=[st.mean(r['cost_change_pct'] for r in rows if r['model_seed']==seed)
               for seed in sorted({r['model_seed'] for r in rows})]
        averages[arm]['initialization_mean_cost_range_pct']=[min(means),max(means)]
        averages[arm]['cost_wins']=sum(r['cost_change_pct']<0 for r in rows)
        averages[arm]['n_cases']=len(rows)
    result=dict(comparisons_vs_rule_all=comparisons,averages=averages,training=stages,
        limitations=['three initializations crossed with the same three evaluation days',
                     'three updates per model; not a convergence or significance study',
                     'common behaviour collection; alternative learners not iterated on their own behaviour',
                     'external calibration incomplete; unfinished and unadmitted work must be checked separately'])
    save(root/'summary.json',result)
    print(json.dumps(averages,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('candidates','learning'))
    p.add_argument('folder')
    a=p.parse_args()
    (candidate_summary if a.kind=='candidates' else learning_summary)(a.folder)
