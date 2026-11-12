"""Independent timing reconciliation and bounded interpretation of demand tests."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from six_step_learning import gz_read,sha
from evaluate_observed_demand import OUT,CASES,VARIANTS,LOADS,RULES,SEEDS,read
from evaluate_arrival_order import HASHES
from verify_job_completion import raw_metrics,assert_metrics
from yard_rl.v7.evaluation_probe import digest

FLOWS=('TRUCK','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD')


def independent_flows(rows,end):
    buckets=defaultdict(list)
    for r in rows:
        start,finish=r['origin_s'],r['finish_s']
        if start is None or start>end:
            continue
        assert finish is None or finish>=start-1e-6
        elapsed=(min(end,finish) if finish is not None else end)-start
        for flow in (r['flow'],'ALL_ORDERS','TRUCK' if r['flow'].startswith('GATE_') else 'VESSEL'):
            buckets[flow].append((max(0.,elapsed),finish is None or finish>end))
    return {k:dict(n=len(v),total_s=math.fsum(x[0] for x in v),unfinished=sum(x[1] for x in v)) for k,v in buckets.items()}


def summarize(group):
    out=dict(runs=len(group),mean_phi_krw=st.mean(d['result']['phi_krw'] for d in group),
        mean_vessel_cost=st.mean(d['result']['c_vessel'] for d in group),
        mean_unfinished_trucks=st.mean(d['result']['n_job_censored'] for d in group))
    for flow in FLOWS:
        vals=[d['completion_by_flow']['at24'][flow] for d in group]
        out[flow]=dict(n=sum(x['n'] for x in vals),mean_min=math.fsum(x['total_s'] for x in vals)/sum(x['n'] for x in vals)/60,
                      mean_unfinished=st.mean(x['unfinished'] for x in vals))
    return out


def source_crosscheck():
    """Visit counts and hour bins recomputed without the calibration reader."""
    audit=read(OUT/'input-audit.json');corrected=read(OUT/'input-audit-corrected.json')
    assert audit['observed']['trucks']==corrected['observed']['trucks']
    assert audit['observed']['reservation_selection']==corrected['observed']['reservation_selection']
    assert audit['synthetic']==corrected['synthetic']
    observed=corrected['observed'];root=Path(observed['private_root'])
    for item in observed['manifest']:
        assert sha(root/item['path'])==item['sha256']
    counts=defaultdict(int);hours=[0]*24;entered=0
    for path in sorted(root.glob('기사/방문/*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                if not row['gate_in_time']:
                    continue
                t=datetime.fromisoformat(row['gate_in_time'])
                entered+=1;counts[row['in_out']]+=1;hours[t.hour]+=1
    assert entered==observed['entered']==34304
    assert counts=={'IN':15865,'OUT':18439}
    assert hours==observed['trucks']['all_hour_counts']
    # Old policy output logs independently reproduce the original evaluation mix.
    old_counts=defaultdict(int)
    for day in range(3):
        for load in LOADS:
            rows=gz_read(Path('outputs/reports/yr332_completion_learning/orders')/f'SF_SPT-0-{load}-{day}-0.json.gz')
            for row in rows:old_counts[row['flow']]+=1
    assert dict(old_counts)==audit['synthetic']['trucks']['evaluation']['flow_counts']
    cluster=read(OUT/'arrival-clustering-audit.json')
    assert cluster['entered_order_records']==entered
    for path,h in cluster['source_sha256'].items():assert sha(root/path)==h
    return dict(source_files=len(observed['manifest']),actual_entered=entered,old_evaluation_trucks=sum(old_counts.values()),
                vessel_snapshot_correction_preserves_all_experiment_inputs=True,
                independent_visit_and_logged_synthetic_counts_match=True)


def verify():
    manifest=read(OUT/'campaign.json');head=manifest['repro']['code']['git_head']
    assert manifest['repro']['code']['git_dirty'] is False
    for path,h in manifest['sources'].items():
        assert sha(path)==h,('current source changed',path)
        blob=subprocess.check_output(['git','show',f'{head}:{path}'])
        assert hashlib.sha256(blob).hexdigest()==h,('frozen committed source differs',path)
    inputs={(r['variant'],r['load']):r for r in read(OUT/'input-preflight.json')['rows']}
    data={};cohorts={};vessel_cohorts={};trucks=0;all_jobs=0
    for case in CASES:
        name='-'.join(map(str,case))+'.json.gz'
        d=gz_read(OUT/'evaluation'/name);rows=gz_read(OUT/'orders'/name);flows=gz_read(OUT/'flow-orders'/name)
        assert d['case']==list(case)
        raw=raw_metrics(rows);assert_metrics(raw,d['result'])
        assert raw==d['metrics_by_flow'] and digest(rows)==d['orders_sha256']
        assert len(rows)==case[3] and len({r['job_id'] for r in flows})==len(flows)
        for row in rows:
            assert row['admitted'] and row['review_count']==1
            assert abs(row['attempt']['review_s']-row['notice_s'])<=1e-6
            stamps=[row['times'][k] for k in ('gate_in_s','block_in_s','service_start_s','job_done_s','gate_out_s') if row['times'][k] is not None]
            assert all(a<=b+1e-6 for a,b in zip(stamps,stamps[1:]))
        by_job={r['job_id']:r for r in flows}
        for row in rows:
            f=by_job[row['job_id']]
            assert f['origin_s']==row['times']['gate_in_s'] and f['finish_s']==row['times']['job_done_s']
        for h in (24,26):
            recomputed=independent_flows(flows,h*3600.)
            for flow,values in recomputed.items():
                reported=d['completion_by_flow'][f'at{h}'][flow]
                for key,value in values.items():
                    assert math.isclose(value,reported[key],rel_tol=1e-12,abs_tol=1e-6),(case,flow,key)
                assert math.isclose(reported['mean_s'],values['total_s']/values['n'],rel_tol=1e-12,abs_tol=1e-6)
        input_key=(case[0],case[3]);sig=(d['schedule_sha256'],d['probe']['metadata']['input_sha256'])
        assert d['schedule_sha256']==inputs[input_key]['schedule_sha256']
        if input_key in cohorts:assert cohorts[input_key]==sig
        cohorts[input_key]=sig
        vessel=digest([{k:r[k] for k in ('job_id','block','flow','origin_s','planned_release_s','stream')}
            for r in flows if r['flow'].startswith('VESSEL_')])
        if case[3] in vessel_cohorts:assert vessel_cohorts[case[3]]==vessel
        vessel_cohorts[case[3]]=vessel
        trucks+=len(rows);all_jobs+=len(flows);data[case]=d
    for case,d in data.items():
        if case[-1]:
            first=data[(*case[:-1],0)]
            assert first['result']==d['result'] and first['orders_sha256']==d['orders_sha256']
            assert all(first['probe'][k]==d['probe'][k] for k in HASHES)
    compatibility=list((OUT/'evaluation').glob('compatibility-*.json.gz'));assert len(compatibility)==2
    for path in compatibility:
        d=gz_read(path);_,arm,seed,load,_=d['case']
        old=gz_read(Path('outputs/reports/yr332_completion_learning/evaluation')/f'{arm}-{seed}-{load}-0-0.json.gz')
        assert d['exact_old_default'] and d['result']==old['result']
        assert all(d['probe'][k]==old['probe'][k] for k in HASHES)
        assert d['orders_sha256']==old['orders_sha256']
    xml=ET.parse(OUT/'contract-tests.xml').getroot()
    assert not xml.findall('.//failure') and not xml.findall('.//error')
    diagnostics=read(OUT/'saved-learning-diagnostic.json')
    for path,h in diagnostics['source_sha256'].items():assert sha(path)==h
    unique=[d for case,d in data.items() if not case[-1]]
    by_variant={v:{arm:summarize([d for d in unique if d['case'][0]==v and d['case'][1]==arm]) for arm in (*RULES,'initial','trained')} for v in VARIANTS}
    by_load={v:{str(load):{arm:summarize([d for d in unique if d['case'][0]==v and d['case'][1]==arm and d['case'][3]==load]) for arm in (*RULES,'initial','trained')} for load in LOADS} for v in VARIANTS}
    by_model={v:{str(seed):{arm:summarize([d for d in unique if d['case'][0]==v and d['case'][1]==arm and d['case'][2]==seed]) for arm in ('initial','trained')} for seed in SEEDS} for v in VARIANTS}
    pct=lambda new,old:100*(old-new)/old if old else None
    reduction={v:{base:{flow:pct(by_variant[v]['trained'][flow]['mean_min'],by_variant[v][base][flow]['mean_min']) for flow in FLOWS} for base in (*RULES,'initial')} for v in VARIANTS}
    environmental={f'{a}_to_{b}':{arm:{flow:pct(by_variant[b][arm][flow]['mean_min'],by_variant[a][arm][flow]['mean_min']) for flow in FLOWS} for arm in (*RULES,'initial','trained')} for a,b in zip(VARIANTS,VARIANTS[1:])}
    decomposition={}
    for a,b in zip(VARIANTS,VARIANTS[1:]):
        decomposition[a+'_to_'+b]={}
        for arm in (*RULES,'initial','trained'):
            old,new=by_variant[a][arm],by_variant[b][arm]
            composition=within=0.
            for flow in ('GATE_IN','GATE_OUT'):
                p0=old[flow]['n']/old['TRUCK']['n'];p1=new[flow]['n']/new['TRUCK']['n']
                composition+=(p1-p0)*old[flow]['mean_min']
                within+=p1*(new[flow]['mean_min']-old[flow]['mean_min'])
            change=new['TRUCK']['mean_min']-old['TRUCK']['mean_min']
            assert math.isclose(change,composition+within,abs_tol=1e-10)
            decomposition[a+'_to_'+b][arm]=dict(actual_change_min=change,
                composition_at_old_type_means_min=composition,within_type_means_at_new_shares_min=within)
    verification=dict(status='PASS',evaluations=len(CASES),unique_evaluations=len(unique),exact_repeats=3,
        old_default_exact_reproductions=2,truck_rows=trucks,all_order_rows=all_jobs,common_inputs=len(cohorts),
        source_commit=head,committed_source_hashes=len(manifest['sources']),tests_passed=len(xml.findall('.//testcase')),
        source_crosscheck=source_crosscheck(),verification_script_sha256=sha(Path(__file__)))
    summary=dict(by_variant=by_variant,by_load=by_load,by_model=by_model,reduction_percent_vs=reduction,
        environmental_time_reduction_percent=environmental,mean_change_accounting=decomposition,
        accounting_note='Exploratory exact arithmetic decomposition, not causal attribution. Primary metrics and adoption criteria unchanged.',
        definition='Truck gate entry to job completion; vessel stream planned start to yard completion; unfinished censored at 24h',
        aggregation='Job-count-weighted means. Three frozen models share each input; models are not independent input days.',
        load_weight={str(load):load/sum(LOADS) for load in LOADS},
        uncertainty='One new input at each load, no significance test; diagnostic only. No final unseen real-data period is available.')
    save(OUT/'verification.json',verification);save(OUT/'summary.json',summary)
    print(json.dumps(dict(verification=verification,reduction_percent_vs_initial={v:reduction[v]['initial'] for v in VARIANTS}),ensure_ascii=False,indent=2))


if __name__=='__main__':verify()
