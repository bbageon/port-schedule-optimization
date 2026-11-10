"""Independently reconcile completion timing, model fits and paired comparisons."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as st
import subprocess
import xml.etree.ElementTree as ET

from benchmark_default_eval import save
from completion_learning import OUT, CASES, SEEDS, LOADS, RULES
from six_step_learning import gz_read, sha, load_net, labelset
from verify_job_completion import raw_metrics, assert_metrics
from horizon_metrics import aggregate
from evaluate_arrival_order import HASHES
from yard_rl.v7.evaluation_probe import digest


def bootstrap(xs):
    rng=random.Random(9917899);n=len(xs)
    b=sorted(st.mean(rng.choices(xs,k=n)) for _ in range(10000))
    return [b[249],b[9749]]


def summarize(group):
    output=dict(runs=len(group),mean_phi_krw=st.mean(d['result']['phi_krw'] for d in group),
        mean_vessel_cost=st.mean(d['result']['c_vessel'] for d in group),
        mean_unfinished_trucks=st.mean(d['result']['n_job_censored'] for d in group))
    raw=[d['metrics_by_flow']['ALL'] for d in group]
    output['exit_diagnostics']=dict(
        gate_to_exit_mean_min=sum(r['mean_turn_time_s']*r['n_trucks'] for r in raw)/sum(r['n_trucks'] for r in raw)/60,
        job_done_to_exit_mean_min=sum(r['post_job_gate_s'] for r in raw)/sum(r['n_job_done'] for r in raw)/60,
        mean_job_done=st.mean(r['n_job_done'] for r in raw),
        mean_job_done_exit_pending=st.mean(r['n_exit_pending'] for r in raw))
    for flow in ('TRUCK','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'):
        vals=[d['completion_by_flow']['at24'][flow] for d in group]
        output[flow]=dict(n=sum(v['n'] for v in vals),
            mean_min=sum(v['total_s'] for v in vals)/sum(v['n'] for v in vals)/60,
            mean_unfinished=st.mean(v['unfinished'] for v in vals))
    return output


def verify():
    import torch
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.integrated.evalkit import paired
    from yard_rl.experiments.gate_harness import judge_performance
    torch.set_num_threads(1)
    manifest=json.loads((OUT/'campaign.json').read_text(encoding='utf-8'))
    head=manifest['repro']['code']['git_head']
    assert manifest['repro']['code']['git_dirty'] is False
    for path,h in manifest['sources'].items():
        relative=Path(path).resolve().relative_to(Path.cwd()).as_posix()
        raw=subprocess.check_output(['git','show',f'{head}:{relative}'])
        assert hashlib.sha256(raw).hexdigest()==h,('committed source differs',path)
    corpus_pairs=0
    for seed in SEEDS:
        root=OUT/f'model-{seed}';net=load_net(root/'initial.pt')
        trainer=CraneTrainer(net,loss_kind='pairwise')
        rows=json.loads((root/'training.json').read_text(encoding='utf-8'))['rows']
        assert len(rows)==6
        for r in rows:
            assert sha(r['corpus'])==r['corpus_sha256'] and sha(r['checkpoint'])==r['checkpoint_sha256']
            data=gz_read(r['corpus']);c=data['result']['crane'];corpus_pairs+=c['labeled']
            assert data['result']['truck_time_contract']=='gate-in-to-job-done-v1'
            assert c['factual_mismatch']==c['force_failed']==c['hook_errors']==0
            fit=trainer.fit(labelset(data),seed=r['input_seed'])
            assert asdict(fit)==r['fit']
            actual=load_net(r['checkpoint']).state_dict()
            assert all(torch.equal(v,actual[k]) for k,v in net.state_dict().items())
    data={};cohorts={};order_count=0;flow_count=0
    for case in CASES:
        name='-'.join(map(str,case))+'.json.gz'
        d=gz_read(OUT/'evaluation'/name);rows=gz_read(OUT/'orders'/name)
        flows=gz_read(OUT/'flow-orders'/name)
        assert d['case']==list(case)
        raw=raw_metrics(rows);assert_metrics(raw,d['result'])
        assert raw==d['metrics_by_flow'] and digest(rows)==d['orders_sha256']
        for h in (24,26):assert aggregate(flows,h*3600.)==d['completion_by_flow'][f'at{h}']
        assert all(r['review_count']==1 and r['admitted'] and
            abs(r['attempt']['review_s']-r['notice_s'])<=1e-6 for r in rows)
        input_key=case[2:4];cohort=(d['probe']['flow_timing']['cohort_sha256'],
                                   d['probe']['metadata']['input_sha256'])
        if input_key in cohorts:assert cohorts[input_key]==cohort
        cohorts[input_key]=cohort
        order_count+=len(rows);flow_count+=len(flows);data[case]=d
    assert order_count==sum(c[2] for c in CASES)
    for case in CASES:
        if case[-1]:
            d,old=data[case],data[(*case[:-1],0)]
            assert d['result']==old['result'] and d['orders_sha256']==old['orders_sha256']
            assert all(d['probe'][k]==old['probe'][k] for k in HASHES)
    unique=[d for case,d in data.items() if not case[-1]]
    by_arm={arm:summarize([d for d in unique if d['case'][0]==arm]) for arm in (*RULES,'initial','trained')}
    by_load={str(load):{arm:summarize([d for d in unique if d['case'][0]==arm and d['case'][2]==load])
                        for arm in (*RULES,'initial','trained')} for load in LOADS}
    by_model={str(seed):{arm:summarize([d for d in unique if d['case'][0]==arm and d['case'][1]==seed])
                         for arm in ('initial','trained')} for seed in SEEDS}
    deltas=[];paired_rows=[]
    for load in LOADS:
        for day in range(3):
            baseline=data[('SF_SPT',0,load,day,0)]
            models=[data[('trained',s,load,day,0)] for s in SEEDS]
            delta=st.mean(d['result']['phi_krw'] for d in models)-baseline['result']['phi_krw']
            deltas.append(delta)
            paired_rows.append(dict(load=load,day=day,delta_phi_krw=delta,
                baseline_phi_krw=baseline['result']['phi_krw']))
    minimum=by_arm['SF_SPT']['mean_phi_krw']*.01
    paired_result=paired(deltas,delta_interest=minimum)
    lo,hi=bootstrap(deltas)
    direction='학습 정책 비용 증가' if lo>0 else ('학습 정책 비용 감소' if hi<0 else '차이 방향 미확정')
    paired_result=replace(paired_result,ci_lo=lo,ci_hi=hi,
        label=f'짝 재표집 95% 구간: {direction}; 검정력·최소 검출 효과는 기존 짝 비교 근사')
    trained=[d for d in unique if d['case'][0]=='trained']
    guards=dict(completion=all(d['result']['n_job_censored']==0 for d in trained),
        backlog_zero=all(d['completion_by_flow']['at24']['ALL_ORDERS']['unfinished']==0 for d in trained),
        physical_valid=all(d['result']['policy_exceptions']==d['probe']['ledger_audit']['skipped']==0 for d in unique),
        vessel_protection=all(d['result']['c_vessel']<=data[('SF_SPT',0,d['case'][2],d['case'][3],0)]['result']['c_vessel']+1e-6
                              for d in trained))
    performance=judge_performance(paired_result,minimum_improvement=minimum,interest_effect=minimum,hard_guards=guards)
    def pct(a,b):return 100*(b-a)/b if b else None
    comparisons={arm:{flow:pct(by_arm['trained'][flow]['mean_min'],by_arm[arm][flow]['mean_min'])
                            for flow in ('TRUCK','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD')}
                 for arm in (*RULES,'initial')}
    xmls=[Path('outputs/reports/yr332_real_order_information')/f for f in
          ('tests-contract.xml','tests-runtime.xml','tests-calibration.xml')]
    tests=[]
    for p in xmls:
        root=ET.parse(p).getroot()
        cases=root.findall('.//testcase');assert not root.findall('.//failure') and not root.findall('.//error')
        tests.extend((c.get('classname'),c.get('name')) for c in cases)
    verification=dict(status='PASS',evaluations=len(CASES),unique_evaluations=len(unique),
        exact_repeats=2,truck_rows=order_count,all_order_rows=flow_count,common_inputs=len(cohorts),
        training_updates=18,training_pairs=corpus_pairs,checkpoints_exact_refit=18,
        source_commit=head,committed_source_hashes=len(manifest['sources']),tests_passed=len(set(tests)),
        all_orders_reviewed_once_at_receipt=True,verification_script_sha256=sha(Path(__file__)))
    summary=dict(by_arm=by_arm,by_load=by_load,by_model=by_model,reduction_percent_vs=comparisons,
        paired_cost=asdict(paired_result),paired_rows=paired_rows,performance=performance.as_dict(),
        definition='Truck gate entry to job completion, censored at 24h; vessel planned stream start to yard completion',
        aggregation='Order weighted means; three models pooled equally on the same nine input days; repeats excluded',
        statistical_notes='Reported CI uses 10000 paired bootstrap replicates; MDE, power and TOST use the existing paired-comparison approximation.',
        limitations=['synthetic physical scenario; measured information calibration only',
            'first matched pre-entry reservation subset and stale reported bookings',
            'transport capacity 3 and cycle 470s remain assumptions',
            'daily 24h observation, not continuous-operation confirmation'])
    save(OUT/'verification.json',verification);save(OUT/'summary.json',summary)
    print(json.dumps(dict(verification=verification,comparison=comparisons,performance=performance.as_dict()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    verify()
