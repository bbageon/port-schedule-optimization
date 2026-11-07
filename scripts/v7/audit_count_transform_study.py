"""Independent raw arithmetic and final Dashboard audit for YR-332-s."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

OUT=Path('outputs/reports/yr332_count_transform')
REFERENCE=Path('outputs/reports/yr332_classical_eval/evaluation')


def read(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt',encoding='utf-8') as f:
            return json.load(f)
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def metrics(d):
    r=d['result']; p=d['probe']; op=p['operational']; counts=p['flow_status_counts']
    done=lambda flow:counts.get(flow+'/DONE',0)
    return dict(cost=r['phi_krw'],turn=r['mean_turn_time_s'],admitted=r['admitted'],
        censored=r['n_censored'],vessel_done=done('VESSEL_LOAD')+done('VESSEL_DISCHARGE'),
        discharge=done('VESSEL_DISCHARGE'),loading=done('VESSEL_LOAD'),
        delay=op['vessel_delay_s'],vessel_unfinished=op['vessel_unfinished'],idle=op['vessel_idle_s'],
        truck_yard_unfinished=sum(value for key,value in counts.items()
            if key.startswith(('GATE_IN/','GATE_OUT/')) and not key.endswith('/DONE')),
        p90=r['p90_turn_time_s'],p99=r['p99_turn_time_s'],vessel_cost=r['c_vessel'])


def audit():
    from yard_rl.experiments.gate_harness import judge_runtime_evidence,judge_claim_alignment
    summary=read(OUT/'summary.json'); verification=read(OUT/'verification.json'); manifest=read(OUT/'campaign.json')
    artifacts={**verification['artifact_sha256'],**manifest['input_sha256'],
        **manifest['prior_artifact_sha256'],**manifest['sources']}
    revision=read(OUT/'analysis-revision.json')
    assert verification['analysis_revision_sha256']==sha(OUT/'analysis-revision.json')
    for key in ('original_report','revised_analysis','amendment','diagnosis','supplemental_plan'):
        assert sha(revision[key])==revision[key+'_sha256'],key
        artifacts[revision[key]]=revision[key+'_sha256']
    diagnosis=read(revision['diagnosis'])
    for key in ('source_code','source','extended'):
        assert sha(diagnosis[key])==diagnosis[key+'_sha256']
        artifacts[diagnosis[key]]=diagnosis[key+'_sha256']
    # Independently check the architecture exception and current-architecture repeat.
    initial_checks=[]
    replay_fields=('candidates_sha256','choices_sha256','events_sha256','jobs_sha256',
                   'records_sha256','result_sha256')
    for seed in (9913501,9913502,9913503):
        for load in (3500,7500,15000):
            old=read(REFERENCE/f'initial-{seed}-{load}.json.gz')
            current=read(OUT/'evaluation'/f'clipped-{seed}-initial-{load}.json.gz')
            matches=old['result']==current['result'] and all(
                old['probe'][k]==current['probe'][k] for k in replay_fields)
            assert matches==((seed,load)!=(9913501,15000))
            initial_checks.append(dict(seed=seed,load=load,exact=matches))
    current=read(OUT/'evaluation'/'clipped-9913501-initial-15000.json.gz')
    repeat=read(OUT/'evaluation'/'clipped-9913501-initial-15000-repeat.json.gz')
    assert current['result']==repeat['result']
    for key in (*replay_fields,'metadata','flow_status_counts','decision_context',
                'operational','main_choices_sha256','count_histogram'):
        assert current['probe'][key]==repeat['probe'][key],key
    runtime=judge_runtime_evidence(manifest['repro'],artifact_hashes=artifacts).as_dict()
    assert runtime['status']=='PASS',runtime
    cache={}
    def get(p):
        key=str(p)
        if key not in cache:
            cache[key]=read(p)
        return cache[key]
    checks=0
    def equal(a,b):
        nonlocal checks
        assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-7),(a,b)
        checks+=1
    independently_computed=[]
    for row in summary['comparisons']:
        arm,seed,n,baseline=(row[k] for k in ('arm','model_seed','load','baseline'))
        b=get(OUT/'evaluation'/f'{arm}-{seed}-final-{n}.json.gz')
        if baseline=='own_initial':
            a=get(OUT/'evaluation'/f'{arm}-{seed}-initial-{n}.json.gz')
        elif baseline=='clipped':
            a=get(OUT/'evaluation'/f'clipped-{seed}-final-{n}.json.gz')
        else:
            a=get(REFERENCE/(f'pairwise-{seed}-{n}.json.gz' if baseline=='prior_pairwise'
                            else f'{baseline}-0-{n}.json.gz'))
        av,bv=metrics(a),metrics(b)
        computed={k:row[k] for k in ('arm','model_seed','load','baseline')}
        for k in av:
            dest=k+'_change_pct' if k in ('cost','turn') else k+'_change'
            value=bv[k]-av[k]
            if k in ('cost','turn'):
                value=value*100/av[k]
            equal(row[dest],value)
            computed[dest]=value
        independently_computed.append(computed)
        for d in (a,b):
            r=d['result']
            equal(r['phi_krw'],sum(r[k] for k in ('c_wait','c_vessel','c_move','c_rehandle')))
    for avg in summary['averages']:
        rows=[r for r in independently_computed if r['arm']==avg['arm'] and r['baseline']==avg['baseline']
              and (avg['load'] is None or r['load']==avg['load'])]
        equal(avg['n'],len(rows))
        for k in rows[0]:
            if k not in ('arm','model_seed','load','baseline'):
                equal(avg[k],sum(r[k] for r in rows)/len(rows))
        for k,metric,sign in [('admission_drop','admitted_change',-1),
                ('unfinished_increase','censored_change',1),
                ('yard_unfinished_increase','truck_yard_unfinished_change',1),
                ('vessel_drop','vessel_done_change',-1),('turn_wins','turn_change_pct',-1),
                ('cost_wins','cost_change_pct',-1)]:
            equal(avg[k],sum(sign*r[metric]>0 for r in rows))
    names={'기존 상한':'clipped','상한만 제거':'linear','로그 변환':'log1p'}
    reported={}; actual={}
    for line in (OUT/'README.md').read_text(encoding='utf-8').splitlines():
        cells=[x.strip() for x in line.split('|')][1:-1]
        if len(cells)!=5 or cells[0] not in names or not cells[1].endswith('%'):
            continue
        arm=names[cells[0]]
        for cell,baseline,metric in zip(cells[1:],('clipped','clipped','own_initial','spt'),
                ('turn_change_pct','cost_change_pct','turn_change_pct','turn_change_pct'),strict=True):
            key=f'{arm}/{baseline}/{metric}'
            reported[key]=float(cell[:-1])
            rs=[r for r in independently_computed if r['arm']==arm and r['baseline']==baseline]
            actual[key]=statistics.mean(r[metric] for r in rs)
    assert len(reported)==12
    alignment=judge_claim_alignment(reported,actual,absolute_tolerance=.00050001).as_dict()
    assert alignment['status']=='PASS',alignment
    result=dict(status='PASS',scalar_checks=checks,comparisons=len(independently_computed),
        initial_architecture_comparisons=initial_checks,current_initial_repeat_exact=True,
        runtime=runtime,claim_alignment=alignment,summary_sha256=sha(OUT/'summary.json'),
        report_sha256=sha(OUT/'README.md'),audit_source=Path(__file__).as_posix(),audit_source_sha256=sha(__file__))
    save(OUT/'independent-arithmetic-audit.json',result)
    print(json.dumps(dict(status='PASS',scalar_checks=checks,reported_values=reported),ensure_ascii=False))
    return result


def finalize(pin):
    from yard_rl.experiments.gate_harness import audit_dashboard,judge_claim_alignment,GateOutcome,GateStatus
    data=read(OUT/'independent-arithmetic-audit.json')
    assert data['summary_sha256']==sha(OUT/'summary.json')
    assert data['report_sha256']==sha(OUT/'README.md')
    upstream=subprocess.check_output(['git','rev-parse','--abbrev-ref','@{upstream}'],encoding='utf-8').strip()
    board=audit_dashboard('.',task_id='YR-332-s',expected_state='done',
        spec_path='.claude/docs/dashboard-task-specs/YR-332-s-feasible-count-scale.md',
        evidence_paths=[str(OUT/p) for p in ('summary.json','README.md','verification.json','campaign.json',
                                            'independent-arithmetic-audit.json')],
        evidence_commits=[pin],remote_ref=upstream,pin_commit=pin).as_dict()
    assert board['status']=='PASS',board
    overview=next(line for line in Path('.claude/Dashboard/README.md').read_text(encoding='utf-8').splitlines()
                  if line.startswith('- **후보 수 입력 상한 비교 완료'))
    values=[float(v) for v in re.findall(r'([+-]\d+\.\d+)%',overview)]
    summary=read(OUT/'summary.json')
    def mean(arm,baseline,key):
        return next(r[key] for r in summary['averages'] if r['arm']==arm
                    and r['baseline']==baseline and r['load'] is None)
    actual=[mean('linear','clipped','turn_change_pct'),mean('linear','clipped','cost_change_pct'),
            mean('log1p','clipped','turn_change_pct'),mean('linear','spt','turn_change_pct')]
    assert len(values)==len(actual)==4
    claims=judge_claim_alignment({str(i):v for i,v in enumerate(values)},
        {str(i):v for i,v in enumerate(actual)},absolute_tolerance=.00050001).as_dict()
    assert claims['status']=='PASS',claims
    board['evidence']['overview_claim_alignment']=claims
    save(OUT/'final-dashboard-audit.json',board)
    verdict=read(OUT/'final-verdict.json')
    verdict['dashboard_alignment']=dict(status='PASS',pin_commit=pin,evidence='final-dashboard-audit.json')
    verdict['common_gates']['reliability']=GateOutcome('reliability',GateStatus.PASS,
        'Dashboard·실행 코드·원자료·보고 수치·원격 커밋이 일치함',
        evidence=dict(runtime=data['runtime'],claims=data['claim_alignment'],dashboard=board,
            arithmetic_scalar_checks=data['scalar_checks'])).as_dict()
    save(OUT/'final-verdict.json',verdict)
    print('Final Dashboard and remote evidence: PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',metavar='PIN_COMMIT')
    args=parser.parse_args()
    if args.finalize:
        finalize(args.finalize)
    else:
        audit()
