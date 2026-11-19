"""Archive exact frozen sources and publish scoped gates after results are pushed."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import zipfile

from global_policy_inputs import OUT,PREREG,SEEDS,ARMS,read,save
from global_policy_plan import check
from six_step_learning import sha


def source_snapshot():
    check();campaign=read(OUT/'campaign.json');rows=[];path=OUT/'source-snapshot.zip'
    assert not path.exists()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,expected in campaign['sources'].items():
            raw=Path(name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
            committed=subprocess.check_output(['git','show',campaign['commit']+':'+name]);match='exact'
            if raw!=committed:
                assert raw.replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),name
                match='line_endings_only'
            archive.writestr(name,raw);rows.append(dict(path=name,sha256=expected,git_match=match))
    save(OUT/'source-publication.json',dict(passed=True,source_commit=campaign['commit'],files=len(rows),
        exact=sum(r['git_match']=='exact' for r in rows),line_endings_only=sum(r['git_match']!='exact' for r in rows),
        sources=rows,archive=path.as_posix(),archive_sha256=sha(path),
        purpose='Frozen physical source bytes; existing CRLF preserved without changing training sources'))


def publish(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,judge_claim_alignment,
        combine_reliability,judge_scenario_validity,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    cfg=check();s=read(OUT/'summary.json');v=read(OUT/'verification.json');campaign=read(OUT/'campaign.json')
    assert v['passed'] and read(OUT/'source-publication.json')['passed']
    remote='origin/'+subprocess.check_output(['git','branch','--show-current'],text=True,encoding='utf-8').strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,remote],check=True)
    artifacts=dict(v['artifact_hashes']);artifacts[(OUT/'verification.json').as_posix()]=sha(OUT/'verification.json')
    for name,h in artifacts.items():
        blob=subprocess.check_output(['git','show',commit+':'+name])
        assert hashlib.sha256(blob).hexdigest()==h,('Published artifact bytes differ',name)
    stamp=campaign['repro'];assert stamp['code']['git_head']==campaign['commit']
    runtime=judge_runtime_evidence(stamp,artifact_hashes=artifacts)
    spec=Path('.claude/docs/dashboard-task-specs/YR-332-bc-global-context-policy-exploration.md')
    board=audit_dashboard('.',task_id='YR-332-bc',expected_state='done',spec_path=spec,
        evidence_paths=[OUT/'README.md',OUT/'summary.json',OUT/'campaign.json'],
        evidence_commits=[commit],remote_ref=remote,pin_commit=commit)
    report=(OUT/'README.md').read_text('utf-8');declared={};truth={}
    for flow,label in [('TRUCK','평균 트럭'),('VESSEL','평균 본선 작업'),('GATE_IN','반입'),
                       ('GATE_OUT','반출'),('VESSEL_DISCHARGE','양하'),('VESSEL_LOAD','적하')]:
        line=re.search(r'^\| '+label+r'\(분\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([+\-\d.]+)% \|$',report,re.M)
        assert line,label
        for i,arm in enumerate(('initial','zero','global','SF_SPT'),1):
            declared[flow+'_'+arm]=float(line[i]);truth[flow+'_'+arm]=s['groups'][arm]['flows'][flow]['mean_min']
        declared[flow+'_change']=float(line[5]);truth[flow+'_change']=s['paired_change']['flow_pct'][flow]
    line=re.search(r'총비용 변화는 ([+\-\d.]+)%',report);assert line
    declared['cost_change']=float(line[1]);truth['cost_change']=s['paired_change']['cost_pct']
    reliability=combine_reliability(runtime,board,judge_claim_alignment(declared,truth,absolute_tolerance=.00501))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,physical_constraints=True,
        ledger_conservation=True,deterministic_replay=True),flow_checks={},anchors={},continuous_operation=False,
        request_real_terminal_claim=False)
    guards={k:s['conditions'][k] for k in ('outbound_no_worse_than_zero','vessel_time_no_worse_than_zero',
        'vessel_no_worse_than_zero','unfinished_no_worse_than_zero','all_final_jobs_complete')}
    performance=GateOutcome('performance',GateStatus.FAIL if not all(guards.values()) else GateStatus.INCONCLUSIVE,
        '합성 탐색의 작업별 보호·완주 조건 미충족' if not all(guards.values()) else
        '작은 합성 표본의 고정 예산 탐색; 실제 성능 확증과 정책 채택 보류',evidence=dict(
        metric='terminal_total_cost',baseline='SF-SPT',hard_guards=guards,
        candidate_minus_baseline=s['uncertainty']['global_vs_rule']['cost_difference'],
        candidate_minus_baseline_ci95_raw=s['uncertainty']['global_vs_rule']['cost_difference_ci95'],
        primary_global_vs_zero=s['uncertainty']['global_vs_zero'],conditions=s['conditions'],
        numerical_adoption_pass=s['adoption_numerical_pass'],model_pairs=3,synthetic_curves=3,diagnostic_only=True))
    gates=attach_common_gates(dict(task='YR-332-bc',evidence_commit=commit,
        scope='Explicit user-authorized synthetic exploration; prediction errors report-only; no real-terminal claim'),
        ResearchGateReport(performance,reliability,scenario))
    save(OUT/'dashboard-audit.json',board.as_dict());save(OUT/'current-gates.json',gates)
    save(OUT/'publication.json',dict(passed=reliability.status is GateStatus.PASS,evidence_commit=commit,
        remote_ref=remote,artifact_count=len(artifacts),artifact_hashes=artifacts,
        source_publication=read(OUT/'source-publication.json')))
    for name in ('dashboard-audit.json','current-gates.json'):
        path=OUT/name;path.with_name(name+'.sha256').write_text(sha(path)+'\n',encoding='ascii',newline='\n')
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in gates['common_gates'].items() if isinstance(r,dict) and 'status' in r},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('source','publish'));p.add_argument('--commit');a=p.parse_args()
    if a.mode=='source':source_snapshot()
    else:
        assert a.commit;publish(a.commit)
