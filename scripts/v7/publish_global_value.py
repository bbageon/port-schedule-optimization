"""Pin source bytes and publish the evidence gates without changing research criteria."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import zipfile

from global_value_inputs import OUT, PREREG, SEEDS, read, save
from global_value_experiment import check
from six_step_learning import sha


def source_snapshot():
    cfg=check();campaign=read(OUT/'campaign.json');rows=[]
    path=OUT/'source-snapshot.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,expected in campaign['sources'].items():
            raw=Path(name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
            committed=subprocess.check_output(['git','show',campaign['commit']+':'+name])
            match='exact'
            if raw!=committed:
                assert raw.replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),name
                match='line_endings_only'
            archive.writestr(name,raw)
            rows.append(dict(path=name,sha256=expected,git_match=match))
    save(OUT/'source-publication.json',dict(passed=True,source_commit=campaign['commit'],
        files=len(rows),exact=sum(r['git_match']=='exact' for r in rows),
        line_endings_only=sum(r['git_match']!='exact' for r in rows),sources=rows,
        archive=path.as_posix(),archive_sha256=sha(path),
        purpose='Exact physical source bytes retained, including pre-existing CRLF input files'))


def publish(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,
        judge_claim_alignment,combine_reliability,judge_scenario_validity,
        GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    cfg=check();s=read(OUT/'summary.json');verification=read(OUT/'verification.json')
    campaign=read(OUT/'campaign.json');assert verification['passed']
    remote='origin/'+subprocess.check_output(['git','branch','--show-current'],text=True,encoding='utf-8').strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,remote],check=True)
    artifacts=dict(verification['artifact_hashes'])
    for name in ('summary.json','README.md','verification.json','campaign.json','source-publication.json',
                 'source-snapshot.zip','prediction-comparison.png','prediction-comparison.svg','REPRODUCE.md'):
        path=OUT/name;artifacts[path.as_posix()]=sha(path)
    for name in ('audit_global_value.py','report_global_value.py','plot_global_value.py','publish_global_value.py'):
        path=Path('scripts/v7')/name;artifacts[path.as_posix()]=sha(path)
    for name in ('current-gates-initial.json','publication-initial.json'):
        path=OUT/name
        if path.exists():artifacts[path.as_posix()]=sha(path)
    if s['policy']:
        for ext in ('png','svg'):
            path=OUT/f'policy-comparison.{ext}';artifacts[path.as_posix()]=sha(path)
    for name,h in artifacts.items():
        blob=subprocess.check_output(['git','show',commit+':'+name])
        assert hashlib.sha256(blob).hexdigest()==h,('Published artifact bytes differ',name)
    seeds=dict(models=list(SEEDS),curve_bank=[9964000],
        inputs=sorted({9964212}|{c['input_seed'] for c in cfg['train']+cfg['validation']+cfg['holdout']}),
        smoke_actions=[9969212],smoke_updates=[9969213],
        common_actions=[9965000+c['input_seed'] for c in cfg['train']+cfg['validation']],
        common_subsamples=[9966000+c['input_seed'] for c in cfg['train']+cfg['validation']],
        prediction_shuffle=[seed+9967000 for seed in SEEDS])
    if s['policy']:
        seeds.update(policy_actions=[seed+1000+i for seed in SEEDS for i in range(12)],
            policy_subsamples=[seed+2000+i for seed in SEEDS for i in range(6)],
            policy_updates=[seed+3000+i for seed in SEEDS for i in range(6)])
    stamp=dict(experiment='YR-332-az',code=dict(git_head=campaign['commit'],git_dirty=False),
        seeds=seeds,params=cfg,prereg=PREREG.as_posix(),
        provenance='Frozen clean campaign and exact source archive; current analysis artifacts independently committed')
    runtime=judge_runtime_evidence(stamp,artifact_hashes=artifacts)
    board=audit_dashboard('.',task_id='YR-332-az',expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-332-az-global-value-context.md'),
        evidence_paths=[OUT/'README.md',OUT/'summary.json',OUT/'campaign.json'],
        evidence_commits=[commit],remote_ref=remote,pin_commit=commit)
    line=re.search(r'^\| 전체 \| ([\d.]+) \| ([\d.]+) \| ([+\-\d.]+)% \|$',(OUT/'README.md').read_text('utf-8'),re.M)
    assert line
    declared=dict(zero=float(line[1]),global_rmse=float(line[2]),change=float(line[3]))
    truth=dict(zero=s['prediction']['groups']['zero']['rmse'],global_rmse=s['prediction']['groups']['global']['rmse'],
               change=s['prediction']['rmse_change_pct'])
    reliability=combine_reliability(runtime,board,judge_claim_alignment(declared,truth,absolute_tolerance=.00501))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,physical_constraints=True,
        ledger_conservation=True,deterministic_replay=True),flow_checks={},anchors={},continuous_operation=False,
        request_real_terminal_claim=False)
    policy=s['policy']
    if policy is None:
        performance=GateOutcome('performance',GateStatus.INCONCLUSIVE,
            '사전 예측 기준 미통과로 정책 성능 비교 미실행; 기존 정책 성능 미통과 상태 유지',
            evidence=dict(prediction_gate=s['prediction'],new_policy_evaluated=False))
    else:
        performance=GateOutcome('performance',GateStatus.INCONCLUSIVE if policy['adoption_numerical_pass'] else GateStatus.FAIL,
            '합성 정책 진단; 작업별 보호·규칙·미완료 조건과 실제 항만 확증을 구분',
            evidence={k:policy[k] for k in ('paired_change','vs_rule','conditions','adoption_numerical_pass')})
    gates=attach_common_gates(dict(task='YR-332-az',evidence_commit=commit,
        scope='Explicit two-stage synthetic diagnostic; no real-terminal or policy-adoption claim'),
        ResearchGateReport(performance,reliability,scenario))
    save(OUT/'dashboard-audit.json',board.as_dict());save(OUT/'current-gates.json',gates)
    save(OUT/'publication.json',dict(passed=reliability.status is GateStatus.PASS,evidence_commit=commit,
        remote_ref=remote,artifact_count=len(artifacts),artifact_hashes=artifacts,
        source_publication=read(OUT/'source-publication.json')))
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in gates['common_gates'].items() if isinstance(r,dict) and 'status' in r},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['source','publish']);p.add_argument('--commit')
    args=p.parse_args()
    if args.mode=='source':source_snapshot()
    else:
        assert args.commit;publish(args.commit)
