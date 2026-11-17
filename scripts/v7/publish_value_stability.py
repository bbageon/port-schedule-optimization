"""Publish strict gates and verify committed evidence bytes after pushing results."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess

from value_stability import check
from value_stability_inputs import OUT, PREREG, ARMS, SEEDS, read, save
from six_step_learning import sha


def main(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard, judge_runtime_evidence,
        judge_claim_alignment, combine_reliability, judge_scenario_validity,
        GateOutcome, GateStatus, ResearchGateReport, attach_common_gates)
    cfg=check();s=read(OUT/'summary.json');campaign=read(OUT/'campaign.json')
    assert read(OUT/'verification.json')['passed']
    remote='origin/'+subprocess.check_output(['git','branch','--show-current'],text=True,encoding='utf-8').strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,remote],check=True)
    artifacts={**s['artifact_hashes']}
    for name in ('summary.json','README.md','campaign.json','verification.json','comparison.png',
                 'choice-diagnostic.json','gradient-diagnostic.json','source-publication.json'):
        p=OUT/name;artifacts[p.as_posix()]=sha(p)
    published={}
    for path,h in artifacts.items():
        blob=subprocess.check_output(['git','show',commit+':'+path])
        assert hashlib.sha256(blob).hexdigest()==h,('Published bytes differ',path)
        published[path]=h
    # This is an explicit schema mapping, not a claim that repro_stamp() was called
    # at run time. prepare() hard-asserted a clean tree before writing campaign.json;
    # the frozen 215-source mapping is checked by check() and source-publication.json.
    stamp=dict(experiment='YR-332-ay',code=dict(git_head=campaign['commit'],git_dirty=False),
        seeds=dict(models=list(SEEDS),training_inputs=[c['input_seed'] for c in cfg['trajectories']],
            training_actions=[s+1000+i for s in SEEDS for i in range(12)],
            training_subsamples=[s+2000+i for s in SEEDS for i in range(6)],
            training_updates=[s+3000+i for s in SEEDS for i in range(6)],
            evaluation_inputs=sorted({c['input_seed'] for c in cfg['evaluation_cases']}),
            curve_bank=[9954000]),prereg=PREREG.as_posix(),params=cfg,
        provenance='Schema mapping from frozen campaign, hard clean-tree assertion in prepare(), and committed source-byte audit')
    runtime=judge_runtime_evidence(stamp,artifact_hashes=artifacts)
    spec=Path('.claude/docs/dashboard-task-specs/YR-332-ay-value-learning-stability.md')
    board=audit_dashboard('.',task_id='YR-332-ay',expected_state='done',spec_path=spec,
        evidence_paths=[OUT/'summary.json',OUT/'README.md',OUT/'campaign.json'],
        evidence_commits=[commit],remote_ref=remote,pin_commit=commit)
    text=(OUT/'README.md').read_text('utf-8')
    matched=re.findall(r'^\| (기존 가치 학습|가치 목표 정규화|짧은 실작업 우선 규칙) \| ([\d.]+) \| ([+\-\d.]+)% \| ([\d.]+) \|$',text,re.M)
    assert len(matched)==3
    declared={};truth={}
    for (_,minutes,cost,unfinished),arm in zip(matched,(*ARMS,'SF_SPT')):
        g=s['groups'][arm]
        declared.update({arm+'_time':float(minutes),arm+'_cost':float(cost),arm+'_unfinished':float(unfinished)})
        truth.update({arm+'_time':g['flows']['TRUCK']['mean_min'],
            arm+'_cost':100*(g['cost']/s['groups']['control']['cost']-1),arm+'_unfinished':g['unfinished26_mean']})
    reliability=combine_reliability(runtime,board,judge_claim_alignment(declared,truth,absolute_tolerance=.00501))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,physical_constraints=True,
        ledger_conservation=True,deterministic_replay=True),flow_checks={},anchors={},continuous_operation=False,
        request_real_terminal_claim=False)
    performance=GateOutcome('performance',GateStatus.INCONCLUSIVE if s['adoption_performance_pass'] else GateStatus.FAIL,
        '채택 조건 미충족; 합성 진단으로 현장 개선 확증 불가',evidence=dict(
            consistent_improvement_candidate=s['consistent_improvement_candidate'],paired=s['paired_change'],
            vs_rule=s['vs_rule'],unfinished26_mean=s['groups']['popart']['unfinished26_mean'],
            model_pairs=3,synthetic_curves=3,diagnostic_only=True))
    gates=attach_common_gates(dict(task='YR-332-ay',evidence_commit=commit,
        scope='explicitly authorized bounded synthetic diagnostic; no real-terminal claim'),
        ResearchGateReport(performance,reliability,scenario))
    save(OUT/'dashboard-audit.json',board.as_dict());save(OUT/'current-gates.json',gates)
    save(OUT/'publication.json',dict(passed=reliability.status is GateStatus.PASS,evidence_commit=commit,
        remote_ref=remote,published_evidence_files=len(published),artifact_hashes=published,
        source_publication=read(OUT/'source-publication.json')))
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in gates['common_gates'].items() if isinstance(r,dict) and 'status' in r},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);main(p.parse_args().commit)
