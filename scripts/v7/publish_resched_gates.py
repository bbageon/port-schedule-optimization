"""Keep deployment gates strict while recording the explicit bounded diagnostic."""
import argparse
import json
from pathlib import Path
import re

from resched_ppo import OUT, save
from six_step_learning import sha


def publish(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,judge_claim_alignment,
        combine_reliability,judge_scenario_validity,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    s=json.loads((OUT/'summary.json').read_text('utf-8'))
    m=json.loads((OUT/'campaign.json').read_text('utf-8'))
    artifacts={**s['artifact_hashes'],**{(OUT/p).as_posix():sha(OUT/p) for p in
                ('summary.json','README.md','campaign.json','schedule-window.json','schedule-window.png',
                 'inference-benchmark.json','verification.json','publication-byte-audit.json')}}
    runtime=judge_runtime_evidence(m['repro'],artifact_hashes=artifacts)
    board=audit_dashboard('.',task_id='YR-332-av',expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-332-av-resched-ppo-diagnostic.md'),
        evidence_paths=[OUT/'summary.json',OUT/'README.md',OUT/'campaign.json'],
        evidence_commits=[commit],remote_ref='origin/강화학습-판매',pin_commit=commit)
    body=(OUT/'README.md').read_text('utf-8')
    matches=re.findall(r'^\| [^|]+ \| ([\d.]+) \| ([+\-\d.]+)% \|$',body,re.M)
    assert len(matches)==len(s['groups'])
    declared=[];truth=[]
    for (minutes,change),g in zip(matches,s['groups']):
        declared.extend([float(minutes),float(change)])
        truth.extend([g['flows']['TRUCK']['mean_min'],g['vs_rule_percent']])
    alignment=judge_claim_alignment({str(i):v for i,v in enumerate(declared)},
        {str(i):v for i,v in enumerate(truth)},absolute_tolerance=.00501)
    reliability=combine_reliability(runtime,board,alignment)
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,physical_constraints=True,
        ledger_conservation=True,deterministic_replay=True),flow_checks={},anchors={},continuous_operation=False,
        request_real_terminal_claim=False)
    guards={g['arm']:dict(unfinished26={f:r['unfinished26'] for f,r in g['flows'].items()},
                         vs_rule_percent=g['vs_rule_percent'],vs_rule_cost_percent=g['vs_rule_cost_percent'])
            for g in s['groups'] if g['arm'] in ('mlp_final','transformer_final')}
    incomplete=any(v>0 for g in guards.values() for v in g['unfinished26'].values())
    performance=GateOutcome('performance',GateStatus.FAIL if incomplete else GateStatus.INCONCLUSIVE,
        '미완료가 남아 채택 조건 미충족' if incomplete else '합성3일 진단으로 확증 불가',
        evidence=dict(diagnostic_only=True,guards=guards,shared_scenarios=3,models_per_architecture=3))
    result=attach_common_gates(dict(task='YR-332-av',evidence_commit=commit,
        scope='explicitly user-requested synthetic scheduling diagnostic; not deployment or real-terminal validation'),
        ResearchGateReport(performance,reliability,scenario))
    save(OUT/'dashboard-audit.json',board.as_dict());save(OUT/'current-gates.json',result)
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in result['common_gates'].items() if isinstance(r,dict) and 'status' in r})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);publish(p.parse_args().commit)
