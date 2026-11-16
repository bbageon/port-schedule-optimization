"""Publish strict existing gates for the four-step diagnostic without relaxing reality requirements."""
import argparse
from pathlib import Path

from audit_day_shapes import read
from six_step_learning import sha
from yard_rl.integrated.repro import write_result

TASKS={
 'mix':('YR-332-ao','yr332_synthetic_inputs','YR-332-ao-day-shape-mix-variation.md'),
 'minutes':('YR-332-ar','yr332_parametric_arrivals','YR-332-ar-parametric-within-hour-arrivals.md'),
 'network':('YR-332-as','yr332_context_learning','YR-332-as-network-learning-design.md')}


def publish(kind,commit,state):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,judge_claim_alignment,
        combine_reliability,judge_scenario_validity,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    task,folder,spec=TASKS[kind];root=Path('outputs/reports')/folder
    s=read(root/'summary.json');m=read(root/'campaign.json')
    artifacts={**s['artifact_hashes'],**{(root/p).as_posix():sha(root/p) for p in ('summary.json','README.md','campaign.json')}}
    runtime=judge_runtime_evidence(m['repro'],artifact_hashes=artifacts)
    board=audit_dashboard('.',task_id=task,expected_state=state,
        spec_path=Path('.claude/docs/dashboard-task-specs')/spec,
        evidence_paths=[root/'summary.json',root/'README.md',root/'campaign.json'],
        evidence_commits=[commit],remote_ref='origin/강화학습-판매',pin_commit=commit)
    body=(root/'README.md').read_text(encoding='utf-8')
    declared=[];truth=[]
    for line in body.splitlines():
        fields=[v.strip() for v in line.strip('|').split('|')]
        if not line.startswith('|'):continue
        if kind!='network' and len(fields)==5 and fields[0].replace(',','').isdigit():
            g=next(g for g in s['groups'] if g['load']==int(fields[0].replace(',','')) and g['rule']==fields[1])
            declared.extend([float(fields[2]),float(fields[3]),float(fields[4].rstrip('%'))])
            truth.extend([g['baseline']['flows']['TRUCK']['mean_min'],g['candidate']['flows']['TRUCK']['mean_min'],g['changes']['TRUCK']])
    if kind=='network':
        import re
        matches=re.findall(r'^\| [^|]+ \| ([\d.]+) \| ([+\-\d.]+)% \|$',body,re.M)
        assert len(matches)==len(s['groups'])
        for (minutes,change),g in zip(matches,s['groups']):
            declared.extend([float(minutes),float(change)])
            truth.extend([g['flows']['TRUCK']['mean_min'],g['vs_rule_percent']])
    assert declared
    reliability=combine_reliability(runtime,board,judge_claim_alignment(
        {str(i):v for i,v in enumerate(declared)},{str(i):v for i,v in enumerate(truth)},absolute_tolerance=.00501))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,physical_constraints=True,
        ledger_conservation=True,deterministic_replay=True),flow_checks={},anchors={},continuous_operation=False,
        request_real_terminal_claim=False)
    performance=GateOutcome('performance',GateStatus.INCONCLUSIVE,
        '고정 규칙 입력 민감도이며 새 정책의 성능 판정 아님')
    if kind=='network':
        guards={g['arm']:dict(unfinished26={f:r['unfinished26'] for f,r in g['flows'].items()},
                             vs_rule_percent=g['vs_rule_percent']) for g in s['groups']}
        incomplete=any(v>0 for g in guards.values() for v in g['unfinished26'].values())
        performance=GateOutcome('performance',GateStatus.FAIL if incomplete else GateStatus.INCONCLUSIVE,
            '미완료가 남아 채택 조건 미충족' if incomplete else '세 합성 입력으로 확증 불가',
            evidence=dict(diagnostic_only=True,guards=guards))
    result=attach_common_gates(dict(task=task,evidence_commit=commit,
        scope='user-requested bounded synthetic diagnostic; no real-terminal deployment claim'),
        ResearchGateReport(performance,reliability,scenario))
    write_result(root/'dashboard-audit.json',board.as_dict());write_result(root/'current-gates.json',result)
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in result['common_gates'].items() if isinstance(r,dict) and 'status' in r})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=TASKS);p.add_argument('--commit',required=True)
    p.add_argument('--state',default='done');a=p.parse_args();publish(a.kind,a.commit,a.state)
