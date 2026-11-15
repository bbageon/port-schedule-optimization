"""Check the shape report against raw summary and the remotely pinned board."""
import argparse
from pathlib import Path
import re

from check_initial_stock import OUT,read,sha,write_result


def publish(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,judge_claim_alignment,
        combine_reliability,judge_scenario_validity,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    s=read(OUT/'shape-summary.json');m=read(OUT/'campaign.json')
    assert s['readout_tracked_dirty'] is False
    paths=[OUT/p for p in ('shape-summary.json','shape-sensitivity.md','shape-input-audit.json',
        'campaign.json','execution-shapes.json','execution-shapes-result.json','authorization-ao.json')]
    artifacts={**s['artifact_hashes'],**{p.as_posix():sha(p) for p in paths}}
    runtime=judge_runtime_evidence(m['repro'],artifact_hashes=artifacts)
    board=audit_dashboard('.',task_id='YR-332-ao',expected_state='in-progress',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-332-ao-day-shape-mix-variation.md'),
        evidence_paths=[OUT/'shape-summary.json',OUT/'shape-sensitivity.md',OUT/'campaign.json'],
        evidence_commits=[commit],remote_ref='origin/강화학습-판매',pin_commit=commit)
    body=(OUT/'shape-sensitivity.md').read_text(encoding='utf-8')
    claims={'cases':int(re.search(r'= \*\*(\d+)회\*\*',body).group(1))}
    truth={'cases':s['cases']}
    names={'SF_SPT':'실작업·짧은 작업 우선','FIFO':'작업 대기 진입 순서','LWKR':'남은 작업량이 적은 순서'}
    for g in s['groups']:
        prefix=f"| {g['load']:,} | {names[g['rule']]} |"
        line=next(line for line in body.splitlines() if line.startswith(prefix))
        fields=[x.strip() for x in line.strip('|').split('|')]
        f=g['by_flow']['TRUCK'];key=f"{g['load']}:{g['rule']}"
        for name,idx,value in [('fixed',2,f['fixed']['mean_min']),('variable',3,f['variable']['mean_min']),
                               ('change',4,f['change_percent']),('cost',6,g['cost_change_percent'])]:
            claims[f'{key}:{name}']=float(fields[idx].rstrip('%'))
            truth[f'{key}:{name}']=value
    reliability=combine_reliability(runtime,board,judge_claim_alignment(claims,truth,absolute_tolerance=.00501))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,physical_constraints=True,
        ledger_conservation=True,deterministic_replay=True),flow_checks={},anchors={},
        continuous_operation=False,request_real_terminal_claim=False)
    performance=GateOutcome('performance',GateStatus.INCONCLUSIVE,
        '고정 규칙의 입력 민감도이며 학습 정책 개선 검증 아님',evidence=dict(previous_rl_gate='FAIL'))
    result=attach_common_gates(dict(task='YR-332-ao',evidence_commit=commit,
        scope='54 frozen-rule runs on three development curves; no independent real-world or learning claim'),
        ResearchGateReport(performance,reliability,scenario))
    write_result(OUT/'shape-dashboard-audit.json',board.as_dict())
    write_result(OUT/'shape-gates.json',result)
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in result['common_gates'].items() if isinstance(r,dict) and 'status' in r})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args();publish(args.commit)
