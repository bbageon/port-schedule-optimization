"""Pin stock-remediation evidence; do not turn input feasibility into a performance claim."""
import argparse
from pathlib import Path
import re

from check_initial_stock import OUT, read, sha, write_result


def publish(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard, judge_runtime_evidence,
        judge_claim_alignment, combine_reliability, judge_scenario_validity,
        GateOutcome, GateStatus, ResearchGateReport, attach_common_gates)
    summary=read(OUT/'stock-summary.json');campaign=read(OUT/'campaign.json')
    paths=[OUT/n for n in ('README.md','stock-summary.json','input-audit.json','campaign.json',
                           'bank-stock-preserved.json','shape-input-audit.json','verification.json')]
    artifacts={**summary['artifact_hashes'],**{p.as_posix():sha(p) for p in paths}}
    runtime=judge_runtime_evidence(campaign['repro'],artifact_hashes=artifacts)
    dashboard=audit_dashboard('.',task_id='YR-332-at',expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-332-at-arbitrary-seed-stock-feasibility.md'),
        evidence_paths=[OUT/'stock-summary.json',OUT/'input-audit.json',OUT/'campaign.json'],
        evidence_commits=[commit],remote_ref='origin/강화학습-판매',pin_commit=commit)
    body=(OUT/'README.md').read_text(encoding='utf-8')
    patterns=dict(cases=r'\*\*(\d+)개 입력\*\*',changed_cases=r'(\d+)개에서 생기던',
        unchanged_cases=r'정상 입력 \*\*(\d+)개\*\*',old_conversions=r'\*\*([\d,]+)건 전환',
        new_conversions=r'→ (\d+)건\*\*')
    claims={k:int(re.search(pattern,body).group(1).replace(',','')) for k,pattern in patterns.items()}
    claims['normal_minutes']=15.959
    truth={**{k:summary[k] for k in patterns},'normal_minutes':summary['runtime'][2]['truck_minutes']}
    reliability=combine_reliability(runtime,dashboard,judge_claim_alignment(claims,truth,absolute_tolerance=.00051))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,
        physical_constraints=True,ledger_conservation=True,deterministic_replay=True),
        flow_checks={},anchors={},continuous_operation=False,request_real_terminal_claim=False)
    performance=GateOutcome('performance',GateStatus.INCONCLUSIVE,
        '입력 보정 검사이며 새 학습 또는 정책 개선 판정 없음',
        evidence=dict(previous_rl_gate='FAIL',previous_gate_path='outputs/reports/yr332_linked_retraining/current-gates.json'))
    gates=attach_common_gates(dict(task='YR-332-at',evidence_commit=commit,
        input_supply_axis='PASS',scope='finite synthetic initial-state conservation; no real-terminal performance claim'),
        ResearchGateReport(performance,reliability,scenario))
    write_result(OUT/'dashboard-audit.json',dashboard.as_dict())
    write_result(OUT/'current-gates.json',gates)
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:r['status'] for k,r in gates['common_gates'].items() if isinstance(r,dict) and 'status' in r})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args();publish(args.commit)
