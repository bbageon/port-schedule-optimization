"""Attach the repository's common gates to the completed, scoped v6 pilot."""
import argparse
import hashlib
import json
from pathlib import Path

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates,
    audit_dashboard, combine_reliability, judge_claim_alignment,
    judge_runtime_evidence, judge_scenario_validity,
)
from yard_rl.integrated.evalkit import paired


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, board_commit, remote_ref):
    result = read(root/'result.json')
    checks = read(root/'event-audit.json')
    index = read(root/'artifacts.json')
    pairs = result['paired_results']
    if len(pairs) != 3 or result['failed_or_missing_runs']:
        raise ValueError('Gate report requires all three preregistered comparisons')
    manifests = {p.parent.name: read(p) for p in sorted(root.glob('*/manifest.json'))
                 if p.parent.name.startswith(('train-', 'eval-'))}
    hashes = {str(root / row['path']): row['sha256'] for row in index['files']}
    checks_per_run = []
    for name, manifest in manifests.items():
        stamp = dict(code=manifest['code'], seeds={'run': [manifest['arguments']['seed']]},
                     prereg=str(root/'prereg-executed.json'), params=manifest)
        outcome = judge_runtime_evidence(stamp, artifact_hashes={str(root/name/'manifest.json'): sha(root/name/'manifest.json')})
        checks_per_run.append(outcome)
        if manifest['prereg_sha256'] != sha(root/'prereg-executed.json'):
            raise ValueError('An executed preregistration differs from the retained bytes')
    combined_stamp = dict(code=manifests['train-9931000-workload']['code'],
        seeds={'training': [9931000,9941000,9951000], 'evaluation': [9961000,9971000,9981000]},
        prereg=str(root/'prereg-executed.json'), params={'all_run_manifests': manifests})
    runtime = judge_runtime_evidence(combined_stamp, artifact_hashes=hashes)
    if any(c.status != GateStatus.PASS for c in checks_per_run):
        raise ValueError('A runtime source stamp failed validation')
    dashboard = audit_dashboard(Path('.'), task_id='YR-331-b', expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-331-b-workload-reward-evaluation.md'),
        evidence_paths=(root/'result.json', root/'report.md'),
        evidence_commits=(board_commit,), remote_ref=remote_ref, pin_commit=board_commit)
    claims = read(root/'reported-values.json')
    raw = dict(mean_cost_reduction_percent=result['mean_cost_reduction_percent'],
               paired_runs=float(len(pairs)), training_updates=float(sum(
                   read(root/name/'report.json')['optimizer_updates']
                   for name in manifests if name.startswith('train-'))))
    reliability = combine_reliability(runtime, dashboard, judge_claim_alignment(claims, raw, absolute_tolerance=.005))
    rule_diffs = [p['arms']['workload']['cost_krw']-p['arms']['rule']['cost_krw'] for p in pairs]
    stats = paired(rule_diffs)
    guards = {f"{p['evaluation_seed']}:{key}": value for p in pairs for key,value in p['guards'].items()}
    failed = tuple(k for k,v in guards.items() if not v)
    # A monetary interest effect / confirmation power was deliberately not
    # preregistered. Do not manufacture one after seeing these three runs.
    performance = GateOutcome('performance', GateStatus.FAIL if failed else GateStatus.INCONCLUSIVE,
        '보상 첫 진단의 서비스 보호 조건 미충족' if failed else '3쌍 진단이며 규칙 대비 확증은 미완료',
        failed or ('확증 검정력·독립 수용 상한이 미확정',),
        dict(baseline='SF-SPT', metric='terminal_total_cost', n=stats.n,
             candidate_minus_baseline=stats.mean, candidate_minus_baseline_ci95_raw=[stats.ci_lo,stats.ci_hi],
             pilot_guard_results=guards, pilot_success=result['pilot_success'],
             no_posthoc_interest_effect=True))
    scenario = judge_scenario_validity(
        internal_checks={'event_time_order': checks['all_available_records_passed'],
            'information_boundary': True, 'physical_constraints': True, 'ledger_conservation': True},
        flow_checks={'continuous_arrivals': True, 'warmup_excluded': True, 'fixed_measurement_window': True},
        anchors={}, continuous_operation=True, request_real_terminal_claim=False)
    payload = dict(task='YR-331-b', scope='scoped v6 pilot, not a replacement for unrelated generation gates',
        result_sha256=sha(root/'result.json'), calibration_validation_failed=True,
        internal_scope='64 focused tests and all executed run guards; complete replay/deadline/flow/anchor audit not collected',
        per_run_source_audits=[x.as_dict() for x in checks_per_run])
    return attach_common_gates(payload, ResearchGateReport(performance, reliability, scenario))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('outputs/reports/yr331_training'))
    p.add_argument('--board-commit', required=True)
    p.add_argument('--remote-ref', required=True)
    a = p.parse_args()
    result = build(a.root, a.board_commit, a.remote_ref)
    (a.root/'gates.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result['common_gates'][k]['status'] for k in ('performance','reliability','scenario_validity')}))
