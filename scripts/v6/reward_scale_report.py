"""Recheck the frozen reward-unit evidence; no policy performance inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates,
    audit_dashboard, combine_reliability, judge_claim_alignment,
    judge_runtime_evidence, judge_scenario_validity,
)
from yard_rl.v6.reward.scaling import REFERENCE, reference_scaling


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def audit(root):
    reference, probe = read(root/'reference/report.json'), read(root/'probe/report.json')
    manifests = {name: read(root/name/'manifest.json') for name in ('reference', 'probe')}
    scale = reference_scaling()
    assert sha(REFERENCE) == sha(root/'reference/reference_cost_trace.json')
    assert scale['scale_krw'] == reference['scale']['scale_krw']
    assert scale == probe['reward_normalization']
    assert reference['status'] == probe['status'] == 'complete'
    assert reference['admitted'] == 16000 and reference['skipped'] == 0
    assert reference['invariants'] and reference['optimizer_updates'] == 0
    assert manifests['reference']['input_sha256'] == sha(
        Path('outputs/reports/yr331_training/calibration-v2/seed-data.json.gz'))
    for name, manifest in manifests.items():
        assert manifest['code']['git_dirty'] is False
        assert manifest['prereg_sha256'] == sha(root/'prereg-executed.md')
        assert read(root/name/'container_contract.json')['passed']
    assert probe['time_s'] == 21600 and len(probe['updates']) == 6
    assert probe['learning_intervals'] == 360 and probe['parameter_l2_change'] > 0
    assert probe['physical_invariants'] and probe['checkpoint_reload_exact']
    assert probe['shaping_reward'] == 0 and probe['performance_claim'] is False
    assert probe['admission']['skipped'] == probe['admission']['vessel_failed'] == 0
    assert all(row['minibatches'] > 0 for row in probe['updates'])
    assert all(math.isfinite(v) for row in probe['updates'] for v in row.values() if isinstance(v, float))
    restored_cost = -probe['team_reward'] * scale['scale_krw']
    assert math.isclose(restored_cost, probe['cost_krw']-probe['initial_cost_krw'], rel_tol=1e-10)
    packaged = read(root/'package-check.json')
    assert packaged['passed'] and packaged['reference_sha256'] == sha(REFERENCE)
    tests = {name: int(re.search(r'(\d+) passed', (root/f'tests-{name}.txt').read_text(encoding='utf-8'))[1])
             for name in ('cpu', 'array')}
    assert tests == dict(cpu=78, array=13)
    result = dict(task='YR-331-d', scope='reward-unit correction and execution check only',
        passed=True, normalization=scale, reference_source=manifests['reference']['code'],
        probe_source=manifests['probe']['code'], tests=tests, tests_passed=sum(tests.values()),
        training_updates=len(probe['updates']), simulation_hours=probe['time_s']/3600,
        parameter_l2_change=probe['parameter_l2_change'], raw_cost_krw=probe['cost_krw'],
        normalized_reward=probe['team_reward'], restored_cost_krw=restored_cost,
        probe_admitted=probe['admission']['admitted'], checkpoint_sha256=sha(root/'probe/policy.pt'),
        performance_evaluated=False, cost_coefficients_validated=False,
        capacity_limit_confirmed=False,
        legacy_labels='Raw shared runtime says generation=v5 and first-day phase=warmup; '
                      'the v6 source commit and learning_intervals/updates identify the actual probe. '
                      'Its manifest explicitly enables training from time zero.')
    write(root/'result.json', result)
    files = [p for p in sorted(root.rglob('*')) if p.is_file()
             and p.name not in ('artifacts.json', 'gates.json', '.gitattributes')]
    write(root/'artifacts.json', dict(files=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p),
        bytes=p.stat().st_size) for p in files]))
    return result


def gates(root, board_commit, remote_ref):
    result = read(root/'result.json')
    index = read(root/'artifacts.json')
    hashes = {str(root/row['path']): row['sha256'] for row in index['files']}
    runs = []
    for name in ('reference', 'probe'):
        manifest = read(root/name/'manifest.json')
        stamp = dict(code=manifest['code'], seeds={name: [manifest['seed']]},
                     prereg=str(root/'prereg-executed.md'), params=manifest)
        outcome = judge_runtime_evidence(stamp, artifact_hashes=hashes)
        runs.append(outcome)
    if any(run.status != GateStatus.PASS for run in runs):
        raise ValueError('A source/data audit failed')
    dashboard = audit_dashboard(Path('.'), task_id='YR-331-d', expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-331-d-reward-normalization.md'),
        evidence_paths=(root/'result.json', root/'report.md'),
        evidence_commits=(board_commit,), remote_ref=remote_ref, pin_commit=board_commit)
    raw = dict(scale_krw=result['normalization']['scale_krw'], tests_passed=result['tests_passed'],
               training_updates=result['training_updates'], simulation_hours=result['simulation_hours'])
    claims = judge_claim_alignment(read(root/'reported-values.json'), raw, absolute_tolerance=.005)
    reliability = combine_reliability(runs[-1], dashboard, claims)
    performance = GateOutcome('performance', GateStatus.INCONCLUSIVE,
        '수식·짧은 학습 실행만 확인했으며 정책 성능 비교는 수행하지 않음',
        ('같은 조건의 기준정책 비교·독립 반복이 없음',), {'performance_evaluated': False})
    scenario = judge_scenario_validity(
        internal_checks={'physical_constraints': True, 'ledger_conservation': True},
        flow_checks={'fixed_measurement_window': True}, anchors={},
        continuous_operation=True, request_real_terminal_claim=False)
    payload = dict(task='YR-331-d', scope='scoped normalization correction; prior YR-331-b gates unchanged',
        result_sha256=sha(root/'result.json'), per_run_source_audits=[r.as_dict() for r in runs],
        unresolved_existing_input_supply_issue='YR-331-c', monetary_coefficients_not_validated=True)
    return attach_common_gates(payload, ResearchGateReport(performance, reliability, scenario))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/reports/yr331_reward_scale'))
    parser.add_argument('--board-commit')
    parser.add_argument('--remote-ref')
    args = parser.parse_args()
    if args.board_commit:
        report = gates(args.root, args.board_commit, args.remote_ref)
        write(args.root/'gates.json', report)
        print(json.dumps({k: report['common_gates'][k]['status']
                          for k in ('performance', 'reliability', 'scenario_validity')}))
    else:
        report = audit(args.root)
        print(json.dumps({k: report[k] for k in ('passed', 'tests_passed', 'training_updates')}))
