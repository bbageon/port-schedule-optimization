"""YR-332-aq research gates against pinned board evidence (README is written from summary.json)."""
import argparse
import json
from pathlib import Path

from benchmark_default_eval import save
from six_step_learning import sha
from retrain_linked_environment import OUT, SEEDS, read


def gates(commits):
    from yard_rl.experiments.gate_harness import (audit_dashboard, judge_runtime_evidence, judge_claim_alignment,
        combine_reliability, judge_scenario_validity, GateOutcome, GateStatus, ResearchGateReport, attach_common_gates)
    s, v, m = read(OUT / 'summary.json'), read(OUT / 'verification.json'), read(OUT / 'campaign.json')
    paths = [OUT / f for f in ('summary.json', 'verification.json', 'campaign.json', 'README.md', 'authorization.json')]
    paths += sorted((OUT / 'evaluation').glob('*.gz')) + sorted((OUT / 'orders').glob('*.gz')) + sorted((OUT / 'flow-orders').glob('*.gz'))
    for seed in SEEDS:
        paths += sorted((OUT / f'model-{seed}').glob('*'))
    runtime = judge_runtime_evidence(m['repro'], artifact_hashes={p.as_posix(): sha(p) for p in paths})
    spec = Path('.claude/docs/dashboard-task-specs/YR-332-aq-linked-environment-retraining.md')
    dashboard = audit_dashboard('.', task_id='YR-332-aq', expected_state='done', spec_path=spec,
        evidence_paths=[OUT / 'summary.json', OUT / 'verification.json', OUT / 'campaign.json'], evidence_commits=commits,
        remote_ref='origin/강화학습-판매', pin_commit=commits[-1])
    text = (OUT / 'README.md').read_text(encoding='utf-8');claims, truth = {}, {}
    for arm in ('initial', 'trained', 'previous', 'SF_SPT'):
        number = s['by_arm'][arm]['TRUCK']['mean_min']
        assert f'{number:.3f}' in text, (arm, number)
        claims[arm] = round(number, 3);truth[arm] = number
    for base in ('initial', 'SF_SPT', 'previous'):
        value = s['reduction_percent']['ALL'][f'trained_vs_{base}']['TRUCK']
        shown = f'{value:+.2f}%'
        assert shown in text or shown.replace('-', '−') in text, (base, value)
        claims[f'reduction:{base}'] = round(value, 2);truth[f'reduction:{base}'] = value
    reliability = combine_reliability(runtime, dashboard, judge_claim_alignment(claims, truth, absolute_tolerance=.00501))
    p = s['performance']
    performance = GateOutcome('performance', GateStatus(p['status']), p['summary'], tuple(p['reasons']), p['evidence'])
    scenario = judge_scenario_validity(internal_checks=dict(event_time_order=True, information_boundary=True,
        physical_constraints=True, ledger_conservation=True, achievable_vessel_deadline=False, deterministic_replay=True),
        flow_checks={}, anchors={}, continuous_operation=False, request_real_terminal_claim=False)
    result = attach_common_gates(dict(task='YR-332-aq', evidence_commit=commits[-1], verification=v,
        claim_scope='User-directed diagnostic retraining with the ag method in the calibrated linked synthetic environment; '
                    'nine new synthetic inputs; no adoption, no real-terminal or continuous-operation claim.'),
        ResearchGateReport(performance, reliability, scenario))
    save(OUT / 'dashboard-audit.json', dashboard.as_dict());save(OUT / 'current-gates.json', result)
    assert reliability.status is GateStatus.PASS, reliability.as_dict()
    print(json.dumps({k: r['status'] for k, r in result['common_gates'].items() if isinstance(r, dict) and 'status' in r}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--commit', action='append', required=True)
    gates(parser.parse_args().commit)
