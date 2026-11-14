"""YR-332-al research gates against pinned board evidence (README is written from summary.json)."""
import argparse
import json
from pathlib import Path

from benchmark_default_eval import save
from six_step_learning import sha
from evaluate_visit_links import OUT, VARIANTS, read


def gates(commits):
    from yard_rl.experiments.gate_harness import (audit_dashboard, judge_runtime_evidence, judge_claim_alignment,
        combine_reliability, judge_scenario_validity, GateOutcome, GateStatus, ResearchGateReport, attach_common_gates)
    s, v, m = read(OUT / 'summary.json'), read(OUT / 'verification.json'), read(OUT / 'campaign.json')
    paths = [OUT / f for f in ('summary.json', 'verification.json', 'campaign.json', 'README.md', 'authorization.json',
        'link-definition.json', 'demand-profiles.json', 'input-preflight.json', 'contract-tests.xml')]
    paths += sorted((OUT / 'evaluation').glob('*.gz')) + sorted((OUT / 'orders').glob('*.gz')) + sorted((OUT / 'flow-orders').glob('*.gz'))
    runtime = judge_runtime_evidence(m['repro'], artifact_hashes={p.as_posix(): sha(p) for p in paths})
    spec = Path('.claude/docs/dashboard-task-specs/YR-332-al-multi-order-visit-link.md')
    dashboard = audit_dashboard('.', task_id='YR-332-al', expected_state='done', spec_path=spec,
        evidence_paths=[OUT / 'summary.json', OUT / 'verification.json', OUT / 'campaign.json'], evidence_commits=commits,
        remote_ref='origin/강화학습-판매', pin_commit=commits[-1])
    text = (OUT / 'README.md').read_text(encoding='utf-8');claims, truth = {}, {}
    for variant in VARIANTS:
        for arm in ('initial', 'trained', 'SF_SPT'):
            number = s['by_variant'][variant][arm]['TRUCK']['mean_min']
            assert f'{number:.3f}' in text, (variant, arm, number)
            claims[variant + ':' + arm] = round(number, 3);truth[variant + ':' + arm] = number
        gap = s['trained_gap_percent'][variant]['ALL']['trained_vs_SF_SPT']['TRUCK']
        assert f'{gap:+.2f}%' in text, (variant, gap)
        claims[f'{variant}:gap'] = round(gap, 2);truth[f'{variant}:gap'] = gap
    reliability = combine_reliability(runtime, dashboard, judge_claim_alignment(claims, truth, absolute_tolerance=.00501))
    performance = GateOutcome('performance', GateStatus.INCONCLUSIVE,
        '고정 정책의 방문 연결 민감도 진단이며 정책 성능 확증을 수행하지 않음',
        ('물량별 입력 하나이며 저장 모델은 연결 없는 입력으로 학습됨',),
        dict(metric='terminal_total_cost', baseline='SF-SPT', new_training=False, no_adoption=True,
             conditional_results=s['trained_gap_percent']))
    scenario = judge_scenario_validity(internal_checks=dict(event_time_order=True, information_boundary=True,
        physical_constraints=True, ledger_conservation=True, achievable_vessel_deadline=False, deterministic_replay=True),
        flow_checks={}, anchors={}, continuous_operation=False, request_real_terminal_claim=False)
    result = attach_common_gates(dict(task='YR-332-al', evidence_commit=commits[-1], verification=v,
        claim_scope='Multi-container visits linked with the observed pattern mix, chassis capacity, inbound-first stops and one exit; '
                    '3,500 inputs match block and visit clustering; larger loads are sensitivity; transport (i), queue definition (f), '
                    'day variation (ao) and high-volume bunching (ap) remain.'),
        ResearchGateReport(performance, reliability, scenario))
    save(OUT / 'dashboard-audit.json', dashboard.as_dict());save(OUT / 'current-gates.json', result)
    assert reliability.status is GateStatus.PASS, reliability.as_dict()
    print(json.dumps({k: r['status'] for k, r in result['common_gates'].items() if isinstance(r, dict) and 'status' in r}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--commit', action='append', required=True)
    gates(p.parse_args().commit)
