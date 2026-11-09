"""Pin the completed diagnostic without turning unresolved calibration into PASS."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from benchmark_default_eval import save
from six_step_learning import sha
from retrain_reachable_teacher import OUT
from yard_rl.integrated.evalkit import PairedResult
from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, attach_common_gates, audit_dashboard,
    judge_claim_alignment, judge_performance, judge_runtime_evidence, report_from_dict)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finalize(pin):
    summary, verified, manifest = [read(OUT/p) for p in
                                   ('summary.json','verification.json','campaign.json')]
    assert sha(OUT/'README.md') == verified['report_sha256']
    assert sha(OUT/'summary.json') == verified['summary_sha256']
    for path, expected in {**verified['artifact_sha256'], **manifest['sources']}.items():
        assert sha(path) == expected, path
    artifacts = {p.as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()
                 and p.name not in ('final-verdict.json','final-dashboard-audit.json')}
    artifacts.update(manifest['sources'])
    for p in ('scripts/v7/report_reachable_teacher.py','scripts/v7/finalize_reachable_teacher.py',
              'tests/v7/test_reachable_teacher.py','tests/v7/test_crane_count_transform.py'):
        artifacts[p] = sha(p)
    runtime = asdict(judge_runtime_evidence(manifest['repro'], artifact_hashes=artifacts, root='.'))
    assert runtime['status'] == 'PASS', runtime
    actual, reported = {}, {}
    table = {(r['arm'],r['load']):r for r in summary['aggregate']}
    for line in (OUT/'README.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.split('|')][1:-1]
        if len(cells) == 4 and cells[0].isdigit():
            seed = int(cells[0])
            for control,value in zip(('initial','legacy_force-5','sf_spt'),cells[1:],strict=True):
                label = f'model/{seed}/{control}'
                actual[label] = next(r['mean_percent'] for r in summary['model_turn_change_percent']
                    if r['model_seed']==seed and r['control']==control)
                reported[label] = float(value)
        if len(cells) == 6 and cells[0] in ('legacy_force','reachable'):
            for key,value in zip(('sampled','labeled','no_witness','force_failed','truncated_horizon_pairs'),cells[1:],strict=True):
                label = f'teacher/{cells[0]}/{key}'
                actual[label] = summary['training']['totals'][cells[0]][key]
                reported[label] = float(value)
        if len(cells) == 7 and cells[1].isdigit():
            arm,load = cells[0],int(cells[1])
            for key,value in zip(('unexited24','unexited26','yard_unfinished26','vessel_unfinished','vessel_idle_s'),cells[2:],strict=True):
                label = f'completion/{load}/{arm}/{key}'
                actual[label] = table[(arm,load)][key]/(3600 if key=='vessel_idle_s' else 1)
                reported[label] = float(value)
        if len(cells) == 6 and cells[0].isdigit():
            load = int(cells[0])
            for arm, value in zip(('initial','legacy_force-5','reachable-2','reachable-5','sf_spt'),
                                  cells[1:], strict=True):
                label = f'{load}/{arm}'
                actual[label], reported[label] = table[(arm,load)]['turn_s24']/60, float(value)
        if len(cells) == 5 and any(r['control']==cells[0] for r in summary['comparisons']):
            for i, metric in enumerate(('turn_s24','cost','unexited24','unexited26')):
                row = next(r for r in summary['comparisons'] if
                           r['control']==cells[0] and r['metric']==metric)
                label = f'{cells[0]}/{metric}'
                actual[label] = row['paired_percent' if i<2 else 'paired_raw']['mean']
                reported[label] = float(cells[i+1])
    assert len(actual) == 134, len(actual)
    claims = asdict(judge_claim_alignment(reported, actual, absolute_tolerance=.00051))
    assert claims['status'] == 'PASS', claims
    remote = subprocess.check_output(['git','rev-parse','--abbrev-ref','@{upstream}'],encoding='utf-8').strip()
    boards = {}
    for task, stem in (('YR-332-ab','reachable-teacher-alternatives'),
                       ('YR-332-c','v7-daily-learning-check')):
        boards[task] = asdict(audit_dashboard('.', task_id=task, expected_state='done',
            spec_path=f'.claude/docs/dashboard-task-specs/{task}-{stem}.md',
            evidence_paths=[str(OUT/p) for p in ('summary.json','README.md','verification.json')],
            evidence_commits=[pin], remote_ref=remote, pin_commit=pin))
        assert boards[task]['status'] == 'PASS', boards[task]
    cost = next(r for r in summary['comparisons'] if r['control']=='sf_spt' and r['metric']=='cost')
    performance = judge_performance(PairedResult(**cost['paired_percent']),
        minimum_improvement=0., interest_effect=1., hard_guards=summary['guards']).as_dict()
    performance['evidence']['units'] = 'percent of paired SF-SPT total cost; 9 input days, model seeds averaged'
    gates = dict(performance=performance,
        reliability=GateOutcome('reliability', GateStatus.PASS, '코드·독립 재계산·주문 장부·보고·원격 상태 일치',
            evidence=dict(runtime=runtime, dashboard=boards['YR-332-ab'],
                daily_learning_dashboard=boards['YR-332-c'], claim_alignment=claims)).as_dict(),
        scenario_validity=GateOutcome('scenario_validity', GateStatus.FAIL,
            '교사 선택 계약 보정 완료; 고정 합성 환경 진단이며 실측 운반·예약 보정은 미확정',
            reasons=['YR-332-i active transport allocation, travel/return/handoff data missing',
                     'YR-332-i LOAD approximation and achievable deadlines unconfirmed',
                     'YR-332-f appointment realism']).as_dict())
    verdict = dict(task='YR-332-ab-c', teacher_contract='PASS',
        retraining_completed=True, evaluation_completed=True, calibration='UNCONFIRMED',
        training_scope='controlled common-behaviour diagnostic; final iteration fixed in advance',
        teacher_scope='shared input-score intervention through the existing resolver; network representability not proved',
        policy_architecture_changed=False, common_gates=gates)
    save(OUT/'final-dashboard-audit.json', boards)
    save(OUT/'final-verdict.json', attach_common_gates(verdict, report_from_dict(verdict)))
    print(json.dumps(dict(teacher='PASS', performance=performance['status'], reliability='PASS',
                          scenario_validity='FAIL'), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    finalize(p.parse_args().pin)
