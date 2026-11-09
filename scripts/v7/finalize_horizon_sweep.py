"""Freeze the four-K diagnostic verdict after evidence and board are pushed."""
import argparse
from dataclasses import asdict
from pathlib import Path
import re
import subprocess

from benchmark_default_eval import save
from six_step_learning import sha
from horizon_sweep import OUT, SOURCE, KS, LOADS
from report_horizon_sweep import read, ARMS, LABELS
from yard_rl.integrated.evalkit import PairedResult
from yard_rl.experiments.gate_harness import (GateOutcome, GateStatus,
    attach_common_gates, audit_dashboard, judge_claim_alignment, judge_performance,
    judge_runtime_evidence, report_from_dict, revalidate_pass_evidence)


def report_claims(summary):
    actual, shown = {}, {}
    table = {r['arm']: r for r in summary['aggregate']}
    names = {label: metric for metric, label in LABELS.items()}
    comparison = lambda k, m, ctrl='initial', end='24': next(r for r in summary['comparisons']
        if r['k'] == k and r['control'] == ctrl and r['metric'] == f'{m}_mean_s_{end}')
    numbers = lambda text: [float(x) for x in re.findall(r'[+-]?\d+(?:\.\d+)?', text)]
    def add(key, expect, printed):
        assert key not in actual
        actual[key], shown[key] = expect, printed
    for line in (OUT/'README.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.split('|')][1:-1]
        if len(cells) == 6 and cells[0] in names:
            m = names[cells[0]]
            add(f'{m}/initial', table['initial'][m+'_mean_s_24']/60, float(cells[1]))
            for k, cell in zip(KS, cells[2:], strict=True):
                minutes, pct = numbers(cell)
                add(f'{m}/k{k}/minutes', table[f'k{k}'][m+'_mean_s_24']/60, minutes)
                add(f'{m}/k{k}/reduction', comparison(k, m)['reduction_percent']['mean'], pct)
        elif len(cells) == 7 and cells[0].isdigit() and int(cells[0]) in LOADS:
            load = int(cells[0])
            for arm, cell in zip(ARMS, cells[1:], strict=True):
                row = next(r for r in summary['by_load'] if r['arm'] == arm and r['load'] == load)
                add(f'load/{load}/{arm}', row['TRUCK_mean_s_24']/60, float(cell))
        elif len(cells) == 5 and cells[0].isdigit() and int(cells[0]) in KS:
            k = int(cells[0])
            r = comparison(k, 'TRUCK')['reduction_percent']
            for field, v in zip(('mean', 'ci_lo', 'ci_hi'), numbers(cells[1]), strict=True):
                add(f'uncertainty/{k}/{field}', r[field], v)
            for field, cell, ctrl, end in (('k3', cells[2], 'k3', '24'),
                    ('sf_spt', cells[3], 'sf_spt', '24'), ('26h', cells[4], 'initial', '26')):
                add(f'comparison/{k}/{field}', comparison(k, 'TRUCK', ctrl, end)['reduction_percent']['mean'], numbers(cell)[0])
        elif len(cells) == 5 and cells[0] in ARMS:
            arm = cells[0]
            for metric, cell in zip(('TRUCK_unfinished_24', 'VESSEL_DISCHARGE_unfinished_24',
                    'VESSEL_DISCHARGE_not_ready_24', 'VESSEL_LOAD_unfinished_24'), cells[1:], strict=True):
                add(f'backlog/{arm}/{metric}', table[arm][metric], float(cell))
        elif len(cells) == 6 and cells[0].isdigit() and int(cells[0]) in KS:
            k = int(cells[0])
            r = next(r for r in summary['training'] if r['k'] == k)
            for metric, cell in zip(('labeled', 'zero_raw_gap', 'sign_flip_vs_k3', 'truncated'), cells[1:5], strict=True):
                add(f'train/{k}/{metric}', r['totals'][metric], float(cell))
            add(f'train/{k}/effective_h', r['effective_horizon_s']['mean']/3600, float(cells[5]))
    assert len(actual) == 140, len(actual)
    claim = judge_claim_alignment(shown, actual, absolute_tolerance=.00051)
    assert claim.status is GateStatus.PASS, claim
    return claim


def finalize(pin):
    manifest, verified, summary = [read(OUT/p) for p in ('campaign.json', 'verification.json', 'summary.json')]
    assert verified['status'] == 'PASS' and verified['exact_refits'] == 72
    assert verified['runs'] == 147 and verified['previous_runs_exact'] == 63 and verified['new_repeats_exact'] == 3
    for p, h in {**manifest['sources'], **manifest['reused_artifacts'], **verified['artifact_sha256']}.items():
        assert sha(p) == h, p
    assert sha(OUT/'summary.json') == verified['summary_sha256']
    assert sha(OUT/'README.md') == verified['report_sha256']
    claims = report_claims(summary)
    artifacts = {**manifest['sources'], **manifest['reused_artifacts'], **verified['artifact_sha256']}
    artifacts.update({p.as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()
        and p.name not in ('final-verdict.json', 'final-dashboard-audit.json')})
    artifacts.update({p.as_posix(): sha(p) for p in (Path(__file__), Path('scripts/v7/report_horizon_sweep.py'))})
    runtime = judge_runtime_evidence(manifest['repro'], artifact_hashes=artifacts, root='.')
    assert runtime.status is GateStatus.PASS, runtime
    remote = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', '@{upstream}'], encoding='utf-8').strip()
    board = audit_dashboard('.', task_id='YR-323', expected_state='done',
        spec_path='.claude/docs/dashboard-task-specs/YR-323-crane-horizon-vs-load.md',
        evidence_paths=[str(OUT/p) for p in ('README.md', 'summary.json', 'verification.json')],
        evidence_commits=[pin], remote_ref=remote, pin_commit=pin)
    assert board.status is GateStatus.PASS, board
    performance = {}
    for k in KS:
        cost = next(r for r in summary['comparisons'] if r['k'] == k and r['control'] == 'sf_spt' and r['metric'] == 'cost')
        performance[f'k{k}'] = judge_performance(PairedResult(**cost['candidate_minus_control_percent']),
            minimum_improvement=0., interest_effect=1., hard_guards=summary['guards'][f'k{k}']).as_dict()
    statuses = {r['status'] for r in performance.values()}
    combined = GateStatus.FAIL if statuses == {'FAIL'} else GateStatus.INCONCLUSIVE
    gates = dict(performance=GateOutcome('performance', combined,
            '네 K의 규칙 대비 비용·완주·본선 보호를 각각 판정; 결과만으로 K를 자동 채택하지 않음',
            evidence=dict(by_k=performance, selection='no deployment selection')).as_dict(),
        reliability=GateOutcome('reliability', GateStatus.PASS,
            '고정 질문·72회 재학습·147회 평가·전체 주문·반복 실행·보고와 원격 기록 일치',
            evidence=dict(runtime=asdict(runtime), dashboard=asdict(board), claim_alignment=asdict(claims))).as_dict(),
        scenario_validity=GateOutcome('scenario_validity', GateStatus.FAIL,
            '고정 합성 환경의 K 진단 완료; 실제 운반·예약 보정은 여전히 미확정',
            reasons=('YR-332-i transport allocation/travel/handoff unresolved',
                     'YR-332-f actual appointment errors unresolved'),
            evidence=dict(unchanged_environment_sources=read(SOURCE/'campaign.json')['sources'],
                user_authorization='2026-11-10 explicit K1 K2 K3 K6 synthetic diagnostic request')).as_dict())
    verdict = dict(task='YR-323', learning_completed=True, evaluation_completed=True,
        ks=list(KS), policy_adopted=False, experimental_axis='teacher horizon only',
        total_independent_evaluation_inputs=9, common_gates=gates)
    result = attach_common_gates(verdict, report_from_dict(verdict))
    save(OUT/'final-dashboard-audit.json', asdict(board))
    save(OUT/'final-verdict.json', result)
    ok, reason = revalidate_pass_evidence(report_from_dict(result), root='.')
    assert ok, reason
    print({k: v['status'] for k, v in performance.items()}, flush=True)
    print('reliability PASS; scenario calibration FAIL; no K adopted.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    finalize(p.parse_args().pin)
