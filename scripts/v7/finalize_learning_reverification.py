"""Pin the independent audit; preserve the original performance/calibration verdicts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from reverify_learning_result import OUT, SOURCE, ARMS, read, sha, close
from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, attach_common_gates, audit_dashboard, judge_claim_alignment,
    judge_runtime_evidence, report_from_dict, revalidate_pass_evidence)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',
                          encoding='utf-8', newline='\n')


def write_report():
    verified = read(OUT/'verification.json')
    data = read(OUT/'independent-ledger-audit.json')
    assert verified['status'] == data['status'] == 'PASS'
    assert verified['replay_runs'] == 12 and verified['replay_exact']
    lines = ['# 하루 학습 결과 독립 재검증', '',
        '2026-11-09. YR-332-c의 기존 결과를 다시 검산했다. 정책·환경·평가 지표를 바꾸지 않았다.', '',
        '**학습 전 대비 4.548676% 감소와 부하별 약 1분·3분·8.6분 단축을 재현했다.**',
        '기존 학습 정책보다 3.571907% 느리다는 결과도 유지된다. 이번 수정의 성능 우위는 성립하지 않는다.', '',
        '## 진행사항', '',
        f'- 기존 평가 208회의 주문 장부 {verified["order_rows"]:,}건을 별도 코드로 검산했다.',
        '- 독립 입력은 9개다. 위 건수에는 같은 입력을 여러 정책으로 실행한 기록이 포함된다.',
        '- 오더별 접수·시각 순서·24/26시간 관측 기록·미완료 수를 확인했다. 누락은 0건이다.',
        '- 같은 입력끼리 주문 ID·유형·블록·도착·통지·내부 이동 시각을 대조했다.',
        '- 저장한 가중치 36개를 원래 자료에서 다시 학습한 결과가 정확히 일치했다.',
        '- 세 부하×네 정책 12회를 새 프로세스에서 재실행했다. 원래 결과와 12회 모두 정확히 일치했다.',
        '- 일치 대상은 전체 결과·주문별 기록·후보·선택·사건·작업·본선 지표이며, 실행 속도는 제외했다.',
        f'- 기존 소스·가중치·원자료의 파일 지문 {data["original_hashes_verified"]}개를 확인했다.', '',
        '## 부하별 평균 체류시간', '',
        '단위는 분이다. 학습 정책은 초기값 셋×평가일 셋, 규칙은 평가일 셋의 평균이다.',
        '작업 규칙은 SF-SPT, 즉 실행 가능한 실작업 중 예상 작업시간이 짧은 것을 우선한다.', '',
        '| 하루 오더 | 학습 전 | 기존 학습 | 수정 학습 | 작업 규칙 | 학습 전 대비 단축 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in data['by_load']:
        lines.append(f'| {row["load"]} | '+' | '.join(f'{row[k]:.3f}' for k in (*ARMS, 'saved_minutes'))+' |')
    lines += ['', '단축 시간은 모두 기존 보고와 일치한다. 수정 학습은 세 부하 모두 기존 학습보다 느리다.', '',
        '## 평균 방식과 관측 시간 점검', '',
        '아래는 수정 학습의 변화율(%)이다. 음수는 더 짧음, 양수는 더 길음을 뜻한다.', '',
        '| 비교 기준 | 24시간 입력별 비율 평균 | 24시간 전체 트럭 누적시간 비율 | 26시간 입력별 비율 평균 |',
        '|---|---:|---:|---:|']
    for label, control in (('학습 전', 'initial'), ('기존 학습', 'legacy_force-5'), ('작업 규칙', 'sf_spt')):
        a, b = [next(r for r in data['sensitivity'] if r['control'] == control and r['metric'] == metric)
                for metric in ('turn_s24', 'turn_s26')]
        lines.append(f'| {label} | {a["mean_input_percent"]:.3f} | {a["pooled_truck_time_percent"]:.3f} | {b["mean_input_percent"]:.3f} |')
    lines += ['', '원래 4.55%는 초기값 셋을 먼저 평균하고, 9개 입력 각각의 감소율을 같은 비중으로 평균한 값이다.',
        '모든 트럭의 누적 체류시간을 기준으로 하면 4.26% 감소다. 혼잡한 입력의 비중이 달라져 생기는 차이다.',
        '26시간까지 관측해도 학습 전보다 개선, 기존 학습과 작업 규칙보다 악화라는 방향은 같다.',
        '보조 계산으로 원래 24시간 지표나 판정을 교체하지 않았다.', '',
        '## 해석과 한계', '',
        '- 24시간이 끝날 때 미출차 차량은 그때까지의 시간만 포함한다. 모든 차량이 나갈 때까지의 평균은 아니다.',
        '- 원래 9개 입력 단위의 95% 불확실성 범위도 재현했다: 학습 전 대비 체류 약 3.23~5.87% 감소.',
        '- 초기값에 따라 악화한 모델도 있다. 평균 개선을 모든 모델의 안정적 개선으로 해석하지 않는다.',
        '- 새 12회는 같은 입력의 재현 확인이다. 추가 독립 표본이나 실제 터미널 검증으로 세지 않는다.',
        '- 비용·본선의 집계는 원자료와 대조하고 실제 재실행으로 확인했다. 별도 비용 모델을 검증한 것은 아니다.',
        '- 성능 채택 미통과와 운반·예약 현실성 미확정 판정을 유지한다. 연속 30일 확증을 시작하지 않았다.', '',
        '## 예정사항', '',
        '등록된 YR-332-i는 실제 운반차 투입·이동·인계 시간 보정, YR-332-f는 예약과 실제 도착 차이 보정이다.',
        '실제 항만에 대한 효과를 판단하려면 이 자료 근거가 필요하다. 이번 요청은 재검증까지 완료했다.', '',
        '[고정 계획](../../../docs/research/v7-discharge/learning-result-reverification.md) ·',
        '[독립 계산](independent-ledger-audit.json) · [실행 검증](verification.json) ·',
        '[기존 보고](../yr332_reachable_teacher_corrected/README.md)']
    assert len(lines) <= 200
    (OUT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def finalize(pin):
    manifest = read(OUT/'manifest.json')
    verified = read(OUT/'verification.json')
    calculated = read(OUT/'independent-ledger-audit.json')
    assert verified['status'] == calculated['status'] == 'PASS'
    assert verified['replay_runs'] == 12 and verified['replay_exact']
    assert verified['checkpoint_refits'] == 36
    for p, expected in {**manifest['sources'], **verified['artifact_sha256']}.items():
        assert sha(p) == expected, p
    old_summary = read(SOURCE/'summary.json')
    for row in calculated['comparisons']:
        old = next(r for r in old_summary['comparisons']
                   if (r['control'], r['metric']) == (row['control'], row['metric']))
        for key in ('paired_raw', 'paired_percent'):
            if row[key] is not None:
                close(row[key]['mean'], old[key]['mean'])
    actual, reported = {}, {}
    labels = {'학습 전': 'initial', '기존 학습': 'legacy_force-5', '작업 규칙': 'sf_spt'}
    for line in (OUT/'README.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.split('|')][1:-1]
        if len(cells) == 6 and cells[0].isdigit():
            row = next(r for r in calculated['by_load'] if r['load'] == int(cells[0]))
            for key, value in zip((*ARMS, 'saved_minutes'), cells[1:], strict=True):
                actual[f'load/{row["load"]}/{key}'] = row[key]
                reported[f'load/{row["load"]}/{key}'] = float(value)
        if len(cells) == 4 and cells[0] in labels:
            control = labels[cells[0]]
            for column, metric, field in ((1, 'turn_s24', 'mean_input_percent'),
                (2, 'turn_s24', 'pooled_truck_time_percent'), (3, 'turn_s26', 'mean_input_percent')):
                row = next(r for r in calculated['sensitivity']
                           if r['control'] == control and r['metric'] == metric)
                actual[f'comparison/{control}/{column}'] = row[field]
                reported[f'comparison/{control}/{column}'] = float(cells[column])
    assert len(actual) == 24, len(actual)
    claims = judge_claim_alignment(reported, actual, absolute_tolerance=.00051)
    assert claims.status is GateStatus.PASS, claims
    artifacts = {**read(SOURCE/'campaign.json')['sources'],
                 **read(SOURCE/'verification.json')['artifact_sha256'], **manifest['sources']}
    artifacts.update({p.as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file()
                      and p.name != 'final-verdict.json'})
    artifacts['scripts/v7/finalize_learning_reverification.py'] = sha(__file__)
    runtime = judge_runtime_evidence(manifest['repro'], artifact_hashes=artifacts, root='.')
    assert runtime.status is GateStatus.PASS, runtime
    remote = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', '@{upstream}'], encoding='utf-8').strip()
    board = audit_dashboard('.', task_id='YR-332-c', expected_state='done',
        spec_path='.claude/docs/dashboard-task-specs/YR-332-c-v7-daily-learning-check.md',
        evidence_paths=[str(OUT/p) for p in ('README.md', 'verification.json', 'independent-ledger-audit.json')],
        evidence_commits=[pin], remote_ref=remote, pin_commit=pin)
    assert board.status is GateStatus.PASS, board
    original_gates = read(SOURCE/'final-verdict.json')['common_gates']
    gates = dict(performance=original_gates['performance'],
        scenario_validity=original_gates['scenario_validity'],
        reliability=GateOutcome('reliability', GateStatus.PASS,
            '독립 장부 검산·12회 실제 재실행·36개 가중치 재학습·보고와 원격 기록 일치',
            evidence=dict(runtime=asdict(runtime), dashboard=asdict(board),
                          claim_alignment=asdict(claims))).as_dict())
    verdict = dict(task='YR-332-c-reverification', recomputation='PASS', execution_replay='PASS',
        original_conclusion_changed=False, additional_independent_inputs=0, common_gates=gates)
    result = attach_common_gates(verdict, report_from_dict(verdict))
    save(OUT/'final-verdict.json', result)
    ok, reason = revalidate_pass_evidence(report_from_dict(result), root='.')
    assert ok, reason
    print('Reverification PASS; performance FAIL and scenario validity FAIL unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', help='Finalize gates after pushing this board/results commit; omit to write the report.')
    pin = parser.parse_args().pin
    if pin:
        finalize(pin)
    else:
        write_report()
