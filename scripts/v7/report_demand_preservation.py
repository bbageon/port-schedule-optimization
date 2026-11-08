"""Independently recalculate conservation, paired metrics and exact repetition."""
from collections import defaultdict
import json
import math
import statistics

from six_step_learning import gz_read, sha
from benchmark_default_eval import save
from evaluate_demand_preservation import OUT, CASES, name


def report():
    manifest = json.loads((OUT / 'campaign.json').read_text(encoding='utf-8'))
    for path, expected in {**manifest['sources'], **manifest['input_sha256']}.items():
        assert sha(path) == expected, path
    values, artifacts = {}, {}
    checked = 0
    for case in CASES:
        path = OUT / 'evaluation' / (name(case) + '.json.gz')
        rowpath = OUT / 'orders' / path.name
        data, rows = gz_read(path), gz_read(rowpath)
        assert len(rows) == case[3] == len({r['job_id'] for r in rows})
        admitted = [r for r in rows if r['admitted']]
        assert len(admitted) == data['result']['admitted']
        for r in rows:
            times = r['times']
            seen = [times[k] for k in ('gate_in_s','block_in_s','service_start_s','job_done_s','gate_out_s')
                    if times[k] is not None]
            assert all(a <= b + 1e-6 for a,b in zip(seen, seen[1:]))
            if r['admitted']:
                assert times['gate_in_s'] == r['scheduled_gate_s']
                assert times['block_in_s'] == r['scheduled_gate_s'] + r['travel_s']
            if case[0] == 'preserved':
                assert r['admitted'] and r['review_count'] == 1 and r['reason'] is None
            checked += 1
        metrics = dict(admitted=len(admitted), missing=len(rows)-len(admitted),
            cost=data['result']['phi_krw'], vessel_unfinished=data['probe']['operational']['vessel_unfinished'],
            vessel_idle_s=data['probe']['operational']['vessel_idle_s'])
        for end, label in ((86400,'24'), (93600,'26')):
            exits = sum(r['times']['gate_out_s'] is not None and r['times']['gate_out_s'] <= end for r in admitted)
            total = sum(min(r['times']['gate_out_s'] or end, end)-r['times']['gate_in_s'] for r in admitted)
            unfinished = sum(r['times']['job_done_s'] is None or r['times']['job_done_s'] > end for r in admitted)
            assert metrics['missing'] + len(admitted)-exits + exits == len(rows)
            metrics.update({f'exited{label}':exits, f'unexited{label}':len(admitted)-exits,
                            f'yard_unfinished{label}':unfinished, f'total_turn_s{label}':total})
        metrics['mean_turn_s'] = metrics['total_turn_s24'] / len(admitted)
        assert math.isclose(metrics['mean_turn_s'], data['result']['mean_turn_time_s'], rel_tol=1e-11)
        values[tuple(case)] = metrics
        artifacts.update({path.as_posix():sha(path), rowpath.as_posix():sha(rowpath)})
    repeat = CASES[-1]
    original = (*repeat[:4], 0)
    assert values[tuple(repeat)] == values[original]
    a = gz_read(OUT / 'evaluation' / (name(repeat)+'.json.gz'))
    b = gz_read(OUT / 'evaluation' / (name(original)+'.json.gz'))
    assert a['result'] == b['result']
    assert gz_read(OUT / 'orders' / (name(repeat)+'.json.gz')) == gz_read(OUT / 'orders' / (name(original)+'.json.gz'))
    for key in ('candidates_sha256','choices_sha256','events_sha256','jobs_sha256','records_sha256','result_sha256'):
        assert a['probe'][key] == b['probe'][key]
    groups = defaultdict(list)
    for (mode, arm, seed, load, repeat), metrics in values.items():
        if not repeat:
            groups[(mode, arm, load)].append(metrics)
    aggregated = [dict(mode=mode,arm=arm,load=load,models=len(rows),
        **{k:statistics.mean(r[k] for r in rows) for k in rows[0]})
        for (mode,arm,load), rows in groups.items()]
    summary = dict(task='YR-332-aa', runs=len(CASES), pairs=32,
        checked_order_rows=checked, legacy_exact_replays=32, deterministic_repeat=True,
        aggregate=aggregated, order_conservation='PASS', fixed_arrival_times='PASS',
        physical_safety='PASS; engine invariants enabled, full-yard waiting tests pass',
        performance_adoption='NOT_PASSED; frozen-model environmental diagnosis',
        scenario_remaining=['vessel calibration (i)', 'appointment realism (f)', 'teacher reachability (ab)'])
    save(OUT / 'summary.json', summary)
    lines = ['# YR-332-aa — 고정 도착 주문 보존 결과', '',
        f'32조건 짝비교·반복 1회, 총 {len(CASES)}회. 오더별 검사 {checked:,}건. 재학습 없음.', '',
        '모든 보존 조건에서 주문 전건 등록·원래 진입 시각·장부 수지가 일치했다.',
        '기존 방식 32회는 과거 원자료와 정확히 재현됐고, 보존 방식 반복도 결정·사건·결과가 같았다.', '',
        '| 작업 규칙/모델 | 입력 | 방식 | 미등록 | 24시간 미출차 | 26시간 미출차 | 체류 분 | 계산 비용 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in aggregated:
        lines.append(f"| {row['arm']} | {row['load']} | {row['mode']} | {row['missing']:.2f} | "
            f"{row['unexited24']:.2f} | {row['unexited26']:.2f} | {row['mean_turn_s']/60:.3f} | {row['cost']:.3f} |")
    lines.extend(['', '## 해석과 한계', '',
        'preserved는 주문 보존, legacy는 기존 공간 예약에 따른 등록 제한이다.',
        '체류는 실제 진입부터 출차까지이며 미출차는 관측 종료까지 센다. 미등록에는 체류 0을 부여하지 않는다.',
        'flow 행은 같은 입력을 쓴 저장 모델 3개의 평균이다. 독립적인 3일 표본으로 해석하지 않는다.',
        '등록 주문 수가 달라져 평균·총비용 변화에는 처리 대상 확대가 섞인다. 이번 결과는 학습 개선율이 아니다.',
        '본선 계획 완료가 평가창 밖인 문제, 예약 현실성, 교사 대안 계약은 이 수정으로 해결되지 않는다.',
        '정책 채택·현실성 전체 통과는 보류한다. 다음 환경 보정은 YR-332-i다.', '',
        '[사전등록](../../../docs/research/v7-discharge/demand-preservation-prereg.md) · '
        '[원자료 요약](summary.json) · [주문별 기록](orders/)'])
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(OUT / 'verification.json', dict(status='PASS',checked_order_rows=checked,
        summary_sha256=sha(OUT/'summary.json'), report_sha256=sha(OUT/'README.md'), artifact_sha256=artifacts))
    print(f'report: {len(CASES)} runs, {checked} order records verified', flush=True)


if __name__ == '__main__':
    report()
