"""Build the bounded diagnostic report and check its numerical evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from yard_rl.experiments.gate_harness import judge_claim_alignment


def main():
    out = Path('outputs/reports/yr332_dispatch_reuse')
    load = lambda n: json.loads((out/n).read_text(encoding='utf-8'))
    latency, stream = load('latency.json'), load('stream.json')
    assert not latency['environment']['tracked_dirty'] and not stream['tracked_dirty']
    assert len(latency['rows']) == 22 and len(stream['rows']) == 3
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    # The benchmark recorded __file__ as absolute; resolve its known repo path
    # so the evidence can also be verified from another checkout.
    source_hashes = {('scripts/v7/benchmark_dispatch_reuse.py'
        if path.replace('\\', '/').endswith('/scripts/v7/benchmark_dispatch_reuse.py') else path): expected
        for path, expected in latency['sha256'].items()}
    for path, expected in source_hashes.items():
        assert digest(Path(path)) == expected, path
    for row in latency['rows']:
        for arm in row['arms'].values():
            assert arm['same_candidates'] and arm['same_assignment']
            assert len(arm['samples_ms']) == 20
            assert statistics.median(arm['samples_ms']) == arm['median_ms']
    for row in stream['rows']:
        assert row['same_trajectory_and_cost']
        assert all(a['remaining'] == 0 and len(a['samples_ms']) == 5 for a in row['arms'].values())
    suites = [r.attrib for r in ET.parse(out/'tests.xml').getroot().iter('testsuite')]
    assert sum(int(s['failures'])+int(s['errors']) for s in suites) == 0, suites
    tests = sum(int(s['tests']) for s in suites)
    find = lambda case, history=0: next(r for r in latency['rows'] if
        (r['orders'], r['history'], r['case']) == (1000, history, case))
    for case in ('crane_move', 'yard_change', 'reservation'):
        stats = find(case)['arms']['batch_index_plans']['cache_delta']
        assert stats['plan_misses'] == 1000 and stats['plan_hits'] == 0
    stats = find('arrival')['arms']['batch_index_plans']['cache_delta']
    assert stats['plan_hits'] == 1000 and stats['plan_misses'] == 1
    before, after = 'batch_common', 'batch_index_plans'
    reductions = [(1-r['arms'][after]['median_ms']/r['arms'][before]['median_ms'])*100 for r in stream['rows']]
    all_reductions = [(1-r['arms'][after]['median_ms']/r['arms']['scalar_scan']['median_ms'])*100 for r in stream['rows']]
    reuse_counts = ', '.join(f"{r['arms'][after]['cache']['plan_hits']}/{r['arms'][after]['cache']['plan_misses']}"
                             for r in stream['rows'])
    lines = ['# 상태별 목록·작업 계획 재사용 평가 — 2026-11-05', '',
        'YR-332-j. 연구 실행 경로에 상태 목록, 일괄 평가, 계획·슬롯 재사용을 적용했다.',
        '**같은 작업을 고르면서 계산 시간이 줄었다. 터미널 대기나 비용 자체를 줄인 연구 결과는 아니다.**', '',
        '## 진행사항', '',
        '- 기존 오더 상태를 삽입·변경·취소·삭제·이동에 맞춰 관리한다. 후보 검색에서 완료 기록을 제외한다.',
        '- 공통 정보와 같은 입력의 신경망 계산을 공유한다. 점수는 매 결정 다시 계산한다.',
        '- 야드·크레인·예약·작업의 물리 속성이 같을 때 계획을 재사용하고 시작 시각은 갱신한다.',
        '- 야드 변경은 블록 전체, 크레인 위치 변경은 해당 크레인의 계획을 다시 계산한다.',
        '- 적재 위치 탐색도 생성 안에서 공유한다. 최종 공동 계획·충돌 검사·실제 예약은 원래 경로로 수행한다.',
        '- 사용자 정의 위치 선택 함수는 재사용하지 않는다. 복제·저장 복원은 상태 목록을 재연결하고 캐시는 비운다.',
        '- 입력 9칸의 의미·보상·실작업 우선·필수/강제 선택·운영 기본 후보 상한 12는 유지한다.',
        '[구현 계약](../../../docs/research/v7-discharge/dispatch-reuse.md) · [사전등록](../../../docs/research/v7-discharge/dispatch-cache-prereg.md).', '',
        '## 결정 한 번의 처리시간', '',
        'Intel Core Ultra 9 285K, CPU 1스레드, Python 3.13.9, PyTorch 2.14.0+cpu.',
        '시드 9,903,321. 같은 상태에서 네 구현의 실행 순서를 섞어 조건별 20회 측정했다.',
        '단위 ms는 1,000분의 1초이며 아래 값은 중앙값이다. 최초 계산도 모델 로딩 시간은 제외한다.',
        '오더 1,000건을 모두 후보로 포함한 진단이다. 실제 행동 실행·통신은 제외하고 상태 변경과',
        '후보 생성→입력 준비→점수→공동 실행 가능성 검사를 포함한다.', '',
        '| 상태 | 기존 개별 계산(ms) | 일괄+공통 정보(ms) | 상태 목록까지(ms) | 계획·슬롯 재사용까지(ms) |',
        '|---|---:|---:|---:|---:|']
    labels = dict(cold='최초 계산', unchanged='동일 상태', arrival='신규 도착 1건', cancel='취소 1건',
                  crane_move='크레인 이동', yard_change='적재 상태 변경', reservation='예약 변경')
    reported, raw = {}, {}
    for case, label in labels.items():
        r = find(case)
        vals = [a['median_ms'] for a in r['arms'].values()]
        lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
        for arm, v in zip(r['arms'], vals):
            reported[case+'_'+arm], raw[case+'_'+arm] = round(v, 3), v
    lines += ['', '신규 도착에서는 기존 계획 1,000개를 재사용하고 새 1개만 계산했다.',
        '크레인·야드·예약 변경에서는 1,000개를 모두 다시 계산했다. 그래도 같은 적재 위치를',
        '반복해서 찾는 계산을 공유하므로 최초/재계산 조건에서도 이득이 있었다.',
        '오더가 모두 현재 대기 중이면 상태 목록만 추가한 효과는 거의 없다.', '']
    h = find('unchanged', 10000)['arms']
    lines += [f"완료 기록 10,000건을 추가하면 일괄 계산은 {h[before]['median_ms']:.3f}ms, "
              f"상태 목록 추가는 {h['batch_index']['median_ms']:.3f}ms, "
              f"계획 재사용까지 적용하면 {h[after]['median_ms']:.3f}ms였다.",
        '목록 관리는 완료 이력이 많을 때 의미가 있었으며, 모든 조건에서 큰 속도 향상을 보장하지 않는다.',
        '12/100개 조건과 원시 시간·상태 변경 비용·느린 쪽 통계는 [latency.json](latency.json)에 보존했다.', '',
        '## 작업이 계속 진행되는 경우', '',
        '두 크레인·외부 반입/반출·본선 양하/적하·장비 정지/복구·본선 계획 변경이 있는 작은 환경이다.',
        '기존 7작업에 반입 60건을 40초 간격으로 추가했다. 세 무작위 모델, 네 구현을 각각 5회 실행했다.',
        '기본 후보 상한 12를 양쪽에 유지했다. 아래는 한 결정이 아니라 **전체 시뮬레이션의 계산시간**이다.', '',
        '| 모델 시드 | 개별 계산(ms) | 일괄+공통 정보(ms) | 상태 목록까지(ms) | 계획·슬롯 재사용까지(ms) |',
        '|---|---:|---:|---:|---:|']
    for r in stream['rows']:
        vals = [a['median_ms'] for a in r['arms'].values()]
        lines.append('| '+str(r['seed'])+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
        for arm, v in zip(r['arms'], vals):
            reported[str(r['seed'])+'_'+arm], raw[str(r['seed'])+'_'+arm] = round(v, 3), v
    lines += ['', f'기존 개별 계산보다 {min(all_reductions):.1f}~{max(all_reductions):.1f}%, '
        f'일괄 계산만 적용한 방식보다 추가로 {min(reductions):.1f}~{max(reductions):.1f}% 짧았다.',
        '세 모델 모두 네 구현의 사건 기록·작업 시작/완료·비용 결과 해시가 같았고 67작업이 모두 완료됐다.',
        f'계획 재사용/신규 계산은 모델별 {reuse_counts}회였다. 작업이 진행되면 계획이',
        '자주 무효화되므로, 정적인 반복 상태의 빠른 시간을 실제 운영 전체 속도로 해석하면 안 된다.',
        '여기서는 반복 적재 위치 탐색과 신경망 입력 준비를 줄인 효과가 함께 남았다.',
        '[연속 실행 원자료](stream.json).', '', '## 검증·재현·판정', '',
        f'- 관련 검사 **{tests} 통과**, 실패 0: [JUnit](tests.xml). 저장소 전체 시험을 뜻하지 않는다.',
        '- 22조건×4구현의 후보/계획·선택 일치, 세 모델×4구현의 연속 궤적·비용 일치를 확인했다.',
        '- 처음에는 기존 시험 관측 함수가 새 공통 정보 인자를 받지 못했다. 관측 조건은 유지하고',
        '  함수 인자 전달만 고쳤다. 초기 31통과/1실패 기록과 교사 관측 검사 중단 경위는 보존한다.',
        '- 연속 결과 첫 저장은 ActionMix 객체의 JSON 직렬화 오류로 실패했다. 계산식 변경 없이 저장 형식만 고쳤다.',
        f"- 결정 지연 실행 커밋 `{latency['environment']['commit']}`; 연속 실행 `{stream['commit']}`. 둘 다 시작 시 추적 파일 변경 없음.",
        '- source/artifact 해시와 수치 대조는 verification.json, 원격 반영 뒤 작업 목록 감사는 dashboard-audit.json.',
        '- 구현 동등성과 계산 지연 개선은 이 진단 범위에서 확인했다. 새 학습이나 비용 개선 판정은 하지 않았다.',
        '- 실제 차량 대수·인계시간·본선 마감의 현실 보정은 YR-332-i에 남아 있다.', '',
        '## 한계와 예정사항', '',
        '- 야드 변경 감지는 한 번 전체 적재 상태를 읽는 보수적 방식이며 특정 스택만 추적하는 구현은 아니다.',
        '- 한 결정의 많은 오더 진단은 빈 야드·반입 위주다. 연속 진단도 67작업의 작은 환경이다.',
        '- 새 예정/현재 집합 인코더, 학습된 다양한 모델, 42크레인 동시 부하·큰 혼잡 야드·현장 통신은 미검증.',
        '- YR-332-l: 큰 혼잡 야드와 여러 크레인이 겹쳐 판단할 때 시간·메모리를 잰다. 적용 규모의 한계를 확인하기 위해서다.',
        '- YR-332-i: 운반 용량·인계시간·본선 마감을 보정한다. 실제 처리 능력과 학습 효과를 판단하기 위한 선결이다.', '',
        '## 재현 (저장소 루트, PYTHONPATH=src)', '', '```text',
        'python scripts/v7/benchmark_dispatch_reuse.py --out outputs/reports/yr332_dispatch_reuse/latency.json',
        'python scripts/v7/benchmark_dispatch_stream.py --out outputs/reports/yr332_dispatch_reuse/stream.json',
        'python scripts/v7/summarize_dispatch_reuse.py', '```']
    assert len(lines) <= 200
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    alignment = judge_claim_alignment(reported, raw, absolute_tolerance=.000501)
    assert alignment.status.value == 'PASS'
    result = dict(schema='yard.v7.dispatch-reuse-verification.v1', claim_alignment=alignment.as_dict(),
        test_suites=suites, snapshot_conditions=22, snapshot_arms=88, stream_runs=60,
        source_sha256=source_hashes,
        artifact_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()
                        and p.name not in ('verification.json','dashboard-audit.json')},
        scope=dict(computation_equivalence='PASS in tested conditions',
            policy_performance='INCONCLUSIVE: no terminal cost improvement claim',
            scenario_validity='INCONCLUSIVE: physical calibration pending',
            authorization_for_new_experiment=False))
    (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(f'PASS: {tests} tests; 88 snapshot comparisons; 60 stream runs; {len(reported)} numerical claims.')


if __name__ == '__main__':
    main()
