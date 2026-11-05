"""Verify paired daily evidence and archive raw decision timing without loss."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from yard_rl.experiments.gate_harness import judge_claim_alignment

OUT = Path('outputs/reports/yr332_default_eval')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    campaign = json.loads((OUT/'campaign/campaign.json').read_text(encoding='utf-8'))
    assert len(campaign['rows']) == 9 and not campaign['tracked_dirty']
    for path, expected in campaign['source_sha256'].items():
        assert sha(Path(path).read_bytes()) == expected, path
    archives, pairs = {}, []
    for row in campaign['rows']:
        assert all(row['equal'].values()), row
        arms = {}
        for mode, info in row['arms'].items():
            path = Path(info['path'])
            archive = path.with_suffix('.json.gz')
            raw = path.read_bytes() if path.exists() else gzip.decompress(archive.read_bytes())
            assert sha(raw) == info['sha256'], path
            data = json.loads(raw)
            assert data['result']['policy_exceptions'] == 0
            assert data['result']['rollout_worlds'] == 0 and data['result']['n_crane_labels'] == 0
            p = data['probe']
            assert p['metadata']['blocks'] == 21 and p['metadata']['cranes'] == 42
            assert len(p['raw_generation_ns']) == len(p['raw_selection_ns']) == p['decision']['count']
            archive.write_bytes(gzip.compress(raw, mtime=0))
            archives[archive.as_posix()] = dict(sha256=sha(archive.read_bytes()), raw_sha256=sha(raw))
            arms[mode] = data
        a, b = arms['reference'], arms['optimized']
        for field in row['equal']:
            if field == 'input':
                assert a['probe']['metadata']['input_sha256'] == b['probe']['metadata']['input_sha256']
            elif field == 'model':
                assert a['model_sha256'] == b['model_sha256']
            else:
                assert a['probe'][field] == b['probe'][field], field
        assert a['result'] == b['result']
        pairs.append(dict(load=row['load'], seed=row['seed'], arms=arms,
            reduction_percent=(1-b['wall_seconds']/a['wall_seconds'])*100))
    suites = []
    for name in ('focused-tests.xml', 'regression-tests.xml'):
        suites += [r.attrib for r in ET.parse(OUT/name).getroot().iter('testsuite')]
    assert sum(int(s['errors'])+int(s['failures']) for s in suites) == 0
    tests = sum(int(s['tests']) for s in suites)
    faster = sum(p['reduction_percent'] > 0 for p in pairs)
    slower_tail = sum(p['arms']['optimized']['probe']['decision']['p95_ms'] >
                      p['arms']['reference']['probe']['decision']['p95_ms'] for p in pairs)
    report = ['# 기본 하루 환경의 평가 속도 비교 — 2026-11-05', '',
        'YR-332-l. 21블록·42크레인에서 현재 9입력 신경망의 계산 최적화를 비교했다.',
        '학습·교사·재배정 없이 같은 입력/모델의 개별 계산과 전체 개선을 9쌍 실행했다.',
        f'전체 계산은 9쌍 중 {faster}쌍에서 짧아졌다. 결정 시간의 느린 5% 경계는 {slower_tail}쌍에서 길어졌다.',
        '이 결과는 계산 동등성과 속도 진단이며 학습 효과나 실제 항만 성능의 판정은 아니다.', '',
        '## 진행사항', '',
        '- 기본 하루 실행 경로에 optimized/reference/batch 평가 설정과 관측자를 연결했다.',
        '- 기본값 optimized는 상태 목록·계획/슬롯 재사용·일괄 추론을 사용한다.',
        '- reference는 개별 추론·전체 목록 탐색·계획 재사용 끔이다. 정렬키 공유는 양쪽 공통이다.',
        '- 관측/대조 설정을 교사와 함께 켜면 거절한다. 학습 기본 동작은 유지한다.',
        '- 관측 24시간+배수 2시간, 초기 적재율 기본 0.65, CPU 1스레드·새 프로세스 순차 실행.',
        '- 입력/모델 시드 3개, 각각 3,500/7,500/15,000 트럭과 기존 본선 생성 입력을 사용했다.',
        '- 호환되는 v7 학습 체크포인트가 없어 고정 무작위 모델 세 개를 사용했다.',
        '- 후보 기본값 12와 필수 후보 초과 보존을 양쪽 유지했다. 후보 제한 해제 시험은 아니다.', '',
        '## 시간·메모리', '',
        '전체 시간은 준비·실행·계측·최종 장부를 포함하고 모델 로딩과 현장 통신은 제외한다.',
        '아래 초 단위 값은 같은 입력 3개의 산술평균이다. 단축률 범위는 각 입력의 짝비교 결과다.', '',
        '| 하루 트럭 | 개별 계산 전체(초) | 개선 전체(초) | 입력별 단축률 범위 |',
        '|---:|---:|---:|---:|']
    claims, raw_claims = {}, {}
    for load in (3500,7500,15000):
        group = [p for p in pairs if p['load'] == load]
        vals = [statistics.mean(p['arms'][m]['wall_seconds'] for p in group) for m in ('reference','optimized')]
        reduction = [p['reduction_percent'] for p in group]
        report.append(f'| {load:,} | {vals[0]:.3f} | {vals[1]:.3f} | {min(reduction):.1f}~{max(reduction):.1f}% |')
        for mode, value in zip(('reference','optimized'), vals):
            claims[f'{load}-{mode}'], raw_claims[f'{load}-{mode}'] = round(value,3), value
    report += ['', '결정 시간은 한 블록에서 가능한 크레인 작업을 고르는 시간이다. 후보 생성부터 공동 안전 검사까지이며',
        '실제 작업 적용·기록 해시 계산은 제외한다. 42크레인 동시 서버 요청의 응답시간이 아니다.', '',
        '| 트럭/입력 시드 | 결정 중앙값(ms), 기존→개선 | 느린 5% 경계(ms), 기존→개선 | 최대 결정(ms), 기존→개선 | 프로세스 최대 실제 메모리(MiB), 기존→개선 |',
        '|---|---:|---:|---:|---:|']
    for p in pairs:
        a,b = [p['arms'][m] for m in ('reference','optimized')]
        ad,bd = a['probe']['decision'],b['probe']['decision']
        report.append(f"| {p['load']:,}/{p['seed']} | {ad['median_ms']:.3f}→{bd['median_ms']:.3f} | "
            f"{ad['p95_ms']:.3f}→{bd['p95_ms']:.3f} | "
            f"{ad['max_ms']:.3f}→{bd['max_ms']:.3f} | "
            f"{a['memory_after']['peak_rss_bytes']/2**20:.1f}→{b['memory_after']['peak_rss_bytes']/2**20:.1f} |")
    report += ['', '메모리는 PyTorch·시뮬레이터·계측 버퍼를 모두 포함한다. 캐시 단독 사용량이 아니다.',
        '1ms는 0.001초, MiB는 약 105만 바이트다. 현장 응답시간·메모리 예산이 없어 절대적인 실시간 적합 판정은 하지 않는다.', '',
        '## 계산 단계별 변화', '',
        '아래는 각 입력의 하루 누적 계산시간이다. 후보 생성에는 상태 대조·계획 재계산·재사용이 함께 포함된다.',
        '개별 하위 함수의 원인은 별도 계측 전 단정하지 않는다. 신경망/선택 절감이 후보 생성 증가로 상쇄되는지 본다.', '',
        '| 트럭/입력 시드 | 후보 생성 누적(초), 기존→개선 | 점수·선택 누적(초), 기존→개선 | 계획 재사용/신규 계산 |',
        '|---|---:|---:|---:|']
    for p in pairs:
        a,b = [p['arms'][m]['probe'] for m in ('reference','optimized')]
        report.append(f"| {p['load']:,}/{p['seed']} | {a['generation']['total_s']:.3f}→{b['generation']['total_s']:.3f} | "
            f"{a['selection']['total_s']:.3f}→{b['selection']['total_s']:.3f} | "
            f"{b['cache']['plan_hits']}/{b['cache']['plan_misses']} |")
    report += ['',
        '## 동등성·실제 부하', '',
        '9쌍 모두 입력·모델·후보 계획·선택·사건·완료 상태/시각·오더 기록·운영 비용이 정확히 같았다.',
        '완료·잔여·투입 수는 입력별로 아래에 남긴다. 계산 결과가 같다는 것이 운영 성능 통과를 뜻하지 않는다.', '',
        '트럭 미완료는 투입된 작업 기준이며, 미투입 수는 요청 수−투입 수다. 서로 합쳐 숨기지 않는다.', '',
        '| 트럭/입력 시드 | 투입 | 트럭 완료/미완료 | 본선 완료/미완료 | 최대 트럭 대기열(한 블록) | 최대 후보(한 크레인) |',
        '|---|---:|---:|---:|---:|---:|']
    for p in pairs:
        a=p['arms']['optimized']; probe=a['probe']
        counts=probe['flow_status_counts']
        count = lambda prefix, done: sum(v for key,v in counts.items()
            if key.startswith(prefix) and (key.endswith('/DONE') == done))
        report.append(f"| {p['load']:,}/{p['seed']} | {a['result']['admitted']} | "
            f"{count('GATE_', True)}/{count('GATE_', False)} | "
            f"{count('VESSEL_', True)}/{count('VESSEL_', False)} | "
            f"{probe['max_waiting_trucks_after_dispatch']} | {probe['max_candidates_per_crane']} |")
    report += ['', f'관련 검사 {tests}개 통과·실패 0. 저장소 전체 회귀를 뜻하지 않는다.',
        '원시 결정 시간·시간대별 통계·계획 재사용 횟수는 campaign/*.json.gz에 손실 없이 보존한다.',
        'archive-index.json의 압축/원본 해시로 검증한다. campaign.json은 짝비교 요약과 원본 해시다.',
        f"실행 커밋 `{campaign['commit']}`, 추적 파일 변경 없는 상태에서 시작했다.", '',
        '## 해석·한계·예정사항', '',
        '- 처리 속도가 빨라도 신경망의 작업 판단이 더 좋아진 것은 아니다. 모델은 전후 고정했다.',
        '- 단 하루·세 입력/모델이며 학습 모델·여러 날 누적 이력·후보 제한 해제·실제 통신은 미검증이다.',
        '- 입력마다 두 구현을 한 번씩 실행했다. 같은 입력의 시간 반복이 아니므로 작은 시간 차이의 통계적 확증은 아니다.',
        '- 블록들은 시뮬레이션 사건 순서로 처리된다. 42개 실제 동시 요청의 서버 부하 시험과 구분한다.',
        '- YR-332-i: 양하 운반 용량·인계시간·마감을 보정해 실제 처리 능력 해석의 근거를 맞춘다.',
        '- YR-332-n: 후보 생성의 단계별 비용을 나눠 증가 원인을 확인하고 같은 작업 결과로 계산을 보정한다.',
        '- YR-332-m: 요청 오더와 실제 투입 수의 차이·작업 잔여 원인을 사유별 장부와 대조한다.',
        '- YR-332-c: 학습 구조와 학습 효과는 별도 과제다. 이번에는 학습을 실행하지 않았다.', '',
        '## 재현', '', '```text',
        'PYTHONPATH=src python scripts/v7/benchmark_default_eval.py --out outputs/reports/yr332_default_eval/campaign',
        'PYTHONPATH=src python scripts/v7/summarize_default_eval.py', '```']
    assert len(report)<=200
    (OUT/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8', newline='\n')
    write(OUT/'archive-index.json', archives)
    alignment=judge_claim_alignment(claims,raw_claims,absolute_tolerance=.000501)
    assert alignment.status.value=='PASS'
    write(OUT/'verification.json', dict(schema='yard.v7.daily-evaluation-verification.v1',
        pairs=9, runs=18, test_suites=suites, claim_alignment=alignment.as_dict(),
        sources=campaign['source_sha256'], artifact_sha256={
            p.as_posix():sha(p.read_bytes()) for p in [OUT/'campaign/campaign.json',OUT/'archive-index.json',
                OUT/'report.md',OUT/'focused-tests.xml',OUT/'regression-tests.xml']},
        scope=dict(equivalence='PASS in tested conditions', policy_performance='INCONCLUSIVE; not evaluated',
                   scenario_validity='INCONCLUSIVE; calibration pending',training_performed=False)))
    print(f'PASS: 9 exact paired runs, {tests} tests, archived raw timings')


if __name__ == '__main__':
    main()
