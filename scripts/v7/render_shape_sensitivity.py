"""Render the already-verified fixed-rule shape results without cherry-picking."""
from check_initial_stock import OUT,read


def main():
    s=read(OUT/'shape-summary.json')
    lines=['# 하루 평균 곡선과 변동 곡선: 고정 규칙 민감도',
        '', '2026-11-16. YR-332-ao의 하루 모양 축. 재고 보정(at) 후 동일 초기 배치에서 비교했다.',
        '', '## ① 진행사항', '',
        '- 원래 학습 곡선 첫 3개 × 오더 3,500/7,500/15,000건 × 평균/변동 × 규칙 3개 = **54회**.',
        '- 두 팔의 오더 수·작업 종류·초기 재고·본선·물리는 같다. 평균은 학습 곡선 24개의 점별 평균이다.',
        '- 정책과 비용 정의를 고정했다. 이 결과는 새 신경망의 학습 성능이 아니다.',
        '- 모든 실행은 접수 누락·중복·정책 예외·방문 연결 지연 없이 26시간까지 계산됐다.',
        '  “실행 종료”는 “모든 오더 완료”와 다르다. 미완료도 장부와 평균에 남겼다.',
        '- 54개 결과를 전체 오더 시각 원자료에서 재계산했다. 앞 실행과 겹치는 3조건은',
        '  원자료와 후보/선택/사건/작업/기록/결과 해시 6종이 정확히 같다.',
        '', '평균은 **오더당 게이트 진입→작업 완료**이며 24시간 경계에서 관측을 자른 값이다.',
        '완료된 건만 골라 계산한 평균이 아니다. 한 방문이 여러 오더를 처리할 수 있으므로 트럭 한 대의 최종 작업 완료 평균과도 구분한다.',
        '아래는 각 물량·규칙에서 3개 배경을 오더 수로 가중한 평균이다. 변화율의 +는 더 오래 걸림이다.',
        '', '| 하루 오더 | 규칙 | 평균 곡선(분) | 변동 곡선(분) | 변화 | 날짜별 변화 범위 | 비용 변화 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    names={'SF_SPT':'실작업·짧은 작업 우선','FIFO':'작업 대기 진입 순서','LWKR':'남은 작업량이 적은 순서'}
    for g in s['groups']:
        f=g['by_flow']['TRUCK']
        lines.append(f"| {g['load']:,} | {names[g['rule']]} | {f['fixed']['mean_min']:.3f} | {f['variable']['mean_min']:.3f} | {f['change_percent']:+.2f}% | {g['pair_min_percent']:+.2f}~{g['pair_max_percent']:+.2f}% | {g['cost_change_percent']:+.2f}% |")
    lines += ['', '비용은 기존 터미널 비용 식을 그대로 적용한 내부 지표다. 실제 발생 비용으로 검증했다는 뜻이 아니다.',
        '', '## 유형별 변화와 미완료', '',
        '실작업·짧은 작업 우선 규칙의 세 배경 합산 결과다. 모든 규칙·개별 배경의 값은 JSON에 보존했다.',
        '본선 양하/적하는 **스트림 계획 시작→각 야드 작업 완료**다. 트럭 시간이나 선박의 전체 접안 시간과 동일한 지표가 아니다.',
        '', '| 하루 오더 | 작업 | 평균 곡선(분) | 변동 곡선(분) | 변화 | 24h 미완료 평균/변동 | 26h 미완료 평균/변동 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    fnames={'GATE_IN':'반입','GATE_OUT':'반출','VESSEL_DISCHARGE':'양하','VESSEL_LOAD':'적하'}
    for g in s['groups']:
        if g['rule']!='SF_SPT':continue
        for flow,name in fnames.items():
            f=g['by_flow'][flow];a,b=f['fixed'],f['variable']
            lines.append(f"| {g['load']:,} | {name} | {a['mean_min']:.3f} | {b['mean_min']:.3f} | {f['change_percent']:+.2f}% | {a['unfinished']:,}/{b['unfinished']:,} | {a['unfinished26']:,}/{b['unfinished26']:,} |")
    lines += ['', '## ② 해석과 범위', '',
        f"- 개별 27쌍의 오더당 트럭 시간 변화는 **{s['min_truck_change']:+.2f}~{s['max_truck_change']:+.2f}%**다.",
        f"  변화 크기의 중앙값은 **{s['median_absolute_truck_change']:.2f}%**다. 세 날짜 평균만으로 개별 날짜 차이를 지우지 않는다.",
        '- 3개 개발 곡선의 민감도다. 새 학습·독립 실측 기간 평가·실항만 성능·통계적 확증이 아니다.',
        '- 시간대별 묶음이 달라 방문 구성도 바뀐다. 동일 연결 알고리즘을 유지했지만 동일한 방문 명단은 아니다.',
        f"  방문 수 변화의 최대 절댓값은 **{s['max_absolute_visit_change']:.3f}%**다. 순수한 시각 이동만의 효과로 좁혀 해석하지 않는다.",
        '- 큰 물량에서는 배수 2시간 뒤에도 미완료가 남는다. 관측 평균의 차이를 최종 완주 시간의 차이라고 할 수 없다.',
        '- 7,500·15,000건은 부담 시험이다. 13일 요약 통계에 보정한 합성 구조이며 실제 터미널 총물량 검증은 아니다.',
        '- 현재 시간대 안 도착은 여전히 관측 분별 형판을 사용한다. 하루 곡선만 수식으로 바꾼 단계다.',
        '- 입력·기록 정합과 공급 보존은 확인했다. 운반·대기열·초기 배치의 외부 근거와 지속운영 현실성은 남는다.',
        '', '## ③ 예정사항', '',
        '- ao의 반출 비율 변동은 하루 곡선과 다른 입력 축이므로 별도 비교로 남긴다.',
        '- ar: 관측 분별 배열 없이 수식과 매개변수로 방문 도착을 만든다. 완전한 수식 기반 합성 경로를 위해 필요하다.',
        '- 323-d: 같은 자료 맞추기·이전 자료 유지·새 상태에서의 학습을 각각 나눈다.',
        '  저장 모델에서 앞 회차 정답 손실을 확인했으므로, 큰 신경망 교체 전에 학습 방식의 영향을 확인한다.',
        '', '## 재현', '',
        '- [전체 집계](shape-summary.json), [실행 목록](execution-shapes-result.json), [설정·소스 고정](campaign.json).',
        '- [사전등록](../../../docs/research/v7-discharge/initial-stock-prereg.md), [짝 입력](shape-input-audit.json).',
        '- 실행 핵심 코드 고정점 305d0eca, 캠페인 시작 체크아웃 df4e7a14. 프로세스 12개, 각 연산 스레드 1개.',
        '  실행 중 핵심 코드·입력 해시는 매 실행 전에 검사했다. 이후 문서·읽기 전용 분석 추가는 실행 소스를 바꾸지 않았다.',
        '- `scripts/v7/report_initial_stock.py shapes`로 원자료를 검산하고 이 문서는 `render_shape_sensitivity.py`로 생성한다.',
        '- [재고 보정](README.md), [저장 모델 진단](../yr332_network_review/saved-pair-retention.md).']
    assert len(lines)<=200
    (OUT/'shape-sensitivity.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('rendered',len(lines),'lines')


if __name__=='__main__':main()
