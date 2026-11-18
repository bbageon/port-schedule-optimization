"""Render both the prediction gate and, when run, truck AND vessel outcomes."""
from pathlib import Path

from global_value_inputs import OUT, SEEDS, ARMS, read


def write_report():
    s=read(OUT/'summary.json');p=s['prediction'];v=read(OUT/'verification.json')
    decision='통과' if p['passed'] else '미통과'
    lines=['# 전역 대기 정보의 비용 예측·조건부 정책 비교','',
        f'**새 날짜의 예측 오차 {p["rmse_change_pct"]:+.2f}%, 사전 예측 기준 {decision}.**',
        '전역 대기량은 비용 예측망에만 넣었다. Transformer의 작업 선택 입력·총비용·보상창은 유지했다.','',
        '## ① 진행사항','',
        f'- 고정 행동 정책으로 학습12일·검증9입력의 공통 경험 {s["selected_common_samples"]:,}개를 모았다.',
        '- 이전 정규화 모델3개의 공유 표현·선택망은 고정하고, 비용 예측 부분만 같은 예산으로200회 반복했다.',
        '- 두 군의 차이는21블록 대기 외부 트럭 오더 합계 한 칸이다. 대조군은 같은 칸에0을 넣었다.',
        '- 3,500건 동작 시험에서 기록만 추가한 실행의 후보·행동·사건·비용·난수 흐름이 정확히 같았다.',
        f'- 관련 검사{v["unit_tests"]}건, 예측 학습6개 재실행, 공통 경험1회 재실행과 원장 재계산을 확인했다.','',
        '| 검증 물량 | 전역 정보 없음 | 전역 정보 있음 | 예측 오차 변화 |','|---|---:|---:|---:|',
        f'| 전체 | {p["groups"]["zero"]["rmse"]:.6f} | {p["groups"]["global"]["rmse"]:.6f} | {p["rmse_change_pct"]:+.2f}% |']
    for load in ('3500','7500','15000'):
        lines.append(f'| {int(load):,}건 | {p["groups"]["zero"]["by_load"][load]:.6f} | {p["groups"]["global"]["by_load"][load]:.6f} | {p["by_load_change_pct"][load]:+.2f}% |')
    lines+=['','오차는 RMSE(큰 오차를 더 크게 반영한 평균 예측 오차)다. 낮을수록 좋다.',
        '단위는 고정 보상 눈금이며 분·원화 청구액이 아니다. 날짜와 모델에 같은 무게를 줬다.','',
        '| 시간대 | 예측 오차 변화 |','|---|---:|']
    for hour,change in p['by_time_change_pct'].items():
        start=int(hour)*6;lines.append(f'| {start:02d}–{start+6:02d}시 | {change:+.2f}% |')
    lines+=['','| 시작 모델 | 예측 오차 변화 |','|---|---:|']
    for seed,change in p['by_seed_change_pct'].items():lines.append(f'| {seed} | {change:+.2f}% |')
    lines+=['','### 조건을 바꾸지 않고 판정했는가','',
        '- 전체 오차5% 이상 감소·모델3쌍 모두 감소·물량별 악화2% 이하·시간대별 악화5% 이하를 미리 정했다.',
        '- 0/20/50/100/200회 곡선을 보존했고 마지막200회만 판정했다. 유리한 회차를 골라 쓰지 않았다.']
    names=dict(overall_reduction='전체 오차5% 감소',all_seed_pairs_improve='세 모델 모두 감소',
               load_guard='물량별 악화 제한',time_guard='시간대별 악화 제한',actor_frozen='선택망 고정',
               observation_contract='관측 시각·행동 불변')
    lines+=['- '+', '.join(f'{names[k]}: {"통과" if ok else "미통과"}' for k,ok in p['conditions'].items())+'.','',
        '## ② 해석','']
    if s['policy'] is None:
        lines+=['- 사전 예측 기준을 모두 만족하지 못해 PPO 재학습과 새 정책 성능 비교로 확대하지 않았다.',
            '- 이번에는 작업 선택을 고정했다. 따라서 예측 오차 변화는 트럭·본선 작업시간의 개선율이 아니다.',
            '- 이 한 대기 요약·고정 행동 정책·학습 예산에서의 결과다. 모든 전역 정보가 쓸모없다는 결론은 아니다.']
    else:
        q=s['policy'];g=q['groups'];d=q['paired_change']
        lines+=['- 예측 기준을 통과해 PPO3쌍을 같은 시작 모델·같은12일·같은6갱신으로 추가 비교했다.',
            '- 예측 진단의 학습 가중치는 정책 학습에 사용하지 않았다. 두 군 모두 ay 시작 모델에서 새 Adam으로 출발했다.',
            f'- 정책 평가{q["evaluations"]}회와 재실행{q["repeats"]}회를 했다. 마지막 모델만 최종 판정했다.','',
            '| 지표 | 전역 정보 없음 | 전역 정보 있음 | 짧은 실작업 우선 규칙 | 변경군 변화 |',
            '|---|---:|---:|---:|---:|']
        names={'TRUCK':'평균 트럭','VESSEL':'평균 본선 작업','GATE_IN':'반입','GATE_OUT':'반출',
               'VESSEL_DISCHARGE':'양하','VESSEL_LOAD':'적하'}
        for flow,name in names.items():
            a,b,r=(g[k]['flows'][flow]['mean_min'] for k in (*ARMS,'SF_SPT'))
            lines.append(f'| {name}(분) | {a:.3f} | {b:.3f} | {r:.3f} | {d["flow_pct"][flow]:+.2f}% |')
        lines+=['',f'- 총비용 변화{d["cost_pct"]:+.2f}%, 본선 지연비용{d["vessel_cost_pct"]:+.2f}%.',
            f'- 26시간 뒤 전체 미완료는 실행당 평균{g["zero"]["unfinished26_mean"]:.2f}→{g["global"]["unfinished26_mean"]:.2f}건이다.',
            '- 트럭은 오더당 게이트 진입→작업완료다. 본선은 계획된 묶음 시작→개별 야드 작업완료이며 선박 입출항 시간이 아니다.',
            '- 미완료도24시간 관측 종료까지 경과시간을 평균에 포함한다. 전건의 최종 완료시간을 관측한 평균이 아니다.',
            '- 입력·모델 동일 가중이 주 평균이며 오더 가중 보조 평균과24/26시간 미완료는 summary.json에 보존했다.',
            f'- 작업별 보호 조건을 포함한 개선 후보 판정: {"통과" if q["consistent_candidate"] else "미통과"}.',
            '- 규칙 대비와 실제 항만 검증까지 통과한 것으로 해석하지 않으며 기본 정책은 유지한다.']
    obs=s['observation'];age=obs['before24_age_s'];after=obs['after24_age_s']
    lines+=['',f'- 학습 관측 지연: 평균{age["mean"]:.2f}초, 95%가{age["p95"]:.2f}초 이내, 최대{age["max"]:.2f}초.',
        f'- 관측 대기 오더 범위{obs["waiting_orders_range"]}, 입력 범위{obs["feature_range"]}. 입력 상한은 자르지 않았다.']
    if after['n']:lines.append(f'- 24시간 뒤 배수 구간은 마지막 관측을 유지하여 최대{after["max"]:.2f}초 오래된 값이다. 이 구간은 학습 표본에서 제외했다.')
    lines+=['- 같은 시각의 review·도착·작업시작은 기존 사건 순서를 따른다. 뒤에 처리되는 동시각 사건을 사후 입력하지 않았다.',
        '- 새로운 합성 날짜이지만 보정 가족은 같다. 독립 실제 운영 기간을 검증한 것은 아니다.',
        '- 운반 흐름·대기열 정의의 실측 확인이 남아 현실성은 미통과다. 현장 성능을 주장하지 않는다.','',
        '## ③ 예정사항','',
        '- YR-332-i: 운반 장비의 투입·주행·인계 자료를 확인한다. 본선 작업의 시간·용량을 실제 흐름에 맞춰 판단하기 위해 필요하다.',
        '- YR-332-f: 블록 대기열의 관측 단위를 확인한다. 대기 오더 수와 실제 차량 대수를 혼동하지 않기 위해 필요하다.',
        '- 두 작업 모두 외부 자료 대기 상태다. 이번 진단으로 그 선결을 해소했다고 하지 않는다.','',
        '[실행 사전등록](../../../docs/research/v7-discharge/global-value-context-prereg.md) · [수치](summary.json) · [검산](verification.json)',
        '', '문헌 근거: [MAPPO](https://arxiv.org/abs/2103.01955)의 전역 학습 정보 활용과',
        '[중앙 평가자 분석](https://arxiv.org/abs/2408.14597)의 효과·편향·분산 한계를 함께 고려했다.','']
    assert len(lines)<=200
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':write_report()
