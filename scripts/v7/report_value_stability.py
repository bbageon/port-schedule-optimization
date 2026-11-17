"""Report the fixed comparison, including negative or inconclusive outcomes."""
import math
import statistics as st

from value_stability_inputs import OUT, ARMS, SEEDS, LOADS, read
from report_synthetic_pipeline import FLOW_NAMES


def main():
    s=read(OUT/'summary.json');g=s['groups'];d=s['paired_change'];rule=s['vs_rule']
    grad=read(OUT/'gradient-diagnostic.json');choice=read(OUT/'choice-diagnostic.json')
    labels=dict(control='기존 가치 학습',popart='가치 목표 정규화',SF_SPT='짧은 실작업 우선 규칙')
    verdict=('세 초기값에 걸쳐 개선 후보 조건을 만족했다.' if s['consistent_improvement_candidate'] else
             '가치 목표 정규화의 일관된 정책 개선을 확인하지 못했다.')
    lines=['# 남은 비용 예측의 학습 안정화 비교', '', f'**{verdict}**',
        f'기존 학습 대비 트럭 시간 {d["flow_pct"]["TRUCK"]:+.2f}%, 총비용 {d["cost_pct"]:+.2f}%다. '
        f'짧은 실작업 우선 규칙 대비 트럭 시간은 {rule["flow_pct"]["TRUCK"]:+.2f}%다.',
        '음수 학습 신호는 유지했다. 새로 학습한 여섯 모델을 같은 입력에서 비교한 결과다.', '',
        '## ① 진행사항', '',
        '- 같은 초기값 3쌍, 모델당 새 합성 12일·6갱신·12,288개 선택 경험·768최적화 단계.',
        '- 기존 방식과 PopArt(비용 예측의 학습 눈금을 바꾸되 원래 단위 예측을 보존)를 비교했다.',
        '- 새로 생성한 19곡선을 학습12·동작시험1·검증3·최종시험3으로 분리했다.',
        '- 최종시험은 3곡선×3물량의 9입력이다. 검증·시험 본 평가126회와 결정적 재실행2회를 했다.',
        '- 총비용·24시간 보상·26시간 실행·정책 입력·학습량·행동 학습 식을 그대로 유지했다.', '',
        '| 최종 시험 | 평균 트럭 작업시간(분) | 총비용 변화: 기존 대비 | 26시간 후 미완료 오더: 회당 |',
        '|---|---:|---:|---:|']
    for a in (*ARMS,'SF_SPT'):
        x=g[a];lines.append(f'| {labels[a]} | {x["flows"]["TRUCK"]["mean_min"]:.3f} | '
            f'{100*(x["cost"]/g["control"]["cost"]-1):+.2f}% | {x["unfinished26_mean"]:.2f} |')
    lines += ['', '변화의 **−는 감소, +는 증가**다. 트럭 시간은 오더당 게이트 진입→작업완료이며 출차까지가 아니다.',
        '미완료 건도 24시간 종료까지 관측한 경과시간을 평균에 포함한다. 최종 완료시간을 모두 관측한 평균은 아니다.',
        '사전등록대로 입력 날짜와 모델에 같은 무게를 줬다. 과거 물량 가중 평균과 직접 비교하지 않는다.',
        f'오더 수로 가중한 보조 평균은 기존 {g["control"]["flows"]["TRUCK"]["order_weighted_min"]:.3f}분, '
        f'정규화 {g["popart"]["flows"]["TRUCK"]["order_weighted_min"]:.3f}분, '
        f'규칙 {g["SF_SPT"]["flows"]["TRUCK"]["order_weighted_min"]:.3f}분이다.', '',
        '### 초기값·물량에 따라 같은 결론인가', '',
        '| 모델 초기값 | 기존(분) | 정규화(분) | 트럭 시간 변화 | 총비용 변화 |', '|---:|---:|---:|---:|---:|']
    for seed,row in s['by_seed'].items():
        a,b=[row['groups'][x]['flows']['TRUCK']['mean_min'] for x in ARMS]
        lines.append(f'| {seed} | {a:.3f} | {b:.3f} | {row["change"]["flow_pct"]["TRUCK"]:+.2f}% | {row["change"]["cost_pct"]:+.2f}% |')
    lines += ['', '| 하루 트럭 오더 | 기존(분) | 정규화(분) | 규칙(분) | 정규화의 시간 변화 |', '|---:|---:|---:|---:|---:|']
    for load,row in s['by_load'].items():
        a,b,c=[row['groups'][x]['flows']['TRUCK']['mean_min'] for x in (*ARMS,'SF_SPT')]
        lines.append(f'| {int(load):,} | {a:.3f} | {b:.3f} | {c:.3f} | {row["change"]["flow_pct"]["TRUCK"]:+.2f}% |')
    lines += ['', '초기값 세 개만으로 일반적인 개선을 확증하지 않는다. 7,500·15,000건은 실측 고물량 검증이 남은 부담 시험이다.', '',
        '### 작업 종류별 결과', '', '| 종류 | 기존(분) | 정규화(분) | 변화 | 26시간 미완료: 기존/정규화 회당 |', '|---|---:|---:|---:|---:|']
    for f in ('GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'):
        a,b=[g[x]['flows'][f] for x in ARMS]
        lines.append(f'| {FLOW_NAMES[f]} | {a["mean_min"]:.3f} | {b["mean_min"]:.3f} | {d["flow_pct"][f]:+.2f}% | {a["unfinished26_mean"]:.2f}/{b["unfinished26_mean"]:.2f} |')
    lines += ['', '본선 작업은 계획된 작업 묶음 시작→개별 야드 작업완료다. 선박 전체의 항만 체류시간과는 다르다.',
        f'본선 지연 비용 변화는 {d["vessel_cost_pct"]:+.2f}%, 전체 미완료는 회당 {d["unfinished26_delta"]:+.2f}건이다.', '',
        '### 비용 예측 자체는 나아졌나', '', '| 하루 오더 | 기존 예측 오차 | 정규화 예측 오차 |', '|---:|---:|---:|']
    for load in LOADS:
        a,b=[next(v['rmse'] for v in s['value_validation'] if v['arm']==arm and v['load']==load) for arm in ARMS]
        lines.append(f'| {load:,} | {a:.4f} | {b:.4f} |')
    lines += ['', '오차는 새 검증 경험에서 실제 남은 비용과 예측의 차이를 제곱평균한 뒤 제곱근을 취한 값이다.',
        '양쪽 모두 원래 고정 보상 눈금으로 복원해서 잰다. 각 정책이 확률 추첨으로 만든 경험이므로 방문 상태는 다를 수 있다.',
        '학습에 사용한 경험의 오차와 새 검증 오차는 분리했다. 정규화된 학습 손실 숫자를 서로 직접 비교하지 않는다.']
    for arm in ARMS:
        ts=[r for r in s['training']['rows'] if r['arm']==arm]
        before=math.sqrt(st.mean(r['before_rmse']**2 for r in ts));after=math.sqrt(st.mean(r['after_rmse']**2 for r in ts))
        gr=[r['critic_actor_ratio'] for r in grad['rows'] if r['arm']==arm]
        cs=[r for r in s['training']['signal_cells'] if r['arm']==arm and r['load']==15000 and r['hour_bin']<2]
        lines.append(f'- {labels[arm]}: 학습자료 예측 오차 {before:.4f}→{after:.4f}; '
            f'고물량 오전 양수 행동 신호 {sum(r["positive"] for r in cs):,}/{sum(r["n"] for r in cs):,}; '
            f'첫 묶음의 가치/행동 기울기 크기 비율 중앙값 {st.median(gr):.2f}배.')
    lines += ['기울기 비율은 갱신 첫 묶음만 측정했다. Adam을 거친 실제 파라미터 이동량이나 전체 학습의 원인 비율이 아니다.', '',
        '### 같은 상태에서의 선택 차이', '', '| 초기값 | 크레인 하나의 선택 상황 수 | 선택 확률 분포의 평균 차이 | 최고점 후보 불일치 |', '|---:|---:|---:|---:|']
    for r in choice['rows']:
        if r['load']==0:
            lines.append(f'| {r["seed"]} | {r["single_crane_n"]:,} | {100*r["single_crane_mean_total_variation"]:.3f}% | {100*r["single_crane_argmax_disagreement"]:.2f}% |')
    lines += ['', '양쪽 정책에서 저장한 검증 상태를 합쳐 같은 합법 후보에 두 모델을 적용했다.',
        '표는 크레인 하나가 고르는 상황만 따로 센다. 여러 크레인에 대한 선택 순서 변화와 실작업 우선순위 변화를 섞지 않기 위해서다.',
        '확률 분포 차이는 바뀐 확률 질량이다. 작은 확률 차이라도 최고점 후보가 바뀔 수 있다. 이 표는 선택 결과의 인과 효과를 재현한 실험이 아니다.', '',
        '## ② 해석', '', f'- {verdict}',
        '- 음수 비용을 보상으로 쓰는 것은 비용 최소화를 보상 최대화로 표현하는 방법이다.',
        '- 행동 신호의 음수는 “예상보다 나빴으니 선택 확률을 낮추라”는 뜻이다. 예를 들어 예측 비용10·실제8이면 상대 신호+2, 실제12이면−2다.',
        '- 현재 구현은 이 상대 신호를 두 날짜 전체에서 한 번 더 표준화한다. 고물량 날짜의 음수 편중만으로 잘못된 행동이라고 단정할 수 없다.',
        '- 이번 비교에서는 음수 신호를 삭제하지 않았다. 삭제 실험을 하지 않았으므로 음수 제거의 효과를 입증한 것도 아니다.',
        '- 학습은 확률 추첨, 주 평가는 최고점 선택이다. 그 차이를 유지한 비교이며 두 실행 방식이 같다고 주장하지 않는다.',
        '- 모든 학습 쌍은 새로 시작했다. 기존 체크포인트에서 완전하게 이어 학습한 결과가 아니다.',
        '- 정규화의 Adam 누적값은 보존했다. 출력 보존은 검사했지만 최적화 궤적의 좌표 불변성은 주장하지 않는다.',
        f'- 채택 성능 게이트: {"통과" if s["adoption_performance_pass"] else "미통과"}. 기록·코드·검산은 별도 확인하며 현실성은 미통과를 유지한다.',
        '- 동일한 합성 생성기의 새 추첨이다. 독립 실측 기간·운반 흐름·현장 대기열을 검증한 결과가 아니다. 기본 정책은 바꾸지 않았다.', '',
        '## ③ 예정사항', '',
        '- 운반 흐름 확인(YR-332-i): 차량 투입·주행·인계 기록으로 양하 연결을 검증한다. 실제 환경의 성능을 말하기 전에 필요한 외부 자료가 남아 있다.',
        '- 대기열 정의 확인(YR-332-f): 현장의 줄과 모델의 대기 상태가 같은 뜻인지 확인한다. 자료가 도착하면 현실성 게이트를 다시 판정한다.', '',
        '증거: [사전등록](../../../docs/research/v7-discharge/value-stability-prereg.md), [실행 설정](campaign.json), [전체 수치](summary.json),',
        '[같은 상태 선택](choice-diagnostic.json), [기울기 진단](gradient-diagnostic.json), [검산](verification.json), [발행](publication.json).',
        '그림: [학습·평가 비교](comparison.png).',
        '검산 명령과 재현 범위: [재현 안내](REPRODUCE.md). 모델 여섯 개가 같은 학습 날짜12개를 공유했다.',
        '재현 주의: 구형 곡선 보정 JSON 1개는 Git과 로컬의 줄바꿈만 달랐다. 수치는 같으며 [원래 바이트 사본](frozen-sources/calibration-periodic_linear_24.json)과 [대조 기록](source-publication.json)을 보존했다.',
        '방법 근거: [PopArt 원 논문](https://arxiv.org/abs/1602.07714), [PPO 원 논문](https://arxiv.org/abs/1707.06347).']
    assert len(lines)<=200,len(lines)
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
