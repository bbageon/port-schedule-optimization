"""Separate label fitting, simulation outcomes, and real-terminal claim eligibility."""
from __future__ import annotations

from pathlib import Path
import statistics as st

from audit_day_shapes import read
from context_learning import OUT,ARMS,SEEDS
from report_synthetic_pipeline import audit,combined,FLOW_NAMES
from yard_rl.integrated.repro import write_result


def main():
    rows,artifacts=audit(OUT);assert len(rows)==72
    arms=('initial',*ARMS,'SF_SPT','FIFO','LWKR')
    grouped={arm:combined([r for r in rows if r['case']['arm']==arm]) for arm in arms}
    reference=grouped['SF_SPT']['flows']['TRUCK']['mean_min']
    groups=[]
    for arm in arms:
        group=grouped[arm]
        groups.append(dict(arm=arm,**group,vs_rule_percent=100*(group['flows']['TRUCK']['mean_min']/reference-1)))
    paired=[]
    for arm in ARMS:
        for seed in SEEDS:
            sel=[r for r in rows if r['case']['arm']==arm and r['case']['model_seed']==seed]
            initial=combined([r for r in rows if r['case']['arm']=='initial' and r['case']['model_seed']==seed])
            g=combined(sel)
            paired.append(dict(arm=arm,seed=seed,**g,
                change_vs_initial=100*(g['flows']['TRUCK']['mean_min']/initial['flows']['TRUCK']['mean_min']-1)))
    label_reports=[read(OUT/f'model-{s}'/'training.json') for s in SEEDS]
    candidate_stats={arm:dict(max_candidates=max(r['probe']['max_candidates_per_crane'] for r in rows if r['case']['arm']==arm),
                             decision=[r['probe']['decision'] for r in rows if r['case']['arm']==arm],
                             total_seconds=sum(r['seconds'] for r in rows if r['case']['arm']==arm)) for arm in arms}
    comparisons=[]
    for baseline,candidate in [('point_current','point_replay'),('point_half','point_replay'),('point_replay','point_steps'),
                               ('point_replay','mean_context'),('mean_context','star_attention')]:
        a,b=grouped[baseline],grouped[candidate]
        comparisons.append(dict(baseline=baseline,candidate=candidate,
            change_percent={f:100*(b['flows'][f]['mean_min']/a['flows'][f]['mean_min']-1) for f in FLOW_NAMES},
            cost_change=100*(b['cost']/a['cost']-1)))
    result=dict(cases=len(rows),groups=groups,by_seed=paired,comparisons=comparisons,
                fitting=label_reports,latency=candidate_stats,artifact_hashes=artifacts,
                scope='3 synthetic days, one per load, 3 model initializations; no final real-world claim')
    write_result(OUT/'summary.json',result)
    names=dict(initial='학습 전 MLP',point_current='현재 자료 50회',point_replay='누적 자료 50회',point_half='절반 자료 50회',
        point_steps='누적 자료 200회',mean_context='후보 평균 맥락망',star_attention='자원 중심 Attention',
        SF_SPT='실작업·짧은 작업 우선',FIFO='작업 대기 진입 순서',LWKR='남은 작업량 우선')
    lines=['# 합성 하루의 학습 방식·후보 맥락망 비교','','## ① 진행사항','',
        '- 학습6일·선택 검사3일에서 새 비교 정답을 수집했다. 행동 정책·미래비용 창은 모든 망이 같다.',
        '- 초기값3개×학습 팔6개를 학습하고 미사용 합성 하루3개에서 규칙 포함72회 평가했다.',
        '- 물량마다 평가일이 하나다. 모델 초기값3개를 독립적인 실제 날짜3개로 세지 않는다.',
        '- 평균은 오더 수로 가중한 게이트 진입→작업 완료이며 24시간 미완료 경과시간을 포함한다. +는 더 오래 걸림이다.','',
        '| 방법 | 트럭 평균(분) | 짧은 실작업 우선 대비 |', '|---|---:|---:|']
    for g in groups:lines.append(f'| {names[g["arm"]]} | {g["flows"]["TRUCK"]["mean_min"]:.3f} | {g["vs_rule_percent"]:+.2f}% |')
    lines += ['', '### 한 요소씩 바꾼 결과','','| 대조 → 변경 | 트럭 | 반입 | 반출 | 양하 | 적하 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append('| '+names[c['baseline']]+' → '+names[c['candidate']]+' | '+
                     ' | '.join(f'{c["change_percent"][f]:+.2f}%' for f in FLOW_NAMES)+' |')
    lines += ['', '양하·적하는 스트림 계획 시작→개별 야드 작업 완료다. 선박 전체 체류시간과 다르다.',
        '', '### 초기값별 트럭 평균','','| 방법 | 초기값1 | 초기값2 | 초기값3 |','|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append('| '+names[arm]+' | '+' | '.join(f'{next(r for r in paired if r["arm"]==arm and r["seed"]==s)["flows"]["TRUCK"]["mean_min"]:.3f}' for s in SEEDS)+' |')
    lines += ['', '### 26시간 뒤 미완료: 모델당 세 평가일 합산의 평균','','| 방법 | 반입 | 반출 | 양하 | 적하 |','|---|---:|---:|---:|---:|']
    for g in groups:
        divisor=1 if g['arm'] in ('SF_SPT','FIFO','LWKR') else 3
        lines.append('| '+names[g['arm']]+' | '+' | '.join(f'{g["flows"][f]["unfinished26"]/divisor:.1f}' for f in ('GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'))+' |')
    lines += ['', '## ② 해석','',
        '- 후보 평균망과 Attention 망의 비교는 같은 10개 특징·후보 집합·정답·갱신 예산에서 집계 방식만 바꾼 것이다.',
        '- 후보별 MLP와 맥락망의 비교에는 다른 후보 정보를 읽는 효과도 포함된다.',
        '- 관계는 같은 크레인 자원 경쟁이다. 트럭 방문 선후관계 전체나 PPO를 구현했다고 주장하지 않는다.',
        '- 학습 손실·순위와 운영 시간이 같은 방향으로 움직인다고 가정하지 않는다. 모든 초기값·미완료를 함께 본다.',
        '- 시간대 곡선·작업 비율·분별 모형은 개발 자료 보정 합성이다. 새로운 실제 항만 기간 검증이 아니다.',
        '- 운반 장비·대기열·지속운영 현실성은 미확인이다. 현장 채택이나 전역 최적성의 증거로 삼지 않는다.',
        '', '## ③ 예정사항','',
        '- 323-d/as는 이 진단 결과를 기준으로 후속 채택 여부를 판정한다. 실패한 팔은 기본값으로 바꾸지 않는다.',
        '- i/f의 외부 운영 자료 확인과 323-e의 미사용 기간 평가는 현황판의 선행 조건을 유지한다.',
        '', '재현: [전체 수치](summary.json), [설정](campaign.json), [사전등록](../../../docs/research/v7-discharge/context-learning-prereg.md).']
    assert len(lines)<=200
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print([(r['arm'],r['flows']['TRUCK']['mean_min'],r['vs_rule_percent']) for r in groups])


if __name__=='__main__':main()
