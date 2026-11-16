"""Separate label fitting, simulation outcomes, and real-terminal claim eligibility."""
from __future__ import annotations

from pathlib import Path
import statistics as st

from audit_day_shapes import read
from context_learning import OUT,ARMS,SEEDS
from report_synthetic_pipeline import audit,combined,FLOW_NAMES
from six_step_learning import sha
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
    for arm in ('initial',*ARMS):
        for seed in SEEDS:
            sel=[r for r in rows if r['case']['arm']==arm and r['case']['model_seed']==seed]
            initial=combined([r for r in rows if r['case']['arm']=='initial' and r['case']['model_seed']==seed])
            g=combined(sel)
            paired.append(dict(arm=arm,seed=seed,**g,
                change_vs_initial=100*(g['flows']['TRUCK']['mean_min']/initial['flows']['TRUCK']['mean_min']-1)))
    label_reports=[read(OUT/f'model-{s}'/'training.json') for s in SEEDS]
    corpus=read(OUT/'corpus-audit.json')
    for r in rows:
        c=r['case']
        if c['model_seed']:
            path=OUT/f'model-{c["model_seed"]}'/f'{c["arm"]}.pt'
            assert sha(path)==r['checkpoint_sha256'];artifacts[path.as_posix()]=sha(path)
    for seed,d in zip(SEEDS,label_reports):
        assert d['seed']==seed
        path=OUT/f'model-{seed}'/'training.json';artifacts[path.as_posix()]=sha(path)
        for i,h in d['corpus_sha256'].items():
            path=OUT/'corpora'/f'{int(i):02d}.json.gz'
            assert sha(path)==h;artifacts[path.as_posix()]=h
        for r in d['rows']:
            assert sha(OUT/f'model-{seed}'/f'{r["arm"]}.pt')==r['checkpoint_sha256']
    for name in ('corpus-audit.json','network-latency.json','authorization.json','training-audit.json'):
        path=OUT/name;artifacts[path.as_posix()]=sha(path)
    by_load=[dict(arm=arm,load=load,**combined([r for r in rows if r['case']['arm']==arm and r['case']['load']==load]))
             for arm in arms for load in (3500,7500,15000)]
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
                fitting=label_reports,corpus=corpus,by_load=by_load,latency=candidate_stats,artifact_hashes=artifacts,
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
    lines += ['', '### 물량별 트럭 평균(분)','','| 방법 | 3,500건 | 7,500건 | 15,000건 |','|---|---:|---:|---:|']
    for arm in arms:
        lines.append('| '+names[arm]+' | '+' | '.join(f'{next(r for r in by_load if r["arm"]==arm and r["load"]==load)["flows"]["TRUCK"]["mean_min"]:.3f}' for load in (3500,7500,15000))+' |')
    lines += ['', '전체 평균에서 물량별 오더 비중은13.46%/28.85%/57.69%다. 많은 물량의 결과가 더 크게 반영된다.']
    lines += ['', '### 한 요소씩 바꾼 결과','','| 대조 → 변경 | 트럭 | 반입 | 반출 | 양하 | 적하 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append('| '+names[c['baseline']]+' → '+names[c['candidate']]+' | '+
                     ' | '.join(f'{c["change_percent"][f]:+.2f}%' for f in FLOW_NAMES)+' |')
    lines += ['', '양하·적하는 스트림 계획 시작→개별 야드 작업 완료다. 선박 전체 체류시간과 다르다.',
        '', '### 초기값별 트럭 평균','','| 방법 | 초기값1 | 초기값2 | 초기값3 |','|---|---:|---:|---:|']
    for arm in ('initial',*ARMS):
        lines.append('| '+names[arm]+' | '+' | '.join(f'{next(r for r in paired if r["arm"]==arm and r["seed"]==s)["flows"]["TRUCK"]["mean_min"]:.3f}' for s in SEEDS)+' |')
    lines += ['', '### 비교 정답을 맞추는 정도','',
        f'- 새 학습 정답{corpus["by_split"]["train"]["pairs"]}쌍, 선택용 검사{corpus["by_split"]["selection"]["pairs"]}쌍이다. 후보 행이나 반복 추출을 새 정답으로 세지 않는다.',
        f'- 절반 자료 팔은{corpus["half_train_pairs"]}쌍이다. 모든 팔의 갱신당128쌍은 복원 추출이며 고유 자료 수와 다르다.',
        f'- 두 후보의10개 입력이 신경망 정밀도에서 완전히 같은데 비용 차이가 있는 비교는 전체{len(corpus["identical_float32_features_with_nonzero_gap"])}쌍이다.',
        '  교사가 동일 입력 후보를 먼저 제외하므로 이 수치는 입력 정보의 충분성을 입증하지 않는다.',
        '- 질문 한도432개 중 실제412개를 물었고307개에 정답을 만들었다.105개는 단일 공유 점수 변경으로 대안을 찾지 못했다.',
        '  이는 가능한 모든 공동 배정에서 대안이 없다는 증명이 아니다.',
        '- 아래 세 초기값의 평균. 순위는 정규화 비용 차이의 절댓값이1e-9를 넘는 비교만 세어 미세한 계산 오차를 제외한다.',
        '  오차는 동점을 포함한 전체 비교의 정규화 SmoothL1이다. 원화 차이 기준의 비동점 수와 분모가 다를 수 있다.',
        f'- 순위 판정에 쓰인 유효 비교는 학습{label_reports[0]["rows"][0]["train"]["nonzero"]}쌍·선택{label_reports[0]["rows"][0]["validation"]["nonzero"]}쌍이며 세 초기값이 같은 비교를 사용한다.',
        '', '| 방법 | 학습 순위 정답률 | 선택용 순위 정답률 | 선택용 오차 |','|---|---:|---:|---:|']
    for arm in ARMS:
        reports=[next(r for r in d['rows'] if r['arm']==arm) for d in label_reports]
        lines.append(f'| {names[arm]} | {st.mean(r["train"]["accuracy"] for r in reports)*100:.1f}% | {st.mean(r["validation"]["accuracy"] for r in reports)*100:.1f}% | {st.mean(r["validation"]["loss"] for r in reports):.4f} |')
    lines += ['', '### 26시간 뒤 미완료: 모델당 세 평가일 합산의 평균','','| 방법 | 반입 | 반출 | 양하 | 적하 |','|---|---:|---:|---:|---:|']
    for g in groups:
        divisor=1 if g['arm'] in ('SF_SPT','FIFO','LWKR') else 3
        lines.append('| '+names[g['arm']]+' | '+' | '.join(f'{g["flows"][f]["unfinished26"]/divisor:.1f}' for f in ('GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'))+' |')
    bench=read(OUT/'network-latency.json')
    generation_minutes=[r['seconds']/60 for r in corpus['rows']]
    training_seconds=[r['seconds'] for r in label_reports]
    lines += ['', '### 후보 보존과 계산 시간','',
        f'- 하루 정답 수집은{min(generation_minutes):.1f}~{max(generation_minutes):.1f}분, 한 초기값에서 여섯 방법의 망 학습은{min(training_seconds):.1f}~{max(training_seconds):.1f}초였다.',
        '- 각각 동시 실행 중인 개별 작업의 경과시간이다. 합을 전체 실험 경과시간으로 읽지 않는다.',
        '', '| 방법 | 매개변수 | 최대 후보 | 전체 결정의 95% 시간 범위(ms) | 1,200후보 신경망만(ms) |',
        '|---|---:|---:|---:|---:|']
    for arm in ('point_replay','mean_context','star_attention'):
        b=next(r for r in bench['rows'] if r['arm']==arm and r['candidates']==1200)
        times=[r['p95_ms'] for r in candidate_stats[arm]['decision']]
        lines.append(f'| {names[arm]} | {b["parameters"]} | {candidate_stats[arm]["max_candidates"]} | {min(times):.3f}~{max(times):.3f} | {b["p95_ms"]:.3f} |')
    lines += ['', '각 실행에서 결정100번 중95번이 끝나는 시간을 구해 그 범위를 표시했다. 전체 결정은 후보 생성과 배정을 포함한다.',
        '실행들은 병렬 진단 환경에서 측정했다. 신경망 단독 수치는 후보 생성 비용을 포함하지 않으며 현장 응답시간 보장이 아니다.',
        '정책별 작업 경로와 후보 수가 달라 전체 결정 시간의 차이를 신경망 계산량만의 효과로 해석하지 않는다.',
        '후보 개수 제한은 없고 제공된 실행 가능 후보를 모두 평가한다. 생성기 밖의 모든 물리적 대안에 대한 최적성 보장은 아니다.']
    lines += ['', '## ② 해석','',
        '- 후보 평균망과 Attention 망의 비교는 같은 10개 특징·후보 집합·정답·갱신 예산에서 집계 방식만 바꾼 것이다.',
        '- 후보별 MLP와 맥락망의 비교에는 다른 후보 정보를 읽는 효과도 포함된다.',
        '- 관계는 같은 크레인 자원 경쟁이다. 트럭 방문 선후관계 전체나 PPO를 구현했다고 주장하지 않는다.',
        '- 맥락망의 미학습 운영 대조는 없다. 최종 망끼리의 비교를 맥락망 자체의 학습 전후 이득으로 해석하지 않는다.',
        '- 세 초기값은 같은 정답을 사용했다. 세 번의 독립적인 환경·행동 정책 자료 수집으로 세지 않는다.',
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
