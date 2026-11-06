"""Independent arithmetic audit and readable report for the frozen 71-run study."""
import json
import math
import copy
from functools import lru_cache
from pathlib import Path
import statistics

import evaluate_classical_rules as e

LABELS = dict(fcfs_arrival='도착순',spt='최단 작업',nearest='최근접',
    vessel_fcfs='본선 우선·도착순',vessel_nearest='본선 우선·최근접',
    unidirectional='단방향',distance_wait='거리·대기 복합',rule_all='기존 최단 작업(SF_SPT)',
    initial='학습 전',pointwise='기존 학습',pairwise='비용 차이 학습',
    coverage='비교 표본 확대',flow='작업 방향 추가(최신)')


@lru_cache(maxsize=None)
def read(arm,load,seed=0):
    return e.gz_read(e.path_for((arm,seed,load,0)))


def outcome(d):
    r=d['result']; o=d['probe']['operational']; c=d['probe']['flow_status_counts']
    return dict(cost=r['phi_krw'],turn=r['mean_turn_time_s'],admitted=r['admitted'],
        censored=r['n_censored'],vessel_done=sum(v for k,v in c.items()
        if k.split('/')[0] in ('VESSEL_LOAD','VESSEL_DISCHARGE') and k.split('/')[1]=='DONE'),
        discharge=c.get('VESSEL_DISCHARGE/DONE',0),loading=c.get('VESSEL_LOAD/DONE',0),
        delay=o['vessel_delay_s'],vessel_unfinished=o['vessel_unfinished'],idle=o['vessel_idle_s'],
        truck_yard_unfinished=sum(v for k,v in c.items()
            if k.split('/')[0] in ('GATE_IN','GATE_OUT') and k.split('/')[1]!='DONE'),
        p90=r['p90_turn_time_s'],p99=r['p99_turn_time_s'],
        vessel_cost=r['c_vessel'])


def validated_summary(raw):
    """Keep raw evidence, replace the mislabeled 26h ledger residual in analysis.

    run_episode synchronizes ExecutionRecords only through 24h. Engine job
    statuses are the actual 26h snapshot, and describe yard-service completion,
    not gate exits. No simulation, selection or official 24h metric is changed.
    """
    summary=copy.deepcopy(raw)
    for rows_key,averages_key in [('comparisons','averages'),('learning_comparisons','learning_averages')]:
        for row in summary[rows_key]:
            n,s=row['load'],row['model_seed'] or 0
            a=read('initial',n,s) if row['baseline']=='own_initial' else read(row['baseline'],n)
            b=read(row['arm'],n,s)
            row.pop('truck_unfinished_26h_change')
            row['truck_yard_unfinished_26h_change']=(outcome(b)['truck_yard_unfinished']
                                                    -outcome(a)['truck_yard_unfinished'])
        summary[averages_key]=e.aggregate(summary[rows_key])
    summary['measurement_correction']=dict(
        excluded_raw_fields=['probe.operational.trucks_unfinished','truck_unfinished_26h_change'],
        reason='ExecutionRecords are synchronized through 24h, not the final 26h snapshot.',
        replacement='26h unfinished yard jobs from final engine GATE_IN/GATE_OUT statuses',
        gate_exit_26h_residual='NOT_COLLECTED',
        raw_summary_sha256=e.sha(e.OUT/'summary.json'),policy_and_official_24h_metrics_unchanged=True)
    return summary


def audit(summary):
    checks=0
    def close(a,b):
        nonlocal checks
        assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-7),(a,b)
        checks+=1
    for row in summary['comparisons']+summary['learning_comparisons']:
        n,s=row['load'],row['model_seed'] or 0
        b=read(row['arm'],n,s)
        a=read('initial',n,s) if row['baseline']=='own_initial' else read(row['baseline'],n)
        assert a['probe']['metadata']==b['probe']['metadata'];checks+=1
        av,bv=outcome(a),outcome(b)
        close(row['cost_change_pct'],(bv['cost']-av['cost'])*100/av['cost'])
        close(row['turn_change_pct'],(bv['turn']-av['turn'])*100/av['turn'])
        close(row['turn_before_min'],av['turn']/60)
        close(row['turn_after_min'],bv['turn']/60)
        mapping=dict(admitted_change='admitted',censored_24h_change='censored',
            vessel_done_26h_change='vessel_done',discharge_done_change='discharge',
            loading_done_change='loading',vessel_delay_26h_s_change='delay',
            vessel_unfinished_26h_change='vessel_unfinished',vessel_idle_26h_s_change='idle',
            truck_yard_unfinished_26h_change='truck_yard_unfinished',p90_turn_s_change='p90',
            p99_turn_s_change='p99',vessel_cost_change='vessel_cost')
        for metric,key in mapping.items():close(row[metric],bv[key]-av[key])
    for rows_key,averages_key in [('comparisons','averages'),('learning_comparisons','learning_averages')]:
        for avg in summary[averages_key]:
            rows=[r for r in summary[rows_key] if r['arm']==avg['arm'] and r['baseline']==avg['baseline']
                  and (avg['load'] is None or r['load']==avg['load'])]
            close(avg['n'],len(rows))
            for key in rows[0]:
                if key not in ('arm','model_seed','load','baseline'):
                    close(avg[key],sum(r[key] for r in rows)/len(rows))
    for case in e.CASES:
        if case[3]:continue
        d=read(case[0],case[2],case[1]);op=d['probe']['operational']
        counts=d['probe']['flow_status_counts']
        close(sum(v for k,v in counts.items() if k.split('/')[0] in ('GATE_IN','GATE_OUT')),
              d['result']['admitted'])
        close(op['trucks_unfinished'],d['result']['n_censored'])
        for ship in op['ships']:
            streams=[r for r in op['streams'] if r['ship']==ship['ship']]
            plan=max(r['planned_s'] for r in streams)
            done=all(r['done'] and r['actual_s'] is not None for r in streams)
            finish=max(r['actual_s'] for r in streams) if done else op['horizon_s']
            close(ship['planned_s'],plan)
            close(ship['delay_s'],max(0.,min(finish,op['horizon_s'])-plan))
            assert ship['done']==done;checks+=1
        close(op['vessel_delay_s'],sum(r['delay_s'] for r in op['ships']))
        close(op['vessel_idle_s'],sum(r['idle_s'] for r in op['ship_idle'].values()))
    return dict(status='PASS',scalar_checks=checks,comparisons=len(summary['comparisons']),
                learning_comparisons=len(summary['learning_comparisons']),
                analysis_source=Path(__file__).resolve().relative_to(Path.cwd()).as_posix(),
                analysis_sha256=e.sha(__file__))


def build_report(summary):
    means={(r['arm'],r['baseline'],r['load']):r for r in summary['averages']}
    learning={(r['arm'],r['load']):r for r in summary['learning_averages']}
    lines=['# 고전 배차 규칙 7종 대비 저장 학습 정책 평가','',
        '2026-11-06, YR-332-v. 재학습 없이 저장된 모든 모델과 규칙을 새로 실행했다.',
        '실제 터미널 운영 규칙 재현이나 현장 성능 입증이 아닌, 현재 고정 시뮬레이션의 비교다.','',
        '## 진행사항','',
        '- 핵심 5종: 도착순·최단 작업·최근접·본선 우선 도착순·본선 우선 최근접.',
        '- 보조 2종: 단방향·거리와 대기시간 복합. 기존 SF_SPT는 이력 대조용으로 별도 실행.',
        '- 최근접 규칙의 거리 0 오류를 첫 취급 위치까지의 거리로 교정했다.',
        '- 21블록·42크레인, 초기 적재율 0.65, 관측 24시간+정리 2시간, 후보 수 제한·재배정 없음.',
        '- 같은 3입력일×초기화 3개, 학습 전+네 학습 방식(각 3회 갱신)의 모델 15개 전부 평가.',
        '- 총 71회: 7규칙×3일 + 15모델×3일 + 기존 규칙×3일 + 반복 2회.','',
        '### 평균 트럭 체류시간 — 분','',
        '| 방식 | 3,500건 | 7,500건 | 15,000건 |', '|---|---:|---:|---:|']
    for arm in (*e.RULES,'flow','rule_all'):
        values=[]
        for n in e.LOADS:
            ds=[read(arm,n,s) for s in e.SEEDS] if arm=='flow' else [read(arm,n)]
            values.append(statistics.mean(d['result']['mean_turn_time_s']/60 for d in ds))
        lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', '최신 모델 행은 모델 3개의 평균이다. 미완료 트럭은 24시간 종료까지의 체류만 포함한다.',
        '미투입 트럭은 평균에 포함되지 않으며, 뒤의 투입·미완료 표로 함께 판단한다.',
        '두 최단 작업 규칙은 표시 평균이 같지만 동률에서 선택·사건이 달라 동일 정책은 아니다.','',
        '### 최신 학습 정책은 각 규칙보다 나은가','',
        '| 비교 규칙 | 체류 변화 | 비용 변화 | 체류 개선/9쌍 | 본선 대기 차이(시간) |',
        '|---|---:|---:|---:|---:|']
    for base in e.RULES:
        r=means['flow',base,None]
        lines.append(f'| {LABELS[base]} | {r["turn_change_pct"]:+.2f}% | {r["cost_change_pct"]:+.2f}% | '
                     f'{r["turn_wins"]}/9 | {r["vessel_idle_26h_s_change"]/3600:+.2f} |')
    lines += ['', '음수는 학습 정책이 줄였다는 뜻이다. 날짜별 변화율을 계산한 뒤 날짜·모델에 같은 비중을 준다.',
        '본선 대기는 26시간까지 누적된 안벽 크레인 대기를 선박별 크레인 수로 나눈 시간의 합이다.',
        '이는 실제 출항 지연이 아니며 원화 비용 변화도 현장 금전 절감으로 해석하지 않는다.','',
        '### 학습 방식별 결과를 모두 공개','',
        '| 방식 | 도착순 대비 체류 | 최단 작업 대비 체류 | 같은 초기 모델 대비 체류 |',
        '|---|---:|---:|---:|']
    for arm in e.ARMS[1:]:
        a,b,c=means[arm,'fcfs_arrival',None],means[arm,'spt',None],learning[arm,None]
        lines.append(f'| {LABELS[arm]} | {a["turn_change_pct"]:+.2f}% | {b["turn_change_pct"]:+.2f}% | '
                     f'{c["turn_change_pct"]:+.2f}% |')
    lines += ['', '학습 전 모델도 규칙과 다른 선택을 한다. 규칙 대비 차이 전체를 학습 효과로 부르지 않는다.',
        '모델을 사후 선택하지 않았으며, 최신 설정이 이전 설정보다 낫다고 가정하지 않는다.','',
        '### 최신 모델에서 함께 확인한 손해','',
        '| 비교 규칙 | 투입 감소 | 24시간 미완료 증가 | 26시간 본선 야드 완료 감소 |',
        '|---|---:|---:|---:|']
    for base in e.RULES:
        r=means['flow',base,None]
        lines.append(f'| {LABELS[base]} | {r["admission_drop"]}/9 | {r["unfinished_increase"]}/9 | {r["vessel_drop"]}/9 |')
    lines += ['', '각 숫자는 학습 정책이 해당 규칙보다 나빠진 비교의 수다. 평균 체류 감소만으로 운영 채택하지 않는다.',
        '세부 양하·적하·트럭 90/99분위·26시간 야드 잔여·비용 네 항은 검증된 비교값에 보존했다.','',
        '### 15,000건 입력의 실제 작업 장부','',
        '| 방식 | 투입 트럭 | 24시간 미출차 | 26시간 야드 작업 미완료 | 26시간 본선 야드 완료 |',
        '|---|---:|---:|---:|---:|']
    for arm in (*e.RULES,'flow'):
        ds=[read(arm,15000,s) for s in e.SEEDS] if arm=='flow' else [read(arm,15000)]
        values=[statistics.mean(outcome(d)[k] for d in ds)
                for k in ('admitted','censored','truck_yard_unfinished','vessel_done')]
        lines.append(f'| {LABELS[arm]} | '+' | '.join(f'{v:,.2f}' for v in values)+' |')
    lines += ['', '소수는 모델 3개의 평균이다. 투입되지 않은 트럭이 정책마다 달라진 원인은 아직 분리하지 못했다.',
        '같은 주문을 입력했다는 사실만으로 실제 처리 대상까지 같다고 간주하지 않는다.','',
        '### 보조 잔여 지표의 시간 기준 교정','',
        '실행 후 의미 대조에서 출입 장부가 24시간까지만 갱신됨을 확인했다. 최초 수집한',
        '`operational.trucks_unfinished`와 원래 summary의 `truck_unfinished_26h_change`는',
        '26시간 출차 잔여로 사용할 수 없어 보고에서 제외했다. 71개 원자료 모두 24시간 미출차 수와 일치한다.',
        '26시간 엔진 상태에서 야드 작업 미완료를 새로 집계했고 투입 건수 보존을 확인했다.',
        '출차와 야드 작업 완료는 다른 지표다. 26시간 출차 잔여는 미수집으로 남긴다.',
        '원자료·최초 summary는 보존하고 validated-summary에 교정 결과를 따로 저장했다.',
        '공식 24시간 체류·비용 및 정책 선택·재현 대조 결과는 바뀌지 않았다.','',
        '## 해석','',
        '- 비교 기준을 넓힌 결과는 같은 환경에서 선택 방식의 장단점을 보여준다.',
        '- 모든 방식의 후보 생성 규칙·공동 안전 제약·필수 작업은 같다. 공개 정보의 표현과 기억은 다르다.',
        '  학습 입력에는 거리·시간·대기 상한이 있으며, 본선 대기열 진입 시각과 단방향 기억을 그대로 주지 않는다.',
        '- 최근접·단방향·본선 우선은 문헌 규칙을 네 작업 흐름에 맞춘 변형이다. 부산항 실제 순서가 아니다.',
        '- 최단 작업은 시뮬레이터의 계획 시간을 정확히 안다. 현실의 시간 추정 오차는 검증하지 않았다.',
        '- 비용·시간의 현실 근거, 정책별 투입 차이, 예약 오차와 양하 용량의 기존 문제는 해소되지 않았다.',
        '- 3일은 이미 본 입력이다. 9쌍을 독립 9일로 해석하거나 새 잠금평가·통계적 확증이라고 하지 않는다.','',
        '### 본선 최종 지연의 관측 한계','',
        '| 입력 물량 | 도착순의 계획 완료 범위(시간) | 26시간에 미완료 본선 |',
        '|---|---:|---:|']
    for n in e.LOADS:
        o=read('fcfs_arrival',n)['probe']['operational'];plans=[r['planned_s']/3600 for r in o['ships']]
        lines.append(f'| {n:,}건 | {min(plans):.1f}–{max(plans):.1f} | {o["vessel_unfinished"]}/{len(o["ships"])}척 |')
    lines += ['', '마감 전 평가가 끝나면 지연이 0이어도 최종 지연이 없다는 증거가 아니다.',
        '완료하지 못한 배의 지연은 관측 종료까지의 하한이다. 본선 대기·야드 완료량과 구분한다.','',
        '## 검증과 예정사항','',
        '- 관련 검사 53건 및 7규칙의 별도 입력 하루 실행 점검 통과.',
        '- 기존 모델 45회·SF_SPT 3회·도착순 3회: 후보·선택·사건·작업·트럭 장부·결과가 정확 재현.',
        '- 최근접·단방향 반복 각 1회도 정확 재현. 모든 입력·모델·실행 산출물 해시 대조.',
        '- 독립 산술 대조와 실행 설정 검증 결과는 verification.json에 기록한다.',
        '- 다음은 등록된 비교 기준·작업시간·비용 감사(t)와 투입·잔여 장부 대조(m)다.',
        '  본선 평가창이 마감까지 도달하지 못하는 한계도 t의 관측 종료 정합 감사에 포함한다.',
        '- 예약 현실화(f)와 양하 물리 보정(i)은 기존 선행 항목으로 유지한다.','',
        '[사전등록](../../../docs/research/v7-discharge/classical-rules-prereg.md) · '
        '[검증된 비교값](validated-summary.json) · [최초 집계](summary.json) · [검증](verification.json) · [실행 목록](campaign.json) · '
        '[설정](runtime-configuration.json) · [최종 판정](final-verdict.json)']
    assert len(lines)<=200
    (e.OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def main():
    raw=json.loads((e.OUT/'summary.json').read_text(encoding='utf-8'))
    summary=validated_summary(raw)
    e.save(e.OUT/'validated-summary.json',summary)
    result=audit(summary)
    e.save(e.OUT/'independent-arithmetic-audit.json',result)
    p=e.OUT/'verification.json';v=json.loads(p.read_text(encoding='utf-8'))
    v['independent_arithmetic']=result
    v['operational_count_dimensions_valid']=v.pop('all_operational_counts_valid',True)
    v['measurement_contract']=summary['measurement_correction']
    e.save(p,v)
    build_report(summary)
    from yard_rl.experiments.gate_harness import (GateOutcome, GateStatus,
        ResearchGateReport, attach_common_gates)
    latest=next(r for r in summary['averages'] if r['arm']=='flow'
                and r['baseline']=='rule_all' and r['load'] is None)
    violation=any(latest[k] for k in ('admission_drop','unfinished_increase','vessel_drop'))
    performance=GateOutcome('performance',GateStatus.FAIL if violation else GateStatus.INCONCLUSIVE,
        '운영 보호 조건 미통과' if violation else '세 입력일 진단이며 확증하지 않음',
        ('기존 최단 작업 규칙 대비 최신 모델의 사전 보호 조건을 확인함',),
        dict(baseline='SF_SPT',candidate='flow',comparisons=9,independent_days=3,
             inferential_ci_reported=False,mean_comparison=latest))
    reliability=GateOutcome('reliability',GateStatus.INCONCLUSIVE,
        '실행·산술 검증 통과, 최종 Dashboard와 원격 커밋 대조 대기',
        evidence=dict(runtime=v['runtime'],arithmetic=result,exact_prior_replays=51,exact_repeats=2))
    reality=GateOutcome('scenario_validity',GateStatus.FAIL,
        '작업시간·비용·예약 오차·양하 물리·투입 장부의 기존 미확정 유지',
        ('실자료 보정 미완료', '평가가 본선 작업 마감 전에 종료됨'),
        dict(preserved_tasks=['YR-332-t','YR-332-m','YR-332-f','YR-332-i'],
             operational_claim=False,authorization_record='authorization.json'))
    verdict=attach_common_gates(dict(task='YR-332-v',retraining=False,
        fresh_evaluations=71,operational_adoption='NOT_PASSED',
        dashboard_alignment=dict(status='PENDING'),
        scope='Descriptive comparison on three previously examined simulation days'),
        ResearchGateReport(performance,reliability,reality))
    e.save(e.OUT/'final-verdict.json',verdict)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
