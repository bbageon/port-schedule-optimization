"""Human-readable audit and final evidence, with no edits to frozen experiment inputs."""
import argparse
from pathlib import Path
import json
import statistics as st

from benchmark_default_eval import save
from six_step_learning import sha,gz_read
from evaluate_observed_demand import OUT,VARIANTS,LOADS,RULES,SEEDS,CASES,PREREG,read


def markdown(path, lines):
    assert len(lines)<=200,(path,len(lines))
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def audit_report():
    a=read(OUT/'input-audit-corrected.json');o=a['observed'];actual=o['trucks'];syn=a['synthetic']['trucks']
    selection=o['reservation_selection'];cluster=read(OUT/'arrival-clustering-audit.json')
    cases=read(OUT/'input-preflight.json')['rows']
    lines=['# 실제 오더와 합성 입력의 비교','',
        '실자료는 오더 비율·시간대·예약 오차를 보정하는 데 사용할 수 있다. 터미널 전체 물량, 물리 처리 능력, 차량 방문 단위는 추가 확인이 필요하다.',
        '2026-10-28~11-09의 이미 검토한 자료를 사용했다. 최종 독립 평가 자료가 아니다.','',
        '## 작업 구성과 시간대','',
        '| 항목 | 관측 진입 오더 | 기존 학습 입력 | 기존 평가 입력 |',
        '|---|---:|---:|---:|']
    for title,key,fmt in [('오더 수','n',lambda x:f'{x:,}'),('22~06시 비중','night_22_to_06_share',lambda x:f'{x*100:.2f}%'),
                          ('가장 많은 시각','busiest_hour',lambda x:f'{x}~{x+1}시'),('최대 1시간 비중','peak_hour_share',lambda x:f'{x*100:.2f}%')]:
        lines.append('| '+title+' | '+' | '.join(fmt(d[key]) for d in (actual,syn['training'],syn['evaluation']))+' |')
    for flow,label in [('GATE_IN','반입 비중'),('GATE_OUT','반출 비중')]:
        lines.append('| '+label+' | '+' | '.join(f'{d["flow_counts"][flow]/d["n"]*100:.2f}%' for d in (actual,syn['training'],syn['evaluation']))+' |')
    lines+=['','기존 학습과 평가 모두 관측 표본보다 반출 비중이 높다. 그 자체로 반입 우선 학습의 원인이 증명된 것은 아니다.',
        '관측 일별 오더는 669~3,710건이다. 수집 대상이 전체 터미널인지 확인되지 않아 3,500/7,500/15,000건의 실험 물량을 이 수치로 대체하지 않았다.','',
        '| 시각 | 실제 오더 비중 | 기존 평가 비중 |','|---|---:|---:|']
    for h,(n,s) in enumerate(zip(actual['all_hour_counts'],syn['evaluation']['all_hour_counts'])):
        lines.append(f'| {h:02d}~{h+1:02d}시 | {100*n/actual["n"]:.2f}% | {100*s/syn["evaluation"]["n"]:.2f}% |')
    lines+=['','## 이번에 적용한 두 입력 축','',
        '반출 확률은 18,439 / 34,304로 바꾼다. 그다음 시간대별 도착 비중을 위 실제 24칸으로 바꾼다.',
        '전건 예약 조건을 유지한다. 비율과 시간대 보정에는 예약 연결 여부에 관계없이 진입 오더 전체를 포함했다.',
        '유형·시간대의 결합분포, 분 단위 몰림, 날짜 간 변동, 본선·야드 물리 조건은 추가 확인 대상이다.','',
        '| 보정 조건 | 새 입력 반출 비중 | 새 입력 야간 비중 |','|---|---:|---:|']
    for variant in VARIANTS:
        group=[r for r in cases if r['variant']==variant];total=sum(r['load'] for r in group)
        out=sum(r['flows']['GATE_OUT'] for r in group);night=sum(sum(r['hour_counts'][h] for h in (22,23,0,1,2,3,4,5)) for r in group)
        lines.append(f'| {variant} | {100*out/total:.2f}% | {100*night/total:.2f}% |')
    lines+=['','시간대 안의 도착 간격은 기존 층화 방식이다. 따라서 시간대 비중이 맞아도 실제 몰림이 재현된 것은 아니다.','',
        '## 예약 보정의 선택 편향','',
        '| 자료 | 진입 오더 | 야간 비중 |','|---|---:|---:|']
    for key,label in [('included','첫 예약 연결 가능'),('excluded','첫 예약 연결 불가')]:
        d=selection['distributions'][key];lines.append(f'| {label} | {d["n"]:,} | {100*d["night_22_to_06_share"]:.2f}% |')
    lines+=['','예약 보정에서 제외된 자료는 야간 비중이 높다. 기존 예약 오차 표본을 전체 오더와 같은 구성이라고 간주할 수 없다.',
        '위 포함군 29,285건은 13일 전체 대조용이다. 기존 정보 보정값은 앞 9일 20,936건으로 만들었으며 이번 비교에서 고정했다.','',
        '## 오더 수와 차량 방문은 다르다','',
        f'- 진입 오더 {cluster["entered_order_records"]:,}건을 차량 식별값·진입 시각으로 묶으면 {cluster["distinct_truck_entry_timestamp_keys"]:,}묶음이다.',
        f'- 같은 차량·시각에 여러 오더가 있는 묶음에 속한 기록은 {cluster["records_in_multi_order_truck_timestamps"]:,}건이다.',
        '- 이 묶음이 실제 차량 방문 단위인지는 확인이 필요하다. 컨테이너가 다른 오더를 중복으로 삭제하지 않는다.',
        '- 현재 모형의 외부 작업 한 건을 실제 차량 한 대와 바로 대응하지 않는다. 평균도 오더 기준임을 명시한다.','',
        '## 분 단위 몰림','',
        '각 날짜·시간의 실제 오더 수를 똑같이 맞춘 뒤, 분별 건수가 얼마나 들쭉날쭉한지 비교했다.',
        '아래 지수는 분별 건수의 분산을 평균으로 나눈 값이며 클수록 같은 시간 안의 몰림이 크다.','',
        '| 단위 | 실제 자료 | 고르게 배분한 대조 |','|---|---:|---:|']
    for key,label in [('matched_hour_dispersion','오더'),('grouped_truck_timestamp_dispersion','차량·진입 시각 묶음')]:
        d=cluster[key];lines.append(f'| {label} | {d["actual_mean"]:.3f} | {d["stratified_control_mean"]:.3f} |')
    lines+=['','단순 물량 차이를 맞춰도 합성 도착이 더 고르다. 차량에 연결된 여러 오더와 실제 도착 변동을 구분해 후속 보정해야 한다.','',
        '## 본선·블록·완료 시각의 적용 한계','',
        f'- 본선 목록 {o["vessels"]["snapshot_rows"]:,}행 중 관측 13일 안에 atw가 있는 것은 {o["vessels"]["atw_in_window"]["n"]:,}행이다. 전체 행을 13일 처리량으로 세지 않는다.',
        '- atw는 기록된 작업 시각이다. 안벽 작업인지 야드 완료인지 확인되기 전에는 준비 시각이나 크레인 서비스시간으로 넣지 않는다.',
        '- 스케줄 848행은 날짜별 반복 스냅샷이다. 터미널·연도·선박·항차로 합친 277항차 중 실제 접안·이안 쌍은 269개다.',
        '- 최신 스냅샷의 실제 시각으로 본 관측기간 동시 접안은 평균 2.705척, 중앙 3척이다. 실제 동시 작업 중인 크레인 수와 다르다.',
        '- 최초 감사의 중복 접안 집계는 별도 정정했다. 트럭 보정값·27개 합성 입력 감사 결과는 전부 동일하다.',
        f'- 블록으로 보이는 위치 접두가 있는 진입 기록은 {actual["known_block_n"]:,}건이다. 실제 위치 코드와 수평 21블록 모형 사이의 대응은 미확정이다.',
        '- 진입→완료 시각이 연결되는 기록은 26,566건이고, 나머지 7,738건은 완료 시각이 없다. 완료된 기록만의 평균을 전체 성능 기준으로 쓰지 않는다.',
        '- 진입 후 취소 기록은 210건이며 그중 완료 시각이 있는 것은 11건이다. 취소 의미·방문 단위 확인 전에는 물리 취소를 임의 구현하지 않는다.',
        '- 완료보다 출차가 이른 기록은 79건이다. 원자료를 수정하거나 완료 시각으로 강제 학습시키지 않는다.',
        '- 운반 대수·왕복 시간·순수 크레인 작업시간·대기열 정의는 기존 i/f에서 확인해야 한다.','',
        '[수정 감사 원값](input-audit-corrected.json) · [최초 감사 및 정정 내역](audit-correction.json) · [몰림·방문 단위](arrival-clustering-audit.json) · [고정 보정값](demand-profiles.json).']
    markdown(OUT/'input-comparison.md',lines)


def report():
    s=read(OUT/'summary.json');v=read(OUT/'verification.json');learning=read(OUT/'saved-learning-diagnostic.json')['summary']
    lines=['# 실제 오더 분포 보정 후 고정 정책 비교','',
        '작업 비율과 시간대 분포를 한 축씩 보정했다. 학습 가중치는 고정했으므로 환경 변경의 영향과 기존 모델의 성향을 보는 진단이다.',
        '트럭 지표는 오더별 게이트 진입→작업 완료이며, 24시간 안에 끝나지 않은 오더도 창 끝까지의 시간을 포함한다.','',
        '## 조건별 결과','',
        '| 조건 | 학습 전 모델 평균(분) | 학습 후 모델 평균(분) | 짧은 실작업 우선(분) | 학습 전 대비 감소 | 규칙 대비 감소 |',
        '|---|---:|---:|---:|---:|---:|']
    labels={'original':'기존 입력','observed_mix':'작업 비율 보정','observed_mix_hours':'비율+시간대 보정'}
    for variant in VARIANTS:
        d=s['by_variant'][variant];r=s['reduction_percent_vs'][variant]
        lines.append(f'| {labels[variant]} | {d["initial"]["TRUCK"]["mean_min"]:.3f} | {d["trained"]["TRUCK"]["mean_min"]:.3f} | {d["SF_SPT"]["TRUCK"]["mean_min"]:.3f} | {r["initial"]["TRUCK"]:.2f}% | {r["SF_SPT"]["TRUCK"]:.2f}% |')
    lines+=['','감소율의 음수는 시간이 늘었다는 뜻이다. 서로 다른 입력 조건에서 평균이 낮아진 것을 추가 학습의 효과로 읽지 않는다.','',
        '| 조건 | 작업 | 학습 전(분) | 학습 후(분) | 규칙(분) | 학습 전 대비 감소 |','|---|---|---:|---:|---:|---:|']
    flow_labels={'GATE_IN':'반입','GATE_OUT':'반출','VESSEL_DISCHARGE':'양하','VESSEL_LOAD':'적하'}
    for variant in VARIANTS:
        d=s['by_variant'][variant]
        for flow,label in flow_labels.items():
            lines.append(f'| {labels[variant]} | {label} | {d["initial"][flow]["mean_min"]:.3f} | {d["trained"][flow]["mean_min"]:.3f} | {d["SF_SPT"][flow]["mean_min"]:.3f} | {s["reduction_percent_vs"][variant]["initial"][flow]:.2f}% |')
    lines+=['','양하·적하는 스트림 계획 시작→개별 야드 완료 시간이며 선박 전체 체류시간이 아니다.','',
        '## 물량별 결과','',
        '| 조건 | 오더/일 | 학습 전(분) | 학습 후(분) | 규칙(분) | 학습 후 미완료(평균 건) | 규칙 미완료(건) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for load in LOADS:
            d=s['by_load'][variant][str(load)]
            lines.append(f'| {labels[variant]} | {load:,} | {d["initial"]["TRUCK"]["mean_min"]:.3f} | {d["trained"]["TRUCK"]["mean_min"]:.3f} | {d["SF_SPT"]["TRUCK"]["mean_min"]:.3f} | {d["trained"]["mean_unfinished_trucks"]:.2f} | {d["SF_SPT"]["mean_unfinished_trucks"]:.0f} |')
    lines+=['','전체 평균은 오더 수 가중이다. 15,000건 조건의 가중치는 57.69%이며 물량별 결과도 함께 봐야 한다.','',
        '## 기존 규칙 7개와의 비교','',
        '| 규칙 | 기존 입력(분) | 작업 비율 보정(분) | 비율+시간대 보정(분) |','|---|---:|---:|---:|']
    for arm in (*RULES,'initial','trained'):
        lines.append('| '+arm+' | '+' | '.join(f'{s["by_variant"][x][arm]["TRUCK"]["mean_min"]:.3f}' for x in VARIANTS)+' |')
    lines+=['','SF_SPT=실작업 중 짧은 작업 우선, FIFO=먼저 통지된 작업, LIFO=나중 통지된 작업, SPT=짧은 작업, NEAREST=가까운 작업, LWKR=오래 기다린 트럭 작업, RANDOM=무작위 기준이다.',
        '여기서 FIFO는 실제 블록 도착순인 FCFS_ARRIVAL과 다르다. 기존 ag와 같은 일곱 규칙을 고정한 비교다.',
        '규칙마다 같은 실행 가능 후보와 입력을 사용했다. 모델 3개는 같은 오더를 보며 독립된 평가일 3개로 세지 않는다.','',
        '## 저장된 학습 자료의 순위 진단','',
        f'- 기존 비교 {learning["pairs"]}쌍 중 반입·반출을 직접 비교한 것은 {learning["mixed_truck_pairs"]}쌍이다.',
        f'- 시뮬레이션 비용 비교는 반입 {learning["teacher_flows"]["GATE_IN"]}쌍, 반출 {learning["teacher_flows"]["GATE_OUT"]}쌍을 선호했다.',
        f'- 최종 모델은 같은 입력의 {learning["network_flows"]["final"]["GATE_IN"]}쌍에서 반입에 우선 점수를 줬고, 반출이 유리했던 22쌍 중 {learning["final_IN_when_teacher_OUT"]}쌍에서도 반입을 우선했다.',
        '- 저장된 비교 입력에 대한 점수 검사다. 실제 실행 선택 비율이나 새로운 자료에서의 예측 정확도와 다르다.',
        '- 자료 구성·특징 구분·회차별 학습과 이전 사례 유지 문제를 먼저 구분해야 한다. 반복 횟수만 늘리면 해결된다고 확정하지 않는다.','',
        '## 검증과 한계','',
        f'- 본 비교 {v["evaluations"]}회(고유 {v["unique_evaluations"]}회·반복 {v["exact_repeats"]}회), 기본 경로 과거 재현 {v["old_default_exact_reproductions"]}회, 관련 검사 {v["tests_passed"]}건 통과.',
        f'- 트럭 장부 {v["truck_rows"]:,}행, 전체 작업 {v["all_order_rows"]:,}행을 저장 시각에서 별도로 검산했다.',
        f'- 실행 코드 `{v["source_commit"][:8]}`, 코드·설정·가중치 {v["committed_source_hashes"]}개 해시 확인.',
        '- 물량별 신규 입력 시드가 하나다. 통계적 확증·최적성 증명·실제 터미널 성능을 주장하지 않는다.',
        '- 실제 오더/차량 단위와 수집 범위, 분 단위 몰림·유형별 시간대 결합, 순수 작업·운반 시간이 남아 있다.',
        '- 원본 13일은 이미 개발에 사용했다. 새 기간은 미확보이므로 독립 최종 평가는 수행하지 않았다.',
        '- 보정은 CPU 하루 실행의 명시적 옵션이다. 연속 월·배열 경로의 검증으로 승계하지 않는다.','',
        '[실제/합성 비교표](input-comparison.md) · [전체 원값](summary.json) · [검산](verification.json) · [실행 설정](campaign.json) · [학습 순위 진단](saved-learning-diagnostic.json) · [최종 판정](current-gates.json).']
    markdown(OUT/'README.md',lines)


def gates(commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,judge_claim_alignment,
        combine_reliability,judge_scenario_validity,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    s=read(OUT/'summary.json');v=read(OUT/'verification.json');m=read(OUT/'campaign.json')
    paths=[OUT/f for f in ('summary.json','verification.json','campaign.json','README.md','input-comparison.md',
        'input-audit.json','input-audit-corrected.json','audit-correction.json','input-preflight.json','demand-profiles.json',
        'arrival-clustering-audit.json','saved-learning-diagnostic.json','contract-tests.xml')]
    paths += sorted((OUT/'evaluation').glob('*.gz'))+sorted((OUT/'orders').glob('*.gz'))+sorted((OUT/'flow-orders').glob('*.gz'))
    runtime=judge_runtime_evidence(m['repro'],artifact_hashes={p.as_posix():sha(p) for p in paths})
    spec=Path('.claude/docs/dashboard-task-specs/YR-332-aj-observed-order-representativeness.md')
    dashboard=audit_dashboard('.',task_id='YR-332-aj',expected_state='done',spec_path=spec,
        evidence_paths=[OUT/'summary.json',OUT/'verification.json',OUT/'campaign.json'],evidence_commits=[commit],
        remote_ref='origin/강화학습-판매',pin_commit=commit)
    report_text=(OUT/'README.md').read_text(encoding='utf-8');claims={};truth={}
    for variant in VARIANTS:
        for arm in ('initial','trained','SF_SPT'):
            number=s['by_variant'][variant][arm]['TRUCK']['mean_min']
            assert f'{number:.3f}' in report_text
            claims[variant+':'+arm]=round(number,3);truth[variant+':'+arm]=number
    reliability=combine_reliability(runtime,dashboard,judge_claim_alignment(claims,truth,absolute_tolerance=.000501))
    # This preregistered study is environmental sensitivity, not a policy
    # selection experiment. Do not manufacture an inferential CI from three
    # fixed workload conditions or upgrade any previous failed policy gate.
    performance=GateOutcome('performance',GateStatus.INCONCLUSIVE,
        '고정 정책의 환경 민감도 진단이며 정책 성능 확증을 수행하지 않음',
        ('물량별 신규 입력 하나이며 독립 실자료 기간 미확보',),
        dict(metric='terminal_total_cost',baseline='SF-SPT',new_training=False,
            prior_policy_gate='FAIL: outputs/reports/yr332_completion_learning/current-gates.json',
            no_adoption=True,conditional_results=s['reduction_percent_vs']))
    scenario=judge_scenario_validity(internal_checks=dict(event_time_order=True,information_boundary=True,
        physical_constraints=True,ledger_conservation=True,achievable_vessel_deadline=False,deterministic_replay=True),
        flow_checks={},anchors={},continuous_operation=False,request_real_terminal_claim=False)
    result=attach_common_gates(dict(task='YR-332-aj',evidence_commit=commit,verification=v,
        claim_scope='Observed input marginals calibrated; incomplete physical and arrival-process validity; no new training or independent real-data evaluation.'),
        ResearchGateReport(performance,reliability,scenario))
    save(OUT/'dashboard-audit.json',dashboard.as_dict());save(OUT/'current-gates.json',result)
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print(json.dumps({k:r['status'] for k,r in result['common_gates'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('audit','report','gates'));p.add_argument('--commit')
    args=p.parse_args()
    if args.command=='audit':audit_report()
    elif args.command=='report':report()
    else:gates(args.commit)
