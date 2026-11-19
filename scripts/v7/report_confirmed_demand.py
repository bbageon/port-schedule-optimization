"""Evidence-based Korean report and common gate publication for YR-332-bb."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import statistics as st
import xml.etree.ElementTree as ET

from confirmed_demand_support import BASE,OUT,read,save,sha

NAMES = dict(curve='하루 곡선',mix='반출 비율',minutes='분별 몰림',links='방문 연결',all='네 축 모두')


def report(root):
    source,inputs,verify,build = (read(root/n) for n in ('source-summary.json','input-check.json','verification.json','build-record.json'))
    assert verify['status']=='PASS' and inputs['status']=='PASS'
    old,new = source['cohorts']['legacy'],source['cohorts']['confirmed']
    old_fits,new_fits = (read(root/arm/'curve-fit.json')['fits'] for arm in ('legacy','confirmed'))
    distances = [.5*sum(abs(x-y) for x,y in zip(a['hour_masses'],b['hour_masses'])) for a,b in zip(old_fits,new_fits)]
    minute = read(root/'confirmed/minute-implementation-check.json')
    suites = ET.parse(root/'tests-contract.xml').getroot().findall('testsuite')
    assert suites and all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
    tests = sum(int(s.get('tests')) for s in suites)
    effects=[]
    for arm in NAMES:
        pairs=[r for r in inputs['pairs'] if r['case']['arm']==arm]
        effects.append(dict(arm=arm,cases=len(pairs),arrival_shift_s=st.mean(r['mean_absolute_arrival_shift_s'] for r in pairs),
            visits_min=min(r['visit_count_change'] for r in pairs),visits_max=max(r['visit_count_change'] for r in pairs),
            changed_orders=sum(r['changed_fields'].get('arrival_s',0) for r in pairs)))
    result=dict(task='YR-332-bb',source=dict(legacy=old,confirmed=new,excluded=source['excluded_orders']),
        fit_dates=len(new_fits),fit_passed=sum(r['within_noise'] for r in new_fits),
        curve_distance=dict(mean=st.mean(distances),max=max(distances),max_date=new_fits[distances.index(max(distances))]['date']),
        minute_runtime=minute,axis_effects=effects,input_cases=inputs['cases'],pairs=len(inputs['pairs']),repeats=len(inputs['repeats']),
        generated_orders=inputs['generated_orders'],replay_orders=inputs['replay_orders'],tests=tests,
        independent_checks=len(verify['checks']),replayed_files=len(verify['replay_matches']),
        code_commit=build['repro']['code']['git_head'],scope='Input recalibration; no policy performance evaluation')
    save(root/'summary.json',result)
    rows='\n'.join(f"| {NAMES[r['arm']]} | {r['cases']} | {r['arrival_shift_s']:.3f} | {r['visits_min']:+d} ~ {r['visits_max']:+d} |" for r in effects)
    minute_rows='\n'.join(f"| {label} | {minute['errors'][key]:.6f} | {limit:.2f} | 통과 |"
        for label,key,limit in [('1분 단위 몰림','index_1m',.10),('10분 단위 몰림','index_10m',.15),('인접 분 상관','lag1',.05)])
    body=f'''# YR-332-bb — 수요 보정의 진입 시각 근거 통일

## ① 진행사항

**합성 도착곡선·반출 비율·분별 몰림·방문 연결의 보정 표본을 같은 원본 시각 기준으로 통일했다.**
기존 생성 수식을 유지한 새 보정과 입력 묶음을 만들었다. 원본에 명시된 진입 시각을 확인할 수 없는 601건은 사용하지 않았다.
예약·통지는 앞서 검증한 ba(예약 정보의 시각 근거 보정)의 파일을 공통 사용했다. 기존 설정과 과거 실험 결과는 그대로 남긴다.

| 보정에 쓰는 자료 | 기존 | 확인 후 |
|---|---:|---:|
| 진입 오더 | {old['orders']:,} | {new['orders']:,} |
| 묶인 트럭 방문 | {old['visits']:,} | {new['visits']:,} |
| 블록·용량을 확인한 방문 패턴 | {old['eligible_patterns']:,} | {new['eligible_patterns']:,} |
| 전체 반출 비율(%) | {100*old['gate_out_share']:.6f} | {100*new['gate_out_share']:.6f} |

오더는 컨테이너 작업 한 건, 방문은 여러 작업을 할 수 있는 트럭 한 번의 진입이다. 따라서 601건 제외와 방문 20번 감소는 다른 집계다.
기존 13개 개발 날짜(10/28~11/09)를 그대로 사용했다. 예약·통지의 9일 보정/4일 확인 구분과 혼동하지 않는다.

**하루 곡선은 {result['fit_passed']}/{result['fit_dates']}일 모두 기존 적합 기준을 통과했다.**
곡선 차이는 두 분포에서 겹치지 않는 시간대별 비중으로 쟀다. 평균{100*st.mean(distances):.4f}%, 최대{100*max(distances):.4f}%({result['curve_distance']['max_date']})다.
24개 주기 선형 성분과 비음수·합1 조건, 각 날짜1,000회 표본 비교의95% 기준을 유지했다.
반출 비율은 곡선과 같은 부모 날짜·보간 비율로 계산한다. 학습용 기준 곡선·비율에 선택/확인 조건을 섞지 않았다.

분별 몰림은 기존의 ‘인접 분 상관 + 10분 변동’ 수식으로 다시 보정했다.
아래는 **실제 생성 코드로100회 생성한 평균과 확인된 원자료 통계의 차이**다. 임계값을 바꾸지 않았다.

| 분별 적합 검사 | 오차 | 허용 오차 | 판정 |
|---|---:|---:|---|
{minute_rows}

‘몰림’은 일정 시간 동안 들어오는 방문 수의 변동을 뜻한다. 지수가 1이면 무작위 도착 수준이며, 표는 지수 자체가 아니라 원자료와의 차이다.
‘인접 분 상관’은 앞 분과 다음 분의 방문 수가 함께 움직이는 정도다. 세 검사를 통과했다는 것은 지정한 세 통계가 허용 범위에 든다는 뜻이다.
기존 표본을 넣은 보정값·반복100회 결과는 이전 저장본과 정확히 일치했다.
중간 분별 모형의 부족한10분 몰림도 보존했고, 기존10분 변동 보정까지 적용한 최종 모형을 사용한다.
분별 표의 시간 키를 숫자0~23 순으로 정렬해 JSON 저장 순서가 난수 비교를 흔들지 않게 했다.

**합성 입력 {inputs['cases']}개, 반복 생성 {len(inputs['repeats'])}개, 관련 시험 {tests}개 통과.**
본 비교{inputs['generated_orders']:,}오더와 반복{inputs['replay_orders']:,}오더를 실제 접수 변환까지 검사했다.
처음3개 학습 곡선×3물량×6조건54입력, 나머지16곡선×3물량48입력이다.
총 19개 곡선 모두에서 완성 묶음을 사용했다. 다음 표는 같은 시드의 9개 입력 쌍에서 각 축을 바꾼 효과다. 평균은 입력별 평균을 같은 비중으로 합쳤다.

| 바꾼 축 | 비교 쌍 | 도착시각 절대 차이 평균(초) | 방문 수 변화 범위 |
|---|---:|---:|---|
{rows}

위 시각 차이는 **합성 입력의 진입 시각 차이**다. 대기시간·작업 완료시간·성능 개선율이 아니다.
곡선·작업 구성이 바뀌면 시간대별 방문 묶음과 추첨 시각의 배정도 바뀐다. 같은 오더 번호의 시각 차이는 이 재생성까지 포함하므로 곡선 분포 차이와 크기가 다르다.
곡선/분별/방문 단독 변경에서는 오더 종류·대상·초기 야드·본선 계획이 같았다.
분별 변경은 시간대별 물량과 방문 연결까지 같았다. 방문 변경은 시간대별 물량을 유지했다.
반출 비율 변경은 작업 구성과 대상·배치도 바꿀 수 있으므로 같은 물리 조건이라고 보고하지 않는다.

원본600파일 불변, 별도 원본 연결·방문 재집계·곡선 최적성·분별 수식·입력 연결 검산 **{len(verify['checks'])}항목** 통과.
보정 핵심 **{len(verify['replay_matches'])}파일**은 다시 실행해 바이트까지 같았고, 입력 묶음의19개 곡선도 같은 인자로 복원됐다.
[독립 검산](verification.json) · [입력 검사](input-check.json) · [수치 요약](summary.json) · [시험](tests-contract.xml)

## ② 해석

**확인된 시각을 기준으로 입력을 만드는 경로는 연결됐다.** 표본과 보정값 변화는 작지만 성능 영향도 작다고 단정하지 않는다.
기존 실제 도착 분포를 그대로 재생하는 대신, 보정한 계수로 합성 곡선과 분별 도착을 생성한다.
방문 패턴도 보정한 빈도에 따라 합성 오더를 묶는다. 본선·정책 신경망·보상은 바꾸지 않았다.

- **기록 정합**: 보정·코드·입력·재현 근거가 맞는다. 최종 Dashboard·원격 commit도 공통 검사로 대조한다.
- **현실성**: 전체 통과가 아니다. 운반 흐름·대기 수 정의·고물량 도착 근거가 남는다.
- **성능**: 새 학습과 정책 평가를 하지 않았으므로 판단 보류다. 이전 모델의 개선율을 갱신하지 않는다.

19개 곡선의 부모 날짜·보간 비율은 기존 진단과 같고 보정된 계수만 다르다.
새로운 실제 날짜나 독립 성능 검증용 표본이라고 부르지 않는다. 7,500·15,000건은 기존 민감도 조건이며 관측 범위 밖이다.
트럭의 실제 진입을 별도 센서로 입증한 것도 아니다. 여기서 ‘확인’은 원본 요청의 명시 시각이 연결된다는 뜻이다.

## ③ 예정사항

- **YR-332-bc(전역 대기 정보의 정책 성능 비교)**: 이번 보정값을 두 학습 조건과 규칙에 공통 적용하고, 새 곡선·시드를 실행 전에 고정한다. 예측 오차만으로는 알 수 없었던 실제 정책의 비용·작업시간 차이를 확인하는 등록 작업이다.
- **YR-332-f(대기 수 정의 확인)**: 집계 대상·관측 시각·작업 시작 기록을 확인한다. 실제와 모형에서 같은 대기열을 비교하기 위해 필요하다.
- **YR-332-i(운반 흐름 확인)**: 본선 시각의 단계와 차량별 적재·인계·귀환 기록을 확인한다. 운반 대수·시간의 실측 근거를 맞추는 작업이다.
- **YR-332-ap(고물량 몰림 근거 확인)**: 더 많은 물량이 들어오는 날의 방문 자료를 확보해 외삽 범위를 확인한다. f·i·ap는 외부 자료 대기이며, bc의 합성환경 탐색과 구분한다.

## 새 입력 사용과 재현

[bundle.json](bundle.json)은 정보·곡선·수요 파일의 경로와 해시를 묶는다. 파일이 달라지면 로더가 거절한다.
과거 실험 경로는 바꾸지 않았으며 새 실험은 아래처럼 묶음을 명시한다. 코드 실행 기준은 `{result['code_commit'][:8]}`다.

```python
from yard_rl.v7.input_bundle import InputBundle
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.profiles import build_h21_profile

inputs = InputBundle("outputs/reports/yr332_confirmed_demand/bundle.json")
stage = build_stage(profile=build_h21_profile(), **inputs.case("train", 0, 3500))
```

재보정: `python scripts/v7/build_confirmed_demand.py --out <새 출력 경로>`.
입력 검사: `python scripts/v7/check_confirmed_demand.py --root <보정 경로> --out <새 검사 경로>`.
환경은 `PYTHONPATH=src;scripts/v7`, 저장소의 `../yr331-env/Scripts/python.exe`다. 원본은 `오더데이터/`를 읽는다.
기존 출력 덮어쓰기를 거절한다. [고정 계약](../../../docs/research/v7-discharge/confirmed-demand-prereg.md) · [실행 설정](build-record.json).
'''
    assert len(body.splitlines())<=200
    (root/'README.md').write_text(body,encoding='utf-8',newline='\n')
    print(dict(report='written',checks=len(verify['checks']),inputs=inputs['cases'],tests=tests))


def publish(root,commit):
    from yard_rl.experiments.gate_harness import (audit_dashboard,judge_runtime_evidence,judge_claim_alignment,
        combine_reliability,judge_scenario_validity,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates)
    summary,verify,build,inputs=(read(root/n) for n in ('summary.json','verification.json','build-record.json','input-check.json'))
    assert verify['status']=='PASS' and all(verify['checks'].values())
    assert sha(BASE/'scripts/v7/verify_confirmed_demand.py')==verify['verification_script_sha256']
    paths=[root/n for n in ('README.md','summary.json','verification.json','build-record.json','input-check.json',
        'bundle.json','bank.json','demand-base.json','source-summary.json','source-manifest.json','tests-contract.xml')]
    paths += [root/arm/n for arm in ('legacy','confirmed') for n in ('daily.json','curve-fit.json',
        'minute-templates.json','visit-link.json','correlated-minute-calibration.json','multiscale-minute-calibration.json','minute-implementation-check.json')]
    paths += [BASE/'scripts/v7'/n for n in ('verify_confirmed_demand.py','check_confirmed_demand.py','report_confirmed_demand.py')]
    artifacts={**build['source_sha256'],**{p.relative_to(BASE).as_posix():sha(p) for p in paths}}
    runtime=judge_runtime_evidence(build['repro'],artifact_hashes=artifacts)
    assert judge_runtime_evidence(inputs['repro'],artifact_hashes=artifacts).status is GateStatus.PASS
    board=audit_dashboard(BASE,task_id='YR-332-bb',expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-332-bb-confirmed-demand-calibration.md'),
        evidence_paths=[p.relative_to(BASE) for p in paths],evidence_commits=[commit],remote_ref='origin/강화학습-판매',pin_commit=commit)
    body=(root/'README.md').read_text(encoding='utf-8')
    match=re.search(r'합성 입력 (\d+)개, 반복 생성 (\d+)개, 관련 시험 (\d+)개 통과',body)
    claims=dict(zip(('inputs','repeats','tests'),map(int,match.groups())))
    truth=dict(inputs=inputs['cases'],repeats=len(inputs['repeats']),tests=summary['tests'])
    for label,key in [('진입 오더','orders'),('묶인 트럭 방문','visits'),('블록·용량을 확인한 방문 패턴','eligible_patterns'),('전체 반출 비율(%)','gate_out_share')]:
        line=next(s for s in body.splitlines() if s.startswith('| '+label+' |'))
        values=[float(x.strip().replace(',','')) for x in line.strip('|').split('|')[1:]]
        for arm,value in zip(('legacy','confirmed'),values):
            claims[f'{arm}:{key}']=value
            truth[f'{arm}:{key}']=summary['source'][arm][key]*(100 if key=='gate_out_share' else 1)
    for row in summary['axis_effects']:
        line=next(s for s in body.splitlines() if s.startswith('| '+NAMES[row['arm']]+' |'))
        fields=[s.strip() for s in line.strip('|').split('|')]
        claims[row['arm']+':shift']=float(fields[2]);truth[row['arm']+':shift']=row['arrival_shift_s']
    board_line=next(s for s in (BASE/'.claude/Dashboard/done.md').read_text(encoding='utf-8').splitlines()
                    if s.startswith('| YR-332-bb |'))
    board_match=re.search(r'(\d+)입력\+(\d+)재생·(\d+)시험·(\d+)독립검사',board_line)
    assert board_match,'Dashboard result counts missing'
    for key,value,expected in zip(('inputs','repeats','tests','independent'),map(int,board_match.groups()),
                                 (inputs['cases'],len(inputs['repeats']),summary['tests'],len(verify['checks']))):
        claims['board:'+key]=value;truth['board:'+key]=expected
    alignment=judge_claim_alignment(claims,truth,absolute_tolerance=.0005001)
    reliability=combine_reliability(runtime,board,alignment)
    scenario=judge_scenario_validity(internal_checks=dict(deterministic_replay=all(verify['replay_matches'].values())),
        flow_checks={},anchors={},continuous_operation=False,request_real_terminal_claim=False)
    performance=GateOutcome('performance',GateStatus.INCONCLUSIVE,'입력 보정·연결만 검증, 새 학습과 정책 성능 미측정')
    gates=attach_common_gates(dict(task='YR-332-bb',evidence_commit=commit,demand_time_provenance_axis='PASS',
        scope='Confirmed-source synthetic calibration and 102 input checks; no scheduling performance claim',
        unresolved_context=['YR-332-f queue definition','YR-332-i transport stages','YR-332-ap high-volume evidence']),
        ResearchGateReport(performance,reliability,scenario))
    for name,value in [('dashboard-audit.json',board.as_dict()),('current-gates.json',gates)]:
        path=root/name;save(path,value);path.with_name(path.name+'.sha256').write_text(sha(path)+'\n',encoding='utf-8',newline='\n')
    assert reliability.status is GateStatus.PASS,reliability.as_dict()
    print({k:v['status'] for k,v in gates['common_gates'].items() if isinstance(v,dict) and 'status' in v})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('report','publish'));p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--commit');a=p.parse_args()
    report(a.root.resolve()) if a.command=='report' else publish(a.root.resolve(),a.commit)
