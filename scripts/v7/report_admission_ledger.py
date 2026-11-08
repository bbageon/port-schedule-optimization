"""Independent ledger arithmetic, cohort accounting and report for YR-332-m."""
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

import audit_admission_ledger as e

LABELS=dict(fcfs_arrival='도착순',spt='최단 작업',nearest='최근접',vessel_fcfs='본선 우선·도착순',
    vessel_nearest='본선 우선·최근접',unidirectional='한 방향 이동',distance_wait='거리·대기 복합',
    flow='기존 학습',linear='상한 제거 학습',default_reference='최초 입력·기존 계산',
    default_optimized='최초 입력·개선 계산')


def measurement(rows,end):
    entered=[r for r in rows if r['times']['gate_in_s'] is not None and r['times']['gate_in_s']<=end]
    exits=sum(r['times']['gate_out_s'] is not None and r['times']['gate_out_s']<=end for r in entered)
    done=sum(r['times']['job_done_s'] is not None and r['times']['job_done_s']<=end for r in entered)
    duration=sum((min(r['times']['gate_out_s'],end) if r['times']['gate_out_s'] is not None else end)
                 -r['times']['gate_in_s'] for r in entered)
    return dict(entered=len(entered),exited=exits,censored=len(entered)-exits,
        yard_done=done,yard_unfinished=len(entered)-done,exit_travel=done-exits,
        mean_turn_s=duration/len(entered) if entered else None,total_turn_s=duration)


def summary_row(case,d,orders):
    arm,seed,n,repeat=case
    reasons=Counter(r['reason_group'] for r in orders if not r['admitted'])
    failures=[r for r in orders if r['reason_group']=='CAPACITY']
    cap=[r['attempt']['capacity'] for r in failures]
    audit=d['probe']['ledger_audit']
    c24,c26=measurement(orders,86400),measurement(orders,93600)
    assert c24['entered']==d['result']['admitted']
    assert c24['censored']==d['result']['n_censored']
    assert math.isclose(c24['mean_turn_s'],d['result']['mean_turn_time_s'],rel_tol=1e-11)
    assert c26['censored']==sum(r['admitted'] and r['state26']!='EXITED' for r in orders)
    assert c26['yard_unfinished']==sum(r['admitted'] and r['engine_status']!='DONE' for r in orders)
    assert Counter(r['flow']+'/'+r['state24'] for r in orders)==Counter(audit['states24'])
    assert Counter(r['flow']+'/'+r['state26'] for r in orders)==Counter(audit['states26'])
    vessel=Counter(r['flow']+'/'+r['status'] for r in audit['vessel_jobs'])
    return dict(arm=arm,model_seed=seed,load=n,repeat=repeat,input=len(orders),admitted=c24['entered'],
        skipped=len(orders)-c24['entered'],reasons=dict(reasons),
        capacity_failures=len(cap),physical_capacity_failures=sum(c['physical_free']<=c['margin'] for c in cap),
        reservation_only_failures=sum(c['physical_free']>c['margin'] for c in cap),
        discharge_reservation_pivotal=sum(c['free_without_discharge']>c['margin'] for c in cap),
        rejected_before_arrival=sum(r['attempt']['review_s']<r['scheduled_gate_s'] for r in failures),
        max_failed_notice_lead_s=max((r['scheduled_gate_s']-r['attempt']['review_s'] for r in failures),default=0),
        stats24=c24,stats26=c26,exits_24_to_26=c26['exited']-c24['exited'],
        by_flow={f:{str(t):measurement([r for r in orders if r['flow']==f],t) for t in (86400,93600)}
                 for f in ('GATE_IN','GATE_OUT')},
        vessel_unfinished={f:sum(v for k,v in vessel.items() if k.startswith(f+'/') and not k.endswith('/DONE'))
                           for f in ('VESSEL_LOAD','VESSEL_DISCHARGE')})


def report():
    from yard_rl.experiments.gate_harness import judge_runtime_evidence,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates
    manifest=json.loads((e.OUT/'campaign.json').read_text(encoding='utf-8'))
    assert manifest['sources']==e.source_hashes()
    for p,h in manifest['input_sha256'].items():assert e.sha(p)==h,p
    runs={};orders={};rows=[];checks=0
    for case in e.CASES:
        d=e.gz_read(e.path_for(case));source=e.gz_read(e.source_path(case))
        rr=e.gz_read(e.OUT/'orders'/f'{e.name(case)}.json.gz')
        assert d['result']==source['result']
        for k in e.HASHES:assert d['probe'][k]==source['probe'][k]
        assert d['source_sha256']==e.sha(e.source_path(case))
        assert len(rr)==len({r['job_id'] for r in rr})
        assert all(r['review_count']==1 for r in rr)
        for r in rr:
            assert r['admitted']==(r['times']['gate_in_s'] is not None)
            assert r['admitted']!=(r['reason'] is not None)
            checks+=2
        a=d['probe']['ledger_audit']
        assert a['projection24_exact'] and a['projection26_matches_engine'] and a['original_records_unchanged']
        assert a['input_orders']==a['admitted']+a['skipped'] and not a['unreviewed']
        row=summary_row(case,d,rr)
        runs[case]=d;orders[case]=rr
        if not case[3]:rows.append(row)
    repeated=('flow',9913501,15000,1);original=(*repeated[:3],0)
    assert orders[repeated]==orders[original]
    assert runs[repeated]['probe']['ledger_audit']==runs[original]['probe']['ledger_audit']
    cohorts=[]
    for n in e.LOADS:
        base=orders[('spt',0,n,0)];bi={r['job_id']:r for r in base if r['admitted']}
        for arm in ('flow','linear'):
            for seed in e.SEEDS:
                trial=orders[(arm,seed,n,0)];ti={r['job_id']:r for r in trial if r['admitted']}
                common=set(bi)&set(ti)
                pair=dict(arm=arm,model_seed=seed,load=n,common_admitted=len(common),
                    base_only_ids=sorted(set(bi)-set(ti)),candidate_only_ids=sorted(set(ti)-set(bi)))
                for end,label in ((86400,'24'),(93600,'26')):
                    b=measurement(base,end);t=measurement(trial,end)
                    bc=measurement([bi[k] for k in sorted(common)],end)
                    tc=measurement([ti[k] for k in sorted(common)],end)
                    pair['all_turn_change_pct_'+label]=100*(t['mean_turn_s']/b['mean_turn_s']-1)
                    pair['common_turn_change_pct_'+label]=100*(tc['mean_turn_s']/bc['mean_turn_s']-1)
                cohorts.append(pair)
    averages=[]
    for arm,n in sorted({(r['arm'],r['load']) for r in rows}):
        rs=[r for r in rows if r['arm']==arm and r['load']==n]
        averages.append(dict(arm=arm,load=n,models=len(rs),
            **{k:statistics.mean(r[k] for r in rs) for k in ('admitted','skipped','capacity_failures',
                'physical_capacity_failures','reservation_only_failures','discharge_reservation_pivotal','exits_24_to_26')},
            censored24=statistics.mean(r['stats24']['censored'] for r in rs),
            censored26=statistics.mean(r['stats26']['censored'] for r in rs),
            yard_unfinished26=statistics.mean(r['stats26']['yard_unfinished'] for r in rs),
            exit_travel26=statistics.mean(r['stats26']['exit_travel'] for r in rs)))
    summary=dict(task='YR-332-m',cases=len(e.CASES),distinct_cases=len(rows),exact_replays=len(e.CASES),
        order_level_checks=checks,rows=rows,averages=averages,cohorts=cohorts,
        default_pair_orders_exact=orders[('default_reference',9913320,3500,0)]==orders[('default_optimized',9913320,3500,0)],
        repeated_orders_exact=True,simulation_changed=False,official_metric_changed=False,
        scope='Fixed inputs and policies. Shared-admission subsets are post-treatment diagnostics, not causal estimates.')
    e.save(e.OUT/'summary.json',summary)
    lines=['# 투입·완료 장부 감사 결과','',
        '2026-11-08 · YR-332-m. 정책·비용·물리를 바꾸지 않고 미투입 사유와 24/26시간 장부를 대조했다.','',
        '## 진행사항','',f'- 고정 조건 {len(e.CASES)}회 모두 기존 결과와 후보·선택·사건·작업·장부 해시가 같았다.',
        f'- 오더별 독립 확인 {checks:,}건 통과. 전체 입력 = 투입 + 사유별 미투입을 전건 맞췄다.',
        '- 24시간 원장 보존, 26시간 별도 장부는 엔진의 실제 사건 시각과 일치했다. 추가 반복의 전건 장부도 같다.','',
        '| 방식 | 물량 | 투입 | 미투입 | 24시간 미출차 | 26시간 미출차 | 26시간 작업 미완료 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in averages:
        lines.append(f'| {LABELS[r["arm"]]} | {r["load"]:,} | {r["admitted"]:.2f} | {r["skipped"]:.2f} | '
                     f'{r["censored24"]:.2f} | {r["censored26"]:.2f} | {r["yard_unfinished26"]:.2f} |')
    lines+=['','모델 3개의 평균은 소수로 표시했다. 미투입은 체류 평균의 분모에 들어가지 않는다.',
        '26시간에 작업을 끝냈어도 게이트로 이동 중이면 미출차다. 두 숫자를 같은 잔여로 부르지 않는다.','',
        '## 해석','',
        '| 방식·물량 | 공간 부족 실패 | 실제 빈 칸도 부족 | 빈 칸은 있으나 예약 때문에 실패 | 양하 예정분을 빼면 한도 통과 |',
        '|---|---:|---:|---:|---:|']
    for r in averages:
        if r['skipped']:
            lines.append(f'| {LABELS[r["arm"]]}·{r["load"]:,} | {r["capacity_failures"]:.2f} | '
                f'{r["physical_capacity_failures"]:.2f} | {r["reservation_only_failures"]:.2f} | '
                f'{r["discharge_reservation_pivotal"]:.2f} |')
    total=Counter()
    for row in rows:total.update(row['reasons'])
    lines+=['',f'전체 미투입 사유(각 고정 실행 합산): {dict(total)}.',
        '공간 판단은 물리 빈 칸에서 아직 도착하지 않은 반입·양하 예정분과 다른 예약을 뺀 값이다.',
        '양하 예약이 판단을 가로막았다는 산술과, 그 예약을 없애도 안전하다는 주장은 다르다.',
        '안내자는 각 오더를 한 번만 검토한다. 실패한 오더는 이후 공간이 생겨도 재시도하지 않는다.',
        '따라서 같은 명목 입력을 넣어도 정책에 따라 실제 투입 집합이 달라질 수 있다.',
        '공통 투입 ID만 따로 계산한 값도 summary.json에 보존했다. 정책이 만든 사후 집합이므로 공정한 인과 효과로 해석하지 않는다.',
        '24시간 미완료 노출과 26시간까지의 추가 대기를 분리했다. 관측창 이후의 최종 체류는 여전히 미확정이다.',
        '이번 감사는 단일 날짜의 과부하가 정상인지, 본선 용량·시간 값이 현실적인지 승인하지 않는다.','',
        '## 예정사항','',
        '- YR-332-w: 학습에서 비교한 행동이 실제 선택 제약 안에서 가능한지 확인한다. 실행으로 구현할 수 없는 선호를 배우는지 가리기 위해서다.',
        '- YR-332-i·YR-332-d: 양하 용량과 완료를 관측하는 평가창의 기존 보정 과제는 유지한다.','',
        '[사전등록](../../../docs/research/v7-discharge/admission-ledger-prereg.md) · [오더별 원자료](orders/) · [전체 집계](summary.json)']
    assert len(lines)<=200
    (e.OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    artifacts={p.as_posix():e.sha(p) for folder in ('evaluation','orders') for p in (e.OUT/folder).glob('*.gz')}
    runtime=asdict(judge_runtime_evidence(manifest['repro'],artifact_hashes={**manifest['sources'],
        **manifest['input_sha256'],**artifacts},root='.'))
    assert runtime['status']=='PASS',runtime
    e.save(e.OUT/'verification.json',dict(status='PASS',runtime=runtime,source_integrity=True,
        exact_replays=len(e.CASES),order_level_checks=checks,artifact_sha256=artifacts,
        summary_sha256=e.sha(e.OUT/'summary.json'),report_sha256=e.sha(e.OUT/'README.md')))
    gates=ResearchGateReport(
        GateOutcome('performance',GateStatus.FAIL,'기존 정책 채택 미통과 유지; 이번 실행은 성능 개선 실험이 아님'),
        GateOutcome('reliability',GateStatus.INCONCLUSIVE,'장부·재현 검증 완료, Dashboard 원격 대조 대기'),
        GateOutcome('scenario_validity',GateStatus.FAIL,'미투입의 영구 제외와 관측창·본선 현실 보정 미해결'))
    e.save(e.OUT/'final-verdict.json',attach_common_gates(dict(task='YR-332-m',operational_adoption='NOT_PASSED',
        summary_sha256=e.sha(e.OUT/'summary.json'),verification_sha256=e.sha(e.OUT/'verification.json')),gates))
    print(json.dumps(dict(cases=len(e.CASES),checks=checks,reason_counts=total,
        selected=[r for r in averages if r['load']==15000 and r['arm'] in ('spt','flow','linear')]),ensure_ascii=False))


if __name__=='__main__':report()
