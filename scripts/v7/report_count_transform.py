"""Verify raw evidence and report the preregistered count-input experiment."""
import json
import math
from pathlib import Path
import statistics

import count_transform_study as e
from report_classical_rules import outcome

LABEL = dict(clipped='기존 상한',linear='상한만 제거',log1p='로그 변환')
RULES = ('fcfs_arrival','spt','nearest','vessel_fcfs','vessel_nearest','unidirectional','distance_wait','rule_all')


def compare(a,b):
    av,bv = outcome(a),outcome(b)
    return dict(cost_change_pct=100*(bv['cost']/av['cost']-1),
        turn_change_pct=100*(bv['turn']/av['turn']-1),
        **{k+'_change':bv[k]-av[k] for k in av if k not in ('cost','turn')})


def aggregate(rows):
    out=[]
    for arm,base in sorted({(r['arm'],r['baseline']) for r in rows}):
        for load in (None,*e.LOADS):
            rs=[r for r in rows if r['arm']==arm and r['baseline']==base
                and (load is None or r['load']==load)]
            out.append(dict(arm=arm,baseline=base,load=load,n=len(rs),
                **{k:statistics.mean(r[k] for r in rs) for k in rs[0]
                   if k not in ('arm','baseline','load','model_seed')},
                admission_drop=sum(r['admitted_change']<0 for r in rs),
                unfinished_increase=sum(r['censored_change']>0 for r in rs),
                yard_unfinished_increase=sum(r['truck_yard_unfinished_change']>0 for r in rs),
                vessel_drop=sum(r['vessel_done_change']<0 for r in rs),
                turn_wins=sum(r['turn_change_pct']<0 for r in rs),
                cost_wins=sum(r['cost_change_pct']<0 for r in rs)))
    return out


def report():
    import torch
    torch.set_num_threads(1)
    manifest=json.loads((e.OUT/'campaign.json').read_text(encoding='utf-8'))
    assert manifest['sources']==e.source_hashes(), 'source changed after registration'
    for group in ('input_sha256','prior_artifact_sha256'):
        for p,h in manifest[group].items():
            assert e.sha(p)==h,p
    runs={tuple(c):e.gz_read(e.case_path(c)) for c in e.CASES}
    def read(arm,seed,phase,load):
        return runs[(arm,seed,phase,load,0)]
    def old(arm,load,seed=0):
        return e.gz_read(e.REFERENCE/'evaluation'/f'{arm}-{seed}-{load}.json.gz')
    exact=[]
    scalar_checks=0
    for case,d in runs.items():
        arm,seed,phase,load,repeat=case
        assert d['result']['policy_exceptions']==0
        if d['checkpoint']:
            assert e.sha(d['checkpoint'])==d['checkpoint_sha256']
            model=e.load_net(d['checkpoint'])
            assert model.count_transform==arm
            assert e.weights_hash(model)==d['weights_sha256']
        op=d['probe']['operational']
        assert 'trucks_unfinished' not in op
        assert op['gate_exit_censored_24h']==d['result']['n_censored']
        assert sum(v for k,v in d['probe']['flow_status_counts'].items()
                   if k.split('/')[0] in ('GATE_IN','GATE_OUT'))==d['result']['admitted']
        base=None
        if repeat:
            base=read(arm,seed,phase,load)
        elif arm=='clipped':
            base=old('initial' if phase=='initial' else 'flow',load,seed)
        elif arm=='spt':
            base=old('spt',load)
        if base:
            assert d['result']==base['result'],case
            assert d['probe']['metadata']==base['probe']['metadata'],case
            for k in e.HASHES:
                assert d['probe'][k]==base['probe'][k],(case,k)
            exact.append(e.case_name(case))
    recovery_rows=0
    recovery_saturated=0
    raw_count_max=0
    for seed,i in e.RECOVERIES:
        d=e.gz_read(e.recovery_path(seed,i))
        assert all(d[k] for k in ('exact_original_features','exact_original_choices','exact_original_result'))
        assert d['source_sha256']==e.sha(d['source'])
        original=e.gz_read(d['source'])
        assert len(d['raw_counts'])==len(original['samples'])
        for sample,n in zip(original['samples'],d['raw_counts'],strict=True):
            assert sample['row'][7]==min(n/18.,1.)
        recovery_rows+=len(d['raw_counts'])
        recovery_saturated+=sum(n>=18 for n in d['raw_counts'])
        raw_count_max=max(raw_count_max,max(d['raw_counts']))
    training=[]
    for seed in e.SEEDS:
        d=json.loads((e.OUT/'training'/f'{seed}.json').read_text(encoding='utf-8'))
        assert len(d['rows'])==9 and d['baseline_all_iterations_exact']
        training.extend(d['rows'])
        for row in d['rows']:
            a,i=row['arm'],row['iteration']
            net=e.load_net(e.model_path(a,seed,f'iteration-{i}'))
            assert e.weights_hash(net)==row['weights_sha256']
            assert net.count_transform==a
            if a=='clipped':
                source=e.load_net(e.PRIOR/f'model-{seed}'/f'flow-{i}.pt')
                assert e.weights_hash(net)==e.weights_hash(source)
        initial=[e.weights_hash(e.load_net(e.model_path(a,seed,'initial'))) for a in e.TRANSFORMS]
        assert len(set(initial))==1
    rows=[]
    for arm in e.TRANSFORMS:
        for seed in e.SEEDS:
            for load in e.LOADS:
                d=read(arm,seed,'final',load)
                baselines=dict(clipped=read('clipped',seed,'final',load),
                    own_initial=read(arm,seed,'initial',load),
                    prior_pairwise=old('pairwise',load,seed))
                baselines.update({a:old(a,load) for a in RULES})
                for base,a in baselines.items():
                    assert a['probe']['metadata']==d['probe']['metadata']
                    row=dict(arm=arm,model_seed=seed,load=load,baseline=base,**compare(a,d))
                    # Independently recompute the main relative measures by subtraction.
                    for key,raw in (('turn_change_pct','mean_turn_time_s'),('cost_change_pct','phi_krw')):
                        expected=(d['result'][raw]-a['result'][raw])*100/a['result'][raw]
                        assert math.isclose(row[key],expected,rel_tol=1e-10,abs_tol=1e-9)
                        scalar_checks+=1
                    rows.append(row)
    averages=aggregate(rows)
    lookup={(r['arm'],r['baseline'],r['load']):r for r in averages}
    decisions={}
    for a in ('linear','log1p'):
        r=lookup[(a,'clipped',None)]
        guards=all(r[k]==0 for k in ('admission_drop','unfinished_increase','yard_unfinished_increase','vessel_drop'))
        decisions[a]=dict(primary=a=='linear',cost_and_turn_improve=
            r['cost_change_pct']<0 and r['turn_change_pct']<0,guards_pass=guards,
            candidate=r['cost_change_pct']<0 and r['turn_change_pct']<0 and guards,
            default_changed=False)
    summary=dict(comparisons=rows,averages=averages,decisions=decisions,
        recovered_input_rows=recovery_rows,recovered_saturated_rows=recovery_saturated,
        recovered_max_count=raw_count_max,new_evaluations=len(runs),training_updates=len(training),
        independent_input_days=3,exact_replays=len(exact),
        scope='Same previously examined three diagnostic days; no operational adoption or statistical confirmation.',
        measurement_contract=dict(turntime='gate-in to gate-out, censored at 24h',
            yard_completion='engine state at 26h',gate_exit_26h='NOT_COLLECTED',
            vessel_delay='lower bound when unfinished or deadline outside observation'))
    e.save(e.OUT/'summary.json',summary)
    lines=['# 후보 수 입력 상한 비교 결과','',
        '2026-11-07 · YR-332-s. 같은 최신 10입력 정책에서 8번째 입력만 바꾼 진단이다.','',
        '## 진행사항','',
        f'- 학습 입력일 9개를 재생해 원래 후보 수 {recovery_rows}행을 복원했다. 기존 선택·결과는 모두 같다.',
        f'- 복원 표본 중 {recovery_saturated}행이 기존 상한에 걸렸고, 최대 후보 수는 {raw_count_max}개였다.',
        '- 같은 선택 쌍·비용·초기 가중치·회차별 학습/검증 분할·갱신 횟수를 유지했다.',
        f'- 3변환×3모델×3회차 = {len(training)}회 학습. 기존 설정의 9개 중간/최종 가중치 정확 재현.',
        f'- 학습 전후 54회·반복 2회·SPT 3회 = {len(runs)}회 평가. 기존/반복 {len(exact)}회 결과·궤적 정확 대조.','',
        '| 입력 표현 | 기존 설정 대비 체류 | 기존 설정 대비 비용 | 자체 학습 전 대비 체류 | 최단 작업 대비 체류 |',
        '|---|---:|---:|---:|---:|']
    for arm in e.TRANSFORMS:
        r=lookup[(arm,'clipped',None)]; learning=lookup[(arm,'own_initial',None)]; spt=lookup[(arm,'spt',None)]
        lines.append(f'| {LABEL[arm]} | {r["turn_change_pct"]:+.3f}% | {r["cost_change_pct"]:+.3f}% | '
                     f'{learning["turn_change_pct"]:+.3f}% | {spt["turn_change_pct"]:+.3f}% |')
    lines+=['','음수는 줄었다는 뜻이다. 변화율은 날짜별로 구한 뒤 모델·날짜에 같은 비중을 준다.',
        '입력 변경은 학습 전 행동도 바꿀 수 있으므로, 각 표현의 자체 학습 전후 효과를 따로 표시한다.','',
        '| 입력 표현·시점 | 3,500건 평균 체류(분) | 7,500건 | 15,000건 |','|---|---:|---:|---:|']
    for arm in e.TRANSFORMS:
        for phase in ('initial','final'):
            values=[statistics.mean(read(arm,s,phase,n)['result']['mean_turn_time_s']/60 for s in e.SEEDS)
                    for n in e.LOADS]
            lines.append(f'| {LABEL[arm]}·'+('학습 전' if phase=='initial' else '학습 후')+' | '+
                         ' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','## 해석','',
        '| 기존 설정보다 나빠진 비교 수/9쌍 | 투입 감소 | 24시간 미출차 증가 | 26시간 야드 잔여 증가 | 본선 완료 감소 |',
        '|---|---:|---:|---:|---:|']
    for a in ('linear','log1p'):
        r=lookup[(a,'clipped',None)]
        lines.append(f'| {LABEL[a]} | {r["admission_drop"]} | {r["unfinished_increase"]} | '
                     f'{r["yard_unfinished_increase"]} | {r["vessel_drop"]} |')
    for a,d in decisions.items():
        lines.extend(['',f'- {LABEL[a]}: 체류·비용 평균 동시 감소 {d["cost_and_turn_improve"]}, '
                      f'보호 조건 전 쌍 통과 {d["guards_pass"]}. '+
                      ('추가 검증 후보이나 기본값은 유지한다.' if d['candidate'] else '사전 지정 채택 조건 미통과, 기본값은 유지한다.')])
    lines+=['','표현 변경과 실제 처리 개선은 다르다. 입력 상한을 없앤 사실만으로 성능 개선이라고 하지 않는다.',
        '3초기 모델×3일은 독립 9일이 아니다. 이미 본 평가일이며 수렴·현장 우월성의 확증이 아니다.',
        '같은 주문을 넣어도 투입된 트럭 수가 달라질 수 있다. 평균 체류만으로 전체 수요 개선을 주장하지 않는다.',
        '본선 마감이 평가창 밖에 있거나 미완료이면 지연은 하한이다. 본선 대기는 출항 지연과 다르다.',
        '공식 출차 장부는 24시간, 야드 잔여는 26시간이다. 26시간 출차 잔여는 미수집이다.',
        '기존 원화 비용·작업시간·예약 오차·양하 물리의 외부 타당성 미해결은 그대로다.','',
        '## 예정사항','',
        '- YR-332-t: 비교 시간·비용의 근거를 맞춘다. 학습 점수와 실제 운영 개선의 관계를 판단하기 위해서다.',
        '- YR-332-m: 정책별 투입·완료 장부를 대조한다. 미투입이나 관측 종료 때문에 평균이 좋아지는지 가리기 위해서다.',
        '- 예약 현실화(f)·양하 보정(i)은 기존 선행 조건으로 유지한다. 다른 개선 축을 이번 변경에 섞지 않는다.','',
        '[사전등록](../../../docs/research/v7-discharge/count-transform-prereg.md) · [원값·전체 비교](summary.json) · [검증](verification.json)']
    (e.OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    verification=dict(status='PASS',source_integrity=True,original_evidence_unchanged=True,
        restored_labels=recovery_rows,recovery_episodes=9,training_updates=27,
        exact_training_checkpoints=9,exact_replays=exact,arithmetic_scalar_checks=scalar_checks,
        artifact_sha256={p.as_posix():e.sha(p) for folder in ('evaluation','recovered','models','training')
                        for p in (e.OUT/folder).iterdir() if p.is_file()},
        report_sha256=e.sha(e.OUT/'README.md'),summary_sha256=e.sha(e.OUT/'summary.json'))
    e.save(e.OUT/'verification.json',verification)
    from yard_rl.experiments.gate_harness import (GateOutcome,GateStatus,
        ResearchGateReport,attach_common_gates)
    primary=lookup[('linear','rule_all',None)]
    violation=any(primary[k] for k in ('admission_drop','unfinished_increase','vessel_drop'))
    performance=GateOutcome('performance',GateStatus.FAIL if violation else GateStatus.INCONCLUSIVE,
        '운영 보호 조건 미통과' if violation else '세 입력일 진단이며 확증하지 않음',
        evidence=dict(baseline='SF_SPT',candidate='linear',independent_days=3,
            inferential_ci_reported=False,mean_comparison=primary))
    reliability=GateOutcome('reliability',GateStatus.INCONCLUSIVE,
        '실행·자료·산술 검증 통과, 최종 Dashboard와 원격 커밋 대조 대기',
        evidence=dict(verification='verification.json',exact_replays=len(exact)))
    reality=GateOutcome('scenario_validity',GateStatus.FAIL,
        '작업시간·비용·예약·양하 물리·투입 장부와 본선 관측창의 미해결 유지',
        evidence=dict(preserved_tasks=['YR-332-t','YR-332-m','YR-332-f','YR-332-i'],
            operational_claim=False,authorization_record='authorization.json'))
    verdict=attach_common_gates(dict(task='YR-332-s',retraining=True,
        default_transform='clipped',decisions=decisions,operational_adoption='NOT_PASSED',
        dashboard_alignment=dict(status='PENDING'),
        verification_sha256=e.sha(e.OUT/'verification.json'),
        scope='Bounded input-transform diagnostic, not a global gate override or operational adoption.'),
        ResearchGateReport(performance,reliability,reality))
    e.save(e.OUT/'final-verdict.json',verdict)
    print(json.dumps(dict(decisions=decisions,overall=[r for r in averages
        if r['load'] is None and r['baseline'] in ('clipped','own_initial','spt')]),ensure_ascii=False),flush=True)


if __name__=='__main__':
    report()
