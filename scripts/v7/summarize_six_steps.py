"""Recompute diagnostic summaries from hashed, retained experiment evidence."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st

from benchmark_default_eval import save


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            return json.load(f)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified(arm):
    path=Path(arm['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==arm['sha256'],path
    return read(path)


def candidate_summary(folder):
    stamp=read(Path(folder)/'campaign.json')
    stage=stamp['stage']
    mechanism=stamp.get('mechanism','slots')
    names=(('legacy','existence' if mechanism=='existence' else 'shared')
           if stage==1 else ('cap12','all'))
    pairs=[]
    seen={}
    repeat_checks=[]
    for row in stamp['rows']:
        a,b=[verified(row['arms'][name]) for name in names]
        for name,d in zip(names,(a,b)):
            key=(row['load'],row['seed'],name)
            hashes={k:v for k,v in d['probe'].items() if k.endswith('_sha256')}
            hashes['input']=d['probe']['metadata']['input_sha256']
            hashes['model']=d['model_sha256']
            if key in seen:
                assert hashes==seen[key],('repeat changed results',key)
                repeat_checks.append(dict(load=row['load'],seed=row['seed'],arm=name,equal=True))
            else:
                seen[key]=hashes
        assert row['equal']['input'] and row['equal']['model']
        if stage==1:
            assert all(row['equal'].values())
        p=dict(load=row['load'],seed=row['seed'],repeat=row['repeat'],equal=row['equal'],arms={})
        for name,d in zip(names,(a,b)):
            probe=d['probe']
            p['arms'][name]=dict(wall_seconds=d['wall_seconds'],run_seconds=probe['run_seconds'],
                generation_seconds=probe['generation']['total_s'],selection_seconds=probe['selection']['total_s'],
                p95_ms=probe['decision']['p95_ms'],max_ms=probe['decision']['max_ms'],
                peak_rss_bytes=d['memory_after']['peak_rss_bytes'],max_candidates=probe['max_candidates_per_crane'],
                outcomes=d['result'],flow_status=probe['flow_status_counts'],metadata=probe['metadata'])
            if 'recall' in probe:
                p['arms'][name]['recall']=probe['recall']
        p['wall_change_pct']=100*(b['wall_seconds']/a['wall_seconds']-1)
        pairs.append(p)
    loads={}
    for load in sorted({p['load'] for p in pairs}):
        primary=[p for p in pairs if p['load']==load and p['repeat']==0]
        out={}
        for name in names:
            vals=[p['arms'][name] for p in primary]
            out[name]={k:st.mean(v[k] for v in vals) for k in
                ('wall_seconds','run_seconds','generation_seconds','selection_seconds','p95_ms','peak_rss_bytes')}
            out[name]['max_ms']=max(v['max_ms'] for v in vals)
            out[name]['max_candidates']=max(v['max_candidates'] for v in vals)
        out['wall_change_pct']=100*(out[names[1]]['wall_seconds']/out[names[0]]['wall_seconds']-1)
        out['all_repeat_changes_pct']=[p['wall_change_pct'] for p in pairs if p['load']==load]
        if stage == 2:
            for name in names:
                audits=[p['arms'][name]['recall'] for p in primary]
                totals={k:sum(a.get(k,0) for a in audits) for k in
                        ('eligible','retained','with_action','top_retained','mandatory',
                         'joint_checks','independent_top_joint_conflicts','audit_seconds')}
                totals['recall']=totals['retained']/max(1,totals['eligible'])
                totals['top_retention']=totals['top_retained']/max(1,totals['with_action'])
                totals['max_eligible']=max(a['max_eligible'] for a in audits)
                out[name]['recall']=totals
        loads[str(load)]=out
    result=dict(stage=stage,mechanism=mechanism,commit=stamp['commit'],n_pairs=len(pairs),
                loads=loads,pairs=pairs,repeat_checks=repeat_checks,
                interpretation='timing and candidate diagnostics; no trained-policy performance claim')
    save(Path(folder)/'summary.json',result)
    if stage==2 and len(pairs)==9:
        candidate_report(Path(folder),result)
    print(json.dumps(dict(stage=stage,mechanism=mechanism,n_pairs=len(pairs),loads=loads),indent=2))


def candidate_report(folder, summary):
    lines=['# 6단계 실험 2 — 기본 12칸과 전체 후보', '',
           f'실행 코드 `{summary["commit"]}`. 21블록·42크레인·3물량·입력 3개씩, 총 9쌍·18회.',
           '두 경로에 존재 확인 단축만 적용했다. 재취급 자리 재사용은 껐다.',
           '학습 전 고정 9입력 모델이며, 같은 쌍의 입력·모델 동일성을 확인했다.', '',
           '## 진행사항', '',
           '| 하루 예약 | 제한 방식 후보 보존 | 모델 최고 순위 보존 | 전체 방식 보존 | 최대 실행 가능 후보 |',
           '|---|---:|---:|---:|---:|']
    for load,d in summary['loads'].items():
        a,b=d['cap12']['recall'],d['all']['recall']
        lines.append(f'| {int(load):,}건 | {a["recall"]:.2%} | {a["top_retention"]:.2%} | '
                     f'{b["recall"]:.2%} | {b["max_eligible"]} |')
    lines+=['', '보존율은 매 결정에서 생성된 실행 가능 행동을 분모로 합산했다.',
            '같은 작업이 여러 시점에 다시 집계되며, 전체 잠재 오더의 비율이 아니다. 대기 행동은 분모에서 뺀다.',
            '최고 순위는 현재 모델의 점수 기준이다. 미래 비용의 실제 최선이라는 뜻이 아니다.',
            '필수 후보는 기존 방식도 12개를 넘어 보존하며, 양쪽 모두 필수 후보 누락은 없었다.', '',
            '| 하루 예약 | 비용 변화 | 평균 체류 변화 | 투입 변화 | 24시간 미완료 변화 | 26시간 본선 완료 변화 |',
            '|---|---:|---:|---:|---:|---:|']
    for load in summary['loads']:
        rows=[r for r in summary['pairs'] if r['load']==int(load)]
        metrics=[]
        for r in rows:
            aa,bb=[r['arms'][k] for k in ('cap12','all')]
            a,b=aa['outcomes'],bb['outcomes']
            vessel=lambda d:sum(n for k,n in d['flow_status'].items()
                                if k.startswith('VESSEL_') and k.endswith('/DONE'))
            metrics.append((100*(b['phi_krw']/a['phi_krw']-1),
                            100*(b['mean_turn_time_s']/a['mean_turn_time_s']-1),
                            b['admitted']-a['admitted'],b['n_censored']-a['n_censored'],
                            vessel(bb)-vessel(aa)))
        m=[st.mean(v[i] for v in metrics) for i in range(5)]
        lines.append(f'| {int(load):,}건 | {m[0]:+.2f}% | {m[1]:+.2f}% | {m[2]:+.2f}대 | '
                     f'{m[3]:+.2f}대 | {m[4]:+.2f}건 |')
    lines+=['', '각 입력에서 제한 방식→전체 후보로 바뀐 비율을 계산한 뒤 세 입력을 평균했다.',
            '비용·체류·미완료는 증가가 불리하고, 본선 완료는 감소가 불리하다.',
            '미완료 트럭도 24시간까지의 체류를 평균에 포함하지만 그 이후의 추가 대기는 포함하지 않는다.', '',
            '| 하루 예약 | 전체 계산 초(제한→전체) | 느린 5% 경계 밀리초 | 최장 결정 밀리초 | 최대 메모리 MiB |',
            '|---|---:|---:|---:|---:|']
    for load,d in summary['loads'].items():
        a,b=d['cap12'],d['all']
        lines.append(f'| {int(load):,}건 | {a["wall_seconds"]:.2f}→{b["wall_seconds"]:.2f} | '
                     f'{a["p95_ms"]:.2f}→{b["p95_ms"]:.2f} | {a["max_ms"]:.2f}→{b["max_ms"]:.2f} | '
                     f'{a["peak_rss_bytes"]/2**20:.1f}→{b["peak_rss_bytes"]/2**20:.1f} |')
    lines+=['', '시간과 느린 5% 경계·메모리는 입력 세 개 평균, 최장 결정은 세 실행의 최대다.',
            '전체 시간에는 후보 보존 검사도 포함한다. 검사 시간은 원자료에 따로 보존했다.',
            '선택과 작업 경로가 달라지므로 이 표를 동일 작업량의 순수 계산 속도 비교로 해석하지 않는다.', '',
            '| 하루 예약 | 후보 생성 초(제한→전체) | 선택 계산 초(제한→전체) |',
            '|---|---:|---:|']
    for load,d in summary['loads'].items():
        a,b=d['cap12'],d['all']
        lines.append(f'| {int(load):,}건 | {a["generation_seconds"]:.2f}→{b["generation_seconds"]:.2f} | '
                     f'{a["selection_seconds"]:.2f}→{b["selection_seconds"]:.2f} |')
    lines+=['', '이 값도 입력별 다른 작업 경로를 포함한 평균이며, 후보 하나의 계산 비용을 뜻하지 않는다.', '',
            '## 해석', '',
            '후보 누락을 없애는 경로는 구현·관측됐으나 후보 확대 자체의 운영 개선을 보장하지 않는다.',
            '기존 제한은 대기·본선 마감·작업 시간·재취급을 계산한 별도 규칙 점수로 후보를 선별한다.',
            '상한 해제에는 이 사전 선별을 제거하는 효과도 포함된다. 학습 전 모델이 넓어진 선택을 잘한다는 전제는 없다.',
            '후속 학습은 전체 후보를 공통 조건으로 사용하며, 이번 결과를 학습 성능으로 주장하지 않는다.',
            '작은 독립 상태의 실행 가능 작업 열거 시험을 통과했지만, 큰 상태의 모든 물리 행동에 대한 전역 증명은 아니다.',
            '실제 최선 행동 대비 선택 후회 비용과 결정시간 99% 경계는 이 캠페인에서 측정하지 않았다.',
            '예약·물리 용량의 현실 보정이 남아 있어 실제 터미널 성능 판정은 미통과다.', '',
            '## 다음 순서', '',
            'YR-332-g: 같은 상태에서 8번째 입력이 실제 실행 가능한 실작업 수인지 확인한다.',
            '기존 트럭 대기 수와 얼마나 달랐는지 먼저 확인하고, 수정 입력의 성능은 재학습 후 판단한다.', '',
            '원자료: [campaign](campaign.json), [해시 대조·전체 수치](summary.json).']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def learning_summary(folder):
    root=Path(folder)
    markers=list(root.glob('campaign-stage-*.json'))
    assert markers,'no completed learning campaign'
    for p in markers:
        stamp=read(p)
        assert len(stamp['completed'])=={4:36,5:12,6:12}[stamp['stage']], 'wait for the current campaign to finish'
    # The runtime writer explicitly uses UTF-8/LF. Verify that text hashes will
    # survive Git newline handling without rewriting experiment evidence.
    for p in root.rglob('*.json'):
        assert b'\r\n' not in p.read_bytes(),('noncanonical JSON newlines',p)
    data=[read(p) for p in sorted((root/'evaluation').glob('*.json.gz'))]
    by_key={(d['arm'],d['model_seed'],d['load']):d for d in data}
    comparisons=[]
    for d in data:
        if d['arm'].startswith('rule'):
            continue
        base=by_key.get(('rule_all',None,d['load']))
        if base is None:
            continue
        a,b=base['result'],d['result']
        capped=by_key[('rule_cap12',None,d['load'])]
        c=capped['result']
        vessel_done=lambda x:sum(n for k,n in x['probe']['flow_status_counts'].items()
                                if k.startswith('VESSEL_') and k.endswith('/DONE'))
        comparisons.append(dict(arm=d['arm'],model_seed=d['model_seed'],load=d['load'],seed=d['seed'],
            cost_change_pct=100*(b['phi_krw']/a['phi_krw']-1),
            turn_change_pct=100*(b['mean_turn_time_s']/a['mean_turn_time_s']-1),
            vessel_cost_change=b['c_vessel']-a['c_vessel'],
            cost_vs_cap12_pct=100*(b['phi_krw']/c['phi_krw']-1),
            turn_vs_cap12_pct=100*(b['mean_turn_time_s']/c['mean_turn_time_s']-1),
            censored_vs_cap12_change=b['n_censored']-c['n_censored'],
            vessel_done_vs_cap12_change=vessel_done(d)-vessel_done(capped),
            admitted_change=b['admitted']-a['admitted'],censored_24h_change=b['n_censored']-a['n_censored'],
            completed_24h_change=(b['n_trucks']-b['n_censored'])-(a['n_trucks']-a['n_censored']),
            vessel_done_26h_change=vessel_done(d)-vessel_done(base)))
    stages=[]
    for seed_dir in sorted(root.glob('model-*')):
        for p in sorted(seed_dir.glob('stage-*.json')):
            run=read(p)
            for r in run['rows']:
                assert hashlib.sha256(Path(r['corpus']).read_bytes()).hexdigest()==r['corpus_sha256'],r['corpus']
            corpora=[read(r['corpus']) for r in run['rows']]
            samples=[s for c in corpora for s in c['samples']]
            flows=Counter(s['flow'] for s in samples)
            cranes=Counter(f'{s["decision_key"][1]}/{s["crane"]}' for s in samples[::2])
            pairs=Counter(f'{a["flow"]}/{b["flow"]}' for c in corpora
                          for a,b in zip(c['samples'][::2],c['samples'][1::2]))
            stages.append(dict(stage=run['stage'],model_seed=run['model_seed'],
                pairs=len(samples)//2,unique_cranes=len(cranes),flow_samples=dict(flows),
                comparison_types=dict(pairs),
                decision_time_range_s=[min(s['decision_key'][0] for s in samples),
                                       max(s['decision_key'][0] for s in samples)] if samples else None,
                zero_pairs=sum(r['stats']['zero'] for r in run['rows']),
                factual_mismatch=sum(r['stats']['factual_mismatch'] for r in run['rows']),
                forced_failure=sum(r['stats']['force_failed'] for r in run['rows']),
                collection_seconds=sum(r['collection_seconds'] for r in run['rows']),
                fit_seconds=sum(f['seconds'] for r in run['rows'] for f in r['fits'].values()),
                worlds=0 if run['stage']==6 else sum(r['stats']['worlds'] for r in run['rows'])))
    averages={}
    for arm in sorted({r['arm'] for r in comparisons}):
        rows=[r for r in comparisons if r['arm']==arm]
        averages[arm]={k:st.mean(r[k] for r in rows) for k in
            ('cost_change_pct','turn_change_pct','admitted_change','censored_24h_change','vessel_done_26h_change',
             'cost_vs_cap12_pct','turn_vs_cap12_pct','censored_vs_cap12_change','vessel_done_vs_cap12_change')}
        means=[st.mean(r['cost_change_pct'] for r in rows if r['model_seed']==seed)
               for seed in sorted({r['model_seed'] for r in rows})]
        averages[arm]['training_run_mean_cost_range_pct']=[min(means),max(means)]
        averages[arm]['cost_wins']=sum(r['cost_change_pct']<0 for r in rows)
        averages[arm]['n_cases']=len(rows)
    paired_changes=[]
    for before,after in [('initial','pointwise'),('pointwise','pairwise'),
                         ('pairwise','coverage'),('coverage','flow')]:
        for (arm,seed,load),d in by_key.items():
            if arm!=after or (before,seed,load) not in by_key:
                continue
            base=by_key[(before,seed,load)]
            a,b=base['result'],d['result']
            assert base['probe']['metadata']['input_sha256']==d['probe']['metadata']['input_sha256']
            vessel_done=lambda x:sum(n for k,n in x['probe']['flow_status_counts'].items()
                                    if k.startswith('VESSEL_') and k.endswith('/DONE'))
            paired_changes.append(dict(before=before,after=after,model_seed=seed,load=load,
                cost_change_pct=100*(b['phi_krw']/a['phi_krw']-1),
                turn_change_pct=100*(b['mean_turn_time_s']/a['mean_turn_time_s']-1),
                admitted_change=b['admitted']-a['admitted'],
                censored_24h_change=b['n_censored']-a['n_censored'],
                vessel_done_26h_change=vessel_done(d)-vessel_done(base)))
    direct_averages={}
    for before,after in sorted({(r['before'],r['after']) for r in paired_changes}):
        rows=[r for r in paired_changes if (r['before'],r['after'])==(before,after)]
        direct_averages[f'{before}->{after}']={k:st.mean(r[k] for r in rows) for k in
            ('cost_change_pct','turn_change_pct','admitted_change','censored_24h_change','vessel_done_26h_change')}
        direct_averages[f'{before}->{after}'].update(n_cases=len(rows),
            cost_wins=sum(r['cost_change_pct']<0 for r in rows))
    adoption={}
    for arm in averages:
        rows=[r for r in comparisons if r['arm']==arm]
        violations=dict(admission_drop=sum(r['admitted_change']<0 for r in rows),
                        unfinished_increase=sum(r['censored_24h_change']>0 for r in rows),
                        vessel_completion_drop=sum(r['vessel_done_26h_change']<0 for r in rows))
        adoption[arm]=dict(status='FAIL' if any(violations.values()) else 'INCONCLUSIVE',
            baseline='same-candidate SF_SPT',case_count=len(rows),guard_violations=violations,
            scope='bounded diagnostic adoption guards; not a statistical confirmation test',
            reason='observed protection failures' if any(violations.values()) else
                   'three training runs on three evaluation days do not establish confirmation')
    result=dict(comparisons_vs_rule_all=comparisons,averages=averages,training=stages,
        direct_comparisons=paired_changes,direct_averages=direct_averages,
        diagnostic_adoption=adoption,
        artifact_sha256={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.rglob('*')) if p.is_file()
                        and (p.suffix in ('.pt','.gz') or p.name.startswith(('stage-','campaign-stage-')))},
        artifact_encoding='UTF-8/LF JSON verified before hashing; binary artifacts unchanged',
        limitations=['three initializations crossed with the same three evaluation days',
                     'each training run also uses different training inputs; run variation is not initialization alone',
                     'three updates per model; not a convergence or significance study',
                     'common behaviour collection; alternative learners not iterated on their own behaviour',
                     'external calibration incomplete; unfinished and unadmitted work must be checked separately'])
    save(root/'summary.json',result)
    learning_reports(root,result)
    print(json.dumps(averages,indent=2))


def learning_reports(root, summary):
    descriptions={4:'동일 자료로 개별 값과 비용 차이 학습',
                  5:'같은 수집 정책에서 비교 표본 선정 변경',
                  6:'동일 자료에 작업 방향 정보 한 칸 추가'}
    arms={4:['initial','pointwise','pairwise'],5:['pairwise','coverage'],6:['coverage','flow']}
    labels={'initial':'학습 전','pointwise':'개별 값 학습','pairwise':'비용 차이 학습',
            'coverage':'비교 표본 변경','flow':'작업 방향 추가'}
    links={4:[('initial','pointwise'),('pointwise','pairwise')],
           5:[('pairwise','coverage')],6:[('coverage','flow')]}
    finished=[]
    for stage in (4,5,6):
        stamp=root/f'campaign-stage-{stage}.json'
        if not stamp.exists():
            continue
        campaign=read(stamp)
        if len(campaign['completed'])!={4:36,5:12,6:12}[stage]:
            continue
        finished.append(stage)
        rows=[r for r in summary['training'] if r['stage']==stage]
        assert len(rows)==3
        lines=[f'# 6단계 실험 {stage} — {descriptions[stage]}', '',
               f'실행 코드 `{campaign["commit"]}`. 초기 모델 3개·각 3회 갱신·평가일 3개를 고정했다.',
               '3,500/7,500/15,000건의 서로 같은 평가 입력을 모든 모델에 사용했다.',
               '세 학습 반복은 초기 가중치와 학습 입력이 함께 다르다. 반복 간 차이를 초기값 하나의 영향으로 해석하지 않는다.',
               '전체 후보·수정된 실작업 수 입력·기존 비용식을 고정하고 존재 확인 단축만 적용했다.', '',
               '기준 규칙은 SF_SPT(실제 상자 처리 작업 우선, 그 안에서 예상 소요시간이 짧은 작업 우선)다.',
               '필수 작업 처리와 크레인 간 충돌 검사는 양쪽에 공통으로 적용한다.', '',
               '## 진행사항', '',
               '| 모델 | 같은 후보의 규칙 대비 비용 변화 | 체류 변화 | 24시간 미완료 변화 | 26시간 본선 완료 변화 | 비용 감소 사례 |',
               '|---|---:|---:|---:|---:|---:|']
        for arm in arms[stage]:
            a=summary['averages'][arm]
            assert a['n_cases']==9
            lines.append(f'| {labels[arm]} | {a["cost_change_pct"]:+.2f}% | {a["turn_change_pct"]:+.2f}% | '
                         f'{a["censored_24h_change"]:+.2f}대 | {a["vessel_done_26h_change"]:+.2f}건 | '
                         f'{a["cost_wins"]}/9 |')
        lines+=['', '음수 비용·체류 변화는 감소를 뜻한다. 미완료 증가는 불리하고, 본선 완료 증가는 유리하다.',
                '9개는 초기 모델 3개와 같은 평가일 3개의 조합이다. 독립된 평가일 9개가 아니다.',
                '세 물량을 같은 비중으로 평균한다. 실제 항만의 물량 빈도로 가중한 평균은 아니다.']
        for arm in arms[stage]:
            low,high=summary['averages'][arm]['training_run_mean_cost_range_pct']
            lines.append(f'{labels[arm]}의 학습 반복별 비용 변화 평균은 {low:+.2f}%~{high:+.2f}%다(세 물량 평균의 범위, 신뢰구간 아님).')
        lines+=['',
                '| 모델 | 원래 12개 선별 규칙 대비 비용 변화 | 체류 변화 | 미완료 변화 | 본선 완료 변화 |',
                '|---|---:|---:|---:|---:|']
        for arm in arms[stage]:
            a=summary['averages'][arm]
            lines.append(f'| {labels[arm]} | {a["cost_vs_cap12_pct"]:+.2f}% | {a["turn_vs_cap12_pct"]:+.2f}% | '
                         f'{a["censored_vs_cap12_change"]:+.2f}대 | {a["vessel_done_vs_cap12_change"]:+.2f}건 |')
        lines+=['', '원래 규칙과는 후보 범위도 다르므로 이 두 번째 표를 학습 방법 하나의 효과로 해석하지 않는다.',
                '기준 규칙을 바꿔 생긴 차이를 숨기지 않기 위한 원래 시스템 대비 결과다.', '',
                '| 직접 비교 | 비용 변화 | 체류 변화 | 투입 변화 | 24시간 미완료 변화 | 26시간 본선 완료 변화 |',
                '|---|---:|---:|---:|---:|---:|']
        for before,after in links[stage]:
            a=summary['direct_averages'][f'{before}->{after}']
            lines.append(f'| {labels[before]}→{labels[after]} | {a["cost_change_pct"]:+.2f}% | '
                         f'{a["turn_change_pct"]:+.2f}% | {a["admitted_change"]:+.2f}대 | '
                         f'{a["censored_24h_change"]:+.2f}대 | {a["vessel_done_26h_change"]:+.2f}건 |')
        lines+=['', '같은 초기 모델·같은 평가 입력끼리 변화를 계산한 뒤 평균했다.', '',
                '| 직접 비교 | 하루 예약 | 비용 변화 | 체류 변화 | 미완료 변화 | 본선 완료 변화 |',
                '|---|---:|---:|---:|---:|---:|']
        for before,after in links[stage]:
            for load in (3500,7500,15000):
                part=[r for r in summary['direct_comparisons']
                      if (r['before'],r['after'],r['load'])==(before,after,load)]
                assert len(part)==3
                m={k:st.mean(r[k] for r in part) for k in
                   ('cost_change_pct','turn_change_pct','censored_24h_change','vessel_done_26h_change')}
                lines.append(f'| {labels[before]}→{labels[after]} | {load:,}건 | {m["cost_change_pct"]:+.2f}% | '
                             f'{m["turn_change_pct"]:+.2f}% | {m["censored_24h_change"]:+.2f}대 | '
                             f'{m["vessel_done_26h_change"]:+.2f}건 |')
        lines+=['', '물량별로 같은 평가일의 세 학습 반복을 평균했다.', '',
                '| 초기 모델 | 유효 비교 쌍 | 비교한 크레인 수 | 비용 차이 0인 쌍 | 새 미래 실행 수 | 자료 수집 초 | 가중치 갱신 초 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f'| {r["model_seed"]} | {r["pairs"]} | {r["unique_cranes"]} | {r["zero_pairs"]} | '
                         f'{r["worlds"]} | {r["collection_seconds"]:.2f} | {r["fit_seconds"]:.3f} |')
        lines+=['', '유효 비교 쌍과 크레인 수는 세 회차 합계다. 같은 크레인은 블록까지 구분해 한 번 센다.',
                '초기 모델 번호는 같은 실험을 재현하기 위한 난수 시작값이다.',
                '가중치 갱신 시간은 4단계의 경우 두 학습기 합계다. 자료 생성 시간과 같은 뜻이 아니다.',
                '6단계는 5단계 자료를 재사용하므로 새 미래 실행과 수집 시간은 0이다.',
                f'분기 시작 시 원래 선택 불일치 {sum(r["factual_mismatch"] for r in rows)}건, '
                f'대안 적용 실패 {sum(r["forced_failure"] for r in rows)}건이다.',
                '이 실행 중 검사는 분기 첫 선택을 대조한다. 모든 학습 분기의 3시간 전체 궤적을 본 실행과 대조한 검사는 아니다.']
        if stage==5:
            lines+=['', '| 모델 번호 | 비교 쌍 기존→변경 | 크레인 수 기존→변경 | 반입/반출/적하/양하 표본 기존→변경 |',
                    '|---|---:|---:|---|']
            for r in rows:
                base=next(v for v in summary['training'] if v['stage']==4 and v['model_seed']==r['model_seed'])
                flow=lambda d:'/'.join(str(d['flow_samples'].get(k,0)) for k in
                                      ('GATE_IN','GATE_OUT','VESSEL_LOAD','VESSEL_DISCHARGE'))
                lines.append(f'| {r["model_seed"]} | {base["pairs"]}→{r["pairs"]} | '
                             f'{base["unique_cranes"]}→{r["unique_cranes"]} | {flow(base)}→{flow(r)} |')
            lines+=['', '한 비교는 선택·대안 두 표본을 만든다. 유형별 표본 수만으로 전체 작업의 대표성을 보장하지 않는다.',
                    '자료 수집의 실제 선택 해시와 비용·투입·잔여는 4단계와 같음을 실행 중 대조했다.']
        lines+=['',
                '## 해석과 한계', '',
                '평균 비용이나 학습 오차 하나만으로 채택하지 않는다. 처리량·미완료·본선 완료도 함께 확인한다.',
                '미완료 트럭은 24시간까지의 체류를 포함하지만 그 이후 시간은 아직 포함하지 않는다.',
                '3회 갱신은 원인 분리를 위한 짧은 진단이며 충분한 학습이나 통계적 확증을 뜻하지 않는다.',
                '개별 값 학습 정책이 공통 자료를 수집했다. 다른 학습기가 자기 행동으로 계속 자료를 다시 모은 실험은 아니다.',
                '예약 오차·본선 용량·미투입 작업의 현실 보정이 남아 실제 항만의 절감 성과를 주장하지 않는다.',
                '원자료의 검증 손실은 학습에서 빼 둔 비교 쌍의 지표이며, 이 표의 별도 날짜 운영 평가와 구분한다.', '',
                '같은 후보의 규칙 대비 보호 조건을 위반한 실행 수는 아래와 같다. 서로 겹치는 실행도 각각 센다.', '']
        for arm in arms[stage]:
            g=summary['diagnostic_adoption'][arm]
            v=g['guard_violations']
            lines.append(f'- {labels[arm]}: 투입 감소 {v["admission_drop"]}회, 미완료 증가 '
                         f'{v["unfinished_increase"]}회, 본선 완료 감소 {v["vessel_completion_drop"]}회 / 9회.')
        lines+=['', '보호 조건 미충족은 이번 운영 채택의 미통과를 뜻한다. 해당 방법이 항상 실패한다는 증명은 아니다.', '',
                '## 다음 순서', '']
        lines += {4:['YR-332-q: 같은 수집 정책·입력에서 비교 대상을 고르는 방식만 바꾼다.',
                    '자료의 범위가 넓어지는지와 운영 효과를 분리 확인한다.'],
                  5:['YR-332-r: 같은 자료에 컨테이너를 놓는 작업인지 꺼내는 작업인지만 추가한다.',
                     '기존 초기 가중치를 복사하고 새 입력 가중치는 0으로 시작한다.'],
                  6:['6단계 진단은 여기까지다. 결과를 종합하고 현실 보정 작업과 장기 학습의 선행 조건을 구분한다.']}[stage]
        lines += ['', f'원자료: [실행 기록](campaign-stage-{stage}.json), '
                  f'[단계 종료 집계](stage{stage}-summary.json). 모델과 학습 자료는 `model-*/`에 보존한다.']
        (root/f'stage{stage}-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if finished:
        save(root/f'stage{max(finished)}-summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('candidates','learning'))
    p.add_argument('folder')
    a=p.parse_args()
    (candidate_summary if a.kind=='candidates' else learning_summary)(a.folder)
