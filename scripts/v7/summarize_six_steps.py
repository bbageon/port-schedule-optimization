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
        vessel_done=lambda x:sum(n for k,n in x['probe']['flow_status_counts'].items()
                                if k.startswith('VESSEL_') and k.endswith('/DONE'))
        comparisons.append(dict(arm=d['arm'],model_seed=d['model_seed'],load=d['load'],seed=d['seed'],
            cost_change_pct=100*(b['phi_krw']/a['phi_krw']-1),
            turn_change_pct=100*(b['mean_turn_time_s']/a['mean_turn_time_s']-1),
            vessel_cost_change=b['c_vessel']-a['c_vessel'],
            admitted_change=b['admitted']-a['admitted'],censored_24h_change=b['n_censored']-a['n_censored'],
            completed_24h_change=(b['n_trucks']-b['n_censored'])-(a['n_trucks']-a['n_censored']),
            vessel_done_26h_change=vessel_done(d)-vessel_done(base)))
    stages=[]
    for seed_dir in sorted(root.glob('model-*')):
        for p in sorted(seed_dir.glob('stage-*.json')):
            run=read(p)
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
                worlds=0 if run['stage']==6 else sum(r['stats']['worlds'] for r in run['rows'])))
    averages={}
    for arm in sorted({r['arm'] for r in comparisons}):
        rows=[r for r in comparisons if r['arm']==arm]
        averages[arm]={k:st.mean(r[k] for r in rows) for k in
            ('cost_change_pct','turn_change_pct','admitted_change','censored_24h_change','vessel_done_26h_change')}
        means=[st.mean(r['cost_change_pct'] for r in rows if r['model_seed']==seed)
               for seed in sorted({r['model_seed'] for r in rows})]
        averages[arm]['initialization_mean_cost_range_pct']=[min(means),max(means)]
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
    result=dict(comparisons_vs_rule_all=comparisons,averages=averages,training=stages,
        direct_comparisons=paired_changes,direct_averages=direct_averages,
        artifact_sha256={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.rglob('*')) if p.is_file()
                        and (p.suffix in ('.pt','.gz') or p.name.startswith(('stage-','campaign-stage-')))},
        limitations=['three initializations crossed with the same three evaluation days',
                     'three updates per model; not a convergence or significance study',
                     'common behaviour collection; alternative learners not iterated on their own behaviour',
                     'external calibration incomplete; unfinished and unadmitted work must be checked separately'])
    save(root/'summary.json',result)
    print(json.dumps(averages,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('candidates','learning'))
    p.add_argument('folder')
    a=p.parse_args()
    (candidate_summary if a.kind=='candidates' else learning_summary)(a.folder)
