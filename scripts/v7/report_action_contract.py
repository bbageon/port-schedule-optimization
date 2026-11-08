"""Recheck reachability certificates and partition saved validation metrics."""
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

import audit_action_contract as e
from action_reachability import replay_certificate


def family_summary(rows):
    unreachable=[r for r in rows if not r['proof']['reachable']]
    gap=lambda r:abs(r['phi_picked']-r['phi_alternative'])
    normalized=lambda r:abs(r['target_picked']-r['target_alternative'])
    total_gap=sum(gap(r) for r in rows)
    total_normalized=sum(normalized(r) for r in rows)
    return dict(pairs=len(rows),mixed_tier=sum(r['mixed_service_tier'] for r in rows),
        unreachable=len(unreachable),reachable=len(rows)-len(unreachable),
        forced_success=sum(r['forced_selected'] for r in rows),
        unreachable_alternative_better=sum(r['phi_alternative']<r['phi_picked'] for r in unreachable),
        unreachable_abs_cost_gap_share=sum(gap(r) for r in unreachable)/total_gap if total_gap else 0.,
        unreachable_abs_target_gap_share=sum(normalized(r) for r in unreachable)/total_normalized if total_normalized else 0.,
        unreachable_median_abs_cost_gap=statistics.median(gap(r) for r in unreachable) if unreachable else None,
        reachable_median_abs_cost_gap=statistics.median(gap(r) for r in rows if r not in unreachable),
        clipped_shared_witness=sum(r['score_witnesses'].get('clipped',{}).get('target_selected',False) for r in rows),
        linear_shared_witness=sum(r['score_witnesses'].get('linear',{}).get('target_selected',False) for r in rows))


def validation(case,run):
    import torch
    from torch.nn.functional import smooth_l1_loss
    seed,it,_=case;root=e.PRIOR/f'model-{seed}';output=[];rechecked=[]
    for arm,stage in (('pairwise',4),('coverage',5),('flow',5),('linear',5)):
        data=e.gz_read(root/f'labels-{stage}-{it}.json.gz')
        samples=data['samples'];n=len(samples)//2
        val_ids=set(torch.randperm(n,generator=torch.Generator().manual_seed(data['seed'])).tolist()[:max(1,int(n*.2))])
        audit={r['pair_index']:r for r in run['pairs'] if r['stage']==stage}
        model=(e.frozen.model_path('linear',seed,f'iteration-{it}') if arm=='linear' else root/f'{arm}-{it}.pt')
        net=e.load_net(model)
        indices=[j for i in range(n) if i in val_ids for j in (2*i,2*i+1)]
        xs=[]
        for j in indices:
            row=list(samples[j]['row'])
            if net.include_flow:row.append(float(samples[j]['is_store']))
            if arm=='linear':
                a=audit[j//2];graph=run['graphs'][a['graph']]
                selected=a['picked'] if j%2==0 else a['alternative']
                row[7]=graph['items'][selected]['raw_count']/18.
            xs.append(row)
        with torch.no_grad():pred=net(torch.tensor(xs,dtype=torch.float32))
        target=torch.tensor([samples[j]['target'] for j in indices],dtype=torch.float32)
        pg=pred[::2]-pred[1::2];yg=target[::2]-target[1::2]
        loss=smooth_l1_loss(pg,yg,reduction='none',beta=1.)
        nonzero=yg.abs()>1e-9
        acc=float((pg[nonzero].sign()==yg[nonzero].sign()).float().mean()) if nonzero.any() else 0.
        if arm=='linear':
            training=json.loads((e.frozen.OUT/'training'/f'{seed}.json').read_text(encoding='utf-8'))
            expected=next(r['fit'] for r in training['rows'] if r['arm']=='linear' and r['iteration']==it)
        else:
            train_stage={'pairwise':4,'coverage':5,'flow':6}[arm]
            training=json.loads((root/f'stage-{train_stage}.json').read_text(encoding='utf-8'))
            expected=next(r['fits'][arm] for r in training['rows'] if r['it']==it)
        assert math.isclose(float(loss.mean()),expected['val_pair_loss'],rel_tol=1e-6,abs_tol=1e-7),(case,arm)
        assert acc==expected['val_order_accuracy'],(case,arm,acc,expected['val_order_accuracy'])
        assert int(nonzero.sum())==expected['val_nonzero_pairs']
        rechecked.append(dict(arm=arm,model_seed=seed,iteration=it,val_pair_loss=float(loss.mean()),
            val_order_accuracy=acc,val_nonzero_pairs=int(nonzero.sum()),checkpoint_sha256=e.sha(model)))
        for k,i in enumerate(sorted(val_ids)):
            a=audit[i]
            output.append(dict(arm=arm,model_seed=seed,iteration=it,pair_index=i,
                unreachable=not a['proof']['reachable'],nonzero=bool(nonzero[k]),
                correct=bool(pg[k].sign()==yg[k].sign()),loss=float(loss[k]),
                teacher_alternative_better=bool(yg[k]>1e-9),model_prefers_alternative=bool(pg[k]>0)))
    return output,rechecked


def report():
    import torch
    from yard_rl.experiments.gate_harness import judge_runtime_evidence,GateOutcome,GateStatus,ResearchGateReport,attach_common_gates
    torch.set_num_threads(1)
    manifest=json.loads((e.OUT/'campaign.json').read_text(encoding='utf-8'))
    assert manifest['sources']==e.source_hashes()
    for p,h in manifest['input_sha256'].items():assert e.sha(p)==h,p
    data={c:e.gz_read(e.OUT/'evaluation'/f'{e.name(c)}.json.gz') for c in e.CASES}
    pairs=[];values=[];rechecked=[];proofs=0
    for case,d in data.items():
        assert all(d[k] for k in ('exact_features','exact_original_choices','exact_original_result'))
        for p,h in d['source_sha256'].items():assert e.sha(p)==h,p
        for row in d['pairs']:
            graph=d['graphs'][row['graph']]
            def compatible(a,b):return graph['oracle']['/'.join(map(str,sorted((a,b))))]
            replay_certificate(graph['items'],row['alternative'],row['proof'],compatible)
            assert row['forced_selected'] and row['forced_joint_safe']
            assert row['forced_rank'][0]==-1
            proofs+=1
        if not case[2]:
            pairs.extend([dict(r,model_seed=case[0],iteration=case[1]) for r in d['pairs']])
            val,check=validation(case,d);values.extend(val);rechecked.extend(check)
    original=data[e.CASES[0]];repeat=data[e.CASES[-1]]
    assert original['pairs']==repeat['pairs'] and original['graphs']==repeat['graphs']
    assert original['result']==repeat['result']
    for k in e.frozen.HASHES:assert original['probe'][k]==repeat['probe'][k]
    families={str(stage):family_summary([r for r in pairs if r['stage']==stage]) for stage in (4,5)}
    accuracy=[]
    for arm in ('pairwise','coverage','flow','linear'):
        for group in ('all','reachable','unreachable'):
            rs=[r for r in values if r['arm']==arm and (group=='all' or r['unreachable']==(group=='unreachable'))]
            nz=[r for r in rs if r['nonzero']]
            accuracy.append(dict(arm=arm,group=group,pairs=len(rs),nonzero=len(nz),
                accuracy=sum(r['correct'] for r in nz)/len(nz) if nz else None,
                mean_pair_loss=statistics.mean(r['loss'] for r in rs) if rs else None,
                correct_unexecutable_alt=sum(r['unreachable'] and r['teacher_alternative_better']
                    and r['model_prefers_alternative'] for r in nz)))
    summary=dict(task='YR-332-w',replay_episodes=len(data),original_training_days=9,
        certificate_checks=proofs,training_families=families,validation=accuracy,
        validation_records=values,validation_checkpoint_rechecks=rechecked,
        pair_rows=pairs,repeat_exact=True,new_training=False,
        scope='Fixed historical pairs; exact two-crane reachability with independently varying scores is an upper bound. Shared-feature score witnesses are not neural-network realizability proofs. No learning-effect causal claim.')
    e.save(e.OUT/'summary.json',summary)
    lines=['# 학습 대안의 실제 선택 가능성 감사','',
        '2026-11-08 · YR-332-w. 저장된 학습 자료를 만든 9일과 반복 1회를 그대로 재생했다. 새 학습이나 교사 비용 계산은 하지 않았다.','',
        '## 진행사항','',
        '- 당시 입력·실제 선택·하루 결과와 기존 복원 실행의 해시를 정확 대조했다.',
        f'- 후보의 필수 등급·실작업 우선·두 크레인 공동 안전 제약을 유지한 판정 증거 {proofs}건을 다시 검사했다.',
        '- 다른 크레인이 먼저 고른 작업에 따라 결과가 달라질 수 있으므로 가능한 첫 선택을 모두 검사했다.',
        '- 현재 2크레인 결정에 한정한다. 3크레인 이상이면 판정하지 않고 중단하도록 했다.','',
        '| 학습 자료 | 비교 쌍 | 실작업/준비 혼합 | 점수만으로 대안 선택 불가 | 그중 교사는 대안이 낫다고 평가 |',
        '|---|---:|---:|---:|---:|']
    for stage,label in (('4','기존 표본'),('5','확대 표본·최신 학습')):
        r=families[stage]
        lines.append(f'| {label} | {r["pairs"]} | {r["mixed_tier"]} | {r["unreachable"]} | {r["unreachable_alternative_better"]} |')
    lines+=['','두 자료는 같은 9일에서 수집했다. 합쳐서 독립 396개 상황이라고 해석하지 않는다.',
        '선택 불가는 신경망 점수를 임의로 바꿔도 고정 우선순위와 공동 배정 제약에 막힌다는 뜻이다.',
        '선택 가능 판정은 후보별 점수가 자유롭다는 상한이다. 같은 입력에 같은 점수를 주는 증인도 따로 기록했다.',
        '그 증인을 현재 크기의 신경망이 실제 구현할 수 있다는 증명은 아니다.','',
        '| 학습 자료 | 선택 불가 쌍의 절대 비용 차이 비중 | 정규화 목표 차이 비중 | 같은 입력·같은 점수로 선택 성공 |',
        '|---|---:|---:|---:|']
    for stage,label in (('4','기존 표본'),('5','확대 표본·최신 학습')):
        r=families[stage]
        lines.append(f'| {label} | {100*r["unreachable_abs_cost_gap_share"]:.2f}% | {100*r["unreachable_abs_target_gap_share"]:.2f}% | {r["clipped_shared_witness"]}/{r["reachable"]} |')
    lines+=['','목표 차이 비중은 학습 신호 크기의 진단이며 실제 기울기 기여나 성능 저하의 원인 비율이 아니다.','',
        '## 해석','',
        '교사의 강제 선택은 우선순위를 -1로 올려 실작업 우선 순서를 우회한다. 필수 작업과 공동 안전 검사는 유지된다.',
        '실행 가능한 물리적 행동과 현재 정책의 점수만으로 선택 가능한 행동은 구분해야 한다.',
        '학습에서 비용이 낮다고 배워도 고정 우선순위가 막으면 그 선호를 실행하지 못할 수 있다.','',
        '| 저장 모델 | 검증 부류 | 비교 쌍(비용 차이 있음) | 비용 순서 정확도 |',
        '|---|---|---:|---:|']
    names=dict(pairwise='비용 차이 학습',coverage='표본 확대',flow='작업 방향 추가',linear='입력 상한 제거')
    groups=dict(all='전체',reachable='대안 선택 가능 상한',unreachable='대안 선택 불가')
    for r in accuracy:
        value='해당 없음' if r['accuracy'] is None else f'{100*r["accuracy"]:.2f}%'
        lines.append(f'| {names[r["arm"]]} | {groups[r["group"]]} | {r["pairs"]} ({r["nonzero"]}) | {value} |')
    lines+=['','9회차의 원래 검증 분할을 그대로 다시 계산했다. 표본 수로 합산한 정확도이며 회차별 정확도의 단순 평균과 다를 수 있다.',
        f'저장 모델 {len(rechecked)}개의 검증 손실·정확도·유효 비교 수가 원래 학습 보고와 일치했다.',
        '이는 비용 순서를 맞히는 정확도다. 터미널 체류시간 개선이나 장기 학습 효과를 대신하지 않는다.',
        '자료를 삭제하거나 우선순위를 풀지 않았다. 이 문제를 고쳤을 때 실제 성능이 좋아지는지는 별도 비교가 필요하다.','',
        '## 예정사항','',
        '- 확인된 계약 문제의 수정은 사전등록한 단일축 비교로 분리한다. 기존 학습·평가 결과는 보존한다.',
        '- YR-332-m·YR-332-i·YR-332-d의 투입 보존·본선 물리·관측창 문제와 혼합하지 않는다.','',
        '[사전등록](../../../docs/research/v7-discharge/action-contract-prereg.md) · [전체 증거](summary.json) · [재생 원자료](evaluation/)']
    assert len(lines)<=200
    (e.OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    artifacts={p.as_posix():e.sha(p) for p in (e.OUT/'evaluation').glob('*.gz')}
    runtime=asdict(judge_runtime_evidence(manifest['repro'],artifact_hashes={**manifest['sources'],
        **manifest['input_sha256'],**artifacts},root='.'))
    assert runtime['status']=='PASS',runtime
    e.save(e.OUT/'verification.json',dict(status='PASS',runtime=runtime,source_integrity=True,
        exact_episodes=len(data),certificate_checks=proofs,validation_checkpoints=len(rechecked),
        artifact_sha256=artifacts,summary_sha256=e.sha(e.OUT/'summary.json'),report_sha256=e.sha(e.OUT/'README.md')))
    gates=ResearchGateReport(
        GateOutcome('performance',GateStatus.FAIL,'기존 정책 채택 미통과 유지; 새 학습 성능 실험 없음'),
        GateOutcome('reliability',GateStatus.INCONCLUSIVE,'재현·도달 가능성 증거·검증 점수 대조 통과, 원격 Dashboard 대조 대기'),
        GateOutcome('scenario_validity',GateStatus.FAIL,'학습/선택 계약과 기존 투입·예약·본선 현실성 미해결'))
    e.save(e.OUT/'final-verdict.json',attach_common_gates(dict(task='YR-332-w',operational_adoption='NOT_PASSED',
        summary_sha256=e.sha(e.OUT/'summary.json'),verification_sha256=e.sha(e.OUT/'verification.json')),gates))
    print(json.dumps(dict(families=families,validation=accuracy,validation_checkpoints=len(rechecked)),ensure_ascii=False))


if __name__=='__main__':report()
