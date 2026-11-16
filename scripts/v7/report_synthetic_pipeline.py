"""Recompute every reported time from saved order ledgers, then compare paired arms."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from six_step_learning import gz_read,sha
from horizon_metrics import aggregate
from verify_job_completion import raw_metrics,assert_metrics
from yard_rl.integrated.repro import write_result
from yard_rl.v7.evaluation_probe import digest

FLOW_NAMES=dict(TRUCK='트럭 전체',GATE_IN='반입',GATE_OUT='반출',VESSEL_DISCHARGE='양하',VESSEL_LOAD='적하')


def audit(root):
    records=[];artifacts={}
    for path in sorted((root/'evaluation').glob('*.json.gz')):
        d=gz_read(path);orders_path=root/'orders'/path.name;flows_path=root/'flow-orders'/path.name
        orders=gz_read(orders_path);flows=gz_read(flows_path)
        assert digest(orders)==d['orders_sha256']
        metrics=raw_metrics(orders);assert metrics==d['metrics'];assert_metrics(metrics,d['result'])
        assert len({r['job_id'] for r in flows})==len(flows)
        assert all(aggregate(flows,h*3600.)==d['completion'][f'at{h}'] for h in (24,26))
        assert d['result']['policy_exceptions']==d['result']['traded_edges']==d['result']['rollout_worlds']==0
        assert d['result']['admitted']==d['case']['load']
        assert all(r['review_count']==1 for r in orders)
        for p in (path,orders_path,flows_path):artifacts[p.as_posix()]=sha(p)
        if 'repeat' not in path.name:records.append(d)
    return records,artifacts


def combined(rows):
    out={}
    for flow in FLOW_NAMES:
        counts=sum(r['completion']['at24'][flow]['n'] for r in rows)
        total=sum(r['completion']['at24'][flow]['total_s'] for r in rows)
        out[flow]=dict(n=counts,mean_min=total/counts/60 if counts else None,
            unfinished24=sum(r['completion']['at24'][flow]['unfinished'] for r in rows),
            unfinished26=sum(r['completion']['at26'][flow]['unfinished'] for r in rows))
    return dict(flows=out,cost=sum(r['result']['phi_krw'] for r in rows)/len(rows))


def inputs(axis):
    root=Path('outputs/reports')/('yr332_synthetic_inputs' if axis=='mix' else 'yr332_parametric_arrivals')
    rows,artifacts=audit(root)
    expected=54 if axis=='mix' else 36
    assert len(rows)==expected,(len(rows),expected)
    if axis=='minutes':
        for name in ('input-audit.json','paired-attribute-audit.json','block-arrival-audit.json'):
            p=root/name;artifacts[p.as_posix()]=sha(p)
    arms=('fixed','variable') if axis=='mix' else ('observed','parametric')
    groups=[];paired=[]
    for load in (3500,7500,15000):
        for rule in ('SF_SPT','FIFO','LWKR') if axis=='mix' else ('SF_SPT','FIFO'):
            armrows={arm:[r for r in rows if r['case']['load']==load and r['case']['rule']==rule and r['case']['arm']==arm] for arm in arms}
            a,b=[combined(armrows[arm]) for arm in arms]
            changes={f:100*(b['flows'][f]['mean_min']/a['flows'][f]['mean_min']-1) for f in FLOW_NAMES}
            groups.append(dict(load=load,rule=rule,baseline=a,candidate=b,changes=changes,cost_change=100*(b['cost']/a['cost']-1)))
            for i in range(3):
                aa,bb=[next(r for r in armrows[arm] if r['case']['index']==i) for arm in arms]
                paired.append(dict(load=load,rule=rule,index=i,change=100*(bb['result']['mean_gate_to_job_done_s']/aa['result']['mean_gate_to_job_done_s']-1)))
                if axis=='minutes':
                    assert aa['case']['seed']==bb['case']['seed']
    summary=dict(cases=len(rows),axis=axis,groups=groups,paired=paired,artifact_hashes=artifacts,
                 change_range=[min(r['change'] for r in paired),max(r['change'] for r in paired)])
    write_result(root/'summary.json',summary)
    title='날짜별 반출 비율' if axis=='mix' else '수식 기반 분별 방문 도착'
    lines=[f'# {title} 비교', '', '## ① 진행사항','',f'- 기존 규칙 {len(rows)}회 실행, 모든 오더 원자료에서 시간·미완료·장부를 재계산했다.',
        '- 표의 +는 더 오래 걸림이다. 각 물량·규칙에서 세 입력의 오더 수로 가중한 평균이다.',
        '- 트럭은 오더당 게이트 진입→작업 완료, 24시간 경계에서 미완료 경과시간을 포함한다.', '',
        '| 물량 | 규칙 | 대조(분) | 변경(분) | 변화 |', '|---:|---|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {g["load"]:,} | {g["rule"]} | {g["baseline"]["flows"]["TRUCK"]["mean_min"]:.3f} | {g["candidate"]["flows"]["TRUCK"]["mean_min"]:.3f} | {g["changes"]["TRUCK"]:+.2f}% |')
    lines += ['', 'SF_SPT는 실작업 중 짧은 작업 우선, FIFO는 작업 대기에 들어온 순서, LWKR는 남은 작업량이 적은 순서다.',
        '', '### 실작업·짧은 작업 우선의 작업별 결과', '', '| 물량 | 종류 | 대조(분) | 변경(분) | 변화 | 26시간 미완료 대조/변경 |',
        '|---:|---|---:|---:|---:|---:|']
    for g in groups:
        if g['rule']!='SF_SPT':continue
        for f in ('GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD'):
            a,b=g['baseline']['flows'][f],g['candidate']['flows'][f]
            lines.append(f'| {g["load"]:,} | {FLOW_NAMES[f]} | {a["mean_min"]:.3f} | {b["mean_min"]:.3f} | {g["changes"][f]:+.2f}% | {a["unfinished26"]}/{b["unfinished26"]} |')
    lines += ['', '## ② 해석','',f'- 개별 짝 비교의 트럭 시간 변화는 {summary["change_range"][0]:+.2f}~{summary["change_range"][1]:+.2f}%다.',
        '- 입력 변화의 영향이다. 새 학습의 개선율이나 실제 터미널 성능으로 해석하지 않는다.',
        '- 세 개발 곡선이며 고물량은 부담 시험이다. 본선은 스트림 시작→개별 야드 작업 완료로 트럭 시간과 기준이 다르다.',
        '- 시간대별 합성 모양의 부모가 같은 개발 자료다. 독립 실측 검증이 아니다.']
    if axis=='mix':
        lines += ['- 방문 연결·재고 제약을 유지한다. 비율 변화는 방문 구성·초기 공급 배치도 바꿀 수 있다.',
                  '- 고정 비율은 학습용24개의 평균이며 선택한3개의 평균과 정확히 같다는 보장은 없다.']
    else:
        assert all(r.get('prior_run_exact') for r in rows if r['case']['arm']=='observed')
        lines += ['- 108쌍에서 비시각 오더 속성·방문 계획·재고·본선·시간대 건수는 같고 시각만 달라졌다.',
                  '- 관측 분별 배열을 쓰는 대조18회는 앞 단계의 동일 조건과 결정·사건·원자료가 정확히 같다.',
                  '- 새 도착은 분별 배열을 재사용하지 않는다. 개발 자료의 집계 통계로 보정한 수식 기반 합성이다.',
                  '- 독립 균등·Dirichlet과 첫 상관 모형의 실패를 보존했다. 기준을 완화하지 않고 입력만 보고 모형을 수정했다.',
                  '- 7,500·15,000건은 같은 강도 변동을 외삽한 부담 시험이며 실측 고물량 적합성을 입증하지 않았다.',
                  '- 시뮬레이터의 실제 블록 도착도 별도 검산했다. 3,500건·짧은 작업 우선의 1분 몰림은1.076→1.073이다.',
                  '  게이트 시각을 목적 블록으로 나눈1.223→1.231과 다르다. 24시간 미도착18/23건도 남겨 조건부 통계임을 표시했다.',
                  '  [블록 도착·도착 간격 원값](block-arrival-audit.json). 현장 블록 도착 자료와의 검증은 아니다.']
    lines += ['', '## ③ 예정사항','',
        '- ao 비율 비교와 ar 분별 도착 비교를 마친 뒤 323-d 학습 진단·as 구조 비교 순서다.',
        '- 외부 운반·대기열(i/f) 근거는 계속 대기이며 현장 성능 주장을 하지 않는다.',
        '', '재현: [전체 집계](summary.json), [실행 설정](campaign.json).',
        '', '입력 그림: [하루 모양·작업 비율·분별 강도](../yr332_synthetic_inputs/synthetic-variation.png).',
        '실패한 모형과 매개변수 근거: [결정 기록](../../../.claude/docs/strategy-history/2026-11-16-YR-332-input-four-step.md).']
    assert len(lines)<=200
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('artifact_hashes','groups','paired')},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('axis',choices=['mix','minutes']);inputs(p.parse_args().axis)
