"""Standalone research figure; no policy or metric selection."""
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('outputs/reports/yr332_value_stability')


def main():
    s=json.loads((ROOT/'summary.json').read_text('utf-8'))
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
                         'svg.hashsalt':'yr332-ay-value-comparison','font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'control':'#64748b','popart':'#087f8c','SF_SPT':'#b46516'}
    labels={'control':'기존 가치 학습','popart':'가치 목표 정규화','SF_SPT':'짧은 실작업 우선'}
    ax=axes[0,0]
    for arm in ('control','popart'):
        rows=s['training']['rows']
        y=[st.mean(r['after_rmse'] for r in rows if r['arm']==arm and r['iteration']==i) for i in range(6)]
        ax.plot(range(1,7),y,marker='o',color=colors[arm],label=labels[arm])
    ax.set(title='학습에 쓴 경험의 비용 예측 오차',xlabel='갱신 횟수 (회차별 물량·날짜가 다름)',ylabel='원래 보상 눈금의 예측 오차')
    ax.legend(frameon=False);ax.grid(alpha=.2)
    ax=axes[0,1];x=np.arange(3)
    for offset,arm in [(-.18,'control'),(.18,'popart')]:
        y=[next(v['rmse'] for v in s['value_validation'] if v['arm']==arm and v['load']==l) for l in (3500,7500,15000)]
        ax.bar(x+offset,y,.34,color=colors[arm],label=labels[arm])
    ax.set(title='새 검증 경험의 비용 예측 오차',xticks=x,xticklabels=['3,500건','7,500건','15,000건'],ylabel='원래 보상 눈금의 예측 오차')
    ax.grid(axis='y',alpha=.2)
    ax=axes[1,0]
    for offset,arm in [(-.24,'control'),(0.,'popart'),(.24,'SF_SPT')]:
        y=[s['by_load'][str(l)]['groups'][arm]['flows']['TRUCK']['mean_min'] for l in (3500,7500,15000)]
        ax.bar(x+offset,y,.23,color=colors[arm],label=labels[arm])
    ax.set(title='최종 시험: 게이트 진입→작업완료',xticks=x,xticklabels=['3,500건','7,500건','15,000건'],ylabel='오더당 관측 평균(분)')
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];flows=('GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD')
    y=[s['paired_change']['flow_pct'][f] for f in flows]
    bars=ax.bar(range(4),y,color=['#087f8c' if v<0 else '#b45309' for v in y])
    ax.bar_label(bars,labels=[f'{v:+.2f}%' for v in y],padding=3)
    ax.axhline(0,color='#475569',linewidth=1)
    ax.set(title='정규화의 작업별 시간 변화',xticks=range(4),xticklabels=['반입','반출','양하','적하'],ylabel='기존 학습 대비 변화(%)')
    ax.margins(y=.22);ax.grid(axis='y',alpha=.2)
    fig.suptitle('가치망 학습만 바꾼 3쌍 비교 · 모델/입력 날짜에 같은 무게',fontsize=15)
    fig.supxlabel('합성 환경 진단 · 음수 행동 신호 유지 · 미완료 작업은 관측 종료까지 포함 · 음수는 시간 감소',fontsize=10)
    fig.savefig(ROOT/'comparison.png',dpi=160)
    fig.savefig(ROOT/'comparison.svg',metadata={'Date':None})
    p=ROOT/'comparison.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text('utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
