"""Standalone research figures; uncertainty bands are the range of three starts."""
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path('outputs/reports/yr332_global_value_context')


def read(p):return json.loads(p.read_text('utf-8'))


def main():
    font_path=Path('C:/Windows/Fonts/malgun.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    summary=read(OUT/'summary.json');p=summary['prediction']
    arms=('zero','global');colors={'zero':'#667085','global':'#176B83'}
    labels={'zero':'전역 대기 정보 없음','global':'전역 대기 정보 있음'}
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for arm in arms:
        rows=[read(OUT/'prediction'/f'{arm}-{seed}'/'result.json') for seed in (9951101,9951102,9951103)]
        epochs=[c['epoch'] for c in rows[0]['curves']]
        matrix=[[r['curves'][j]['validation']['rmse'] for r in rows] for j in range(len(epochs))]
        axes[0].plot(epochs,[st.mean(x*x for x in a)**.5 for a in matrix],marker='o',color=colors[arm],label=labels[arm])
        axes[0].fill_between(epochs,[min(a) for a in matrix],[max(a) for a in matrix],color=colors[arm],alpha=.13)
    axes[0].set(xlabel='비용 예측망 학습 반복',ylabel='새 날짜의 예측 오차 (낮을수록 좋음)',title='선택망을 고정한 예측 학습')
    axes[0].legend(fontsize=10);axes[0].grid(axis='y',alpha=.15)
    for i,arm in enumerate(arms):
        values=[p['groups'][arm]['by_load'][str(load)] for load in (3500,7500,15000)]
        x=[j+(-.19 if i==0 else .19) for j in range(3)]
        axes[1].bar(x,values,width=.36,color=colors[arm],label=labels[arm])
    axes[1].set_xticks(range(3),['3,500건','7,500건','15,000건'])
    axes[1].set(ylabel='예측 오차',title='200회 후 물량별 검증 결과')
    axes[1].grid(axis='y',alpha=.15)
    fig.suptitle(f'전역 대기 한 칸 추가: 전체 예측 오차 {p["rmse_change_pct"]:+.2f}%',fontsize=15)
    fig.supxlabel('공통 경험·동일 학습 예산 | 음영: 시작 모델 3개의 범위 | 예측 개선은 운영 성능 개선과 다름',fontsize=10)
    for ext in ('png','svg'):fig.savefig(OUT/f'prediction-comparison.{ext}',dpi=180)
    plt.close(fig)
    if summary['policy']:
        q=summary['policy'];flows=['TRUCK','VESSEL','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD']
        values=[q['paired_change']['flow_pct'][f] for f in flows]
        fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
        bars=ax.bar(['트럭 평균','본선 평균','반입','반출','양하','적하'],values,
                    color=['#176B83' if v<=0 else '#B54737' for v in values])
        ax.axhline(0,color='#667085',linewidth=.8)
        ax.bar_label(bars,labels=[f'{v:+.2f}%' for v in values],padding=4)
        ax.margins(y=.25);ax.set(ylabel='기존 방식 대비 시간 변화 (%)',
            title='전역 대기 정보의 정책 학습 효과 — 트럭과 본선을 함께 비교')
        fig.supxlabel('−: 시간 감소 / +: 시간 증가 | 동일한 새 합성 시험 입력 | 미완료의 관측 경과시간 포함',fontsize=10)
        for ext in ('png','svg'):fig.savefig(OUT/f'policy-comparison.{ext}',dpi=180)
        plt.close(fig)


if __name__=='__main__':main()
