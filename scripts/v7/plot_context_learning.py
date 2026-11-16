"""Show every initialization; ranges are not confidence intervals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('outputs/reports/yr332_context_learning')
    s=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    names={'initial':'학습 전 MLP','point_current':'현재 자료 · 50회',
           'point_replay':'누적 자료 · 50회','point_half':'절반 자료 · 50회',
           'point_steps':'누적 자료 · 200회','mean_context':'후보 평균 맥락망',
           'star_attention':'자원 중심 Attention','SF_SPT':'짧은 실작업 우선',
           'FIFO':'작업 대기 진입 순서','LWKR':'남은 작업량 우선'}
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,ax=plt.subplots(figsize=(12,7.5),layout='constrained')
    ref=next(g['flows']['TRUCK']['mean_min'] for g in s['groups'] if g['arm']=='SF_SPT')
    for i,g in enumerate(s['groups']):
        value=g['flows']['TRUCK']['mean_min']
        color='#377eb8' if g['arm'] not in ('SF_SPT','FIFO','LWKR') else '#75847c'
        ax.barh(i,value,color=color,alpha=.45,height=.6)
        dots=[r['flows']['TRUCK']['mean_min'] for r in s['by_seed'] if r['arm']==g['arm']]
        if dots:
            ax.plot([min(dots),max(dots)],[i,i],color='#1b2735',lw=1.2)
            ax.scatter(dots,[i]*len(dots),color='#1b2735',s=22,zorder=3)
        ax.text(max([value,*dots])+2,i,f'{value:.1f}분',va='center',fontsize=10)
    ax.axvline(ref,color='#b95435',ls='--',lw=1.2,label='짧은 실작업 우선 규칙')
    ax.set_yticks(range(len(s['groups'])),[names[g['arm']] for g in s['groups']])
    ax.invert_yaxis();ax.set_xlim(left=0,right=ax.get_xlim()[1]*1.12)
    ax.set_xlabel('게이트 진입 → 오더 작업 완료 평균 (분) · 짧을수록 좋음')
    ax.set_title('같은 합성 입력에서 학습 방법 비교',fontsize=17,pad=20)
    ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',frameon=False)
    fig.supxlabel('막대: 오더 수 가중 평균 · 점: 학습 초기값 3개(신뢰구간 아님)\n평가: 물량별 합성 하루 1개씩 · 24시간 미완료 경과시간 포함 · 현장 성능 확증 아님',fontsize=10)
    fig.savefig(root/'learning-comparison.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
