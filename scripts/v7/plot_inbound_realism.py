"""Standalone figure for the saved-trajectory decomposition; no model execution."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from inbound_realism_common import SOURCE, OUT, read


def main():
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    summary=read(SOURCE/'summary.json');sim=read(OUT/'simulation.json')
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.0))
    fig.subplots_adjust(bottom=.27,top=.84,wspace=.36)
    for arm,title,color in [('initial','추가 학습 전','#64748b'),('zero','전역 정보 없음','#e08214'),('global','전역 정보 있음','#137c8b')]:
        values=[summary['by_load'][str(l)]['groups'][arm]['flows']['GATE_IN']['mean_min'] for l in (3500,7500,15000)]
        axes[0].plot(range(3),values,'o-',label=title,color=color,lw=2)
        axes[0].annotate(f'{values[2]:.1f}',(2,values[2]),xytext=(0,-16 if arm=='initial' else 9),textcoords='offset points',ha='center',fontsize=9,color=color)
    axes[0].set(xticks=range(3),xticklabels=['3,500건','7,500건','15,000건'],ylabel='반입 진입→완료 관측시간 (분)',title='큰 차이는 고물량 대조군에서 발생')
    axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.2);axes[0].set_ylim(0,90)
    rows=[next(r for r in sim['aggregates'] if r['arm']==a and r['flow']=='GATE_IN' and r['load']==15000)['at24']['means'] for a in ('zero','global')]
    left=np.zeros(2)
    for k,title,color in [('gate_to_block','진입→블록','#b7c6d9'),('block_to_start','블록에서 작업 시작 대기','#e08214'),('start_to_done','작업 중 관측시간','#137c8b')]:
        values=np.array([r[k] for r in rows]);axes[1].barh([1,0],values,left=left,label=title,color=color)
        if k=='block_to_start':
            for y,x,v in zip([1,0],left,values):axes[1].text(x+v/2,y,f'{v:.1f}분',ha='center',va='center',color='white',fontweight='bold')
        left+=values
    for y,v in zip([1,0],left):axes[1].text(v+1,y,f'총 {v:.1f}분',va='center',fontsize=10)
    axes[1].set(yticks=[1,0],yticklabels=['전역 정보 없음','전역 정보 있음'],xlabel='반입 오더당 관측시간 (분)',title='고물량 반입 차이의 대부분은 대기시간',xlim=(0,98))
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.21),frameon=False,fontsize=9)
    fig.suptitle('반입 시간 감소 원인 검산 — 같은 입력·같은 추가 학습량의 비교',fontsize=15)
    fig.text(.03,.025,'합성 최종시험 27쌍. 미완료 오더는 24시간까지 각 구간에 누적. 실제 항만 개선율이 아님.',fontsize=10,color='#475569')
    fig.savefig(OUT/'inbound-comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
