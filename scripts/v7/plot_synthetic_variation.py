"""Plot frozen development input variation, separately from policy outcomes."""
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from yard_rl.v7.correlated_arrivals import weights


def main():
    root=Path('outputs/reports/yr332_synthetic_inputs')
    bank=json.loads((root/'bank.json').read_text(encoding='utf-8'))
    parameters=json.loads((root/'multiscale-minute-calibration.json').read_text(encoding='utf-8'))['parameters']
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
    hours=np.arange(24)
    for row in bank['splits']['train']:
        axes[0].plot(hours,np.array(row['hour_masses'])*100,color='#377eb8',alpha=.2,lw=1)
    axes[0].plot(hours,np.array(bank['fixed_hour_masses'])*100,color='#1b2735',lw=2,label='기존 고정 하루 곡선')
    axes[0].plot([],[],color='#377eb8',label='서로 다른 합성 하루 24개')
    axes[0].set(title='① 하루 모양: 아침·낮·저녁 물량의 비중이 날짜마다 변함',ylabel='하루 오더 중 비중 (%)',xlabel='시각 (시)',xlim=(0,23),xticks=np.arange(0,24,2))
    axes[0].legend(loc='upper left',frameon=False,ncol=2)
    shares=np.array([r['gate_out_share'] for r in bank['splits']['train']])*100
    axes[1].scatter(np.arange(1,25),shares,color='#d36a27',s=35)
    axes[1].axhline(bank['fixed_gate_out_share']*100,color='#1b2735',ls='--',label='대조 고정 비율 54.44%')
    axes[1].set(title='② 작업 비율: 하루 곡선과 같은 부모·혼합비로 반출 비율도 변함',ylabel='반출 비율 (%)',xlabel='학습용 합성 하루 번호',xlim=(.5,24.5),xticks=np.arange(1,25))
    axes[1].legend(loc='upper left',frameon=False)
    for i,color in enumerate(('#377eb8','#d36a27','#409366')):
        w=weights(random.Random(9949000+i),parameters)
        axes[2].plot(np.arange(60),np.array(w)*120,color=color,alpha=.8,lw=1,label=f'난수 예시 {i+1}')
    axes[2].axhline(2,color='#1b2735',ls='--',lw=1)
    axes[2].set(title='③ 시간대 안 몰림: 같은 시간당 120방문이어도 분별 강도는 달라짐',ylabel='분당 기대 방문 수',xlabel='시간대 안의 분',xlim=(0,59))
    axes[2].legend(loc='upper right',frameon=False,ncol=3)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('개발 자료의 집계 통계로 보정한 합성 입력',fontsize=17)
    fig.supxlabel('①② 학습용 생성 설정 · ③ 생성 수식의 예시 강도이며 실제 방문 건수나 운영 평가 결과가 아님',fontsize=10)
    fig.savefig(root/'synthetic-variation.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
