"""Show a preregistered block/time window; never select a flattering example."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=Path('outputs/reports/yr332_resched_ppo')


def main():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    cases=[('SF_SPT',0,'짧은 실작업 우선'),('old200',9917101,'기존 비교학습·200회'),
           ('mlp_final',9951101,'MLP + PPO'),('transformer_final',9951101,'Transformer + PPO')]
    flows=['GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD']
    labels=['반입','반출','양하','적하'];colors=['#206fbe','#e68628','#2b936f','#9a61b1']
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True,layout='constrained')
    example=[]
    evaluations=[]
    for p in sorted((OUT/'evaluation').glob('*.json.gz')):
        if 'repeat' not in p.name:
            evaluations.append((p,json.loads(gzip.decompress(p.read_bytes()))))
    for ax,(arm,seed,label) in zip(axes,cases):
        path,data=next((p,d) for p,d in evaluations if d['case']['arm']==arm and
                       d['case']['model_seed']==seed and d['case']['load']==3500)
        rows=json.loads(gzip.decompress((OUT/'flow-orders'/path.name).read_bytes()))
        selected=sorted([r for r in rows if r['block']=='Y01' and r['service_start_s'] is not None and
            r['service_start_s']<39600 and (r['yard_done_s'] is None or r['yard_done_s']>32400)],
            key=lambda r:(r['service_start_s'],r['job_id']))
        for r in selected:
            index=flows.index(r['flow'])
            start=max(32400,r['service_start_s'])/3600
            finish=min(39600,r['yard_done_s'] or 39600)/3600
            ax.barh(index,finish-start,left=start,height=.5,color=colors[index],alpha=.8,
                    edgecolor='white',linewidth=.6)
            ax.plot(start,index,marker='|',color='#222222',markersize=8)
        ax.set_yticks(range(4),labels)
        ax.invert_yaxis()
        ax.set_title(f'{label}  ·  구간에 걸친 실작업 {len(selected)}건',loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.22);ax.set_axisbelow(True)
        example.append(dict(arm=arm,model_seed=seed,evaluation=path.name,rows=selected))
    axes[-1].set_xticks([9,9.5,10,10.5,11],['09:00','09:30','10:00','10:30','11:00'])
    axes[-1].set_xlim(9,11)
    axes[-1].set_xlabel('합성 하루의 시각  |  막대: 작업 시작→완료 (이동·대기·크레인별 구분은 미표시)')
    fig.suptitle('같은 입력에서 달라진 작업 순서\nY01 블록 · 하루 3,500건 · 사전에 고정한 09:00–11:00',fontsize=15)
    fig.savefig(OUT/'schedule-window.png',dpi=160)
    fig.savefig(OUT/'schedule-window.svg')
    svg=OUT/'schedule-window.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text('utf-8').splitlines())+'\n',
                   encoding='utf-8',newline='\n')
    plt.close(fig)
    (OUT/'schedule-window.json').write_text(json.dumps(dict(block='Y01',start_s=32400,end_s=39600,
        selection='fixed before evaluation, not a selected improvement example',
        caution='rows are physical service intervals, not crane assignments; overlapping work may use separate cranes',
        examples=example),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print({x['arm']:len(x['rows']) for x in example})


if __name__=='__main__':main()
