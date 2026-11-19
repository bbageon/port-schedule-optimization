"""Standalone scientific figure; validation curves do not choose the final model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path

OUT=Path('outputs/reports/yr332_global_policy')
ARMS=('zero','global')


def main():
    s=json.loads((OUT/'summary.json').read_text('utf-8'));fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    colors={'zero':'#4878a8','global':'#d46b35'}
    for arm in ARMS:
        rows=[r for r in s['development_curve'] if r['arm']==arm]
        axes[0].plot([0,2,4,6],[r['cost']/1e8 for r in rows],marker='o',label=arm,color=colors[arm])
        axes[1].plot([0,2,4,6],[r['flows']['TRUCK']['mean_min'] for r in rows],marker='o',label=arm,color=colors[arm])
    axes[0].set(title='Development: terminal cost',ylabel='Model cost / 100 million',xlabel='PPO update')
    axes[1].set(title='Development: truck completion',ylabel='Minutes (24 h observation)',xlabel='PPO update')
    names=['Truck','Vessel jobs','Inbound','Outbound','Discharge','Loading']
    vals=[s['paired_change']['flow_pct'][f] for f in ('TRUCK','VESSEL','GATE_IN','GATE_OUT','VESSEL_DISCHARGE','VESSEL_LOAD')]
    axes[2].barh(names,vals,color=['#d46b35' if v>0 else '#27806b' for v in vals])
    axes[2].invert_yaxis();axes[2].axvline(0,color='black',linewidth=.7)
    axes[2].set(title='Final test: global vs zero',xlabel='Time change (%) / lower is better')
    for i,v in enumerate(vals):axes[2].text(v,i,f' {v:+.2f}%',va='center',ha='left' if v>=0 else 'right',fontsize=9)
    for ax in axes[:2]:ax.legend();ax.set_xticks([0,2,4,6]);ax.grid(alpha=.2)
    axes[2].margins(x=.3)
    fig.suptitle('Fixed final checkpoint; 3 starting models, 3 synthetic test curves, 3 loads',fontsize=12)
    for ext in ('png','svg'):fig.savefig(OUT/f'policy-comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
