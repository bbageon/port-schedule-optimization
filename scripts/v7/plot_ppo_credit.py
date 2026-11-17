"""Static scientific summary of the saved-state diagnostic, using only measured JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('outputs/reports/yr332_ppo_credit')
s = json.loads((OUT/'summary.json').read_text('utf-8'))
d = json.loads((OUT/'choice-detail.json').read_text('utf-8'))
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
plt.rcParams['svg.hashsalt'] = 'YR-332-ax'
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
loads = (3500, 7500, 15000)
lookup = {(r['group']['load'], r['group']['time_bin']): r['metrics']
          for r in s['tables']['kind_load_time_bin'] if r['group']['kind'] == 'transformer'}
values = np.array([[lookup[load, t]['positive_advantage_fraction']*100 for t in range(4)] for load in loads])
ax = axes[0, 0]
ax.imshow(values, vmin=0, vmax=100, cmap='Blues', aspect='auto')
for y in range(3):
    for x in range(4):
        ax.text(x, y, f'{values[y, x]:.1f}%', ha='center', va='center',
                color='white' if values[y, x] > 55 else '#152536', fontsize=13)
ax.set(xticks=range(4), xticklabels=['0–6시', '6–12시', '12–18시', '18–24시'],
       yticks=range(3), yticklabels=['3,500건', '7,500건', '15,000건'],
       title='① 선택을 장려하는 양수 학습 신호의 비율')
lookup_load = {r['group']['load']: r['metrics'] for r in s['tables']['kind_load']
               if r['group']['kind'] == 'transformer'}
ax = axes[0, 1]
x = np.arange(3)
for offset, name, color, label in [(-.18, 'pre', '#32556b', '갱신 직전'), (.18, 'post', '#58a5a3', '갱신 직후')]:
    bar = ax.bar(x+offset, [lookup_load[l][name]['rmse'] for l in loads], .34, color=color, label=label)
    ax.bar_label(bar, fmt='%.2f', padding=3)
ax.set(xticks=x, xticklabels=['3,500건', '7,500건', '15,000건'], ylim=(0, 5.3),
       title='② 남은 총비용 예측 오차', ylabel='고정 비용 눈금 단위 · 낮을수록 좋음')
ax.legend(frameon=False)
ax.text(.02, .94, '같은 학습 자료에서의 오차 · 새 날 성능 아님', transform=ax.transAxes, fontsize=9, va='top')
detail = [r for r in d['rows'] if r['kind'] == 'transformer' and r['load'] == 0]
ax = axes[1, 0]
for offset, name, color, label in [(-.18, 'initial', '#32556b', '학습 시작'), (.18, 'final', '#58a5a3', '학습 종료')]:
    bar = ax.bar(x+offset, [r[name]['mixed_single_crane_inbound_probability']*100 for r in detail], .34,
                 color=color, label=label)
    ax.bar_label(bar, fmt='%.1f%%', padding=3)
ax.set(xticks=x, xticklabels=['초기값 1', '초기값 2', '초기값 3'], ylim=(0, 100),
       title='③ 반입·반출이 함께 가능할 때 반입 선택 확률', ylabel='트럭 후보 안의 조건부 확률 (%)')
ax.text(.02, .95, '같은 크레인·같은 상태 비교', transform=ax.transAxes, fontsize=9, va='top')
ax.legend(frameon=False, loc='upper right')
ax = axes[1, 1]
for f, offset, color, label in [(0, -.18, '#58a5a3', '반입'), (1, .18, '#bf7653', '반출')]:
    rates = [100*lookup_load[l]['saturation']['cap_counts'][f]/lookup_load[l]['saturation']['flow_counts'][f] for l in loads]
    bar = ax.bar(x+offset, rates, .34, color=color, label=label)
    ax.bar_label(bar, fmt='%.1f%%', padding=3)
ax.set(xticks=x, xticklabels=['3,500건', '7,500건', '15,000건'], ylim=(0, 75),
       title='④ 대기 입력이 2시간 상한에 붙은 후보', ylabel='결정마다 나타난 후보의 비율 (%)')
ax.legend(frameon=False)
fig.suptitle('Transformer 저장 경험 진단: 비용 예측 · 선택 변화 · 대기 상한', fontsize=17, y=.98)
fig.text(.5, .025, '3개 초기값 · 저장 상태 36,864건 | 새 학습·시뮬레이션 없음 | 관측 관계이며 성능 저하의 인과 증명은 아님',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, .06, 1, .94), h_pad=3, w_pad=3)
fig.savefig(OUT/'diagnosis.png', dpi=170)
fig.savefig(OUT/'diagnosis.svg', metadata={'Date': None})
svg = OUT/'diagnosis.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text('utf-8').splitlines())+'\n',
               encoding='utf-8', newline='\n')
