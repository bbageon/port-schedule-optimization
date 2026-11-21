"""YR-332-bf: fixed-seed accounting diagnosis, without policy simulation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.stage.month import (MONTH_FILL_RATIO, build_month, initial_stock_by_block,
                                  plan_month, plan_month_vessels, truck_net_by_block)
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.world.integrated.yard_layout import terminal_layout

SEED = 9_400_777
OUT = Path('outputs/reports/yr332_month_inventory')
PREREG = '.claude/docs/dashboard-task-specs/YR-332-bf-month-inventory-diagnosis.md'


def ledger(initial, trucks, vessels, days, capacity):
    counts = {key: Counter(value) for key, value in trucks.items()}
    for d, rows in vessels.items():
        for r in rows:
            counts.setdefault((d, r['block']), Counter())[r['work']] += r['moves']
    running, rows, totals = dict(initial), [], []
    for d in days:
        for block in sorted(initial):
            flow = {f: counts.get((d.index, block), {}).get(f, 0)
                    for f in ('GATE_IN', 'GATE_OUT', 'DISCHARGE', 'LOAD')}
            start = running[block]
            end = start + flow['GATE_IN'] + flow['DISCHARGE'] - flow['GATE_OUT'] - flow['LOAD']
            # Independent prefix sum, not the running-balance update above.
            check = initial[block] + sum(
                v.get('GATE_IN', 0) + v.get('DISCHARGE', 0) - v.get('GATE_OUT', 0) - v.get('LOAD', 0)
                for (day, b), v in counts.items() if b == block and day <= d.index)
            assert check == end
            running[block] = end
            rows.append(dict(day=d.index, block=block, initial_sod=start, **flow, planned_eod=end))
        totals.append(sum(running.values()))
    summary = dict(initial_total=sum(initial.values()), first_day_total=totals[0],
        minimum_total=min(totals), minimum_day=totals.index(min(totals)), last_day_total=totals[-1],
        minimum_block=min(r['planned_eod'] for r in rows), maximum_block=max(r['planned_eod'] for r in rows),
        negative_block_days=sum(r['planned_eod'] < 0 for r in rows),
        above_capacity_block_days=sum(r['planned_eod'] > capacity for r in rows),
        original_90pct_guard=max(r['planned_eod'] for r in rows) < capacity*.9,
        original_minimum_50pct_guard=min(totals) > .5*totals[0],
        original_last_day_50pct_guard=totals[-1] > .5*totals[0],
        first_day_retention=min(totals)/totals[0],
        flow_totals={f: sum(r[f] for r in rows) for f in ('GATE_IN','GATE_OUT','DISCHARGE','LOAD')})
    return dict(initial=initial, daily_totals=totals, rows=rows, summary=summary)


def main():
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    profile, layout = build_h21_profile(), terminal_layout()
    days = plan_month(SEED)
    built = build_month(SEED, days=days)
    original = digest(built)
    repeated = build_month(SEED, days=days)
    assert digest(repeated) == original
    g = profile.block
    capacity = g.bay_count*g.row_count*g.tier_max
    nominal = {b: int(capacity*MONTH_FILL_RATIO) for b in layout.ids}
    actual = initial_stock_by_block(built)
    assert set(actual) == set(nominal)
    trucks = defaultdict(Counter)
    for e in built['schedule']:
        assert e['flow'] in ('GATE_IN','GATE_OUT')
        trucks[e['day'],e['block']][e['flow']] += 1
    net = truck_net_by_block(built['schedule'])
    for (d,b), flows in trucks.items():
        assert net[d][b] == flows['GATE_OUT'] - flows['GATE_IN']
    base = plan_month_vessels(days, layout, truck_net=net)
    guarded = plan_month_vessels(days, layout, truck_net=net, initial_stock=actual)
    assert base == plan_month_vessels(days, layout, truck_net=net)
    variants = {
        'original_test': ledger(nominal,trucks,base,days,capacity),
        'actual_initial_same_plan': ledger(actual,trucks,base,days,capacity),
        'actual_initial_guarded_plan': ledger(actual,trucks,guarded,days,capacity),
    }
    changed = []
    for d in base:
        assert len(base[d]) == len(guarded[d])
        for a,b in zip(base[d],guarded[d]):
            assert a['key'] == b['key'] and a['moves'] == b['moves']
            if a != b:
                changed.append(dict(day=d,key=a['key'],block=a['block'],moves=a['moves'],
                                    before=a['work'],after=b['work']))
    stamp = repro_stamp(experiment='YR-332-bf-month-inventory',seeds={'month':[SEED]},
        params=dict(days=[d.as_dict() for d in days],fill_ratio=MONTH_FILL_RATIO,
                    capacity_per_block=capacity,policy_run=False,training=False,
                    initial_variants=list(variants)),prereg=PREREG)
    result = dict(task='YR-332-bf',repro=stamp,input_sha256=original,repeat_identical=True,
        schedule_sha256=digest(built['schedule']),orders=len(built['schedule']),
        capacity_per_block=capacity,nominal_fill_ratio=MONTH_FILL_RATIO,
        generated_initial_by_block=actual,initial_stock_difference=sum(actual.values())-sum(nominal.values()),
        days=[d.as_dict() for d in days],variants=variants,
        base_vessels=base,guarded_vessels=guarded,changed_streams=changed,
        sources={str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__),Path(PREREG),*Path('src/yard_rl/v7').rglob('*.py')]},
        caveat='Plan-day mass balance; not time-ordered physical inventory or proof of feasibility.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(orders=result['orders'],capacity=capacity,
        initial_difference=result['initial_stock_difference'],changed_streams=len(changed),
        summaries={k:v['summary'] for k,v in variants.items()}),ensure_ascii=False))


if __name__ == '__main__':
    main()
