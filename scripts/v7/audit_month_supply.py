"""Rebuild the frozen month and independently reconcile clock and plan-day ledgers."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_month_inventory import SEED, OUT as PRIOR
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.information_contract import announcement_time
from yard_rl.v7.stage.fixed_seed import build_fixed_seed, sha
from yard_rl.v7.stage.month import (DAY_S, MONTH_FILL_RATIO, build_month, initial_stock_by_block,
                                   plan_month, plan_month_vessels, truck_net_by_block)
from yard_rl.v7.world.integrated.yard_layout import terminal_layout

OUT = Path('outputs/reports/yr332_month_supply')
SPEC = '.claude/docs/dashboard-task-specs/YR-332-bg-month-supply-contract.md'


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def movements(doc):
    """Read individual moves directly from the serialized workload."""
    supply, exits = {}, {}

    def record(kind, job, block, day, visible, at, cid):
        table = exits if kind in ('GATE_OUT', 'LOAD') else supply
        assert cid not in table
        table[cid] = dict(job=job, kind=kind, block=block, plan_day=day,
                          public_s=visible, event_s=at)

    for block, boxes in doc['initial_inventory'].items():
        for box in boxes:
            record('INITIAL', None, block, -1, 0, 0, box['container_id'])
    for e in doc['schedule']:
        record(e['flow'], e['job_id'], e['block'], e['day'], announcement_time(e),
               e['arrival_s'] + e['travel_s'], e['con_no'])
    for day, streams in doc['vessels_by_day'].items():
        for r in streams:
            for m in range(r['moves']):
                job = f"{r['block']}:J-{r['key']}-{m:04d}"
                cid = 'IN_'+job if r['work'] == 'DISCHARGE' else r['targets'][m]
                record(r['work'], job, r['block'], int(day), doc['days'][int(day)]['t0'],
                       r['start_s'] + m*r['cadence_s'], cid)
    return supply, exits


def verify(doc, audit, previous):
    supply, exits = movements(doc)
    assert exits.keys() <= supply.keys()
    moments = defaultdict(lambda: defaultdict(list))
    plan = Counter()
    waits = []
    spill = defaultdict(Counter)
    for item in (*supply.values(), *exits.values()):
        if item['kind'] != 'INITIAL':
            for boundary_day in range(item['plan_day'], int(item['event_s']//DAY_S)):
                spill[item['block'], boundary_day][item['kind']] += 1
    for cid, s in supply.items():
        block, at = s['block'], s['event_s']
        moments[block]['supply'].append(at)
        plan[block, s['plan_day']] += 1
        if cid in exits:
            e = exits[cid]
            assert block == e['block'] and e['public_s'] <= e['event_s']
            wanted = e['event_s']
            moments[block]['demand'].append(wanted)
            moments[block]['exit'].append(max(wanted, at))
            plan[block, e['plan_day']] -= 1
            if at > wanted:
                waits.append(dict(container=cid, supply=s, demand=e))
    assert len(waits) == audit['planned_waiting_exits']
    prev = {(r['block'],r['day']):r for r in previous['rows']}
    clock = {(r['block'],r['day']):r for r in audit['month_supply']['days']}
    negatives = []
    for block, times in moments.items():
        for values in times.values():
            values.sort()
        src, want, exit_at = (times[k] for k in ('supply','demand','exit'))
        events = sorted(set(src+want+exit_at))
        balance = plan[block,-1]
        initial = len(doc['initial_inventory'][block])
        for d in doc['days']:
            day = d['index']
            begin, end = d['t0'], d['t0']+DAY_S
            balance += plan[block,day]
            assert prev[block,day]['planned_eod'] == balance
            net = bisect_left(src,end) - bisect_left(want,end)
            stock = bisect_left(src,end) - bisect_left(exit_at,end)
            r = clock[block,day]
            assert (r['net_at_close'],r['stock_at_close'],r['waiting_at_close']) == (net,stock,stock-net)
            net0 = initial if day == 0 else clock[block,day-1]['net_at_close']
            stock0 = initial if day == 0 else clock[block,day-1]['stock_at_close']
            ts = events[bisect_left(events,begin):bisect_left(events,end)]
            assert min([net0]+[bisect_right(src,t)-bisect_right(want,t) for t in ts]) == r['min_net']
            assert max([stock0]+[bisect_right(src,t)-bisect_right(exit_at,t) for t in ts]) == r['peak_stock']
            if balance < 0:
                ids = [w['container'] for w in waits if w['supply']['block'] == block
                       and w['demand']['event_s'] < end <= w['supply']['event_s']]
                assert len(ids) == stock-net
                late = spill[block,day]
                assert net-balance == late['LOAD']+late['GATE_OUT']-late['DISCHARGE']-late['GATE_IN']
                negatives.append(dict(**r, plan_day_balance=balance, cutoff_s=end,
                                      boundary_adjustment=net-balance, later_events=dict(late),
                                      waiting_containers=ids))
    assert len(negatives) == previous['summary']['negative_block_days']
    crossed = Counter()
    after_month = Counter()
    for item in (*supply.values(), *exits.values()):
        if item['kind'] != 'INITIAL' and int(item['event_s']//DAY_S) != item['plan_day']:
            crossed[item['kind']] += 1
        if item['event_s'] >= len(doc['days'])*DAY_S:
            after_month[item['kind']] += 1
    save('future-supply.json',waits)
    save('deficit-boundaries.json',negatives)
    return dict(verified_block_days=len(clock), negative_plan_days=len(negatives),
                negative_clock_closes=sum(r['net_at_close']<0 for r in clock.values()),
                any_negative_clock_within_day=sum(r['min_net']<0 for r in clock.values()),
                highest_clock_stock=max(r['peak_stock'] for r in clock.values()),
                waiting_at_any_deficit_close=len(set().union(*(r['waiting_containers'] for r in negatives))),
                moved_across_plan_day=dict(crossed), at_or_after_final_midnight=dict(after_month))


def main():
    baseline = json.loads((OUT/'baseline.json').read_text('utf8'))
    bf = json.loads((PRIOR/'diagnosis.json').read_text('utf8'))
    days = plan_month(SEED)
    built = build_month(SEED, days=days)
    assert digest(built) == baseline['input_sha256'] == bf['input_sha256']
    vessels = plan_month_vessels(days, terminal_layout(), truck_net=truck_net_by_block(built['schedule']),
                                initial_stock=initial_stock_by_block(built))
    document, audit = build_fixed_seed(built, vessels, seed=SEED)
    assert audit['passed'] and sha(document) == baseline['document_sha256']
    assert {k:sha(v) for k,v in document.items()} == baseline['field_hashes']
    previous = bf['variants']['actual_initial_guarded_plan']
    checks = verify(document,audit,previous)
    save('clock-ledger.json',audit['month_supply'])
    stamp = repro_stamp(experiment='YR-332-bg', seeds={'month':[SEED]},
                        params=dict(days=[d.as_dict() for d in days], fill_ratio=MONTH_FILL_RATIO,
                                    lead_mode=built['lead_mode'], baseline_commit=baseline['commit'],
                                    geometry='H21', training=False, policy_run=False), prereg=SPEC)
    assert stamp['code']['git_dirty'] is False
    result = dict(task='YR-332-bg', repro=stamp, checks=checks, fixed_input_unchanged=True,
                  field_hashes=baseline['field_hashes'], document_sha256=baseline['document_sha256'],
                  previous_plan_day_guards=previous['summary'], orders=audit['truck_orders'],
                  exits=audit['exits'], sources=audit['sources'], future_supply=audit['planned_waiting_exits'],
                  max_planned_wait_s=audit['max_planned_wait_lower_bound_s'],
                  space_pressure_block_days=audit['month_supply']['space_pressure_block_days'])
    save('result.json',result)
    print(checks)


if __name__ == '__main__':
    main()
