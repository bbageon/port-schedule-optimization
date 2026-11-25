"""Visit accounting for continuous fixed-cargo crane replays."""
from collections import defaultdict
from math import isclose, isfinite


def earliest_block_time(entry):
    """A lower bound for a planned source, with no crane or queue time assumed."""
    plan = entry.get('visit')
    travel = float(entry['travel_s']) if plan is None else sum(plan['travel_s'][:plan['stop'] + 1])
    return float(entry['arrival_s']) + travel


def visit_plans(schedule):
    """Validate the complete itinerary before any events run. Singletons need no plan."""
    groups = defaultdict(dict)
    for e in schedule:
        link = e.get('visit')
        if link is not None:
            groups[link['id']][e['job_id']] = e
    plans = {}
    for vid, orders in groups.items():
        first = next(iter(orders.values()))
        gate, plan = first['arrival_s'], first['visit']
        stops, travel = plan['stops'], plan['travel_s']
        if (len(orders) < 2 or not stops or len(travel) != len(stops) or travel[0] <= 0
                or any(not isfinite(s) or s < 0 for s in [*travel, plan['exit_travel_s']])):
            raise ValueError(f'{vid}: invalid stop count or travel time')
        seen = set()
        for k, (bid, flow, jobs) in enumerate(stops):
            if not jobs:
                raise ValueError(f'{vid}: empty stop')
            for jid in jobs:
                e = orders.get(jid)
                if (jid in seen or e is None or e['arrival_s'] != gate or e['block'] != bid
                        or e['flow'] != flow or e['visit'] != {**plan, 'stop': k}):
                    raise ValueError(f'{vid}: inconsistent order, gate time or itinerary')
                seen.add(jid)
        if seen != orders.keys():
            raise ValueError(f'{vid}: itinerary omits an order')
        plans[vid] = plan
    return plans


def truck_visits(plans, order_rows, times, end):
    """Use the common gate and last service completion, also for visits spanning midnight."""
    from ..reward.krw import truck_wait_krw

    by_job = {r['job']: r for r in order_rows}
    rows, linked = [], set()
    for vid, plan in plans.items():
        first = by_job[plan['stops'][0][2][0]]
        gate = first['gate']
        previous = gate
        count = 0
        for k, (_, _, jobs) in enumerate(plan['stops']):
            finishes = []
            for jid in jobs:
                rec = times[jid]
                expected = None if previous is None else previous + plan['travel_s'][k]
                if rec.block_arrival is None:
                    if expected is not None and expected <= end:
                        raise ValueError(f'{vid}: missed a block arrival')
                elif expected is None or not isclose(rec.block_arrival, expected, rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f'{vid}: block arrival differs from preceding completion plus travel')
                finishes.append(rec.job_done)
                linked.add(jid)
            count += len(jobs)
            previous = max(finishes) if None not in finishes else None
        finish = previous
        if finish is not None:
            exit_time = finish + plan['exit_travel_s']
            if any(times[j].gate_out is None or not isclose(times[j].gate_out, exit_time,
                   rel_tol=0, abs_tol=1e-6) for _, _, js in plan['stops'] for j in js):
                raise ValueError(f'{vid}: completed orders do not share one exit')
        seconds = (end if finish is None else finish) - gate
        rows.append(dict(visit=vid, orders=count, day=first['day'], gate=gate, finish=finish,
                         seconds=seconds, cost=truck_wait_krw(seconds)))
    for r in order_rows:
        if r['job'] not in linked:
            rows.append(dict(visit=r['job'], orders=1, **{k: r[k] for k in ('day','gate','finish','seconds','cost')}))
    return rows


def stop_intervals(plans, rows, blocks):
    """Assign each second of a completed visit to its current destination block."""
    intervals = defaultdict(list)
    linked = set()
    by_job = {r['job']: r for r in rows} if plans else {}
    for plan in plans.values():
        gate = by_job[plan['stops'][0][2][0]]['gate']
        start = gate
        for bid, _, jobs in plan['stops']:
            finish = max(by_job[j]['finish'] for j in jobs)
            intervals[bid].append((gate, start, finish))
            start = finish
            linked.update(jobs)
    for row in rows:
        if row['job'] not in linked:
            intervals[blocks[row['job']]].append((row['gate'], row['gate'], row['finish']))
    return intervals
