"""Clock-based input accounting for the fixed-cargo monthly runtime."""
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import math

from ..information_contract import announcement_time
from ..world.domain.models import Container
from ..world.domain.validators import ValidationError, validate_scenario
from ..world.integrated.profiles import build_h21_profile
from ..world.integrated.terminal_stream import DIURNAL_DRAIN_S
from .container_contract import ContainerContractError
from .month import DAY_S


def _check_initial_slots(inventory, profile):
    for block, rows in inventory.items():
        boxes, sizes = {}, {}
        for row in rows:
            box = Container(**row)
            if any(type(i) is not int for i in (box.bay, box.row, box.tier)):
                raise ContainerContractError(f'{block}/{box.container_id}: noninteger slot')
            if sizes.setdefault((box.bay, box.row), box.size) != box.size:
                raise ContainerContractError(f'{block}: mixed sizes in initial stack')
            boxes[box.container_id] = box
        try:
            validate_scenario([], boxes, profile)
        except ValidationError as exc:
            raise ContainerContractError(f'{block}: {exc}') from exc


def month_supply_audit(document, by_id, demands):
    """Net at the requested time vs. stock if each exit waits for its own source.

    Equal-time events are simultaneous. Neither projection models service times,
    stack accessibility or completion. Space pressure can queue valid work.
    """
    profile = build_h21_profile()  # run_month restores fixed seeds into this geometry.
    geom = profile.block
    capacity = geom.bay_count * geom.row_count * geom.tier_max
    inventory = document['initial_inventory']
    _check_initial_slots(inventory, profile)
    days = document['days']
    if not days or any(d['index'] != i or d['t0'] != i * DAY_S
                       for i, d in enumerate(days)):
        raise ContainerContractError('Invalid continuous monthly calendar')
    end = len(days) * DAY_S + DIURNAL_DRAIN_S
    start = {d['index']: d['t0'] for d in days}
    for day, rows in document['vessels_by_day'].items():
        if int(day) not in start:
            raise ContainerContractError(f'{day}: no injection day for vessel plan')
        for row in rows:
            if not start[int(day)] <= row['start_s'] <= end:
                raise ContainerContractError(f"{row['key']}: vessel start outside injection window")
    for entry in document['schedule']:
        arrival, travel, lead = (float(entry[k]) for k in ('arrival_s', 'travel_s', 'lead_s'))
        if (not all(math.isfinite(v) and v >= 0 for v in (arrival, travel, lead))
                or announcement_time(entry) > arrival + 1e-6 or arrival + travel > end):
            raise ContainerContractError(f"{entry['job_id']}: invalid truck notice/arrival/travel")

    events = defaultdict(list)
    for s in by_id.values():
        if s['source_kind'] != 'INITIAL':
            events[s['block']].append((s['planned_source_s'], 1, 1))
    for d in demands:
        entry, move = d['entry'], d['move']
        target = entry['target'] if move is None else entry['targets'][move]
        demand_at = d['planned_demand_s']
        source_at = by_id[target]['planned_source_s']
        block_events = events[d['block']]
        if source_at <= demand_at:
            block_events.append((demand_at, -1, -1))
        else:
            block_events.append((demand_at, -1, 0))
            block_events.append((source_at, 0, -1))

    daily = []
    for block, boxes in sorted(inventory.items()):
        net = stock = len(boxes)
        groups = groupby(sorted(events[block]), itemgetter(0))
        group = next(groups, None)
        for day in days:
            low, high = net, stock
            while group is not None and group[0] < day['t0'] + DAY_S:
                for _, net_delta, stock_delta in group[1]:
                    net += net_delta
                    stock += stock_delta
                low, high = min(low, net), max(high, stock)
                group = next(groups, None)
            daily.append(dict(day=day['index'], block=block, net_at_close=net,
                              stock_at_close=stock, waiting_at_close=stock-net,
                              min_net=low, peak_stock=high))
    return dict(capacity_per_block=capacity, days=daily,
                space_pressure_block_days=sum(row['peak_stock'] > capacity for row in daily))
