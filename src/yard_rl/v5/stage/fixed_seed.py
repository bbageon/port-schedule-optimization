"""fixed-cargo-v2: retain arrivals/workloads, assign each exit a real planned source.

This builds a DATASET, not an execution trace. No physical completion, location
after a policy action, or operational realism is inferred from planned sources.
The legacy generator and PPO entrypoint do not automatically consume this file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import random
from types import SimpleNamespace

from .container_contract import ContainerContractError, audit_container_plan
from .orders import orders_from_schedule
from .seed_inventory import enumerate_inventory

SCHEMA = 'yard_rl.v5.fixed_cargo_seed.v2'
RULES = {
    'name': 'fixed-cargo-v2', 'assignment_order': 'planned demand time, then job key',
    'selection': 'uniform unassigned planned sources in block; otherwise earliest future source',
    'rng': 'v5:fixed-cargo-v2:<seed>:<block>', 'max_exits_per_source': 1,
    'source_time': 'planned lower bound, not actual readiness',
    'missing_supply': 'error; no new cargo, dropped jobs, flow changes or appointment shifts',
    'cargo_classes': 'pooled block inventory; import/export/transshipment classes not calibrated',
    'runtime_requirements': ['wait for actual ready cargo', 'follow actual relocated container block'],
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def initial_inventory(built):
    result = {}
    for bid, scn in sorted(built['day0']['scenarios'].items()):
        result[bid] = []
        for cid, box in sorted(scn.containers.items()):
            if box.container_id != cid:
                raise ContainerContractError('Initial dictionary key differs from container identity')
            row = asdict(box)
            # Cloned scenarios retain the prototype's embedded block label.
            # The owning scenario is authoritative; do not move or mutate cargo.
            row['block'] = bid
            row['special_flags'] = sorted(row['special_flags'])
            result[bid].append(row)
    return result


def as_audit_input(document):
    scenarios = {bid: SimpleNamespace(containers={
        box['container_id']: SimpleNamespace(container_id=box['container_id']) for box in boxes})
        for bid, boxes in document['initial_inventory'].items()}
    return dict(day0=dict(scenarios=scenarios), schedule=document['schedule'])


def fixed_seed_audit(document):
    if document.get('schema') != SCHEMA or document.get('rules') != RULES:
        raise ContainerContractError('Unsupported seed schema or generation rules')
    sources, demands = enumerate_inventory(document['initial_inventory'], document['schedule'],
                                           document['vessels_by_day'], document['seed'])
    if sources != document['sources']:
        raise ContainerContractError('Source registry disagrees with initial/incoming cargo')
    orders, records = orders_from_schedule(as_audit_input(document))
    if [asdict(o) for o in orders.values()] != document['orders']:
        raise ContainerContractError('Public six-field orders disagree with the physical plan')
    contract = audit_container_plan(as_audit_input(document), document['vessels_by_day'])
    if not contract['passed']:
        return dict(schema=SCHEMA, passed=False, contract=contract, ready_for_training=False)
    by_id = {s['container']: s for s in sources}
    matrix, gaps, waiting, invalid = Counter(), [], [], []
    for d in demands:
        e, m = d['entry'], d['move']
        cid = e.get('target') if m is None else (e.get('targets') or [])[m]
        source = by_id.get(cid)
        if source is None:
            continue  # The contract reports the missing source.
        if m is None and e['size_ft40'] != (source['size'] == 'FT40'):
            invalid.append(dict(job=d['job'], reason='outbound size disagrees with target'))
        if m is None and 'size_class' in e and list(e['size_class']) != [source['size'][2:], 'GP', None]:
            invalid.append(dict(job=d['job'], reason='outbound legacy size class disagrees with target'))
        matrix[f"{source['source_kind']}->{d['kind']}"] += 1
        gap = d['planned_demand_s'] - source['planned_source_s']
        gaps.append(gap)
        if gap < 0:
            waiting.append(dict(job=d['job'], container=cid, planned_wait_lower_bound_s=-gap))
    gaps.sort()
    if invalid:
        contract['passed'] = False
        contract['violations']['outbound_size_mismatch'] = len(invalid)
    return dict(schema=SCHEMA, passed=contract['passed'], contract=contract,
                truck_orders=len(orders), sources=len(sources), exits=len(demands),
                ending_unassigned_sources=len(sources) - len(demands),
                source_exit_counts=dict(sorted(matrix.items())),
                source_size_counts=dict(sorted(Counter(s['size'] for s in sources).items())),
                planned_waiting_exits=len(waiting), waiting_examples=waiting[:8],
                max_planned_wait_lower_bound_s=max((w['planned_wait_lower_bound_s'] for w in waiting), default=0),
                planned_source_to_demand_s={
                    'minimum': min(gaps, default=0), 'median': gaps[len(gaps)//2] if gaps else 0,
                    'maximum': max(gaps, default=0), 'meaning': 'planned time difference, NOT actual yard dwell'},
                empty_execution_records=all(r.gate_in_s is None for r in records.values()),
                physical_execution_verified=False, ready_for_training=False,
                scope='fixed-identity input validity only; no policy performance or operational realism claim')


def build_fixed_seed(built, vessels_by_day, *, seed):
    """Input objects remain unchanged. Target assignment never inspects a live world."""
    initial = initial_inventory(built)
    schedule, vessels = copy.deepcopy(built['schedule']), copy.deepcopy(vessels_by_day)
    sources, demands = enumerate_inventory(initial, schedule, vessels, seed)
    by_block, wanted = defaultdict(list), Counter(d['block'] for d in demands)
    for s in sources:
        by_block[s['block']].append(s)
    shortages = {b: dict(sources=len(by_block[b]), exits=n, missing=n-len(by_block[b]))
                 for b, n in wanted.items() if n > len(by_block[b])}
    if shortages:
        raise ContainerContractError('Insufficient real planned supply; generation refused', report=shortages)
    for rows in vessels.values():
        for row in rows:
            if row['work'] == 'LOAD':
                row['targets'] = [None] * row['moves']
    pools, pointers = defaultdict(list), Counter()
    rngs = {b: random.Random(f'v5:fixed-cargo-v2:{seed}:{b}') for b in initial}
    for d in demands:
        b, at = d['block'], d['planned_demand_s']
        future, pool = by_block[b], pools[b]
        p = pointers[b]
        while p < len(future) and future[p]['planned_source_s'] <= at:
            pool.append(future[p])
            p += 1
        if pool:
            index = rngs[b].randrange(len(pool))
            source = pool[index]
            pool[index] = pool[-1]
            pool.pop()
        else:
            source = future[p]
            p += 1  # Reserve the future source exactly once; never create a replacement.
        pointers[b] = p
        if d['move'] is None:
            d['entry'].update(target=source['container'], con_no=source['container'],
                              size_ft40=source['size'] == 'FT40')
            if 'size_class' in d['entry']:
                d['entry']['size_class'] = [source['size'][2:], 'GP', None]
        else:
            d['entry']['targets'][d['move']] = source['container']
    orders, _ = orders_from_schedule(dict(schedule=schedule))
    document = dict(schema=SCHEMA, seed=seed, rules=copy.deepcopy(RULES),
                    days=[d.as_dict() for d in built['days']],
                    month_end_s=built['month_end_s'], lead_mode=built['lead_mode'],
                    initial_inventory=initial, schedule=schedule,
                    vessels_by_day={str(d): rows for d, rows in sorted(vessels.items())},
                    sources=sources, orders=[asdict(o) for o in orders.values()])
    # JSON arrays have one representation; do not leak tuples into the bundle API.
    document = json.loads(canonical_bytes(document))
    audit = fixed_seed_audit(document)
    if not audit['passed']:
        raise ContainerContractError('Generated input failed its independent audit', report=audit)
    return document, audit
