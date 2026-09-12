"""Exogenous container sources and demands; no simulator or policy execution."""
from __future__ import annotations

from collections import Counter
import math
import random

from .container_contract import ContainerContractError, container_no
from .month_engine import SIZE_MIX_FT40


def enumerate_inventory(initial: dict, schedule: list, vessels: dict, seed: int):
    """Planned source times are lower bounds, NEVER actual ready/completion times."""
    sources, demands = [], []

    def add_source(cid, bid, kind, job, at, size, load='FULL'):
        if not isinstance(cid, str) or not cid.strip():
            raise ContainerContractError('Source container must have a fixed name')
        if bid not in initial or not math.isfinite(at) or at < 0:
            raise ContainerContractError(f'{cid}: invalid block or planned source time')
        sources.append(dict(container=cid, block=bid, source_kind=kind, source_job=job,
                            planned_source_s=float(at), size=size, load_status=load))

    for bid, boxes in sorted(initial.items()):
        for box in sorted(boxes, key=lambda c: c['container_id']):
            if box['block'] != bid or not box.get('work_available', True):
                raise ContainerContractError('Held or wrong-block initial cargo needs an explicit release plan')
            add_source(box['container_id'], bid, 'INITIAL', None, 0, box['size'], box['load_status'])

    keys = set()
    for e in schedule:
        key, bid = e['job_id'], e['block']
        if key in keys:
            raise ContainerContractError(f'Duplicate order key: {key}')
        keys.add(key)
        at = float(e['arrival_s']) + float(e['travel_s'])
        if bid not in initial or not math.isfinite(at) or at < 0:
            raise ContainerContractError(f'{key}: invalid block or arrival')
        if e['flow'] == 'GATE_IN':
            add_source(container_no(e), bid, 'GATE_IN', key, at,
                       'FT40' if e['size_ft40'] else 'FT20')
        elif e['flow'] == 'GATE_OUT':
            demands.append(dict(job=key, block=bid, kind='GATE_OUT', planned_demand_s=at,
                                entry=e, move=None))
        else:
            raise ContainerContractError(f'{key}: unsupported truck flow')

    streams = set()
    for day in sorted(vessels, key=int):
        for row in vessels[day]:
            key, bid, count = row['key'], row['block'], row['moves']
            start, cadence = float(row['start_s']), float(row['cadence_s'])
            if (key in streams or bid not in initial or row['work'] not in ('LOAD', 'DISCHARGE')
                    or type(count) is not int or count <= 0 or not math.isfinite(start) or start < 0
                    or not math.isfinite(cadence) or cadence <= 0):
                raise ContainerContractError(f'{key}: invalid or repeated vessel stream')
            streams.add(key)
            # Identical to month_run -> inject_vessel's incoming size generation.
            rng = random.Random(f'v3:month:{seed}:{key}:size')
            for m in range(count):
                jid, at = f'{bid}:J-{key}-{m:04d}', start + m * cadence
                if jid in keys:
                    raise ContainerContractError(f'Duplicate job key: {jid}')
                keys.add(jid)
                if row['work'] == 'DISCHARGE':
                    size = 'FT40' if rng.random() < SIZE_MIX_FT40 else 'FT20'
                    add_source(f'IN_{jid}', bid, 'VESSEL_DISCHARGE', jid, at, size)
                else:
                    demands.append(dict(job=jid, block=bid, kind='VESSEL_LOAD',
                                        planned_demand_s=at, entry=row, move=m))
    names = Counter(s['container'] for s in sources)
    if any(n != 1 for n in names.values()):
        raise ContainerContractError('Duplicate source container identity')
    sources.sort(key=lambda s: (s['planned_source_s'], s['container']))
    demands.sort(key=lambda d: (d['planned_demand_s'], d['job']))
    return sources, demands
