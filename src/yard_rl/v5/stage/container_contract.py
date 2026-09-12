"""Fixed container identities, not interchangeable block inventory.

The public Order schema stays at six fields. Internal STORE jobs have no
target_container because that attribute means *retrieve an existing box*;
their fixed produced identity is IN_<job_id>, as required by the frozen engine.
An input audit is necessary, but not sufficient, for physical feasibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json


class ContainerContractError(ValueError):
    def __init__(self, message, *, report=None):
        super().__init__(message)
        self.report = report


def container_no(entry: dict) -> str:
    """Resolve the engine identity; never manufacture an unrelated CN<order>."""
    flow, key = entry['flow'], entry['job_id']
    if flow == 'GATE_IN':
        expected = f'IN_{key}'
    elif flow == 'GATE_OUT':
        expected = entry.get('target')
    else:
        raise ContainerContractError(f'{key}: unsupported truck flow {flow!r}')
    if not isinstance(expected, str) or not expected.strip():
        raise ContainerContractError(f'{key}: missing fixed container identity')
    if 'con_no' in entry and entry['con_no'] != expected:
        raise ContainerContractError(
            f"{key}: con_no={entry['con_no']!r} != engine container={expected!r}")
    return expected


def namespace_initial_inventory(built: dict) -> None:
    """Give repeated block-local synthetic IDs unique, stable terminal IDs.

    Only fresh scenario data changes; no engine code, geometry or workload does.
    The initial block is an identity prefix, NOT the container's current location.
    """
    counts = Counter(c for s in built['scenarios'].values() for c in s.containers)
    maps = {b: {c: f'{b}-{c}' if counts[c] > 1 else c for c in s.containers}
            for b, s in built['scenarios'].items()}
    names = [new for mapping in maps.values() for new in mapping.values()]
    if len(names) != len(set(names)):
        raise ContainerContractError('Initial container namespace collision')
    for bid, scenario in built['scenarios'].items():
        mapping = maps[bid]
        for old, box in scenario.containers.items():
            box.container_id = mapping[old]
        scenario.containers = {mapping[c]: box for c, box in scenario.containers.items()}
        for job in scenario.jobs:
            if job.target_container is not None:
                job.target_container = mapping[job.target_container]
    for entry in built['schedule']:
        if entry.get('target') is not None:
            entry['target'] = maps[entry['block']][entry['target']]


def audit_container_plan(built: dict, vessels_by_day: dict) -> dict:
    """Read-only finite-visit audit across initial stock, trucks and vessels.

    Each generated box has one source and at most one exit in this input.
    Re-entry needs an explicit new visit/source contract; a repeated day-local
    pickup name is not evidence that the box returned. Planned source times are
    lower bounds, not fabricated physical completion times.
    """
    counts, examples = Counter(), defaultdict(list)
    sources, exits, identities = {}, {}, []
    initial = built.get('day0', built)['scenarios']

    def issue(kind, item, n=1):
        counts[kind] += n
        if len(examples[kind]) < 5:
            examples[kind].append(item)

    def source(cid, bid, key, earliest):
        item = dict(container=cid, block=bid, source=key, earliest_s=float(earliest))
        if cid in sources:
            issue('duplicate_sources', dict(first=sources[cid], repeated=item))
        else:
            sources[cid] = item
        identities.append(['source', cid, bid, key, float(earliest)])

    def consume(cid, bid, key, earliest):
        item = dict(container=cid, block=bid, order=key, earliest_s=float(earliest))
        if cid in exits:
            issue('duplicate_exits', dict(first=exits[cid], repeated=item))
        else:
            exits[cid] = item
        identities.append(['exit', cid, bid, key, float(earliest)])
        origin = sources.get(cid)
        if origin is None:
            issue('missing_sources', item)
        elif origin['block'] != bid:
            issue('wrong_planned_block', dict(source=origin, exit=item))
        elif origin['earliest_s'] > earliest:
            issue('exit_before_possible_arrival', dict(source=origin, exit=item))

    for bid, scn in sorted(initial.items()):
        for cid, box in sorted(scn.containers.items()):
            if box.container_id != cid:
                issue('initial_key_mismatch', dict(key=cid, container=box.container_id, block=bid))
            source(cid, bid, 'INITIAL', 0)

    outgoing, job_keys = [], set()
    for e in built['schedule']:
        if e['job_id'] in job_keys:
            issue('duplicate_order_keys', dict(order=e['job_id']))
        job_keys.add(e['job_id'])
        try:
            cid = container_no(e)
        except ContainerContractError as error:
            issue('order_identity_mismatch', dict(order=e['job_id'], reason=str(error)))
            continue
        at = float(e['arrival_s']) + float(e['travel_s'])
        if e['flow'] == 'GATE_IN':
            source(cid, e['block'], e['job_id'], at)
        else:
            outgoing.append((cid, e['block'], e['job_id'], at))

    stream_count = 0
    for day in sorted(vessels_by_day):
        for row in vessels_by_day[day]:
            stream_count += 1
            key, bid, moves = row['key'], row['block'], int(row['moves'])
            identities.append(['vessel', key, bid, row['work'], moves,
                               row['start_s'], row['cadence_s'], row.get('targets')])
            if moves <= 0 or row['work'] not in ('LOAD', 'DISCHARGE'):
                issue('invalid_vessel_plan', dict(stream=key, work=row['work'], moves=moves))
                continue
            if row['work'] == 'DISCHARGE':
                for m in range(moves):
                    jid = f'{bid}:J-{key}-{m:04d}'
                    source(f'IN_{jid}', bid, jid, row['start_s'] + m * row['cadence_s'])
            else:
                targets = row.get('targets')
                if not isinstance(targets, (list, tuple)) or len(targets) != moves:
                    issue('unbound_vessel_load_streams', dict(stream=key, moves=moves))
                    counts['unbound_vessel_load_moves'] += moves
                    continue
                for m, cid in enumerate(targets):
                    if not isinstance(cid, str) or not cid.strip():
                        issue('invalid_vessel_target', dict(stream=key, index=m, target=cid))
                        continue
                    outgoing.append((cid, bid, f'{bid}:J-{key}-{m:04d}',
                                     row['start_s'] + m * row['cadence_s']))

    for cid, bid, key, at in sorted(outgoing, key=lambda x: (x[3], x[2])):
        consume(cid, bid, key, at)
    payload = json.dumps(identities, ensure_ascii=False, separators=(',', ':')).encode()
    return dict(schema='yard_rl.v5.container_contract.v1', passed=not counts,
                scope='static fixed-identity contract; NOT physical feasibility or performance',
                truck_orders=len(built['schedule']), vessel_streams=stream_count,
                source_containers=len(sources), fixed_exit_containers=len(exits),
                violations=dict(sorted(counts.items())), examples=dict(examples),
                identity_sha256=hashlib.sha256(payload).hexdigest())


def require_container_plan(report: dict) -> None:
    if report.get('passed') is not True:
        raise ContainerContractError(
            f"Container input contract failed before training: {report.get('violations')}",
            report=report)
