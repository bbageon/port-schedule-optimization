"""Demand-consistent initial allocation, before time zero (YR-332-at).

This is a synthetic starting-layout assumption, not a free runtime transfer.
Only uncommitted top boxes can be allocated to another block. Total stock,
sizes, all existing targets and vessel plans are conserved. Infeasible input
raises instead of changing an outbound order into an inbound order.
"""
from __future__ import annotations

from collections import Counter
import copy
import dataclasses
import random

from ..world.sim.stack import YardStacks

MODES = ('preserve', 'strict', 'legacy')


class InitialStockError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def preserve_initial_stock(scenarios, schedule, profile, seed, mode='preserve'):
    """Return repaired copies, or the original objects if no repair is needed.

    ``strict`` validates the original allocation without repairing it;
    ``legacy`` is explicitly for reproducing historical fallback results.
    A failed repair never mutates the input objects.
    """
    if mode not in MODES:
        raise ValueError(f'initial_stock_mode must be one of {MODES}: {mode!r}')
    missing = [e for e in schedule if e['flow'] != e['requested_flow']]
    report = dict(schema='initial-stock-allocation-v1', mode=mode,
                  shortages=dict(sorted(Counter(e['block'] for e in missing).items())),
                  repaired=0, allocations=[], scope='synthetic initial state, not runtime transport')
    if not missing or mode == 'legacy':
        return scenarios, schedule, report
    if mode == 'strict':
        raise InitialStockError('Requested outbound jobs exceed initial available stock', report)
    if any(e['requested_flow'] != 'GATE_OUT' or e.get('fallback_reason') != 'no_free_target'
           for e in missing):
        raise InitialStockError('Unrecognised flow mismatch; refusing to repair', report)

    # Work on copies: failure cannot leak a partially reallocated starting state.
    scns, rows = copy.deepcopy((scenarios, schedule))
    reserved = {b: {j.target_container for j in s.jobs if j.target_container}
                for b, s in scns.items()}
    for e in rows:
        if e.get('target'):
            reserved[e['block']].add(e['target'])
    stacks = {b: YardStacks(profile.block, s.containers) for b, s in scns.items()}
    rng = random.Random(f'v7:initial-stock:{seed}')
    g = profile.block
    # Keep the generator's original per-block empty-slot reserve.
    ceiling = g.bay_count * g.row_count * g.tier_max - g.bay_count
    before_sizes = Counter(c.size for s in scns.values() for c in s.containers.values())

    for e in rows:
        if e['flow'] == e['requested_flow']:
            continue
        bid = e['block']; dst = stacks[bid]
        slots = {}
        if len(dst.containers) < ceiling:
            for bay in range(1, g.bay_count + 1):
                for row in range(1, g.row_count + 1):
                    if dst.top_tier(bay, row) < g.tier_max:
                        pile = dst.stack(bay, row)
                        size = dst.containers[pile[-1]].size if pile else None
                        slots.setdefault(size, []).append((bay, row))
        choices = []
        for donor in sorted(stacks):
            if donor == bid:
                continue
            src = stacks[donor]
            for position in sorted(src._stacks):
                pile = src.stack(*position)
                if not pile or pile[-1] in reserved[donor]:
                    continue
                box = src.containers[pile[-1]]
                if slots.get(None) or slots.get(box.size):
                    choices.append((donor, box.container_id))
        if not choices:
            report['failed_order'] = e['job_id']
            report['unresolved'] = len(missing) - report['repaired']
            raise InitialStockError('No uncommitted top box and compatible initial slot', report)
        donor, old_id = rng.choice(choices)
        src = stacks[donor]
        box = src.containers[old_id]
        bay, row = rng.choice(slots.get(None, []) + slots.get(box.size, []))
        original = src.remove(old_id)
        new_id = f'SUPPLY-{donor}-{old_id}'
        if new_id in dst.containers:
            raise InitialStockError('Initial supply identity collision', report)
        replacement = dataclasses.replace(box, container_id=new_id, block=g.block_id)
        position = dst.place(replacement, bay, row)
        reserved[bid].add(new_id)
        e.update(flow='GATE_OUT', target=new_id, fallback_reason=None)
        report['allocations'].append(dict(order=e['job_id'], donor_block=donor,
            donor_local_id=old_id, initial_position=original, recipient_block=bid,
            target=new_id, final_position=position, size=box.size.value))
        report['repaired'] += 1
    after_sizes = Counter(c.size for s in stacks.values() for c in s.containers.values())
    if before_sizes != after_sizes:
        raise InitialStockError('Initial inventory conservation failed', report)
    for b, s in scns.items():
        scns[b] = dataclasses.replace(s, containers=stacks[b].containers)
    report['initial_counts_before'] = {b: len(s.containers) for b, s in scenarios.items()}
    report['initial_counts_after'] = {b: len(s.containers) for b, s in scns.items()}
    return scns, rows, report
