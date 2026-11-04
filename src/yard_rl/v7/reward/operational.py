"""Currency-free physical ledgers and frozen, per-objective PPO normalization."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from ..schema.lifecycle import censored_turn_time_s
from ..stage.episode import rehandles_of, yc_empty_travel_s
from ..stage.month import month_vessel_idle

KEYS = ('truck_s', 'vessel_s', 'empty_s', 'rehandles')
REFERENCE = Path(__file__).with_name('operational_reference.json')
WEIGHTS = (.25, .25, .25, .25)


def physical_totals(records, *, end_s, vessel_idle, empty_s, rehandles):
    truck = sum(censored_turn_time_s(r, end_s) or 0. for r in records.values())
    values = (truck, sum(float(v[1]) for v in vessel_idle.values()), float(empty_s), float(rehandles))
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Physical ledgers must be finite and nonnegative')
    return dict(zip(KEYS, values))


def runtime_totals(rt, t):
    return physical_totals(rt.bridge.records, end_s=t,
        vessel_idle=month_vessel_idle(rt.mbt, rt.meta, rt.archive),
        empty_s=yc_empty_travel_s(rt.mbt), rehandles=rehandles_of(rt.mbt))


def fit_reference(rows, *, gamma=.999, time_unit_s=60., start_s=86400., end_s=259200.):
    if not (math.isfinite(gamma) and 0 < gamma <= 1 and math.isfinite(time_unit_s)
            and time_unit_s > 0 and 0 <= start_s < end_s and math.isfinite(end_s)):
        raise ValueError('Invalid operational normalization window/discount')
    if not rows or rows[0][0] != 0:
        raise ValueError('Physical reference must start at zero')
    previous, returns = None, [0.] * 4
    samples = [[] for _ in KEYS]
    for row in rows:
        if len(row) != 5 or any(not math.isfinite(float(v)) or v < 0 for v in row):
            raise ValueError('Invalid physical reference row')
        t, *values = row
        if previous is not None:
            dt = t - previous[0]
            changes = [x-y for x,y in zip(values, previous[1:])]
            if dt <= 0 or min(changes) < -1e-6:
                raise ValueError('Physical reference clock/ledgers decreased')
            returns = [gamma**(dt/time_unit_s)*z+dx for z,dx in zip(returns, changes)]
            if start_s < t <= end_s:
                for data, z in zip(samples, returns):
                    data.append(z)
        previous = row
    if previous[0] < end_s or len(samples[0]) < 2:
        raise ValueError('Physical reference does not cover the fit window')
    scales = tuple(statistics.pstdev(x) for x in samples)
    if any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Every physical objective needs positive reference variance')
    return dict(scales=dict(zip(KEYS, scales)), samples=len(samples[0]), gamma=gamma,
        time_unit_s=time_unit_s, fit_start_s=start_s, fit_end_s=end_s,
        discounted_means=dict(zip(KEYS, map(statistics.mean, samples))))


def reference_config(*, gamma=.999, time_unit_s=60., weights=WEIGHTS):
    if len(weights) != 4 or any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.):
        raise ValueError('Four finite nonnegative objective weights must sum to one')
    data = REFERENCE.read_bytes()
    trace = json.loads(data)
    if trace['schema'] != 'yard.v6.operational-reference.v1' or tuple(trace['keys']) != KEYS:
        raise ValueError('Incompatible operational reference')
    fit = fit_reference(trace['rows'], gamma=gamma, time_unit_s=time_unit_s,
                        start_s=trace['fit_start_s'], end_s=trace['fit_end_s'])
    return fit | dict(method='frozen-physical-discounted-return-std', weights=dict(zip(KEYS, weights)),
        reference_sha256=hashlib.sha256(data).hexdigest(), reference_seed=trace['seed'],
        source_commit=trace['code']['git_head'], currency_used=False, mean_centered=False, clipped=False)


def normalized_loss(totals, reference):
    return sum(reference['weights'][k] * totals[k] / reference['scales'][k] for k in KEYS)
