"""Explicit, optional demand calibration for the CPU daily diagnostic.

Flow probability, clock-hour masses and the within-hour arrival process are
independent input axes. Total demand, block assignment, inventory, vessel plans,
service physics and reward stay owned by the existing stage. Without a
`within_hour` field the old stratified (evenly spaced) arrivals are unchanged.
"""
from __future__ import annotations

import bisect
import math
import random

SCHEMA = 'observed-truck-demand-marginals-v1'
WITHIN_HOUR_METHODS = ('poisson', 'observed_visit_minutes')


def validate(profile: dict, *, day_s: float) -> None:
    if profile.get('schema') != SCHEMA:
        raise ValueError('unknown demand profile schema')
    if day_s != 86400.0:
        raise ValueError('observed demand profile requires the 24-hour daily stage')
    if not set(profile).issubset({'schema','gate_out_share','hour_counts','within_hour','source'}):
        raise ValueError('unsupported demand profile field')
    if profile.get('within_hour') is not None:
        validate_within_hour(profile['within_hour'])
    p=profile.get('gate_out_share')
    if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
        raise ValueError('gate_out_share must be a finite probability')
    counts=profile.get('hour_counts')
    if counts is not None:
        if len(counts)!=24 or any(not math.isfinite(x) or x<0 for x in counts) or sum(counts)<=0:
            raise ValueError('hour_counts requires 24 finite nonnegative masses and a positive total')


def hourly_arrivals(seed: int, total: int, counts: list[float]) -> list[float]:
    """Same stratified quantiles as the old curve; only the arrival CDF changes."""
    validate(dict(schema=SCHEMA,hour_counts=counts),day_s=86400.)
    if total < 0:
        raise ValueError('negative order count')
    cumulative=[0.]
    for x in counts:
        cumulative.append(cumulative[-1]+x)
    rng=random.Random(f'h21d:arr:{seed}')
    times=[]
    for i in range(total):
        q=cumulative[-1]*(i+rng.random())/total
        hour=min(23,bisect.bisect_right(cumulative,q)-1)
        fraction=(q-cumulative[hour])/(cumulative[hour+1]-cumulative[hour])
        # The schedule stores milliseconds; never round up into the next day.
        times.append(min(86399.999,round((hour+fraction)*3600.,3)))
    return times


def validate_within_hour(spec: dict) -> None:
    """`poisson`: independent uniform times inside each hour (random, no clustering).
    `observed_visit_minutes`: per-minute counts of observed truck visits for each
    date and clock hour (aggregate counts, no identifiers)."""
    if not isinstance(spec, dict) or spec.get('method') not in WITHIN_HOUR_METHODS:
        raise ValueError('unknown within-hour arrival method')
    if spec['method'] == 'poisson':
        if not set(spec).issubset({'method', 'source'}):
            raise ValueError('unsupported within-hour field')
        return
    if not set(spec).issubset({'method', 'dates', 'minute_counts', 'source'}):
        raise ValueError('unsupported within-hour field')
    dates = spec.get('dates')
    if not dates or len(set(dates)) != len(dates):
        raise ValueError('observed templates need distinct dates')
    counts = spec.get('minute_counts')
    if not isinstance(counts, dict) or set(counts) != {str(h) for h in range(24)}:
        raise ValueError('observed templates need all 24 clock hours')
    for hour, days in counts.items():
        if len(days) != len(dates):
            raise ValueError('one minute vector per date is required')
        for minutes in days:
            if len(minutes) != 60 or any(not isinstance(x, int) or x < 0 for x in minutes):
                raise ValueError('minute vectors need 60 nonnegative integers')
        if sum(map(sum, days)) <= 0:
            raise ValueError(f'hour {hour} has no observed arrivals')


def within_hour_arrivals(seed: int, times: list[float], spec: dict) -> tuple[list[float], dict]:
    """Redraw times inside each clock hour only.

    Each hour keeps its count and the entries keep their order, so flows, blocks and
    targets stay attached to the same arrival rank. `observed_visit_minutes` stacks
    whole observed dates of the same clock hour in a seeded random order until the
    hour is filled; only the last date is subsampled. A higher load therefore
    superposes more observed days. Times are uniform inside the drawn minute.
    """
    validate_within_hour(spec)
    hours: dict[int, list[int]] = {}
    for i, t in enumerate(times):
        if not 0 <= t < 86400:
            raise ValueError('arrival outside the 24-hour stage')
        hours.setdefault(int(t // 3600), []).append(i)
    out = list(times)
    meta = dict(method=spec['method'], hours={})
    for hour, index in sorted(hours.items()):
        index.sort(key=lambda i: (times[i], i))
        n = len(index)
        rng = random.Random(f"h21d:within:{spec['method']}:{seed}:{hour}")
        if spec['method'] == 'poisson':
            offsets = [3600. * rng.random() for _ in range(n)]
            meta['hours'][str(hour)] = dict(n=n)
        else:
            days = spec['minute_counts'][str(hour)]
            order = list(range(len(days)))
            rng.shuffle(order)
            minutes: list[int] = []
            used = 0
            whole = 0
            while len(minutes) < n:
                day = days[order[used % len(order)]]
                used += 1
                pool = [m for m, c in enumerate(day) for _ in range(c)]
                need = n - len(minutes)
                if len(pool) <= need:
                    minutes += pool
                    whole += len(pool) > 0
                else:
                    minutes += rng.sample(pool, need)
            offsets = [60. * m + 60. * rng.random() for m in minutes]
            meta['hours'][str(hour)] = dict(n=n, dates_used=used, whole_dates=whole,
                                            cycled=used > len(order))
        offsets = sorted(min(3599.999, round(x, 3)) for x in offsets)
        for i, x in zip(index, offsets):
            out[i] = min(86399.999, round(hour * 3600. + x, 3))
    meta['cycled_hours'] = sum(bool(h.get('cycled')) for h in meta['hours'].values())
    return out, meta
