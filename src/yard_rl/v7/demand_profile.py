"""Explicit, optional demand calibration for the CPU daily diagnostic.

Flow probability and clock-hour masses are independent input axes. Total demand,
block assignment, inventory, vessel plans, service physics and reward stay owned
by the existing stage. This does not reproduce observed minute-scale bursts.
"""
from __future__ import annotations

import bisect
import math
import random

SCHEMA = 'observed-truck-demand-marginals-v1'


def validate(profile: dict, *, day_s: float) -> None:
    if profile.get('schema') != SCHEMA:
        raise ValueError('unknown demand profile schema')
    if day_s != 86400.0:
        raise ValueError('observed demand profile requires the 24-hour daily stage')
    if not set(profile).issubset({'schema','gate_out_share','hour_counts','source'}):
        raise ValueError('unsupported demand profile field')
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
