"""Frozen reference-return standard deviation; never a hand-picked money unit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

REFERENCE = Path(__file__).with_name('reference_cost_trace.json')


def fit_reference_scale(rows, *, gamma=.999, time_unit_s=60., start_s=86400., end_s=259200.):
    """Carry discounted costs through warmup; fit only the preregistered window."""
    if not (math.isfinite(gamma) and 0 < gamma <= 1 and math.isfinite(time_unit_s)
            and time_unit_s > 0 and math.isfinite(start_s) and math.isfinite(end_s)
            and 0 <= start_s < end_s):
        raise ValueError('Invalid reference discount or fit window')
    if not rows or rows[0][0] != 0:
        raise ValueError('Reference trace must start at time zero')
    z, samples = 0., []
    previous = None
    for raw in rows:
        t, cost = map(float, raw)
        if not all(math.isfinite(v) for v in (t, cost)) or cost < 0:
            raise ValueError('Nonfinite or negative reference cost')
        if previous is not None:
            old_t, old_cost = previous
            if t <= old_t or cost < old_cost - 1e-6:
                raise ValueError('Reference time/cumulative cost must increase')
            z = gamma ** ((t-old_t)/time_unit_s) * z - (cost-old_cost)
            if start_s < t <= end_s:
                samples.append(z)
        previous = t, cost
    if previous[0] < end_s or len(samples) < 2:
        raise ValueError('Reference does not cover the fit window')
    scale = statistics.pstdev(samples)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Reference has no positive finite return scale')
    return dict(method='frozen-reference-discounted-return-std', scale_krw=scale,
                samples=len(samples), gamma=gamma, time_unit_s=time_unit_s,
                fit_start_exclusive_s=start_s, fit_end_inclusive_s=end_s,
                mean_discounted_return_krw=statistics.mean(samples))


def reference_scaling(*, gamma=.999, time_unit_s=60.):
    data = REFERENCE.read_bytes()
    trace = json.loads(data)
    if trace.get('schema') != 'yard.v6.reference-cost-trace.v1':
        raise ValueError('Incompatible frozen cost reference')
    fitted = fit_reference_scale(trace['rows'], gamma=gamma, time_unit_s=time_unit_s,
                                 start_s=trace['fit_start_s'], end_s=trace['fit_end_s'])
    return fitted | dict(reference_sha256=hashlib.sha256(data).hexdigest(),
                         reference_seed=trace['seed'], source_commit=trace['code']['git_head'])


def default_reward_scale(*, gamma=.999, time_unit_s=60.):
    return reference_scaling(gamma=gamma, time_unit_s=time_unit_s)['scale_krw']
