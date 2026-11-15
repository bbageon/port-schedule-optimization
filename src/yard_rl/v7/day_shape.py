"""Optional parametric daily shape bank; does not change stage defaults.

Only mixture weights vary. Shapes are convex interpolations of development-fit
parameters, not a fitted probability law of real days. Random streams for the
shape bank and the existing stage are independent. No day truth enters features.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

SCHEMA = 'synthetic-day-shape-v1'
PEAKS = ((10., 1.5), (15., 2.5), (21., 1.))


def basis_hour_masses():
    """Unit-mass uniform + Gaussian components truncated to the 24-hour day."""
    basis = [[1. / 24] * 24]
    for mean, sigma in PEAKS:
        cdf = lambda t: .5 * (1 + math.erf((t - mean) / (sigma * math.sqrt(2))))
        z = cdf(24) - cdf(0)
        basis.append([(cdf(h + 1) - cdf(h)) / z for h in range(24)])
    return basis


def periodic_linear_basis(knots):
    """Hourly integrals of periodic triangular densities; no midnight discontinuity.

    Knot spacing is 24/knots hours. Each triangle has unit area and support of
    two spacings. A convex mixture is a nonnegative continuous daily curve.
    """
    if type(knots) is not int or not 2 <= knots <= 24:
        raise ValueError('2..24 knots required')
    step = 24. / knots

    def cdf(x):
        if x <= -step:
            return 0.
        if x < 0:
            return (x + step) ** 2 / (2 * step ** 2)
        if x < step:
            return 1 - (step - x) ** 2 / (2 * step ** 2)
        return 1.

    basis = []
    for i in range(knots):
        centre = (i + .5) * step
        basis.append([sum(cdf(h + 1 - centre + offset) - cdf(h - centre + offset)
                          for offset in (-24., 0., 24.)) for h in range(24)])
    return basis


def hour_masses(weights, basis=None):
    basis = basis_hour_masses() if basis is None else basis
    if not basis or any(len(b) != 24 or any(not math.isfinite(x) or x < 0 for x in b)
                        or not math.isclose(sum(b), 1., abs_tol=1e-10) for b in basis):
        raise ValueError('unit-mass nonnegative 24-hour basis required')
    if (len(weights) != len(basis) or any(not math.isfinite(x) or x < 0 for x in weights)
            or not math.isclose(sum(weights), 1., abs_tol=1e-10, rel_tol=0)):
        raise ValueError('finite nonnegative mixture weights summing to one required')
    return [sum(w * b[h] for w, b in zip(weights, basis)) for h in range(24)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def build_bank(fits, *, seed, sizes=None, basis=None, family='original_gaussians'):
    """Fit parameters only; no observed hourly/minute vectors used by this draw."""
    if len(fits) < 2 or len({x['date'] for x in fits}) != len(fits):
        raise ValueError('at least two distinct development dates required')
    basis = basis_hour_masses() if basis is None else basis
    for item in fits:
        hour_masses(item['weights'], basis)
    sizes = dict(train=24, selection=6, holdout=6) if sizes is None else sizes
    if set(sizes) != {'train', 'selection', 'holdout'} or any(
            type(n) is not int or n < 1 for n in sizes.values()):
        raise ValueError('positive train/selection/holdout sizes required')
    splits = {}
    for split_index, split in enumerate(('train', 'selection', 'holdout')):
        rng = random.Random(f'ao:day-shape:{seed}:{split}')
        rows = []
        for i in range(sizes[split]):
            a, b = rng.sample(fits, 2)
            fraction = rng.random()
            w = [(1 - fraction) * x + fraction * y for x, y in zip(a['weights'], b['weights'])]
            masses = hour_masses(w, basis)
            rows.append(dict(id=f'{split}-{i:03d}', parents=[a['date'], b['date']],
                             fraction=fraction, weights=w, hour_masses=masses,
                             input_seed=9934100 + 100 * split_index + i,
                             shape_sha256=digest(masses)))
        splits[split] = rows
    train = splits['train']
    fixed = [sum(row['hour_masses'][h] for row in train) / len(train) for h in range(24)]
    return dict(schema=SCHEMA, seed=seed, family=family, basis_hour_masses=basis, splits=splits,
                fixed_hour_masses=fixed, fixed_sha256=digest(fixed),
                development_fit_sha256=digest(fits),
                sampling='uniform distinct parameter pair and uniform convex interpolation',
                scope='bounded synthetic conditions; not a fitted real-day probability law')


def demand_for_shape(base, masses, *, source):
    """Deep copy existing profile, replacing only hourly masses and provenance."""
    from .demand_profile import validate
    result = copy.deepcopy(base)
    result['hour_counts'] = list(masses)
    result['source'] = dict(base_source=result.get('source'), synthetic_day_shape=copy.deepcopy(source))
    validate(result, day_s=86400.)
    return result
