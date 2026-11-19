"""Shared immutable paths and pure calibration helpers for YR-332-bb."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from audit_day_shapes import fit_simplex, tvd
from recheck_transport_queue import digest as sha
from yard_rl.v7.day_shape import periodic_linear_basis

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'outputs/reports/yr332_confirmed_demand'
BA = BASE/'outputs/reports/yr332_gate_time_provenance'
PREREG = 'docs/research/v7-discharge/confirmed-demand-prereg.md'
OLD_TEMPLATES = BASE/'outputs/reports/yr332_visit_arrivals/visit-arrival-templates.json'
OLD_FIT = BASE/'outputs/reports/yr332_day_shapes/calibration-periodic_linear_24.json'
OLD_MIX = BASE/'outputs/reports/yr332_day_flow_volume/measurement.json'
OLD_MINUTES = BASE/'outputs/reports/yr332_synthetic_inputs'
OLD_DEMAND = BASE/'outputs/reports/yr332_visit_links/demand-profiles.json'
OLD_BANK = BASE/'outputs/reports/yr332_global_value_context/bank.json'
ARMS = ('legacy', 'curve', 'mix', 'minutes', 'links', 'all')
LOADS = (3500, 7500, 15000)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep canonical numeric hour insertion order; lexical sort changes MC rows.
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def ordered_templates(value):
    if set(value['minute_counts']) != {str(h) for h in range(24)}:
        raise ValueError('all 24 source clock hours required')
    dates = value['dates']
    if dates != sorted(set(dates)):
        raise ValueError('unique chronological development dates required')
    counts = {str(h):value['minute_counts'][str(h)] for h in range(24)}
    if any(len(rows) != len(dates) or any(len(r) != 60 or any(type(n) is not int or n < 0 for n in r) for r in rows)
           for rows in counts.values()):
        raise ValueError('nonnegative integer minute counts for every date-hour required')
    return dict(value,minute_counts=counts)


def fit_days(daily):
    basis = periodic_linear_basis(24)
    fits = []
    for index, row in enumerate(sorted(daily,key=lambda r:r['date'])):
        counts, n = row['hour_counts'], row['orders']
        if n <= 0 or len(counts) != 24 or sum(counts) != n:
            raise ValueError('daily order conservation failed')
        target = np.array(counts)/n
        error,weights,fitted = fit_simplex(target,basis)
        rng = np.random.default_rng(9934000+index)
        noise = .5*np.abs(rng.multinomial(n,fitted,size=1000)/n-fitted).sum(axis=1)
        threshold = float(np.quantile(noise,.95)); residual = tvd(target,fitted)
        fits.append(dict(date=row['date'],orders=n,hour_counts=counts,weights=weights.tolist(),
            hour_masses=fitted.tolist(),squared_error=error,residual_tvd=residual,noise_tvd_p95=threshold,
            within_noise=residual <= threshold,zero_weight_components=np.flatnonzero(weights < 1e-8).tolist()))
    return dict(schema='day-shape-calibration-v1',family='periodic_linear_24',fits=fits,
                input_family_pass=all(r['within_noise'] for r in fits),passed_dates=sum(r['within_noise'] for r in fits),
                total_dates=len(fits),development_dates=[r['date'] for r in fits])


@contextmanager
def bind(module, **values):
    """Redirect old offline fitters sequentially; restore all globals on failure."""
    previous = {k:getattr(module,k) for k in values}
    try:
        for k,v in values.items():setattr(module,k,v)
        yield
    finally:
        for k,v in previous.items():setattr(module,k,v)


def make_cases(bank):
    rows = []
    for split, shapes in bank['splits'].items():
        for index, shape in enumerate(shapes):
            arms = ARMS if split == 'train' and index < 3 else ('all',)
            for load in LOADS:
                rows.extend(dict(split=split,index=index,seed=shape['input_seed'],load=load,arm=arm) for arm in arms)
    return rows
