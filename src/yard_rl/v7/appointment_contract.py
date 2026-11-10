"""Synthetic public reservations, independent of physical arrival realization.

The daily physical arrivals and notice-to-arrival samples remain fixed. We vary
only the accuracy of the reported reservation: R = A - clipped Gaussian error.
This is a controlled information experiment, not a calibrated booking process.
Negative R is a reservation before the observation window, not a missing value.
"""
from __future__ import annotations

import math
import random

SCHEMA = 'all-reserved-synthetic-v1'


def reservation_time(entry: dict) -> float:
    """Read a public reservation; old unversioned synthetic inputs replay exactly.

    Legacy monthly BNCT aliases can still contain day-local reserved_s. Those
    inputs historically used arrival_s as R; do not silently reinterpret them.
    New tagged inputs must contain a finite reservation; never fill a missing R
    with the private actual arrival.
    """
    schema = entry.get('reservation_schema')
    if schema is None:
        value = entry['arrival_s']
    elif schema == SCHEMA:
        value = entry.get('reserved_s')
    else:
        raise ValueError(f"{entry.get('job_id')}: unknown reservation schema {schema!r}")
    if value is None or isinstance(value, bool):
        raise ValueError(f"{entry.get('job_id')}: all orders require a finite reservation")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{entry.get('job_id')}: invalid reservation {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{entry.get('job_id')}: all orders require a finite reservation")
    return value


def add_reservations(schedule: list[dict], *, seed: int, sigma_s: float) -> dict:
    """Sample once per input seed; changing sigma never consumes physical RNGs."""
    sigma_s = float(sigma_s)
    if not math.isfinite(sigma_s) or sigma_s < 0:
        raise ValueError('appointment_sigma_s must be finite and nonnegative')
    rng = random.Random(f'v7:appointment:{seed}')
    for entry in schedule:
        actual = float(entry['arrival_s'])
        if not math.isfinite(actual):
            raise ValueError(f"{entry.get('job_id')}: invalid actual arrival")
        error = sigma_s * max(-2.0, min(2.0, rng.gauss(0.0, 1.0)))
        entry['reserved_s'] = round(actual - error, 3)
        entry['reservation_schema'] = SCHEMA
        reservation_time(entry)
    return dict(schema=SCHEMA, all_orders_reserved=True,
                error_model='normal-clipped-at-2-sigma', sigma_s=sigma_s,
                mean_before_clipping_s=0.0, empirical_calibration=False,
                physical_arrival='unchanged-diurnal-curve',
                public_reservation='actual-minus-synthetic-error',
                public_block_eta='reservation-plus-layout-travel',
                notice='unchanged-actual-minus-existing-lead',
                announcement_grid='existing-60-second-floor-grid',
                boundary='reservations-can-precede-or-follow-observation-window')


def validate_reservation_metadata(schedule, contract) -> None:
    """A new saved input cannot downgrade individual rows to the legacy fallback."""
    if contract is None:
        return
    if contract.get('schema') != SCHEMA or contract.get('all_orders_reserved') is not True:
        raise ValueError('Invalid all-reserved reservation contract metadata')
    for entry in schedule:
        if entry.get('reservation_schema') != SCHEMA:
            raise ValueError(f"{entry.get('job_id')}: reservation schema missing from new input")
        reservation_time(entry)


def require_exact_reservations(schedule, *, consumer: str) -> None:
    """Reject noisy reservations in legacy array paths that assume R == A."""
    for entry in schedule:
        if reservation_time(entry) != float(entry['arrival_s']):
            raise ValueError(f'{consumer} does not support separate reservation/arrival '
                             'times; use the CPU stage path or explicitly sigma=0')
