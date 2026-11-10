"""Notice visibility and calibrated public booking information, separate from physics."""
from bisect import bisect_right
import hashlib
import json
import math
import random

from .appointment_contract import SCHEMA

NOTICE_SCHEMA = 'exact-receipt-v1'


def notice_time(entry):
    return round(max(0.0, float(entry['arrival_s']) - float(entry['lead_s'])), 6)


def announcement_time(entry, period_s=60.0):
    schema = entry.get('notice_schema')
    if schema == NOTICE_SCHEMA:
        return notice_time(entry)
    if schema is not None:
        raise ValueError(f'unknown notice schema: {schema}')
    # Saved unversioned inputs keep their original replay contract.
    notice = max(0.0, float(entry['arrival_s']) - float(entry['lead_s']))
    return round((notice // period_s) * period_s, 6)


def apply_information_profile(schedule, profile, seed):
    if profile.get('schema') != 'empirical-reservation-notice-pairs-v1':
        raise ValueError('unknown empirical information profile')
    pools = {}
    for flow in ('GATE_IN', 'GATE_OUT'):
        data = profile['flows'][flow]
        cumulative, n = [], 0
        for error, lead, count in data['samples']:
            if (not all(math.isfinite(x) for x in (error, lead, count))
                    or lead < 0 or count <= 0 or int(count) != count):
                raise ValueError('invalid empirical timing/count')
            n += count
            cumulative.append(n)
        if not n or n != data['n'] or not math.isfinite(data['eta_offset_s']):
            raise ValueError('empty or inconsistent empirical pool')
        pools[flow] = (data, cumulative, n)
    rng = random.Random(f'v7:empirical-information:{seed}')
    boundary_clamped = 0
    for e in schedule:
        data, cumulative, n = pools[e['flow']]
        error, lead, _ = data['samples'][bisect_right(cumulative, rng.randrange(n))]
        a = float(e['arrival_s'])
        e['reserved_s'] = round(a-error, 3)
        e['reservation_schema'] = SCHEMA
        e['lead_s'] = min(float(lead), a)
        boundary_clamped += lead > a
        e['eta_offset_s'] = float(data['eta_offset_s'])
    return dict(schema=SCHEMA, all_orders_reserved=True, empirical_calibration=True,
        profile_schema=profile['schema'], profile_sha256=hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        calibration_dates=profile['calibration_dates'], error_model='joint empirical pairs by flow; tails retained',
        physical_arrival='unchanged-diurnal-curve', public_reservation='actual-minus-sampled-report-error',
        public_block_eta='reservation-plus-training-flow-median-error-plus-layout-travel',
        notice='actual-minus-joint-sampled-lead; pre-window receipts visible at zero',
        pre_window_notices=boundary_clamped, announcement_grid=NOTICE_SCHEMA)
