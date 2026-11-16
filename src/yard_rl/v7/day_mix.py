"""Jointly interpolate daily shape and outbound proportion, using development aggregates."""
from __future__ import annotations

import copy
import math


def add_daily_mix(bank, daily):
    by_date = {row['date']: row['gate_out_share'] for row in daily}
    if len(by_date) != len(daily) or any(not math.isfinite(p) or not 0 <= p <= 1
                                       for p in by_date.values()):
        raise ValueError('distinct dates with valid outbound proportions required')
    result = copy.deepcopy(bank)
    for rows in result['splits'].values():
        for row in rows:
            a, b = row['parents']; f = row['fraction']
            if not 0 <= f <= 1:
                raise ValueError('invalid interpolation fraction')
            row['gate_out_share'] = (1 - f) * by_date[a] + f * by_date[b]
    result['fixed_gate_out_share'] = sum(r['gate_out_share'] for r in result['splits']['train']) / len(result['splits']['train'])
    result['mix_scope'] = 'same parent parameters and fraction as the daily shape; bounded design, not a real-day probability law'
    return result
