"""Explicit, hash-pinned synthetic inputs; historical defaults remain immutable."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .day_shape import demand_for_shape, hour_masses
from .demand_profile import validate

SCHEMA = 'yard_rl.calibrated_input_bundle.v1'
ASSETS = frozenset(('information', 'bank', 'demand'))


class InputBundle:
    """Load a reviewed bundle once and obtain ordinary build_stage kwargs."""

    def __init__(self, path):
        path = Path(path).resolve()
        spec = json.loads(path.read_text(encoding='utf-8'))
        if spec.get('schema') != SCHEMA or set(spec.get('assets', {})) != ASSETS:
            raise ValueError('complete calibrated input bundle required')
        self._data = {}
        for name, reference in spec['assets'].items():
            asset = (path.parent / reference['path']).resolve()
            raw = asset.read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference['sha256']:
                raise ValueError(f'input asset changed: {name}')
            self._data[name] = json.loads(raw)
        info, bank, demand = (self._data[k] for k in ('information', 'bank', 'demand'))
        if not info.get('all_orders_reserved') or info.get('source_event_contract', {}).get('log_clock_substitution') is not False:
            raise ValueError('source-confirmed reservation information required')
        if bank.get('schema') != 'synthetic-day-shape-v1' or bank.get('initial_stock_mode') != 'preserve':
            raise ValueError('supported synthetic bank and inventory preservation required')
        if set(bank.get('splits', {})) != {'train', 'selection', 'holdout'}:
            raise ValueError('explicit synthetic splits required')
        validate(demand, day_s=86400.)
        for rows in bank['splits'].values():
            if not rows:
                raise ValueError('empty synthetic split')
            for row in rows:
                expected = hour_masses(row['weights'], bank['basis_hour_masses'])
                if expected != row['hour_masses'] or type(row['input_seed']) is not int:
                    raise ValueError('inconsistent synthetic curve or seed')
                self._profile(row)
        self.spec = copy.deepcopy(spec)

    def _profile(self, row):
        p = demand_for_shape(self._data['demand'], row['hour_masses'],
            source=dict(bundle_schema=SCHEMA, shape_id=row['id'], shape_sha256=row['shape_sha256']))
        p['gate_out_share'] = row['gate_out_share']
        validate(p, day_s=86400.)
        return p

    def case(self, split, index, load):
        if split not in self._data['bank']['splits'] or type(index) is not int or index < 0:
            raise ValueError('valid split and nonnegative curve index required')
        if type(load) is not int or load <= 0:
            raise ValueError('positive integer order count required')
        row = self._data['bank']['splits'][split][index]
        return dict(load=load, seed=row['input_seed'], information_profile=copy.deepcopy(self._data['information']),
                    demand_profile=self._profile(row), initial_stock_mode='preserve')
