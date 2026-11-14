"""YR-332-an: the BNCT realized-time alias follows calibrated arrivals.

Only `gate_in_s` may change. Saved aj/ak inputs stay reproducible with
`bnct_time_sync=False`; the default path without calibration is untouched.
"""
import json
from pathlib import Path

import pytest

from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.order_schema import attach, sync_realized_gate_in
from yard_rl.v7.world.integrated.profiles import build_h21_profile

AK = Path('outputs/reports/yr332_visit_arrivals')
INFO = Path('outputs/reports/yr332_real_order_information/profile.json')
SEEDS = {3500: 9921100, 7500: 9921110, 15000: 9921120}


def info():
    return json.loads(INFO.read_text(encoding='utf-8'))


def ak_profiles():
    return json.loads((AK / 'demand-profiles.json').read_text(encoding='utf-8'))['profiles']


def saved_digests():
    rows = json.loads((AK / 'input-preflight.json').read_text(encoding='utf-8'))['rows']
    return {(r['variant'], r['load']): r['schedule_sha256'] for r in rows}


def test_alias_helper_only_touches_existing_alias():
    entry = attach(dict(job_id='D-1', block='Y01', arrival_s=10.0, size_ft40=True))
    entry['arrival_s'] = 25.0
    assert sync_realized_gate_in(entry) and entry['gate_in_s'] == 25.0
    assert entry['reserved_s'] == 10.0          # reservation contract owns this field
    assert not sync_realized_gate_in(entry)
    plain = dict(arrival_s=3.0)
    assert not sync_realized_gate_in(plain) and plain == dict(arrival_s=3.0)


def test_uncalibrated_path_has_nothing_to_resync():
    args = dict(load=600, seed=9920997, profile=build_h21_profile(), information_profile=info())
    new, old = build_stage(**args), build_stage(**args, bnct_time_sync=False)
    assert new['bnct_gate_in_resynced'] == 0 and 'bnct_gate_in_resynced' not in old
    assert new['schedule'] == old['schedule']
    assert all(e['gate_in_s'] == e['arrival_s'] for e in new['schedule'])


@pytest.mark.parametrize('variant', ['observed_mix_hours', 'observed_hours_poisson',
                                     'observed_hours_visit_minutes'])
def test_calibrated_inputs_change_only_the_alias(variant):
    args = dict(load=1200, seed=9920996, profile=build_h21_profile(), information_profile=info(),
                demand_profile=ak_profiles()[variant])
    new, old = build_stage(**args), build_stage(**args, bnct_time_sync=False)
    assert new['scenarios'] == old['scenarios'] and new['vessel_schedule'] == old['vessel_schedule']
    changed = 0
    for a, b in zip(old['schedule'], new['schedule']):
        assert {k: v for k, v in a.items() if k != 'gate_in_s'} == \
               {k: v for k, v in b.items() if k != 'gate_in_s'}
        assert b['gate_in_s'] == b['arrival_s']
        changed += a['gate_in_s'] != b['gate_in_s']
    assert changed == new['bnct_gate_in_resynced'] > 0


@pytest.mark.parametrize('load', sorted(SEEDS))
def test_saved_ak_inputs_reproduce_with_legacy_alias(load):
    """How to rebuild the saved aj/ak input digests after the fix."""
    digests, profiles = saved_digests(), ak_profiles()
    for variant in ('observed_mix_hours', 'observed_hours_visit_minutes'):
        args = dict(load=load, seed=SEEDS[load], profile=build_h21_profile(),
                    information_profile=info(), demand_profile=profiles[variant])
        legacy = build_stage(**args, bnct_time_sync=False)
        assert digest(legacy['schedule']) == digests[(variant, load)]
        synced = build_stage(**args)
        assert digest(synced['schedule']) != digests[(variant, load)]
        assert all(e['gate_in_s'] == e['arrival_s'] for e in synced['schedule'])
