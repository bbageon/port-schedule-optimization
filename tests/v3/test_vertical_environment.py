"""Input identity and continuous-stage wiring for opt-in vertical geometry."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from yard_rl.v3.layouts import VerticalEnvironment, synthetic_vertical_spec
from yard_rl.v3.stage.month import plan_days
from yard_rl.v3.stage.month_run import run_month
from yard_rl.v3.world.integrated.profiles import build_h21_profile
from yard_rl.v3.world.integrated.yard_layout import terminal_layout


def environment():
    profile = build_h21_profile()
    spec = synthetic_vertical_spec(profile)
    return VerticalEnvironment.from_dict(spec, profile, terminal_layout()), profile, spec


def entry():
    return dict(job_id='D00-test', block='Y01', arrival_s=3600, day=0,
                lead_s=900, flow='GATE_IN', requested_flow='GATE_IN',
                target=None, size_ft40=True, travel_base_s=190, travel_s=195,
                exit_travel_s=312, con_no='C-unchanged')


def test_only_physical_travel_changes_and_separate_exit_residual_is_preserved():
    env, _, _ = environment()
    source = entry()
    saved = deepcopy(source)
    row = env.adapt_schedule([source])[0]
    assert source == saved and row is not source
    assert row['travel_base_s'] == pytest.approx(205.2)
    assert row['travel_s'] - row['travel_base_s'] == pytest.approx(5)
    assert row['exit_travel_s'] - row['travel_base_s'] == pytest.approx(12)
    changed = {k for k in source if row[k] != source[k]}
    assert changed == {'travel_base_s', 'travel_s', 'exit_travel_s'}
    json.dumps(env.manifest(canonical_input={'test': 'same'}, schedule=[row]), allow_nan=False)


def test_new_route_is_not_clamped_to_legacy_180_420_seconds():
    env, profile, spec = environment()
    spec['geometry']['gate_approach_m'] = 3000
    env = VerticalEnvironment.from_dict(spec, profile, terminal_layout())
    assert env.adapt_schedule([entry()])[0]['travel_s'] > 600


@pytest.mark.parametrize('key,value', [
    ('travel_base_s', float('nan')), ('travel_s', float('inf')),
    ('exit_travel_s', -1)])
def test_invalid_source_travel_is_rejected(key, value):
    env, _, _ = environment()
    source = entry()
    source[key] = value
    with pytest.raises(ValueError):
        env.adapt_schedule([source])


def test_impossible_derived_exit_is_not_silently_clipped():
    env, profile, spec = environment()
    spec['geometry']['gate_approach_m'] = 1
    env = VerticalEnvironment.from_dict(spec, profile, terminal_layout())
    source = entry()
    source['exit_travel_s'] = 60
    with pytest.raises(ValueError, match='exit_travel'):
        env.adapt_schedule([source])


def test_geometry_must_match_stock_and_clear_the_parking_rail():
    _, profile, spec = environment()
    spec['geometry']['bay_count'] = 12
    with pytest.raises(ValueError, match='mismatch'):
        VerticalEnvironment.from_dict(spec, profile, terminal_layout())
    spec = synthetic_vertical_spec(profile)
    spec['geometry']['land_road_clearance_m'] = 13
    with pytest.raises(ValueError, match='parking'):
        VerticalEnvironment.from_dict(spec, profile, terminal_layout())


@pytest.mark.parametrize('kwargs', [
    dict(arm='RL'), dict(arm='RL_TIME'),
    dict(arm='NO_REALLOC', labels_per_day=1),                 # training without the opt-in
    dict(arm='RL_TIME', labels_per_day=1),                    # same, time-only
    dict(arm='RL', labels_per_day=1, layout_training=True),   # opt-in but full candidates
    dict(arm='NO_REALLOC', layout_training=True),             # opt-in without training
])
def test_unqualified_policy_or_training_is_rejected_before_simulation(kwargs):
    _, _, spec = environment()
    with pytest.raises(ValueError):
        run_month(seed=99194001, environment_spec=spec, admission_mode='PRESERVE',
                  supply_mode='COUNT_BALANCED', **kwargs)


def test_layout_training_opt_in_requires_an_environment():
    from yard_rl.v3.actors import BuyerNet, SellerNet
    with pytest.raises(ValueError, match='opt-in'):
        run_month(seed=99194001, arm='RL_TIME', seller_net=SellerNet(), buyer_net=BuyerNet(),
                  labels_per_day=1, layout_training=True,
                  admission_mode='PRESERVE', supply_mode='COUNT_BALANCED')


def test_layout_training_labels_and_fits_inside_the_vertical_environment():
    """The teacher runs in the vertical simulator: branch worlds get the day's
    exploration rate, counterfactual labels are produced and the student is updated.
    Tiny stage; no performance claim."""
    import torch
    from yard_rl.v3.actors import BuyerNet, SellerNet
    _, _, spec = environment()
    torch.manual_seed(1)
    seller, buyer = SellerNet(), BuyerNet()
    days = plan_days(99194001, (40, 40))
    seen, fits = [], []

    def on_fit(day, rows):
        fits.append((day.index, len(rows)))
        return {'n_seller': len(rows)}

    result = run_month(seed=99194001, days=days, arm='RL_TIME', seller_net=seller,
        buyer_net=buyer, labels_per_day=2, workers=1, explore_of_day=lambda d: 0.37,
        on_fit=on_fit, on_day=seen.append, layout_training=True,
        admission_mode='PRESERVE', supply_mode='COUNT_BALANCED', environment_spec=spec)
    assert [d.explore for d in seen] == [0.37, 0.37]
    assert result.environment_manifest['runtime_schedule_sha256']
    assert result.policy_exceptions == 0 and result.n_space == 0
    assert fits and sum(n for _, n in fits) > 0
    assert sum(d.worlds for d in result.days) > 0


def test_continuous_stage_preserves_canonical_requests_and_records_physical_input():
    # Tiny full 21-block stage, not a performance or independent-seed test.
    from yard_rl.v3.eval.seed_bank import create_month_payload, digest
    from yard_rl.v3.stage.supply_plan import balance_vessel_supply
    env, profile, spec = environment()
    days = plan_days(99194001, (21,))
    payload = create_month_payload(99194001, days=days)
    initial = {b: len(s['containers']) for b, s in payload['initial_scenarios'].items()}
    vessels, _ = balance_vessel_supply(payload['vessels'], payload['schedule'],
                                       initial, {b: 1440 for b in initial})
    expected = dict(schedule_sha256=digest(payload['schedule']),
                    initial_scenarios_sha256=digest(payload['initial_scenarios']),
                    vessels_sha256=digest(vessels))
    result = run_month(seed=99194001, days=days, arm='NO_REALLOC',
        admission_mode='PRESERVE', supply_mode='COUNT_BALANCED',
        capture_requests=True, diagnose_admissions=True, capture_daily=True,
        expected_input=expected, environment_spec=spec)
    assert result.environment_manifest['canonical_input'] == expected
    assert result.environment_manifest['runtime_schedule_sha256'] != expected['schedule_sha256']
    assert result.admitted == 21 and result.skipped == 0
    assert result.request_summary['all_requests_admitted']
    assert result.container_flow_summary['passed']
    assert result.policy_exceptions == 0
    assert result.n_space == result.n_time == 0
    assert len(result.days) == 1
    assert result.daily_observation['samples'] > 0
    assert result.vessel_admissions
    assert all(r['structural_min_overrun_s'] == 0 for r in result.vessel_admissions)
    assert all(r['phys_min_completion_s'] <= r['planned_completion_s'] <= r['etd_s']
               for r in result.vessel_admissions)
    json.dumps(asdict(result), allow_nan=False)
