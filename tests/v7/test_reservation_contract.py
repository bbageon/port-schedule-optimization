"""Reservation existence must not disclose the realized arrival or alter demand."""
import copy

import pytest

from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.stage.orders import V3Announcer, build_stage, orders_from_schedule
from yard_rl.v7.world.domain.enums import InformationLevel
from yard_rl.v7.appointment_contract import (
    SCHEMA, add_reservations, reservation_time, require_exact_reservations)
from yard_rl.v7.world.integrated.candidates import _visible_eta_of
from yard_rl.v7.world.integrated.profiles import build_h21_profile


def entry(actual=1800, reserved=1500):
    return dict(job_id='one', block='Y01', flow='GATE_IN', target=None, con_no='IN_one',
                arrival_s=actual, lead_s=actual, travel_s=300, travel_base_s=240,
                exit_travel_s=60, size_ft40=False, reserved_s=reserved,
                reservation_schema=SCHEMA)


class RecordingTerminal:
    def __init__(self):
        self.admitted = []

    def admit_external_job(self, block, job, **kwargs):
        self.admitted.append((block, job, kwargs))


@pytest.mark.parametrize('reserved', [900, 1500, 2100, -600, 87000])
def test_announcer_uses_reservation_for_public_eta_and_actual_for_events(reserved):
    e = entry(reserved=reserved)
    orders, records = orders_from_schedule({'schedule': [e]})
    assert orders['one'].in_out_reserve_s == reserved
    assert records['one'].gate_in_s is None
    assert records['one'].block_in_s is None
    terminal = RecordingTerminal()
    V3Announcer([e]).review(terminal, 0)
    _, job, kwargs = terminal.admitted[0]
    assert job.appointment_gate_time == job.notified_gate_in_s == reserved
    assert job.provided_eta == job.estimated_block_arrival == reserved + 240
    assert job.actual_gate_in == kwargs['gate_in_s'] == 1800
    assert job.actual_block_arrival == 2100


def test_no_admission_before_notice_and_clone_preserves_both_clocks():
    e = entry()
    e['lead_s'] = 600  # notice=1200, exactly on the existing review grid
    ann = V3Announcer([e])
    terminal = RecordingTerminal()
    ann.review(terminal, 1140)
    assert not terminal.admitted
    ann.review(terminal, 1200)
    clone = ann.window(1200, 1800).clone_fresh()
    other = RecordingTerminal()
    clone.review(other, 1200)
    assert ann.n_admitted == clone.n_admitted == 1
    for i in (terminal, other):
        job = i.admitted[0][1]
        assert (job.appointment_gate_time, job.actual_gate_in) == (1500, 1800)


def test_changing_unobserved_actual_does_not_change_public_order_or_eta():
    first, second = entry(1800, 1500), entry(2400, 1500)
    # Same public notice and reservation, different hidden realized arrival.
    a, _ = orders_from_schedule({'schedule': [first]})
    b, _ = orders_from_schedule({'schedule': [second]})
    assert a == b
    jobs = []
    for e in (first, second):
        terminal = RecordingTerminal()
        V3Announcer([e]).review(terminal, 0)
        jobs.append(terminal.admitted[0][1])
    assert jobs[0].actual_gate_in != jobs[1].actual_gate_in
    assert _visible_eta_of(jobs[0], InformationLevel.PRE_ADVICE) == _visible_eta_of(
        jobs[1], InformationLevel.PRE_ADVICE)


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), True, 'not-a-time'])
def test_missing_or_invalid_explicit_reservation_never_falls_back_to_actual(bad):
    e = entry(reserved=bad)
    with pytest.raises(ValueError, match='reservation'):
        orders_from_schedule({'schedule': [e]})
    with pytest.raises(ValueError, match='reservation'):
        V3Announcer([e])


def test_absent_new_reservation_and_unknown_schema_are_rejected():
    e = entry()
    del e['reserved_s']
    with pytest.raises(ValueError, match='reservation'):
        reservation_time(e)
    e['reservation_schema'] = 'unrecognized'
    with pytest.raises(ValueError, match='schema'):
        reservation_time(e)


def test_legacy_month_alias_is_not_reinterpreted_as_new_reservation():
    e = entry(actual=90000, reserved=3600)
    del e['reservation_schema']
    assert reservation_time(e) == 90000
    orders, _ = orders_from_schedule({'schedule': [e]})
    assert orders['one'].in_out_reserve_s == 90000


@pytest.mark.parametrize('sigma', [-1, float('nan'), float('inf')])
def test_invalid_error_scale_is_rejected(sigma):
    with pytest.raises(ValueError, match='appointment_sigma_s'):
        add_reservations([entry()], seed=42, sigma_s=sigma)


def test_generation_preserves_physical_demand_and_repeats_exactly():
    profile = build_h21_profile()
    base = build_stage(load=80, seed=9915101, profile=profile, appointment_sigma_s=0)
    noisy = build_stage(load=80, seed=9915101, profile=profile)
    repeat = build_stage(load=80, seed=9915101, profile=profile)
    assert noisy['reservation_contract']['sigma_s'] == 600
    assert not noisy['reservation_contract']['empirical_calibration']
    assert digest(noisy['schedule']) == digest(repeat['schedule'])
    for a, b in zip(base['schedule'], noisy['schedule']):
        assert {k: v for k, v in a.items() if k != 'reserved_s'} == {
            k: v for k, v in b.items() if k != 'reserved_s'}
        assert a['arrival_s'] == a['reserved_s']
        assert b['reservation_schema'] == SCHEMA
        assert abs(b['arrival_s'] - b['reserved_s']) <= 1200 + 1e-9
    assert any(e['reserved_s'] < e['arrival_s'] for e in noisy['schedule'])
    assert any(e['reserved_s'] > e['arrival_s'] for e in noisy['schedule'])
    assert digest(base['scenarios']) == digest(noisy['scenarios'])
    assert digest(base['vessel_schedule']) == digest(noisy['vessel_schedule'])


def test_month_shifts_public_and_private_times_together():
    from yard_rl.v7.stage.month import build_month, plan_days
    days = plan_days(9915101, loads=(30, 30, 30))
    month = build_month(9915101, days=days, n_days=3)
    for day in days:
        profile = build_h21_profile()
        daily = build_stage(load=day.load, seed=day.seed, profile=profile, fill_ratio=.45)
        actual = [e for e in month['schedule'] if e['day'] == day.index]
        for m, d in zip(actual, daily['schedule']):
            for key in ('arrival_s', 'reserved_s', 'gate_in_s'):
                assert m[key] == d[key] + day.t0
            assert m['lead_s'] == d['lead_s']
            assert m['reservation_schema'] == SCHEMA
    orders, _ = orders_from_schedule(month)
    for e in month['schedule']:
        assert orders[e['job_id']].in_out_reserve_s == round(e['reserved_s'], 3)


def test_unsupported_array_path_cannot_silently_disclose_actual_arrival():
    with pytest.raises(ValueError, match='separate reservation/arrival'):
        require_exact_reservations([entry()], consumer='legacy array')
    require_exact_reservations([entry(reserved=1800)], consumer='legacy array')


def test_real_gate_record_stays_empty_until_event_even_with_earlier_reservation():
    from test_cargo_capacity_wait import world
    terminal, sim, ann, bridge, _ = world(terminal_type='scheduled', full=False)
    e = copy.deepcopy(next(iter(ann.by_epoch.values()))[0])
    e.update(reservation_schema=SCHEMA, reserved_s=-600)
    V3Announcer([e]).review(terminal, 0)
    bridge._sync(terminal, 0)
    assert bridge.records['incoming'].gate_in_s is None
    bridge._sync(terminal, 60)
    assert bridge.records['incoming'].gate_in_s == 60
    assert sim.jobs['incoming'].provided_eta == -300


def test_new_saved_contract_cannot_silently_fall_back_to_actual():
    e = entry()
    contract = add_reservations([e], seed=42, sigma_s=600)
    del e['reservation_schema']
    with pytest.raises(ValueError, match='schema missing'):
        orders_from_schedule(dict(schedule=[e], reservation_contract=contract))


def test_saved_seed_round_trip_preserves_both_times_and_contract(tmp_path):
    from test_fixed_seed import data
    from yard_rl.v7.stage.fixed_seed import build_fixed_seed
    from yard_rl.v7.stage.seed_bundle import load_seed_bundle, save_seed_bundle
    from yard_rl.v7.stage.cargo_input import restore_input
    original = data()
    original['reservation_contract'] = add_reservations(original['schedule'], seed=42, sigma_s=600)
    doc, audit = build_fixed_seed(original, {}, seed=9900306)
    assert audit['passed']
    info = save_seed_bundle(doc, tmp_path / 'reserved.gz')
    restored_doc, _ = load_seed_bundle(tmp_path / 'reserved.gz', expected_sha256=info['sha256'])
    restored = restore_input(restored_doc, seed=9900306, days=original['days'], lead_mode='DIST')
    assert restored['reservation_contract'] == original['reservation_contract']
    assert restored['schedule'][0]['reserved_s'] == original['schedule'][0]['reserved_s']
    assert restored['schedule'][0]['arrival_s'] == original['schedule'][0]['arrival_s']
    orders, records = orders_from_schedule(restored)
    assert orders['pickup'].in_out_reserve_s != original['schedule'][0]['arrival_s']
    assert records['pickup'].gate_in_s is None


def test_array_entrypoints_reject_unimplemented_noisy_reservations():
    pytest.importorskip('jax')
    from yard_rl.v7.gpu.host_terminal import to_terminal_world
    from yard_rl.v7.gpu.admission import schedule_arrays
    with pytest.raises(ValueError, match='separate reservation/arrival'):
        to_terminal_world(build_h21_profile(), {'scenarios': {}, 'schedule': [entry()]}, lead_s=0)
    with pytest.raises(ValueError, match='separate reservation/arrival'):
        schedule_arrays([entry()], block_index={'Y01': 0}, row_of_job={'one': 0},
                        lead_s=0, n_epochs=2)
