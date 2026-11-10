import copy
import pytest

from yard_rl.v7.information_contract import (NOTICE_SCHEMA, announcement_time,
                                            apply_information_profile, notice_time)
from yard_rl.v7.stage.orders import V3Announcer, build_stage, orders_from_schedule
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.appointment_contract import require_exact_reservations


def entry():
    return dict(job_id='one', block='Y01', flow='GATE_IN', target=None, con_no='IN_one',
                arrival_s=151.25, lead_s=30, travel_s=300, travel_base_s=240,
                exit_travel_s=60, size_ft40=False, reserved_s=90,
                reservation_schema='all-reserved-synthetic-v1', notice_schema=NOTICE_SCHEMA)


def profile():
    return dict(schema='empirical-reservation-notice-pairs-v1', calibration_dates=['a','b'],
                flows={f: dict(n=3, eta_offset_s=60, samples=[[60, 10, 2], [-90, 20, 1]])
                       for f in ('GATE_IN','GATE_OUT')})


class Recorder:
    def __init__(self):
        self.calls = []
    def admit_external_job(self, block, job, **kwargs):
        self.calls.append((job, kwargs))


def test_exact_notice_not_visible_on_prior_minute_or_in_branch():
    e = entry()
    ann = V3Announcer([e]); term = Recorder()
    for t in (0, 120, 121.249999):
        ann.review(term, t)
    assert not term.calls
    ann.review(term, 121.25)
    assert len(term.calls) == 1
    other = Recorder()
    ann.window(121, 122).clone_fresh().review(other, 121.25)
    assert other.calls[0][1] == term.calls[0][1]
    orders, _ = orders_from_schedule({'schedule': [e]})
    assert orders['one'].copino_notice_s == notice_time(e) == 121.25


def test_zero_lead_has_notice_at_actual_entry():
    e=entry();e['lead_s']=0
    assert announcement_time(e) == e['arrival_s']


def test_legacy_saved_input_preserves_floor_contract():
    e=entry();e.pop('notice_schema')
    assert announcement_time(e) == 120


def test_joint_calibration_preserves_physics_and_reproduces():
    before=[dict(entry(), job_id=str(i), flow='GATE_OUT' if i%2 else 'GATE_IN') for i in range(300)]
    a,b=copy.deepcopy(before),copy.deepcopy(before)
    apply_information_profile(a,profile(),5);apply_information_profile(b,profile(),5)
    assert a==b
    assert {(round(e['arrival_s']-e['reserved_s']), e['lead_s']) for e in a} == {(60,10),(-90,20)}
    for old,new in zip(before,a):
        for key in ('arrival_s','travel_s','travel_base_s','block','target','flow'):
            assert old[key]==new[key]


def test_public_eta_uses_fixed_calibration_not_actual_future():
    a=entry();a['eta_offset_s']=60
    b=dict(a, arrival_s=211.25, lead_s=90)
    terms=[]
    for e in (a,b):
        term=Recorder();V3Announcer([e]).review(term,121.25);terms.append(term.calls[0][0])
    assert terms[0].provided_eta==terms[1].provided_eta==390
    assert terms[0].actual_gate_in!=terms[1].actual_gate_in


def test_pre_window_notice_is_explicit_not_negative():
    p=profile()
    for f in p['flows'].values():f.update(n=1,samples=[[100000,200000,1]])
    e=entry();contract=apply_information_profile([e],p,1)
    assert notice_time(e)==0 and e['reserved_s']<0
    assert contract['pre_window_notices']==1


def test_array_conversion_fails_instead_of_rounding_receipt_early():
    with pytest.raises(ValueError,match='exact receipt'):
        require_exact_reservations([entry()],consumer='test')


def test_default_stage_versions_notice_contract():
    built=build_stage(load=20,seed=9917199,profile=build_h21_profile())
    assert built['reservation_contract']['announcement_grid']==NOTICE_SCHEMA
    assert all(e['notice_schema']==NOTICE_SCHEMA for e in built['schedule'])


def test_calibration_and_synthetic_sigma_cannot_silently_mix():
    with pytest.raises(ValueError,match='choose empirical'):
        build_stage(load=10,seed=1,profile=build_h21_profile(),
                    information_profile=profile(),appointment_sigma_s=0)
