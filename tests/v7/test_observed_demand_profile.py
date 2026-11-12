"""Calibrate exogenous demand without changing jobs, timing semantics or physics."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from yard_rl.v7.demand_profile import SCHEMA, hourly_arrivals, validate
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.profiles import build_h21_profile


@pytest.mark.parametrize('counts', [[0]*24, [1]*23, [-1]+[1]*23, [float('nan')]+[1]*23])
def test_invalid_hour_mass_fails(counts):
    with pytest.raises(ValueError):
        hourly_arrivals(1,100,counts)


def test_empty_hours_and_last_millisecond():
    counts=[0]*24;counts[-1]=1
    times=hourly_arrivals(5,300,counts)
    assert len(times)==300 and times==sorted(times)
    assert all(82800<=t<86400 for t in times)
    assert times==hourly_arrivals(5,300,counts)


def test_reject_non_daily_or_bad_flow_probability():
    with pytest.raises(ValueError):
        validate(dict(schema=SCHEMA),day_s=3600)
    with pytest.raises(ValueError):
        validate(dict(schema=SCHEMA,gate_out_share=1.1),day_s=86400)


def test_separate_mix_and_clock_axes_preserve_background_and_order_attributes():
    info=json.loads(Path('outputs/reports/yr332_real_order_information/profile.json').read_text(encoding='utf-8'))
    args=dict(load=600,seed=9920999,profile=build_h21_profile(),information_profile=info)
    base=build_stage(**args)
    same=build_stage(**args,demand_profile=dict(schema=SCHEMA,gate_out_share=.6))
    mix=build_stage(**args,demand_profile=dict(schema=SCHEMA,gate_out_share=.5375145755597015))
    hours=[1]*12+[3]*12
    clock=build_stage(**args,demand_profile=dict(schema=SCHEMA,gate_out_share=.5375145755597015,hour_counts=hours))
    assert base['schedule']==same['schedule']
    for other in (mix,clock):
        assert base['scenarios']==other['scenarios']
        assert base['vessel_schedule']==other['vessel_schedule']
        assert other['flow_fallbacks_total']==0
        for a,b in zip(base['schedule'],other['schedule']):
            for key in ('job_id','block','size_ft40','travel_s','travel_base_s','exit_travel_s'):
                assert a[key]==b[key], key
    assert [r['arrival_s'] for r in base['schedule']]==[r['arrival_s'] for r in mix['schedule']]
    assert Counter(r['flow'] for r in mix['schedule'])==Counter(r['flow'] for r in clock['schedule'])
    assert sum(r['arrival_s']<43200 for r in clock['schedule'])==150
    # Information is recomputed from changed arrival and the same flow-specific
    # calibration. No actual completion timestamp enters the generated stage.
    for a,b in zip(mix['schedule'],clock['schedule']):
        for key in ('flow','target','con_no'):
            assert a[key]==b[key], key
        assert 0<=b['lead_s']<=b['arrival_s']
    assert 'job_done_time' not in clock['schedule'][0]
