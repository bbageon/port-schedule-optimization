from types import SimpleNamespace

import pytest

from yard_rl.v7.reward.gate_age import StopGateAgeClock, truck_points
from yard_rl.v7.stage.month_visits import stop_intervals, truck_visits, visit_plans


def test_multibox_stop_across_midnight_counts_once_and_preserves_gate_age():
    gate = 23 * 3600
    plan = dict(id='v', stops=[['a', 'GATE_IN', ['x', 'y']], ['b', 'GATE_OUT', ['z']]],
                travel_s=[60, 180], exit_travel_s=120)
    schedule, rows, times = [], [], {}
    for jid, k, age in [('x', 0, 65), ('y', 0, 70), ('z', 1, 100)]:
        bid, flow, _ = plan['stops'][k]
        schedule.append(dict(job_id=jid, block=bid, flow=flow, arrival_s=gate, visit=dict(plan, stop=k)))
        rows.append(dict(job=jid, gate=gate, finish=gate+age*60, day=0))
        times[jid] = SimpleNamespace(block_arrival=gate+(73*60 if k else 60),
                                     job_done=gate+age*60, gate_out=gate+102*60)
    plans = visit_plans(schedule)
    visit, = truck_visits(plans, rows, times, gate+120*60)
    assert (visit['day'], visit['orders'], visit['seconds']) == (0, 3, 100*60)
    blocks = {e['job_id']: e['block'] for e in schedule}
    segments = stop_intervals(plans, rows, blocks)
    assert segments['b'] == [(gate, gate+70*60, gate+100*60)]
    clocks = [StopGateAgeClock(v) for v in segments.values()]
    for age in (0, 59, 60, 65, 70, 75, 90, 100, 120):
        assert sum(c.minutes_at(gate+age*60) for c in clocks) == pytest.approx(truck_points(min(age, 100)))
    times['z'].job_done = times['z'].gate_out = None
    pending, = truck_visits(plans, rows, times, gate+80*60)
    assert pending['finish'] is None and pending['seconds'] == 80*60
    times['z'].block_arrival = gate+65*60
    with pytest.raises(ValueError, match='preceding completion'):
        truck_visits(plans, rows, times, gate+80*60)
    schedule[-1]['visit']['stop'] = 0
    with pytest.raises(ValueError, match='inconsistent'):
        visit_plans(schedule)
