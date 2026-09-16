from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from yard_rl.v3.stage.daily_observation import DailyObserver, action_mix, snapshot
from yard_rl.v3.stage.month import DAY_S, plan_days
from yard_rl.v3.world.domain.enums import JobFlow, JobStatus
from yard_rl.v3.world.domain.models import Job


def world():
    waiting = Job('wait', JobFlow.GATE_OUT, 0, 0, 10, status=JobStatus.WAITING)
    future = Job('future', JobFlow.GATE_IN, 0, 100, 110)
    running = Job('run', JobFlow.GATE_IN, 0, 0, 0,
                  status=JobStatus.RUNNING, service_start=5)
    vessel = Job('vessel', JobFlow.VESSEL_DISCHARGE, float('inf'), None, None)
    ready = Job('ready', JobFlow.VESSEL_LOAD, 0, None, None, status=JobStatus.RELEASED)
    done = Job('done', JobFlow.GATE_OUT, 0, 0, 0, status=JobStatus.DONE, service_end=8)
    crane = NS(state=NS(available_at=70.0))
    a = NS(jobs={j.job_id:j for j in [waiting, future, running, vessel, ready, done]},
           kpis=NS(queue_area_s=0.0), fleet=NS(all=lambda: [crane]))
    b = NS(jobs={}, kpis=NS(queue_area_s=0.0), fleet=NS(all=lambda: []))
    return NS(blocks={'Y01': a, 'Y02': b})


def bridge():
    return NS(n_space=0, n_time=0, market=NS(seller=NS(trail=[])))


def test_physical_queue_excludes_future_running_done_and_counts_unreleased_work():
    m = world()
    before = deepcopy(m.blocks['Y01'].jobs)
    row = snapshot(m, 20)
    assert m.blocks['Y01'].jobs == before  # observation has no state effect
    a = row['blocks']['Y01']
    assert a['truck_queue'] == 1
    assert a['yard_jobs_unfinished'] == 5
    assert a['yard_jobs_not_released'] == 2
    assert a['yard_jobs_released_unfinished'] == 3
    assert a['yard_jobs_in_service'] == 1
    assert a['crane_committed_remaining_s'] == 50
    assert row['total'] == a


def test_action_denominator_is_committed_changes_and_empty_is_null():
    zero = dict(spatial=0, temporal=0, decisions=0)
    assert action_mix(zero, zero)['spatial_share'] is None
    out = action_mix(zero, dict(spatial=3, temporal=1, decisions=20))
    assert out['spatial_share'] == .75 and out['temporal_share'] == .25


def test_midnight_carryover_exact_integral_and_arrival_date_are_distinct():
    m, b, samples = world(), bridge(), []
    days = plan_days(9900720, (3, 3))
    obs = DailyObserver(days, seed=9900720, arm='RL', sample_s=43200, on_sample=samples.append)
    obs.observe(m, 0, b)
    obs.observe(m, 43200, b)
    m.blocks['Y01'].kpis.queue_area_s = DAY_S * 2
    b.n_space, b.n_time = 3, 1
    obs.observe(m, DAY_S, b)
    assert obs.rows[0]['queue']['truck_queue_mean'] == 2
    assert obs.rows[0]['actions']['spatial_share'] == .75
    assert obs.rows[0]['end'] == obs.starts[1]
    # This decision at the boundary belongs to the new day, not the old day.
    b.n_time += 2
    obs.observe(m, DAY_S, b)  # duplicate review must not overwrite the start
    obs.observe(m, DAY_S + 43200, b)
    obs.observe(m, DAY_S * 2, b)
    obs.observe(m, DAY_S * 2 + 3600, b)  # drain is not another calendar day
    assert obs.rows[1]['actions']['spatial'] == 0
    assert obs.rows[1]['actions']['temporal'] == 2
    assert len(samples) == 5
    records = {'a': NS(gate_in_s=DAY_S+10, gate_out_s=DAY_S+30),
               'b': NS(gate_in_s=100, gate_out_s=None),
               'c': NS(gate_in_s=None, gate_out_s=None)}
    schedule = [dict(job_id=k, day=d) for k,d in [('a',0),('b',0),('c',1)]]
    obs.finish(schedule, records, DAY_S*2+7200)
    c = obs.rows[0]['cohort']
    assert c['requested'] == 2 and c['completed_by_cutoff'] == 1 and c['censored_by_cutoff'] == 1
    assert c['actual_gate_entries_on_calendar_day'] == 1
    assert obs.rows[1]['cohort']['actual_gate_entries_on_calendar_day'] == 1
    assert obs.rows[1]['cohort']['not_entered_by_cutoff'] == 1


@pytest.mark.parametrize('interval', [0, -60, 61, 3601, float('nan')])
def test_invalid_sampling_interval_is_rejected(interval):
    with pytest.raises(ValueError):
        DailyObserver(plan_days(1, (3,)), seed=1, arm='RL', sample_s=interval)


def test_missing_state_samples_are_not_silently_reported_complete():
    obs = DailyObserver(plan_days(1, (3,)), seed=1, arm='NO_REALLOC')
    m, b = world(), bridge()
    obs.observe(m, 0, b)
    with pytest.raises(RuntimeError, match='Incomplete'):
        obs.observe(m, DAY_S, b)


def test_monthly_judge_keeps_operating_state_in_day_output():
    from yard_rl.v3.eval.month_judge import _day_row
    from yard_rl.v3.stage.month_run import DayReport
    d = DayReport(index=1, load=3500, label='test', train=True,
                  operational={'start': {'total': {'truck_queue': 10}}})
    assert _day_row(d)['operational'] == d.operational
