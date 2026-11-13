"""YR-332-ak: within-hour arrival redraw changes only gaps inside each clock hour."""
from collections import Counter
import gzip
import json
from pathlib import Path

import pytest

from yard_rl.v7.demand_profile import SCHEMA, hourly_arrivals, validate, within_hour_arrivals
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.profiles import build_h21_profile

AJ = Path('outputs/reports/yr332_order_representativeness')
INFO = Path('outputs/reports/yr332_real_order_information/profile.json')
# Arrival-derived fields; every other schedule field must be identical.
TIME_FIELDS = {'arrival_s', 'lead_s', 'reserved_s', 'eta_offset_s'}


def template(counts_by_hour):
    return dict(method='observed_visit_minutes', dates=['d0', 'd1'],
                minute_counts={str(h): counts_by_hour(h) for h in range(24)})


def aj_profile():
    return json.loads((AJ / 'demand-profiles.json').read_text(encoding='utf-8'))['profiles']['observed_mix_hours']


@pytest.mark.parametrize('spec', [dict(method='burst'), dict(method='poisson', extra=1),
    dict(method='observed_visit_minutes', dates=['a'], minute_counts={str(h): [[1] * 60] for h in range(23)}),
    dict(method='observed_visit_minutes', dates=['a'], minute_counts={str(h): [[1] * 59] for h in range(24)}),
    dict(method='observed_visit_minutes', dates=['a'], minute_counts={str(h): [[-1] + [1] * 59] for h in range(24)}),
    dict(method='observed_visit_minutes', dates=['a'], minute_counts={str(h): [[0] * 60] for h in range(24)})])
def test_invalid_within_hour_specs_fail(spec):
    with pytest.raises(ValueError):
        validate(dict(schema=SCHEMA, within_hour=spec), day_s=86400.)


@pytest.mark.parametrize('spec', [dict(method='poisson'),
    template(lambda h: [[1] * 60, [0] * 30 + [2] * 30])])
def test_counts_order_and_determinism(spec):
    base = hourly_arrivals(7, 900, [1] * 12 + [3] * 12)
    times, meta = within_hour_arrivals(7, base, spec)
    assert Counter(int(t // 3600) for t in times) == Counter(int(t // 3600) for t in base)
    assert times == sorted(times) and all(0 <= t < 86400 for t in times)
    assert times == within_hour_arrivals(7, base, spec)[0]
    assert times != within_hour_arrivals(8, base, spec)[0]
    assert times != base and meta['cycled_hours'] == 0


def test_observed_minutes_follow_template_and_stack_whole_dates():
    one_minute = template(lambda h: [[0] * 5 + [3] + [0] * 54, [0] * 5 + [2] + [0] * 54])
    base = [3600. * h + 1800. for h in range(24) for _ in range(5)]
    times, meta = within_hour_arrivals(3, base, one_minute)
    assert all(300 <= t % 3600 < 360 for t in times)
    assert all(m['whole_dates'] == 2 and m['dates_used'] == 2 for m in meta['hours'].values())
    more = [3600. * h + 1800. for h in range(24) for _ in range(9)]
    _, cycled = within_hour_arrivals(3, more, one_minute)
    assert cycled['cycled_hours'] == 24


def test_old_profile_inputs_are_identical_to_aj():
    """Before/after identity: the frozen aj inputs rebuild to the saved digests."""
    info = json.loads(INFO.read_text(encoding='utf-8'))
    for load, seed in ((3500, 9921100), (7500, 9921110), (15000, 9921120)):
        built = build_stage(load=load, seed=seed, profile=build_h21_profile(),
                            information_profile=info, demand_profile=aj_profile())
        assert 'within_hour_arrivals' not in built
        with gzip.open(AJ / 'evaluation' / f'observed_mix_hours-SF_SPT-0-{load}-0.json.gz', 'rt',
                       encoding='utf-8') as stream:
            old = json.load(stream)
        assert old['profile'] == aj_profile()
        assert digest(built['schedule']) == old['schedule_sha256']


def test_redraw_keeps_every_non_time_field_and_information_bounds():
    info = json.loads(INFO.read_text(encoding='utf-8'))
    args = dict(load=1200, seed=9920998, profile=build_h21_profile(), information_profile=info)
    base = build_stage(**args, demand_profile=aj_profile())
    spec = template(lambda h: [[1] * 60, [0] * 50 + [6] * 10])
    for within in (dict(method='poisson'), spec):
        new = build_stage(**args, demand_profile={**aj_profile(), 'within_hour': within})
        assert new['scenarios'] == base['scenarios'] and new['vessel_schedule'] == base['vessel_schedule']
        assert new['flow_fallbacks_total'] == base['flow_fallbacks_total'] == 0
        assert new['within_hour_arrivals']['cycled_hours'] == 0
        hours = lambda b: Counter(int(e['arrival_s'] // 3600) for e in b['schedule'])
        assert hours(new) == hours(base)
        for a, b in zip(base['schedule'], new['schedule']):
            assert {k: v for k, v in a.items() if k not in TIME_FIELDS} == \
                   {k: v for k, v in b.items() if k not in TIME_FIELDS}
            assert int(a['arrival_s'] // 3600) == int(b['arrival_s'] // 3600)
            assert 0 <= b['lead_s'] <= b['arrival_s']
        assert [e['arrival_s'] for e in new['schedule']] == sorted(e['arrival_s'] for e in new['schedule'])
        assert digest(new['schedule']) != digest(base['schedule'])
        assert 'job_done_time' not in new['schedule'][0]
