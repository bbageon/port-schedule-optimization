import copy
from collections import Counter

import pytest

from yard_rl.v7.day_mix import add_daily_mix
from yard_rl.v7.demand_profile import within_hour_arrivals, validate_within_hour


def test_mix_keeps_joint_parents_and_training_only_control():
    rows = [dict(parents=['a', 'b'], fraction=f, hour_masses=[1 / 24] * 24) for f in (0., .5)]
    bank = dict(splits=dict(train=rows, selection=[dict(rows[0], fraction=1.)], holdout=[rows[0]]))
    old = copy.deepcopy(bank)
    mixed = add_daily_mix(bank, [dict(date='a', gate_out_share=.4), dict(date='b', gate_out_share=.8)])
    assert bank == old
    assert [r['gate_out_share'] for r in mixed['splits']['train']] == pytest.approx([.4, .6])
    assert mixed['fixed_gate_out_share'] == pytest.approx(.5)
    assert mixed['splits']['selection'][0]['gate_out_share'] == .8


def test_parametric_arrivals_preserve_counts_replay_and_no_observed_arrays():
    times = [10. + i * 30 for i in range(100)] + [4000. + i for i in range(400)]
    spec = dict(method='dirichlet_minutes', alpha=2.)
    first, meta = within_hour_arrivals(120, times, spec)
    assert (first, meta) == within_hour_arrivals(120, times, spec)
    assert first != within_hour_arrivals(121, times, spec)[0]
    assert Counter(int(x // 3600) for x in first) == Counter(int(x // 3600) for x in times)
    assert first == sorted(first) and len(first) == len(times)
    assert all(0 <= x < 86400 for x in first)
    with pytest.raises(ValueError):
        validate_within_hour(dict(spec, minute_counts={}))


@pytest.mark.parametrize('alpha', [0, -1, float('nan'), float('inf'), None])
def test_invalid_concentration(alpha):
    with pytest.raises(ValueError):
        validate_within_hour(dict(method='dirichlet_minutes', alpha=alpha))
