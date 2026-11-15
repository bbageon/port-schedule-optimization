"""Day-shape changes must not leak other input changes or held-out means."""
import copy
import math

import pytest

from yard_rl.v7.day_shape import basis_hour_masses, build_bank, demand_for_shape, hour_masses


def fits():
    return [dict(date='d0', weights=[.4, .2, .35, .05]),
            dict(date='d1', weights=[.6, .3, .1, 0]),
            dict(date='d2', weights=[.2, .1, .5, .2])]


def test_truncated_components_conserve_probability():
    for row in basis_hour_masses():
        assert len(row) == 24 and min(row) >= 0
        assert math.isclose(sum(row), 1., abs_tol=1e-12)
    with pytest.raises(ValueError):
        hour_masses([.1] * 4)
    with pytest.raises(ValueError):
        hour_masses([float('nan'), 0, 0, 1])


def test_reproducible_disjoint_shapes_and_mean_uses_train_only():
    bank = build_bank(fits(), seed=82)
    assert bank == build_bank(fits(), seed=82)
    bigger = build_bank(fits(), seed=82, sizes=dict(train=24, selection=10, holdout=20))
    assert bank['splits']['train'] == bigger['splits']['train']
    assert bank['fixed_hour_masses'] == bigger['fixed_hour_masses']
    identities = [x['shape_sha256'] for rows in bank['splits'].values() for x in rows]
    assert len(set(identities)) == 36
    for h in range(24):
        assert bank['fixed_hour_masses'][h] == sum(r['hour_masses'][h] for r in bank['splits']['train']) / 24
    for rows in bank['splits'].values():
        for row in rows:
            assert math.isclose(sum(row['hour_masses']), 1., abs_tol=1e-12)
            for i, w in enumerate(row['weights']):
                assert min(r['weights'][i] for r in fits()) <= w <= max(r['weights'][i] for r in fits())


def test_constant_family_reduces_to_fixed_shape():
    data = fits()
    for row in data:
        row['weights'] = [.4, .2, .35, .05]
    bank = build_bank(data, seed=83)
    for row in bank['splits']['train']:
        assert row['hour_masses'] == pytest.approx(bank['fixed_hour_masses'], abs=1e-15)


def test_profile_changes_only_shape_and_provenance_without_mutation():
    base = dict(schema='observed-truck-demand-marginals-v1', gate_out_share=.5375,
                hour_counts=[1] * 24, within_hour=dict(method='poisson'), source=dict(old=True))
    before = copy.deepcopy(base)
    new = demand_for_shape(base, hour_masses(fits()[0]['weights']), source=dict(seed=34))
    assert base == before
    assert {k: v for k, v in new.items() if k not in ('hour_counts', 'source')} == {
        k: v for k, v in base.items() if k not in ('hour_counts', 'source')}
    new['within_hour']['source'] = 'new'
    assert base == before
