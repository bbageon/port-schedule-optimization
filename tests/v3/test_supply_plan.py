from copy import deepcopy
import pytest

from yard_rl.v3.stage.supply_plan import balance_vessel_supply


def stream(key, moves, work='LOAD', start=0):
    return dict(key=key, block='Y01', moves=moves, work=work,
                type_offset=0 if work == 'DISCHARGE' else 1, start_s=start,
                cadence_s=60, ship='S1')


def test_negative_stock_corrected_without_changing_volume_time_or_truck_input():
    original = {0: [stream('A', 4), stream('B', 3, start=60)]}
    trucks = [dict(block='Y01', flow='GATE_OUT', arrival_s=100)]
    saved = deepcopy((original, trucks))
    revised, report = balance_vessel_supply(original, trucks, {'Y01': 5}, {'Y01': 12})
    assert (original, trucks) == saved
    assert report['original_end_balance']['Y01'] == -3
    assert report['corrected_end_balance']['Y01'] == 5
    assert [x['key'] for x in report['changes']] == ['A']
    for old, new in zip(original[0], revised[0]):
        assert {k:v for k,v in old.items() if k not in ('work','type_offset')} == {
            k:v for k,v in new.items() if k not in ('work','type_offset')}
    assert not report['timely_service_proven']


def test_valid_original_is_unchanged_even_if_another_balance_is_closer_to_initial():
    original = {0: [stream('A', 3)]}
    revised, report = balance_vessel_supply(original, [], {'Y01': 5}, {'Y01': 12})
    assert revised == original and report['changes'] == []


def test_infeasible_input_fails_without_clipping_or_creating_stock():
    original = {0: [stream('A', 100)]}
    with pytest.raises(ValueError, match='no feasible'):
        balance_vessel_supply(original, [], {'Y01': 5}, {'Y01': 12})
    assert original[0][0]['work'] == 'LOAD'


def test_exact_subset_and_deterministic_tie_break():
    original = {0: [stream('A', 4), stream('B', 4, start=60), stream('C', 4, start=120)]}
    revised, report = balance_vessel_supply(original, [], {'Y01': 1}, {'Y01': 6})
    assert report['corrected_end_balance']['Y01'] == 5
    assert [x['key'] for x in report['changes']] == ['A','B']
    assert balance_vessel_supply(original, [], {'Y01': 1}, {'Y01': 6}) == (revised, report)


def test_excess_stock_uses_opposite_direction_without_adding_vessel_moves():
    original = {0: [stream('A', 4, 'DISCHARGE'), stream('B', 3, 'DISCHARGE')]}
    revised, report = balance_vessel_supply(original, [], {'Y01': 5}, {'Y01': 8})
    assert report['corrected_end_balance']['Y01'] == 4
    assert report['changes'][0]['key'] == 'A'
    assert sum(x['moves'] for x in revised[0]) == 7


def test_supply_mode_cannot_be_silently_ignored_by_legacy_admission():
    from yard_rl.v3.stage.month_run import run_month
    with pytest.raises(ValueError, match='requires PRESERVE'):
        run_month(seed=1, supply_mode='COUNT_BALANCED')
    with pytest.raises(ValueError, match='supply_mode must'):
        run_month(seed=1, supply_mode='TYPO')


def test_invalid_geometry_is_rejected_even_when_final_stock_would_fit():
    with pytest.raises(ValueError, match='initial stock'):
        balance_vessel_supply({0:[stream('A',3)]}, [], {'Y01':10}, {'Y01':8})
