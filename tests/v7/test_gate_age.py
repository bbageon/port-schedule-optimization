from math import fsum

import pytest

from yard_rl.v7.reward.gate_age import GateAgeTruckClock, MAX_SURCHARGE, truck_points


@pytest.mark.parametrize(("age", "extra"), ((0, 0), (60, 0), (70, 100/60), (75, 3.75),
                                           (90, 15), (120, 45)))
def test_points_at_gate_age(age, extra):
    assert truck_points(age) == pytest.approx(age + MAX_SURCHARGE * extra)


def test_gate_age_is_preserved_across_block_arrival_and_midnight():
    gate = 23.5 * 3600
    block, finish = gate + 50*60, gate + 70*60
    clock = GateAgeTruckClock([(gate, finish)])
    assert clock.minutes_at(block) == pytest.approx(50)
    assert clock.minutes_at(finish) - clock.minutes_at(block) == pytest.approx(20 + MAX_SURCHARGE*100/60)
    assert clock.minutes_at(finish + 1800) == pytest.approx(truck_points(70))


def test_window_charges_each_overlapping_truck_only_for_new_time():
    intervals = [(0, 70*60), (600, 600 + 10800), (0, 3600), (3600, 3600), (3700, 9100)]
    clock = GateAgeTruckClock(intervals)
    for t in [0, 600, 3600, 4200, 5400, 6100, 9100, 2500000]:
        direct = fsum(truck_points(max(0, min(t, finish) - gate)/60) for gate, finish in intervals)
        assert clock.minutes_at(t) == pytest.approx(direct)
    assert clock.total_minutes == pytest.approx(fsum(truck_points((f-g)/60) for g, f in intervals))
    assert GateAgeTruckClock([]).minutes_at(123) == 0
