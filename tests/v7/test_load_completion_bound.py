"""LOAD lower bounds must hold for finite fleets, including their first batch."""
import heapq
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v7'))
from audit_load_bound import fixture, run_case
from yard_rl.v7.world.integrated.scenario_gen import phys_min_completion_s, _min_yc_cycle_s
from yard_rl.v7.world.integrated.transfer import TransferFleet
from yard_rl.v7.world.integrated.vessel import VesselWorkType


@pytest.mark.parametrize('moves', [2, 3, 8, 32])
def test_load_can_finish_before_continuous_average_supply_bound(moves):
    result = run_case(units=3, moves=moves)
    for v in result['vessels']:
        assert v['with_targets_s'] <= v['actual_completion_s'] + 1e-8


def relaxed_finish(units, transfer, cadence, moves, earliest, start):
    """No yard contention: all boxes request the real fleet at earliest time."""
    fleet = TransferFleet('relaxed', 'YT', units, transfer)
    events, arrivals = [], []
    for k in range(moves):
        at = fleet.request(earliest, str(k))
        if at is not None:
            heapq.heappush(events, (at, str(k)))
    while events:
        at, _ = heapq.heappop(events)
        arrivals.append(at)
        nxt = fleet.dispatch_pending(at)
        if nxt:
            heapq.heappush(events, nxt)
    assert len(arrivals) == moves and fleet.waiting_count() == 0
    end = start
    for at in arrivals:
        end = max(end + cadence, at)
    return end


@pytest.mark.parametrize('units', [1, 2, 3, 6])
@pytest.mark.parametrize('transfer', [30., 470.])
@pytest.mark.parametrize('cadence', [20., 3600./27.5, 600.])
def test_bound_matches_independent_relaxed_event_model(units, transfer, cadence):
    profile, scenario = fixture(units=units, moves=1)
    profile = replace(profile, transfer=replace(profile.transfer, move_time_s=transfer))
    targets = list(scenario.containers.values())
    earliest = 600. + _min_yc_cycle_s(profile, targets[0])
    for moves in (1, 2, 3, 4, 8, 32):
        actual = relaxed_finish(units, transfer, cadence, moves, earliest, start=600.)
        lower = phys_min_completion_s(profile, work=VesselWorkType.LOAD, start_s=600.,
            moves=moves, cadence_s=cadence, load_targets=targets * moves)
        assert lower == pytest.approx(actual, abs=1e-8)
        without = phys_min_completion_s(profile, work=VesselWorkType.LOAD, start_s=600.,
                                        moves=moves, cadence_s=cadence)
        assert without <= actual + 1e-8
