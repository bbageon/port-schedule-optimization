"""A physical lower bound must not exceed a completed, feasible vessel trace."""
from dataclasses import replace

import pytest

from test_discharge_backpressure import discharge_fixture, finish
from yard_rl.v7.world.integrated.engine import TerminalSimulator
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.world.integrated.scenario_gen import phys_min_completion_s
from yard_rl.v7.world.integrated.vessel import VesselWorkType


@pytest.mark.parametrize('moves', [1, 2, 3])
def test_short_discharge_can_finish_before_first_vehicle_returns(moves):
    profile, scenario = discharge_fixture(moves=moves)
    profile = replace(profile, transfer=build_h21_profile().transfer)
    cadence = 3600./27.5
    scenario.vessels[0].plan = replace(scenario.vessels[0].plan, sts_move_interval_s=cadence)
    sim = TerminalSimulator(profile, scenario, yard_handover_cap=3)
    finish(sim)
    actual = sim.vessels['V0'].truth.actual_completion_s
    lower = phys_min_completion_s(profile, work=VesselWorkType.DISCHARGE,
                                 start_s=0., moves=moves, cadence_s=cadence)
    assert actual == pytest.approx(moves*cadence)
    assert lower <= actual+1e-8


@pytest.mark.parametrize('units', [1, 3])
@pytest.mark.parametrize('backpressure', [False, True])
@pytest.mark.parametrize('delay', [0, 1200])
def test_discharge_bound_survives_shared_fleet_and_yard_delay(units, backpressure, delay):
    profile, scenario = discharge_fixture(ships=2, moves=8, units=units, yard_delay=delay)
    profile = replace(profile, transfer=replace(profile.transfer, move_time_s=470.))
    # One vehicle must finish 16 full transport/stack cycles before comparing completion.
    # The original 8,000 s fixture truncates this intentionally slow configuration.
    scenario = replace(scenario, horizon_s=16000.)
    sim = TerminalSimulator(profile, scenario,
                            yard_handover_cap=units if backpressure else None)
    finish(sim)
    assert sim.unfinished_backlog() == 0
    for vessel in sim.vessels.values():
        lower = phys_min_completion_s(profile, work=vessel.work_type, start_s=0.,
                                     moves=8, cadence_s=vessel.plan.sts_move_interval_s)
        assert lower <= vessel.truth.actual_completion_s+1e-8


def test_lower_bound_is_not_a_claim_of_achievable_deadline():
    profile, scenario = discharge_fixture(moves=8, yard_delay=1200)
    sim = TerminalSimulator(profile, scenario, yard_handover_cap=1)
    finish(sim)
    bound = phys_min_completion_s(profile, work=VesselWorkType.DISCHARGE,
                                 start_s=0., moves=8, cadence_s=20.)
    assert sim.vessels['V0'].truth.actual_completion_s > bound
