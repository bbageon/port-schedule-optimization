"""YR-332-h: yard delay must reach discharge and the existing cost ledger."""
import copy
from dataclasses import replace

import pytest

from yard_rl.v7.reward.phi import terminal_cost_krw
from yard_rl.v7.stage.episode import _sim_from
from yard_rl.v7.world.domain.enums import ContainerSize, JobFlow, JobStatus, LoadStatus
from yard_rl.v7.world.domain.models import Job
from yard_rl.v7.world.integrated.baselines import ResolverPolicy, ServiceFirstSPTPreference, run_joint_episode, _apply
from yard_rl.v7.world.integrated.candidates import CandidateGenerator
from yard_rl.v7.world.integrated.cost_config import RewardCalculator
from yard_rl.v7.world.integrated.engine import TerminalSimulator, TerminalDecision
from yard_rl.v7.world.integrated.fixtures import build_integrated_profile
from yard_rl.v7.world.integrated.scenario import InjectedEvent, TerminalScenario
from yard_rl.v7.world.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType


def discharge_fixture(*, ships=1, moves=8, yard_delay=0, units=3):
    p = build_integrated_profile()
    p = replace(p, cranes=(p.cranes[0],), transfer=replace(p.transfer, n_units=units, move_time_s=30.0))
    jobs, vessels = [], []
    for i in range(ships):
        vid = f"V{i}"
        vessels.append(VesselProcess(vid, VesselWorkType.DISCHARGE,
            VesselPlan(0., 7000., None, 8000., moves, 20., quay_buffer_cap=3)))
        jobs.extend(Job(job_id=f"{vid}-{j:03}", flow=JobFlow.VESSEL_DISCHARGE,
            release_time=0., actual_gate_in=None, actual_block_arrival=None,
            target_container=None, inbound_size=ContainerSize.FT40,
            inbound_load=LoadStatus.FULL, vessel_id=vid) for j in range(moves))
    events = [] if not yard_delay else [InjectedEvent(0., "EQUIPMENT_DOWN", "YC-A"),
                                      InjectedEvent(float(yard_delay), "EQUIPMENT_UP", "YC-A")]
    return p, TerminalScenario("discharge-causal-probe", 0, 8000., 0., {}, jobs, vessels, events)


class TracedSimulator(TerminalSimulator):
    def reset(self):
        super().reset()
        self.peak = 0

    def _sts_move(self, vid):
        super()._sts_move(vid)
        self.peak = max(self.peak, sum(self._discharge_pipeline.values()))


def run_discharge(*, cap, **kwargs):
    p, sc = discharge_fixture(**kwargs)
    sim = TracedSimulator(p, sc, yard_handover_cap=cap)
    finish(sim)
    return sim


def finish(sim):
    if sim._pending:
        dp = TerminalDecision(sim.clock, sim._pending)
        gen = CandidateGenerator()
        gb = {c: gen.generate(sim, c, sim.info_level) for c in dp.crane_ids}
        _apply(sim, ResolverPolicy(ServiceFirstSPTPreference()).decide(sim, dp, gb))
    run_joint_episode(sim, ResolverPolicy(ServiceFirstSPTPreference()),
                      RewardCalculator.numeraire_v1())
    return sim


def result(sim):
    idle = sum(v.sts_wait_accum_s for v in sim.vessels.values())
    cost = terminal_cost_krw({}, end_s=sim.clock,
        vessel_idle={vid: (50_000., v.sts_wait_accum_s) for vid, v in sim.vessels.items()})
    return dict(idle_s=idle, vessel_cost_krw=cost.vessel,
        completion_s={vid: v.truth.actual_completion_s for vid, v in sim.vessels.items()},
        peak=sim.peak, remaining=sum(j.status != JobStatus.DONE for j in sim.jobs.values()),
        pipeline_remaining=sum(sim._discharge_pipeline.values()))


def test_yard_delay_reaches_vessel_and_reward_only_with_connection():
    off_fast = result(run_discharge(cap=None))
    off_slow = result(run_discharge(cap=None, yard_delay=600))
    on_fast = result(run_discharge(cap=3))
    on_slow = result(run_discharge(cap=3, yard_delay=600))
    assert off_fast['completion_s'] == off_slow['completion_s']
    assert off_fast['idle_s'] == off_slow['idle_s']
    assert on_slow['completion_s']['V0'] > on_fast['completion_s']['V0']
    assert on_slow['idle_s'] > on_fast['idle_s']
    assert on_slow['vessel_cost_krw'] > on_fast['vessel_cost_krw']
    assert on_slow['remaining'] == on_slow['pipeline_remaining'] == 0


@pytest.mark.parametrize('cap', [1, 3, 6])
def test_shared_capacity_and_other_ship_resume(cap):
    sim = run_discharge(cap=cap, ships=2, yard_delay=600)
    assert 0 < sim.peak <= cap
    assert all(v.done and v.buffer_level == 0 for v in sim.vessels.values())
    assert all(j.status == JobStatus.DONE for j in sim.jobs.values())
    assert len(sim.stacks.containers) == 16
    assert sum(sim._discharge_pipeline.values()) == 0
    assert sim.transfer.waiting_count() == 0


def test_in_transit_arrival_does_not_release_capacity_before_yard_completion():
    p, sc = discharge_fixture(yard_delay=600)
    sim = TerminalSimulator(p, sc, yard_handover_cap=1)
    sim.run_until_decision()
    assert sum(sim._discharge_pipeline.values()) == 1
    assert not any(j.status == JobStatus.DONE for j in sim.jobs.values())
    assert sim.vessels['V0'].remaining_moves == 7
    assert sim.vessels['V0'].sts_wait_accum_s > 0


def test_snapshot_and_reset_preserve_configuration_and_conservation():
    p, sc = discharge_fixture(ships=2, yard_delay=600)
    sim = TracedSimulator(p, sc, yard_handover_cap=3)
    sim.run_until_decision()
    twin = copy.deepcopy(sim)
    finish(sim)
    finish(twin)
    assert result(sim) == result(twin)
    assert sim.event_log == twin.event_log
    twin.reset()
    assert twin.yard_handover_cap == 3 and twin._discharge_pipeline == {}


def test_stage_default_tracks_actual_fleet_and_legacy_off_is_explicit():
    p, sc = discharge_fixture(units=5)
    assert _sim_from(sc, p).yard_handover_cap == 5
    assert _sim_from(sc, p, discharge_backpressure=False).yard_handover_cap is None


@pytest.mark.parametrize('cap', [0, -1, 1.5, True])
def test_invalid_capacity_fails_before_simulation(cap):
    p, sc = discharge_fixture()
    with pytest.raises(ValueError, match='positive integer'):
        TerminalSimulator(p, sc, yard_handover_cap=cap)
