"""Physical end-transfer checks; these small fixtures are not performance runs."""
import copy
from dataclasses import replace

import pytest

from yard_rl.v3.layouts.runtime import BoundedYardStacks, VerticalTerminalSimulator
from yard_rl.v3.world.contract.schema import CandidateKind
from yard_rl.v3.world.domain.enums import (ContainerSize, InformationLevel, JobFlow,
                                           JobStatus, LoadStatus)
from yard_rl.v3.world.domain.models import Container, Job
from yard_rl.v3.world.integrated.baselines import (ResolverPolicy,
    ServiceFirstSPTPreference, _apply)
from yard_rl.v3.world.integrated.candidates import CandidateGenerator
from yard_rl.v3.world.integrated.engine import CraneAssignment, TerminalSimulator
from yard_rl.v3.world.integrated.jobplan import JobRef
from yard_rl.v3.world.integrated.ledger import assert_ledger_identity
from yard_rl.v3.world.integrated.policy_config import LEGACY_DEFAULT
from yard_rl.v3.world.integrated.profiles import build_h21_profile
from yard_rl.v3.world.integrated.scenario import TerminalScenario
from yard_rl.v3.world.integrated.vessel import VesselPlan, VesselProcess, VesselWorkType
from yard_rl.v3.world.sim.constraints import ConstraintViolation
from yard_rl.v3.world.sim.travel_time import move_container


def profile():
    p = build_h21_profile()
    return replace(p, block=replace(p.block, bay_count=6, row_count=2, tier_max=2),
        cranes=tuple(replace(c, service_bay_min=1, service_bay_max=6) for c in p.cranes),
        transfer=replace(p.transfer, n_units=1, move_time_s=5))


def box(cid, bay, row=1, tier=1):
    return Container(cid, ContainerSize.FT20, LoadStatus.FULL, 'B', bay, row, tier)


def job(jid, flow, target=None, *, vessel=None, arrival=0):
    external = flow in (JobFlow.GATE_IN, JobFlow.GATE_OUT)
    store = flow in (JobFlow.GATE_IN, JobFlow.VESSEL_DISCHARGE)
    return Job(jid, flow, release_time=0, actual_gate_in=arrival if external else None,
        actual_block_arrival=arrival if external else None, target_container=target,
        inbound_size=ContainerSize.FT20 if store else None,
        inbound_load=LoadStatus.FULL if store else None, vessel_id=vessel,
        exit_travel_s=10 if external else None)


def vessel(vid, kind, moves=1):
    return VesselProcess(vid, kind, VesselPlan(0, 9000, None, 10000, moves, 2))


def scenario(jobs=(), containers=(), vessels=()):
    return TerminalScenario('vertical-fixture', 17, 8000, 2000,
        {c.container_id: c for c in containers}, list(jobs), list(vessels))


def run(sim):
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    policy = ResolverPolicy(ServiceFirstSPTPreference(), 'SF_SPT')
    plans = {}
    for _ in range(2000):
        decision = sim.run_until_decision()
        if decision is None:
            return list(plans.values())
        choices = {c: gen.generate(sim, c, InformationLevel.PRE_ADVICE)
                   for c in decision.crane_ids}
        _apply(sim, policy.decide(sim, decision, choices))
        for plan in sim._active_plans.values():
            plans[(plan.crane_id, plan.job_id, plan.start_s)] = plan
        sim.check_invariants()
    raise AssertionError('Small physical fixture failed to terminate')


def ref_for(sim, jid, cid):
    return sim._jobref(sim.jobs[jid], sim.fleet.spec(cid), sim.fleet.get(cid))


@pytest.mark.parametrize('flow,cid,end', [
    (JobFlow.GATE_IN, 'YC-L', 0), (JobFlow.GATE_OUT, 'YC-L', 0),
    (JobFlow.VESSEL_DISCHARGE, 'YC-W', 7), (JobFlow.VESSEL_LOAD, 'YC-W', 7)])
def test_end_transfer_changes_actual_moves_time_distance_and_corridor(flow, cid, end):
    target = None if flow in (JobFlow.GATE_IN, JobFlow.VESSEL_DISCHARGE) else 'target'
    sim = VerticalTerminalSimulator(profile(), scenario([job('j', flow, target)],
                                                        [box('target', 3)]))
    sim.jobs['j'].status = JobStatus.WAITING
    ref = ref_for(sim, 'j', cid)
    plan = sim._plan(cid, ref)
    move = plan.moves[-1]
    endpoint = move.src if target is None else move.dst
    assert endpoint == (end, 1.5, 1)
    state = sim.fleet.get(cid).state
    expected = move_container(sim.fleet.spec(cid), sim.profile.block,
                              state.position_bay, state.trolley_row, move.src, move.dst)
    positioning = sim.fleet.spec(cid).truck_positioning_time_s if sim.jobs['j'].is_external_truck else 0
    assert plan.duration_s == pytest.approx(expected.duration_s + positioning)
    assert plan.loaded_gantry_m == pytest.approx(expected.loaded_gantry_m)
    assert plan.loaded_gantry_m > 0
    assert plan.empty_gantry_m == pytest.approx(expected.empty_gantry_m)
    assert plan.corridor[0] <= end <= plan.corridor[1]
    assert (plan.end_bay, plan.end_row) == (expected.end_bay, expected.end_row)
    assert plan.lane_id == ('LANDSIDE' if cid == 'YC-L' else 'WATERSIDE')
    assert ref.eligible_crane_ids == (cid,)


def test_wrong_role_is_rejected_in_candidates_planning_and_direct_commit():
    sim = VerticalTerminalSimulator(profile(), scenario([job('j', JobFlow.GATE_OUT, 'c')],
                                                        [box('c', 3)]))
    sim.jobs['j'].status = JobStatus.WAITING
    ref = ref_for(sim, 'j', 'YC-L')
    assert not sim._dispatchable(sim.jobs['j'], 'YC-W')
    assert ref_for(sim, 'j', 'YC-W') is None
    assert sim._plan('YC-W', ref) is None
    assert sim._plan('YC-W', replace(ref, kind=CandidateKind.PRE_REHANDLE)) is None
    assert not sim.candidates_for('YC-W')
    sim._pending = ('YC-W',)
    with pytest.raises(ConstraintViolation, match='NOT_DISPATCHABLE'):
        sim.assign('YC-W', CraneAssignment('YC-W', CandidateKind.SERVE, ref))
    assert not sim.reservations.active() and not sim._active_plans


def test_parking_does_not_add_storage_or_rehandle_capacity():
    p = profile()
    containers = [box(f'{b}-{r}-{t}', b, r, t)
                  for b in range(1, 7) for r in (1, 2) for t in (1, 2)]
    sim = VerticalTerminalSimulator(p, scenario(containers=containers))
    spec = sim.profile.cranes[0]
    assert (spec.service_bay_min, spec.service_bay_max) == (-2, 9)
    assert isinstance(sim.stacks, BoundedYardStacks)
    assert sim.stacks.find_slot(ContainerSize.FT20, spec, -2, 0) is None
    assert not sim.stacks.rehandle_capacity_ok('1-1-1', spec)
    with pytest.raises(ConstraintViolation, match='INVALID_SLOT'):
        sim.stacks.place(box('bad', 0), 0, 1)
    assert len(sim.stacks.containers) == 24
    assert p.cranes[0].service_bay_min == 1 and p.cranes[0].service_bay_max == 6
    assert sim.input_profile == p


@pytest.mark.parametrize('flow,target_bay,worker,blocker,position,direction', [
    (JobFlow.VESSEL_LOAD, 1, 'YC-W', 'YC-L', 0, -1),
    (JobFlow.GATE_OUT, 6, 'YC-L', 'YC-W', 7, 1)])
def test_retreat_reopens_opposite_end_stock_and_keeps_paid_nonpassing_motion(
        flow, target_bay, worker, blocker, position, direction):
    vessels = [vessel('V', VesselWorkType.LOAD)] if flow == JobFlow.VESSEL_LOAD else []
    sim = VerticalTerminalSimulator(profile(), scenario(
        [job('retrieve', flow, 'c', vessel='V' if vessels else None)],
        [box('c', target_bay)], vessels), enable_cost_ledger=True)
    # Idle cranes at either interface obstruct stock near that end. The
    # existing escape policy must make a paid move to open the corridor.
    sim.fleet.get(blocker).state.position_bay = position
    sim.reservations.set_idle_position(blocker, position)
    sim.jobs['retrieve'].status = JobStatus.RELEASED
    candidate = ref_for(sim, 'retrieve', worker)
    plan = sim._plan(worker, candidate)
    assert sim.reservations.reject_reason(sim._reservation(plan)) == 'CRANE_INTERFERENCE'
    plans = run(sim)
    retreats = [p for p in plans if p.kind == CandidateKind.REPOSITION]
    assert retreats and sum(p.empty_gantry_m for p in retreats) > 0
    assert any(p.crane_id == blocker and direction * (p.end_bay - position) > 0
               for p in retreats)
    assert all(p.duration_s > 0 for p in retreats)
    assert sim.jobs['retrieve'].status == JobStatus.DONE
    assert all(v.done for v in sim.vessels.values())
    sim.check_invariants()
    assert_ledger_identity(sim.cost)


@pytest.mark.parametrize('slot', [(0, 1), (7, 1), (1, 0), (1.5, 1)])
def test_custom_store_selector_cannot_bypass_stock_bounds(slot):
    sim = VerticalTerminalSimulator(profile(), scenario([job('in', JobFlow.GATE_IN)]))
    sim.store_slot_selector = lambda *args: slot
    with pytest.raises((RuntimeError, ConstraintViolation)):
        ref_for(sim, 'in', 'YC-L')
    assert not sim.stacks.containers and not sim.reservations.active()


def test_all_four_flows_finish_with_rehandles_conservation_and_time_cost_ledgers():
    jobs = [job('in', JobFlow.GATE_IN), job('out', JobFlow.GATE_OUT, 'CG'),
            job('load', JobFlow.VESSEL_LOAD, 'CV', vessel='VL'),
            job('discharge', JobFlow.VESSEL_DISCHARGE, vessel='VD')]
    sim = VerticalTerminalSimulator(profile(), scenario(jobs,
        [box('CG', 6), box('blocker', 6, 1, 2), box('CV', 1, 2)],
        [vessel('VL', VesselWorkType.LOAD), vessel('VD', VesselWorkType.DISCHARGE)]),
        enable_cost_ledger=True)
    plans = run(sim)
    assert all(j.status == JobStatus.DONE for j in sim.jobs.values())
    assert all(v.done for v in sim.vessels.values())
    assert set(sim.stacks.containers) == {'blocker', 'IN_in', 'IN_discharge'}
    assert sum(p.rehandles for p in plans) >= 1
    assert not sim.reservations.active() and sim.unfinished_backlog() == 0
    for jid in ('in', 'out'):
        record = sim.time_ledger.records[jid]
        assert record.gate_in <= record.block_arrival <= record.service_start < record.job_done
        assert record.gate_out == pytest.approx(record.job_done + 10)
    expected_wait = sum(r.job_done - r.block_arrival for r in sim.time_ledger.records.values())
    assert sim.time_ledger.block_area_s == pytest.approx(expected_wait)
    totals = sim.cost.ledger.term_totals()
    assert totals['crane_travel'] == pytest.approx(sum(p.loaded_gantry_m for p in plans))
    assert totals['empty_travel'] == pytest.approx(sum(p.empty_gantry_m for p in plans))
    assert_ledger_identity(sim.cost)


def test_discharge_capacity_is_shared_and_other_vessel_wakes_after_storage():
    class Observed(VerticalTerminalSimulator):
        def _sts_move(self, vid):
            super()._sts_move(vid)
            self.observed.append((self.clock, sum(self._discharge_pipeline.values())))
    jobs = [job(f'{v}-{i}', JobFlow.VESSEL_DISCHARGE, vessel=v)
            for v in ('VA', 'VB') for i in range(2)]
    sim = Observed(profile(), scenario(jobs, vessels=[
        vessel('VA', VesselWorkType.DISCHARGE, 2),
        vessel('VB', VesselWorkType.DISCHARGE, 2)]), yard_handover_cap=1)
    sim.observed = []
    run(sim)
    assert max(n for _, n in sim.observed) == 1
    assert sum(sim._discharge_pipeline.values()) == 0
    assert len(sim.stacks.containers) == 4
    assert all(j.status == JobStatus.DONE for j in sim.jobs.values())
    assert all(v.done and v.buffer_level == 0 for v in sim.vessels.values())
    assert sim.transfer.waiting_count() == 0


def test_copy_and_reset_preserve_runtime_class_roles_and_bounded_stacks():
    sim = VerticalTerminalSimulator(profile(), scenario([job('in', JobFlow.GATE_IN)]))
    twin = copy.deepcopy(sim)
    assert isinstance(twin, VerticalTerminalSimulator)
    assert isinstance(twin.stacks, BoundedYardStacks)
    assert twin.profile == sim.profile
    assert run(twin) and sim.jobs['in'].status == JobStatus.PLANNED
    twin.reset()
    assert twin.jobs['in'].status == JobStatus.PLANNED
    assert [twin.fleet.get(cid).state.position_bay for cid in twin._rail_order] == [-2, 9]
    assert not twin.stacks.containers


def test_horizontal_base_runtime_stays_unextended():
    p = profile()
    sim = TerminalSimulator(p, scenario([job('in', JobFlow.GATE_IN)]))
    sim.jobs['in'].status = JobStatus.WAITING
    ref = sim._jobref(sim.jobs['in'], sim.fleet.spec('YC-L'), sim.fleet.get('YC-L'))
    plan = sim._plan('YC-L', ref)
    assert plan.moves[0].src[0] == plan.moves[0].dst[0]
    assert plan.loaded_gantry_m == 0
    assert sim.profile == p and not isinstance(sim.stacks, BoundedYardStacks)
