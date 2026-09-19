"""Opt-in perpendicular end transfer on the existing nonpassing crane rail.

Stock stays in bays 1..N. Empty cranes can retreat beyond either interface;
those parking bays add travel space, never storage capacity or free movement.
This synthetic layout does not represent a measured terminal installation.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math

from ..world.contract.schema import CandidateKind
from ..world.contract.state import LaneGraph
from ..world.domain.enums import ServiceMode
from ..world.integrated.engine import TerminalSimulator
from ..world.integrated.events import EventKind
from ..world.integrated.vessel import VesselWorkType
from ..world.sim.constraints import ConstraintViolation
from ..world.sim.stack import YardStacks
from ..world.sim.travel_time import move_container


class BoundedYardStacks(YardStacks):
    """Crane travel ranges may exceed the physical stock rectangle."""

    def _stock_spec(self, spec):
        return replace(spec, service_bay_min=max(1, spec.service_bay_min),
                       service_bay_max=min(self.geom.bay_count, spec.service_bay_max))

    def find_slot(self, size, spec, near_bay, near_row, exclude=frozenset()):
        return super().find_slot(size, self._stock_spec(spec), near_bay, near_row,
                                 exclude=exclude)

    def rehandle_capacity_ok(self, target_id, spec):
        return super().rehandle_capacity_ok(target_id, self._stock_spec(spec))

    def place(self, container, bay, row):
        if (type(bay) is not int or type(row) is not int
                or not 1 <= bay <= self.geom.bay_count
                or not 1 <= row <= self.geom.row_count):
            raise ConstraintViolation('INVALID_SLOT', 'Transfer/parking space is not storage')
        return super().place(container, bay, row)


class VerticalTerminalSimulator(TerminalSimulator):
    """Two fixed job roles, full stock access, paid retreat, and no passing.

    The first profile crane handles external trucks; the second handles vessel
    work. Both retain the complete stock range. End transfer uses one exclusive
    lane per interface. The finite discharge pipeline is shared by all vessels
    in this block, from STS dispatch through completed yard storage. It is a
    synthetic reservation capacity, not a measured physical transfer buffer.
    """

    def __init__(self, profile, scenario, *, landside_bay=0, waterside_bay=None,
                 transfer_row=None, yard_handover_cap=1, **kwargs):
        if len(profile.cranes) != 2 or len({c.crane_id for c in profile.cranes}) != 2:
            raise ValueError('Vertical runtime requires exactly two distinct cranes')
        n = profile.block.bay_count
        water = n + 1 if waterside_bay is None else waterside_bay
        row = (profile.block.row_count + 1) / 2 if transfer_row is None else transfer_row
        gap = profile.safety_gap_bay
        if (not all(math.isfinite(v) for v in (landside_bay, water, row, gap))
                or not landside_bay < 1 or not water > n or gap <= 0
                or not 1 <= row <= profile.block.row_count):
            raise ValueError('Interfaces must flank the stock area with a positive safety gap')
        if type(yard_handover_cap) is not int or yard_handover_cap <= 0:
            raise ValueError('An explicit positive vessel handover capacity is required')
        self.input_profile = profile
        self.landside_crane, self.waterside_crane = (c.crane_id for c in profile.cranes)
        self.landside_bay, self.waterside_bay = float(landside_bay), float(water)
        self.end_transfer_row = float(row)
        self.parking_bays = (math.floor(landside_bay) - math.ceil(gap),
                             math.ceil(water) + math.ceil(gap))
        cranes = tuple(replace(c, service_bay_min=self.parking_bays[0],
                               service_bay_max=self.parking_bays[1]) for c in profile.cranes)
        runtime_profile = replace(profile, cranes=cranes,
            lane_graph=LaneGraph(('LANDSIDE', 'WATERSIDE'), ()))
        super().__init__(runtime_profile, scenario, yard_handover_cap=yard_handover_cap,
                         **kwargs)

    def reset(self):
        super().reset()
        self.stacks = BoundedYardStacks(self.profile.block,
                                        copy.deepcopy(self.scenario.containers))
        self._rail_order = (self.landside_crane, self.waterside_crane)
        for cid, bay in zip(self._rail_order, self.parking_bays):
            yc = self.fleet.get(cid)
            yc.state.position_bay = float(bay)
            yc.state.trolley_row = self.end_transfer_row
            self.reservations.set_idle_position(cid, bay)
        if self._check:
            self.check_invariants()

    def _role_allows(self, job, crane_id):
        if job.is_external_truck:
            return crane_id == self.landside_crane
        if job.is_vessel_linked:
            return crane_id == self.waterside_crane
        return False

    def _can_sts_process(self, vessel):
        if (vessel.work_type == VesselWorkType.DISCHARGE
                and sum(self._discharge_pipeline.values()) >= self.yard_handover_cap):
            return False
        return super()._can_sts_process(vessel)

    def _complete(self, crane_id):
        super()._complete(crane_id)
        # The base engine wakes the completed job's vessel. A block-wide cap
        # also requires waking other blocked vessels; clearing their marker
        # before queueing ensures that no duplicate wake is scheduled.
        for vid in sorted(self.vessels):
            vessel = self.vessels[vid]
            if (vessel.started and not vessel.done
                    and vessel.work_type == VesselWorkType.DISCHARGE
                    and vessel.sts_blocked_since_s is not None
                    and self._can_sts_process(vessel)):
                vessel.sts_blocked_since_s = None
                self.queue.push(self.clock, EventKind.STS_MOVE, vid)

    def check_invariants(self):
        super().check_invariants()
        counts = self._discharge_pipeline.values()
        if any(n < 0 for n in counts) or sum(counts) > self.yard_handover_cap:
            raise ConstraintViolation('HANDOVER_CAPACITY', 'Block discharge pipeline exceeded')

    def _interface(self, job):
        return ((self.landside_bay, self.end_transfer_row, 1), 'LANDSIDE') \
            if job.is_external_truck else \
            ((self.waterside_bay, self.end_transfer_row, 1), 'WATERSIDE')

    def _dispatchable(self, job, crane_id):
        return self._role_allows(job, crane_id) and super()._dispatchable(job, crane_id)

    def _jobref(self, job, spec, yc):
        if not self._role_allows(job, spec.crane_id):
            return None
        ref = super()._jobref(job, spec, yc)
        if ref is None:
            return None
        _, lane = self._interface(job)
        return replace(ref, lane_id=lane, eligible_crane_ids=(spec.crane_id,))

    def _store_slot(self, job, spec, bay, row, exclude=frozenset()):
        dest = super()._store_slot(job, spec, bay, row, exclude=exclude)
        if dest is not None and not (type(dest[0]) is int and type(dest[1]) is int
                                    and 1 <= dest[0] <= self.profile.block.bay_count
                                    and 1 <= dest[1] <= self.profile.block.row_count):
            raise ConstraintViolation('INVALID_SLOT', 'Store selector escaped the stock area')
        return dest

    def _plan(self, crane_id, ref, *, extra_exclude=frozenset()):
        if ref.kind != CandidateKind.REPOSITION:
            job = self.jobs.get(ref.job_id)
            if job is None or not self._role_allows(job, crane_id):
                return None
        plan = super()._plan(crane_id, ref, extra_exclude=extra_exclude)
        if plan is None or ref.kind != CandidateKind.SERVE:
            return plan
        job = self.jobs[ref.job_id]
        interface, lane = self._interface(job)
        spec, geom = self.fleet.spec(crane_id), self.profile.block
        state = self.fleet.get(crane_id).state
        moves = list(plan.moves)
        index = 0 if job.service_mode == ServiceMode.STORE else len(moves) - 1
        old = moves[index]
        start = ((state.position_bay, state.trolley_row) if index == 0
                 else moves[index - 1].dst[:2])
        src = interface if job.service_mode == ServiceMode.STORE else old.src
        dst = old.dst if job.service_mode == ServiceMode.STORE else interface
        actual = move_container(spec, geom, *start, src, dst)
        moves[index] = replace(old, src=src, dst=dst,
            duration_s=actual.duration_s, loaded_gantry_m=actual.loaded_gantry_m,
            empty_gantry_m=actual.empty_gantry_m)
        return replace(plan, moves=tuple(moves), lane_id=lane,
            duration_s=plan.duration_s - old.duration_s + actual.duration_s,
            loaded_gantry_m=plan.loaded_gantry_m - old.loaded_gantry_m + actual.loaded_gantry_m,
            empty_gantry_m=plan.empty_gantry_m - old.empty_gantry_m + actual.empty_gantry_m,
            end_bay=actual.end_bay, end_row=actual.end_row,
            corridor=(min(plan.corridor[0], interface[0]),
                      max(plan.corridor[1], interface[0])))
