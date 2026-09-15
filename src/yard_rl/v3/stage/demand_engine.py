"""Preserve declared demand while retaining the engine's physical service guards.

Legacy month admission discards requests when a box or a future storage reservation
is unavailable at notice time. This opt-in contract registers the request and its
original arrival instead. It does not create containers, add storage, change flows,
reschedule arrivals, or charge a new cost. Unserved trucks remain censored in the
existing gate-in cost; outstanding vessel moves remain on their original manifest.
"""
from __future__ import annotations

import bisect
import random

from ..world.domain.enums import JobFlow, JobStatus
from ..world.integrated.events import EventKind
from ..world.integrated.multiblock import JobRecord, TransferError
from ..world.integrated.time_contract import TruckTimes
from .month_engine import MonthTerminal


class DemandTerminal(MonthTerminal):
    """Request registration is distinct from permission to store/retrieve a box."""

    def admit_external_job(self, bid, job, *, gate_in_s, travel_s):
        jid = job.job_id
        if jid in self.ledger.records:
            raise TransferError(f"{jid}: already registered")
        if bid not in self.blocks:
            raise TransferError(f"{jid}: unknown block {bid}")
        sim = self.blocks[bid]
        if sim.time_ledger is None or job.exit_travel_s is None:
            raise TransferError(f"{jid}: missing truck time contract")
        arr = gate_in_s + travel_s
        full_end = sim.scenario.horizon_s + sim.scenario.drain_window_s
        if gate_in_s < sim.clock - 1e-9 or arr <= sim.clock + 1e-9 or arr > full_end:
            raise TransferError(f"{jid}: invalid arrival gate={gate_in_s} block={arr}")
        if job.flow not in (JobFlow.GATE_IN, JobFlow.GATE_OUT):
            raise TransferError(f"{jid}: not an external truck")
        if job.flow == JobFlow.GATE_OUT and job.target_container is not None:
            tgt = job.target_container
            if tgt not in sim.stacks.containers or any(
                    j.target_container == tgt for j in sim.jobs.values()):
                raise TransferError(f"{jid}: target absent or already claimed: {tgt}")
        # Waiting requests consume no physical storage slot. The unchanged
        # _dispatchable/_store_slot/_plan/commit checks still guard real storage.
        job.actual_gate_in, job.actual_block_arrival = gate_in_s, arr
        sim.jobs[jid] = job
        sim.queue.push(arr, EventKind.BLOCK_ARRIVAL, jid)
        tl = sim.time_ledger
        tl.records[jid] = TruckTimes(gate_in=gate_in_s)
        i = bisect.bisect_left(tl._a_sorted, gate_in_s)
        tl._a_sorted.insert(i, gate_in_s)
        if i < tl._a_idx:
            tl._a_idx += 1
            tl._n_inside += 1
        self.ledger.register(JobRecord(job_id=jid, origin_block=bid, owner=bid,
            flow=job.flow.value, a_gate_in=gate_in_s))

    def bind_available_targets(self, t):
        """Bind released unassigned work to real, unclaimed stock; never to future stock.

        Future vessel moves do not preclaim the day's entire inventory at midnight.
        Jobs wait in release-time/job-ID order. Selection within available stock is
        deterministic random sampling, without looking at stack accessibility.
        """
        def ready_at(job):
            return job.actual_block_arrival if job.is_external_truck else job.release_time

        for bid, sim in self.blocks.items():
            pending = [j for j in sim.jobs.values()
                if j.flow in (JobFlow.GATE_OUT, JobFlow.VESSEL_LOAD)
                and j.target_container is None and j.status != JobStatus.DONE
                and ready_at(j) <= t]
            if not pending:
                continue
            claimed = {j.target_container for j in sim.jobs.values()
                if j.target_container is not None}
            available = sorted(set(sim.stacks.containers) - claimed)
            for job in sorted(pending, key=lambda j: (ready_at(j), j.job_id)):
                if not available:
                    break
                index = random.Random(f"v3:demand:{bid}:{job.job_id}").randrange(len(available))
                job.target_container = available.pop(index)
                if not hasattr(self, "demand_bindings"):
                    self.demand_bindings = []
                self.demand_bindings.append({"job_id": job.job_id, "at_s": t,
                    "block": bid, "target": job.target_container, "flow": job.flow.value})
