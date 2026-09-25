"""Move a dependent job, not a container. Preserve elapsed time and identity."""
import bisect
import heapq

from ..world.domain.enums import JobStatus
from ..world.integrated.events import EventKind
from ..world.integrated.multiblock import TransferError


def _gate_index(ledger, jid, t):
    gate_in = ledger.records[jid].gate_in
    index = bisect.bisect_left(ledger._a_sorted, gate_in)
    if index < len(ledger._a_sorted) and ledger._a_sorted[index] == gate_in:
        return index
    # Daily pruning removes consumed A indices, not live trucks or occupancy.
    if gate_in < t:
        return None
    raise TransferError(f'{jid}: missing future gate-in accounting index')


def validate_dependent_move(terminal, jid, dst):
    rec = terminal.ledger.records.get(jid)
    if rec is None or rec.owner == dst:
        return
    src = rec.owner
    a, b = terminal.blocks[src], terminal.blocks[dst]
    job = a.jobs[jid]
    if job.status not in (JobStatus.PLANNED, JobStatus.WAITING, JobStatus.RELEASED):
        raise TransferError(f'{jid}: dependent work already started')
    t = terminal.now
    if a.clock != t or b.clock != t:
        raise TransferError('Dependent routing requires synchronized clocks')
    if job.is_external_truck:
        if a.time_ledger is None or b.time_ledger is None or jid not in a.time_ledger.records:
            raise TransferError(f'{jid}: missing dependent time ledger')
        _gate_index(a.time_ledger, jid, t)


def move_dependent(terminal, jid, dst):
    validate_dependent_move(terminal, jid, dst)
    rec = terminal.ledger.records.get(jid)
    if rec is None or rec.owner == dst:
        return
    src, t = rec.owner, terminal.now
    a, b = terminal.blocks[src], terminal.blocks[dst]
    job = a.jobs[jid]
    ready = None
    if job.is_external_truck:
        ta, tb = a.time_ledger, b.time_ledger
        times = ta.records[jid]
        old_arr = job.actual_block_arrival
        if times.block_arrival is not None:
            # B remains the first observed block arrival. The extra road leg is
            # a separate readiness constraint and is still charged in A->O.
            ready = t + terminal.layout.block_to_block_s(src, dst)
        elif job.actual_gate_in > t:
            new_arr = job.actual_gate_in + terminal.layout.gate_to_block_s(dst)
        else:
            new_arr = max(t, old_arr) + terminal.layout.block_to_block_s(src, dst)
        index = _gate_index(ta, jid, t)
        ta.records.pop(jid)
        if index is not None:
            ta._a_sorted.pop(index)
            if index < ta._a_idx:
                ta._a_idx -= 1
        if times.gate_in < t:
            ta._n_inside -= 1
        tb.records[jid] = times
        index = bisect.bisect_left(tb._a_sorted, times.gate_in)
        tb._a_sorted.insert(index, times.gate_in)
        # Boundary A==t has not been integrated yet, and must be consumed once.
        if times.gate_in < t:
            tb._a_idx += 1
            tb._n_inside += 1
        if jid in ta._in_block:
            tb._in_block[jid] = ta._in_block.pop(jid)
        if jid in a.kpis._waiting:
            b.kpis._waiting[jid] = a.kpis._waiting.pop(jid)
        if ready is None:
            job.actual_block_arrival = new_arr
            for field in ('provided_eta', 'estimated_block_arrival'):
                if getattr(job, field, None) is not None:
                    setattr(job, field, getattr(job, field) + new_arr - old_arr)
            b.queue.push(new_arr, EventKind.BLOCK_ARRIVAL, jid)
        else:
            b.cargo_not_before[jid] = ready
            b.queue.push(ready, EventKind.ETA_UPDATED, jid)
    elif job.status == JobStatus.PLANNED:
        b.queue.push(job.release_time, EventKind.JOB_RELEASED, jid)
    a.queue._heap = [e for e in a.queue._heap if not (
        e.payload == jid and e.kind_name in ('BLOCK_ARRIVAL', 'JOB_RELEASED', 'ETA_UPDATED'))]
    heapq.heapify(a.queue._heap)
    b.jobs[jid] = a.jobs.pop(jid)
    rec.owner, rec.version = dst, rec.version + 1
    rec.transfer_history += ((src, dst, t),)
    terminal.relocations.append(dict(job=jid, source=src, destination=dst,
                                    time_s=t, road_ready_s=ready))
    terminal.update_order_location(jid, dst)
    a._clear_yields()
    b._clear_yields()
    a._refresh_rates()
    b._refresh_rates()
