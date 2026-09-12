"""Post-execution observations only; do not choose actions or alter the world."""
from collections import Counter
from dataclasses import asdict


def capture(runtime):
    terminal = runtime.mbt
    jobs, blocks, vessels = [], {}, {}
    for bid, sim in sorted(terminal.blocks.items()):
        blocks[bid] = dict(clock=sim.clock,end=sim.end,geometry=asdict(sim.profile.block),
            inventory=len(sim.stacks.containers),reservation_free_slots=terminal.free_slots(bid),
            cranes={cid:asdict(sim.fleet.get(cid)) for cid in sim.fleet.ids()},
            active_plans={cid:asdict(plan) for cid,plan in sim._active_plans.items()},
            pending_events=[asdict(e) for e in sorted(sim.queue._heap)],
            yard_handover_cap=sim.yard_handover_cap,
            discharge_pipeline=dict(sim._discharge_pipeline),
            last_events=sim.event_log[-50:])
        for vid,vessel in sim.vessels.items():
            vessels[vid] = dict(home_block=bid,**asdict(vessel))
        for jid,job in sorted(sim.jobs.items()):
            if job.status.value == 'DONE':
                continue
            cid = job.target_container or 'IN_' + jid
            source = terminal.sources[cid]
            box = sim.stacks.containers.get(cid)
            basic = {c:bool(sim._dispatchable(job,c)) for c in sim.fleet.ids()}
            row = dict(block=bid,**asdict(job),container=cid,source=source,
                expected_container_block=terminal.locations[cid],
                source_ready_s=terminal.ready_times.get(cid),exit_s=terminal.exit_times.get(cid),
                actual_container=None if box is None else asdict(box),
                blockers_above=[] if box is None else sim.stacks.blockers_above(cid),
                extra_road_ready_s=sim.cargo_not_before.get(jid),
                basic_dispatchable_by_crane=basic,
                idle_basic_dispatchable_cranes=[c for c,ok in basic.items() if ok and sim.fleet.get(c).idle],
                execution_record=None if jid not in runtime.bridge.records else asdict(runtime.bridge.records[jid]),
                order=None if jid not in runtime.bridge.orders else asdict(runtime.bridge.orders[jid]))
            if job.status.value in ('RUNNING','ASSIGNED') or job.assigned_crane is not None:
                reason = 'SERVICE_IN_PROGRESS'
            elif job.target_container is not None and box is None:
                reason = 'SOURCE_NOT_READY' if cid not in terminal.ready_times else 'READY_CONTAINER_NOT_IN_BLOCK'
            elif sim.clock < sim.cargo_not_before.get(jid,0):
                reason = 'EXTRA_ROAD_TRAVEL_PENDING'
            elif job.status.value not in ('WAITING','RELEASED'):
                reason = 'NOT_RELEASED_TO_YARD'
            elif not any(basic.values()):
                reason = 'BASIC_PHYSICAL_CONSTRAINT'
            elif not row['idle_basic_dispatchable_cranes']:
                reason = 'BASIC_READY_BUT_CRANES_BUSY'
            else:
                reason = 'BASIC_READY_REQUIRES_JOINT_PLAN_CHECK'
            row['observed_category'] = reason
            jobs.append(row)
    return dict(time_s=runtime.time_s,cargo=terminal.cargo_report(),jobs=jobs,blocks=blocks,vessels=vessels,
        categories=dict(Counter(j['observed_category'] for j in jobs)),
        categories_by_flow={flow:dict(Counter(j['observed_category'] for j in jobs if j['flow']==flow))
                            for flow in sorted({j['flow'] for j in jobs})},
        caveat='Post-run snapshot; basic dispatchability is NOT a full joint plan or a causal policy verdict.')


def json_default(value):
    if isinstance(value,(set,frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)
