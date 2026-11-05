"""Conservative candidate-planning reuse, scoped to a simulator and a decision.

Physical fingerprints are checked once per generation, including direct edits.
Any yard change invalidates the whole block's plans: no unproven local dependency
shortcut. Time-dependent features and reservation feasibility are never cached.
Actual assignment and dry-run commit continue to call the uncached planner.
"""
from contextlib import contextmanager
from dataclasses import replace


class DispatchCache:
    def __init__(self):
        self.states = {}
        self.active = None
        self.stats = dict(plan_hits=0, plan_misses=0, slot_hits=0, slot_misses=0,
                          invalidations=0, fingerprints=0)

    def __deepcopy__(self, memo):
        # Plans are a derived optimization, never shared between CF worlds.
        out = type(self)()
        memo[id(self)] = out
        return out

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    @contextmanager
    def scope(self, sim, cid, enabled=True):
        previous = self.active
        self.active = None
        try:
            if enabled and sim.slot_selector is None and sim.store_slot_selector is None:
                yc = sim.fleet.get(cid)
                # Include values, not merely object identity: scenario tests and
                # integrations can edit container attributes without a stack event.
                yard = tuple((key, c.container_id, c.bay, c.row, c.tier, c.size,
                              c.load_status, c.work_available, c.special_flags)
                             for key, c in sim.stacks.containers.items())
                piles = tuple((key, tuple(ids)) for key, ids in sim.stacks._stacks.items())
                signature = (sim.profile.block, sim.stacks.geom, sim.fleet.spec(cid),
                             yc.state.position_bay, yc.state.trolley_row,
                             sim.reservations.reserved_slots(), yard, piles)
                self.stats['fingerprints'] += 1
                old = self.states.get(cid)
                if old is None or old['signature'] != signature:
                    if old is not None:
                        self.stats['invalidations'] += 1
                    old = dict(signature=signature, plans={}, slots={})
                    self.states[cid] = old
                old['seen'] = set()
                self.active = old
            yield
        finally:
            if self.active is not None:
                # Completed/cancelled/removed jobs must not accumulate cached plans.
                seen = self.active.pop('seen')
                self.active['plans'] = {key: value for key, value in self.active['plans'].items()
                                        if key in seen}
            self.active = previous

    def slot(self, key, compute):
        state = self.active
        if state is None:
            return compute()
        if key not in state['slots']:
            state['slots'][key] = compute()
            self.stats['slot_misses'] += 1
        else:
            self.stats['slot_hits'] += 1
        return state['slots'][key]

    def plan(self, sim, cid, ref):
        state = self.active
        if state is None:
            return sim._plan(cid, ref)
        job = sim.jobs.get(ref.job_id)
        job_key = None if job is None else (job.job_id, job.flow, job.target_container,
                                            job.inbound_size, job.inbound_load)
        key = (cid, ref, job_key)
        state['seen'].add(key)
        if key not in state['plans']:
            state['plans'][key] = sim._plan(cid, ref)
            self.stats['plan_misses'] += 1
        else:
            self.stats['plan_hits'] += 1
        plan = state['plans'][key]
        # Duration/moves are time-independent with built-in selectors. Start isn't.
        return replace(plan, start_s=sim.clock) if plan is not None and plan.start_s != sim.clock else plan
