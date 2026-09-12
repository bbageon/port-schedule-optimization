"""Fixed cargo lifecycle with causal, globally ordered cross-block events.

This is an opt-in execution contract, not a policy or a substitute container pool.
The frozen world modules are unchanged. PPO still observes only 60-second reviews.
"""
from bisect import bisect_left
from collections import Counter
from dataclasses import replace

from .month_engine import MonthTerminal
from .cargo_moves import move_dependent, validate_dependent_move
from .seed_inventory import enumerate_inventory
from ..world.contract.schema import CandidateKind
from ..world.domain.enums import JobFlow, JobStatus
from ..world.integrated.engine import TerminalSimulator, TerminalDecision
from ..world.integrated.events import EventKind
from ..world.integrated.multiblock import TransferError, JobRecord
from ..world.integrated.time_contract import TruckTimes


class CargoBlock(TerminalSimulator):
    def __init__(self, *args, **kwargs):
        self.cargo_terminal = None
        self.cargo_bid = None
        self.cargo_not_before = {}
        super().__init__(*args, **kwargs)

    def _dispatchable(self, job, crane_id):
        if self.clock < self.cargo_not_before.get(job.job_id, 0):
            return False
        return super()._dispatchable(job, crane_id)

    def _handle(self, event):
        super()._handle(event)
        if event.kind_name == 'ETA_UPDATED' and event.payload in self.cargo_not_before:
            self.cargo_not_before.pop(event.payload)
            self._clear_yields()

    def _complete(self, crane_id):
        plan = self._active_plans.get(crane_id)
        job = self.jobs.get(plan.job_id) if plan and plan.kind == CandidateKind.SERVE else None
        terminal = self.cargo_terminal
        if job and terminal:
            terminal.check_completion(self.cargo_bid, job, self.clock)
        super()._complete(crane_id)
        if job and terminal:
            terminal.completed(self.cargo_bid, job, self.clock)

    def _transfer_request(self, vid):
        terminal = self.cargo_terminal
        home = terminal.vessel_home[vid] if terminal else self.cargo_bid
        if terminal and home != self.cargo_bid:
            target = terminal.blocks[home]
            # Global event order guarantees no unprocessed earlier home event.
            target._advance(self.clock)
            target._transfer_request(vid)
            target._refresh_rates()
            terminal.remote_handoffs += 1
        else:
            super()._transfer_request(vid)


class CargoTerminal(MonthTerminal):
    def __init__(self, blocks, *, document, layout, **kwargs):
        self._cargo_time = 0.0
        super().__init__(blocks, **kwargs)
        self.layout = layout
        sources, demands = enumerate_inventory(document['initial_inventory'], document['schedule'],
                                               document['vessels_by_day'], document['seed'])
        self.sources = {s['container']: s for s in sources}
        self.locations = {s['container']: s['block'] for s in sources}
        self.exit_jobs = {}
        for d in demands:
            e, m = d['entry'], d['move']
            cid = e['target'] if m is None else e['targets'][m]
            self.exit_jobs[cid] = d['job']
        self.ready_times = {s['container']: 0.0 for s in sources if s['source_kind'] == 'INITIAL'}
        self.exit_times = {}
        self.vessel_home = {}
        self.orders, self.records = {}, {}
        self.relocations = []
        self.pending_admissions = 0
        self.remote_handoffs = 0
        self.completed_jobs = Counter()
        for bid, sim in self.blocks.items():
            sim.cargo_terminal, sim.cargo_bid = self, bid

    @property
    def now(self):
        return self._cargo_time

    def update_order_location(self, jid, bid):
        order = self.orders.get(jid)
        if order is not None and order.con_loc != bid:
            record = self.records[jid]
            record.prev_con_loc, record.con_swap_reason = order.con_loc, 'FIXED_CARGO_LOCATION'
            # Physical follow-up routing, NOT a new seller action for GATE_OUT.
            self.orders[jid] = replace(order, con_loc=bid)

    def resolve_entry(self, entry):
        if entry['flow'] != 'GATE_OUT':
            return entry
        cid = entry['target']
        bid = self.locations[cid]
        if bid == entry['block']:
            return entry
        self.update_order_location(entry['job_id'], bid)
        self.relocations.append(dict(job=entry['job_id'], source=entry['block'], destination=bid,
                                    time_s=self.now, phase='at_notice', road_ready_s=None))
        return entry | {'block': bid, 'travel_s': self.layout.gate_to_block_s(bid)}

    def admit_external_job(self, bid, job, *, gate_in_s, travel_s):
        if job.flow != JobFlow.GATE_OUT:
            cid = 'IN_' + job.job_id
            if (job.flow != JobFlow.GATE_IN or cid not in self.sources
                    or self.sources[cid]['source_job'] != job.job_id or cid in self.ready_times):
                raise TransferError(f'{job.job_id}: invalid or already produced incoming cargo')
            return super().admit_external_job(bid, job, gate_in_s=gate_in_s, travel_s=travel_s)
        jid, cid = job.job_id, job.target_container
        if (cid not in self.sources or self.exit_jobs.get(cid) != jid
                or cid in self.exit_times or bid != self.locations[cid]):
            raise TransferError(f'{jid}: invalid fixed cargo reservation')
        if jid in self.ledger.records or bid not in self.blocks:
            raise TransferError(f'{jid}: duplicate work or unknown block')
        sim = self.blocks[bid]
        arr = gate_in_s + travel_s
        if (sim.time_ledger is None or job.exit_travel_s is None
                or gate_in_s < sim.clock - 1e-9 or arr <= sim.clock + 1e-9 or arr > sim.end):
            raise TransferError(f'{jid}: invalid admission time or accounting')
        if cid in self.ready_times and cid not in sim.stacks.containers:
            raise TransferError(f'{jid}: ready cargo missing from its actual stack')
        job.actual_gate_in, job.actual_block_arrival = gate_in_s, arr
        sim.jobs[jid] = job
        sim.queue.push(arr, EventKind.BLOCK_ARRIVAL, jid)
        tl = sim.time_ledger
        tl.records[jid] = TruckTimes(gate_in=gate_in_s)
        i = bisect_left(tl._a_sorted, gate_in_s)
        tl._a_sorted.insert(i, gate_in_s)
        if i < tl._a_idx:
            tl._a_idx += 1
            tl._n_inside += 1
        self.ledger.register(JobRecord(jid, bid, bid, job.flow.value, a_gate_in=gate_in_s))
        self.pending_admissions += cid not in self.ready_times

    def commit(self, txn):
        self.validate(txn)
        cid = 'IN_' + txn.job_id
        consumer = self.exit_jobs.get(cid)
        if consumer:
            validate_dependent_move(self, consumer, txn.dst)
        super().commit(txn)
        if cid in self.sources:
            self.locations[cid] = txn.dst
            if consumer:
                move_dependent(self, consumer, txn.dst)

    def check_completion(self, bid, job, t):
        cid = job.target_container
        if cid is None:
            cid = 'IN_' + job.job_id
            if cid not in self.sources or cid in self.ready_times:
                raise RuntimeError(f'{job.job_id}: unplanned or repeated cargo birth')
        elif (self.exit_jobs.get(cid) != job.job_id or cid in self.exit_times
                or cid not in self.ready_times or self.ready_times[cid] > t):
            raise RuntimeError(f'{job.job_id}: invalid or premature fixed cargo exit')
        if self.locations[cid] != bid:
            raise RuntimeError(f'{job.job_id}: cargo owner differs from execution block')

    def completed(self, bid, job, t):
        if job.target_container is None:
            cid = 'IN_' + job.job_id
            self.blocks[bid].stacks.containers[cid].block = bid
            self.ready_times[cid] = t
        else:
            self.exit_times[job.target_container] = t
        self.completed_jobs[job.flow.value] += 1

    def cargo_report(self):
        return dict(planned_sources=len(self.sources), planned_exits=len(self.exit_jobs),
            ready_sources=len(self.ready_times), completed_exits=len(self.exit_times),
            physical_inventory=sum(len(s.stacks.containers) for s in self.blocks.values()),
            expected_inventory=len(self.ready_times)-len(self.exit_times),
            pending_admissions=self.pending_admissions, dependent_reroutes=len(self.relocations),
            remote_vessel_handoffs=self.remote_handoffs, completed_jobs=dict(self.completed_jobs),
            active_unfinished_jobs=dict(Counter(j.flow.value for s in self.blocks.values()
                for j in s.jobs.values() if j.status != JobStatus.DONE)),
            active_unfinished_vessels=sum(not v.done for s in self.blocks.values()
                                         for v in s.vessels.values()),
            routing_examples=self.relocations[:10])

    def run(self, policy_fn, review_fn=None, cost_fn=None):
        """Globally ordered events: a receiving block can never outrun a message.

        Only affected blocks integrate between events; all blocks synchronize at
        the existing market grid. No extra PPO boundary or world copy is created.
        """
        end = max(s.end for s in self.blocks.values())
        epochs, ei = sorted(set(self._extra_epochs)), 0
        totals, last = {b:0.0 for b in self.blocks}, {b:0.0 for b in self.blocks}
        for s in self.blocks.values():
            s.cost.cut()
            s.review_epochs = []
        guard = 0
        while self._cargo_time < end:
            guard += 1
            if guard > self.LOOP_GUARD:
                raise RuntimeError('Fixed cargo event loop guard')
            candidates = [end]
            if ei < len(epochs):
                candidates.append(epochs[ei])
            for s in self.blocks.values():
                nt, wt = s.queue.peek_time(), s._next_wake_time()
                candidates.extend(x for x in (nt, wt) if x is not None and x <= end)
            t = min(candidates)
            if t < self._cargo_time - 1e-9:
                raise RuntimeError('Cross-block event scheduled in the past')
            self._cargo_time = t
            grid = ei < len(epochs) and abs(epochs[ei]-t) < 1e-9
            affected = set()
            # Completion priority precedes arrivals, independent of block order.
            while True:
                due = [(s.queue._heap[0].priority, b, s) for b,s in self.blocks.items()
                       if s.queue.peek_time() is not None and s.queue.peek_time() <= t + 1e-9]
                if not due:
                    break
                _, b, s = min(due, key=lambda x:(x[0],x[1]))
                s._process_next_event()
                affected.add(b)
            for b,s in self.blocks.items():
                wt = s._next_wake_time()
                if wt is not None and wt <= t + 1e-9:
                    affected.add(b)
            if grid or t == end:
                affected.update(self.blocks)
            for b in sorted(affected):
                s = self.blocks[b]
                s._advance(t)
                s.review_epochs = [t]
                while True:
                    event = s.run_until_decision()
                    if isinstance(event, TerminalDecision):
                        policy_fn(s, event)
                    else:
                        break
                if cost_fn:
                    totals[b] += cost_fn(s, last[b], t, s.cost.cut())
                    last[b] = t
            if grid:
                ei += 1
                for s in self.blocks.values():
                    self._sync_locks(s)
                if review_fn:
                    review_fn(self, t)
                for s in self.blocks.values():
                    s.review_epochs = [t]
                    while True:
                        event = s.run_until_decision()
                        if isinstance(event, TerminalDecision):
                            policy_fn(s, event)
                        else:
                            break
                # Review can admit work at this exact timestamp; next iteration
                # consumes those events before any clock moves forward.
        for s in self.blocks.values():
            if not s.terminal:
                s._finalize()
        self.ledger.harvest(self.blocks)
        report = self.cargo_report()
        if report['physical_inventory'] != report['expected_inventory']:
            raise RuntimeError('Physical cargo conservation failed')
        return dict(totals=totals, route_cost_s=self.route_cost_s,
                    terminal_total=sum(totals.values()), end=end)
