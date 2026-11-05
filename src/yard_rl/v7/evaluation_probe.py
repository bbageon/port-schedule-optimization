"""Optional read-only evidence for the real daily evaluation path (no teacher)."""
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import statistics
import time


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, 'value'):
        return value.value
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default,
                                    separators=(',', ':')).encode()).hexdigest()


def timing_summary(samples):
    values = sorted(samples)
    if not values:
        return dict(count=0, median_ms=0, p95_ms=0, max_ms=0, total_s=0)
    return dict(count=len(values), median_ms=statistics.median(values)/1e6,
                p95_ms=values[math.ceil(.95*len(values))-1]/1e6,
                max_ms=values[-1]/1e6, total_s=sum(values)/1e9)


class EvaluationProbe:
    """Timing excludes hashing/logging/apply. Whole-episode wall time includes them.

    A decision is one block's available crane(s), not all 42 cranes concurrently.
    Exact candidate plans, chosen actions, events and final accounting are hashed.
    """

    def __init__(self):
        self.generation_ns, self.selection_ns, self.samples = [], [], []
        self.candidates = hashlib.sha256()
        self.choices = hashlib.sha256()
        self.metadata = {}
        self.result = {}

    def start(self, built, mbt):
        self.metadata = dict(blocks=len(mbt.blocks),
            cranes=sum(len(s.profile.cranes) for s in mbt.blocks.values()),
            initial_containers=sum(len(s.stacks.containers) for s in mbt.blocks.values()),
            truck_orders=len(built['schedule']),
            initial_vessel_jobs=sum(len(s.jobs) for s in mbt.blocks.values()),
            input_sha256=digest((built['schedule'], built.get('vessel_schedule'),
                {b: [(k, asdict(c)) for k, c in s.stacks.containers.items()]
                 for b, s in mbt.blocks.items()})))
        self.block_names = {id(s): b for b, s in mbt.blocks.items()}
        self.started = time.perf_counter()

    def decision(self, sim, dp, generated, assignment, gen_ns, select_ns):
        self.generation_ns.append(gen_ns)
        self.selection_ns.append(select_ns)
        block = self.block_names[id(sim)]
        counts = [len(generated[c].items) for c in dp.crane_ids]
        self.samples.append((sim.clock, block, counts, len(sim.jobs), sim.kpis.waiting_count()))
        prefix = (sim.clock, block)
        # repr is deterministic for these immutable dataclass plans/references.
        self.candidates.update(repr((prefix, [(c, generated[c].items)
                                              for c in dp.crane_ids])).encode())
        self.choices.update(repr((prefix, [(c, assignment[c])
                                           for c in sorted(assignment)])).encode())

    def finish(self, mbt, records, episode):
        run_seconds = time.perf_counter() - self.started
        states = {b: [(j.job_id, j.status.value, j.service_start, j.service_end)
                       for j in s.jobs.values()] for b, s in mbt.blocks.items()}
        cache = Counter()
        counts = Counter()
        for sim in mbt.blocks.values():
            cache.update(sim.dispatch_cache.stats)
            counts.update((j.flow.value, j.status.value) for j in sim.jobs.values())
        hourly = {}
        for meta, ns in zip(self.samples, self.total_ns):
            hourly.setdefault(str(int(meta[0]//3600)), []).append(ns)
        self.result = dict(metadata=self.metadata, run_seconds=run_seconds,
            generation=timing_summary(self.generation_ns), selection=timing_summary(self.selection_ns),
            decision=timing_summary(self.total_ns),
            hourly={h: timing_summary(v) for h, v in hourly.items()},
            max_candidates_per_crane=max((max(s[2]) for s in self.samples), default=0),
            max_jobs_in_deciding_block=max((s[3] for s in self.samples), default=0),
            max_waiting_trucks_after_dispatch=max((s[4] for s in self.samples), default=0),
            candidates_sha256=self.candidates.hexdigest(), choices_sha256=self.choices.hexdigest(),
            events_sha256=digest({b: s.event_log for b, s in mbt.blocks.items()}),
            jobs_sha256=digest(states), records_sha256=digest({k: asdict(v) for k,v in records.items()}),
            result_sha256=digest(episode.as_dict()),
            flow_status_counts={f'{flow}/{status}': n for (flow,status),n in sorted(counts.items())},
            cache=dict(cache), raw_generation_ns=self.generation_ns,
            raw_selection_ns=self.selection_ns, decision_context=self.samples)

    @property
    def total_ns(self):
        return [a+b for a,b in zip(self.generation_ns, self.selection_ns)]
