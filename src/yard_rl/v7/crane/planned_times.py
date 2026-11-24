"""Decision-time information for the ETA/ETW experiment. No realized future timestamps."""
from collections import defaultdict
import heapq
import math

from ..world.domain.enums import JobStatus


ARMS = {"eta": ("eta",), "etw": ("etw",), "both": ("eta", "etw"),
        "both_need": ("eta", "etw", "need")}


def encode_gap(due, now):
    if due is None:
        return [0., 0.]
    if not math.isfinite(due):
        raise ValueError("Nonfinite public planned time")
    hours = (due - now) / 3600.
    return [1., math.copysign(math.log1p(abs(hours)), hours)]


class PublicSchedule:
    def __init__(self, arm):
        self.columns = ARMS[arm]
        self.arrivals = defaultdict(list)
        self.announced = defaultdict(set)
        self.decisions = 0
        self.visible = dict.fromkeys(self.columns, 0)

    def notice(self, block, job):
        if "eta" in self.columns and job.provided_eta is not None:
            self.announced[id(block)].add(job.job_id)
            heapq.heappush(self.arrivals[id(block)], (job.provided_eta, job.job_id))

    def candidates(self, sim, pairs):
        now = sim.clock
        heap = self.arrivals[id(sim)]
        while heap:
            job = sim.jobs.get(heap[0][1])
            if job is not None and job.status == JobStatus.PLANNED and job.provided_eta == heap[0][0]:
                break
            heapq.heappop(heap)
        next_eta = heap[0][0] if heap else None
        next_start = min((v.plan.planned_start_s for v in sim.vessels.values()
                          if not v.started and not v.done), default=None) if "etw" in self.columns else None
        result = []
        for _, candidate in pairs:
            ref = candidate.job_ref
            job = sim.jobs.get(ref.job_id) if ref else None
            vessel = sim.vessels.get(job.vessel_id) if job and job.vessel_id else None
            row = []
            if "eta" in self.columns:
                eta = (job.provided_eta if job and job.is_external_truck and job.status == JobStatus.PLANNED
                       and job.job_id in self.announced[id(sim)]
                       else next_eta)
                row.extend(encode_gap(eta, now))
            if "etw" in self.columns:
                start = vessel.plan.planned_start_s if vessel and not vessel.started else next_start
                row.extend(encode_gap(start, now))
            if "need" in self.columns:
                need = None
                if vessel and vessel.started and not vessel.done:
                    # Orders are numbered by the public planned stream cadence. Delay
                    # the original slot by progress actually observed up to this decision.
                    interval = vessel.plan.sts_move_interval_s
                    done = vessel.plan.total_moves - vessel.remaining_moves
                    next_planned = vessel.plan.planned_start_s + (done + 1) * interval
                    need = job.release_time + interval + max(0., now - next_planned)
                row.extend(encode_gap(need, now))
            result.append(row)
        self.decisions += 1
        for i, name in enumerate(self.columns):
            self.visible[name] += any(row[2*i] for row in result)
        return result
