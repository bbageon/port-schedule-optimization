"""Opt-in, conserved public workload potential (YR-331).

The current H21 experiment has identical full-span cranes and mutually routable
inbound blocks. This adapter fails on other geometry; it is not a dispatcher.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from ..world.contract.schema import CandidateKind
from ..world.domain.enums import JobStatus


def targets(capacity, fixed, movable, safe, available=None):
    a, f, m, c = (np.asarray(v, dtype=float) for v in (capacity, fixed, movable, safe))
    u = np.ones_like(a) if available is None else np.asarray(available, dtype=float)
    if (a.ndim != 1 or not len(a) or any(v.shape != a.shape for v in (f, m, c, u))
            or not all(np.isfinite(v).all() for v in (a, f, m, c, u))
            or np.any(a <= 0) or np.any(f < 0) or np.any(m < 0)
            or np.any(c <= 0) or np.any(c >= 1) or np.any(u < 0) or np.any(u > 1)):
        raise ValueError('Invalid workload/capacity group')
    high = np.maximum(f, a * c * u)
    total = f.sum() + min(m.sum(), (high - f).sum())
    lo, hi = 0.0, float((high / a).max())
    for _ in range(55):
        level = (lo + hi) / 2
        if np.minimum(high, np.maximum(f, a * level)).sum() < total:
            lo = level
        else:
            hi = level
    return np.minimum(high, np.maximum(f, a * hi))


def potential(capacity, fixed, movable, safe):
    a, f, m = (np.asarray(v, dtype=float) for v in (capacity, fixed, movable))
    goal = targets(a, f, m, safe)
    return -float((((f + m - goal) ** 2) / a).sum() / a.sum())


@dataclass(frozen=True)
class WorkloadConfig:
    service_s: dict[str, float]
    capacity_fraction: dict[str, float]
    safe_fraction: float
    horizon_s: float = 900.0
    eta: float = 1.0

    def __post_init__(self):
        vals = [*self.service_s.values(), *self.capacity_fraction.values(),
                self.safe_fraction, self.horizon_s, self.eta]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('Nonfinite workload configuration')
        if (not self.service_s or not self.capacity_fraction or min(vals[:-1]) <= 0
                or self.safe_fraction >= 1 or self.eta < 0
                or max(self.capacity_fraction.values()) > 1):
            raise ValueError('Invalid workload configuration')
        for flow in ('GATE_IN', 'GATE_OUT', 'VESSEL_LOAD', 'VESSEL_DISCHARGE'):
            if flow not in self.service_s:
                raise ValueError(f'Missing frozen service weight: {flow}')


class WorkloadMeter:
    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.last = None

    def bind(self, mbt, bridge):
        self.mbt, self.bridge = mbt, bridge
        self.keys = [(b, c) for b, s in sorted(mbt.blocks.items()) for c in sorted(s.fleet.ids())]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.capacity = np.array([self.config.horizon_s * self.config.capacity_fraction[c]
                                  for b, c in self.keys])
        self.safe = np.full(len(self.keys), self.config.safe_fraction)
        # Preserve the original request BEFORE the market can change it. Future
        # rows are stored but never enter a snapshot until their public notice.
        self.original = {k: (o.copino_notice_s, o.in_out_reserve_s)
                         for k, o in bridge.orders.items()}
        self.pending = sorted((notice, k) for k, (notice, _) in self.original.items())
        self.cursor, self.known = 0, set()
        for sim in mbt.blocks.values():
            for cid in sim.fleet.ids():
                spec = sim.fleet.spec(cid)
                if (spec.service_bay_min, spec.service_bay_max) != (1, sim.profile.block.bay_count):
                    raise ValueError('YR-331 adapter requires full-span shared H21 cranes')

    def snapshot(self, t):
        while self.cursor < len(self.pending) and self.pending[self.cursor][0] <= t:
            self.known.add(self.pending[self.cursor][1])
            self.cursor += 1
        fixed, movable = np.zeros(len(self.keys)), np.zeros(len(self.keys))
        expected, n_jobs = 0.0, 0

        def add(bid, job, flow, can_move):
            nonlocal expected, n_jobs
            sim = self.mbt.blocks[bid]
            weight = self.config.service_s[flow]
            cid = None if job is None else job.assigned_crane
            if cid is not None:
                plan = sim.active_plan(cid)
                if plan is None or plan.job_id != job.job_id or plan.kind != CandidateKind.SERVE:
                    raise RuntimeError('Assigned useful job has no matching committed plan')
                # The committed plan duration is public at dispatch (candidate
                # features already contain it), not a future realized timestamp.
                weight *= max(0.0, 1.0 - max(0.0, t - plan.start_s) / plan.duration_s)
                ids = [self.index[(bid, cid)]]
                can_move = False
            else:
                ids = [self.index[(bid, c)] for c in sim.fleet.ids()]
            shares = self.capacity[ids] / self.capacity[ids].sum()
            (movable if can_move else fixed)[ids] += shares * weight
            expected += weight
            n_jobs += 1

        for key in sorted(self.known):
            rec = self.bridge.records[key]
            if rec.job_done_s is not None and rec.job_done_s <= t:
                self.known.remove(key)
                continue
            if self.original[key][1] > t + self.config.horizon_s:
                continue
            order = self.bridge.orders[key]
            sim = self.mbt.blocks[order.con_loc]
            job = sim.jobs.get(key)
            if job is not None and job.status == JobStatus.DONE:
                continue
            # A one-shot market decision is irrevocable; it becomes fixed work,
            # even if its delayed truck has yet to enter the gate.
            can_move = (order.is_inbound and key not in self.bridge.market.decided
                        and (rec.gate_in_s is None or rec.gate_in_s > t))
            add(order.con_loc, job, 'GATE_IN' if order.is_inbound else 'GATE_OUT', can_move)
        for bid, sim in self.mbt.blocks.items():
            for job in sim.jobs.values():
                if (job.is_vessel_linked and job.status != JobStatus.DONE
                        and job.release_time <= t + self.config.horizon_s):
                    add(bid, job, job.flow.value, False)
        error = abs(float((fixed + movable).sum()) - expected)
        if error > 1e-7 * max(1, expected):
            raise RuntimeError('Workload shares do not conserve useful work')
        goal = targets(self.capacity, fixed, movable, self.safe)
        p = -float((((fixed + movable - goal) ** 2) / self.capacity).sum() / self.capacity.sum())
        self.last = dict(time_s=t, potential=p, jobs=n_jobs, total_work_s=expected,
                         fixed_work_s=float(fixed.sum()), movable_work_s=float(movable.sum()),
                         conservation_error_s=error, rho=((fixed + movable) / self.capacity).tolist(),
                         target_rho=(goal / self.capacity).tolist())
        return p

    def report(self):
        return dict(config=asdict(self.config), last=self.last)
