"""Realized telemetry and original-request audit, never fed into policy features."""
from collections import Counter, defaultdict
import math

import numpy as np

from ..reward.krw import truck_wait_krw
from ..world.contract.schema import CandidateKind
from ..world.domain.enums import JobStatus
from .journal import write_json


class Observer:
    def __init__(self, output, days, *, calibration=False):
        self.output, self.days, self.calibration = output, days, calibration
        self.completed, self.rows, self.idle_ready = set(), [], Counter()
        self.last_t, self.last_idle = None, []
        self.day_snapshots, self.potentials = [], []
        self.max_conservation_error = 0.0
        self.max_shaping = 0.0

    def __call__(self, rt):
        t = rt.time_s
        if self.last_t is not None and t > self.last_t:
            for cid in self.last_idle:
                self.idle_ready[cid] += t - self.last_t
        self.last_t, self.last_idle = t, []
        for bid, sim in rt.mbt.blocks.items():
            ready = any(j.status in (JobStatus.WAITING, JobStatus.RELEASED)
                        and (not j.target_container or j.target_container in sim.stacks.containers)
                        for j in sim.jobs.values())
            for cid in sim.fleet.ids():
                plan = sim.active_plan(cid)
                if ready and (plan is None or plan.kind != CandidateKind.SERVE):
                    self.last_idle.append(cid)
            for job in sim.jobs.values():
                if job.status != JobStatus.DONE or job.job_id in self.completed:
                    continue
                self.completed.add(job.job_id)
                if job.service_start is None or job.service_end is None:
                    raise RuntimeError('Completed job has no realized service interval')
                self.rows.append(dict(job=job.job_id, block=bid, crane=job.assigned_crane,
                    flow=job.flow.value, start=job.service_start, end=job.service_end))
        if rt.workload and rt.workload.last:
            w = rt.workload.last
            self.max_conservation_error = max(self.max_conservation_error, w['conservation_error_s'])
            self.max_shaping = max(self.max_shaping, abs(rt.last_shaping_reward))
            if round(t) % 900 == 0:
                self.potentials.append(w | {'shaping': rt.last_shaping_reward})
        if round(t) % 86400 == 0 and (not self.day_snapshots or self.day_snapshots[-1]['time_s'] != t):
            for sim in rt.mbt.blocks.values():
                sim.check_invariants()
            self.day_snapshots.append(dict(time_s=t, cost_krw=rt.cost_krw,
                c_vessel=rt.cost_breakdown['c_vessel'], cargo=rt.mbt.cargo_report(),
                request=self.request_metrics(rt, t), updates=len(rt.updates)))
            write_json(self.output / 'day-snapshots.json', self.day_snapshots)

    @staticmethod
    def original_orders(rt):
        # The runner binds this immutable input for diagnostics; no observation
        # or policy computation calls this evaluator.
        return rt.original_requests

    def request_metrics(self, rt, t, cohort=None):
        times, delays, queue, completed = [], [], [], 0
        actual_wait, original_wait = 0.0, 0.0
        for key, (request, notice) in self.original_orders(rt).items():
            if request > t or (cohort is not None and not cohort[0] <= request < cohort[1]):
                continue
            rec = rt.bridge.records[key]
            end = min(t, rec.gate_out_s) if rec.gate_out_s is not None else t
            tt = max(0.0, end - request)
            times.append(tt)
            original_wait += truck_wait_krw(tt)
            if rec.gate_in_s is not None:
                actual_wait += truck_wait_krw(max(0.0, end - rec.gate_in_s))
            current_request = rt.bridge.orders[key].in_out_reserve_s
            delays.append(max(0.0, current_request - request))
            completed += rec.gate_out_s is not None and rec.gate_out_s <= t
            if rec.block_in_s is not None and rec.block_in_s <= t:
                stop = min(t, rec.service_start_s) if rec.service_start_s is not None else t
                queue.append(max(0.0, stop - rec.block_in_s))
        return dict(requested=len(times), completed=int(completed), unfinished=len(times)-completed,
            original_request_wait_krw=original_wait, actual_gate_wait_krw=actual_wait,
            deferred_count=sum(x > .001 for x in delays), defer_mean_s=float(np.mean(delays)) if delays else 0,
            request_to_exit_mean_s=float(np.mean(times)) if times else 0,
            request_to_exit_p90_s=float(np.quantile(times, .9)) if times else 0,
            block_queue_mean_s=float(np.mean(queue)) if queue else 0)

    def finish(self, rt):
        self(rt)
        write_json(self.output / 'completed-jobs.json', self.rows)
        write_json(self.output / 'workload-samples.json', self.potentials)
        periods = [(d.t0, d.t1) for d in self.days if d.is_train]
        cohorts = [self.request_metrics(rt, rt.time_s, p) for p in periods]
        market = rt.bridge.market
        trades = []
        for lo, hi in periods:
            seller = [r for r in market.seller.trail if lo <= r['t'] < hi]
            buyer = [r for r in market.buyer.trail if lo <= r['t'] < hi]
            ledger = [r for r in rt.bridge.ledger if lo <= r['t'] < hi]
            trades.append(dict(start_s=lo, end_s=hi, seller_decisions=len(seller),
                buyer_calls=len(buyer), buyer_accepts=sum(r['action'] == 'BUY' for r in buyer),
                committed_space=sum(r['ok'] and r['kind'] == 'SPACE' for r in ledger),
                committed_time=sum(r['ok'] and r['kind'] == 'TIME' for r in ledger),
                completed_traded_jobs=sum(r['ok'] and rt.bridge.records[r['doc_key']].job_done_s is not None
                                         for r in ledger)))
        return dict(days=self.day_snapshots, measured_cohorts=cohorts,
                    measured_trades=trades,
                    whole=self.request_metrics(rt, rt.time_s),
                    realized_service_jobs=len(self.rows), idle_ready_s=dict(self.idle_ready),
                    max_conservation_error_s=self.max_conservation_error,
                    max_abs_shaping_reward=self.max_shaping)

    def calibration_report(self):
        samples = defaultdict(list)
        busy = Counter()
        for row in self.rows:
            duration = row['end'] - row['start']
            samples[row['flow']].append(duration)
            busy[row['crane']] += duration
        if None in busy:
            raise RuntimeError('Crane ownership lost in completed telemetry')
        stats = {flow: dict(n=len(v), mean_s=float(np.mean(v)),
                           cv=float(np.std(v) / np.mean(v))) for flow, v in samples.items()}
        all_services = [r['end'] - r['start'] for r in self.rows]
        mean, cv = float(np.mean(all_services)), float(np.std(all_services) / np.mean(all_services))
        safe = 300 / (300 + .5 * (1 + cv ** 2) * mean)
        capacity = {c: min(1.0, max(.1, s / (s + self.idle_ready[c]))) for c, s in busy.items()}
        config = dict(service_s={k: v['mean_s'] for k, v in stats.items()},
                      capacity_fraction=capacity, safe_fraction=safe, horizon_s=900.0, eta=1.0)
        if not math.isfinite(safe):
            raise RuntimeError('Insufficient calibration observations')
        return dict(config=config, service_stats=stats, productive_occupied_s=dict(busy),
                    nonproductive_ready_queue_s=dict(self.idle_ready),
                    capacity_estimator='60-second reference-policy sampling; includes joint blocking and pre-work',
                    safe_fraction_scope='M/G/1 diagnostic approximation, not empirically certified queue guarantee')
