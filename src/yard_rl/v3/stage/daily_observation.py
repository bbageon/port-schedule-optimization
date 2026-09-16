"""Read-only operational observations; never fed to actors or the cost function.

Queue averages use the engine's continuous queue integral. Peaks are sampled,
not exact event-level maxima. Day boundaries precede admission/reallocation.
"""
from __future__ import annotations

import math

from ..world.domain.enums import JobStatus
from .month import DAY_S


def snapshot(mbt, t):
    blocks = {}
    for bid, sim in mbt.blocks.items():
        row = dict(truck_queue=0, truck_yard_unfinished=0,
                   vessel_yard_unfinished=0, other_yard_unfinished=0,
                   yard_jobs_not_released=0, yard_jobs_released_unfinished=0,
                   yard_jobs_in_service=0, crane_committed_remaining_s=0.0,
                   truck_queue_area_s=float(sim.kpis.queue_area_s))
        for j in sim.jobs.values():
            if j.status in (JobStatus.DONE, JobStatus.CANCELLED):
                continue
            group = ('truck' if j.is_external_truck else
                     'vessel' if j.is_vessel_linked else 'other')
            row[group + '_yard_unfinished'] += 1
            released = (j.actual_block_arrival is not None and j.actual_block_arrival <= t
                        if j.is_external_truck else j.status != JobStatus.PLANNED)
            row['yard_jobs_released_unfinished' if released else 'yard_jobs_not_released'] += 1
            started = j.service_start is not None and j.service_start <= t
            row['yard_jobs_in_service'] += int(started)
            row['truck_queue'] += int(j.is_external_truck and released and not started)
        for crane in sim.fleet.all():
            row['crane_committed_remaining_s'] += max(0.0, crane.state.available_at - t)
        row['yard_jobs_unfinished'] = sum(row[k + '_yard_unfinished']
                                         for k in ('truck', 'vessel', 'other'))
        blocks[bid] = row
    totals = {key: sum(b[key] for b in blocks.values()) for key in next(iter(blocks.values()))}
    return dict(at_s=float(t), total=totals, blocks=blocks)


def _actions(bridge):
    return dict(spatial=bridge.n_space, temporal=bridge.n_time,
                decisions=len(bridge.market.seller.trail))


def action_mix(start, end):
    counts = {key: end[key] - start[key] for key in start}
    n = counts['spatial'] + counts['temporal']
    return dict(**counts, committed_changes=n,
                spatial_share=counts['spatial'] / n if n else None,
                temporal_share=counts['temporal'] / n if n else None,
                denominator='successful spatial + temporal changes; calendar decision day')


class DailyObserver:
    def __init__(self, days, *, seed, arm, sample_s=300.0, on_sample=None):
        if not math.isfinite(sample_s) or sample_s < 60 or sample_s % 60 or DAY_S % sample_s:
            raise ValueError('daily_sample_s must divide 86400 and be a positive multiple of 60')
        self.days, self.seed, self.arm = days, seed, arm
        self.sample_s, self.on_sample = float(sample_s), on_sample
        self.n_days = len(days)
        self.rows, self.samples = {}, []
        self.current, self.last_t, self.sample_count = None, None, 0
        self.starts, self.action_starts = {}, {}

    @property
    def metadata(self):
        return dict(schema='yard_rl.v3.daily-observation.v1', seed=self.seed, arm=self.arm,
                    sample_s=self.sample_s, samples=self.sample_count,
                    timing='synchronized review before admission, reallocation and new vessels',
                    queue='arrived at block, service not started; external trucks only',
                    queue_mean='exact continuous engine queue area / calendar-day seconds',
                    queue_max='maximum among regular samples; not event-level maximum',
                    workload='unfinished yard-job counts; committed crane seconds separate',
                    turn_time='original request-day cohort, gate-in to gate-out/cutoff',
                    independent_unit='whole month, not daily rows')

    def observe(self, mbt, t, bridge):
        end = self.n_days * DAY_S
        if t > end or abs(t / self.sample_s - round(t / self.sample_s)) > 1e-7:
            return
        if self.last_t == t:
            return
        state = snapshot(mbt, t)
        actions = _actions(bridge)
        boundary = abs(t / DAY_S - round(t / DAY_S)) < 1e-9
        if boundary:
            index = round(t / DAY_S)
            if index:
                self._close(index - 1, state, actions)
            if index < self.n_days:
                self.current = index
                self.starts[index], self.action_starts[index] = state, actions
                self.samples = []
        if t < end:
            self.samples.append(state)
        self.last_t = t
        self.sample_count += 1
        if self.on_sample:
            self.on_sample(dict(seed=self.seed, arm=self.arm, day_index=min(int(t // DAY_S), self.n_days-1),
                                boundary=boundary, **state))

    def _close(self, index, end, actions):
        start = self.starts[index]
        if len(self.samples) != round(DAY_S / self.sample_s):
            raise RuntimeError(f'Incomplete daily queue samples: day {index}')
        def summary(key):
            a, b = (start['total'], end['total']) if key is None else (start['blocks'][key], end['blocks'][key])
            values = [s['total'] if key is None else s['blocks'][key] for s in self.samples]
            return dict(truck_queue_mean=(b['truck_queue_area_s'] - a['truck_queue_area_s']) / DAY_S,
                        truck_queue_sampled_max=max(v['truck_queue'] for v in values),
                        released_yard_jobs_sampled_mean=sum(v['yard_jobs_released_unfinished'] for v in values) / len(values),
                        released_yard_jobs_sampled_max=max(v['yard_jobs_released_unfinished'] for v in values))
        self.rows[index] = dict(schema='yard_rl.v3.daily-observation.v1', seed=self.seed,
            arm=self.arm, day_index=index, evaluation_day=self.days[index].is_train,
            requested_trucks=self.days[index].load, sample_s=self.sample_s, samples=len(self.samples),
            start=start, end=end, queue=summary(None),
            blocks={bid: summary(bid) for bid in start['blocks']},
            actions=action_mix(self.action_starts[index], actions))

    def finish(self, schedule, records, end_s):
        if len(self.rows) != self.n_days:
            raise RuntimeError('Daily observation missing calendar boundaries')
        arrival_counts = [0] * self.n_days
        for rec in records.values():
            at = rec.gate_in_s
            if at is not None and 0 <= at < self.n_days * DAY_S:
                arrival_counts[int(at // DAY_S)] += 1
        # Recover deferred original requests as well as actual gate arrivals.
        by_day = [[] for _ in self.days]
        for e in schedule:
            by_day[int(e['day'])].append(records[e['job_id']])
        for index, cohort in enumerate(by_day):
            entered = [r for r in cohort if r.gate_in_s is not None and r.gate_in_s <= end_s]
            done = [r for r in entered if r.gate_out_s is not None and r.gate_out_s <= end_s]
            self.rows[index]['cohort'] = dict(requested=len(cohort), entered_by_cutoff=len(entered),
                completed_by_cutoff=len(done), censored_by_cutoff=len(entered)-len(done),
                not_entered_by_cutoff=len(cohort)-len(entered),
                actual_gate_entries_on_calendar_day=arrival_counts[index])
