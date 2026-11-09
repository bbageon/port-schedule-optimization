"""Read-only, fixed-cohort order timing for the K-hour teacher experiment."""
from collections import defaultdict
import math

from audit_admission_ledger import LedgerProbe
from yard_rl.v7.evaluation_probe import digest

FLOWS = ('GATE_IN', 'GATE_OUT', 'VESSEL_DISCHARGE', 'VESSEL_LOAD')


def duration(start, finish, end):
    """Restricted elapsed time: never discard unfinished work or count future time."""
    if start is None or start > end:
        return None
    if finish is not None and finish < start - 1e-6:
        raise ValueError(('completion precedes origin', start, finish))
    return max(0., min(finish, end) - start if finish is not None else end - start)


def aggregate(rows, end):
    """Truck A->O; vessel batch planned start->yard completion, same fixed cohort.

    The mixed ALL_ORDERS mean is an order-response indicator, not physical truck
    turn time or ship port time. Ready->yard completion is conditional/descriptive.
    """
    buckets = defaultdict(list)
    for row in rows:
        value = duration(row['origin_s'], row['finish_s'], end)
        if value is None:
            continue
        for key in (row['flow'], 'ALL_ORDERS',
                    'TRUCK' if row['flow'].startswith('GATE_') else 'VESSEL'):
            buckets[key].append((row, value))
    result = {}
    for key, values in buckets.items():
        ready = [duration(r['ready_s'], r['yard_done_s'], end) for r, _ in values]
        ready = [x for x in ready if x is not None]
        total = math.fsum(v for _, v in values)
        result[key] = dict(n=len(values), total_s=total, mean_s=total/len(values),
            unfinished=sum(r['finish_s'] is None or r['finish_s'] > end for r, _ in values),
            not_ready=sum(r['ready_s'] is None or r['ready_s'] > end for r, _ in values),
            ready_n=len(ready), ready_mean_s=math.fsum(ready)/len(ready) if ready else None)
    return result


class FlowProbe(LedgerProbe):
    """Observe all pre-existing vessel jobs, including those not physically released."""

    def start(self, built, mbt):
        super().start(built, mbt)
        self.vessel_origins = {}
        for bid, sim in mbt.blocks.items():
            for jid, job in sim.jobs.items():
                if job.flow.value.startswith('VESSEL_'):
                    v = sim.vessels[job.vessel_id]
                    self.vessel_origins[jid] = dict(job_id=jid, block=bid,
                        flow=job.flow.value, origin_s=v.plan.planned_start_s,
                        planned_release_s=job.release_time, stream=job.vessel_id)

    def finish(self, mbt, records, episode):
        super().finish(mbt, records, episode)
        rows = []
        for r in self.rows:
            t = r['times']
            assert r['admitted'] and t['gate_in_s'] == r['scheduled_gate_s']
            rows.append(dict(job_id=r['job_id'], block=r['block'], flow=r['flow'],
                origin_s=t['gate_in_s'], ready_s=t['block_in_s'],
                service_start_s=t['service_start_s'], yard_done_s=t['job_done_s'],
                finish_s=t['gate_out_s'], planned_release_s=None, stream=None))
        seen = set()
        for bid, sim in mbt.blocks.items():
            releases = {jid: t for t, kind, jid in sim.event_log
                        if kind in ('VESSEL_RELEASED', 'JOB_RELEASED')}
            for jid, job in sim.jobs.items():
                if not job.flow.value.startswith('VESSEL_'):
                    continue
                seen.add(jid)
                row = dict(self.vessel_origins[jid], ready_s=releases.get(jid),
                    service_start_s=job.service_start, yard_done_s=job.service_end,
                    finish_s=job.service_end)
                # planned_release_s is deliberately not an origin for discharge:
                # its actual physical release is controlled by the transport chain.
                stamps = [row[k] for k in ('origin_s', 'ready_s', 'service_start_s', 'finish_s')
                          if row[k] is not None]
                assert all(a <= b+1e-6 for a, b in zip(stamps, stamps[1:])), (jid, row)
                if job.service_start is not None:
                    assert row['ready_s'] is not None, jid
                rows.append(row)
        assert seen == self.vessel_origins.keys()
        assert len({r['job_id'] for r in rows}) == len(rows)
        self.flow_rows = sorted(rows, key=lambda r: r['job_id'])
        cohort = [{k: r[k] for k in ('job_id', 'block', 'flow', 'origin_s',
                                    'planned_release_s', 'stream')} for r in self.flow_rows]
        self.result['flow_timing'] = dict(cohort_sha256=digest(cohort),
            rows_sha256=digest(self.flow_rows), at24=aggregate(rows, 86400.),
            at26=aggregate(rows, 93600.), definitions=dict(
                truck='actual gate entry to gate exit',
                vessel='stream planned start to individual yard completion; batch response time',
                all_orders='job-count-weighted mixture; not physical truck or vessel port time',
                ready='actual block arrival/release to yard completion; conditional diagnostic',
                unfinished='retain all fixed-origin orders and count only observed elapsed time'))
        a = self.result['flow_timing']['at24']['TRUCK']
        assert a['n'] == episode.admitted
        assert math.isclose(a['mean_s'], episode.breakdown['mean_turn_time_s'], rel_tol=1e-11)
        assert a['unfinished'] == episode.breakdown['n_censored']
