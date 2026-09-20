"""Read-only replay of a stalled legacy month up to the stall (YR-317-j).

Fixed 7e2fb14 source, the same frozen inputs as the independent evaluation, the
NO_REALLOC policy (no networks involved). Every block simulator is retained; when
the watched block has idle cranes and waiting jobs with an unchanged signature for
10 consecutive 300 s samples after ``--from-day``, the block is diagnosed on a
private deep copy, snapshotted, and the replay stops. Nothing is written back to
the evaluation evidence.
"""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import argparse
import copy
import gzip
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

WORKSPACE = Path('/mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매')
SOURCE = Path('/home/geonu/yr317-v3-independent-7e2fb14')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(SOURCE / 'src'))
sys.path.insert(0, str(SOURCE / 'scripts/v3'))
os.chdir(SOURCE)


class StallCaptured(Exception):
    pass


def diagnose(sim):
    from yard_rl.v3.world.domain.enums import JobStatus, InformationLevel
    from yard_rl.v3.world.integrated.candidates import CandidateGenerator
    from yard_rl.v3.world.integrated.policy_config import LEGACY_DEFAULT
    from yard_rl.v3.eval.seed_bank import primitive
    snap = copy.deepcopy(sim)
    details = []
    for jid, j in sorted(snap.jobs.items()):
        if j.status not in (JobStatus.WAITING, JobStatus.RELEASED):
            continue
        target = snap.stacks.containers.get(j.target_container)
        row = dict(job=jid, flow=j.flow.value, status=j.status.value, assigned_crane=j.assigned_crane,
                   target=j.target_container, target_present=target is not None,
                   available=None if target is None else target.work_available,
                   target_slot=None if target is None else (target.bay, target.row, target.tier), per_crane={})
        for cid in snap.fleet.ids():
            try:
                spec, yc = snap.fleet.spec(cid), snap.fleet.get(cid)
                d = dict(dispatchable=snap._dispatchable(j, cid))
                ref = snap._jobref(j, spec, yc)
                d['ref_present'] = ref is not None
                if ref is not None:
                    plan = snap._plan(cid, ref)
                    d['plan_present'] = plan is not None
                    if plan is not None:
                        d.update(corridor=list(plan.corridor), end_bay=plan.end_bay, duration_s=plan.duration_s,
                                 reject=snap.reservations.reject_reason(snap._reservation(plan)))
                d['reason'] = ('NOT_DISPATCHABLE' if not d['dispatchable'] else 'NO_JOBREF' if ref is None
                               else 'NO_PLAN' if not d.get('plan_present') else d['reject'] or 'FEASIBLE')
            except Exception:
                d = dict(reason='INSPECTION_EXCEPTION', error=traceback.format_exc()[-400:])
            row['per_crane'][cid] = d
        if target is not None:
            try:
                row['blockers'] = snap.stacks.blockers_above(j.target_container)
            except Exception:
                pass
        details.append(row)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    candidates = {}
    for cid in snap.fleet.ids():
        gc = gen.generate(snap, cid, InformationLevel.PRE_ADVICE)
        candidates[cid] = [dict(kind=g.kind.value, job=None if g.job_ref is None else g.job_ref.job_id,
                                feasible=g.feasible, mandatory=g.mandatory, reason=g.mask_reason,
                                end_bay=None if g.plan is None else g.plan.end_bay) for g in gc.items]
    reasons = Counter(d['reason'] for r in details for d in r['per_crane'].values())
    return snap, dict(clock=sim.clock, day=sim.clock / 86400,
        positions={c.crane_id: dict(bay=c.state.position_bay, row=c.state.trolley_row, idle=c.idle, yielded=c.yielded,
                                    down=c.down, assigned=c.state.assigned_job) for c in sim.fleet.all()},
        status_counts=dict(Counter(j.status.value for j in sim.jobs.values())),
        pending=list(sim._pending), last_decision_at=sim._last_decision_at, escape_at=sim._escape_at,
        escape_count=sim.deadlock_escape_count, escape_mode=sim.escape_mode,
        active_reservations=primitive(sim.reservations.active()), idle_barriers=sim.reservations.idle_positions(),
        deadlock_corridors=snap.interference_deadlock_corridors(), job_reasons=dict(reasons),
        candidates=candidates, kept_feasible={cid: sum(1 for c in cs if c['feasible'] and c['kind'] != 'WAIT')
                                             for cid, cs in candidates.items()},
        jobs=details, last_events=[list(e) for e in sim.event_log[-30:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--block', required=True)
    parser.add_argument('--from-day', type=float, required=True)
    parser.add_argument('--cpu', type=int, default=19)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    from run_independent_evaluation import expected_month, read
    from yard_rl.v3.stage import month_run
    from yard_rl.v3.stage.month import plan_month
    from yard_rl.v3.stage.month_run import run_month
    from yard_rl.v3.world.domain.enums import JobStatus
    from yard_rl.v3.eval.seed_bank import file_sha
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if not head.startswith('7e2fb14') or subprocess.check_output(['git', 'status', '--porcelain'], text=True):
        raise RuntimeError('Expected the clean frozen 7e2fb14 checkout')
    cfg = read(WORKSPACE / 'outputs/reports/yr317_v3_independent_eval/config.json')
    expected = expected_month(SimpleNamespace(workspace=WORKSPACE), cfg, args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source_commit=head, source_checkout=str(SOURCE), seed=args.seed, block=args.block,
                    from_day=args.from_day, cpu=args.cpu, pid=os.getpid(), expected_input=expected,
                    config_sha256=file_sha(WORKSPACE / 'outputs/reports/yr317_v3_independent_eval/config.json'),
                    probe_sha256=file_sha(__file__), arm='NO_REALLOC',
                    purpose='Read-only stall replay for mechanism diagnosis; not a performance run')
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    from yard_rl.v3.stage import daily_observation
    holder = {}
    original_observe = daily_observation.DailyObserver.observe
    def observe_hook(self, m, t, bridge):
        holder['terminal'] = m
        return original_observe(self, m, t, bridge)
    daily_observation.DailyObserver.observe = observe_hook
    started = time.monotonic()
    stable, signature, captured = Counter(), {}, {}
    def sim_for(block):
        term = holder.get('terminal')
        return None if term is None else term.blocks.get(block)
    def observe(row):
        if row['at_s'] < args.from_day * 86400:
            return
        sim = sim_for(args.block)
        if sim is None:
            raise RuntimeError(f'block {args.block} not found on the terminal')
        jobs = [j for j in sim.jobs.values() if j.status in (JobStatus.WAITING, JobStatus.RELEASED)]
        idle = all(c.state.assigned_job is None for c in sim.fleet.all())
        sig = (tuple((j.job_id, j.target_container, j.status.value) for j in jobs),
               tuple(c.state.position_bay for c in sim.fleet.all()))
        if not jobs or not idle:
            stable[args.block] = 0
            return
        stable[args.block] = stable[args.block] + 1 if signature.get(args.block) == sig else 1
        signature[args.block] = sig
        if stable[args.block] not in (1, 10):
            return
        snap, detail = diagnose(sim)
        detail.update(block=args.block, observed_at_s=row['at_s'], consecutive_samples=stable[args.block],
                      waiting_jobs=len(jobs), elapsed_s=time.monotonic() - started)
        with (out / 'observations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(detail, ensure_ascii=False, allow_nan=False, default=str) + '\n')
        print(json.dumps({k: detail[k] for k in ('block', 'day', 'consecutive_samples', 'waiting_jobs',
                                                 'job_reasons', 'kept_feasible', 'escape_count', 'positions')},
                         ensure_ascii=False, default=str), flush=True)
        if stable[args.block] == 10:
            path = out / f'{args.block}-snapshot.pkl.gz'
            with gzip.open(path, 'wb') as stream:
                pickle.dump(snap, stream, protocol=4)
            captured[args.block] = dict(snapshot=path.name, sha256=file_sha(path), diagnosis_at_s=row['at_s'])
            raise StallCaptured()
    def on_day(d):
        print(json.dumps(dict(event='day', day=d.index, elapsed_s=time.monotonic() - started)), flush=True)
    status = 'completed_without_capture'
    try:
        with torch.inference_mode():
            run_month(arm='NO_REALLOC', seed=args.seed, days=plan_month(args.seed), workers=1, explore=0,
                      admission_mode='PRESERVE', supply_mode='COUNT_BALANCED', capture_requests=True,
                      diagnose_admissions=True, capture_daily=True, daily_sample_s=300.0,
                      expected_input=expected, on_observation=observe, on_day=on_day)
    except StallCaptured:
        status = 'stall_captured'
    except BaseException:
        status = 'failed'
        (out / 'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=1), encoding='utf-8')
        raise
    finally:
        daily_observation.DailyObserver.observe = original_observe
        (out / 'summary.json').write_text(json.dumps(dict(state=status, elapsed_s=time.monotonic() - started,
                                                         captured=captured), ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(event='finished', state=status, elapsed_s=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
