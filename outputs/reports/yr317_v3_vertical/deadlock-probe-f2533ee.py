"""Read-only replay instrumentation; fixed f2533ee code, inputs and policy."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

workspace = Path(__file__).resolve().parents[3]
source = Path('/home/geonu/yr317-v3-vertical-f2533ee')
out = Path(__file__).with_suffix('')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(source/'src'))
os.chdir(source)
from yard_rl.v3.eval.__main__ import _load_nets
from yard_rl.v3.eval.contracts import runtime_identity, network_identity, write_json
from yard_rl.v3.eval.seed_bank import primitive, file_sha
from yard_rl.v3.layouts.environment import VerticalEnvironment
from yard_rl.v3.stage.month import plan_days
from yard_rl.v3.stage.month_run import run_month
from yard_rl.v3.world.domain.enums import JobStatus, InformationLevel
from yard_rl.v3.world.integrated.candidates import CandidateGenerator
from yard_rl.v3.world.integrated.policy_config import LEGACY_DEFAULT
import torch


class StableDiagnosticCaptured(Exception):
    pass


def diagnose(sim):
    # All planner/candidate introspection runs on a private snapshot. The live
    # simulation receives no extra policy calls, assignments or mutations.
    snap = copy.deepcopy(sim)
    details = []
    for jid, j in sorted(snap.jobs.items()):
        if j.status not in (JobStatus.WAITING, JobStatus.RELEASED):
            continue
        target = snap.stacks.containers.get(j.target_container)
        cid = snap.landside_crane if j.is_external_truck else snap.waterside_crane
        row = dict(job=jid, flow=j.flow.value, status=j.status.value,
            assigned_crane=j.assigned_crane, target=j.target_container,
            target_present=target is not None, available=None if target is None else target.work_available,
            target_slot=None if target is None else (target.bay, target.row, target.tier), role=cid)
        try:
            spec, yc = snap.fleet.spec(cid), snap.fleet.get(cid)
            row['dispatchable'] = snap._dispatchable(j, cid)
            ref = snap._jobref(j, spec, yc)
            row['ref_present'] = ref is not None
            if target is not None:
                row['blockers'] = snap.stacks.blockers_above(j.target_container)
                row['rehandle_capacity'] = snap.stacks.rehandle_capacity_ok(j.target_container, spec)
            if ref is not None:
                plan = snap._plan(cid, ref)
                row['plan_present'] = plan is not None
                if plan is not None:
                    row.update(corridor=plan.corridor, end_bay=plan.end_bay,
                        duration_s=plan.duration_s,
                        reject=snap.reservations.reject_reason(snap._reservation(plan)))
            row['reason'] = ('UNBOUND_TARGET' if not j.inbound_size and j.target_container is None else
                'TARGET_MISSING' if j.target_container is not None and target is None else
                'TARGET_UNAVAILABLE' if target is not None and not target.work_available else
                'NOT_DISPATCHABLE' if not row['dispatchable'] else
                'NO_JOBREF' if ref is None else
                'NO_PLAN' if not row.get('plan_present') else row['reject'] or 'FEASIBLE')
        except Exception:
            row['reason'], row['error'] = 'INSPECTION_EXCEPTION', traceback.format_exc()
        details.append(row)
    gen = CandidateGenerator(config=LEGACY_DEFAULT)
    candidates = {}
    for cid in snap.fleet.ids():
        candidates[cid] = [dict(kind=g.kind.value, job=None if g.job_ref is None else g.job_ref.job_id,
            feasible=g.feasible, reason=g.mask_reason,
            end_bay=None if g.plan is None else g.plan.end_bay)
            for g in gen.generate(snap, cid, InformationLevel.PRE_ADVICE).items]
    return snap, dict(clock=sim.clock, positions={c.crane_id: dict(
        bay=c.state.position_bay, row=c.state.trolley_row, idle=c.idle, yielded=c.yielded,
        down=c.down, assigned=c.state.assigned_job) for c in sim.fleet.all()},
        status_counts=dict(Counter(j.status.value for j in sim.jobs.values())),
        pending=list(sim._pending), last_decision_at=sim._last_decision_at,
        escape_at=sim._escape_at, escape_count=sim.deadlock_escape_count,
        active_reservations=primitive(sim.reservations.active()),
        idle_barriers=sim.reservations.idle_positions(),
        deadlock_corridors=snap.interference_deadlock_corridors(),
        role_job_reasons=dict(Counter(r['reason'] for r in details)),
        jobs=details, candidates=candidates,
        discharge_pipeline=dict(sim._discharge_pipeline),
        transfer_busy_until=list(sim.transfer.busy_until), transfer_pending=len(sim.transfer.pending),
        next_event=sim.queue.peek_time(), last_events=sim.event_log[-20:])


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if b'python' in (proc/'cmdline').read_bytes() and os.sched_getaffinity(int(proc.name)) == {19}:
                raise RuntimeError(f'CPU19 already occupied: {proc.name}')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    os.sched_setaffinity(0, {19})
    os.nice(10)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if not head.startswith('f2533ee') or subprocess.check_output(['git', 'status', '--porcelain'], text=True):
        raise RuntimeError('Expected clean frozen f2533ee')
    manifest = json.loads((Path(__file__).parent/'run-f2533ee/manifest.json').read_text())
    if manifest['seed'] != 99194001 or manifest['counts'] != [60, 60]:
        raise RuntimeError('Diagnostic input changed')
    checkpoint = Path(manifest['checkpoint'])
    if file_sha(checkpoint) != manifest['checkpoint_sha256']:
        raise RuntimeError('Fixed model changed')
    seller, buyer, _ = _load_nets(str(checkpoint))
    before = [network_identity(n) for n in (seller, buyer)]
    out.mkdir(exist_ok=False)
    write_json(out/'manifest.json', dict(source_commit=head, source_checkout=str(source),
        cpu=19, pid=os.getpid(), source_manifest_sha256=file_sha(Path(__file__).parent/'run-f2533ee/manifest.json'),
        diagnostic_script_sha256=file_sha(__file__), expected_input=manifest['expected_input'],
        environment_spec=manifest['environment_spec'], checkpoint_sha256=manifest['checkpoint_sha256'],
        runtime=runtime_identity(), purpose='Fixed-input deadlock replay; not a performance run'))
    sims, stable, signatures, captured = {}, Counter(), {}, {}
    original = VerticalEnvironment.make_sim
    def retain(env, scenario, profile, block_id):
        sim = original(env, scenario, profile, block_id)
        sims[block_id] = sim
        return sim
    VerticalEnvironment.make_sim = retain
    started = time.monotonic()
    def observe(row):
        if row['at_s'] < 30*3600:
            return
        for bid in ('Y05', 'Y14'):
            sim = sims[bid]
            jobs = [j for j in sim.jobs.values() if j.status in (JobStatus.WAITING, JobStatus.RELEASED)]
            idle = all(c.state.assigned_job is None for c in sim.fleet.all())
            signature = (tuple((j.job_id,j.target_container,j.status.value) for j in jobs),
                         tuple(c.state.position_bay for c in sim.fleet.all()))
            if not jobs or not idle:
                stable[bid] = 0
                continue
            stable[bid] = stable[bid]+1 if signatures.get(bid) == signature else 1
            signatures[bid] = signature
            if stable[bid] not in (1, 10) or bid in captured:
                continue
            snap, detail = diagnose(sim)
            detail.update(block=bid, observed_at_s=row['at_s'], consecutive_samples=stable[bid],
                          elapsed_s=time.monotonic()-started)
            with (out/'observations.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(detail,allow_nan=False)+'\n')
            print(json.dumps({k:detail[k] for k in ('block','observed_at_s','consecutive_samples','role_job_reasons','positions')}),flush=True)
            if stable[bid] == 10:
                path = out/(bid+'-snapshot.pkl.gz')
                with gzip.open(path,'wb') as stream:
                    pickle.dump(snap,stream,protocol=4)
                captured[bid] = dict(snapshot=path.name,sha256=file_sha(path),diagnosis=detail)
        if len(captured) == 2:
            raise StableDiagnosticCaptured()
    print(json.dumps(dict(event='started',pid=os.getpid(),cpu=19)),flush=True)
    status = 'completed'
    try:
        with torch.inference_mode():
            run_month(arm='NO_REALLOC', seed=99194001, days=plan_days(99194001,(60,60)),
                seller_net=seller,buyer_net=buyer,workers=1,explore=0,
                admission_mode='PRESERVE',supply_mode='COUNT_BALANCED',capture_requests=True,
                diagnose_admissions=True,capture_daily=True,daily_sample_s=300,
                expected_input=manifest['expected_input'],environment_spec=manifest['environment_spec'],
                on_observation=observe,on_day=lambda d:print(json.dumps(dict(event='day',day=d.index,elapsed_s=time.monotonic()-started)),flush=True))
    except StableDiagnosticCaptured:
        status='stable_cases_captured'
    except BaseException:
        status='failed'
        write_json(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        VerticalEnvironment.make_sim=original
        write_json(out/'summary.json',dict(state=status,elapsed_s=time.monotonic()-started,
            captured=captured,original_runtime_not_modified=True,
            frozen_networks=before==[network_identity(n) for n in (seller,buyer)]))
    print(json.dumps(dict(event='finished',state=status,captured=list(captured),elapsed_s=time.monotonic()-started)),flush=True)


if __name__ == '__main__':
    main()
