"""Tiny frozen-policy vertical-runtime qualification, never performance evidence.

Use a clean frozen checkout, an external fresh --out, and an explicit free --cpu.
Runs two continuous days on 21 blocks, sequentially for NO_REALLOC and RL_TIME.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ARMS = ('NO_REALLOC', 'RL_TIME')


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonl(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def qualify(args):
    import torch
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import network_identity, runtime_identity, write_json
    from yard_rl.v3.eval.seed_bank import (create_month_payload, digest, file_sha,
        primitive, validate_payload, write_bundle)
    from yard_rl.v3.layouts import synthetic_vertical_spec
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
    from yard_rl.v3.stage.month import plan_days
    from yard_rl.v3.stage.month_run import run_month
    from yard_rl.v3.stage.supply_plan import balance_vessel_supply
    from yard_rl.v3.world.integrated.profiles import build_h21_profile

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not hasattr(os, 'sched_setaffinity') or args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError('Qualification requires an available explicit Linux CPU')
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Use a clean frozen source checkout')
    if args.source_commit is not None:
        expected = subprocess.check_output(['git', 'rev-parse', '--verify',
                                           args.source_commit + '^{commit}'], text=True).strip()
        if expected != head:
            raise RuntimeError('Requested source commit differs from executed checkout')
    if args.out.is_relative_to(ROOT):
        raise ValueError('Output must be outside the clean frozen source checkout')
    checkpoint_hash = file_sha(args.checkpoint)
    days = plan_days(args.seed, tuple(args.counts))
    payload = create_month_payload(args.seed, days=days)
    canonical_validation = validate_payload(payload, require_month=False)
    initial = {b: len(s['containers']) for b, s in payload['initial_scenarios'].items()}
    profile = build_h21_profile()
    capacity = profile.block.bay_count * profile.block.row_count * profile.block.tier_max
    balanced, supply_audit = balance_vessel_supply(payload['vessels'], payload['schedule'],
                                                  initial, {b: capacity for b in initial})
    expected_input = dict(schedule_sha256=digest(payload['schedule']),
        initial_scenarios_sha256=digest(payload['initial_scenarios']), vessels_sha256=digest(balanced))
    spec = synthetic_vertical_spec(profile, n_blocks=21)
    runtime = runtime_identity()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(at=now(), purpose='physical qualification only; no performance inference',
        claim_eligible=False, new_training_runs=0, independent_performance_runs=0,
        seed=args.seed, counts=args.counts, arms=list(ARMS), days=primitive(days), cpu=args.cpu,
        pid=os.getpid(), source_commit=head, source_checkout=str(ROOT), git_dirty=False,
        runner_sha256=file_sha(__file__), checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash, expected_input=expected_input, runtime=runtime,
        environment_spec=spec, environment_spec_sha256=digest(spec),
        canonical_validation=canonical_validation, supply_plan_audit=supply_audit,
        canonical_bundle_sha256=write_bundle(args.out/'canonical-input.json.gz', payload),
        balanced_vessels_sha256=write_bundle(args.out/'balanced-vessels.json.gz', balanced))
    write_json(args.out/'manifest.json', manifest)
    write_json(args.out/'environment-spec.json', spec)
    summaries = []
    for arm in ARMS:
        folder = args.out/arm
        folder.mkdir()
        seller, buyer, checkpoint_description = _load_nets(str(args.checkpoint))
        weights = [network_identity(n) for n in (seller, buyer)]
        started = time.monotonic()
        write_json(folder/'manifest.json', dict(at=now(), arm=arm,
            qualification_manifest_sha256=file_sha(args.out/'manifest.json'),
            checkpoint_description=checkpoint_description, networks=weights))
        write_json(args.out/'progress.json', dict(at=now(), state='running', arm=arm,
            completed_arms=len(summaries), planned_arms=len(ARMS), pid=os.getpid()))

        def on_day(day):
            row = dict(at=now(), elapsed_s=time.monotonic()-started, **day.as_dict())
            with (folder/'days.jsonl').open('a', encoding='utf-8') as stream:
                jsonl(stream, row)
            write_json(args.out/'progress.json', dict(at=now(), state='running', arm=arm,
                completed_arms=len(summaries), planned_arms=len(ARMS), pid=os.getpid(),
                day_index_completed=day.index, days_planned=len(days), elapsed_s=row['elapsed_s']))
            print(json.dumps(dict(arm=arm, day=day.index, elapsed_s=row['elapsed_s'])), flush=True)

        reset_rollout_calls()
        with gzip.open(folder/'operating-state.jsonl.gz', 'wt', encoding='utf-8') as stream:
            with torch.inference_mode():
                result = run_month(arm=arm, seed=args.seed, days=days,
                    seller_net=seller, buyer_net=buyer, workers=1, explore=0,
                    admission_mode='PRESERVE', supply_mode='COUNT_BALANCED',
                    capture_requests=True, diagnose_admissions=True, capture_daily=True,
                    daily_sample_s=300, expected_input=expected_input, environment_spec=spec,
                    on_day=on_day, on_observation=lambda row: jsonl(stream, row))
        write_json(folder/'result.json', primitive(asdict(result)))
        for name, rows in (('requests', result.request_ledger),
                           ('container-links', result.container_links)):
            with gzip.open(folder/(name+'.jsonl.gz'), 'wt', encoding='utf-8') as stream:
                for row in rows:
                    jsonl(stream, row)
        with (folder/'daily-final.jsonl').open('w', encoding='utf-8') as stream:
            for day in result.days:
                jsonl(stream, day.as_dict())
        identity_fields = ('job_id', 'requested_day', 'flow', 'requested_block',
                           'requested_arrival_s', 'lead_s', 'requested_target', 'travel_s')
        request_hash = digest([{k: r[k] for k in identity_fields} for r in result.request_ledger])
        checks = dict(all_requests_admitted=result.request_summary['all_requests_admitted'],
            all_requests_completed=result.request_summary['states'].get('COMPLETED', 0) == sum(args.counts),
            no_unbound_jobs=result.request_summary['unbound_jobs_at_end'] == 0,
            feasible_external_deadlines=bool(result.vessel_admissions) and all(
                r['structural_min_overrun_s'] == 0 for r in result.vessel_admissions),
            request_recording=result.request_summary['recording_ok'],
            admission_counts=result.request_summary['announcer_counts_match'],
            physical_invariants_enabled=result.request_summary['physical_invariants_enabled'],
            container_flow_passed=result.container_flow_summary['passed'],
            vessel_recording=result.vessel_work_summary['recording_ok'],
            all_vessel_work_completed=result.vessel_work_summary['all_requested_work_completed'],
            no_dropped_vessel_moves=result.vessel_work_summary['unadmitted_moves'] == 0,
            no_policy_exceptions=result.policy_exceptions == 0,
            no_rollouts_or_training=rollout_calls() == 0,
            no_spatial_actions=result.n_space == 0,
            frozen_networks=weights == [network_identity(n) for n in (seller, buyer)],
            runtime_unchanged=runtime == runtime_identity(),
            checkpoint_unchanged=file_sha(args.checkpoint) == checkpoint_hash,
            canonical_input_matches=result.environment_manifest['canonical_input'] == expected_input,
            two_complete_daily_records=len(result.days) == 2 and all(
                d.operational.get('day_index') == d.index and 'cohort' in d.operational for d in result.days))
        summary = dict(arm=arm, passed=all(v is True for v in checks.values()), checks=checks,
            elapsed_s=time.monotonic()-started, requested_identity_sha256=request_hash,
            runtime_schedule_sha256=result.environment_manifest['runtime_schedule_sha256'],
            cost_krw=sum(d.phi_krw for d in result.days), spatial=result.n_space, temporal=result.n_time,
            request_summary=result.request_summary, vessel_work_summary=result.vessel_work_summary,
            result_sha256=file_sha(folder/'result.json'), raw_artifacts={p.name:file_sha(p)
                for p in sorted(folder.iterdir()) if p.name.endswith(('.jsonl', '.jsonl.gz'))})
        write_json(folder/'qualification.json', summary)
        summaries.append(summary)
        if not summary['passed']:
            raise RuntimeError(f'{arm} qualification failed; all evidence retained')
    paired = (len({r['requested_identity_sha256'] for r in summaries}) == 1
              and len({r['runtime_schedule_sha256'] for r in summaries}) == 1)
    summary = dict(at=now(), passed=paired, purpose=manifest['purpose'], claim_eligible=False,
        new_training_runs=0, independent_performance_runs=0, same_input_across_arms=paired,
        seed=args.seed, counts=args.counts, arms=summaries,
        limitation='One tiny diagnostic trajectory; no robustness, superiority, or training claim.')
    write_json(args.out/'summary.json', summary)
    if not paired:
        raise RuntimeError('Policies received different inputs; evidence retained')
    write_json(args.out/'progress.json', dict(at=now(), state='completed', passed=True,
        completed_arms=2, planned_arms=2, claim_eligible=False))
    print(json.dumps(dict(passed=True, out=str(args.out), purpose=manifest['purpose'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/v3/month-02/ckpt_029.pt')
    parser.add_argument('--source-commit')
    parser.add_argument('--seed', type=int, default=99_194_001)
    parser.add_argument('--counts', type=int, nargs=2, default=[60, 60])
    parser.add_argument('--cpu', required=True, type=int)
    args = parser.parse_args()
    if args.seed < 0 or min(args.counts) <= 0 or not 0 <= args.cpu < 20:
        parser.error('Use a nonnegative diagnostic seed, two positive counts, and CPU 0..19')
    args.out, args.checkpoint = args.out.resolve(), args.checkpoint.resolve()
    if args.out.exists():
        parser.error('Output must be fresh; existing evidence is never overwritten')
    os.chdir(ROOT)
    try:
        qualify(args)
    except BaseException:
        if args.out.is_dir():
            from yard_rl.v3.eval.contracts import write_json
            write_json(args.out/'failure.json', dict(at=now(), traceback=traceback.format_exc()))
            write_json(args.out/'progress.json', dict(at=now(), state='failed', claim_eligible=False))
        raise


if __name__ == '__main__':
    main()
