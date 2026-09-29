"""Pinned paired evaluation: original assignment versus reallocation, same crane policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from yard_rl.v6.ppo.checkpoint import load_policy, save_checkpoint
from yard_rl.v6.ppo.journal import RunJournal, write_json
from yard_rl.v6.ppo.original_comparison import OriginalComparisonRuntime, assignment_audit
from yard_rl.v6.ppo.provenance import code_stamp, file_sha256
from yard_rl.v6.ppo.runtime import DebugStop, PPOConfig
from yard_rl.v6.reward.counterfactual import rollout_calls
from yard_rl.v6.reward.operational import KEYS, normalized_loss, reference_config
from yard_rl.v6.stage.month import plan_days
from yard_rl.v6.stage.month_run import run_month
from yard_rl.v6.stage.seed_bundle import load_seed_bundle

SPEC = Path('.claude/docs/dashboard-task-specs/YR-331-f-original-assignment-comparison.md')
TRAIN_SEEDS = (9931000, 9941000, 9951000)
EVAL_SEEDS = (9961000, 9971000, 9981000)


def parameter_hash(policy):
    digest = hashlib.sha256()
    for key, value in policy.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def original_request_totals(rt, original, t):
    total, requested, unfinished = 0., 0, 0
    for order in original:
        request = order['in_out_reserve_s']
        if request > t:
            continue
        rec = rt.bridge.records[order['doc_key']]
        end = min(t, rec.gate_out_s) if rec.gate_out_s is not None else t
        total += max(0., end - request)
        requested += 1
        unfinished += int(rec.gate_out_s is None or rec.gate_out_s > t)
    return dict(original_request_s=total, requested=requested, unfinished=unfinished)


def save_compressed(path, data):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def run_one(args):
    torch.set_num_threads(1)
    stamp = code_stamp()
    input_hash = file_sha256(args.seed_bundle)
    checkpoint_hash = file_sha256(args.checkpoint)
    document, audit = load_seed_bundle(args.seed_bundle, expected_sha256=input_hash)
    days = plan_days(args.seed, loads=[row['load'] for row in document['days']])
    config = PPOConfig(reward_mode='operational')
    reference = reference_config(gamma=config.gamma, time_unit_s=config.time_unit_s)
    policy = load_policy(args.checkpoint)
    initial_hash = parameter_hash(policy)
    smoke = args.smoke
    window = (3600., 7200.) if smoke else (86400., 172800.)
    endpoint = 7200. if smoke else 259200.
    stop = endpoint if smoke else None
    out = Path(args.output)
    manifest = dict(code=stamp, ppo=asdict(config), arguments=vars(args),
        seed=args.seed, arm=args.arm, learning_days=0, optimizer_updates=0,
        prereg_path=args.prereg, prereg_sha256=file_sha256(args.prereg),
        input_path=args.seed_bundle, input_sha256=input_hash, input_audit=audit,
        checkpoint_path=args.checkpoint, checkpoint_sha256=checkpoint_hash,
        initial_parameter_sha256=initial_hash, measurement_window_s=list(window),
        intervention_s=window[0], endpoint_s=endpoint, stop_s=stop,
        action_mode='sample; role-local RNG; same learned crane dispatcher',
        normalization=reference, claim_scope='paired reassignment diagnostic, not original TOS trace')
    journal = RunJournal(out, days, manifest)
    snapshots, trace = {}, []

    def boundary(rt):
        raw = rt.physical
        if not trace or trace[-1][0] != rt.time_s:
            trace.append([rt.time_s] + [raw[key] for key in KEYS])
        if rt.time_s in (0., *window, endpoint):
            assignment = assignment_audit(document['orders'], rt.bridge.orders)
            if args.arm == 'original' or rt.time_s <= window[0]:
                assert assignment['original_fields_preserved'], assignment
                assert rt.bridge.traded_edges == rt.bridge.n_space == rt.bridge.n_time == 0
            assert parameter_hash(policy) == initial_hash
            snapshots[str(int(rt.time_s))] = dict(physical=raw,
                objective=normalized_loss(raw, reference),
                request=original_request_totals(rt, document['orders'], rt.time_s),
                cargo=rt.mbt.cargo_report(), assignments=assignment,
                n_space=rt.bridge.n_space, n_time=rt.bridge.n_time,
                parameter_sha256=initial_hash, updates=len(rt.updates))
            write_json(out/'snapshots.json', snapshots)
        journal.boundary(rt)

    rt = OriginalComparisonRuntime(policy, reallocate=args.arm == 'reallocated',
        seed=args.seed, intervention_s=window[0], config=config, stop_s=stop,
        on_boundary=boundary)
    write_json(out/'random-streams.json', dict(seeds=rt.role_seeds,
        coupling='identical per-role stream seeds in both arms; no shared market/crane stream'))
    started, cf_before = time.perf_counter(), rollout_calls()
    try:
        try:
            result = run_month(seed=args.seed, days=days, ppo=rt, seed_data=document,
                on_admission=journal.admission, on_day=journal.day_report,
                on_container_contract=journal.container_contract)
            write_json(out/'month-result.json', asdict(result))
        except DebugStop:
            if not smoke or rt.time_s != stop:
                raise
        assert rollout_calls() == cf_before
        assert not rt.updates and not rt.buffer and not rt.training
        assert parameter_hash(policy) == initial_hash
        assert file_sha256(args.seed_bundle) == input_hash
        assert file_sha256(args.checkpoint) == checkpoint_hash
        for sim in rt.mbt.blocks.values():
            sim.check_invariants()
        journal.save_admissions()
        assert journal.admissions['skipped'] == journal.admissions['vessel_failed'] == 0
        expected = (sum((max(0., entry['arrival_s']-entry['lead_s'])//60)*60 <= stop
                        for entry in document['schedule']) if smoke else len(document['schedule']))
        assert journal.admissions['admitted'] == expected
        assignments = assignment_audit(document['orders'], rt.bridge.orders)
        if args.arm == 'original':
            assert assignments['original_fields_preserved']
            assert rt.bridge.traded_edges == rt.bridge.n_space == rt.bridge.n_time == 0
            assert rt.role_counts['buyer'] == 0
        save_checkpoint(out/'policy-final.pt', rt)
        restored = load_policy(out/'policy-final.pt')
        assert parameter_hash(restored) == initial_hash
        save_compressed(out/'orders-and-records.json.gz', dict(
            orders=[asdict(order) for order in rt.bridge.orders.values()],
            records=[asdict(record) | {'_stamped': sorted(stage.name for stage in record._stamped)}
                     for record in rt.bridge.records.values()]))
        write_json(out/'physical-trace.json', dict(keys=KEYS, rows=trace))
        first, last = (snapshots[str(int(t))] for t in window)
        measured = {key: last['physical'][key]-first['physical'][key] for key in KEYS}
        report = rt.report() | dict(status='complete', arm=args.arm, seed=args.seed,
            smoke=smoke, protocol_complete=True, parameter_unchanged=True,
            initial_parameter_sha256=initial_hash, final_parameter_sha256=parameter_hash(policy),
            checkpoint_reload_exact=True, assignments=assignments,
            forced_market=dict(rt.forced_market), measurement_window_s=list(window),
            measured_physical=measured, measured_objective=normalized_loss(measured, reference),
            measured_original_request_s=last['request']['original_request_s']-first['request']['original_request_s'],
            evaluation_endpoint=snapshots[str(int(endpoint))],
            input_file_unchanged=True, checkpoint_file_unchanged=True, invariants=True,
            admission=journal.admissions, cf_calls=0, source=stamp,
            elapsed_s=time.perf_counter()-started, performance_claim=False)
        write_json(out/'report.json', report)
        write_json(out/'status.json', dict(status='complete', time_s=rt.time_s, updates=0))
        print(json.dumps(dict(status='complete', arm=args.arm, seed=args.seed,
                              measured_objective=report['measured_objective'])), flush=True)
    except BaseException as exc:
        journal.fail(exc, rt)
        write_json(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def campaign(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'prereg-executed.md'
    snapshot.write_bytes(SPEC.read_bytes())
    data = Path('outputs/reports/yr331_training').resolve()
    models = Path('outputs/reports/yr331_operational/campaign').resolve()
    jobs, finished = [], []
    pairs = list(zip(TRAIN_SEEDS, EVAL_SEEDS))[:1] if args.smoke else zip(TRAIN_SEEDS, EVAL_SEEDS)
    for train_seed, eval_seed in pairs:
        for arm in ('original', 'reallocated'):
            name = f'eval-{eval_seed}-{arm}'
            command = [sys.executable, str(Path(__file__).resolve()), '--seed', str(eval_seed),
                '--arm', arm, '--seed-bundle', str(data/f'eval-{eval_seed}-cost/seed-data.json.gz'),
                '--checkpoint', str(models/f'train-{train_seed}-operational/policy-final.pt'),
                '--prereg', str(snapshot), '--output', str(root/name)]
            if args.smoke:
                command.append('--smoke')
            jobs.append(dict(name=name, command=command))
    write_json(root/'registered.json', dict(jobs=jobs, expected_runs=len(jobs), smoke=args.smoke))

    def invoke(job):
        with (root/f"{job['name']}.log").open('x', encoding='utf-8') as log:
            rc = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT).returncode
        return job | dict(exit_code=rc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(invoke, job) for job in jobs]):
            row = future.result()
            finished.append(row)
            write_json(root/'progress.json', dict(completed=finished, registered_runs=len(jobs)))
            print(json.dumps(dict(name=row['name'], exit_code=row['exit_code'])), flush=True)
    complete = len(finished) == len(jobs) and all(row['exit_code'] == 0 for row in finished)
    write_json(root/'campaign.json', dict(completed=finished, registered_runs=len(jobs), all_completed=complete))
    if not complete:
        raise RuntimeError('Registered comparison incomplete; preserve failed runs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['original', 'reallocated'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed-bundle')
    parser.add_argument('--prereg')
    args = parser.parse_args()
    campaign(args) if args.campaign else run_one(args)
