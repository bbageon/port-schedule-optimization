"""YR-306 reference evaluation; frozen src/v5, fixed cargo, no policy updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from yard_rl.v5.ppo.checkpoint import load_policy
from yard_rl.v5.ppo.journal import RunJournal, write_json
from yard_rl.v5.ppo.provenance import code_stamp, file_sha256
from yard_rl.v5.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v5.reward.counterfactual import rollout_calls
from yard_rl.v5.stage.cargo_input import restore_input
from yard_rl.v5.stage.episode import _rule_policy
from yard_rl.v5.stage.month import DAY_S, DayPlan
from yard_rl.v5.stage.month_run import run_month
from yard_rl.v5.stage.seed_bundle import load_seed_bundle

ARMS = ('final', 'initial', 'rule')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def policy_digest(policy):
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def savings(candidate, reference):
    require(all(math.isfinite(v) and v >= 0 for v in (candidate, reference)),
            'Costs must be finite and nonnegative')
    difference = reference - candidate
    return dict(saving_krw=difference,
                saving_percent=None if reference == 0 else difference / reference * 100)


class RuleRuntime(PPORuntime):
    """Use the existing actor seam: KEEP + unmodified existing SF-SPT executor."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        require(not self.training, 'Rules may not train')
        rule, errors = _rule_policy('SF_SPT')
        self.rule_errors = errors

        def execute(sim, dp):
            bid = self.block_of[id(sim)]
            # Exactly the same realized-event synchronization as CraneActor.
            self.bridge._sync(self.mbt, sim.now, block_ids=(bid,))
            rule(sim, dp)
            require(errors['n'] == 0, 'SF-SPT exception fallback invalidates comparison')
            for assignment in sim.last_assignments().values():
                self.crane_actions[assignment.action.name] += 1
                self.role_counts['crane'] += 1

        self.execute = execute

    def select(self, role, bid, t, rows, mask=None):
        require(role == 'seller', 'KEEP-only baseline must never receive a buyer offer')
        require(len(rows) > 0 and (mask is None or bool(mask[0])), 'KEEP row unavailable')
        require(len(rows[0]) == 21 and float(rows[0][12]) == 1.0,
                'Seller KEEP encoding changed')
        self.role_counts[role] += 1
        return 0  # Seller.decide defines coords[0] = None (KEEP).


class EvaluationJournal(RunJournal):
    def __init__(self, output, days, manifest, expected_digest):
        self.expected_digest = expected_digest
        self.last_cost = 0.0
        super().__init__(output, days, manifest)

    def event(self, kind, row):
        if kind == 'start':
            row = {**row, 'learning_days': 0, 'evaluation_days': len(self.days)}
        super().event(kind, row)

    def boundary(self, runtime):
        t = runtime.time_s
        if self.last_time == t:
            return
        self.last_time = t
        require(not runtime.training and not runtime.updates and not runtime.buffer,
                'Frozen evaluation collected learning data')
        if t > 0 and t % DAY_S == 0 and t <= len(self.days) * DAY_S:
            require(policy_digest(runtime.policy) == self.expected_digest, 'Policy changed')
            for sim in runtime.mbt.blocks.values():
                sim.check_invariants()
            day = self.days[int(t // DAY_S) - 1]
            row = dict(day=day.index + 1, load=day.load, time_s=t, training=False,
                       measurement_day=0 < day.index < len(self.days) - 1,
                       interval_cost_krw=runtime.cost_krw - self.last_cost,
                       cost_krw=runtime.cost_krw, cost_breakdown=runtime.cost_breakdown,
                       cargo=runtime.mbt.cargo_report(), updates=0,
                       policy_digest=self.expected_digest, physical_invariants_checked=True,
                       roles=dict(runtime.role_counts), crane_actions=dict(runtime.crane_actions))
            self.daily.append(row)
            self.last_cost = runtime.cost_krw
            write_json(self.output / 'days.json', self.daily)
            self.event('day', row)
        if t % 3600 == 0:
            self.save_admissions()
            state = dict(state='running', phase='frozen-evaluation', time_s=t,
                         completed_days=len(self.daily), updates=0, cost_krw=runtime.cost_krw,
                         admitted=self.admissions['admitted'], skipped=self.admissions['skipped'],
                         wall_seconds=time.perf_counter() - self.started)
            write_json(self.output / 'status.json', state)
            self.event('progress', state)


def residual_summary(runtime):
    jobs = [job for sim in runtime.mbt.blocks.values() for job in sim.jobs.values()
            if job.status.value != 'DONE']
    vessels = [v for sim in runtime.mbt.blocks.values() for v in sim.vessels.values()]
    return dict(unfinished_jobs=len(jobs),
                overdue_vessel_yard_jobs=sum(j.vessel_id is not None and j.deadline is not None
                                             and j.deadline < runtime.time_s for j in jobs),
                unfinished_streams_past_planned_completion=sum(not v.done and
                    v.plan.planned_completion_s is not None and
                    v.plan.planned_completion_s < runtime.time_s for v in vessels),
                caveat='Job deadline counts and ship-stream completion are distinct; no feasibility verdict')


def run_arm(*, arm, original_run, seed_bundle, output):
    original_run, seed_bundle, output = map(Path, (original_run, seed_bundle, output))
    require(arm in ARMS, 'Unknown arm')
    require(not output.exists(), 'Refusing to overwrite an evaluation')
    torch.set_num_threads(1)
    require(len(os.sched_getaffinity(0)) == 1, 'Evaluation must use one CPU')
    stamp = code_stamp()
    original, original_report = read(original_run / 'manifest.json'), read(original_run / 'report.json')
    require(original_report['state'] == 'completed', 'Original training must be complete')
    for key in ('source_sha256', 'python', 'torch', 'numpy'):
        require(stamp[key] == original['code'][key], f'Original {key} changed')
    originals = {p.name: file_sha256(p) for p in original_run.iterdir() if p.is_file()}
    document, _ = load_seed_bundle(seed_bundle, expected_sha256=original['fixed_seed']['sha256'])
    days = [DayPlan(d['index'], d['load'], d['label'], d['seed'], d['t0'], d['n_days'])
            for d in original['days']]
    restore_input(document, seed=original['seed'], days=days, lead_mode='DIST')
    checkpoint_name = 'final.pt' if arm == 'final' else 'initial.pt'
    checkpoint = original_run / checkpoint_name
    reference_key = 'final_checkpoint' if arm == 'final' else 'initial_checkpoint'
    require(file_sha256(checkpoint) == original_report[reference_key]['sha256'], 'Checkpoint changed')
    torch.manual_seed(original['seed'])
    policy = load_policy(checkpoint)
    torch.manual_seed(original['seed'])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    digest = policy_digest(policy)
    manifest = dict(task='YR-306', purpose='USER_AUTHORIZED_REFERENCE_ONLY', arm=arm,
                    formal_gate_override=False, user_sequence_exception_date='2026-09-14',
                    code=stamp, source_original=original['code'], seed=original['seed'],
                    days=original['days'], ppo=original['ppo'], training=False,
                    action_mode='argmax' if arm != 'rule' else 'KEEP+existing-SF-SPT',
                    checkpoint=dict(path=str(checkpoint.resolve()), sha256=originals[checkpoint_name],
                                    observer_only=arm == 'rule'), policy_digest=digest,
                    fixed_seed=dict(path=str(seed_bundle.resolve()), sha256=original['fixed_seed']['sha256']),
                    measurement_window_s=[DAY_S, (len(days) - 1) * DAY_S],
                    input_scope='training-input reuse, not held-out generalization',
                    original_files_sha256=originals)
    journal = EvaluationJournal(output, days, manifest, digest)
    runtime_type = RuleRuntime if arm == 'rule' else PPORuntime
    runtime = runtime_type(policy, config=PPOConfig(**original['ppo']), seed=original['seed'],
                           training=False, on_boundary=journal.boundary)
    before_cf = rollout_calls()
    try:
        result = run_month(seed=original['seed'], days=days, ppo=runtime, seed_data=document,
                           on_admission=journal.admission, on_day=journal.day_report,
                           on_container_contract=journal.container_contract)
        write_json(output / 'month_result.json', asdict(result))
        write_json(output / 'cohort_reports.json', dict(
            note='Inherited cohort train flag describes original input phase; evaluation never trains',
            final=[d.as_dict() for d in result.days]))
        cargo = runtime.mbt.cargo_report()
        checks = dict(
            frozen_policy=policy_digest(policy) == digest,
            no_updates=not runtime.updates and runtime.learning_intervals == 0,
            no_counterfactual=rollout_calls() == before_cf,
            all_trucks=result.admitted == len(document['schedule']) and result.skipped == 0,
            all_vessels=len(result.vessel_admissions) == sum(map(len, document['vessels_by_day'].values()))
                        and all(v['ok'] for v in result.vessel_admissions),
            no_policy_exceptions=result.policy_exceptions == 0 and
                                 getattr(runtime, 'rule_errors', {'n': 0})['n'] == 0,
            inventory_balance=cargo['physical_inventory'] == cargo['expected_inventory'],
            all_days=len(journal.daily) == len(days),
            fixed_end=runtime.time_s == len(days) * DAY_S + 7200,
            original_files_preserved=all(file_sha256(original_run / name) == value
                                        for name, value in originals.items()),
            seed_preserved=file_sha256(seed_bundle) == original['fixed_seed']['sha256'])
        for sim in runtime.mbt.blocks.values():
            sim.check_invariants()
        checks['physical_invariants'] = True
        measurement = math.fsum(d['interval_cost_krw'] for d in journal.daily if d['measurement_day'])
        report = {**runtime.report(), 'state': 'completed' if all(checks.values()) else 'invalid',
                  'arm': arm, 'training': False, 'claim_scope': 'REFERENCE_ONLY', 'checks': checks,
                  'measurement_cost_krw': measurement, 'cargo': cargo,
                  'residual': residual_summary(runtime), 'admitted': result.admitted,
                  'skipped': result.skipped, 'code': stamp,
                  'wall_seconds': time.perf_counter() - journal.started}
        write_json(output / 'report.json', report)
        require(all(checks.values()), f'Evaluation checks failed: {checks}')
        write_json(output / 'status.json', dict(state='completed', time_s=runtime.time_s,
                                                completed_days=len(days), updates=0))
        journal.event('completed', dict(arm=arm, cost_krw=runtime.cost_krw,
                                       measurement_cost_krw=measurement, checks=checks))
        return report
    except BaseException as error:
        journal.fail(error, runtime)
        raise


def summarize(root):
    root = Path(root)
    reports = {arm: read(root / arm / 'report.json') for arm in ARMS}
    manifests = {arm: read(root / arm / 'manifest.json') for arm in ARMS}
    for arm in ARMS:
        report = reports[arm]
        require(report['state'] == 'completed' and all(report['checks'].values()), 'Invalid arm')
        for key in ('seed', 'days', 'fixed_seed', 'measurement_window_s'):
            require(manifests[arm][key] == manifests['final'][key], f'Unpaired {key}')
        require(manifests[arm]['code']['source_sha256'] == manifests['final']['code']['source_sha256'],
                'Environment source differs')
    comparisons = {}
    for candidate, reference in (('final', 'rule'), ('initial', 'rule'), ('final', 'initial')):
        comparisons[candidate + '_vs_' + reference] = {
            'measurement_window': savings(reports[candidate]['measurement_cost_krw'],
                                               reports[reference]['measurement_cost_krw']),
            'whole_window': savings(reports[candidate]['cost_krw'], reports[reference]['cost_krw'])}
    return dict(state='completed', claim_scope='REFERENCE_ONLY', formal_performance_pass=False,
                measurement_window_s=manifests['final']['measurement_window_s'],
                limitations=['training data reused', 'one dependent month', 'argmax not sampling',
                             'WAIT and current cost issues unchanged', 'no operational validity claim'],
                comparisons=comparisons,
                arms={arm: {k: reports[arm][k] for k in ('cost_krw', 'measurement_cost_krw',
                       'cost_breakdown', 'cargo', 'residual', 'checks')} for arm in ARMS},
                artifact_sha256={f'{arm}/{name}': file_sha256(root / arm / name)
                                 for arm in ARMS for name in ('report.json', 'manifest.json', 'days.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--original-run', type=Path)
    parser.add_argument('--seed-bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        write_json(args.output / 'comparison.json', summarize(args.output))
    else:
        require(args.arm and args.original_run and args.seed_bundle, 'Worker arguments missing')
        run_arm(arm=args.arm, original_run=args.original_run, seed_bundle=args.seed_bundle, output=args.output)


if __name__ == '__main__':
    main()
