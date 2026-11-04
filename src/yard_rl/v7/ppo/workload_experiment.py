"""Reproducible v6 workload reward pilot; run from a clean source worktree."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import torch

from ..reward.counterfactual import rollout_calls
from ..reward.workload import WorkloadConfig, WorkloadMeter
from ..stage.episode import _rule_policy
from ..stage.fixed_seed import build_fixed_seed
from ..stage.month import build_month, plan_days, plan_month_vessels, truck_net_by_block
from ..stage.month_run import run_month
from ..stage.seed_bundle import save_seed_bundle, load_seed_bundle
from ..world.integrated.terminal_stream import OBS_24H
from ..world.integrated.yard_layout import terminal_layout
from .checkpoint import load_policy, save_checkpoint
from .journal import RunJournal, write_json
from .model import BlockPolicy
from .provenance import code_stamp, file_sha256
from .runtime import DebugStop, PPOConfig, PPORuntime
from .workload_observer import Observer


class ReferenceRuntime(PPORuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rule, self.rule_exceptions = _rule_policy('SF_SPT')
        def execute(sim, dp):
            rule(sim, dp)
            if self.rule_exceptions['n']:
                raise RuntimeError('Reference rule fell back after a policy exception')
        self.execute = execute

    def select(self, role, bid, t, rows, mask=None):
        self.role_counts[role] += 1
        return 0 if role == 'seller' else 1  # KEEP / REJECT, never reached buyer normally.


def run(args):
    torch.set_num_threads(1)
    stamp = code_stamp()
    prereg = Path(args.prereg)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    days = plan_days(args.seed, loads=args.loads)
    config = PPOConfig(reward_mode='legacy-krw', reward_scale_krw=args.reward_scale_krw)
    registration = json.loads(prereg.read_text(encoding='utf-8'))
    if registration.get('schema') == 'yr331.pilot.v1' and config.reward_scale_krw != 1_000_000.:
        raise ValueError('YR-331-b preregistered a historical scale: use --reward-scale-krw 1000000 '
                         'for replay or provide a new preregistration for the new formula')
    if args.calibration:
        frozen = json.loads(Path(args.calibration).read_text(encoding='utf-8'))['config']
        workload = WorkloadMeter(WorkloadConfig(**frozen)) if args.arm == 'workload' else None
    else:
        workload = None
    if args.arm == 'workload' and workload is None:
        raise ValueError('Workload arm requires frozen calibration')
    torch.manual_seed(args.init_seed or args.seed)
    policy = load_policy(args.checkpoint) if args.checkpoint else BlockPolicy()
    manifest = dict(generation='v6', scope='CPU diagnostic pilot', code=stamp,
        prereg_sha256=file_sha256(prereg), arguments=vars(args), ppo=asdict(config),
        learning_window_s=[86400, (len(days)-1)*86400],
        calibration_sha256=file_sha256(args.calibration) if args.calibration else None,
        checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
        days=[d.as_dict() for d in days])
    journal = RunJournal(out, days, manifest)
    observer = Observer(out, days, calibration=args.mode == 'calibrate')
    start, calls = time.perf_counter(), rollout_calls()
    rt = None
    try:
        if args.seed_bundle:
            document, audit = load_seed_bundle(args.seed_bundle,
                                               expected_sha256=file_sha256(args.seed_bundle))
        else:
            built = build_month(args.seed, days=days)
            ships = plan_month_vessels(days, terminal_layout(), obs=OBS_24H,
                                      truck_net=truck_net_by_block(built['schedule']))
            document, audit = build_fixed_seed(built, ships, seed=args.seed)
        artifact = save_seed_bundle(document, out / 'seed-data.json.gz')
        write_json(out / 'input-audit.json', audit | {'artifact': artifact})
        rt_class = ReferenceRuntime if args.arm == 'rule' else PPORuntime
        def boundary(runtime):
            observer(runtime)
            journal.boundary(runtime)
        rt = rt_class(policy, config=config, seed=args.seed,
            training=args.mode == 'train', sample_actions=True, workload=workload,
            stop_s=args.stop_s, learning_window_s=(86400, (len(days)-1)*86400),
            on_boundary=boundary, on_update=journal.update)
        rt.original_requests = {o['doc_key']: (o['in_out_reserve_s'], o['copino_notice_s'])
                                for o in document['orders']}
        initial = {k: v.clone() for k, v in policy.state_dict().items()}
        interrupted = False
        try:
            result = run_month(seed=args.seed, days=days, ppo=rt, seed_data=document,
                on_day=journal.day_report, on_admission=journal.admission,
                on_container_contract=journal.container_contract)
            write_json(out / 'month-result.json', asdict(result))
        except DebugStop:
            if args.stop_s is None:
                raise
            interrupted = True
        for sim in rt.mbt.blocks.values():
            sim.check_invariants()
        if rollout_calls() != calls:
            raise RuntimeError('Counterfactual rollout entered a pure PPO run')
        telemetry = observer.finish(rt)
        report = rt.report() | dict(generation='v6', status='smoke' if interrupted else 'complete',
            wall_s=time.perf_counter()-start, mode=args.mode, arm=args.arm, seed=args.seed,
            optimizer_updates=len(rt.updates), telemetry=telemetry, cargo=rt.mbt.cargo_report(),
            parameter_l2_change=float(sum(((v-initial[k])**2).sum() for k,v in policy.state_dict().items()).sqrt()),
            cf_calls=0, invariants=True, admission=journal.admissions,
            source_stamp=stamp, seed_artifact=artifact)
        if args.mode == 'train' and not interrupted and len(rt.updates) != 24*(len(days)-2):
            raise RuntimeError('Missing registered PPO updates')
        save_checkpoint(out / 'policy-final.pt', rt)
        report['checkpoint_sha256'] = file_sha256(out / 'policy-final.pt')
        if args.mode == 'calibrate':
            write_json(out / 'calibration.json', observer.calibration_report())
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(status=report['status'], wall_s=report['wall_s'], time_s=rt.time_s))
        journal.event('finished', dict(time_s=rt.time_s, cost_krw=rt.cost_krw, updates=len(rt.updates)))
        return report
    except Exception as exc:
        write_json(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            details=getattr(exc, 'report', None),
            wall_s=time.perf_counter()-start, time_s=rt.time_s if rt else None))
        write_json(out / 'status.json', dict(status='failed', error=repr(exc)))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['calibrate', 'train', 'eval'], required=True)
    p.add_argument('--arm', choices=['rule', 'cost', 'workload'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--init-seed', type=int)
    p.add_argument('--reward-scale-krw', type=float,
                   help='Explicit historical scale for replay; default uses frozen reference data')
    p.add_argument('--loads', type=int, nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prereg', default='docs/research/v6-workload-reward/training-prereg.json')
    p.add_argument('--calibration')
    p.add_argument('--checkpoint')
    p.add_argument('--seed-bundle')
    p.add_argument('--stop-s', type=float)
    args = p.parse_args()
    if len(args.loads) < 3 or not 9900000 <= args.seed < 9990000:
        p.error('Use at least three days and a reserved diagnostic seed')
    if args.arm == 'rule' and args.mode == 'train':
        p.error('Rule reference cannot train')
    if args.mode == 'eval' and args.arm != 'rule' and not args.checkpoint:
        p.error('Frozen evaluation needs a trained checkpoint')
    run(args)


if __name__ == '__main__':
    main()
