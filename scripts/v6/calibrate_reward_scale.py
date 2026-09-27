"""Replay the preregistered reference rule and retain every minute's raw cost."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from yard_rl.v6.ppo.journal import RunJournal, write_json
from yard_rl.v6.ppo.model import BlockPolicy
from yard_rl.v6.ppo.provenance import code_stamp, file_sha256
from yard_rl.v6.ppo.runtime import PPOConfig
from yard_rl.v6.ppo.workload_experiment import ReferenceRuntime
from yard_rl.v6.reward.scaling import fit_reference_scale
from yard_rl.v6.stage.month import plan_days
from yard_rl.v6.stage.month_run import run_month
from yard_rl.v6.stage.seed_bundle import load_seed_bundle


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(9911000)
    stamp = code_stamp()
    seed = 9911000
    days = plan_days(seed, loads=[3500, 5000, 7500])
    document, audit = load_seed_bundle(args.seed_bundle, expected_sha256=file_sha256(args.seed_bundle))
    config = PPOConfig(reward_scale_krw=1.)  # Raw currency accounting; NO optimizer.
    out = Path(args.output)
    manifest = dict(code=stamp, seed=seed, ppo=asdict(config),
                    prereg=args.prereg, prereg_sha256=file_sha256(args.prereg),
                    input_sha256=file_sha256(args.seed_bundle), input_audit=audit,
                    rule='KEEP+SF_SPT', scope='reward-unit calibration, not capacity validation')
    journal = RunJournal(out, days, manifest)
    rows = []

    def boundary(rt):
        if not rows or rows[-1][0] != rt.time_s:
            rows.append([rt.time_s, rt.cost_krw])
        journal.boundary(rt)

    rt = ReferenceRuntime(BlockPolicy(), config=config, seed=seed, training=False,
                          on_boundary=boundary)
    try:
        result = run_month(seed=seed, days=days, ppo=rt, seed_data=document,
                           on_admission=journal.admission, on_day=journal.day_report,
                           on_container_contract=journal.container_contract)
        for sim in rt.mbt.blocks.values():
            sim.check_invariants()
        if result.admitted != len(document['schedule']) or result.skipped or result.policy_exceptions:
            raise RuntimeError('Reference dropped requests or encountered policy errors')
        if rt.updates or rt.rule_exceptions['n']:
            raise RuntimeError('Reference unexpectedly learned or fell back')
        rows = [row for row in rows if row[0] <= 259200]
        if [r[0] for r in rows] != list(range(0, 259201, 60)):
            raise RuntimeError('Reference minute grid is incomplete')
        trace = dict(schema='yard.v6.reference-cost-trace.v1', seed=seed,
                     code=stamp, rule=manifest['rule'], input_sha256=manifest['input_sha256'],
                     fit_start_s=86400., fit_end_s=259200., rows=rows)
        write_json(out/'reference_cost_trace.json', trace)
        fitted = fit_reference_scale(rows)
        write_json(out/'scale.json', fitted)
        write_json(out/'report.json', dict(status='complete', admitted=result.admitted,
            skipped=result.skipped, invariants=True, optimizer_updates=0,
            minute_boundaries=len(rows), final_cost_krw=rt.cost_krw,
            scale=fitted, code=stamp))
        write_json(out/'status.json', dict(status='complete', time_s=rt.time_s))
        print(json.dumps(fitted), flush=True)
    except BaseException as exc:
        journal.fail(exc, rt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed-bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prereg', default='.claude/docs/dashboard-task-specs/YR-331-d-reward-normalization.md')
    run(p.parse_args())
