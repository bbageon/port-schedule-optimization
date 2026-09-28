"""Collect a fixed-rule physical reference before fitting or training a policy."""
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
from yard_rl.v6.reward.operational import KEYS, runtime_totals, fit_reference
from yard_rl.v6.stage.month import plan_days
from yard_rl.v6.stage.month_run import run_month
from yard_rl.v6.stage.seed_bundle import load_seed_bundle


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(9911000)
    stamp = code_stamp()
    document, audit = load_seed_bundle(args.seed_bundle, expected_sha256=file_sha256(args.seed_bundle))
    config = PPOConfig(reward_mode='legacy-krw', reward_scale_krw=1.)  # Reference never learns.
    days = plan_days(9911000, loads=[3500,5000,7500])
    out = Path(args.output)
    manifest = dict(code=stamp, seed=9911000, ppo=asdict(config),
        prereg=args.prereg, prereg_sha256=file_sha256(args.prereg),
        input_sha256=file_sha256(args.seed_bundle), input_audit=audit,
        rule='KEEP+SF_SPT', scope='physical normalization calibration, no performance claim')
    journal = RunJournal(out, days, manifest)
    rows = []
    def boundary(rt):
        if not rows or rows[-1][0] != rt.time_s:
            values = runtime_totals(rt, rt.time_s)
            rows.append([rt.time_s]+[values[k] for k in KEYS])
        journal.boundary(rt)
    rt = ReferenceRuntime(BlockPolicy(), config=config, seed=9911000, training=False, on_boundary=boundary)
    try:
        result = run_month(seed=9911000, days=days, ppo=rt, seed_data=document,
            on_admission=journal.admission, on_container_contract=journal.container_contract,
            on_day=journal.day_report)
        for sim in rt.mbt.blocks.values():
            sim.check_invariants()
        assert result.admitted == len(document['schedule']) and not result.skipped
        assert not result.policy_exceptions and not rt.updates and not rt.rule_exceptions['n']
        rows = [r for r in rows if r[0] <= 259200]
        assert [r[0] for r in rows] == list(range(0,259201,60))
        trace = dict(schema='yard.v6.operational-reference.v1', keys=KEYS, seed=9911000,
            code=stamp, input_sha256=manifest['input_sha256'], fit_start_s=86400., fit_end_s=259200., rows=rows)
        write_json(out/'operational_reference.json', trace)
        fitted = fit_reference(rows)
        write_json(out/'scales.json', fitted)
        write_json(out/'report.json', dict(status='complete', admitted=result.admitted, skipped=0,
            invariants=True, optimizer_updates=0, code=stamp, fit=fitted, minute_boundaries=len(rows)))
        write_json(out/'status.json', dict(status='complete', time_s=rt.time_s))
        print(json.dumps(fitted), flush=True)
    except BaseException as exc:
        journal.fail(exc, rt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed-bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prereg', default='.claude/docs/dashboard-task-specs/YR-331-e-operational-normalization.md')
    run(p.parse_args())
