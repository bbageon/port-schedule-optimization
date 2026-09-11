"""First continuous v5 learning run: 1 warmup + N-2 learning + 1 cooldown day."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path

import torch

from .. import V4_BASE_COMMIT
from ..eval.guards import DIAGNOSTIC_BAND
from ..reward.counterfactual import rollout_calls
from ..stage.month import DAY_S, DayPlan, MONTH_FILL_RATIO, plan_month
from ..stage.month_run import run_month
from .journal import RunJournal, write_json
from .model import BlockPolicy
from .provenance import code_stamp
from .runtime import PPOConfig, PPORuntime


def make_plan(seed, n_days, load=None):
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or seed // 100_000 * 100_000 != DIAGNOSTIC_BAND):
        raise ValueError("Use an integer diagnostic seed in [9900000, 9999999]")
    if isinstance(n_days, bool) or not isinstance(n_days, int) or not 3 <= n_days <= 30:
        raise ValueError("Continuous run requires 3 to 30 days")
    if seed + 1000 * n_days > 9999999:
        raise ValueError("Daily seeds must stay within the diagnostic band")
    if load is not None:
        if isinstance(load, bool) or not isinstance(load, int) or load < 1:
            raise ValueError("Debug load must be a positive integer")
        return [DayPlan(i, load, "fixed-load-debug", seed + 1000 * (i + 1), i * DAY_S, n_days)
                for i in range(n_days)]
    return plan_month(seed, n_days=n_days)


def run_continuous(*, output, seed=9900306, n_days=30, load=None, config=None):
    days = make_plan(seed, n_days, load)
    config = config or PPOConfig()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a NEW output directory")
    torch.set_num_threads(1)
    stamp = code_stamp()  # Refuse dirty or misidentified source BEFORE writing outputs.
    torch.manual_seed(seed)
    policy = BlockPolicy()
    manifest = {"generation": "v5", "algorithm": "shared-block-PPO",
                "purpose": "first-continuous-execution-validation; NO performance claim",
                "code": stamp, "v4_base_commit": V4_BASE_COMMIT,
                "seed": seed, "days": [d.as_dict() for d in days], "ppo": asdict(config),
                "learning_window_s": [DAY_S, (n_days - 1) * DAY_S],
                "initial_occupancy": MONTH_FILL_RATIO, "action_mode": "sample-all-days",
                "day_metric": "calendar interval cost, not arrival-cohort cost",
                "counterfactual_worlds_allowed": 0,
                "recovery": "day checkpoints are weights only, not physical world resume"}
    journal = RunJournal(output, days, manifest)
    runtime = PPORuntime(policy, config=config, seed=seed, training=True,
                         learning_window_s=manifest["learning_window_s"],
                         on_update=journal.update, on_boundary=journal.boundary)
    before_cf = rollout_calls()
    try:
        initial = journal.checkpoint("initial.pt", runtime)
        result = run_month(seed=seed, days=days, ppo=runtime,
                           on_admission=journal.admission, on_day=journal.day_report)
        # Keep the returned evidence even when a final validation rejects the run.
        write_json(output / "month_result.json", asdict(result))
        write_json(output / "cohort_reports.json", {"note": "Inherited cohort metric; distinct from calendar rewards",
                                                     "live": [d.as_dict() for d in result.live],
                                                     "final": [d.as_dict() for d in result.days]})
        journal.admissions.update(admitted=result.admitted, skipped=result.skipped,
            vessels=result.vessel_admissions,
            vessel_failed=sum(not a["ok"] for a in result.vessel_admissions))
        if result.truck_skips and not journal.admissions["truck_failures"]:
            journal.admissions["truck_failures"] = result.truck_skips
        journal.save_admissions()
        for sim in runtime.mbt.blocks.values():
            sim.check_invariants()
        cf = rollout_calls() - before_cf
        if cf or result.policy_exceptions:
            raise RuntimeError("Continuous PPO invoked a counterfactual world or policy exception")
        expected = (n_days - 2) * int(DAY_S // 60)
        if runtime.learning_intervals != expected or len(journal.daily) != n_days:
            raise RuntimeError("Learning/day boundaries did not cover exactly the registered window")
        learning_cost = sum(d["interval_cost_krw"] for d in journal.daily if d["train"])
        if not math.isclose(-runtime.learning_reward * config.reward_scale_krw,
                            learning_cost, rel_tol=1e-10, abs_tol=1e-4):
            raise RuntimeError("Learning rewards do not telescope to the calendar-window cost")
        if any(d["updates"] or d["learning_reward"] for d in journal.daily if not d["train"]):
            raise RuntimeError("Warmup/cooldown leaked into learning")
        if result.skipped or any(not a["ok"] for a in result.vessel_admissions):
            raise RuntimeError("Some external trucks or vessels could not be admitted; inspect logs")
        final = journal.checkpoint("final.pt", runtime)
        report = runtime.report()
        report.update(state="completed", counterfactual_worlds=cf, n_days=n_days,
                      learning_days=n_days - 2, learning_interval_cost_krw=learning_cost,
                      admitted=result.admitted, skipped=result.skipped,
                      initial_checkpoint=initial, final_checkpoint=final,
                      engine_end_s=max(s.end for s in runtime.mbt.blocks.values()),
                      claim_scope="NO_PERFORMANCE_CLAIM", code=stamp,
                      restart="No physical-world resume; restart same seed in a NEW directory")
        write_json(output / "report.json", report)
        write_json(output / "status.json", {"state": "completed", "time_s": runtime.time_s,
                                             "completed_days": n_days, "updates": len(runtime.updates)})
        journal.event("completed", {"time_s": runtime.time_s, "updates": len(runtime.updates),
                                     "counterfactual_worlds": cf})
        return runtime, report
    except BaseException as error:
        try:
            journal.fail(error, runtime)
        except Exception as log_error:
            error.add_note(f"Failure journal could not be saved: {log_error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9900306)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--debug-load", type=int, help="Fixed low load for short wiring tests only")
    args = parser.parse_args(argv)
    run_continuous(output=args.output, seed=args.seed, n_days=args.days, load=args.debug_load)


if __name__ == "__main__":
    main()
