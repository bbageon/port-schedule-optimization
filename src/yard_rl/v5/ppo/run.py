"""Bounded debugging entry point, separate from inherited CF experiment commands."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from ..eval.guards import DIAGNOSTIC_BAND
from .. import V4_BASE_COMMIT
from ..reward.counterfactual import rollout_calls
from ..stage.month import DAY_S, DayPlan
from ..stage.month_run import run_month
from .checkpoint import load_policy, save_checkpoint
from .model import BlockPolicy
from .runtime import DebugStop, PPOConfig, PPORuntime


def run_debug(*, seed=9900302, load=300, duration_s=7200.0,
              config=None, policy=None, training=True, on_update=None):
    if int(seed) // 100_000 * 100_000 != DIAGNOSTIC_BAND:
        raise ValueError("Debugging requires a diagnostic seed in [9900000, 9999999]")
    if load < 1 or not (0 < duration_s <= DAY_S):
        raise ValueError("Debug load must be positive and duration at most one simulated day")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    policy = policy if policy is not None else BlockPolicy()
    runtime = PPORuntime(policy, config=config, seed=seed, training=training,
                         stop_s=duration_s, on_update=on_update)
    # Always build two continuous days. The debug cutoff never modifies engine end.
    days = [DayPlan(index=i, load=load, label="v5-debug", seed=seed + 1000 * (i + 1),
                    t0=i * DAY_S, n_days=2) for i in range(2)]
    before_cf = rollout_calls()
    started = time.perf_counter()
    try:
        run_month(seed=seed, days=days, ppo=runtime)
    except DebugStop:
        pass
    report = runtime.report()
    source_hash = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for source in sorted(root.rglob("*.py")):
        source_hash.update(source.relative_to(root).as_posix().encode())
        source_hash.update(source.read_text(encoding="utf-8").encode())
    report.update(seed=seed, load=load, training=training,
                  v4_base_commit=V4_BASE_COMMIT, source_sha256=source_hash.hexdigest(),
                  requested_duration_s=duration_s, wall_seconds=time.perf_counter() - started,
                  counterfactual_worlds=rollout_calls() - before_cf,
                  engine_end_s=max(s.end for s in runtime.mbt.blocks.values()),
                  engine_policy_exceptions=0)
    if report["counterfactual_worlds"]:
        raise RuntimeError("PPO unexpectedly invoked a counterfactual world")
    return runtime, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="v5 unified PPO: short implementation/debug run")
    parser.add_argument("--seed", type=int, default=9900302)
    parser.add_argument("--load", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--rollout-intervals", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, help="Existing v5 weights for a new world")
    parser.add_argument("--eval", action="store_true", help="Greedy execution, no optimizer updates")
    parser.add_argument("--output", type=Path, required=True, help="NEW output directory")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if args.eval and args.checkpoint is None:
        parser.error("--eval requires --checkpoint")
    config = PPOConfig(rollout_intervals=args.rollout_intervals, epochs=args.epochs)
    policy = load_policy(args.checkpoint) if args.checkpoint else None
    runtime, report = run_debug(seed=args.seed, load=args.load, duration_s=args.seconds,
                                config=config, policy=policy, training=not args.eval,
                                on_update=lambda row: print(json.dumps({"update": row}), flush=True))
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.eval:
        save_checkpoint(args.output / "policy.pt", runtime)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
