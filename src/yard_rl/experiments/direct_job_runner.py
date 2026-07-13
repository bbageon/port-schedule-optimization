"""YR-027 external-truck Direct-Job Cost-Q experiment pipeline.

This runner is intentionally separate from the legacy nine-rule ``YardEnv``
pipeline.  It owns distinct seed bands, train-only bucket fitting, validation
checkpoint selection, and locked paired testing for the two SLA arms.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable, Sequence

from ..domain.models import TerminalProfile
from ..domain.scenario import Scenario
from ..envs.direct_job_env import (DirectJobBucketConfig, DirectJobEnv,
                                   DirectJobStepInfo, SLAMode)
from ..experiments.runner import collect_metrics
from ..io.profile_loader import load_profile
from ..io.scenario_gen import GenParams, generate
from ..policies.cost_q import CostQAgent, CostQConfig
from ..policies.direct_baselines import (DirectJobRulePolicy, DirectRule,
                                         direct_baseline_policies)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

STRATEGY_ID = "YR-027-v1"
POLICY_COST_Q = "CostQ+GreedyFallback"
PRIMARY_ARM = SLAMode.OFF.value
DEFAULT_DIRECT_PROFILE = "configs/terminals/hjnc_armg.yaml"


@dataclass(frozen=True)
class DirectExperimentConfig:
    train_episodes: int = 1_000
    validation_episodes: int = 30
    test_episodes: int = 100
    checkpoint_every: int = 50
    n_external: int = 100
    learning_rate_powers: tuple[float, ...] = (0.6, 0.8, 1.0)
    train_seed0: int = 10_000
    validation_seed0: int = 20_000
    test_seed0: int = 30_000
    bootstrap_seed: int = 72_027
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    quick: bool = False
    check_train_invariants: bool = True

    def __post_init__(self) -> None:
        positive = {
            "train_episodes": self.train_episodes,
            "validation_episodes": self.validation_episodes,
            "test_episodes": self.test_episodes,
            "checkpoint_every": self.checkpoint_every,
            "n_external": self.n_external,
            "bootstrap_resamples": self.bootstrap_resamples,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.test_episodes < 2:
            raise ValueError("paired bootstrap requires at least two test episodes")
        if not self.learning_rate_powers:
            raise ValueError("at least one learning-rate power is required")
        if any(not 0.0 < value <= 1.0 for value in self.learning_rate_powers):
            raise ValueError("learning-rate powers must be in (0, 1]")
        if self.drain_window_s <= 0.0 or not math.isfinite(self.drain_window_s):
            raise ValueError("drain_window_s must be finite and positive")
        splits = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(left & right for i, left in enumerate(splits)
               for right in splits[i + 1:]):
            raise ValueError("train/validation/test seed bands must be disjoint")

    @property
    def train_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.train_seed0, self.train_seed0 + self.train_episodes))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.validation_seed0,
                           self.validation_seed0 + self.validation_episodes))

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.test_seed0, self.test_seed0 + self.test_episodes))

    @property
    def strategy_compliant(self) -> bool:
        return (
            not self.quick
            and self.train_episodes >= 1_000
            and self.validation_episodes >= 30
            and self.test_episodes >= 100
            and self.checkpoint_every == 50
            and self.learning_rate_powers == (0.6, 0.8, 1.0)
            and self.bootstrap_resamples >= 10_000
        )


def quick_direct_config() -> DirectExperimentConfig:
    """Small smoke setting; results are explicitly ineligible for a claim."""
    return DirectExperimentConfig(
        train_episodes=12,
        validation_episodes=4,
        test_episodes=4,
        checkpoint_every=4,
        learning_rate_powers=(0.6, 0.8, 1.0),
        bootstrap_resamples=500,
        quick=True,
    )


def direct_gen_params(cfg: DirectExperimentConfig) -> GenParams:
    return GenParams(
        n_external=cfg.n_external,
        n_vessel=0,
        drain_window_s=cfg.drain_window_s,
    )


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _scenario(profile: TerminalProfile, seed: int, params: GenParams,
              n_config: int) -> Scenario:
    scenario = generate(profile, seed, params)
    if "_v0_" not in scenario.scenario_id:
        raise AssertionError(f"scenario metadata is not n_vessel=0: {scenario.scenario_id}")
    if len(scenario.jobs) != n_config:
        raise AssertionError(
            f"fixed N_config mismatch at seed {seed}: {len(scenario.jobs)} != {n_config}"
        )
    if any(not job.is_external_truck for job in scenario.jobs):
        raise AssertionError(f"non-external job generated at seed {seed}")
    return scenario


def _scenario_descriptor(scenario: Scenario) -> dict[str, object]:
    canonical = {
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "horizon_s": scenario.horizon_s,
        "drain_window_s": scenario.drain_window_s,
        "jobs": [
            {
                "job_id": job.job_id,
                "flow": job.flow.value,
                "release_time": job.release_time,
                "actual_gate_in": job.actual_gate_in,
                "block_entry_s": job.actual_block_arrival,
                "provided_eta": job.provided_eta,
                "deadline": job.deadline,
                "target": job.target_container,
                "inbound_size": None if job.inbound_size is None else job.inbound_size.value,
                "inbound_load": None if job.inbound_load is None else job.inbound_load.value,
                "priority_class": job.priority_class,
            }
            for job in sorted(scenario.jobs, key=lambda item: item.job_id)
        ],
        "containers": [
            {
                "container_id": container.container_id,
                "size": container.size.value,
                "load_status": container.load_status.value,
                "block": container.block,
                "bay": container.bay,
                "row": container.row,
                "tier": container.tier,
                "work_available": container.work_available,
                "special_flags": sorted(container.special_flags),
            }
            for container in sorted(scenario.containers.values(),
                                    key=lambda item: item.container_id)
        ],
        "meta": scenario.meta,
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return {
        "seed": scenario.seed,
        "scenario_id": scenario.scenario_id,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _profile_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state() -> dict[str, object]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(command("git", "status", "--porcelain")),
    }


def _run_baseline_episode(policy: DirectJobRulePolicy, env: DirectJobEnv,
                          scenario: Scenario) -> tuple[dict[str, float | int | bool], list[str]]:
    state, info = env.reset(scenario)
    trace: list[str] = []
    while state is not None:
        candidate = policy.act(state, info.candidates)
        trace.append(candidate.job_id)
        state, _cost, _done, info = env.step(candidate)
    return _direct_metrics(env, info), trace


def _run_cost_q_episode(agent: CostQAgent, env: DirectJobEnv, scenario: Scenario,
                        *, train_episode: int | None = None
                        ) -> tuple[dict[str, float | int | bool], list[str]]:
    state, info = env.reset(scenario)
    trace: list[str] = []
    if train_episode is None:
        agent.reset_diagnostics()
    epsilon = 0.0 if train_episode is None else 1.0 / math.sqrt(train_episode + 1.0)
    while state is not None:
        candidate = (agent.act(state, info.candidates) if train_episode is None
                     else agent.act_train(state, info.candidates, epsilon))
        trace.append(candidate.job_id)
        next_state, cost, done, next_info = env.step(candidate)
        if train_episode is not None:
            agent.update(state, candidate, cost, next_state, next_info.candidates, done)
        state, info = next_state, next_info
    diagnostics = None if train_episode is not None else agent.diagnostics.as_dict()
    return _direct_metrics(env, info, diagnostics), trace


def _direct_metrics(env: DirectJobEnv, terminal_info: DirectJobStepInfo,
                    diagnostics: dict[str, int | float] | None = None
                    ) -> dict[str, float | int | bool]:
    base = collect_metrics(env)
    waits_s = env.sim.kpis.wait_samples_s
    n = env.n_config
    sla = env.profile.long_wait_sla_s
    over = sum(wait >= sla for wait in waits_s)
    identity_error = abs(base["mean_wait_min"] - env.cumulative_cost)
    result: dict[str, float | int | bool] = {
        "mean_wait_min": base["mean_wait_min"],
        "p50_wait_min": base["p50_wait_min"],
        "p90_wait_min": base["p90_wait_min"],
        "p95_wait_min": base["p95_wait_min"],
        "max_wait_min": base["max_wait_min"],
        "sla_over_count": over,
        "sla_over_rate": over / n,
        "queue_area_h": base["queue_area_h"],
        "travel_km": base["travel_km"],
        "rehandles": base["rehandles"],
        "completed_external": int(base["completed_external"]),
        "completion_rate": base["completed_external"] / n,
        "backlog": int(base["backlog"]),
        "n_decisions": int(base["n_decisions"]),
        "step_cost_sum": env.cumulative_cost,
        "cost_identity_error": identity_error,
        "episode_success": bool(terminal_info.episode_success),
        "invariants_ok": identity_error <= 1e-9 and bool(terminal_info.episode_success),
    }
    if diagnostics is None:
        result.update({
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "fully_covered_decisions": 0,
            "decision_coverage": 1.0,
            "signatures_checked": 0,
            "visited_signatures": 0,
            "signature_coverage": 1.0,
        })
    else:
        result.update(diagnostics)
    return result


def fit_direct_buckets(profile: TerminalProfile, seeds: Sequence[int], params: GenParams,
                       cfg: DirectExperimentConfig,
                       progress: Callable[[str], None] = print
                       ) -> tuple[DirectJobBucketConfig, list[dict[str, object]]]:
    """Fit all continuous edges using SLA_OFF FIFO training observations only."""
    queue_lengths: list[float] = []
    oldest_waits: list[float] = []
    own_waits: list[float] = []
    reaches: list[float] = []
    services: list[float] = []
    descriptors: list[dict[str, object]] = []
    fifo = DirectJobRulePolicy(DirectRule.FIFO)
    for index, seed in enumerate(seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        descriptors.append(_scenario_descriptor(scenario))
        env = DirectJobEnv(
            profile, sla_mode=SLAMode.OFF, expected_n_config=cfg.n_external,
            check_invariants=cfg.check_train_invariants,
        )
        state, info = env.reset(scenario)
        while state is not None:
            raw = info.raw_global
            queue_lengths.append(raw.queue_length)
            oldest_waits.append(raw.oldest_wait_s)
            for candidate in info.feasible_candidates:
                own_waits.append(candidate.wait_s)
                reaches.append(candidate.reach_s)
                services.append(candidate.estimated_service_s)
            candidate = fifo.act(state, info.candidates)
            state, _cost, _done, info = env.step(candidate)
        if index % max(1, min(100, len(seeds))) == 0 or index == len(seeds):
            progress(f"[bucket] FIFO train observations {index}/{len(seeds)}")
    buckets = DirectJobBucketConfig.fit(
        queue_lengths=queue_lengths,
        oldest_waits_s=oldest_waits,
        own_waits_s=own_waits,
        reaches_s=reaches,
        service_times_s=services,
        sla_s=profile.long_wait_sla_s,
    )
    return buckets, descriptors


def _evaluate_policy(policy: DirectJobRulePolicy | CostQAgent, arm: SLAMode,
                     profile: TerminalProfile, scenarios: Sequence[Scenario],
                     buckets: DirectJobBucketConfig, n_config: int,
                     policy_name: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        env = DirectJobEnv(
            profile, sla_mode=arm, bucket_cfg=buckets,
            expected_n_config=n_config, check_invariants=True,
        )
        if isinstance(policy, CostQAgent):
            metrics, _trace = _run_cost_q_episode(policy, env, scenario)
            name = policy_name or POLICY_COST_Q
        else:
            metrics, _trace = _run_baseline_episode(policy, env, scenario)
            name = policy_name or policy.name
        rows.append({
            "arm": arm.value,
            "policy": name,
            "seed": scenario.seed,
            "scenario_id": scenario.scenario_id,
            "metrics": metrics,
        })
    return rows


def _checkpoint_schedule(cfg: DirectExperimentConfig) -> set[int]:
    checkpoints = set(range(cfg.checkpoint_every, cfg.train_episodes + 1,
                            cfg.checkpoint_every))
    checkpoints.add(cfg.train_episodes)
    return checkpoints


def _train_arm(profile: TerminalProfile, arm: SLAMode,
               train_seeds: Sequence[int], validation_scenarios: Sequence[Scenario],
               params: GenParams, buckets: DirectJobBucketConfig,
               cfg: DirectExperimentConfig, progress: Callable[[str], None]
               ) -> tuple[CostQAgent, dict[str, object], list[dict[str, object]]]:
    checkpoints = _checkpoint_schedule(cfg)
    best: tuple[float, float, int, CostQAgent] | None = None
    validation_log: list[dict[str, object]] = []
    for p in cfg.learning_rate_powers:
        agent = CostQAgent(CostQConfig(learning_rate_power=p, gamma=1.0),
                           seed=cfg.train_seed0 + round(p * 1_000))
        for episode, seed in enumerate(train_seeds, start=1):
            scenario = _scenario(profile, seed, params, cfg.n_external)
            env = DirectJobEnv(
                profile, sla_mode=arm, bucket_cfg=buckets,
                expected_n_config=cfg.n_external,
                check_invariants=cfg.check_train_invariants,
            )
            metrics, _trace = _run_cost_q_episode(
                agent, env, scenario, train_episode=episode - 1,
            )
            if episode in checkpoints:
                evaluated = copy.deepcopy(agent)
                rows = _evaluate_policy(
                    evaluated, arm, profile, validation_scenarios, buckets,
                    cfg.n_external,
                )
                mean_wait = fmean(float(row["metrics"]["mean_wait_min"])
                                  for row in rows)
                fallback_rate = _aggregate_fallback(rows)
                log_row = {
                    "arm": arm.value,
                    "learning_rate_power": p,
                    "episode": episode,
                    "validation_mean_wait_min": mean_wait,
                    "validation_fallback_rate": fallback_rate,
                    "train_last_mean_wait_min": metrics["mean_wait_min"],
                    "table_keys": len(agent.table.q),
                }
                validation_log.append(log_row)
                progress(
                    f"[train:{arm.value}] p={p:.1f} episode={episode}/"
                    f"{cfg.train_episodes} val_wait={mean_wait:.3f} "
                    f"fallback={fallback_rate:.1%}"
                )
                candidate = (mean_wait, p, episode, copy.deepcopy(agent))
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
    if best is None:  # pragma: no cover - schedule is non-empty by construction
        raise AssertionError("no validation checkpoint was evaluated")
    mean_wait, selected_p, selected_episode, selected_agent = best
    selected_agent.reset_diagnostics()
    selection = {
        "p": selected_p,
        "checkpoint_episode": selected_episode,
        "validation_mean_wait_min": mean_wait,
        "tie_break": "mean_wait_min, then lower p, then earlier checkpoint",
    }
    return selected_agent, selection, validation_log


def _aggregate_fallback(rows: Sequence[dict[str, object]]) -> float:
    fallback = sum(int(row["metrics"]["fallback_count"]) for row in rows)
    decisions = sum(int(row["metrics"]["n_decisions"]) for row in rows)
    return fallback / decisions if decisions else 0.0


def _baseline_validation(profile: TerminalProfile, arm: SLAMode,
                         scenarios: Sequence[Scenario], buckets: DirectJobBucketConfig,
                         cfg: DirectExperimentConfig
                         ) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for policy in direct_baseline_policies():
        policy_rows = _evaluate_policy(
            policy, arm, profile, scenarios, buckets, cfg.n_external,
        )
        rows.extend(policy_rows)
        summaries.append({
            "policy": policy.name,
            "validation_mean_wait_min": fmean(
                float(row["metrics"]["mean_wait_min"]) for row in policy_rows
            ),
        })
    order = {policy.name: index for index, policy in enumerate(direct_baseline_policies())}
    selected = min(
        summaries,
        key=lambda row: (float(row["validation_mean_wait_min"]),
                         order[str(row["policy"])]),
    )
    return selected, rows


def _assert_alias_results(rows: Sequence[dict[str, object]]) -> None:
    metrics = ("mean_wait_min", "p50_wait_min", "p95_wait_min", "queue_area_h",
               "travel_km", "rehandles", "sla_over_rate", "backlog")
    aliases = (
        (DirectRule.FIFO.value, DirectRule.LONGEST_WAIT.value),
        (DirectRule.SHORTEST_ESTIMATED_SERVICE_TIME.value,
         DirectRule.IMMEDIATE_COST_GREEDY.value),
    )
    indexed = {(str(row["arm"]), str(row["policy"]), int(row["seed"])): row
               for row in rows}
    arms = {str(row["arm"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    for arm in arms:
        for left, right in aliases:
            for seed in seeds:
                left_row = indexed[(arm, left, seed)]
                right_row = indexed[(arm, right, seed)]
                for metric in metrics:
                    if left_row["metrics"][metric] != right_row["metrics"][metric]:
                        raise AssertionError(
                            f"alias mismatch {arm} seed={seed}: {left} != {right} ({metric})"
                        )


def _summarize(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | bool]]] = {}
    for row in rows:
        groups.setdefault((str(row["arm"]), str(row["policy"])), []).append(row["metrics"])
    keys = (
        "mean_wait_min", "p50_wait_min", "p90_wait_min", "p95_wait_min",
        "max_wait_min", "sla_over_rate", "queue_area_h", "travel_km",
        "rehandles", "completion_rate", "backlog", "fallback_rate",
        "decision_coverage", "signature_coverage",
    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for (arm, policy), metrics_rows in groups.items():
        summary = {key: fmean(float(row[key]) for row in metrics_rows) for key in keys}
        fallback = sum(int(row["fallback_count"]) for row in metrics_rows)
        decisions = sum(int(row["n_decisions"]) for row in metrics_rows)
        checked = sum(int(row["signatures_checked"]) for row in metrics_rows)
        visited = sum(int(row["visited_signatures"]) for row in metrics_rows)
        summary.update({
            "episodes": float(len(metrics_rows)),
            "fallback_count": float(fallback),
            "n_decisions": float(decisions),
            "fallback_rate": fallback / decisions if decisions else 0.0,
            "signature_coverage": visited / checked if checked else 1.0,
        })
        result.setdefault(arm, {})[policy] = summary
    return result


def _paired_statistics(test_rows: Sequence[dict[str, object]],
                       selection: dict[str, dict[str, object]],
                       cfg: DirectExperimentConfig) -> dict[str, object]:
    result: dict[str, object] = {}
    for arm in (SLAMode.OFF.value, SLAMode.ON.value):
        baseline = str(selection[arm]["baseline"]["policy"])
        by_policy = {
            policy: sorted(
                (row for row in test_rows
                 if row["arm"] == arm and row["policy"] == policy),
                key=lambda row: int(row["seed"]),
            )
            for policy in (baseline, POLICY_COST_Q)
        }
        seeds = [int(row["seed"]) for row in by_policy[baseline]]
        if seeds != [int(row["seed"]) for row in by_policy[POLICY_COST_Q]]:
            raise AssertionError("paired test seed ordering mismatch")
        arm_stats: dict[str, object] = {
            "baseline": baseline,
            "alternative": POLICY_COST_Q,
        }
        for label, metric_name in (("mean_wait", "mean_wait_min"),
                                   ("p95_wait", "p95_wait_min")):
            base_values = [float(row["metrics"][metric_name])
                           for row in by_policy[baseline]]
            cost_values = [float(row["metrics"][metric_name])
                           for row in by_policy[POLICY_COST_Q]]
            stats = paired_bootstrap(
                base_values,
                cost_values,
                metric=MetricSpec(metric_name, MetricDirection.MINIMIZE),
                seeds=seeds,
                seed=cfg.bootstrap_seed + (0 if arm == PRIMARY_ARM else 1)
                     + (0 if label == "mean_wait" else 10),
                n_resamples=cfg.bootstrap_resamples,
            )
            arm_stats[label] = stats.as_dict()
        result[arm] = arm_stats
    return result


def _coverage_class(fallback_rate: float) -> str:
    if fallback_rate == 0.0:
        return "PURE_COST_Q"
    if fallback_rate <= 0.05:
        return "HYBRID_ACCEPTABLE"
    return "COVERAGE_INSUFFICIENT"


def _acceptance(summary: dict[str, dict[str, dict[str, float]]],
                paired: dict[str, object], rows: Sequence[dict[str, object]],
                cfg: DirectExperimentConfig) -> dict[str, object]:
    primary = paired[PRIMARY_ARM]
    mean_ci = primary["mean_wait"]["difference_ci"]
    p95_ci = primary["p95_wait"]["percent_change_ci"]
    fallback_rate = summary[PRIMARY_ARM][POLICY_COST_Q]["fallback_rate"]
    coverage = _coverage_class(fallback_rate)
    mean_improved = float(mean_ci["upper"]) < 0.0
    p95_guardrail = p95_ci is not None and float(p95_ci["upper"]) <= 5.0
    completion_ok = all(float(row["metrics"]["completion_rate"]) == 1.0 for row in rows)
    backlog_ok = all(int(row["metrics"]["backlog"]) == 0 for row in rows)
    invariants_ok = all(bool(row["metrics"]["invariants_ok"]) for row in rows)
    coverage_ok = fallback_rate <= 0.05
    criteria = mean_improved and p95_guardrail and completion_ok and backlog_ok \
        and invariants_ok and coverage_ok
    eligible = cfg.strategy_compliant
    return {
        "quick": cfg.quick,
        "strategy_compliant_setting": cfg.strategy_compliant,
        "primary_arm": PRIMARY_ARM,
        "mean_improved": mean_improved,
        "p95_guardrail": p95_guardrail,
        "completion_ok": completion_ok,
        "backlog_ok": backlog_ok,
        "invariants_ok": invariants_ok,
        "fallback_rate": fallback_rate,
        "coverage_class": coverage,
        "coverage_ok": coverage_ok,
        "criteria_met": criteria,
        "overall": criteria if eligible else None,
        "decision": ("PASS" if criteria else "FAIL") if eligible else "NO_CLAIM_NONCOMPLIANT",
    }


def run_direct_job_experiment(profile_path: str = DEFAULT_DIRECT_PROFILE,
                              out_dir: str = "outputs/reports/exp1_direct_costq_hjnc",
                              cfg: DirectExperimentConfig | None = None,
                              progress: Callable[[str], None] = print) -> Path:
    """Run the complete frozen YR-027 protocol and write reproducible artifacts."""
    cfg = cfg or DirectExperimentConfig()
    started = time.time()
    source_git = _git_state()  # capture before this run creates output artifacts
    if not cfg.quick and (source_git["commit"] == "unknown" or source_git["dirty"]):
        raise RuntimeError(
            "full YR-027 run requires a clean committed source tree for provenance"
        )
    profile = load_profile(profile_path)
    params = direct_gen_params(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    progress(
        f"[YR-027] profile={profile.terminal_id} train={cfg.train_episodes} "
        f"val={cfg.validation_episodes} test={cfg.test_episodes}"
    )
    buckets, train_descriptors = fit_direct_buckets(
        profile, cfg.train_seeds, params, cfg, progress,
    )
    buckets.save(out / "direct_buckets.json")
    validation_scenarios = [
        _scenario(profile, seed, params, cfg.n_external) for seed in cfg.validation_seeds
    ]
    test_scenarios = [
        _scenario(profile, seed, params, cfg.n_external) for seed in cfg.test_seeds
    ]
    seed_manifest = {
        "train": train_descriptors,
        "validation": [_scenario_descriptor(item) for item in validation_scenarios],
        "test": [_scenario_descriptor(item) for item in test_scenarios],
        "disjoint": True,
    }
    _json_dump(out / "seed_manifest.json", seed_manifest)

    agents: dict[str, CostQAgent] = {}
    selection: dict[str, dict[str, object]] = {}
    validation_results: dict[str, object] = {}
    train_log: list[dict[str, object]] = []
    for arm in (SLAMode.OFF, SLAMode.ON):
        baseline_selection, baseline_rows = _baseline_validation(
            profile, arm, validation_scenarios, buckets, cfg,
        )
        agent, cost_selection, arm_log = _train_arm(
            profile, arm, cfg.train_seeds, validation_scenarios,
            params, buckets, cfg, progress,
        )
        agents[arm.value] = agent
        selection[arm.value] = {
            "cost_q": cost_selection,
            "baseline": baseline_selection,
        }
        validation_results[arm.value] = {
            "baseline_rows": baseline_rows,
            "cost_q_checkpoints": arm_log,
        }
        train_log.extend(arm_log)
        agent.save(out / f"agent_{arm.value}.json")
    _json_dump(out / "train_log.json", train_log)
    _json_dump(out / "validation_results.json", validation_results)
    _json_dump(out / "selection.json", selection)

    test_rows: list[dict[str, object]] = []
    for arm in (SLAMode.OFF, SLAMode.ON):
        for policy in direct_baseline_policies():
            progress(f"[test:{arm.value}] {policy.name}")
            test_rows.extend(_evaluate_policy(
                policy, arm, profile, test_scenarios, buckets, cfg.n_external,
            ))
        progress(f"[test:{arm.value}] {POLICY_COST_Q}")
        test_rows.extend(_evaluate_policy(
            copy.deepcopy(agents[arm.value]), arm, profile, test_scenarios,
            buckets, cfg.n_external, POLICY_COST_Q,
        ))
    _assert_alias_results(test_rows)
    summary = _summarize(test_rows)
    paired = _paired_statistics(test_rows, selection, cfg)
    acceptance = _acceptance(summary, paired, test_rows, cfg)

    manifest = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "mode": "quick" if cfg.quick else "full",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": {
            "path": str(profile_path),
            "terminal_id": profile.terminal_id,
            "profile_date": profile.profile_date,
            "assumed": profile.assumed,
            "sha256": _profile_digest(profile_path),
        },
        "git": source_git,
        "config": asdict(cfg),
        "strategy_compliant_setting": cfg.strategy_compliant,
        "clean_source_required": not cfg.quick,
        "generator_params": asdict(params),
        "N_config": cfg.n_external,
        "n_vessel": 0,
        "information_boundary": "BLOCK_ENTRY",
        "transfer_directions": ["TRUCK_TO_YARD", "YARD_TO_TRUCK"],
        "objective": "sum(queue_area_delta_s)/(60*N_config)=mean_wait_min",
        "gamma": 1.0,
        "epsilon": "1/sqrt(episode_index+1)",
        "bucket_fit": "SLA_OFF FIFO, train seeds only, quartiles + 1800s hard edge",
        "sla_threshold_s": profile.long_wait_sla_s,
        "primary_arm": PRIMARY_ARM,
        "alias_assertions": ["FIFO=LONGEST_WAIT",
                             "SHORTEST_ESTIMATED_SERVICE_TIME=IMMEDIATE_COST_GREEDY"],
        "elapsed_s": time.time() - started,
    }
    payload = {
        "manifest": manifest,
        "selection": selection,
        "test_rows": test_rows,
        "summary": summary,
        "paired_statistics": paired,
        "acceptance": acceptance,
    }
    _json_dump(out / "manifest.json", manifest)
    _json_dump(out / "test_results.json", test_rows)
    _json_dump(out / "paired_statistics.json", paired)
    _json_dump(out / "exp1_direct_results.json", payload)

    from .direct_job_report import build_direct_job_report
    report_path = build_direct_job_report(payload, out)
    progress(f"[YR-027] completed in {manifest['elapsed_s']:.1f}s -> {report_path}")
    return report_path
