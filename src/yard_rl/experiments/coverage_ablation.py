"""YR-028 — Direct-Job Cost-Q coverage ablation.

YR-027 v1 의 selected-checkpoint fallback 55.04% 가 (a) checkpoint 선택 규칙
(R1: validation mean_wait 최소 — coverage 무시) 탓인지, (b) v1 상태공간 크기
(≈102만 signature) 탓인지, (c) 학습예산(1,000 ep) 부족 탓인지를 분리한다.

사전등록 (spec: YR-028-cost-q-coverage.md — 실행 전 동결):
- 새 seed band: train 40000+ / validation 50000+ / test 60000+ (YR-027 band 와 분리,
  기존 test seed 재선택 금지 준수).
- 선택 규칙 2종: R1 = min val mean_wait (YR-027 규칙 재현) /
  R2 = val fallback ≤ 5% 인 checkpoint 중 min mean_wait (coverage-gate).
- 학습 1회를 공유해 horizon 1,000(프로토콜 재현)·3,000(증량 축) 에 규칙을 사후 적용
  — 선택은 validation 만 사용하므로 사전등록 하에 정당.
- 상태 축: v1_rich(복원) vs v2_minimal(기존 순수 Cost-Q 참조), SLA_OFF 단일 arm.
- 판정: 어떤 v1 checkpoint 도 gate 를 못 넘으면 STATE_SPACE, 1,000 내에서 넘는데
  R1 이 고-fallback checkpoint 를 골랐으면 CHECKPOINT_RULE, 1,000 초과에서만 넘으면
  BUDGET. 비목표: YR-027 FAIL 재해석 금지 — 성능 판정이 아니라 원인 분리.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Sequence

from ..domain.models import TerminalProfile
from ..domain.scenario import Scenario
from ..envs.direct_job_env import DirectJobBucketConfig, DirectJobEnv, SLAMode
from ..io.profile_loader import load_profile
from ..io.scenario_gen import GenParams
from ..policies.cost_q import CostQAgent, CostQConfig
from ..policies.direct_baselines import (DirectJobRulePolicy, DirectRule,
                                         direct_baseline_policies)
from .direct_job_runner import (_aggregate_fallback, _assert_alias_results,
                                _direct_metrics, _git_state, _json_dump,
                                _profile_digest, _run_baseline_episode,
                                _run_cost_q_episode, _scenario,
                                _scenario_descriptor)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

ABLATION_ID = "YR-028-coverage-ablation"
FALLBACK_GATE = 0.05
ARM = SLAMode.OFF
RULES = ("R1_min_wait", "R2_coverage_gate")


@dataclass(frozen=True)
class AblationConfig:
    train_episodes_v1: int = 3_000
    protocol_horizon: int = 1_000
    train_episodes_v2: int = 1_000
    validation_episodes: int = 30
    test_episodes: int = 100
    checkpoint_every: int = 50
    n_external: int = 100
    learning_rate_powers: tuple[float, ...] = (0.6, 0.8, 1.0)
    train_seed0: int = 40_000
    validation_seed0: int = 50_000
    test_seed0: int = 60_000
    bootstrap_seed: int = 72_028
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    quick: bool = False

    def __post_init__(self) -> None:
        if self.protocol_horizon > self.train_episodes_v1:
            raise ValueError("protocol_horizon must fit inside train_episodes_v1")
        if min(self.train_episodes_v1, self.train_episodes_v2, self.validation_episodes,
               self.test_episodes, self.checkpoint_every, self.n_external) <= 0:
            raise ValueError("all sizes must be positive")
        bands = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy = set(range(10_000, 10_000 + 1_000)) | set(range(20_000, 20_030)) \
            | set(range(30_000, 30_100))
        if any(band & legacy for band in bands):
            raise ValueError("YR-027 seed band 재사용 금지 (사전등록)")

    @property
    def train_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.train_seed0, self.train_seed0 + self.train_episodes_v1))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.validation_seed0,
                           self.validation_seed0 + self.validation_episodes))

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.test_seed0, self.test_seed0 + self.test_episodes))

    def horizons(self, schema: str) -> tuple[int, ...]:
        if schema == "v1_rich":
            return (self.protocol_horizon, self.train_episodes_v1)
        return (self.train_episodes_v2,)


def quick_ablation_config() -> AblationConfig:
    return AblationConfig(
        train_episodes_v1=16, protocol_horizon=8, train_episodes_v2=8,
        validation_episodes=3, test_episodes=4, checkpoint_every=4,
        n_external=12, learning_rate_powers=(0.8,), bootstrap_resamples=200,
        quick=True,
    )


def _gen_params(cfg: AblationConfig) -> GenParams:
    return GenParams(n_external=cfg.n_external, n_vessel=0,
                     drain_window_s=cfg.drain_window_s)


def _fit_buckets(profile: TerminalProfile, seeds: Sequence[int], params: GenParams,
                 cfg: AblationConfig, progress: Callable[[str], None]
                 ) -> DirectJobBucketConfig:
    """FIFO train 관측으로 전 bucket edge fit — v1/v2 arm 이 공유 (같은 궤적)."""
    queue, service, oldest, own, reach = [], [], [], [], []
    fifo = DirectJobRulePolicy(DirectRule.FIFO)
    for index, seed in enumerate(seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        env = DirectJobEnv(profile, sla_mode=ARM, expected_n_config=cfg.n_external,
                           state_schema="v1_rich")
        state, info = env.reset(scenario)
        while state is not None:
            raw = info.raw_global
            queue.append(raw.queue_length)
            oldest.append(raw.oldest_wait_s)
            for c in info.feasible_candidates:
                service.append(c.estimated_service_s)
                own.append(c.wait_s)
                reach.append(c.reach_s)
            state, _cost, _done, info = env.step(fifo.act(state, info.candidates))
        if index % max(1, min(200, len(seeds))) == 0 or index == len(seeds):
            progress(f"[bucket] FIFO train {index}/{len(seeds)}")
    return DirectJobBucketConfig.fit(
        queue_lengths=queue, service_times_s=service, oldest_waits_s=oldest,
        own_waits_s=own, reaches_s=reach, sla_s=profile.long_wait_sla_s)


def _evaluate(policy, schema: str, profile: TerminalProfile,
              scenarios: Sequence[Scenario], buckets: DirectJobBucketConfig,
              cfg: AblationConfig, policy_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        env = DirectJobEnv(profile, sla_mode=ARM, bucket_cfg=buckets,
                           expected_n_config=cfg.n_external, check_invariants=True,
                           state_schema=schema)
        if isinstance(policy, CostQAgent):
            metrics, _ = _run_cost_q_episode(policy, env, scenario)
        else:
            metrics, _ = _run_baseline_episode(policy, env, scenario)
        rows.append({"arm": ARM.value, "policy": policy_name, "seed": scenario.seed,
                     "scenario_id": scenario.scenario_id, "metrics": metrics})
    return rows


@dataclass
class _Best:
    mean: float
    p: float
    episode: int
    agent: CostQAgent

    @property
    def key(self) -> tuple[float, float, int]:
        return (self.mean, self.p, self.episode)


def _train_schema(schema: str, episodes: int, profile: TerminalProfile,
                  validation: Sequence[Scenario], params: GenParams,
                  buckets: DirectJobBucketConfig, cfg: AblationConfig,
                  progress: Callable[[str], None]
                  ) -> tuple[list[dict[str, object]], dict[str, dict[int, _Best]]]:
    """한 schema 를 학습하며 checkpoint 곡선 + (규칙×horizon) 최적 스냅샷을 유지."""
    checkpoints = set(range(cfg.checkpoint_every, episodes + 1, cfg.checkpoint_every))
    checkpoints.update({episodes, *(h for h in cfg.horizons(schema))})
    curve: list[dict[str, object]] = []
    best: dict[str, dict[int, _Best]] = {rule: {} for rule in RULES}
    for p in cfg.learning_rate_powers:
        agent = CostQAgent(CostQConfig(learning_rate_power=p, gamma=1.0),
                           seed=cfg.train_seed0 + round(p * 1_000))
        for episode, seed in enumerate(cfg.train_seeds[:episodes], start=1):
            scenario = _scenario(profile, seed, params, cfg.n_external)
            env = DirectJobEnv(profile, sla_mode=ARM, bucket_cfg=buckets,
                               expected_n_config=cfg.n_external, state_schema=schema)
            _run_cost_q_episode(agent, env, scenario, train_episode=episode - 1)
            if episode not in checkpoints:
                continue
            snapshot = copy.deepcopy(agent)
            rows = _evaluate(snapshot, schema, profile, validation, buckets, cfg,
                             "val")
            mean = fmean(float(r["metrics"]["mean_wait_min"]) for r in rows)
            fallback = _aggregate_fallback(rows)
            curve.append({"schema": schema, "p": p, "episode": episode,
                          "val_mean_wait_min": mean, "val_fallback_rate": fallback,
                          "table_keys": len(agent.table.q)})
            progress(f"[train:{schema}] p={p:.1f} ep={episode}/{episodes} "
                     f"val={mean:.3f} fb={fallback:.1%} keys={len(agent.table.q)}")
            cand = _Best(mean, p, episode, snapshot)
            for horizon in cfg.horizons(schema):
                if episode > horizon:
                    continue
                cur = best[RULES[0]].get(horizon)
                if cur is None or cand.key < cur.key:
                    best[RULES[0]][horizon] = cand
                if fallback <= FALLBACK_GATE:
                    cur = best[RULES[1]].get(horizon)
                    if cur is None or cand.key < cur.key:
                        best[RULES[1]][horizon] = cand
    return curve, best


def _verdict(curve: list[dict[str, object]], best: dict[str, dict[int, _Best]],
             cfg: AblationConfig) -> dict[str, object]:
    v1 = [row for row in curve if row["schema"] == "v1_rich"]
    within_protocol = [r for r in v1 if int(r["episode"]) <= cfg.protocol_horizon]
    gate_hits_protocol = [r for r in within_protocol
                          if float(r["val_fallback_rate"]) <= FALLBACK_GATE]
    gate_hits_any = [r for r in v1 if float(r["val_fallback_rate"]) <= FALLBACK_GATE]
    r1 = best[RULES[0]].get(cfg.protocol_horizon)
    r1_fallback = None
    if r1 is not None:
        r1_fallback = next(
            (float(r["val_fallback_rate"]) for r in within_protocol
             if r["p"] == r1.p and int(r["episode"]) == r1.episode), None)
    if not gate_hits_any:
        primary = "STATE_SPACE"
    elif gate_hits_protocol:
        primary = ("CHECKPOINT_RULE"
                   if r1_fallback is not None and r1_fallback > FALLBACK_GATE
                   else "NONE_REPRODUCED")
    else:
        primary = "BUDGET"
    return {
        "primary_cause": primary,
        "protocol_gate_hits": len(gate_hits_protocol),
        "any_gate_hits": len(gate_hits_any),
        "min_v1_fallback": min((float(r["val_fallback_rate"]) for r in v1),
                               default=None),
        "r1_protocol_pick": None if r1 is None else
            {"p": r1.p, "episode": r1.episode, "val_fallback_rate": r1_fallback},
    }


def run_coverage_ablation(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                          out_dir: str = "outputs/reports/costq_coverage_ablation_hjnc",
                          cfg: AblationConfig | None = None,
                          progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or AblationConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-028 run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-028] profile={profile.terminal_id} v1={cfg.train_episodes_v1}ep "
             f"v2={cfg.train_episodes_v2}ep val={cfg.validation_episodes} "
             f"test={cfg.test_episodes}")

    buckets = _fit_buckets(profile, cfg.train_seeds[:cfg.train_episodes_v2],
                           params, cfg, progress)
    buckets.save(out / "direct_buckets.json")
    validation = [_scenario(profile, s, params, cfg.n_external)
                  for s in cfg.validation_seeds]
    tests = [_scenario(profile, s, params, cfg.n_external) for s in cfg.test_seeds]
    _json_dump(out / "seed_manifest.json", {
        "validation": [_scenario_descriptor(s) for s in validation],
        "test": [_scenario_descriptor(s) for s in tests],
        "train_seed0": cfg.train_seed0, "bands_disjoint_from_yr027": True,
    })

    curve: list[dict[str, object]] = []
    selections: dict[str, dict[str, object]] = {}
    selected_agents: dict[str, tuple[str, CostQAgent]] = {}
    for schema, episodes in (("v1_rich", cfg.train_episodes_v1),
                             ("v2_minimal", cfg.train_episodes_v2)):
        schema_curve, best = _train_schema(schema, episodes, profile, validation,
                                           params, buckets, cfg, progress)
        curve.extend(schema_curve)
        for rule, by_horizon in best.items():
            for horizon, pick in by_horizon.items():
                name = f"CostQ[{schema}|{rule}@{horizon}]"
                selections[name] = {
                    "schema": schema, "rule": rule, "horizon": horizon,
                    "p": pick.p, "episode": pick.episode,
                    "val_mean_wait_min": pick.mean,
                }
                dedup = (schema, pick.p, pick.episode)
                if all(key != dedup for key, _ in selected_agents.values()):
                    selected_agents[name] = (dedup, pick.agent)
                else:
                    twin = next(n for n, (key, _) in selected_agents.items()
                                if key == dedup)
                    selections[name]["same_agent_as"] = twin
        if schema == "v1_rich":
            v1_verdict_input = {rule: dict(by) for rule, by in best.items()}
    verdict = _verdict(curve, v1_verdict_input, cfg)
    _json_dump(out / "checkpoint_curve.json", curve)

    # baseline 선택 (validation 최저 mean, 프로토콜 재현) + locked test
    baseline_rows_val: list[dict[str, object]] = []
    for policy in direct_baseline_policies():
        baseline_rows_val.extend(_evaluate(policy, "v2_minimal", profile, validation,
                                           buckets, cfg, policy.name))
    by_policy = {}
    for row in baseline_rows_val:
        by_policy.setdefault(row["policy"], []).append(
            float(row["metrics"]["mean_wait_min"]))
    baseline_name = min(by_policy, key=lambda k: (fmean(by_policy[k]), k))
    selections["_baseline"] = {"policy": baseline_name,
                               "validation_mean_wait_min": fmean(by_policy[baseline_name])}
    _json_dump(out / "selections.json", selections)

    test_rows: list[dict[str, object]] = []
    for policy in direct_baseline_policies():
        progress(f"[test] {policy.name}")
        test_rows.extend(_evaluate(policy, "v2_minimal", profile, tests, buckets,
                                   cfg, policy.name))
    _assert_alias_results(test_rows)
    for name, (dedup, agent) in selected_agents.items():
        progress(f"[test] {name}")
        schema = selections[name]["schema"]
        test_rows.extend(_evaluate(copy.deepcopy(agent), schema, profile, tests,
                                   buckets, cfg, name))
        agent.save(out / f"agent_{name.replace('|', '_').replace('@', '_at_')}.json")
    _json_dump(out / "test_results.json", test_rows)

    base_rows = sorted((r for r in test_rows if r["policy"] == baseline_name),
                       key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in base_rows]
    paired: dict[str, object] = {}
    for offset, name in enumerate(selected_agents):
        alt_rows = sorted((r for r in test_rows if r["policy"] == name),
                          key=lambda r: int(r["seed"]))
        entry: dict[str, object] = {
            "test_fallback_rate": _aggregate_fallback(alt_rows),
        }
        for m_off, (label, metric) in enumerate((("mean_wait", "mean_wait_min"),
                                                 ("p95_wait", "p95_wait_min"))):
            stats = paired_bootstrap(
                [float(r["metrics"][metric]) for r in base_rows],
                [float(r["metrics"][metric]) for r in alt_rows],
                metric=MetricSpec(metric, MetricDirection.MINIMIZE),
                seeds=seeds, seed=cfg.bootstrap_seed + offset * 10 + m_off,
                n_resamples=cfg.bootstrap_resamples)
            entry[label] = stats.as_dict()
        paired[name] = entry
    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": ABLATION_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"path": str(profile_path),
                        "terminal_id": profile.terminal_id,
                        "assumed": profile.assumed,
                        "sha256": _profile_digest(profile_path)},
            "git": git, "config": asdict(cfg), "arm": ARM.value,
            "fallback_gate": FALLBACK_GATE, "rules": RULES,
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "verdict": verdict, "paired": paired,
    }
    _json_dump(out / "ablation_results.json", payload)
    report = _build_report(payload, curve, out)
    progress(f"[YR-028] completed in {payload['manifest']['elapsed_s']:.1f}s -> {report}")
    return report


def _build_report(payload: dict, curve: list[dict[str, object]], out: Path) -> Path:
    man, verdict = payload["manifest"], payload["verdict"]
    sel, paired = payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-028 — Direct-Job Cost-Q coverage ablation")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. 목적은 YR-027 v1 fallback 55% 의 **원인 분리**")
    L.append("> (checkpoint 규칙 vs 상태공간 vs 예산) — 성능 재판정·FAIL 재해석이 아님 (비목표).")
    L.append("")
    L.append("## 판정")
    L.append("")
    L.append(f"- **primary_cause: `{verdict['primary_cause']}`**")
    L.append(f"- v1 validation fallback 최소값: {verdict['min_v1_fallback']:.1%}"
             if verdict["min_v1_fallback"] is not None else "- v1 곡선 없음")
    L.append(f"- gate(≤{FALLBACK_GATE:.0%}) 통과 checkpoint: 프로토콜(≤{man['config']['protocol_horizon']}ep) "
             f"{verdict['protocol_gate_hits']}개 / 전체(≤{man['config']['train_episodes_v1']}ep) "
             f"{verdict['any_gate_hits']}개")
    if verdict["r1_protocol_pick"]:
        r1 = verdict["r1_protocol_pick"]
        L.append(f"- R1(프로토콜 재현) 선택: p={r1['p']}, episode={r1['episode']}, "
                 f"val fallback={r1['val_fallback_rate']:.1%}"
                 if r1["val_fallback_rate"] is not None else
                 f"- R1 선택: p={r1['p']}, episode={r1['episode']}")
    L.append("")
    L.append("## 선택 결과 × locked test (paired vs "
             f"{sel['_baseline']['policy']})")
    L.append("")
    L.append("| variant | p | episode | val_mean | test fallback | mean_wait Δ [95% CI] | p95 Δ% CI 상한 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, entry in paired.items():
        s = sel[name]
        mw_point = entry["mean_wait"]["difference"]
        mw_ci = entry["mean_wait"]["difference_ci"]
        p95 = entry["p95_wait"].get("percent_change_ci")
        p95_txt = "—" if p95 is None else f"{p95['upper']:+.1f}%"
        L.append(f"| {name} | {s['p']} | {s['episode']} | {s['val_mean_wait_min']:.2f} "
                 f"| {entry['test_fallback_rate']:.1%} "
                 f"| {mw_point:+.3f} [{mw_ci['lower']:+.3f}, {mw_ci['upper']:+.3f}] "
                 f"| {p95_txt} |")
    L.append("")
    L.append("## checkpoint 곡선 (validation, p·schema 별)")
    L.append("")
    L.append("| schema | p | episode | val_mean | val_fallback | table_keys |")
    L.append("|---|---|---|---|---|---|")
    step = max(1, len(curve) // 60)
    for row in curve[::step]:
        L.append(f"| {row['schema']} | {row['p']} | {row['episode']} "
                 f"| {row['val_mean_wait_min']:.2f} | {row['val_fallback_rate']:.1%} "
                 f"| {row['table_keys']} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.coverage_ablation — 원자료 "
             "ablation_results.json·checkpoint_curve.json*")
    path = out / "coverage_ablation_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
