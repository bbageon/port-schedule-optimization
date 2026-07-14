"""YR-030-c — Greedy 기반 잔차 Cost-Q 3-arm 비교 (사전등록 동결 실행기).

사용자 최종 전략 (2026-07-14): Q_total = G(정확 greedy 비용) + ΔQ.
- arm 1: IMMEDIATE_COST_GREEDY (baseline — 기존 휴리스틱 6종 중 validation 선택)
- arm 2: ResidualCostQ[state_job] — ΔQ 키 = 기존 coarse (YardState, JobState)
- arm 3: ResidualCostQ[future]    — ΔQ 키 = future_situation 단독 (§3 원문)

동결: γ=0.95 · p=1.0 · 3,000 ep/arm · ckpt 50 · seed band 110k/120k/130k.
사전등록: .claude/docs/strategy-history/2026-07-14-YR-030-c-residual-costq-prereg.md
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Sequence

from ..envs.direct_job_env import DirectJobBucketConfig, DirectJobEnv, SLAMode
from ..io.profile_loader import load_profile
from ..policies.cost_q import CostQConfig
from ..policies.direct_baselines import (DirectJobRulePolicy, DirectRule,
                                         direct_baseline_policies)
from ..policies.residual_cost_q import KEY_MODES, ResidualCostQAgent
from .coverage_ablation import _evaluate, _gen_params
from .direct_job_runner import (_aggregate_fallback, _assert_alias_results,
                                _git_state, _json_dump, _profile_digest,
                                _run_cost_q_episode, _scenario,
                                _scenario_descriptor)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-030-c-residual-costq"
ARM = SLAMode.OFF
SCHEMA = "v1_final"  # env 전역 상태·일관성 규칙 — 두 RL arm 공통 (사전등록 §3)


@dataclass(frozen=True)
class ResidualConfig:
    train_episodes: int = 3_000
    validation_episodes: int = 30
    test_episodes: int = 100
    checkpoint_every: int = 50
    n_external: int = 100
    learning_rate_power: float = 1.0     # YR-028 R2 승계
    gamma: float = 0.95                  # YR-030-b γ 축 소진 — 사용자 지정값 고정
    key_modes: tuple[str, ...] = KEY_MODES
    train_seed0: int = 110_000
    validation_seed0: int = 120_000
    test_seed0: int = 130_000
    bucket_fit_episodes: int = 1_000
    bootstrap_seed: int = 73_030
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.checkpoint_every, self.n_external,
               self.bucket_fit_episodes) <= 0:
            raise ValueError("all sizes must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if any(m not in KEY_MODES for m in self.key_modes) or not self.key_modes:
            raise ValueError(f"key_modes must be drawn from {KEY_MODES}")
        bands = [set(self.train_seeds), set(self.validation_seeds),
                 set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy = (set(range(10_000, 11_000)) | set(range(20_000, 20_030))
                  | set(range(30_000, 30_100)) | set(range(40_000, 43_000))
                  | set(range(50_000, 50_030)) | set(range(60_000, 60_100))
                  | set(range(70_000, 73_000)) | set(range(80_000, 80_030))
                  | set(range(90_000, 90_100)))
        if any(band & legacy for band in bands):
            raise ValueError("기존 실험(YR-027/028/030-b) seed band 재사용 금지 (사전등록)")

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


def quick_residual_config() -> ResidualConfig:
    return ResidualConfig(
        train_episodes=12, validation_episodes=3, test_episodes=4,
        checkpoint_every=4, n_external=12, bucket_fit_episodes=6,
        bootstrap_resamples=200, quick=True)


@dataclass(frozen=True)
class _CfgShim:
    """coverage_ablation._evaluate/_gen_params 가 기대하는 필드 어댑터."""
    _cfg: ResidualConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _fit_buckets_ctx(profile, seeds: Sequence[int], params, cfg: ResidualConfig,
                     progress: Callable[[str], None]) -> DirectJobBucketConfig:
    """FIFO train 관측으로 기존 5종 + future_situation edge 3종 fit (동결).

    잔여 총량 표본 = 결정별 '전체 feasible 서비스 합 − 후보 자신' (후보 단위,
    사전등록 §2). val/test 재조정 금지 — 본 함수는 train seed 만 받는다.
    """
    waiting_counts, service, longest, own, reach = [], [], [], [], []
    jobs_left, work_left = [], []
    fifo = DirectJobRulePolicy(DirectRule.FIFO)
    for index, seed in enumerate(seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        env = DirectJobEnv(profile, sla_mode=ARM, expected_n_config=cfg.n_external,
                           state_schema="v1_rich")
        state, info = env.reset(scenario)
        while state is not None:
            raw = info.raw_global
            waiting_counts.append(raw.waiting_truck_count)
            longest.append(raw.longest_wait_s)
            jobs_left.append(max(0, raw.waiting_truck_count - 1))
            total = sum(c.estimated_service_s for c in info.feasible_candidates)
            for c in info.feasible_candidates:
                service.append(c.estimated_service_s)
                own.append(c.wait_s)
                reach.append(c.reach_s)
                work_left.append(total - c.estimated_service_s)
            state, _cost, _done, info = env.step(fifo.act(state, info.candidates))
        if index % max(1, min(200, len(seeds))) == 0 or index == len(seeds):
            progress(f"[bucket] FIFO train {index}/{len(seeds)}")
    return DirectJobBucketConfig.fit(
        queue_lengths=waiting_counts, service_times_s=service,
        oldest_waits_s=longest, own_waits_s=own, reaches_s=reach,
        jobs_left_counts=jobs_left, work_left_totals_s=work_left,
        sla_s=profile.long_wait_sla_s)


def _arm_name(key_mode: str) -> str:
    return f"ResidualCostQ[{key_mode}]"


def _train_arm(key_mode: str, profile, validation, params, buckets,
               cfg: ResidualConfig, progress: Callable[[str], None]
               ) -> tuple[list[dict[str, object]], dict[str, object],
                          ResidualCostQAgent]:
    """arm 하나 학습 — checkpoint 곡선 + validation 최저 mean checkpoint 선택."""
    checkpoints = set(range(cfg.checkpoint_every, cfg.train_episodes + 1,
                            cfg.checkpoint_every))
    checkpoints.add(cfg.train_episodes)
    agent = ResidualCostQAgent(
        CostQConfig(learning_rate_power=cfg.learning_rate_power, gamma=cfg.gamma),
        seed=cfg.train_seed0 + KEY_MODES.index(key_mode), key_mode=key_mode)
    curve: list[dict[str, object]] = []
    best: tuple[float, int, ResidualCostQAgent] | None = None
    for episode, seed in enumerate(cfg.train_seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        env = DirectJobEnv(profile, sla_mode=ARM, bucket_cfg=buckets,
                           expected_n_config=cfg.n_external, state_schema=SCHEMA)
        _run_cost_q_episode(agent, env, scenario, train_episode=episode - 1)
        if episode not in checkpoints:
            continue
        snapshot = copy.deepcopy(agent)
        rows = _evaluate(snapshot, SCHEMA, profile, validation, buckets,
                         _CfgShim(cfg), "val")
        mean = fmean(float(r["metrics"]["mean_wait_min"]) for r in rows)
        coverage = fmean(float(r["metrics"]["signature_coverage"]) for r in rows)
        curve.append({"key_mode": key_mode, "episode": episode,
                      "val_mean_wait_min": mean, "val_signature_coverage": coverage,
                      "table_keys": len(agent.table.q)})
        progress(f"[train:{key_mode}] ep={episode}/{cfg.train_episodes} "
                 f"val={mean:.3f} cov={coverage:.1%} keys={len(agent.table.q)}")
        if best is None or (mean, episode) < (best[0], best[1]):
            best = (mean, episode, snapshot)
    mean, episode, selected = best
    selected.reset_diagnostics()
    selection = {"key_mode": key_mode, "gamma": cfg.gamma,
                 "p": cfg.learning_rate_power, "episode": episode,
                 "val_mean_wait_min": mean}
    return curve, selection, selected


def run_residual_experiment(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                            out_dir: str = "outputs/reports/costq_residual_hjnc",
                            cfg: ResidualConfig | None = None,
                            progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or ResidualConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-030-c run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_CfgShim(cfg))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-030-c] profile={profile.terminal_id} schema={SCHEMA} "
             f"arms={list(cfg.key_modes)} gamma={cfg.gamma} "
             f"train={cfg.train_episodes}")

    buckets = _fit_buckets_ctx(
        profile, cfg.train_seeds[:min(cfg.bucket_fit_episodes, cfg.train_episodes)],
        params, cfg, progress)
    buckets.save(out / "direct_buckets.json")
    validation = [_scenario(profile, s, params, cfg.n_external)
                  for s in cfg.validation_seeds]
    tests = [_scenario(profile, s, params, cfg.n_external) for s in cfg.test_seeds]
    _json_dump(out / "seed_manifest.json", {
        "validation": [_scenario_descriptor(s) for s in validation],
        "test": [_scenario_descriptor(s) for s in tests],
        "train_seed0": cfg.train_seed0,
        "bands_disjoint_from_prior_experiments": True,
    })

    curve: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    agents: dict[str, ResidualCostQAgent] = {}
    for key_mode in cfg.key_modes:
        arm_curve, selection, agent = _train_arm(
            key_mode, profile, validation, params, buckets, cfg, progress)
        curve.extend(arm_curve)
        selections[_arm_name(key_mode)] = selection
        agents[_arm_name(key_mode)] = agent
    _json_dump(out / "checkpoint_curve.json", curve)

    baseline_rows_val: list[dict[str, object]] = []
    for policy in direct_baseline_policies():
        baseline_rows_val.extend(_evaluate(policy, SCHEMA, profile, validation,
                                           buckets, _CfgShim(cfg), policy.name))
    by_policy: dict[str, list[float]] = {}
    for row in baseline_rows_val:
        by_policy.setdefault(str(row["policy"]), []).append(
            float(row["metrics"]["mean_wait_min"]))
    baseline_name = min(by_policy, key=lambda k: (fmean(by_policy[k]), k))
    selections["_baseline"] = {
        "policy": baseline_name,
        "validation_mean_wait_min": fmean(by_policy[baseline_name])}
    _json_dump(out / "selections.json", selections)

    test_rows: list[dict[str, object]] = []
    for policy in direct_baseline_policies():
        progress(f"[test] {policy.name}")
        test_rows.extend(_evaluate(policy, SCHEMA, profile, tests, buckets,
                                   _CfgShim(cfg), policy.name))
    _assert_alias_results(test_rows)
    for name, agent in agents.items():
        progress(f"[test] {name}")
        test_rows.extend(_evaluate(copy.deepcopy(agent), SCHEMA, profile, tests,
                                   buckets, _CfgShim(cfg), name))
        agent.save(out / f"agent_{name.replace('|', '_')}.json")
    _json_dump(out / "test_results.json", test_rows)

    base_rows = sorted((r for r in test_rows if r["policy"] == baseline_name),
                       key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in base_rows]
    paired: dict[str, object] = {}
    for offset, name in enumerate(agents):
        alt_rows = sorted((r for r in test_rows if r["policy"] == name),
                          key=lambda r: int(r["seed"]))
        entry: dict[str, object] = {
            "test_fallback_rate": _aggregate_fallback(alt_rows),
            "test_signature_coverage": fmean(
                float(r["metrics"]["signature_coverage"]) for r in alt_rows),
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
        # 사전등록 §4 guardrail 동시 보고: P95 CI 상한 ≤ +5% · completion 100%
        # · backlog 0 · invariant 0 (episode_success + 비용 항등식)
        p95_ci = entry["p95_wait"].get("percent_change_ci")
        entry["guardrails"] = {
            "p95_pct_change_ci_upper": None if p95_ci is None else float(p95_ci["upper"]),
            "p95_within_5pct": None if p95_ci is None else float(p95_ci["upper"]) <= 5.0,
            "completion_all_100pct": all(
                float(r["metrics"]["completion_rate"]) >= 1.0 for r in alt_rows),
            "max_backlog": max(int(r["metrics"]["backlog"]) for r in alt_rows),
            "invariants_all_ok": all(
                bool(r["metrics"]["invariants_ok"]) for r in alt_rows),
        }
        paired[name] = entry

    improved = [name for name in agents
                if float(paired[name]["mean_wait"]["difference_ci"]["upper"]) < 0.0]
    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": EXPERIMENT_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"path": str(profile_path),
                        "terminal_id": profile.terminal_id,
                        "assumed": profile.assumed,
                        "sha256": _profile_digest(profile_path)},
            "git": git, "config": asdict(cfg), "arm": ARM.value,
            "state_schema": SCHEMA,
            "policy": "Q_total = exact greedy G + residual dQ (user strategy 2026-07-14)",
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "paired": paired,
        "arms_improved_vs_baseline": improved,
    }
    _json_dump(out / "residual_results.json", payload)
    report = _build_report(payload, curve, out)
    progress(f"[YR-030-c] completed in {payload['manifest']['elapsed_s']:.1f}s"
             f" -> {report}")
    return report


def _build_report(payload: dict, curve: list[dict[str, object]], out: Path) -> Path:
    sel, paired = payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-030-c — Greedy 기반 잔차 Cost-Q (3-arm)")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. Q_total = 정확한 greedy 비용 G + ΔQ — "
             "잔차분해(state_job) vs 미래맥락(future) 키의 순서품질 효과 검증.")
    L.append("")
    L.append(f"- baseline (validation 선택): **{sel['_baseline']['policy']}**")
    L.append(f"- baseline 을 유의하게 이긴 arm: "
             f"{payload['arms_improved_vs_baseline'] or '**없음**'}")
    L.append("")
    L.append("## locked test — paired vs " + str(sel["_baseline"]["policy"]))
    L.append("")
    L.append("| arm | 선택 ep | val_mean | coverage | mean_wait Δ [95% CI] "
             "| p95 Δ% CI 상한 | guardrail (P95≤+5%/완료100%/backlog0/invariant) |")
    L.append("|---|---|---|---|---|---|---|")
    for name, entry in paired.items():
        s = sel.get(name, {})
        mw = entry["mean_wait"]["difference"]
        ci = entry["mean_wait"]["difference_ci"]
        p95 = entry["p95_wait"].get("percent_change_ci")
        p95_txt = "—" if p95 is None else f"{p95['upper']:+.1f}%"
        g = entry["guardrails"]
        mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
        g_txt = (f"{mark(bool(g['p95_within_5pct']))}/"
                 f"{mark(g['completion_all_100pct'])}/"
                 f"{mark(g['max_backlog'] == 0)}/"
                 f"{mark(g['invariants_all_ok'])}")
        L.append(f"| {name} | {s.get('episode', '—')} "
                 f"| {s.get('val_mean_wait_min', float('nan')):.2f} "
                 f"| {entry['test_signature_coverage']:.1%} "
                 f"| {mw:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}] "
                 f"| {p95_txt} | {g_txt} |")
    L.append("")
    L.append("## checkpoint 곡선 (validation)")
    L.append("")
    L.append("| arm | episode | val_mean | signature coverage | table_keys |")
    L.append("|---|---|---|---|---|")
    step = max(1, len(curve) // 48)
    for row in curve[::step]:
        L.append(f"| {row['key_mode']} | {row['episode']} "
                 f"| {row['val_mean_wait_min']:.2f} "
                 f"| {row['val_signature_coverage']:.1%} | {row['table_keys']} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.residual_costq — 원자료 residual_results.json*")
    path = out / "residual_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
