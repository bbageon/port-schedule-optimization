"""YR-030-b — 사용자 정의 v1 최종안 상태 + greedy-prior Q0 + γ grid.

사용자 지시 (2026-07-14): "다시 내가 state 정의했는데 다시 테스트해볼래?" +
"처음에는 greedy 하게 해서 보상율 0.95 로 해서 하고 주변값들도 적용해서 비교해보자"

- 상태: `v1_final` — YardState(운영단계/크레인위치/대기규모/최장대기/30분초과 수) ×
  JobState(반입반출/트럭대기 4단계/크레인이동 3단계/총작업시간/선행이동 수) +
  상태 일관성 규칙 4건 (env 불변조건).
- 초기화: 미방문 Q0 = greedy 즉시비용 ĉ(j) ("처음에는 greedy") — fallback 불필요,
  학습은 prior 를 덮어쓰며 개선분만 축적 (α₁=1 이므로 첫 실방문이 prior 대체).
- γ grid: {0.90, 0.95, 0.99, 1.0} — 사용자 지정 0.95 + 주변값. γ<1 은 노이즈 큰
  장기 bootstrap 전파 축소 가설. p=1.0 고정 (YR-028 R2 선택값, grid 폭 통제).
- 비교: 강 휴리스틱 6종 (paired) + YR-028 의 v1_rich(Q0=0, γ=1) R2@3000 agent 를
  같은 test band 에서 재평가한 reference.
- seed band: train 70000+ / val 80000+ / test 90000+ (기존 두 실험 band 와 분리).
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
from ..policies.cost_q import CostQAgent, CostQConfig
from ..policies.direct_baselines import direct_baseline_policies
from .coverage_ablation import _evaluate, _fit_buckets, _gen_params
from .direct_job_runner import (_aggregate_fallback, _assert_alias_results,
                                _git_state, _json_dump, _profile_digest,
                                _run_cost_q_episode, _scenario,
                                _scenario_descriptor)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-030-b-v1final-greedy-prior-gamma"
ARM = SLAMode.OFF
SCHEMA = "v1_final"
REFERENCE_AGENT = ("outputs/reports/costq_coverage_ablation_hjnc/"
                   "agent_CostQ[v1_rich_R2_coverage_gate_at_3000].json")
REFERENCE_BUCKETS = "outputs/reports/costq_coverage_ablation_hjnc/direct_buckets.json"


@dataclass(frozen=True)
class V1FinalConfig:
    train_episodes: int = 3_000
    validation_episodes: int = 30
    test_episodes: int = 100
    checkpoint_every: int = 50
    n_external: int = 100
    learning_rate_power: float = 1.0
    gammas: tuple[float, ...] = (0.90, 0.95, 0.99, 1.0)
    train_seed0: int = 70_000
    validation_seed0: int = 80_000
    test_seed0: int = 90_000
    bootstrap_seed: int = 72_030
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    include_reference: bool = True
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.checkpoint_every, self.n_external) <= 0:
            raise ValueError("all sizes must be positive")
        if not self.gammas or any(not 0.0 < g <= 1.0 for g in self.gammas):
            raise ValueError("gammas must be in (0, 1]")
        bands = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy = (set(range(10_000, 11_000)) | set(range(20_000, 20_030))
                  | set(range(30_000, 30_100)) | set(range(40_000, 43_000))
                  | set(range(50_000, 50_030)) | set(range(60_000, 60_100)))
        if any(band & legacy for band in bands):
            raise ValueError("기존 실험(YR-027/028) seed band 재사용 금지 (사전등록)")

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


def quick_v1final_config() -> V1FinalConfig:
    return V1FinalConfig(
        train_episodes=12, validation_episodes=3, test_episodes=4,
        checkpoint_every=4, n_external=12, gammas=(0.95, 1.0),
        bootstrap_resamples=200, include_reference=False, quick=True,
    )


def _gamma_name(gamma: float) -> str:
    return f"CostQ[v1_final|prior|g{gamma:g}]"


def _train_gamma(gamma: float, profile, validation, params, buckets,
                 cfg: V1FinalConfig, progress: Callable[[str], None]
                 ) -> tuple[list[dict[str, object]], dict[str, object], CostQAgent]:
    """γ 하나를 학습 — checkpoint 곡선 + validation 최저 mean checkpoint 선택."""
    checkpoints = set(range(cfg.checkpoint_every, cfg.train_episodes + 1,
                            cfg.checkpoint_every))
    checkpoints.add(cfg.train_episodes)
    agent = CostQAgent(
        CostQConfig(learning_rate_power=cfg.learning_rate_power, gamma=gamma,
                    use_greedy_prior=True),
        seed=cfg.train_seed0 + round(gamma * 1_000))
    curve: list[dict[str, object]] = []
    best: tuple[float, int, CostQAgent] | None = None
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
        curve.append({"gamma": gamma, "episode": episode,
                      "val_mean_wait_min": mean, "val_signature_coverage": coverage,
                      "table_keys": len(agent.table.q)})
        progress(f"[train:g{gamma:g}] ep={episode}/{cfg.train_episodes} "
                 f"val={mean:.3f} cov={coverage:.1%} keys={len(agent.table.q)}")
        if best is None or (mean, episode) < (best[0], best[1]):
            best = (mean, episode, snapshot)
    mean, episode, selected = best
    selected.reset_diagnostics()
    selection = {"gamma": gamma, "p": cfg.learning_rate_power, "episode": episode,
                 "val_mean_wait_min": mean}
    return curve, selection, selected


@dataclass(frozen=True)
class _CfgShim:
    """coverage_ablation._evaluate 가 기대하는 최소 인터페이스."""
    _cfg: V1FinalConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external


def run_v1_final_experiment(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                            out_dir: str = "outputs/reports/costq_v1final_hjnc",
                            cfg: V1FinalConfig | None = None,
                            progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or V1FinalConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-030-b run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_AblationShim(cfg))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-030-b] profile={profile.terminal_id} schema={SCHEMA} "
             f"gammas={list(cfg.gammas)} train={cfg.train_episodes}")

    buckets = _fit_buckets(profile, cfg.train_seeds[:min(1_000, cfg.train_episodes)],
                           params, _AblationShim(cfg), progress)
    buckets.save(out / "direct_buckets.json")
    validation = [_scenario(profile, s, params, cfg.n_external)
                  for s in cfg.validation_seeds]
    tests = [_scenario(profile, s, params, cfg.n_external) for s in cfg.test_seeds]
    _json_dump(out / "seed_manifest.json", {
        "validation": [_scenario_descriptor(s) for s in validation],
        "test": [_scenario_descriptor(s) for s in tests],
        "train_seed0": cfg.train_seed0, "bands_disjoint_from_prior_experiments": True,
    })

    curve: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    agents: dict[str, CostQAgent] = {}
    for gamma in cfg.gammas:
        gamma_curve, selection, agent = _train_gamma(
            gamma, profile, validation, params, buckets, cfg, progress)
        curve.extend(gamma_curve)
        name = _gamma_name(gamma)
        selections[name] = selection
        agents[name] = agent
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
    selections["_baseline"] = {"policy": baseline_name,
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
    reference_name = None
    if cfg.include_reference and Path(REFERENCE_AGENT).exists():
        reference_name = "CostQ[v1_rich|Q0=0|g1](YR-028 ref)"
        ref_agent = CostQAgent.load(REFERENCE_AGENT)
        ref_buckets = DirectJobBucketConfig.load(REFERENCE_BUCKETS)
        progress(f"[test] {reference_name}")
        test_rows.extend(_evaluate(ref_agent, "v1_rich", profile, tests, ref_buckets,
                                   _CfgShim(cfg), reference_name))
    _json_dump(out / "test_results.json", test_rows)

    base_rows = sorted((r for r in test_rows if r["policy"] == baseline_name),
                       key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in base_rows]
    paired: dict[str, object] = {}
    compare_names = list(agents) + ([reference_name] if reference_name else [])
    for offset, name in enumerate(compare_names):
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
        paired[name] = entry

    improved = [name for name in agents
                if float(paired[name]["mean_wait"]["difference_ci"]["upper"]) < 0.0]
    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": EXPERIMENT_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"path": str(profile_path), "terminal_id": profile.terminal_id,
                        "assumed": profile.assumed,
                        "sha256": _profile_digest(profile_path)},
            "git": git, "config": asdict(cfg), "arm": ARM.value,
            "state_schema": SCHEMA, "q0": "greedy immediate-cost prior",
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "paired": paired,
        "gamma_improved_vs_baseline": improved,
    }
    _json_dump(out / "v1final_results.json", payload)
    report = _build_report(payload, curve, out)
    progress(f"[YR-030-b] completed in {payload['manifest']['elapsed_s']:.1f}s -> {report}")
    return report


@dataclass(frozen=True)
class _AblationShim:
    """coverage_ablation._fit_buckets/_gen_params 가 기대하는 필드 어댑터."""
    _cfg: V1FinalConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _build_report(payload: dict, curve: list[dict[str, object]], out: Path) -> Path:
    man, sel, paired = payload["manifest"], payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-030-b — v1 최종안 상태 + greedy-prior Q0 + γ grid")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. 사용자 정의 상태(v1_final)와 "
             "greedy 초기화·할인율의 순서품질 효과 검증.")
    L.append("")
    L.append(f"- baseline (validation 선택): **{sel['_baseline']['policy']}**")
    L.append(f"- γ 중 baseline 을 유의하게 이긴 것: "
             f"{payload['gamma_improved_vs_baseline'] or '**없음**'}")
    L.append("")
    L.append("## locked test — paired vs " + str(sel["_baseline"]["policy"]))
    L.append("")
    L.append("| variant | 선택 ep | val_mean | coverage | mean_wait Δ [95% CI] | p95 Δ% CI 상한 |")
    L.append("|---|---|---|---|---|---|")
    for name, entry in paired.items():
        s = sel.get(name, {})
        mw_point = entry["mean_wait"]["difference"]
        ci = entry["mean_wait"]["difference_ci"]
        p95 = entry["p95_wait"].get("percent_change_ci")
        p95_txt = "—" if p95 is None else f"{p95['upper']:+.1f}%"
        L.append(f"| {name} | {s.get('episode', '—')} "
                 f"| {s.get('val_mean_wait_min', float('nan')):.2f} "
                 f"| {entry['test_signature_coverage']:.1%} "
                 f"| {mw_point:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}] "
                 f"| {p95_txt} |")
    L.append("")
    L.append("## checkpoint 곡선 (validation)")
    L.append("")
    L.append("| γ | episode | val_mean | signature coverage | table_keys |")
    L.append("|---|---|---|---|---|")
    step = max(1, len(curve) // 48)
    for row in curve[::step]:
        L.append(f"| {row['gamma']:g} | {row['episode']} "
                 f"| {row['val_mean_wait_min']:.2f} "
                 f"| {row['val_signature_coverage']:.1%} | {row['table_keys']} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.state_v1_final — 원자료 v1final_results.json*")
    path = out / "v1final_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
