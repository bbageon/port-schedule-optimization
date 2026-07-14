"""YR-012 — 잔차 연속-feature Δ 학습 (함수근사) 실행기 (사전등록 동결).

arm: ResidualDeltaNet (MLP, bucket 없음) vs 휴리스틱 6종(validation 선택
baseline) + YR-030-c tabular state_job agent 재평가 reference (해상도 대조).
사전등록: .claude/docs/strategy-history/2026-07-14-YR-012-residual-delta-net-prereg.md
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
from ..policies.direct_baselines import (DirectJobRulePolicy, DirectRule,
                                         direct_baseline_policies)
from ..policies.residual_cost_q import ResidualCostQAgent
from ..policies.residual_delta_net import (DeltaNetConfig, FeatureScaler,
                                           ResidualDeltaNetAgent,
                                           extract_features)
from .coverage_ablation import _evaluate, _gen_params
from .direct_job_runner import (_git_state, _json_dump, _profile_digest,
                                _run_cost_q_episode, _scenario,
                                _scenario_descriptor)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-012-residual-delta-net"
ARM = SLAMode.OFF
SCHEMA = "v1_final"
ARM_NAME = "ResidualDeltaNet"
REFERENCE_AGENT = ("outputs/reports/costq_residual_hjnc/"
                   "agent_ResidualCostQ[state_job].json")
REFERENCE_BUCKETS = "outputs/reports/costq_residual_hjnc/direct_buckets.json"
REFERENCE_NAME = "ResidualCostQ[state_job](YR-030-c ref)"


@dataclass(frozen=True)
class DeltaExpConfig:
    train_episodes: int = 3_000
    validation_episodes: int = 30
    test_episodes: int = 100
    checkpoint_every: int = 50
    n_external: int = 100
    gamma: float = 0.95
    lr: float = 1e-3
    hidden: int = 64
    train_seed0: int = 140_000
    validation_seed0: int = 150_000
    test_seed0: int = 160_000
    scaler_fit_episodes: int = 1_000
    bootstrap_seed: int = 74_012
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    include_reference: bool = True
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.checkpoint_every, self.n_external,
               self.scaler_fit_episodes) <= 0:
            raise ValueError("all sizes must be positive")
        bands = [set(self.train_seeds), set(self.validation_seeds),
                 set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy = (set(range(10_000, 11_000)) | set(range(20_000, 20_030))
                  | set(range(30_000, 30_100)) | set(range(40_000, 43_000))
                  | set(range(50_000, 50_030)) | set(range(60_000, 60_100))
                  | set(range(70_000, 73_000)) | set(range(80_000, 80_030))
                  | set(range(90_000, 90_100)) | set(range(110_000, 113_000))
                  | set(range(120_000, 120_030)) | set(range(130_000, 130_100)))
        if any(band & legacy for band in bands):
            raise ValueError("기존 실험 seed band 재사용 금지 (사전등록)")

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


def quick_delta_config() -> DeltaExpConfig:
    return DeltaExpConfig(
        train_episodes=12, validation_episodes=3, test_episodes=4,
        checkpoint_every=4, n_external=12, scaler_fit_episodes=4,
        bootstrap_resamples=200, include_reference=False, quick=True)


@dataclass(frozen=True)
class _CfgShim:
    _cfg: DeltaExpConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _fit_scaler(profile, seeds: Sequence[int], params, cfg: DeltaExpConfig,
                progress: Callable[[str], None]) -> FeatureScaler:
    """FIFO train 관측 → 연속 feature mean/std (fit 후 동결, §2)."""
    rows: list[list[float]] = []
    fifo = DirectJobRulePolicy(DirectRule.FIFO)
    for index, seed in enumerate(seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        env = DirectJobEnv(profile, sla_mode=ARM, expected_n_config=cfg.n_external,
                           state_schema=SCHEMA)
        state, info = env.reset(scenario)
        while state is not None:
            rows.extend(extract_features(c) for c in info.feasible_candidates)
            state, _cost, _done, info = env.step(fifo.act(state, info.candidates))
        if index % max(1, min(200, len(seeds))) == 0 or index == len(seeds):
            progress(f"[scaler] FIFO train {index}/{len(seeds)} ({len(rows)} rows)")
    return FeatureScaler.fit(rows)


def _train_delta(profile, validation, params, scaler, cfg: DeltaExpConfig,
                 progress: Callable[[str], None]
                 ) -> tuple[list[dict[str, object]], dict[str, object],
                            ResidualDeltaNetAgent]:
    checkpoints = set(range(cfg.checkpoint_every, cfg.train_episodes + 1,
                            cfg.checkpoint_every))
    checkpoints.add(cfg.train_episodes)
    agent = ResidualDeltaNetAgent(
        DeltaNetConfig(gamma=cfg.gamma, lr=cfg.lr, hidden=cfg.hidden),
        scaler=scaler, seed=cfg.train_seed0)
    curve: list[dict[str, object]] = []
    best: tuple[float, int, ResidualDeltaNetAgent] | None = None
    buckets = DirectJobBucketConfig()  # net 은 bucket 미사용 — env 인코딩용 기본값
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
        curve.append({"arm": ARM_NAME, "episode": episode,
                      "val_mean_wait_min": mean})
        progress(f"[train:{ARM_NAME}] ep={episode}/{cfg.train_episodes} "
                 f"val={mean:.3f}")
        if best is None or (mean, episode) < (best[0], best[1]):
            best = (mean, episode, snapshot)
    mean, episode, selected = best
    selected.reset_diagnostics()
    selection = {"arm": ARM_NAME, "gamma": cfg.gamma, "lr": cfg.lr,
                 "hidden": cfg.hidden, "episode": episode,
                 "val_mean_wait_min": mean}
    return curve, selection, selected


def run_delta_experiment(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                         out_dir: str = "outputs/reports/residual_delta_hjnc",
                         cfg: DeltaExpConfig | None = None,
                         progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or DeltaExpConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-012 run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_CfgShim(cfg))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-012] profile={profile.terminal_id} arm={ARM_NAME} "
             f"gamma={cfg.gamma} lr={cfg.lr} hidden={cfg.hidden} "
             f"train={cfg.train_episodes}")

    scaler = _fit_scaler(
        profile, cfg.train_seeds[:min(cfg.scaler_fit_episodes, cfg.train_episodes)],
        params, cfg, progress)
    scaler.save(out / "feature_scaler.json")
    validation = [_scenario(profile, s, params, cfg.n_external)
                  for s in cfg.validation_seeds]
    tests = [_scenario(profile, s, params, cfg.n_external) for s in cfg.test_seeds]
    _json_dump(out / "seed_manifest.json", {
        "validation": [_scenario_descriptor(s) for s in validation],
        "test": [_scenario_descriptor(s) for s in tests],
        "train_seed0": cfg.train_seed0,
        "bands_disjoint_from_prior_experiments": True,
    })

    curve, selection, agent = _train_delta(profile, validation, params, scaler,
                                           cfg, progress)
    _json_dump(out / "checkpoint_curve.json", curve)
    selections: dict[str, object] = {ARM_NAME: selection}

    buckets = DirectJobBucketConfig()
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
    progress(f"[test] {ARM_NAME}")
    test_rows.extend(_evaluate(copy.deepcopy(agent), SCHEMA, profile, tests,
                               buckets, _CfgShim(cfg), ARM_NAME))
    agent.save(out / "model_ResidualDeltaNet.pt")
    compare_names = [ARM_NAME]
    if cfg.include_reference and Path(REFERENCE_AGENT).exists():
        ref = ResidualCostQAgent.load(REFERENCE_AGENT)
        ref_buckets = DirectJobBucketConfig.load(REFERENCE_BUCKETS)
        progress(f"[test] {REFERENCE_NAME}")
        test_rows.extend(_evaluate(ref, SCHEMA, profile, tests, ref_buckets,
                                   _CfgShim(cfg), REFERENCE_NAME))
        compare_names.append(REFERENCE_NAME)
    _json_dump(out / "test_results.json", test_rows)

    base_rows = sorted((r for r in test_rows if r["policy"] == baseline_name),
                       key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in base_rows]
    paired: dict[str, object] = {}
    for offset, name in enumerate(compare_names):
        alt_rows = sorted((r for r in test_rows if r["policy"] == name),
                          key=lambda r: int(r["seed"]))
        entry: dict[str, object] = {}
        for m_off, (label, metric) in enumerate((("mean_wait", "mean_wait_min"),
                                                 ("p95_wait", "p95_wait_min"))):
            stats = paired_bootstrap(
                [float(r["metrics"][metric]) for r in base_rows],
                [float(r["metrics"][metric]) for r in alt_rows],
                metric=MetricSpec(metric, MetricDirection.MINIMIZE),
                seeds=seeds, seed=cfg.bootstrap_seed + offset * 10 + m_off,
                n_resamples=cfg.bootstrap_resamples)
            entry[label] = stats.as_dict()
        p95_ci = entry["p95_wait"].get("percent_change_ci")
        entry["guardrails"] = {
            "p95_pct_change_ci_upper": (None if p95_ci is None
                                        else float(p95_ci["upper"])),
            "p95_within_5pct": (None if p95_ci is None
                                else float(p95_ci["upper"]) <= 5.0),
            "completion_all_100pct": all(
                float(r["metrics"]["completion_rate"]) >= 1.0 for r in alt_rows),
            "max_backlog": max(int(r["metrics"]["backlog"]) for r in alt_rows),
            "invariants_all_ok": all(
                bool(r["metrics"]["invariants_ok"]) for r in alt_rows),
        }
        paired[name] = entry

    improved = [n for n in compare_names
                if float(paired[n]["mean_wait"]["difference_ci"]["upper"]) < 0.0]
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
            "policy": "Q_total = exact greedy G + Delta_theta(x) — continuous "
                      "features, output-layer zero-init (YR-012 prereg)",
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "paired": paired,
        "improved_vs_baseline": improved,
    }
    _json_dump(out / "delta_net_results.json", payload)
    report = _build_report(payload, curve, out)
    progress(f"[YR-012] completed in {payload['manifest']['elapsed_s']:.1f}s"
             f" -> {report}")
    return report


def _build_report(payload: dict, curve: list[dict[str, object]], out: Path) -> Path:
    sel, paired = payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-012 — 잔차 연속-feature Δ 학습 (함수근사)")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. Q_total = 정확한 greedy G + "
             "Δθ(연속 x) — bucket 해상도 세금 가설의 직접 시험.")
    L.append("")
    L.append(f"- baseline (validation 선택): **{sel['_baseline']['policy']}**")
    L.append(f"- baseline 을 유의하게 이긴 정책: "
             f"{payload['improved_vs_baseline'] or '**없음**'}")
    L.append("")
    L.append("## locked test — paired vs " + str(sel["_baseline"]["policy"]))
    L.append("")
    L.append("| 정책 | 선택 ep | val_mean | mean_wait Δ [95% CI] "
             "| p95 Δ% CI 상한 | guardrail (P95≤+5%/완료100%/backlog0/invariant) |")
    L.append("|---|---|---|---|---|---|")
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
                 f"| {mw:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}] "
                 f"| {p95_txt} | {g_txt} |")
    L.append("")
    L.append("## checkpoint 곡선 (validation)")
    L.append("")
    L.append("| episode | val_mean |")
    L.append("|---|---|")
    step = max(1, len(curve) // 40)
    for row in curve[::step]:
        L.append(f"| {row['episode']} | {row['val_mean_wait_min']:.2f} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.residual_delta_experiment — "
             "원자료 delta_net_results.json*")
    path = out / "delta_net_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
