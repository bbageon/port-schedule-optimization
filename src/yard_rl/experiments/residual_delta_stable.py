"""YR-012-b — Δ-net 학습 안정화 (replay buffer + target network).

YR-012 확정 3 (남은 용의자 = 학습 안정성: checkpoint 곡선 7.97~10.29 진동)의
직접 후속. 정책·feature·잔차 구조는 YR-012 그대로, 학습 절차만 DQN 표준화:
- replay buffer: 상관 깨기 + 경험 재사용 (배치 무작위 샘플)
- target network: bootstrap 과녁을 N gradient step 동안 고정 (정보 없는 변화 차단)
per-step gradient 업데이트 수는 1로 유지 — YR-012 와 계산 예산 등가.

사전등록: .claude/docs/strategy-history/2026-07-15-YR-012-b-delta-stable-prereg.md
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..envs.direct_job_env import DirectJobBucketConfig, DirectJobEnv, SLAMode
from ..io.profile_loader import load_profile
from ..policies.direct_baselines import direct_baseline_policies
from ..policies.residual_delta_net import (DeltaNetConfig, ResidualDeltaNetAgent)
from .coverage_ablation import _evaluate, _gen_params
from .direct_job_runner import (_git_state, _json_dump, _profile_digest,
                                _run_cost_q_episode, _scenario,
                                _scenario_descriptor)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap
from .residual_delta_experiment import _fit_scaler

EXPERIMENT_ID = "YR-012-b-delta-net-stabilized"
ARM = SLAMode.OFF
SCHEMA = "v1_final"
YR012_MODEL = "outputs/reports/residual_delta_hjnc/model_ResidualDeltaNet.pt"
YR012_NAME = "ResidualDeltaNet[online](YR-012 ref)"


@dataclass(frozen=True)
class StableExpConfig:
    train_episodes: int = 3_000
    validation_episodes: int = 30
    test_episodes: int = 100
    checkpoint_every: int = 50
    n_external: int = 100
    gamma: float = 0.95
    lr: float = 1e-3
    hidden: int = 64
    replay_capacity: int = 100_000
    batch_size: int = 64
    min_replay: int = 1_000
    target_syncs: tuple[int, ...] = (500, 2_000)   # grid — 과녁 고정 주기
    train_seed0: int = 170_000
    validation_seed0: int = 180_000
    test_seed0: int = 190_000
    scaler_fit_episodes: int = 1_000
    bootstrap_seed: int = 74_112
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    include_reference: bool = True
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.checkpoint_every, self.n_external,
               self.scaler_fit_episodes) <= 0:
            raise ValueError("all sizes must be positive")
        if not self.target_syncs or any(s <= 0 for s in self.target_syncs):
            raise ValueError("target_syncs must be positive")
        bands = [set(self.train_seeds), set(self.validation_seeds),
                 set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy = (set(range(10_000, 11_000)) | set(range(20_000, 20_030))
                  | set(range(30_000, 30_100)) | set(range(40_000, 43_000))
                  | set(range(50_000, 50_030)) | set(range(60_000, 60_100))
                  | set(range(70_000, 73_000)) | set(range(80_000, 80_030))
                  | set(range(90_000, 90_100)) | set(range(110_000, 113_000))
                  | set(range(120_000, 120_030)) | set(range(130_000, 130_100))
                  | set(range(140_000, 143_000)) | set(range(150_000, 150_030))
                  | set(range(160_000, 160_100)))
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


def quick_stable_config() -> StableExpConfig:
    return StableExpConfig(
        train_episodes=12, validation_episodes=3, test_episodes=4,
        checkpoint_every=4, n_external=12, scaler_fit_episodes=4,
        replay_capacity=2_000, min_replay=20, batch_size=16,
        target_syncs=(50,), bootstrap_resamples=200,
        include_reference=False, quick=True)


@dataclass(frozen=True)
class _CfgShim:
    _cfg: StableExpConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _arm_name(sync: int) -> str:
    return f"DeltaNet[replay|sync{sync}]"


def _train_arm(sync: int, profile, validation, params, scaler,
               cfg: StableExpConfig, progress: Callable[[str], None]
               ) -> tuple[list[dict[str, object]], dict[str, object],
                          ResidualDeltaNetAgent]:
    checkpoints = set(range(cfg.checkpoint_every, cfg.train_episodes + 1,
                            cfg.checkpoint_every))
    checkpoints.add(cfg.train_episodes)
    agent = ResidualDeltaNetAgent(
        DeltaNetConfig(gamma=cfg.gamma, lr=cfg.lr, hidden=cfg.hidden,
                       replay_capacity=cfg.replay_capacity,
                       batch_size=cfg.batch_size, min_replay=cfg.min_replay,
                       target_sync_every=sync),
        scaler=scaler, seed=cfg.train_seed0 + sync)
    name = _arm_name(sync)
    buckets = DirectJobBucketConfig()
    curve: list[dict[str, object]] = []
    best: tuple[float, int, ResidualDeltaNetAgent] | None = None
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
        curve.append({"arm": name, "episode": episode, "val_mean_wait_min": mean})
        progress(f"[train:{name}] ep={episode}/{cfg.train_episodes} val={mean:.3f}")
        if best is None or (mean, episode) < (best[0], best[1]):
            best = (mean, episode, snapshot)
    mean, episode, selected = best
    selected.reset_diagnostics()
    selection = {"arm": name, "target_sync_every": sync, "episode": episode,
                 "val_mean_wait_min": mean}
    return curve, selection, selected


def run_stable_experiment(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                          out_dir: str = "outputs/reports/residual_delta_stable_hjnc",
                          cfg: StableExpConfig | None = None,
                          progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or StableExpConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-012-b run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_CfgShim(cfg))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-012-b] profile={profile.terminal_id} syncs={list(cfg.target_syncs)} "
             f"replay={cfg.replay_capacity} batch={cfg.batch_size} "
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

    curve: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    agents: dict[str, ResidualDeltaNetAgent] = {}
    for sync in cfg.target_syncs:
        arm_curve, selection, agent = _train_arm(sync, profile, validation,
                                                 params, scaler, cfg, progress)
        curve.extend(arm_curve)
        selections[_arm_name(sync)] = selection
        agents[_arm_name(sync)] = agent
    _json_dump(out / "checkpoint_curve.json", curve)

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
    for name, agent in agents.items():
        progress(f"[test] {name}")
        test_rows.extend(_evaluate(copy.deepcopy(agent), SCHEMA, profile, tests,
                                   buckets, _CfgShim(cfg), name))
        agent.save(out / f"model_{name.replace('|', '_')}.pt")
    compare_names = list(agents)
    if cfg.include_reference and Path(YR012_MODEL).exists():
        ref = ResidualDeltaNetAgent.load(YR012_MODEL)
        progress(f"[test] {YR012_NAME}")
        test_rows.extend(_evaluate(ref, SCHEMA, profile, tests, buckets,
                                   _CfgShim(cfg), YR012_NAME))
        compare_names.append(YR012_NAME)
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
            "policy": "Q_total = G + Delta_theta(x); replay buffer + target "
                      "network (per-step 1 update — YR-012 예산 등가)",
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "paired": paired,
        "improved_vs_baseline": improved,
    }
    _json_dump(out / "delta_stable_results.json", payload)
    report = _build_report(payload, curve, out)
    progress(f"[YR-012-b] completed in {payload['manifest']['elapsed_s']:.1f}s"
             f" -> {report}")
    return report


def _build_report(payload: dict, curve: list[dict[str, object]], out: Path) -> Path:
    sel, paired = payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-012-b — Δ-net 학습 안정화 (replay buffer + target network)")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. YR-012 진동(7.97~10.29) 해소가 목적 —")
    L.append("> 정책·feature 불변, 학습 절차만 DQN 표준화 (per-step 1 update 예산 등가).")
    L.append("")
    L.append(f"- baseline (validation 선택): **{sel['_baseline']['policy']}**")
    L.append(f"- baseline 을 유의하게 이긴 정책: "
             f"{payload['improved_vs_baseline'] or '**없음**'}")
    L.append("")
    L.append("## locked test — paired vs " + str(sel["_baseline"]["policy"]))
    L.append("")
    L.append("| 정책 | 선택 ep | val_mean | mean_wait Δ [95% CI] "
             "| p95 Δ% CI 상한 | guardrail |")
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
    L.append("## checkpoint 곡선 (validation, arm 별) — 진동 폭이 1차 관찰 대상")
    L.append("")
    L.append("| arm | episode | val_mean |")
    L.append("|---|---|---|")
    step = max(1, len(curve) // 48)
    for row in curve[::step]:
        L.append(f"| {row['arm']} | {row['episode']} "
                 f"| {row['val_mean_wait_min']:.2f} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.residual_delta_stable — "
             "원자료 delta_stable_results.json*")
    path = out / "delta_stable_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
