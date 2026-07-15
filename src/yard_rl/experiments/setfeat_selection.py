"""YR-033 — checkpoint 선택 프로토콜 보완 (winner's curse 진단·강건 재선택).

YR-012-c: SetFeat[22] 격차 +0.035 (greedy 통계적 동률, CI 0 포함) — 단 선택은
60 checkpoint × val 30일 argmin 이라 winner's curse 의심. YR-012-b 에서 val-test
순위 역전이 실측됨.

방법 (spec YR-033): 학습기·feature·비용 불변. YR-012-c 학습을 **결정론적으로
재실행**(동일 seed·train band → 동일 60 checkpoint 재현) 후, 세 선택 프로토콜을
동일 checkpoint 묶음에 적용해 fresh test 로 비교:
- **P1_val30**: val 30일 argmin (YR-012-c 재현)
- **P2_val90**: val 90일(30일 superset) argmin — 표본 확대
- **P3_val90_smooth3**: 90일 val 곡선 3-checkpoint 이동평균 argmin — 단일 운luck 제거

test 는 선택에 절대 미사용 (spec 범위 밖). 전 checkpoint test 평가는 winner's
curse 진단(val-test Spearman·오선택 optimism·최적선택 하한)에만 사용.

사전등록: .claude/docs/strategy-history/2026-07-15-YR-033-checkpoint-selection-prereg.md
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
from ..policies.direct_baselines import direct_baseline_policies
from ..policies.residual_delta_net import (DeltaNetConfig, ResidualDeltaNetAgent)
from .coverage_ablation import _evaluate, _gen_params
from .direct_job_runner import (_git_state, _json_dump, _profile_digest,
                                _run_cost_q_episode, _scenario,
                                _scenario_descriptor)
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap
from .residual_setfeat_experiment import _fit_scaler_set

EXPERIMENT_ID = "YR-033-checkpoint-selection"
ARM = SLAMode.OFF
SCHEMA = "v1_final"


@dataclass(frozen=True)
class SelectConfig:
    train_episodes: int = 3_000     # YR-012-c 와 동일 (결정론 재현)
    val_episodes: int = 90          # 30(YR-012-c) superset
    val30_episodes: int = 30        # P1 재현용 부분집합
    test_episodes: int = 100
    checkpoint_every: int = 50
    smooth_window: int = 3
    n_external: int = 100
    gamma: float = 0.95
    lr: float = 1e-3
    hidden: int = 64
    train_seed0: int = 200_000      # YR-012-c 와 동일 train band (동일 trajectory)
    validation_seed0: int = 210_000  # YR-012-c val superset
    test_seed0: int = 240_000       # fresh (220k=YR-012-c test 재사용 금지)
    scaler_fit_episodes: int = 1_000
    bootstrap_seed: int = 74_233
    bootstrap_resamples: int = 10_000
    drain_window_s: float = 86_400.0
    quick: bool = False

    def __post_init__(self) -> None:
        if self.val30_episodes > self.val_episodes:
            raise ValueError("val30 은 val 의 부분집합이어야 함")
        if self.smooth_window < 1 or self.smooth_window % 2 == 0:
            raise ValueError("smooth_window 는 홀수 ≥1")
        bands = [set(self.train_seeds), set(self.validation_seeds),
                 set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy = set()
        for lo, n in ((10_000, 1_000), (20_000, 30), (30_000, 100), (40_000, 3_000),
                      (50_000, 30), (60_000, 100), (70_000, 3_000), (80_000, 30),
                      (90_000, 100), (110_000, 3_000), (120_000, 30), (130_000, 100),
                      (140_000, 3_000), (150_000, 30), (160_000, 100),
                      (170_000, 3_000), (180_000, 30), (190_000, 100),
                      (220_000, 100)):
            legacy |= set(range(lo, lo + n))
        # train/val 은 YR-012-c 와 동일 band 를 '의도적으로' 재사용 (결정론 재현·val superset)
        if set(self.test_seeds) & legacy:
            raise ValueError("test band 는 기존 실험과 겹치면 안 됨 (재선택 오염 방지)")

    @property
    def train_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.train_seed0, self.train_seed0 + self.train_episodes))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.validation_seed0,
                           self.validation_seed0 + self.val_episodes))

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.test_seed0, self.test_seed0 + self.test_episodes))


def quick_select_config() -> SelectConfig:
    return SelectConfig(train_episodes=16, val_episodes=6, val30_episodes=3,
                        test_episodes=4, checkpoint_every=4, smooth_window=3,
                        n_external=12, scaler_fit_episodes=4,
                        bootstrap_resamples=200, quick=True)


@dataclass(frozen=True)
class _CfgShim:
    _cfg: SelectConfig

    @property
    def n_external(self) -> int:
        return self._cfg.n_external

    @property
    def drain_window_s(self) -> float:
        return self._cfg.drain_window_s


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """순위 상관 (동점 평균순위 Pearson) — scipy 미사용."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j < len(v) and v[order[j]] == v[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                r[order[k]] = avg
            i = j
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def run_selection_experiment(profile_path: str = "configs/terminals/hjnc_armg.yaml",
                             out_dir: str = "outputs/reports/setfeat_selection_hjnc",
                             cfg: SelectConfig | None = None,
                             progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or SelectConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-033 run requires a clean committed tree")
    profile = load_profile(profile_path)
    params = _gen_params(_CfgShim(cfg))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(f"[YR-033] profile={profile.terminal_id} train={cfg.train_episodes} "
             f"val={cfg.val_episodes} test={cfg.test_episodes} (fresh)")

    # scaler: YR-012-c 와 동일 (같은 train FIFO fit → 결정론)
    scaler = _fit_scaler_set(
        profile, cfg.train_seeds[:min(cfg.scaler_fit_episodes, cfg.train_episodes)],
        params, cfg, progress)
    scaler.save(out / "feature_scaler.json")
    val_scen = [_scenario(profile, s, params, cfg.n_external)
                for s in cfg.validation_seeds]
    test_scen = [_scenario(profile, s, params, cfg.n_external)
                 for s in cfg.test_seeds]
    _json_dump(out / "seed_manifest.json", {
        "validation": [_scenario_descriptor(s) for s in val_scen],
        "test": [_scenario_descriptor(s) for s in test_scen],
        "train_seed0": cfg.train_seed0, "note": "val=YR-012-c superset, test=fresh"})

    # ---- 결정론 재실행: 동일 60 checkpoint 재현 + 각 checkpoint 저장·평가
    buckets = DirectJobBucketConfig()
    agent = ResidualDeltaNetAgent(
        DeltaNetConfig(gamma=cfg.gamma, lr=cfg.lr, hidden=cfg.hidden,
                       use_set_context=True),
        scaler=scaler, seed=cfg.train_seed0)
    checkpoints = set(range(cfg.checkpoint_every, cfg.train_episodes + 1,
                            cfg.checkpoint_every))
    checkpoints.add(cfg.train_episodes)
    records: list[dict] = []
    snapshots: dict[int, ResidualDeltaNetAgent] = {}
    for episode, seed in enumerate(cfg.train_seeds, start=1):
        scenario = _scenario(profile, seed, params, cfg.n_external)
        env = DirectJobEnv(profile, sla_mode=ARM, bucket_cfg=buckets,
                           expected_n_config=cfg.n_external, state_schema=SCHEMA)
        _run_cost_q_episode(agent, env, scenario, train_episode=episode - 1)
        if episode not in checkpoints:
            continue
        snap = copy.deepcopy(agent)
        vrows = _evaluate(snap, SCHEMA, profile, val_scen, buckets, _CfgShim(cfg),
                          "val")
        vmeans = [float(r["metrics"]["mean_wait_min"]) for r in vrows]
        trows = _evaluate(snap, SCHEMA, profile, test_scen, buckets, _CfgShim(cfg),
                          f"ckpt{episode}")
        tmean = fmean(float(r["metrics"]["mean_wait_min"]) for r in trows)
        snap.reset_diagnostics()
        snapshots[episode] = snap
        records.append({
            "episode": episode,
            "val30_mean": fmean(vmeans[:cfg.val30_episodes]),
            "val90_mean": fmean(vmeans),
            "test_mean": tmean,             # 진단 전용 — 선택 미사용
            "_test_rows": trows})
        progress(f"[ckpt {episode}/{cfg.train_episodes}] val30={records[-1]['val30_mean']:.3f} "
                 f"val90={records[-1]['val90_mean']:.3f} test={tmean:.3f}")
    _json_dump(out / "checkpoint_records.json",
               [{k: v for k, v in r.items() if k != "_test_rows"} for r in records])

    # ---- 세 프로토콜 선택 (val 만 사용)
    eps = [r["episode"] for r in records]
    v30 = [r["val30_mean"] for r in records]
    v90 = [r["val90_mean"] for r in records]
    w = cfg.smooth_window // 2
    v90s = [fmean(v90[max(0, i - w):min(len(v90), i + w + 1)])
            for i in range(len(v90))]
    protocols = {
        "P1_val30": eps[min(range(len(v30)), key=lambda i: (v30[i], eps[i]))],
        "P2_val90": eps[min(range(len(v90)), key=lambda i: (v90[i], eps[i]))],
        "P3_val90_smooth3": eps[min(range(len(v90s)), key=lambda i: (v90s[i], eps[i]))],
    }

    # ---- greedy baseline + 선택 checkpoint fresh test paired
    greedy_name = "IMMEDIATE_COST_GREEDY"
    greedy_pol = next(p for p in direct_baseline_policies() if p.name == greedy_name)
    grows = sorted(_evaluate(greedy_pol, SCHEMA, profile, test_scen, buckets,
                             _CfgShim(cfg), greedy_name),
                   key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in grows]
    by_ep = {r["episode"]: r for r in records}
    paired: dict[str, object] = {}
    for offset, (pname, ep) in enumerate(protocols.items()):
        arows = sorted(by_ep[ep]["_test_rows"], key=lambda r: int(r["seed"]))
        entry: dict[str, object] = {"selected_episode": ep,
                                    "val30_mean": by_ep[ep]["val30_mean"],
                                    "val90_mean": by_ep[ep]["val90_mean"]}
        for m_off, (label, metric) in enumerate((("mean_wait", "mean_wait_min"),
                                                 ("p95_wait", "p95_wait_min"))):
            stats = paired_bootstrap(
                [float(r["metrics"][metric]) for r in grows],
                [float(r["metrics"][metric]) for r in arows],
                metric=MetricSpec(metric, MetricDirection.MINIMIZE),
                seeds=seeds, seed=cfg.bootstrap_seed + offset * 10 + m_off,
                n_resamples=cfg.bootstrap_resamples)
            entry[label] = stats.as_dict()
        p95_ci = entry["p95_wait"].get("percent_change_ci")
        entry["guardrails"] = {
            "p95_within_5pct": (None if p95_ci is None
                                else float(p95_ci["upper"]) <= 5.0),
            "completion_all_100pct": all(
                float(r["metrics"]["completion_rate"]) >= 1.0 for r in arows),
            "max_backlog": max(int(r["metrics"]["backlog"]) for r in arows),
            "invariants_all_ok": all(
                bool(r["metrics"]["invariants_ok"]) for r in arows)}
        entry["formal_win"] = float(entry["mean_wait"]["difference_ci"]["upper"]) < 0.0
        paired[pname] = entry

    # ---- winner's curse 진단 (test 는 진단에만)
    greedy_mean = fmean(float(r["metrics"]["mean_wait_min"]) for r in grows)
    best_test_ep = min(records, key=lambda r: r["test_mean"])["episode"]
    diagnostics = {
        "greedy_test_mean": greedy_mean,
        "spearman_val30_test": _spearman(v30, [r["test_mean"] for r in records]),
        "spearman_val90_test": _spearman(v90, [r["test_mean"] for r in records]),
        "best_achievable_test_mean": min(r["test_mean"] for r in records),
        "best_achievable_episode": best_test_ep,
        "best_achievable_delta_vs_greedy": min(r["test_mean"] for r in records) - greedy_mean,
        "p1_optimism_test_minus_best": (by_ep[protocols["P1_val30"]]["test_mean"]
                                        - min(r["test_mean"] for r in records)),
        "test_mean_spread": max(r["test_mean"] for r in records) - min(
            r["test_mean"] for r in records)}

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
            "note": "test 는 선택 미사용 (val 만 선택). test-per-ckpt 는 진단 전용.",
            "elapsed_s": time.time() - started},
        "protocols": protocols, "paired": paired, "diagnostics": diagnostics}
    _json_dump(out / "selection_results.json", payload)
    report = _build_report(payload, records, out)
    progress(f"[YR-033] completed in {payload['manifest']['elapsed_s']:.1f}s "
             f"-> {report}")
    return report


def _build_report(payload: dict, records: list[dict], out: Path) -> Path:
    paired, diag = payload["paired"], payload["diagnostics"]
    L: list[str] = []
    L.append("# YR-033 - checkpoint 선택 프로토콜 보완 (winner's curse)")
    L.append("")
    L.append("> ⚠ 가정 프로파일 + 합성 시나리오. YR-012-c SetFeat[22] 재실행(결정론 동일)")
    L.append("> 후, 선택 프로토콜만 바꿔 fresh test 비교. test 는 선택 미사용.")
    L.append("")
    L.append(f"- greedy fresh test 평균: **{diag['greedy_test_mean']:.3f}분**")
    L.append(f"- 최적선택 하한 (test argmin, 도달불가 상한): "
             f"{diag['best_achievable_test_mean']:.3f}분 "
             f"(Δ vs greedy {diag['best_achievable_delta_vs_greedy']:+.3f}, "
             f"ep{diag['best_achievable_episode']})")
    L.append("")
    L.append("## 선택 프로토콜 × fresh test - paired vs greedy")
    L.append("")
    L.append("| 프로토콜 | 선택 ep | val30 | val90 | mean_wait Δ [95% CI] | 형식승리 | p95 Δ% 상한 | guardrail |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, e in paired.items():
        mw = e["mean_wait"]["difference"]
        ci = e["mean_wait"]["difference_ci"]
        p95 = e["p95_wait"].get("percent_change_ci")
        p95_txt = "-" if p95 is None else f"{p95['upper']:+.1f}%"
        g = e["guardrails"]
        mark = lambda ok: "OK" if ok else "X"  # noqa: E731
        g_txt = (f"{mark(bool(g['p95_within_5pct']))}/{mark(g['completion_all_100pct'])}"
                 f"/{mark(g['max_backlog'] == 0)}/{mark(g['invariants_all_ok'])}")
        L.append(f"| {name} | {e['selected_episode']} | {e['val30_mean']:.2f} "
                 f"| {e['val90_mean']:.2f} | {mw:+.3f} [{ci['lower']:+.3f}, "
                 f"{ci['upper']:+.3f}] | {'✅' if e['formal_win'] else '미달'} "
                 f"| {p95_txt} | {g_txt} |")
    L.append("")
    L.append("## winner's curse 진단")
    L.append("")
    L.append(f"- val-test Spearman: val30 **{diag['spearman_val30_test']:+.3f}** · "
             f"val90 **{diag['spearman_val90_test']:+.3f}** "
             f"(1=완전예측·0=무작위·음수=역상관)")
    L.append(f"- P1(val30) optimism: test − 최적 = "
             f"**+{diag['p1_optimism_test_minus_best']:.3f}분** (노이즈 선택 손실)")
    L.append(f"- checkpoint 간 test 산포: {diag['test_mean_spread']:.3f}분")
    L.append("")
    L.append("## checkpoint 기록 (앞 20)")
    L.append("")
    L.append("| ep | val30 | val90 | test |")
    L.append("|---|---|---|---|")
    for r in records[:20]:
        L.append(f"| {r['episode']} | {r['val30_mean']:.2f} | {r['val90_mean']:.2f} "
                 f"| {r['test_mean']:.2f} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.setfeat_selection - 원자료 "
             "selection_results.json·checkpoint_records.json*")
    path = out / "selection_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
