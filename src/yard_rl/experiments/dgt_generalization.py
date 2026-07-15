"""YR-042 — DGT 근사 프로파일 일반화 게이트.

질문: YR-039 승리(POC-MULTI)가 DGT 치수·ARMG 속도 프로파일에서도 재현되는가?
arm: (a) baseline 2종(val 선택) (b) POC 학습 dueling **zero-shot** (인코딩이
스키마 유도라 무수정 평가 가능 — 프로파일 일반화 시험) (c) DGT **재학습** dueling.
판정: 재학습 CI 상한<0 재현 여부 + zero-shot 의 이전 가능성. guardrail 동시 보고.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..integrated.dqn_learner import CandidateDQNLearner
from ..integrated.profiles import build_dgt_approx_profile
from ..integrated.scenario_gen import TerminalGenParams
from .candidate_dqn_experiment import (BASELINES, _dims, _eval_policy,
                                       _train_variant)
from .direct_job_runner import _git_state, _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-042-dgt-generalization"
POC_MODEL = "outputs/reports/candidate_dqn_poc/model_CandidateDQN[dueling].pt"


@dataclass(frozen=True)
class DgtGenConfig:
    train_episodes: int = 500
    validation_episodes: int = 20
    test_episodes: int = 60
    checkpoint_every: int = 25
    train_seed0: int = 330_000
    validation_seed0: int = 340_000
    test_seed0: int = 350_000
    n_external: int = 40
    n_vessels: int = 2
    bootstrap_seed: int = 76_042
    bootstrap_resamples: int = 10_000
    include_zero_shot: bool = True
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes,
               self.test_episodes, self.checkpoint_every) <= 0:
            raise ValueError("all sizes must be positive")
        bands = [set(self.train_seeds), set(self.validation_seeds),
                 set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        used = set(range(0, 250_000)) | set(range(300_000, 320_100))
        if any(band & used for band in bands):
            raise ValueError("기존 실험 seed 대역 재사용 금지")

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


def quick_dgt_gen_config() -> DgtGenConfig:
    return DgtGenConfig(train_episodes=6, validation_episodes=2, test_episodes=3,
                        checkpoint_every=3, n_external=8, n_vessels=1,
                        bootstrap_resamples=200, include_zero_shot=False,
                        quick=True)


def run_dgt_generalization(out_dir: str = "outputs/reports/dgt_generalization",
                           cfg: DgtGenConfig | None = None,
                           progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or DgtGenConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-042 run requires a clean committed tree")
    profile = build_dgt_approx_profile()
    params = (TerminalGenParams(n_external=cfg.n_external,
                                n_vessels=cfg.n_vessels, vessel_moves=6,
                                horizon_s=7_200.0, drain_window_s=3_600.0)
              if cfg.quick else
              TerminalGenParams(n_external=cfg.n_external,
                                n_vessels=cfg.n_vessels))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dims = _dims(profile, params, cfg.train_seeds[0])
    fit_rows = _eval_policy(profile, params, cfg.train_seeds[:5],
                            preference_factory=BASELINES["BASELINE_VESSEL_WAIT"])
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions)
                                 for r in fit_rows))
    progress(f"[YR-042] profile={profile.terminal_id} dims={dims} "
             f"cost_scale={cost_scale:.1f}")

    base_val = {name: _eval_policy(profile, params, cfg.validation_seeds,
                                   preference_factory=factory)
                for name, factory in BASELINES.items()}
    baseline_name = min(base_val,
                        key=lambda n: (fmean(r.total_cost for r in base_val[n]), n))
    selections: dict[str, object] = {"_baseline": {
        "policy": baseline_name,
        "validation_total_cost": fmean(r.total_cost for r in base_val[baseline_name])}}
    progress(f"[baseline] {baseline_name}")

    policies: dict[str, CandidateDQNLearner] = {}
    if cfg.include_zero_shot and Path(POC_MODEL).exists():
        zs = CandidateDQNLearner.load(POC_MODEL)
        if zs.dims != dims:
            raise RuntimeError(f"zero-shot dims 불일치: {zs.dims} != {dims}")
        policies["DuelingDQN[zero-shot POC]"] = zs
        selections["DuelingDQN[zero-shot POC]"] = {"source": POC_MODEL,
                                                   "trained_on": "POC-MULTI"}
    curve, selection, retrained = _train_variant(
        "dueling", profile, params, cfg, dims, cost_scale, progress)
    policies["DuelingDQN[DGT-retrained]"] = retrained
    selections["DuelingDQN[DGT-retrained]"] = selection
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)

    def rows_of(results, seeds):
        return [{"seed": s, "total_cost": r.total_cost,
                 "mean_wait_min": r.mean_wait_min, "p95_wait_min": r.p95_wait_min,
                 "vessel_delay_min": r.vessel_delay_min,
                 "completion_rate": r.completion_rate, "backlog": r.backlog,
                 "invariants_ok": r.invariants_ok}
                for s, r in zip(seeds, results)]

    test_rows: dict[str, list[dict]] = {}
    progress(f"[test] {baseline_name}")
    test_rows[baseline_name] = rows_of(
        _eval_policy(profile, params, cfg.test_seeds,
                     preference_factory=BASELINES[baseline_name]), cfg.test_seeds)
    for name, learner in policies.items():
        progress(f"[test] {name}")
        test_rows[name] = rows_of(
            _eval_policy(profile, params, cfg.test_seeds, learner=learner),
            cfg.test_seeds)
    retrained.save(out / "model_DuelingDQN_DGT.pt")
    _json_dump(out / "test_results.json", test_rows)

    base_rows = test_rows[baseline_name]
    seeds = [r["seed"] for r in base_rows]
    paired: dict[str, object] = {}
    for offset, name in enumerate(policies):
        alt = test_rows[name]
        entry: dict[str, object] = {}
        for m_off, (label, key_) in enumerate((("total_cost", "total_cost"),
                                               ("mean_wait", "mean_wait_min"),
                                               ("p95_wait", "p95_wait_min"),
                                               ("vessel_delay", "vessel_delay_min"))):
            stats = paired_bootstrap(
                [float(r[key_]) for r in base_rows],
                [float(r[key_]) for r in alt],
                metric=MetricSpec(key_, MetricDirection.MINIMIZE), seeds=seeds,
                seed=cfg.bootstrap_seed + offset * 10 + m_off,
                n_resamples=cfg.bootstrap_resamples)
            entry[label] = stats.as_dict()
        p95_ci = entry["p95_wait"].get("percent_change_ci")
        entry["guardrails"] = {
            "p95_pct_change_ci_upper": (None if p95_ci is None
                                        else float(p95_ci["upper"])),
            "p95_within_5pct": (None if p95_ci is None
                                else float(p95_ci["upper"]) <= 5.0),
            "completion_all_100pct": all(r["completion_rate"] >= 1.0 for r in alt),
            "max_backlog": max(int(r["backlog"]) for r in alt),
            "invariants_all_ok": all(bool(r["invariants_ok"]) for r in alt),
        }
        paired[name] = entry

    improved = [n for n in policies
                if float(paired[n]["total_cost"]["difference_ci"]["upper"]) < 0.0]
    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": EXPERIMENT_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"terminal_id": profile.terminal_id,
                        "assumed": profile.assumed,
                        "approx_note": "육/해측 역할분리·AGV 스케줄 미반영 근사"},
            "git": git, "config": asdict(cfg), "cost_scale": cost_scale,
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "paired": paired,
        "improved_vs_baseline": improved,
    }
    _json_dump(out / "dgt_generalization_results.json", payload)
    report = _build_report(payload, out)
    progress(f"[YR-042] completed in {payload['manifest']['elapsed_s']:.1f}s"
             f" -> {report}")
    return report


def _build_report(payload: dict, out: Path) -> Path:
    sel, paired = payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-042 — DGT 근사 프로파일 일반화 게이트")
    L.append("")
    L.append("> ⚠ DGT-APPROX-2CR (dgt_armg 치수·ARMG 속도, 역할분리 미반영 근사)"
             " + 합성 시나리오. YR-039 승리의 프로파일 강건성 확인.")
    L.append("")
    L.append(f"- baseline (validation 선택): **{sel['_baseline']['policy']}**")
    L.append(f"- baseline 을 유의하게 이긴 정책: "
             f"{payload['improved_vs_baseline'] or '**없음**'}")
    L.append("")
    L.append("## locked test — paired vs " + str(sel["_baseline"]["policy"]))
    L.append("")
    L.append("| 정책 | total_cost Δ [95% CI] | mean 대기 Δ | p95 Δ% 상한 "
             "| guardrail (P95≤+5%/완료/backlog0/invariant) |")
    L.append("|---|---|---|---|---|")
    for name, entry in paired.items():
        mw = entry["total_cost"]["difference"]
        ci = entry["total_cost"]["difference_ci"]
        p95 = entry["p95_wait"].get("percent_change_ci")
        p95_txt = "—" if p95 is None else f"{p95['upper']:+.1f}%"
        g = entry["guardrails"]
        mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
        g_txt = (f"{mark(bool(g['p95_within_5pct']))}/"
                 f"{mark(g['completion_all_100pct'])}/"
                 f"{mark(g['max_backlog'] == 0)}/{mark(g['invariants_all_ok'])}")
        L.append(f"| {name} | {mw:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}] "
                 f"| {entry['mean_wait']['difference']:+.3f} "
                 f"| {p95_txt} | {g_txt} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.dgt_generalization — "
             "원자료 dgt_generalization_results.json*")
    path = out / "dgt_generalization_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
