"""YR-039 Stage B — Candidate DQN/DDQN/Dueling 3-arm 실험 (매핑 §3·§5 동결).

train(생성 시나리오) → 25ep 마다 validation checkpoint (YR-033: 선택은 val 만)
→ locked test 에서 baseline(동정보 휴리스틱, val 선택) 대비 총비용 paired.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.dqn_learner import (CandidateDQNLearner, EpisodeResult,
                                      LearnerConfig, VARIANTS, run_episode)
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.adapter import capture
from ..integrated.fixtures import build_integrated_profile
from ..integrated.qnet import QPreference
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import (TerminalGenParams,
                                       generate_terminal_scenario)
from .direct_job_runner import _git_state, _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-039-candidate-ddqn"
LEVEL = InformationLevel.PRE_ADVICE


class SPTPreference(BaselinePreference):
    """동정보 강 휴리스틱 2안 — 계획 소요시간 최단 우선 (mandatory 는 resolver 가 선치)."""

    def rank(self, sim, crane_id, gc) -> tuple:
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        return (dur,) + super().rank(sim, crane_id, gc)


BASELINES: dict[str, Callable[[], BaselinePreference]] = {
    "BASELINE_VESSEL_WAIT": BaselinePreference,
    "BASELINE_SPT": SPTPreference,
}


@dataclass(frozen=True)
class CandidateDqnConfig:
    train_episodes: int = 500
    validation_episodes: int = 20
    test_episodes: int = 60
    checkpoint_every: int = 25
    variants: tuple[str, ...] = VARIANTS
    train_seed0: int = 300_000
    validation_seed0: int = 310_000
    test_seed0: int = 320_000
    n_external: int = 40
    n_vessels: int = 2
    bootstrap_seed: int = 75_039
    bootstrap_resamples: int = 10_000
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.checkpoint_every) <= 0:
            raise ValueError("all sizes must be positive")
        if any(v not in VARIANTS for v in self.variants) or not self.variants:
            raise ValueError(f"variants must be drawn from {VARIANTS}")
        bands = [set(self.train_seeds), set(self.validation_seeds),
                 set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        legacy_top = 250_000       # 단일야드 전 실험 대역 상한 (10k~240k)
        if any(s < legacy_top for band in bands for s in band):
            raise ValueError("기존 실험 seed 대역(<250k) 재사용 금지 (매핑 §5)")

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


def quick_candidate_dqn_config() -> CandidateDqnConfig:
    return CandidateDqnConfig(train_episodes=6, validation_episodes=2,
                              test_episodes=3, checkpoint_every=3,
                              variants=("ddqn",), n_external=8, n_vessels=1,
                              bootstrap_resamples=200, quick=True)


def _gen_params(cfg: CandidateDqnConfig) -> TerminalGenParams:
    if cfg.quick:
        return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels,
                                 vessel_moves=6, horizon_s=7_200.0,
                                 drain_window_s=3_600.0)
    return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels)


def _sim(profile, seed: int, params) -> TerminalSimulator:
    return TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                             check_invariants=True)


def _eval_policy(profile, params, seeds, *, preference_factory=None,
                 learner: CandidateDQNLearner | None = None) -> list[EpisodeResult]:
    out = []
    for seed in seeds:
        pref = preference_factory() if preference_factory else QPreference()
        out.append(run_episode(_sim(profile, seed, params), level=LEVEL,
                               preference=pref, learner=learner, epsilon=0.0))
    return out


def _dims(profile, params, seed: int) -> tuple[int, int, int, int]:
    sim = _sim(profile, seed, params)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "dims", 0)
    return encoding_dims(encode_observation(state, obs[0]))


def _train_variant(variant: str, profile, params, cfg: CandidateDqnConfig,
                   dims, cost_scale: float, progress
                   ) -> tuple[list[dict], dict, CandidateDQNLearner]:
    learner = CandidateDQNLearner(
        LearnerConfig(variant=variant, cost_scale=cost_scale), dims,
        seed=cfg.train_seed0 + VARIANTS.index(variant))
    explore = random.Random(cfg.train_seed0 + 7 + VARIANTS.index(variant))
    curve: list[dict] = []
    best: tuple[float, int, CandidateDQNLearner] | None = None
    for ep, seed in enumerate(cfg.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        run_episode(_sim(profile, seed, params), level=LEVEL,
                    preference=QPreference(), learner=learner, epsilon=eps,
                    explore_rng=explore, collect=True, learn=True)
        if ep % cfg.checkpoint_every and ep != cfg.train_episodes:
            continue
        snapshot = copy.deepcopy(learner)
        rows = _eval_policy(profile, params, cfg.validation_seeds, learner=snapshot)
        mean = fmean(r.total_cost for r in rows)
        curve.append({"variant": variant, "episode": ep, "val_total_cost": mean,
                      "replay": len(learner.replay),
                      "grad_steps": learner.grad_steps})
        progress(f"[train:{variant}] ep={ep}/{cfg.train_episodes} "
                 f"val_cost={mean:.3f} replay={len(learner.replay)}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snapshot)
    mean, ep, chosen = best
    return curve, {"variant": variant, "episode": ep, "val_total_cost": mean}, chosen


def run_candidate_dqn(out_dir: str = "outputs/reports/candidate_dqn_poc",
                      cfg: CandidateDqnConfig | None = None,
                      progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or CandidateDqnConfig()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-039 run requires a clean committed tree")
    profile = build_integrated_profile()
    params = _gen_params(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dims = _dims(profile, params, cfg.train_seeds[0])
    # 학습 표적 스케일 fit — train band 선두 5일 baseline 결정당 비용 (test 미접촉)
    fit_rows = _eval_policy(profile, params, cfg.train_seeds[:5],
                            preference_factory=BaselinePreference)
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions)
                                 for r in fit_rows))
    progress(f"[YR-039] profile={profile.terminal_id} dims={dims} "
             f"variants={list(cfg.variants)} train={cfg.train_episodes} "
             f"cost_scale={cost_scale:.1f}")

    # ---- baseline 선택 (validation, YR-033: test 미접촉)
    base_val = {name: _eval_policy(profile, params, cfg.validation_seeds,
                                   preference_factory=factory)
                for name, factory in BASELINES.items()}
    baseline_name = min(base_val,
                        key=lambda n: (fmean(r.total_cost for r in base_val[n]), n))
    selections: dict[str, object] = {"_baseline": {
        "policy": baseline_name,
        "validation_total_cost": fmean(r.total_cost for r in base_val[baseline_name])}}
    progress(f"[baseline] {baseline_name} 선택")

    curve: list[dict] = []
    learners: dict[str, CandidateDQNLearner] = {}
    for variant in cfg.variants:
        vcurve, selection, learner = _train_variant(variant, profile, params, cfg,
                                                    dims, cost_scale, progress)
        curve.extend(vcurve)
        selections[f"CandidateDQN[{variant}]"] = selection
        learners[f"CandidateDQN[{variant}]"] = learner
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)

    # ---- locked test
    def rows_of(results: list[EpisodeResult], seeds) -> list[dict]:
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
    for name, learner in learners.items():
        progress(f"[test] {name}")
        test_rows[name] = rows_of(
            _eval_policy(profile, params, cfg.test_seeds, learner=learner),
            cfg.test_seeds)
        learner.save(out / f"model_{name}.pt")
    _json_dump(out / "test_results.json", test_rows)

    base_rows = test_rows[baseline_name]
    seeds = [r["seed"] for r in base_rows]
    paired: dict[str, object] = {}
    for offset, name in enumerate(learners):
        alt = test_rows[name]
        entry: dict[str, object] = {}
        for m_off, (label, key_, direction) in enumerate((
                ("total_cost", "total_cost", MetricDirection.MINIMIZE),
                ("mean_wait", "mean_wait_min", MetricDirection.MINIMIZE),
                ("p95_wait", "p95_wait_min", MetricDirection.MINIMIZE),
                ("vessel_delay", "vessel_delay_min", MetricDirection.MINIMIZE))):
            stats = paired_bootstrap(
                [float(r[key_]) for r in base_rows],
                [float(r[key_]) for r in alt],
                metric=MetricSpec(key_, direction), seeds=seeds,
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

    improved = [n for n in learners
                if float(paired[n]["total_cost"]["difference_ci"]["upper"]) < 0.0]
    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": EXPERIMENT_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"terminal_id": profile.terminal_id,
                        "assumed": profile.assumed},
            "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
            "policy": "Q_cost(global,yc,queue,candidate) via resolver "
                      "QPreference (YR-039 매핑 동결)",
            "elapsed_s": time.time() - started,
        },
        "selections": selections, "paired": paired,
        "variants_improved_vs_baseline": improved,
    }
    _json_dump(out / "candidate_dqn_results.json", payload)
    report = _build_report(payload, curve, out)
    progress(f"[YR-039] completed in {payload['manifest']['elapsed_s']:.1f}s"
             f" -> {report}")
    return report


def _build_report(payload: dict, curve: list[dict], out: Path) -> Path:
    sel, paired = payload["selections"], payload["paired"]
    L: list[str] = []
    L.append("# YR-039 — 동적 후보 Candidate DQN/DDQN/Dueling (통합 터미널)")
    L.append("")
    L.append("> ⚠ 가정 프로파일(POC-MULTI 2-crane) + 합성 시나리오. 1차 지표 = "
             "정규화 누적 터미널 총비용 (YR-038).")
    L.append("")
    L.append(f"- baseline (validation 선택): **{sel['_baseline']['policy']}**")
    L.append(f"- baseline 을 유의하게 이긴 variant: "
             f"{payload['variants_improved_vs_baseline'] or '**없음**'}")
    L.append("")
    L.append("## locked test — paired vs " + str(sel["_baseline"]["policy"]))
    L.append("")
    L.append("| variant | 선택 ep | val_cost | total_cost Δ [95% CI] "
             "| mean 대기 Δ | 본선지연 Δ | p95 대기 Δ% 상한 "
             "| guardrail (P95≤+5%/완료/backlog0/invariant) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, entry in paired.items():
        s = sel.get(name, {})
        mw = entry["total_cost"]["difference"]
        ci = entry["total_cost"]["difference_ci"]
        p95 = entry["p95_wait"].get("percent_change_ci")
        p95_txt = "—" if p95 is None else f"{p95['upper']:+.1f}%"
        g = entry["guardrails"]
        mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
        g_txt = (f"{mark(bool(g['p95_within_5pct']))}/"
                 f"{mark(g['completion_all_100pct'])}/"
                 f"{mark(g['max_backlog'] == 0)}/{mark(g['invariants_all_ok'])}")
        L.append(f"| {name} | {s.get('episode', '—')} "
                 f"| {s.get('val_total_cost', float('nan')):.2f} "
                 f"| {mw:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}] "
                 f"| {entry['mean_wait']['difference']:+.3f} "
                 f"| {entry['vessel_delay']['difference']:+.3f} "
                 f"| {p95_txt} | {g_txt} |")
    L.append("")
    L.append("## checkpoint 곡선 (validation)")
    L.append("")
    L.append("| variant | episode | val_total_cost | replay | grad_steps |")
    L.append("|---|---|---|---|---|")
    step = max(1, len(curve) // 45)
    for row in curve[::step]:
        L.append(f"| {row['variant']} | {row['episode']} "
                 f"| {row['val_total_cost']:.2f} | {row['replay']} "
                 f"| {row['grad_steps']} |")
    L.append("")
    L.append("*생성: yard_rl.experiments.candidate_dqn_experiment — "
             "원자료 candidate_dqn_results.json*")
    path = out / "candidate_dqn_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
