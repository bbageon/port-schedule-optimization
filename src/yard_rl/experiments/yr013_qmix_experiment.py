"""YR-013 — QMIX vs 독립 학습자 판정 실험 (매핑 §4).

질문: 학습 구조만 CTDE(mixer)로 바꾸면 — 정보(COORD on)·예산·실행경로 동일 —
interference 격차가 줄어드는가. 기준선 JointRollout(forbid) 동반 보고.
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
from ..integrated import (JointRolloutGreedy, TerminalSimulator,
                          build_integrated_profile, run_joint_episode)
from ..integrated.adapter import capture
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig,
                                      run_episode)
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.qmix import QmixConfig, QmixLearner
from ..integrated.qnet import QPreference
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import (TerminalGenParams,
                                       generate_terminal_scenario)
from .direct_job_runner import _git_state, _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap
from .yr056_coord_experiment import _rl_rows, _jr_rows

EXPERIMENT_ID = "YR-013-qmix-vs-independent"
LEVEL = InformationLevel.PRE_ADVICE


@dataclass(frozen=True)
class Yr013Config:
    train_episodes: int = 500
    validation_episodes: int = 20
    test_episodes: int = 60
    checkpoint_every: int = 25
    variant: str = "dueling"
    train_seed0: int = 530_000
    validation_seed0: int = 540_000
    test_seed0: int = 550_000
    n_external: int = 40
    n_vessels: int = 2
    bootstrap_seed: int = 75_013
    bootstrap_resamples: int = 10_000
    quick: bool = False

    # 소각·기사용 대역: 단일야드(<250k)·YR-039(300k대)·YR-045(400k대)·YR-056(500k~530k)
    _USED_RANGES = ((0, 250_000), (300_000, 330_000), (400_000, 430_000),
                    (500_000, 530_000))

    def __post_init__(self) -> None:
        bands = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        if any(lo <= s < hi for band in bands for s in band
               for lo, hi in self._USED_RANGES):
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


def quick_yr013_config() -> Yr013Config:
    return Yr013Config(train_episodes=6, validation_episodes=2, test_episodes=3,
                       checkpoint_every=3, n_external=8, n_vessels=1,
                       bootstrap_resamples=200, quick=True)


def _params(cfg: Yr013Config) -> TerminalGenParams:
    if cfg.quick:
        return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels,
                                 vessel_moves=6, horizon_s=7_200.0,
                                 drain_window_s=3_600.0)
    return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels)


def _sim(profile, seed, params) -> TerminalSimulator:
    return TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                             check_invariants=True)


def _eval(profile, params, seeds, learner) -> list:
    return [run_episode(_sim(profile, s, params), level=LEVEL,
                        preference=QPreference(), learner=learner) for s in seeds]


def _train(arm: str, learner, profile, params, cfg, progress):
    """공통 학습 루프 — QMIX 는 joint replay(absorb_joint), INDEP 는 per-crane collect."""
    explore = random.Random(13_100 + (0 if arm == "QMIX" else 1))
    curve, best = [], None
    is_qmix = isinstance(learner, QmixLearner)
    for ep, seed in enumerate(cfg.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        sink: dict | None = {} if is_qmix else None
        run_episode(_sim(profile, seed, params), level=LEVEL, preference=QPreference(),
                    learner=learner, epsilon=eps, explore_rng=explore,
                    collect=not is_qmix, learn=True, joint_sink=sink)
        if is_qmix:
            learner.absorb_joint(sink)
        if ep % cfg.checkpoint_every and ep != cfg.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        mean = fmean(r.total_cost for r in
                     _eval(profile, params, cfg.validation_seeds, snap))
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean,
                      "replay": len(learner.replay)})
        progress(f"[train:{arm}] ep={ep}/{cfg.train_episodes} val_cost={mean:.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    mean, ep, chosen = best
    return curve, {"arm": arm, "episode": ep, "val_total_cost": mean}, chosen


def _paired(base_rows, alt_rows, cfg, tag: int) -> dict:
    seeds = [r["seed"] for r in base_rows]
    out = {}
    for i, key_ in enumerate(("total_cost", "interference", "mean_wait_min",
                              "p95_wait_min")):
        out[key_] = paired_bootstrap(
            [float(r[key_]) for r in base_rows], [float(r[key_]) for r in alt_rows],
            metric=MetricSpec(key_, MetricDirection.MINIMIZE), seeds=seeds,
            seed=cfg.bootstrap_seed + tag * 10 + i,
            n_resamples=cfg.bootstrap_resamples).as_dict()
    return out


def run_yr013(out_dir: str = "outputs/reports/yr013_qmix",
              cfg: Yr013Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr013Config()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-013 run requires a clean committed tree")
    profile = build_integrated_profile()
    params = _params(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc = RewardCalculator.assumed_default()

    sim0 = _sim(profile, cfg.train_seeds[0], params)
    sim0.info_level = LEVEL
    dp0 = sim0.run_until_decision()
    state, obs, _g = capture(sim0, dp0.crane_ids, LEVEL, "dims", 0)
    dims = encoding_dims(encode_observation(state, obs[0]))
    fit = [run_episode(_sim(profile, s, params), level=LEVEL,
                       preference=BaselinePreference())
           for s in cfg.train_seeds[:5]]
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions) for r in fit))
    n_agents = len(profile.cranes) if hasattr(profile, "cranes") else 2
    progress(f"[YR-013] dims={dims} cost_scale={cost_scale:.1f} n_agents={n_agents}")

    learners = {
        "QMIX": QmixLearner(QmixConfig(variant=cfg.variant, n_agents=n_agents,
                                       cost_scale=cost_scale), dims, seed=13_000),
        "INDEP": CandidateDQNLearner(LearnerConfig(variant=cfg.variant,
                                                   cost_scale=cost_scale),
                                     dims, seed=13_000),
    }
    curve, selections, results = [], {}, {}
    for arm, learner in learners.items():
        acurve, sel, chosen = _train(arm, learner, profile, params, cfg, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] RL {arm}")
        results[arm] = _rl_rows(_eval(profile, params, cfg.test_seeds, chosen),
                                cfg.test_seeds)
        chosen.save(out / f"model_{arm}.pt")
    progress("[test] JointRollout (forbid)")
    results["JOINT_ROLLOUT"] = _jr_rows(profile, params, cfg.test_seeds, rc)
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {
        "QMIX_vs_INDEP": _paired(results["INDEP"], results["QMIX"], cfg, 0),
        "QMIX_vs_JR": _paired(results["JOINT_ROLLOUT"], results["QMIX"], cfg, 1),
        "INDEP_vs_JR": _paired(results["JOINT_ROLLOUT"], results["INDEP"], cfg, 2),
    }
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if cfg.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "contract": "itc-v3 (COORD on 양 arm — 차이는 학습 구조만)",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: {k: fmean(float(r[k]) for r in rows)
                         for k in ("total_cost", "interference", "wait_actions",
                                   "mean_wait_min", "completion_rate")}
                  for name, rows in results.items()},
    }
    _json_dump(out / "yr013_results.json", payload)
    report = _report(payload, out)
    progress(f"[YR-013] 완료 ({payload['manifest']['elapsed_s']:.0f}s) → {report}")
    return report


def _report(payload: dict, out: Path) -> Path:
    m, p = payload["means"], payload["paired"]
    lines = ["# YR-013 — QMIX vs 독립 학습자 판정 결과", "",
             "> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 정보(COORD)·예산·실행경로 동일 —",
             "> 차이는 학습 구조(CTDE mixer)뿐. 질문: 협조를 '학습 구조'로 가르치면 격차가 주는가.", ""]
    lines.append("| 정책 | total_cost | interference | WAIT 수 | mean_wait(분) | 완료율 |")
    lines.append("|---|---|---|---|---|---|")
    for name, v in m.items():
        lines.append(f"| {name} | {v['total_cost']:.2f} | {v['interference']:.2f} "
                     f"| {v['wait_actions']:.1f} | {v['mean_wait_min']:.2f} "
                     f"| {v['completion_rate']:.3f} |")
    lines.append("")
    for tag, d in p.items():
        tc, itf = d["total_cost"], d["interference"]
        lines.append(f"- **{tag}**: Δtotal={tc['difference']:+.2f} "
                     f"[{tc['difference_ci']['lower']:+.2f}, {tc['difference_ci']['upper']:+.2f}] · "
                     f"Δinterference={itf['difference']:+.2f} "
                     f"[{itf['difference_ci']['lower']:+.2f}, {itf['difference_ci']['upper']:+.2f}]")
    lines.append("")
    lines.append("*원자료: yr013_results.json · test_results.json (seed별)*")
    path = out / "yr013_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    run_yr013(cfg=quick_yr013_config() if "--quick" in sys.argv else None)
