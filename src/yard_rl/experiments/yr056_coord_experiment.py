"""YR-056 — COORD 협조 feature 경량 실험 (itc-v3).

질문: 상대 의도·경합 feature(COORD)만으로 RL 의 interference 격차(YR-054: 격차의 85%)가
줄어드는가? — QMIX(YR-013) 착수 판단 재료.

설계: dueling(YR-045 RL 최선) 2-arm 동일 학습예산 — COORD(on) vs NO_COORD(ablation off,
정보량 = itc-v2 동일) + JointRollout(forbid WAIT — YR-045 최강 조건) 기준선.
FULL ETA(기본 생성기)·전략 WAIT 제외(YR-052 기본). 신규 seed 대역 500k/510k/520k.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..contract.schema import AblationGroup
from ..domain.enums import InformationLevel
from ..integrated import (JointRolloutGreedy, TerminalSimulator,
                          build_integrated_profile, run_joint_episode)
from ..integrated.adapter import capture
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import (CandidateDQNLearner, EpisodeResult,
                                      LearnerConfig, run_episode)
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.qnet import QPreference
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import (TerminalGenParams,
                                       generate_terminal_scenario)
from .candidate_dqn_experiment import _gen_params  # 동일 물량 파라미터 재사용
from .direct_job_runner import _git_state, _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-056-coord-features"
LEVEL = InformationLevel.PRE_ADVICE
ARMS: dict[str, tuple] = {"COORD": (), "NO_COORD": (AblationGroup.COORD,)}


@dataclass(frozen=True)
class Yr056Config:
    train_episodes: int = 500
    validation_episodes: int = 20
    test_episodes: int = 60
    checkpoint_every: int = 25
    variant: str = "dueling"          # YR-045 에서 RL 최선이던 학습기
    train_seed0: int = 500_000
    validation_seed0: int = 510_000
    test_seed0: int = 520_000
    n_external: int = 40
    n_vessels: int = 2
    bootstrap_seed: int = 75_056
    bootstrap_resamples: int = 10_000
    quick: bool = False

    # 소각·기사용 대역: 단일야드(<250k)·YR-039(300k~330k)·YR-045(400k~430k)
    _USED_RANGES = ((0, 250_000), (300_000, 330_000), (400_000, 430_000))

    def __post_init__(self) -> None:
        bands = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        if any(lo <= s < hi for band in bands for s in band
               for lo, hi in self._USED_RANGES):
            raise ValueError("기존 실험 seed 대역 재사용 금지 (<250k·300k대·400k대)")

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


def quick_yr056_config() -> Yr056Config:
    return Yr056Config(train_episodes=6, validation_episodes=2, test_episodes=3,
                       checkpoint_every=3, n_external=8, n_vessels=1,
                       bootstrap_resamples=200, quick=True)


def _sim(profile, seed: int, params) -> TerminalSimulator:
    return TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                             check_invariants=True)


def _eval_rl(profile, params, seeds, learner, ablation_off) -> list[EpisodeResult]:
    return [run_episode(_sim(profile, s, params), level=LEVEL, preference=QPreference(),
                        learner=learner, ablation_off=ablation_off)
            for s in seeds]


def _train_arm(arm: str, profile, params, cfg: Yr056Config, dims, cost_scale,
               progress) -> tuple[list[dict], dict, CandidateDQNLearner]:
    import copy
    import random
    off = ARMS[arm]
    learner = CandidateDQNLearner(LearnerConfig(variant=cfg.variant,
                                                cost_scale=cost_scale), dims,
                                  seed=56_000 + list(ARMS).index(arm))
    explore = random.Random(56_100 + list(ARMS).index(arm))
    curve: list[dict] = []
    best = None
    for ep, seed in enumerate(cfg.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        run_episode(_sim(profile, seed, params), level=LEVEL, preference=QPreference(),
                    learner=learner, epsilon=eps, explore_rng=explore,
                    collect=True, learn=True, ablation_off=off)
        if ep % cfg.checkpoint_every and ep != cfg.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        rows = _eval_rl(profile, params, cfg.validation_seeds, snap, off)
        mean = fmean(r.total_cost for r in rows)
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean})
        progress(f"[train:{arm}] ep={ep}/{cfg.train_episodes} val_cost={mean:.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    mean, ep, chosen = best
    return curve, {"arm": arm, "episode": ep, "val_total_cost": mean}, chosen


def _rl_rows(results: list[EpisodeResult], seeds) -> list[dict]:
    return [{"seed": s, "total_cost": r.total_cost,
             "interference": float(r.extras["term_contrib"].get("interference", 0.0)),
             "wait_actions": int(r.extras["action_counts"].get("WAIT", 0)),
             "mean_wait_min": r.mean_wait_min, "p95_wait_min": r.p95_wait_min,
             "vessel_delay_min": r.vessel_delay_min,
             "completion_rate": r.completion_rate, "backlog": r.backlog}
            for s, r in zip(seeds, results)]


def _jr_rows(profile, params, seeds, rc) -> list[dict]:
    rows = []
    for s in seeds:
        pol = JointRolloutGreedy(rc, forbid_strategic_wait=True)
        r = run_joint_episode(_sim(profile, s, params), pol, rc, level=LEVEL)
        rows.append({"seed": s, "total_cost": r["total_cost"],
                     "interference": float(r["term_contrib"].get("interference", 0.0)),
                     "wait_actions": int(r["action_mix"]["counts"].get("WAIT", 0)),
                     "mean_wait_min": r["mean_wait_min"],
                     "p95_wait_min": r.get("p95_wait_min", 0.0),
                     "vessel_delay_min": r.get("vessel_delay_min", 0.0),
                     "completion_rate": r["completion_rate"], "backlog": r["backlog"]})
    return rows


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


def run_yr056(out_dir: str = "outputs/reports/yr056_coord",
              cfg: Yr056Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr056Config()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-056 run requires a clean committed tree")
    profile = build_integrated_profile()
    params = _gen_params_of(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc = RewardCalculator.assumed_default()

    sim0 = _sim(profile, cfg.train_seeds[0], params)
    sim0.info_level = LEVEL
    dp0 = sim0.run_until_decision()
    state, obs, _g = capture(sim0, dp0.crane_ids, LEVEL, "dims", 0)
    dims = encoding_dims(encode_observation(state, obs[0]))
    # 학습 표적 스케일 — train 선두 5일 baseline (YR-039 관례, test 미접촉)
    fit = [run_episode(_sim(profile, s, params), level=LEVEL,
                       preference=BaselinePreference())
           for s in cfg.train_seeds[:5]]
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions) for r in fit))
    progress(f"[YR-056] dims={dims} cost_scale={cost_scale:.1f} arms={list(ARMS)}")

    curve, selections, results = [], {}, {}
    for arm in ARMS:
        acurve, sel, learner = _train_arm(arm, profile, params, cfg, dims,
                                          cost_scale, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] RL {arm}")
        results[arm] = _rl_rows(
            _eval_rl(profile, params, cfg.test_seeds, learner, ARMS[arm]),
            cfg.test_seeds)
        learner.save(out / f"model_{arm}.pt")
    progress("[test] JointRollout (forbid)")
    results["JOINT_ROLLOUT"] = _jr_rows(profile, params, cfg.test_seeds, rc)
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {
        "COORD_vs_NO_COORD": _paired(results["NO_COORD"], results["COORD"], cfg, 0),
        "COORD_vs_JR": _paired(results["JOINT_ROLLOUT"], results["COORD"], cfg, 1),
        "NO_COORD_vs_JR": _paired(results["JOINT_ROLLOUT"], results["NO_COORD"], cfg, 2),
    }
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if cfg.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "contract": "itc-v3 (COORD on/off arm)",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: {k: fmean(float(r[k]) for r in rows)
                         for k in ("total_cost", "interference", "wait_actions",
                                   "mean_wait_min", "completion_rate")}
                  for name, rows in results.items()},
    }
    _json_dump(out / "yr056_results.json", payload)
    report = _report(payload, out)
    progress(f"[YR-056] 완료 ({payload['manifest']['elapsed_s']:.0f}s) → {report}")
    return report


def _gen_params_of(cfg: Yr056Config) -> TerminalGenParams:
    if cfg.quick:
        return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels,
                                 vessel_moves=6, horizon_s=7_200.0,
                                 drain_window_s=3_600.0)
    return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels)


def _report(payload: dict, out: Path) -> Path:
    m = payload["means"]
    p = payload["paired"]
    lines = ["# YR-056 — COORD 협조 feature 경량 실험 결과", "",
             "> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 질문: 상대 의도·경합 feature 만으로",
             "> RL 의 interference 격차(YR-054: 85%)가 줄어드는가 — QMIX 착수 판단 재료.", ""]
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
    lines.append("*원자료: yr056_results.json · test_results.json (seed별)*")
    path = out / "yr056_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    run_yr056(cfg=quick_yr056_config() if quick else None)
