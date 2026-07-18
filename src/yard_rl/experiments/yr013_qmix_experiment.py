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
    # 학습예산 사다리 (사용자 지시 2026-07-18): 비어있으면 단일 예산.
    # 예: (500, 1000, 2000) — 한 번 2000ep 학습하며 tier 별 val-best checkpoint 를
    # 각각 test 평가 → "구조 무효 vs 학습량 부족" 판별.
    budget_ladder: tuple[int, ...] = ()

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
        if self.budget_ladder:
            if list(self.budget_ladder) != sorted(set(self.budget_ladder)):
                raise ValueError("budget_ladder 는 오름차순 유일값")
            if self.budget_ladder[-1] != self.train_episodes:
                raise ValueError("budget_ladder 최댓값 == train_episodes 여야 함")

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
    """공통 학습 루프 — QMIX 는 joint replay(absorb_joint), INDEP 는 per-crane collect.

    budget_ladder 지원: 한 번 학습하며 tier(≤ep)별 val-best snapshot 을 따로 유지 —
    같은 학습 궤적에서 예산만 다른 checkpoint 를 얻는다 (재학습 없음·재현 동일).
    반환 chosen: {tier: snapshot} (단일 예산이면 {train_episodes: snap}).
    """
    explore = random.Random(13_100 + (0 if arm == "QMIX" else 1))
    tiers = tuple(cfg.budget_ladder) or (cfg.train_episodes,)
    curve: list[dict] = []
    best: dict[int, tuple] = {}
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
        snap.replay.clear()          # 평가엔 불필요 — tier 다중 보관 메모리 절약
        mean = fmean(r.total_cost for r in
                     _eval(profile, params, cfg.validation_seeds, snap))
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean,
                      "replay": len(learner.replay)})
        progress(f"[train:{arm}] ep={ep}/{cfg.train_episodes} val_cost={mean:.2f}")
        for tier in tiers:
            if ep <= tier and (tier not in best
                               or (mean, ep) < (best[tier][0], best[tier][1])):
                best[tier] = (mean, ep, snap)
    selections = {tier: {"arm": arm, "episode": best[tier][1],
                         "val_total_cost": best[tier][0]} for tier in tiers}
    chosen = {tier: best[tier][2] for tier in tiers}
    return curve, selections, chosen


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
              progress: Callable[[str], None] = print,
              reuse_jr: str | None = None) -> Path:
    """reuse_jr: 같은 test 대역의 선행 run test_results.json — JR(결정적·비학습) 행 재사용."""
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
    tiers = tuple(cfg.budget_ladder) or (cfg.train_episodes,)

    def _name(arm: str, tier: int) -> str:
        return arm if len(tiers) == 1 else f"{arm}@{tier}"

    curve, selections, results = [], {}, {}
    for arm, learner in learners.items():
        acurve, sels, chosen = _train(arm, learner, profile, params, cfg, progress)
        curve.extend(acurve)
        for tier in tiers:
            name = _name(arm, tier)
            selections[name] = sels[tier]
            progress(f"[test] RL {name}")
            results[name] = _rl_rows(
                _eval(profile, params, cfg.test_seeds, chosen[tier]), cfg.test_seeds)
            chosen[tier].save(out / f"model_{name}.pt")
    if reuse_jr:
        import json
        prior = json.loads(Path(reuse_jr).read_text(encoding="utf-8"))
        rows = prior["JOINT_ROLLOUT"]
        if [r["seed"] for r in rows] != list(cfg.test_seeds):
            raise ValueError("reuse_jr 의 test seed 가 현재 config 와 불일치")
        results["JOINT_ROLLOUT"] = rows
        progress(f"[test] JointRollout — 선행 run 재사용 ({reuse_jr})")
    else:
        progress("[test] JointRollout (forbid)")
        results["JOINT_ROLLOUT"] = _jr_rows(profile, params, cfg.test_seeds, rc)
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    for t, tier in enumerate(tiers):
        qn, dn = _name("QMIX", tier), _name("INDEP", tier)
        paired[f"{qn}_vs_{dn}"] = _paired(results[dn], results[qn], cfg, t * 3)
        paired[f"{qn}_vs_JR"] = _paired(results["JOINT_ROLLOUT"], results[qn],
                                        cfg, t * 3 + 1)
        paired[f"{dn}_vs_JR"] = _paired(results["JOINT_ROLLOUT"], results[dn],
                                        cfg, t * 3 + 2)
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
    argv = sys.argv[1:]
    reuse = None
    if "--reuse-jr" in argv:
        reuse = argv[argv.index("--reuse-jr") + 1]
    if "--quick" in argv:
        cfg = quick_yr013_config()
    elif "--ladder" in argv:
        # 예산 사다리 (사용자 지시): 2000ep 1회 학습, tier 별 checkpoint 판정
        cfg = Yr013Config(train_episodes=2000, budget_ladder=(500, 1000, 2000))
    else:
        cfg = None
    out = ("outputs/reports/yr013_qmix_ladder" if "--ladder" in argv
           else "outputs/reports/yr013_qmix")
    run_yr013(out_dir=out, cfg=cfg, reuse_jr=reuse)
