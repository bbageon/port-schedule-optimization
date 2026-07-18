"""YR-061 — 단일 DQN 보상 재설계: 미완료 잔존 페널티 판정 실험.

배경 (2026-07-18 단일 DQN 11조건 진단, strategy-history 박제): 학습부·정규화·후보부
어느 knob 으로도 SERVE 붕괴(0.04~0.09)·REPOSITION 퇴화를 못 막았고, 원인은 학습신호
(총비용)가 미서비스·미완료를 과소처벌하는 것("일 안 해도 낮게")으로 특정됐다.

질문: 학습 표적에만 미완료 잔존 페널티(`LearnerConfig.unserved_terminal_cost`)를 넣으면
— 평가 지표·게이트·비용 ledger 는 그대로 — 퇴화가 사라지고 SERVE·완료율이 회복되는가.

- arm = 페널티 크기 사다리 (0.0 = control, 기존 거동 재현).
- checkpoint 선택은 **학습 목적함수와 동일한 penalized val 비용** (control 은 둘이 같음)
  — "게이밍된 지표로 게이밍된 정책을 뽑는" 순환을 선택 단계에서도 끊는다.
- 판정 지표는 실제(비페널티) total_cost + 행동분포 건전성(YR-044 계약) + 완료율.
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
from ..integrated import TerminalSimulator, build_integrated_profile, run_joint_episode
from ..integrated.adapter import capture
from ..integrated.baselines import (FIFOPreference, ResolverPolicy,
                                    ServiceFirstSPTPreference)
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import (CandidateDQNLearner, EpisodeResult,
                                      LearnerConfig, run_episode)
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.qnet import QPreference
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import (TerminalGenParams,
                                       generate_terminal_scenario)
from .direct_job_runner import _git_state, _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap

EXPERIMENT_ID = "YR-061-reward-unserved-penalty"
LEVEL = InformationLevel.PRE_ADVICE


@dataclass(frozen=True)
class Yr061Config:
    train_episodes: int = 150
    validation_episodes: int = 8
    test_episodes: int = 20
    checkpoint_every: int = 15
    variant: str = "ddqn"              # 진단 판정: ddqn≈dueling 최선, 기본 유지
    train_seed0: int = 600_000
    validation_seed0: int = 610_000
    test_seed0: int = 620_000
    n_external: int = 16               # 진단과 동일한 소형 시나리오 (빠른 판정 루프)
    n_vessels: int = 2
    lr: float = 1e-3
    penalties: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)   # 0.0 = control
    # 2차 축 (phase-1 판정 후 등록): 비어있지 않으면 penalties 대신 γ 사다리를 arm 으로.
    # phase-1 실측 — 미완료가 구조적으로 0 이라 페널티 무발동, 퇴화는 '방치'가 아니라
    # '지연'. 용의자 = 할인 근시(γ^(Δt/60): 1h 후 비용 ×0.046 ≈ 소멸) vs 비할인 평가.
    gammas: tuple[float, ...] = ()                          # 첫 값 = control(0.95)
    # 3차 축 (phase-2 판정 후 등록): SF_SPT 모방 지도학습 — 표현력 vs TD 신호 이분 검정.
    imitation_epochs: int = 30
    imitation_checkpoint_every: int = 5
    bootstrap_seed: int = 75_061
    bootstrap_resamples: int = 10_000
    quick: bool = False

    # 소각·기사용 대역: 단일야드(<250k)·YR-039(300k대)·YR-045(400k대)·YR-056/013(500k~560k)
    _USED_RANGES = ((0, 250_000), (300_000, 330_000), (400_000, 430_000),
                    (500_000, 560_000))

    def __post_init__(self) -> None:
        bands = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        if any(lo <= s < hi for band in bands for s in band
               for lo, hi in self._USED_RANGES):
            raise ValueError("기존 실험 seed 대역 재사용 금지")
        if not self.penalties or self.penalties[0] != 0.0:
            raise ValueError("penalties[0] 은 control(0.0) 이어야 함")
        if list(self.penalties) != sorted(set(self.penalties)):
            raise ValueError("penalties 는 오름차순 유일값")
        if self.gammas:
            if self.gammas[0] != 0.95:
                raise ValueError("gammas[0] 은 control(0.95 — 기존 기본값) 이어야 함")
            if list(self.gammas) != sorted(set(self.gammas)):
                raise ValueError("gammas 는 오름차순 유일값")
            if any(not 0.0 < g <= 1.0 for g in self.gammas):
                raise ValueError("gamma ∈ (0, 1]")

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


def quick_yr061_config() -> Yr061Config:
    return Yr061Config(train_episodes=6, validation_episodes=2, test_episodes=3,
                       checkpoint_every=3, n_external=8, n_vessels=1,
                       penalties=(0.0, 5.0), bootstrap_resamples=200, quick=True)


def _params(cfg: Yr061Config) -> TerminalGenParams:
    if cfg.quick:
        return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels,
                                 vessel_moves=6, horizon_s=7_200.0,
                                 drain_window_s=3_600.0)
    return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels)


def _sim(profile, seed, params) -> TerminalSimulator:
    return TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                             check_invariants=True)


def _eval(profile, params, seeds, learner) -> list[EpisodeResult]:
    return [run_episode(_sim(profile, s, params), level=LEVEL,
                        preference=QPreference(), learner=learner) for s in seeds]


def _objective(rows: list[EpisodeResult], penalty: float) -> float:
    """학습 목적함수와 동형의 선택 기준 — 실제비용 + 페널티×미완료 (control 은 실제비용)."""
    return fmean(r.total_cost + penalty * r.backlog for r in rows)


def _train(arm: str, overrides: dict, cost_scale: float, dims, profile, params,
           cfg: Yr061Config, progress):
    """공통 학습 루프 — 전 arm 동일 초기화 seed·동일 train seed 열 (paired)."""
    penalty = overrides.get("unserved_terminal_cost", 0.0)
    learner = CandidateDQNLearner(
        LearnerConfig(variant=cfg.variant, cost_scale=cost_scale, lr=cfg.lr,
                      **overrides), dims, seed=61_000)
    explore = random.Random(61_100)
    curve: list[dict] = []
    best: tuple | None = None
    for ep, seed in enumerate(cfg.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        run_episode(_sim(profile, seed, params), level=LEVEL, preference=QPreference(),
                    learner=learner, epsilon=eps, explore_rng=explore,
                    collect=True, learn=True)
        if ep % cfg.checkpoint_every and ep != cfg.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval(profile, params, cfg.validation_seeds, snap)
        obj = _objective(rows, penalty)
        raw_mean = fmean(r.total_cost for r in rows)
        swa = _swa(rows)
        curve.append({"arm": arm, "episode": ep, "val_objective": obj,
                      "val_total_cost": raw_mean, "val_serve_when_available": swa})
        progress(f"[train:{arm}] ep={ep}/{cfg.train_episodes} "
                 f"val_obj={obj:.2f} val_cost={raw_mean:.2f} swa={swa:.2f}")
        if best is None or (obj, ep) < (best[0], best[1]):
            best = (obj, ep, snap)
    sel = {"arm": arm, "episode": best[1], "val_objective": best[0]}
    return curve, sel, best[2]


def _swa(rows: list[EpisodeResult]) -> float:
    avail = sum(r.extras["serve_available"] for r in rows)
    taken = sum(r.extras["serve_taken"] for r in rows)
    return taken / avail if avail else 1.0


def _rl_rows(results: list[EpisodeResult], seeds) -> list[dict]:
    rows = []
    for s, r in zip(seeds, results):
        total_actions = max(1, sum(r.extras["action_counts"].values()))
        rows.append({
            "seed": s, "total_cost": r.total_cost,
            "completion_rate": r.completion_rate, "backlog": r.backlog,
            "mean_wait_min": r.mean_wait_min, "p95_wait_min": r.p95_wait_min,
            "interference": float(r.extras["term_contrib"].get("interference", 0.0)),
            "action_counts": r.extras["action_counts"],
            "serve_share": r.extras["action_counts"].get("SERVE", 0) / total_actions,
            "serve_available": r.extras["serve_available"],
            "serve_taken": r.extras["serve_taken"],
        })
    return rows


def _baseline_rows(profile, params, seeds, rc, preference, name) -> list[dict]:
    rows = []
    for s in seeds:
        r = run_joint_episode(_sim(profile, s, params),
                              ResolverPolicy(preference, name), rc, level=LEVEL)
        rows.append({"seed": s, "total_cost": r["total_cost"],
                     "completion_rate": r["completion_rate"], "backlog": r["backlog"],
                     "mean_wait_min": r["mean_wait_min"],
                     "p95_wait_min": r["p95_wait_min"],
                     "interference": float(r["term_contrib"].get("interference", 0.0)),
                     "action_counts": r["action_mix"]["counts"],
                     "serve_share": r["action_mix"]["shares"].get("SERVE", 0.0),
                     "serve_available": r["action_mix"]["serve_available"],
                     "serve_taken": r["action_mix"]["serve_taken"]})
    return rows


def _paired(base_rows, alt_rows, cfg: Yr061Config, tag: int) -> dict:
    seeds = [r["seed"] for r in base_rows]
    out = {}
    for i, key_ in enumerate(("total_cost", "mean_wait_min", "p95_wait_min")):
        out[key_] = paired_bootstrap(
            [float(r[key_]) for r in base_rows], [float(r[key_]) for r in alt_rows],
            metric=MetricSpec(key_, MetricDirection.MINIMIZE), seeds=seeds,
            seed=cfg.bootstrap_seed + tag * 10 + i,
            n_resamples=cfg.bootstrap_resamples).as_dict()
    return out


def _agg(rows: list[dict]) -> dict:
    avail = sum(r["serve_available"] for r in rows)
    taken = sum(r["serve_taken"] for r in rows)
    return {"total_cost": fmean(r["total_cost"] for r in rows),
            "completion_rate": fmean(r["completion_rate"] for r in rows),
            "backlog": fmean(r["backlog"] for r in rows),
            "mean_wait_min": fmean(r["mean_wait_min"] for r in rows),
            "serve_share": fmean(r["serve_share"] for r in rows),
            "serve_when_available": taken / avail if avail else 1.0}


def run_yr061(out_dir: str = "outputs/reports/yr061_reward",
              cfg: Yr061Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr061Config()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-061 run requires a clean committed tree")
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
    progress(f"[YR-061] dims={dims} cost_scale={cost_scale:.2f}")

    if cfg.gammas:
        arm_specs = [(f"g{g:g}", {"gamma": g}) for g in cfg.gammas]
    else:
        arm_specs = [(f"pen{p:g}", {"unserved_terminal_cost": p})
                     for p in cfg.penalties]
    curve, selections, results = [], {}, {}
    for arm, overrides in arm_specs:
        acurve, sel, chosen = _train(arm, overrides, cost_scale, dims, profile,
                                     params, cfg, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] {arm}")
        results[arm] = _rl_rows(_eval(profile, params, cfg.test_seeds, chosen),
                                cfg.test_seeds)
        chosen.save(out / f"model_{arm}.pt")
    progress("[test] baselines (SF_SPT·FIFO)")
    results["SF_SPT"] = _baseline_rows(profile, params, cfg.test_seeds, rc,
                                       ServiceFirstSPTPreference(), "SF_SPT")
    results["FIFO"] = _baseline_rows(profile, params, cfg.test_seeds, rc,
                                     FIFOPreference(), "FIFO")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    control = arm_specs[0][0]
    paired = {}
    for t, (arm, _o) in enumerate(arm_specs[1:], start=1):
        paired[f"{arm}_vs_{control}"] = _paired(results[control], results[arm], cfg, t * 2)
        paired[f"{arm}_vs_SF_SPT"] = _paired(results["SF_SPT"], results[arm], cfg,
                                             t * 2 + 1)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if cfg.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "페널티는 학습 표적 전용 — test total_cost 는 실제(비페널티) 비용",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr061_results.json", payload)
    report = _report(payload, out)
    progress(f"[YR-061] 완료 ({payload['manifest']['elapsed_s']:.0f}s) → {report}")
    return report


def run_yr061_imitation(out_dir: str = "outputs/reports/yr061_imitation",
                        cfg: Yr061Config | None = None,
                        progress: Callable[[str], None] = print) -> Path:
    """phase 3 (prereg §6) — SF_SPT 모방 이분 검정.

    같은 Q-망·인코딩이 SF_SPT 선택을 지도학습으로 배우면 표현력은 충분(병목=TD 신호),
    못 배우면 인코딩/후보 feature 결함. −Q 를 logit 으로 cross-entropy → argmin Q = 모방.
    """
    import torch
    from torch import nn
    cfg = cfg or Yr061Config()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-061 imitation run requires a clean committed tree")
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

    data: list[tuple] = []                     # (enc, pos) — SF_SPT 의 실제 선택
    for s in cfg.train_seeds:
        sink: dict = {}
        run_episode(_sim(profile, s, params), level=LEVEL,
                    preference=ServiceFirstSPTPreference(), joint_sink=sink)
        for _k, recs in sink.get("events", []):
            for _cid, enc, pos in recs:
                if pos is not None:
                    data.append((enc, pos))
    progress(f"[YR-061i] dims={dims} demos={len(data)} "
             f"({len(cfg.train_seeds)} episodes)")

    learner = CandidateDQNLearner(LearnerConfig(variant=cfg.variant, lr=cfg.lr),
                                  dims, seed=61_200)
    rng = random.Random(61_200)
    idx = list(range(len(data)))
    curve: list[dict] = []
    best: tuple | None = None
    for epoch in range(1, cfg.imitation_epochs + 1):
        rng.shuffle(idx)
        losses = []
        for lo in range(0, len(idx), 128):
            batch = [data[i] for i in idx[lo:lo + 128]]
            g, yc, qs, cand, sel = learner._tensors([b[0] for b in batch])
            logits = (-learner.online(g, yc, qs, cand, sel)).masked_fill(
                ~sel, float("-inf"))
            tgt = torch.tensor([b[1] for b in batch], device=learner.device)
            loss = nn.functional.cross_entropy(logits, tgt)
            learner.opt.zero_grad()
            loss.backward()
            learner.opt.step()
            losses.append(float(loss.detach()))
        if epoch % cfg.imitation_checkpoint_every and epoch != cfg.imitation_epochs:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval(profile, params, cfg.validation_seeds, snap)
        mean = fmean(r.total_cost for r in rows)
        swa = _swa(rows)
        curve.append({"arm": "IMITATE", "episode": epoch, "val_total_cost": mean,
                      "val_serve_when_available": swa, "ce_loss": fmean(losses)})
        progress(f"[imitate] epoch={epoch}/{cfg.imitation_epochs} "
                 f"ce={fmean(losses):.3f} val_cost={mean:.2f} swa={swa:.2f}")
        if best is None or (mean, epoch) < (best[0], best[1]):
            best = (mean, epoch, snap)
    selections = {"IMITATE": {"arm": "IMITATE", "episode": best[1],
                              "val_total_cost": best[0]}}
    chosen = best[2]
    progress("[test] IMITATE")
    results = {"IMITATE": _rl_rows(_eval(profile, params, cfg.test_seeds, chosen),
                                   cfg.test_seeds)}
    chosen.save(out / "model_IMITATE.pt")
    progress("[test] baselines (SF_SPT·FIFO)")
    results["SF_SPT"] = _baseline_rows(profile, params, cfg.test_seeds, rc,
                                       ServiceFirstSPTPreference(), "SF_SPT")
    results["FIFO"] = _baseline_rows(profile, params, cfg.test_seeds, rc,
                                     FIFOPreference(), "FIFO")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)
    paired = {"IMITATE_vs_SF_SPT": _paired(results["SF_SPT"], results["IMITATE"],
                                           cfg, 2)}
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID + "-imitation",
                     "mode": "quick" if cfg.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "prereg §6 — 모방 이분 검정 (표현력 vs TD 신호)",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr061_results.json", payload)
    report = _report(payload, out)
    progress(f"[YR-061i] 완료 ({payload['manifest']['elapsed_s']:.0f}s) → {report}")
    return report


def _report(payload: dict, out: Path) -> Path:
    m, p = payload["means"], payload["paired"]
    lines = ["# YR-061 — 미완료 잔존 페널티 판정 결과", "",
             "> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 페널티는 학습 표적 전용 —",
             "> 아래 total_cost 는 전 arm 실제(비페널티) 비용. 판정 축: 퇴화 해소 여부",
             "> (serve_when_available ≥ 0.25 — YR-044 건전성 계약) + 완료율 + 실제비용.", ""]
    lines.append("| arm | total_cost | 완료율 | backlog | serve_share | serve_when_avail | mean_wait(분) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, v in m.items():
        lines.append(f"| {name} | {v['total_cost']:.2f} | {v['completion_rate']:.3f} "
                     f"| {v['backlog']:.1f} | {v['serve_share']:.3f} "
                     f"| {v['serve_when_available']:.3f} | {v['mean_wait_min']:.2f} |")
    lines.append("")
    for tag, d in p.items():
        tc = d["total_cost"]
        lines.append(f"- **{tag}**: Δtotal={tc['difference']:+.2f} "
                     f"[{tc['difference_ci']['lower']:+.2f}, "
                     f"{tc['difference_ci']['upper']:+.2f}]")
    lines.append("")
    lines.append("*원자료: yr061_results.json · test_results.json (seed별)*")
    path = out / "yr061_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if "--imitate" in argv:
        # 3차 (prereg §6): SF_SPT 모방 이분 검정 — 표현력 vs TD 신호.
        cfg = quick_yr061_config() if "--quick" in argv else None
        out = ("outputs/reports/yr061_imitation_quick" if "--quick" in argv
               else "outputs/reports/yr061_imitation")
        run_yr061_imitation(out_dir=out, cfg=cfg)
    else:
        if "--quick" in argv:
            cfg, out = quick_yr061_config(), "outputs/reports/yr061_reward_quick"
        elif "--gamma" in argv:
            # 2차: 할인 근시 검정 — γ 사다리 (ref_s 60 고정, 페널티 0).
            cfg = Yr061Config(gammas=(0.95, 0.99, 0.999, 1.0))
            out = "outputs/reports/yr061_gamma"
        else:
            cfg, out = None, "outputs/reports/yr061_reward"
        run_yr061(out_dir=out, cfg=cfg)
