"""YR-045 — 정정판 locked 재실험 (사전등록 2026-07-16 동결본 집행).

사전등록: .claude/docs/strategy-history/2026-07-16-YR-045-corrected-locked-rerun-prereg.md
- 신규 seed: train 400000~400499 / val 410000~410019 / locked test 420000~420059.
- 3 ETA arm(NO_ETA/ETA_NO_PRE/FULL) × 전략적 WAIT(allow/forbid) 교차.
- 정책: Candidate DQN/DDQN/Dueling(같은 예산) + JointRollout(600s)·Beam·SF-SPT·FIFO.
- 6중 동시 게이트(§6)·보고 의무(§7)·중단 조건(§7).

사전등록이 명시하지 않은 지점의 해석 (최소 변형 원칙, 리포트에 공개):
- 학습은 FULL arm 환경에서 variant 별 1회(500 ep) — §5 의 "variant별 500 episode" 예산을
  준수한다. arm 은 **평가 시** 입력(ETA)·행동(PRE) 제거로 적용 = 배치된 정책의 ablation.
  arm 별 재학습은 예산 3배라 동결 문구와 충돌한다.
- checkpoint 선택은 FULL arm validation 총비용 최저 (§5 문언 그대로).
- SF-SPT·FIFO 는 구조상 전략적 WAIT 불가(WAIT 최하위 선호) → forbid=allow 동일 —
  중복 실행하지 않고 리포트에 동일 표기.
- "비악화" 게이트의 조작적 정의: paired 차이 CI 상한 ≤ max(0, baseline 평균의 +5%)
  (YR-039 하네스 P95 관행 승계). 그 외 게이트는 문언 그대로.

phase 별 산출물을 out_dir 에 저장하고, 존재하면 건너뛴다(다시 실행 = resume).
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, replace as dc_replace
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator
from ..integrated.baselines import (ActionMix, BeamLookahead, FIFOPreference,
                                    JointRolloutGreedy, ResolverPolicy,
                                    ServiceFirstSPTPreference,
                                    assert_healthy_action_mix, run_joint_episode)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig, VARIANTS,
                                      run_episode)
from ..integrated.adapter import capture
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.fixtures import build_integrated_profile
from ..integrated.qnet import QPreference
from ..integrated.resolver import BaselinePreference
from ..integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario
from .direct_job_runner import _git_state, _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap
from .terminal_cost import assert_no_dominance

EXPERIMENT_ID = "YR-045-corrected-locked-rerun"
LEVEL = InformationLevel.PRE_ADVICE
ARMS = ("NO_ETA", "ETA_NO_PRE", "FULL")
WAIT_MODES = ("allow", "forbid")
RC = RewardCalculator.assumed_default()      # §2.5: scale 재적합 금지 — assumed config 동결


@dataclass(frozen=True)
class Yr045Config:
    train_episodes: int = 500
    validation_episodes: int = 20
    test_episodes: int = 60
    checkpoint_every: int = 25
    variants: tuple[str, ...] = VARIANTS
    train_seed0: int = 400_000
    validation_seed0: int = 410_000
    test_seed0: int = 420_000
    n_external: int = 40
    n_vessels: int = 2
    rollout_horizon_s: float = 600.0
    beam_width: int = 3
    bootstrap_seed: int = 75_045
    bootstrap_resamples: int = 10_000
    precheck_train_n: int = 5
    precheck_test_n: int = 5
    quick: bool = False

    def __post_init__(self) -> None:
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.checkpoint_every) <= 0:
            raise ValueError("all sizes must be positive")
        if any(v not in VARIANTS for v in self.variants) or not self.variants:
            raise ValueError(f"variants must be drawn from {VARIANTS}")
        bands = [set(self.train_seeds), set(self.validation_seeds), set(self.test_seeds)]
        if any(a & b for i, a in enumerate(bands) for b in bands[i + 1:]):
            raise ValueError("seed bands must be disjoint")
        # 사전등록 §3: 300000~320059(학습·진단·보정 사용) 및 그 이전 대역 전부 폐기.
        if any(s < 400_000 for band in bands for s in band):
            raise ValueError("YR-045 는 400000 미만 seed 사용 금지 (사전등록 §3)")

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


def quick_yr045_config() -> Yr045Config:
    return Yr045Config(train_episodes=6, validation_episodes=2, test_episodes=3,
                       checkpoint_every=3, variants=("ddqn",), n_external=8,
                       n_vessels=1, bootstrap_resamples=200, precheck_train_n=1,
                       precheck_test_n=1, quick=True)


# ------------------------------------------------------------------- arm 환경
def _gen_params(cfg: Yr045Config) -> TerminalGenParams:
    if cfg.quick:
        return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels,
                                 vessel_moves=6, horizon_s=7_200.0,
                                 drain_window_s=3_600.0)
    return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels)


def scenario_for_arm(profile, seed: int, params, arm: str):
    """arm 별 시나리오 — NO_ETA 는 provided_eta 제거(구조 불변, YR-048 격리 계약)."""
    sc = generate_terminal_scenario(profile, seed, params)
    if arm == "NO_ETA":
        sc = dc_replace(sc, jobs=[dc_replace(j, provided_eta=None) for j in sc.jobs],
                        meta={**sc.meta, "eta_error_s": None, "arm": arm})
    else:
        sc.meta = {**sc.meta, "arm": arm}
    return sc


def generator_for_arm(arm: str) -> CandidateGenerator:
    return CandidateGenerator(block_pre_rehandle=(arm == "ETA_NO_PRE"))


def _sim(profile, sc) -> TerminalSimulator:
    return TerminalSimulator(profile, sc, check_invariants=True, info_level=LEVEL)


# ------------------------------------------------------------------- 공통 행렬
def _row_from_joint(seed: int, r: dict) -> dict:
    m = r["action_mix"]
    return {"seed": seed, "total_cost": r["total_cost"], "n_decisions": r["n_decisions"],
            "completion_rate": r["completion_rate"], "backlog": r["backlog"],
            "mean_wait_min": r["mean_wait_min"], "p95_wait_min": r["p95_wait_min"],
            "vessel_delay_min": r["vessel_delay_min"],
            "sts_wait_s": r["sts_wait_s"], "transfer_wait_s": r["transfer_wait_s"],
            "loaded_travel_m": r["loaded_travel_m"], "empty_travel_m": r["empty_travel_m"],
            "rehandles": r["rehandles"], "combo_truncations": r["combo_truncations"],
            "action_counts": m["counts"], "serve_available": m["serve_available"],
            "serve_taken": m["serve_taken"], "cand_listed": r["cand_listed"],
            "cand_feasible": r["cand_feasible"], "term_contrib": r["term_contrib"],
            "invariants_ok": True}


def _row_from_episode(seed: int, r) -> dict:
    x = r.extras
    return {"seed": seed, "total_cost": r.total_cost, "n_decisions": r.n_decisions,
            "completion_rate": r.completion_rate, "backlog": r.backlog,
            "mean_wait_min": r.mean_wait_min, "p95_wait_min": r.p95_wait_min,
            "vessel_delay_min": r.vessel_delay_min,
            "sts_wait_s": x["sts_wait_s"], "transfer_wait_s": x["transfer_wait_s"],
            "loaded_travel_m": x["loaded_travel_m"], "empty_travel_m": x["empty_travel_m"],
            "rehandles": x["rehandles"], "combo_truncations": 0,
            "action_counts": x["action_counts"],
            "serve_available": x["serve_available"], "serve_taken": x["serve_taken"],
            "cand_listed": x["cand_listed"], "cand_feasible": x["cand_feasible"],
            "term_contrib": x["term_contrib"], "invariants_ok": r.invariants_ok}


def _eval_rl(profile, params, seeds, arm: str, wait_mode: str,
             learner: CandidateDQNLearner | None) -> list[dict]:
    gen = generator_for_arm(arm)
    rows = []
    for seed in seeds:
        sim = _sim(profile, scenario_for_arm(profile, seed, params, arm))
        r = run_episode(sim, level=LEVEL, preference=QPreference(), learner=learner,
                        epsilon=0.0, generator=gen,
                        forbid_strategic_wait=(wait_mode == "forbid"))
        rows.append(_row_from_episode(seed, r))
    return rows


def _eval_pref(profile, params, seeds, arm: str, preference_factory) -> list[dict]:
    """BaselinePreference 계열 (scale fit·SF·FIFO) — run_joint_episode 공통 드라이버."""
    gen = generator_for_arm(arm)
    rows = []
    for seed in seeds:
        sim = _sim(profile, scenario_for_arm(profile, seed, params, arm))
        r = run_joint_episode(sim, ResolverPolicy(preference_factory(), "PREF"), RC,
                              level=LEVEL, generator=gen)
        rows.append(_row_from_joint(seed, r))
    return rows


def _eval_rollout(profile, params, seeds, arm: str, wait_mode: str, cfg: Yr045Config,
                  *, beam: bool) -> list[dict]:
    gen = generator_for_arm(arm)
    rows = []
    for seed in seeds:
        sim = _sim(profile, scenario_for_arm(profile, seed, params, arm))
        if beam:
            pol = BeamLookahead(RC, horizon_s=cfg.rollout_horizon_s, width=cfg.beam_width,
                                generator=gen,
                                forbid_strategic_wait=(wait_mode == "forbid"))
        else:
            pol = JointRolloutGreedy(RC, horizon_s=cfg.rollout_horizon_s, generator=gen,
                                     forbid_strategic_wait=(wait_mode == "forbid"))
        r = run_joint_episode(sim, pol, RC, level=LEVEL, generator=gen)
        rows.append(_row_from_joint(seed, r))
    return rows


# ------------------------------------------------------------------- 학습
def _dims(profile, params) -> tuple[int, int, int, int]:
    sim = _sim(profile, scenario_for_arm(profile, 400_000, params, "FULL"))
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "dims", 0)
    return encoding_dims(encode_observation(state, obs[0]))


def _train_variant(variant: str, profile, params, cfg: Yr045Config, dims,
                   cost_scale: float, progress) -> tuple[list[dict], dict,
                                                         CandidateDQNLearner]:
    learner = CandidateDQNLearner(
        LearnerConfig(variant=variant, cost_scale=cost_scale), dims,
        seed=cfg.train_seed0 + VARIANTS.index(variant))
    explore = random.Random(cfg.train_seed0 + 7 + VARIANTS.index(variant))
    gen = generator_for_arm("FULL")
    curve: list[dict] = []
    best: tuple[float, int, CandidateDQNLearner] | None = None
    for ep, seed in enumerate(cfg.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        run_episode(_sim(profile, scenario_for_arm(profile, seed, params, "FULL")),
                    level=LEVEL, preference=QPreference(), learner=learner,
                    epsilon=eps, explore_rng=explore, generator=gen,
                    collect=True, learn=True)
        if ep % cfg.checkpoint_every and ep != cfg.train_episodes:
            continue
        snapshot = copy.deepcopy(learner)
        rows = _eval_rl(profile, params, cfg.validation_seeds, "FULL", "allow", snapshot)
        mean = fmean(r["total_cost"] for r in rows)
        curve.append({"variant": variant, "episode": ep, "val_total_cost": mean,
                      "replay": len(learner.replay), "grad_steps": learner.grad_steps})
        progress(f"[train:{variant}] ep={ep}/{cfg.train_episodes} "
                 f"val_cost={mean:.3f} replay={len(learner.replay)}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snapshot)
    mean, ep, chosen = best
    return curve, {"variant": variant, "episode": ep, "val_total_cost": mean}, chosen


# ------------------------------------------------------------------- 분석
_GATE_METRICS = (("total_cost", MetricDirection.MINIMIZE),
                 ("mean_wait_min", MetricDirection.MINIMIZE),
                 ("p95_wait_min", MetricDirection.MINIMIZE),
                 ("vessel_delay_min", MetricDirection.MINIMIZE),
                 ("sts_wait_s", MetricDirection.MINIMIZE),
                 ("transfer_wait_s", MetricDirection.MINIMIZE),
                 ("empty_travel_m", MetricDirection.MINIMIZE),
                 ("rehandles", MetricDirection.MINIMIZE))


def _paired(base_rows, alt_rows, cfg: Yr045Config, tag: int) -> dict:
    seeds = [r["seed"] for r in base_rows]
    out = {}
    for i, (key, direction) in enumerate(_GATE_METRICS):
        stats = paired_bootstrap([float(r[key]) for r in base_rows],
                                 [float(r[key]) for r in alt_rows],
                                 metric=MetricSpec(key, direction), seeds=seeds,
                                 seed=cfg.bootstrap_seed + tag * 100 + i,
                                 n_resamples=cfg.bootstrap_resamples)
        out[key] = stats.as_dict()
    return out


def _mix_of(rows) -> ActionMix:
    mix = ActionMix()
    for r in rows:
        for kind, n in r["action_counts"].items():
            mix.counts[kind] = mix.counts.get(kind, 0) + int(n)
        mix.serve_available += int(r["serve_available"])
        mix.serve_taken += int(r["serve_taken"])
    return mix


def _term_shares(rows) -> dict[str, float]:
    tot: dict[str, float] = {}
    for r in rows:
        for k, v in r["term_contrib"].items():
            tot[k] = tot.get(k, 0.0) + float(v)
    s = sum(tot.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in sorted(tot.items())}


def _no_worse(entry: dict, base_mean: float) -> bool:
    """비악화 조작 정의: paired 차이 CI 상한 ≤ max(0, base 평균의 +5%)."""
    ub = float(entry["difference_ci"]["upper"])
    return ub <= max(0.0, 0.05 * abs(base_mean))


def _gates(base_rows, alt_rows, paired: dict) -> dict:
    base_mean = {k: fmean(float(r[k]) for r in base_rows) for k, _ in _GATE_METRICS}
    mix = _mix_of(alt_rows)
    healthy = True
    try:
        assert_healthy_action_mix(mix, label="gate")
    except Exception:
        healthy = False
    shares = _term_shares(alt_rows)
    dom_ok = (max(shares.values()) <= 0.70) if shares else True
    g = {
        "g1_mean_wait_improves": float(paired["mean_wait_min"]["difference_ci"]["upper"]) < 0.0,
        "g2_p95_no_worse": _no_worse(paired["p95_wait_min"], base_mean["p95_wait_min"]),
        "g3_vessel_no_worse": _no_worse(paired["vessel_delay_min"], base_mean["vessel_delay_min"]),
        "g3_sts_no_worse": _no_worse(paired["sts_wait_s"], base_mean["sts_wait_s"]),
        "g3_transfer_no_worse": _no_worse(paired["transfer_wait_s"], base_mean["transfer_wait_s"]),
        "g4_travel_or_rehandle_improves": (
            float(paired["empty_travel_m"]["difference_ci"]["upper"]) < 0.0
            or float(paired["rehandles"]["difference_ci"]["upper"]) < 0.0),
        "g5_completion_backlog_invariants": (
            all(r["completion_rate"] >= 1.0 for r in alt_rows)
            and all(int(r["backlog"]) == 0 for r in alt_rows)
            and all(bool(r["invariants_ok"]) for r in alt_rows)),
        "g6_dominance_le_70pct": dom_ok,
        "g6_action_mix_healthy": healthy,
    }
    g["all_pass"] = all(bool(v) for v in g.values())
    return g


# ------------------------------------------------------------------- 실행
def run_yr045(out_dir: str = "outputs/reports/yr045_locked_rerun",
              cfg: Yr045Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr045Config()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("locked run 은 clean commit 필수 (사전등록 §2.4)")
    profile = build_integrated_profile()
    params = _gen_params(cfg)
    out = Path(out_dir)
    (out / "phase_d").mkdir(parents=True, exist_ok=True)

    # ---- phase A: 선결 — baseline 라이브락·건전성 재확인 (arm별, §2.1)
    pa = out / "phase_a_precheck.json"
    if not pa.exists():
        pre = {}
        seeds = (list(cfg.train_seeds[:cfg.precheck_train_n])
                 + list(cfg.validation_seeds)
                 + list(cfg.test_seeds[:cfg.precheck_test_n]))
        for arm in ARMS:
            rows = _eval_rollout(profile, params, seeds, arm, "allow", cfg, beam=False)
            bad = [r["seed"] for r in rows if r["completion_rate"] < 1.0]
            mix = _mix_of(rows)
            assert_healthy_action_mix(mix, label=f"precheck:{arm}")
            pre[arm] = {"seeds": seeds, "livelock_seeds": bad,
                        "truncations": sum(r["combo_truncations"] for r in rows),
                        "action_mix": mix.as_dict()}
            if bad:
                raise RuntimeError(f"precheck {arm}: 완주 실패 seed {bad} — §2.1 위반, "
                                   "원인 규명 전 locked 실행 금지")
            progress(f"[precheck:{arm}] {len(seeds)} seeds 완주·건전성 통과 "
                     f"(절단 {pre[arm]['truncations']}회)")
        _json_dump(pa, pre)
    else:
        progress("[precheck] 기존 산출물 재사용")

    # ---- phase B: cost scale fit (train 선두 5, FULL) + 지배도 guard
    pb = out / "phase_b_scale.json"
    if pb.exists():
        cost_scale = json.loads(pb.read_text(encoding="utf-8"))["cost_scale"]
        progress(f"[scale] 기존 fit 재사용 cost_scale={cost_scale:.2f}")
    else:
        fit_rows = _eval_pref(profile, params, cfg.train_seeds[:5], "FULL",
                              BaselinePreference)
        cost_scale = max(1e-6, fmean(r["total_cost"] / max(1, r["n_decisions"])
                                     for r in fit_rows))
        shares = _term_shares(fit_rows)
        assert_no_dominance(shares)
        _json_dump(pb, {"cost_scale": cost_scale, "term_shares": shares,
                        "fit_seeds": list(cfg.train_seeds[:5])})
        progress(f"[scale] cost_scale={cost_scale:.2f} 지배도 통과 "
                 f"(최대 항 {max(shares.values()):.1%})")

    # ---- phase C: 학습 (FULL arm) + validation checkpoint 선택
    dims = _dims(profile, params)
    curve_all: list[dict] = []
    selections: dict[str, dict] = {}
    learners: dict[str, CandidateDQNLearner] = {}
    for variant in cfg.variants:
        name = f"DQN[{variant}]"
        model_p = out / f"model_{variant}.pt"
        sel_p = out / f"selection_{variant}.json"
        if model_p.exists() and sel_p.exists():
            learners[name] = CandidateDQNLearner.load(model_p)
            selections[name] = json.loads(sel_p.read_text(encoding="utf-8"))
            progress(f"[train:{variant}] 기존 checkpoint 재사용")
            continue
        curve, sel, chosen = _train_variant(variant, profile, params, cfg, dims,
                                            cost_scale, progress)
        chosen.save(model_p)
        _json_dump(sel_p, sel)
        curve_all.extend(curve)
        selections[name] = sel
        learners[name] = chosen
    if curve_all:
        _json_dump(out / "checkpoint_curve.json", curve_all)

    # ---- phase D: locked test — (정책 × arm × wait_mode) × 60 seeds
    def cond_rows(pol_name: str, arm: str, mode: str, fn) -> list[dict]:
        p = out / "phase_d" / f"{pol_name}__{arm}__{mode}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        t0 = time.time()
        rows = fn()
        _json_dump(p, rows)
        progress(f"[test] {pol_name} {arm}/{mode} 60s… {time.time()-t0:.0f}s "
                 f"cost={fmean(r['total_cost'] for r in rows):.3f}")
        return rows

    results: dict[str, dict] = {}
    ts = cfg.test_seeds
    for arm in ARMS:
        for mode in WAIT_MODES:
            key = f"{arm}/{mode}"
            results.setdefault(key, {})
            results[key]["JOINT_ROLLOUT"] = cond_rows(
                "JOINT_ROLLOUT", arm, mode,
                lambda a=arm, m=mode: _eval_rollout(profile, params, ts, a, m, cfg,
                                                    beam=False))
            results[key]["BEAM"] = cond_rows(
                "BEAM", arm, mode,
                lambda a=arm, m=mode: _eval_rollout(profile, params, ts, a, m, cfg,
                                                    beam=True))
            for name, le in learners.items():
                results[key][name] = cond_rows(
                    name, arm, mode,
                    lambda a=arm, m=mode, L=le: _eval_rl(profile, params, ts, a, m, L))
        # SF·FIFO: 전략적 WAIT 구조상 불가 → allow 만 실행, forbid 는 동일 참조
        for pol_name, factory in (("SF_SPT", ServiceFirstSPTPreference),
                                  ("FIFO", FIFOPreference)):
            rows = cond_rows(pol_name, arm, "allow",
                             lambda a=arm, f=factory: _eval_pref(profile, params, ts,
                                                                 a, f))
            results[f"{arm}/allow"][pol_name] = rows
            results[f"{arm}/forbid"][pol_name] = rows      # 동일 (구조상 전략 WAIT 없음)

    # ---- phase E: 판정 — 게이트(§6) + arm 기여 분리(§4)
    analysis: dict[str, object] = {"gates": {}, "paired_vs_baseline": {},
                                   "arm_contributions": {}, "term_shares": {},
                                   "action_mix": {}}
    tag = 0
    for key, pols in results.items():
        base = pols["JOINT_ROLLOUT"]
        analysis["term_shares"][key] = {n: _term_shares(r) for n, r in pols.items()}
        analysis["action_mix"][key] = {n: _mix_of(r).as_dict() for n, r in pols.items()}
        for name, rows in pols.items():
            if name == "JOINT_ROLLOUT":
                continue
            tag += 1
            pr = _paired(base, rows, cfg, tag)
            analysis["paired_vs_baseline"][f"{key}::{name}"] = pr
            analysis["gates"][f"{key}::{name}"] = _gates(base, rows, pr)
    # ETA 경로 기여: 같은 정책·mode 에서 arm 간 paired (ETA_NO_PRE−NO_ETA=위치선점,
    # FULL−ETA_NO_PRE=선제 재조작). §4: FULL−NO_ETA 단독으로 H2 를 판정하지 않는다.
    for mode in WAIT_MODES:
        for name in list(learners) + ["JOINT_ROLLOUT", "BEAM", "SF_SPT", "FIFO"]:
            for lo, hi, label in (("NO_ETA", "ETA_NO_PRE", "reposition_path"),
                                  ("ETA_NO_PRE", "FULL", "pre_rehandle_path")):
                tag += 1
                pr = _paired(results[f"{lo}/{mode}"][name],
                             results[f"{hi}/{mode}"][name], cfg, tag)
                analysis["arm_contributions"][f"{name}/{mode}/{label}"] = {
                    k: pr[k] for k in ("total_cost", "mean_wait_min")}

    payload = {
        "manifest": {
            "schema_version": 1, "strategy_id": EXPERIMENT_ID,
            "mode": "quick" if cfg.quick else "full",
            "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": {"terminal_id": profile.terminal_id, "assumed": profile.assumed},
            "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
            "interpretations": [
                "학습 FULL arm 1회/variant, arm 은 평가시 ablation",
                "SF/FIFO forbid==allow (구조상 전략 WAIT 불가)",
                "비악화 = paired 차이 CI 상한 <= max(0, base 평균 +5%)"],
            "elapsed_s": time.time() - started},
        "selections": selections, "analysis": analysis,
    }
    _json_dump(out / "yr045_results.json", payload)
    report = _build_report(payload, results, out)
    progress(f"[YR-045] completed in {payload['manifest']['elapsed_s']:.1f}s -> {report}")
    return report


# ------------------------------------------------------------------- 리포트
def compute_condition(out_dir: str, policy: str, arm: str, mode: str,
                      cfg: Yr045Config | None = None,
                      progress: Callable[[str], None] = print) -> Path:
    """locked test 조건 1개만 계산해 phase_d 파일로 저장 — 병렬 실행 단위.

    같은 함수·같은 seed 를 쓰므로 순차 실행(run_yr045)과 결과가 동일하다 (episode 는
    seed 독립·결정론). 오케스트레이션만 병렬화한다. RL 정책은 phase C 모델 필요.
    """
    cfg = cfg or Yr045Config()
    out = Path(out_dir)
    (out / "phase_d").mkdir(parents=True, exist_ok=True)
    p = out / "phase_d" / f"{policy}__{arm}__{mode}.json"
    if p.exists():
        progress(f"[cond] {p.name} 존재 — skip")
        return p
    profile = build_integrated_profile()
    params = _gen_params(cfg)
    ts = cfg.test_seeds
    t0 = time.time()
    if policy == "JOINT_ROLLOUT":
        rows = _eval_rollout(profile, params, ts, arm, mode, cfg, beam=False)
    elif policy == "BEAM":
        rows = _eval_rollout(profile, params, ts, arm, mode, cfg, beam=True)
    elif policy == "SF_SPT":
        rows = _eval_pref(profile, params, ts, arm, ServiceFirstSPTPreference)
    elif policy == "FIFO":
        rows = _eval_pref(profile, params, ts, arm, FIFOPreference)
    elif policy.startswith("DQN["):
        variant = policy[4:-1]
        learner = CandidateDQNLearner.load(out / f"model_{variant}.pt")
        rows = _eval_rl(profile, params, ts, arm, mode, learner)
    else:
        raise ValueError(f"unknown policy {policy}")
    _json_dump(p, rows)
    progress(f"[cond] {p.name} 완료 {time.time()-t0:.0f}s "
             f"cost={fmean(r['total_cost'] for r in rows):.3f}")
    return p


def _fmt_ci(e: dict) -> str:
    ci = e["difference_ci"]
    return f"{e['difference']:+.3f} [{ci['lower']:+.3f}, {ci['upper']:+.3f}]"


def _build_report(payload: dict, results: dict, out: Path) -> Path:
    an = payload["analysis"]
    L: list[str] = []
    L.append("# YR-045 — 정정판 locked 재실험 결과")
    L.append("")
    L.append("> ⚠ 가정 프로파일(POC-MULTI)·합성 시나리오 — 실운영 주장 아님 (주장 게이트: "
             "YR-002/009). 사전등록 동결본 집행, 해석 3건은 manifest 참조.")
    L.append("")
    L.append("## 동시 판정 게이트 (§6) — baseline=JointRollout(같은 arm/mode)")
    L.append("")
    L.append("| 조건::정책 | 대기↓ | P95 | 본선 | STS | 이송 | 이동/재조작↓ | 완주 | 지배도 | 건전성 | **전부** |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    mk = lambda b: "✅" if b else "❌"  # noqa: E731
    for key, g in sorted(an["gates"].items()):
        L.append(f"| {key} | {mk(g['g1_mean_wait_improves'])} | {mk(g['g2_p95_no_worse'])} "
                 f"| {mk(g['g3_vessel_no_worse'])} | {mk(g['g3_sts_no_worse'])} "
                 f"| {mk(g['g3_transfer_no_worse'])} | {mk(g['g4_travel_or_rehandle_improves'])} "
                 f"| {mk(g['g5_completion_backlog_invariants'])} | {mk(g['g6_dominance_le_70pct'])} "
                 f"| {mk(g['g6_action_mix_healthy'])} | {mk(g['all_pass'])} |")
    L.append("")
    L.append("## ETA 경로 기여 분리 (§4) — paired Δ [95% CI]")
    L.append("")
    L.append("| 정책/mode/경로 | total_cost Δ | mean_wait Δ |")
    L.append("|---|---|---|")
    for key, e in sorted(an["arm_contributions"].items()):
        L.append(f"| {key} | {_fmt_ci(e['total_cost'])} | {_fmt_ci(e['mean_wait_min'])} |")
    L.append("")
    L.append("## 총비용 요약 (locked test 60-seed 평균)")
    L.append("")
    conds = sorted(results)
    pols = sorted({n for c in results.values() for n in c})
    L.append("| 정책 | " + " | ".join(conds) + " |")
    L.append("|---|" + "---|" * len(conds))
    for n in pols:
        cells = []
        for c in conds:
            rows = results[c].get(n)
            cells.append(f"{fmean(r['total_cost'] for r in rows):.2f}" if rows else "—")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("*원자료: yr045_results.json·phase_d/ (seed별 행렬·행동분포·항별 기여·절단 횟수)*")
    path = out / "yr045_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


if __name__ == "__main__":                       # pragma: no cover — 실행 진입점
    import argparse

    ap = argparse.ArgumentParser(description="YR-045 locked rerun")
    ap.add_argument("--out", default="outputs/reports/yr045_locked_rerun")
    ap.add_argument("--quick", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("cond", help="locked test 조건 1개 (병렬 worker 단위)")
    c.add_argument("policy")
    c.add_argument("arm", choices=ARMS)
    c.add_argument("mode", choices=WAIT_MODES)
    args = ap.parse_args()
    cfg = quick_yr045_config() if args.quick else None
    if args.cmd == "cond":
        compute_condition(args.out, args.policy, args.arm, args.mode, cfg)
    else:
        run_yr045(args.out, cfg)
