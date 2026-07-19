"""YR-067 — 상태 정규화 × 단일 DQN 트랙 결합 (BC 제외 — 사용자 지시 2026-07-19).

배경: YR-059(병행)가 큰 시나리오(외부 40) INDEP 에서 state_norm 유의 개선(−4.4~−5.6,
TD-RL 최고 80.51)을 실증했으나, 이 트랙(YR-061~065, 외부 16 진단 시나리오)은 전부
정규화 없이 돌았다. 질문: **정규화만으로** 이 트랙의 결론(퇴화·차분 순위)이 바뀌는가.

arm 2개 (BC 는 제외 — 사용자 지시):
- TD_NORM   : scratch TD (YR-061 pen0 프로토콜 그대로) + state_norm — vs CONTROL_TD(70.11).
  주: YR-061 11조건의 A2.state_norm 은 임시 프로토타입(악화 +7.5)이었다 — 본 실험이
  YR-059 정식 구현(P90 fit·itc-v4)으로 그 판정을 교체 검증한다.
- DIFF2400_NORM: 차분 최고 구성(YR-065 창 2400s) + state_norm — vs DIFF2400(78.91).
norm fit 은 YR-059 기계(fit_state_norm, baseline P90) 를 이 시나리오·train 대역에 재적합
(val/test 미접촉)·박제. 평가 지표·비교군·seed 는 YR-061 동결 그대로.
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..domain.enums import InformationLevel
from ..integrated import build_integrated_profile
from ..integrated.adapter import capture
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig,
                                      run_episode)
from ..integrated.encoding import StateNorm, encode_observation, encoding_dims
from ..integrated.resolver import BaselinePreference
from ..integrated.qnet import QPreference
from .direct_job_runner import _git_state, _json_dump
from .yr059_state_norm import fit_state_norm
from .yr061_reward_redesign import (_agg, _paired, _params, _report, _rl_rows,
                                    _sim, _swa, Yr061Config,
                                    quick_yr061_config)
from .yr063_diff_credit import run_diff_episode

EXPERIMENT_ID = "YR-067-state-norm-apply"
LEVEL = InformationLevel.PRE_ADVICE

REUSE_ROWS = {
    "CONTROL_TD": ("outputs/reports/yr061_reward/test_results.json", "pen0"),
    "DIFF2400": ("outputs/reports/yr065_window/test_results.json", "DIFF2400"),
    "SF_SPT": ("outputs/reports/yr061_imitation/test_results.json", "SF_SPT"),
    "FIFO": ("outputs/reports/yr061_imitation/test_results.json", "FIFO"),
}


@dataclass(frozen=True)
class Yr067Config:
    base: Yr061Config = Yr061Config()
    window_s: float = 2_400.0                  # 차분 arm — YR-065 승자 창
    run_td: bool = True
    run_diff: bool = True
    fit_seeds_n: int = 5
    reuse: bool = True


def quick_yr067_config() -> Yr067Config:
    return Yr067Config(base=quick_yr061_config(), window_s=300.0,
                       run_diff=True, reuse=False)


def _load_reused_rows(test_seeds) -> dict:
    out = {}
    for name, (path, key) in REUSE_ROWS.items():
        rows = json.loads(Path(path).read_text(encoding="utf-8"))[key]
        if [r["seed"] for r in rows] != list(test_seeds):
            raise ValueError(f"{name} 재사용 행의 test seed 불일치")
        out[name] = rows
    return out


def _eval_norm(profile, params, seeds, learner, norm) -> list:
    return [run_episode(_sim(profile, s, params), level=LEVEL,
                        preference=QPreference(), learner=learner,
                        state_norm=norm) for s in seeds]


def _train_td(norm, cost_scale, dims, base: Yr061Config, profile, params, progress):
    """YR-061 pen0 프로토콜 그대로 (seed 61,000·ε=1/√ep·150ep) — 차이는 norm 뿐."""
    learner = CandidateDQNLearner(
        LearnerConfig(variant=base.variant, cost_scale=cost_scale, lr=base.lr),
        dims, seed=61_000)
    explore = random.Random(61_100)
    curve, best = [], None
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        run_episode(_sim(profile, seed, params), level=LEVEL, preference=QPreference(),
                    learner=learner, epsilon=eps, explore_rng=explore,
                    collect=True, learn=True, state_norm=norm)
        if ep % base.checkpoint_every and ep != base.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval_norm(profile, params, base.validation_seeds, snap, norm)
        mean, swa = fmean(r.total_cost for r in rows), _swa(rows)
        curve.append({"arm": "TD_NORM", "episode": ep, "val_total_cost": mean,
                      "val_serve_when_available": swa})
        progress(f"[train:TD_NORM] ep={ep}/{base.train_episodes} "
                 f"val_cost={mean:.2f} swa={swa:.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    return curve, {"arm": "TD_NORM", "episode": best[1],
                   "val_total_cost": best[0]}, best[2]


def _train_diff(norm, window_s, dims, base: Yr061Config, profile, params, rc,
                progress):
    """YR-065 승자 구성 그대로 (seed 63,000·창 2400s) — 차이는 norm 뿐."""
    learner = CandidateDQNLearner(
        LearnerConfig(variant=base.variant, lr=base.lr, cost_scale=1.0),
        dims, seed=63_000)
    explore = random.Random(63_100)
    arm = f"DIFF{int(window_s)}_NORM"
    curve, best = [], None
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        info = run_diff_episode(_sim(profile, seed, params), learner=learner, rc=rc,
                                window_s=window_s, epsilon=eps,
                                explore_rng=explore, learn=True, state_norm=norm)
        if ep % base.checkpoint_every and ep != base.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval_norm(profile, params, base.validation_seeds, snap, norm)
        mean, swa = fmean(r.total_cost for r in rows), _swa(rows)
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean,
                      "val_serve_when_available": swa,
                      "credit_mean": info["credit_mean"]})
        progress(f"[train:{arm}] ep={ep}/{base.train_episodes} "
                 f"val_cost={mean:.2f} swa={swa:.2f} "
                 f"D_mean={info['credit_mean']:+.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    return curve, {"arm": arm, "episode": best[1],
                   "val_total_cost": best[0]}, best[2], arm


def run_yr067(out_dir: str = "outputs/reports/yr067_norm",
              cfg: Yr067Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr067Config()
    base = cfg.base
    started = time.time()
    git = _git_state()
    if not base.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-067 run requires a clean committed tree")
    profile = build_integrated_profile()
    params = _params(base)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc = RewardCalculator.assumed_default()

    sim0 = _sim(profile, base.train_seeds[0], params)
    sim0.info_level = LEVEL
    dp0 = sim0.run_until_decision()
    state, obs, _g = capture(sim0, dp0.crane_ids, LEVEL, "dims", 0)
    dims = encoding_dims(encode_observation(state, obs[0]))
    # norm fit — 이 시나리오·train 대역 재적합 (YR-059 기계, val/test 미접촉) + 박제
    norm, detail = fit_state_norm(profile, params,
                                  base.train_seeds[:cfg.fit_seeds_n],
                                  progress=progress)
    _json_dump(out / "state_norm.json",
               {"refs": norm.refs, "clip": norm.clip, "basis": norm.basis,
                "fit_seeds": list(base.train_seeds[:cfg.fit_seeds_n]),
                "detail": detail})
    fit = [run_episode(_sim(profile, s, params), level=LEVEL,
                       preference=BaselinePreference())
           for s in base.train_seeds[:5]]
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions) for r in fit))
    progress(f"[YR-067] dims={dims} cost_scale={cost_scale:.2f} "
             f"norm_refs={len(norm.refs)}")

    curve, selections, results = [], {}, {}
    if cfg.run_td:
        acurve, sel, chosen = _train_td(norm, cost_scale, dims, base, profile,
                                        params, progress)
        curve.extend(acurve)
        selections["TD_NORM"] = sel
        progress(f"[test] TD_NORM (선택 ep={sel['episode']})")
        results["TD_NORM"] = _rl_rows(
            _eval_norm(profile, params, base.test_seeds, chosen, norm),
            base.test_seeds)
        chosen.save(out / "model_TD_NORM.pt")
    if cfg.run_diff:
        acurve, sel, chosen, arm = _train_diff(norm, cfg.window_s, dims, base,
                                               profile, params, rc, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] {arm} (선택 ep={sel['episode']})")
        results[arm] = _rl_rows(
            _eval_norm(profile, params, base.test_seeds, chosen, norm),
            base.test_seeds)
        chosen.save(out / f"model_{arm}.pt")
    if cfg.reuse:
        results.update(_load_reused_rows(base.test_seeds))
        progress("[test] CONTROL_TD/DIFF2400/SF_SPT/FIFO 기존 판정 행 재사용")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    if cfg.reuse:
        pairs = []
        if cfg.run_td:
            pairs.append(("TD_NORM", "CONTROL_TD"))
            pairs.append(("TD_NORM", "SF_SPT"))
        if cfg.run_diff:
            pairs.append((f"DIFF{int(cfg.window_s)}_NORM", "DIFF2400"))
            pairs.append((f"DIFF{int(cfg.window_s)}_NORM", "SF_SPT"))
        for t, (arm, ref) in enumerate(pairs, start=1):
            paired[f"{arm}_vs_{ref}"] = _paired(results[ref], results[arm], base, t)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if base.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "state_norm 결합 (BC 제외 — 사용자 지시). fit=P90 재적합",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr067_results.json", payload)
    report = _report(payload, out, name="yr067_report.md",
                     title="YR-067 — 상태 정규화 결합 판정 결과 (BC 제외)")
    progress(f"[YR-067] 완료 ({payload['manifest']['elapsed_s']:.0f}s) -> {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr067(out_dir=("outputs/reports/yr067_norm_quick" if quick
                       else "outputs/reports/yr067_norm"),
              cfg=quick_yr067_config() if quick else None)
