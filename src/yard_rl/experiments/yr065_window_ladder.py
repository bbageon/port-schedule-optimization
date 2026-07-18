"""YR-065 — 차분 신호 개량: 비교 창(window) 사다리 1200/2400s.

배경 (YR-063): 600s 차분 귀속은 탈퇴화(swa 0.322)에 성공했으나 성능은 전 비교군
열세(85.58) — "일은 하나 순서를 모름". 유력 원인 1순위가 **window 근시**(600s 밖
파급 무시)였다. 창을 20/40분으로 늘려 이 가설을 검정한다.

- arm = window ∈ {1200s, 2400s}. 600s 는 YR-063 결과 행 재사용 (결정론 실증 관례).
- 나머지 전부 YR-063 동결값 그대로 (ddqn·lr 1e-3·ε=1/√ep·seed 63,000/63,100·
  150ep·1-step·WAIT 앵커·SF_SPT base_policy) — 차이는 창 길이뿐.
- 비용 주의: rollout 시간이 창에 비례 — 2400s arm 은 600s 의 ~4배 소요.
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
from ..integrated.dqn_learner import CandidateDQNLearner, LearnerConfig
from ..integrated.encoding import encode_observation, encoding_dims
from .direct_job_runner import _git_state, _json_dump
from .yr061_reward_redesign import (_agg, _eval, _paired, _params, _report,
                                    _rl_rows, _sim, _swa, Yr061Config,
                                    quick_yr061_config)
from .yr063_diff_credit import run_diff_episode

EXPERIMENT_ID = "YR-065-diff-window-ladder"
LEVEL = InformationLevel.PRE_ADVICE

REUSE_ROWS = {
    "DIFF600": ("outputs/reports/yr063_diff/test_results.json", "DIFF"),
    "CONTROL_TD": ("outputs/reports/yr061_reward/test_results.json", "pen0"),
    "BC": ("outputs/reports/yr061_imitation/test_results.json", "IMITATE"),
    "SF_SPT": ("outputs/reports/yr061_imitation/test_results.json", "SF_SPT"),
    "FIFO": ("outputs/reports/yr061_imitation/test_results.json", "FIFO"),
}


@dataclass(frozen=True)
class Yr065Config:
    base: Yr061Config = Yr061Config()
    windows: tuple[float, ...] = (1_200.0, 2_400.0)
    reuse: bool = True


def quick_yr065_config() -> Yr065Config:
    return Yr065Config(base=quick_yr061_config(), windows=(300.0,), reuse=False)


def _load_reused_rows(test_seeds) -> dict:
    out = {}
    for name, (path, key) in REUSE_ROWS.items():
        rows = json.loads(Path(path).read_text(encoding="utf-8"))[key]
        if [r["seed"] for r in rows] != list(test_seeds):
            raise ValueError(f"{name} 재사용 행의 test seed 불일치")
        out[name] = rows
    return out


def _train_window(arm: str, window_s: float, dims, base: Yr061Config,
                  profile, params, rc, progress):
    """YR-063 학습 루프와 동일 — 차이는 window 뿐 (동일 seed 로 paired 성격 유지)."""
    learner = CandidateDQNLearner(
        LearnerConfig(variant=base.variant, lr=base.lr, cost_scale=1.0),
        dims, seed=63_000)
    explore = random.Random(63_100)
    curve: list[dict] = []
    best: tuple | None = None
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        info = run_diff_episode(_sim(profile, seed, params), learner=learner, rc=rc,
                                window_s=window_s, epsilon=eps,
                                explore_rng=explore, learn=True)
        if ep % base.checkpoint_every and ep != base.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval(profile, params, base.validation_seeds, snap)
        mean, swa = fmean(r.total_cost for r in rows), _swa(rows)
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean,
                      "val_serve_when_available": swa,
                      "credit_mean": info["credit_mean"]})
        progress(f"[train:{arm}] ep={ep}/{base.train_episodes} val_cost={mean:.2f} "
                 f"swa={swa:.2f} D_mean={info['credit_mean']:+.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    sel = {"arm": arm, "episode": best[1], "val_total_cost": best[0]}
    return curve, sel, best[2]


def run_yr065(out_dir: str = "outputs/reports/yr065_window",
              cfg: Yr065Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr065Config()
    base = cfg.base
    started = time.time()
    git = _git_state()
    if not base.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-065 run requires a clean committed tree")
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
    progress(f"[YR-065] dims={dims} windows={cfg.windows}")

    curve, selections, results = [], {}, {}
    for w in cfg.windows:
        arm = f"DIFF{int(w)}"
        acurve, sel, chosen = _train_window(arm, w, dims, base, profile, params,
                                            rc, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] {arm} (선택 ep={sel['episode']})")
        results[arm] = _rl_rows(_eval(profile, params, base.test_seeds, chosen),
                                base.test_seeds)
        chosen.save(out / f"model_{arm}.pt")
    if cfg.reuse:
        results.update(_load_reused_rows(base.test_seeds))
        progress("[test] DIFF600/CONTROL_TD/BC/SF_SPT/FIFO 기존 판정 행 재사용")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    if cfg.reuse:
        for t, w in enumerate(cfg.windows, start=1):
            arm = f"DIFF{int(w)}"
            paired[f"{arm}_vs_DIFF600"] = _paired(results["DIFF600"], results[arm],
                                                  base, t * 2)
            paired[f"{arm}_vs_SF_SPT"] = _paired(results["SF_SPT"], results[arm],
                                                 base, t * 2 + 1)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if base.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "차분 window 사다리 — 600s 는 YR-063 행 재사용",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr065_results.json", payload)
    report = _report(payload, out, name="yr065_report.md",
                     title="YR-065 — 차분 window 사다리 판정 결과")
    progress(f"[YR-065] 완료 ({payload['manifest']['elapsed_s']:.0f}s) -> {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr065(out_dir=("outputs/reports/yr065_window_quick" if quick
                       else "outputs/reports/yr065_window"),
              cfg=quick_yr065_config() if quick else None)
