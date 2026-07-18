"""YR-064 — BC(모방) 초기화 + 차분(counterfactual) 신호 미세조정.

배경: YR-062 에서 TD 미세조정은 BC 를 15ep 내 파괴했다 (신용 희석 신호).
YR-063 의 차분 credit 은 scratch 에서 탈퇴화를 일으킨 순위-정렬 신호다.
질문: **차분 신호라면 BC(56.25·swa 0.491)를 보존하면서 개선까지 가는가** —
효율(BC)과 행동 유인(차분)의 결합이 SF_SPT(53.12) 초과의 유력 경로.

알려진 위험 (YR-062 §2 동형): BC 의 Q 는 CE 서수 값, 차분 표적 D 는 비용 차 척도 —
재척도화 충격은 여기도 존재한다. 단 D 는 행동 간 순위와 정렬된 값이라 TD(팀 return
회귀)보다 순위 보존 가능성이 높다는 것이 본 실험의 가설. lr 사다리 완충 + ep0 포함
val-best 선택(미세조정 순손해 시 BC 회귀가 정직한 선택)은 YR-062 와 동일.
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
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import LearnerConfig
from .direct_job_runner import _git_state, _json_dump
from .yr061_reward_redesign import (_agg, _eval, _paired, _params, _report,
                                    _rl_rows, _sim, _swa, Yr061Config,
                                    quick_yr061_config)
from .yr062_bc_finetune import BC_CHECKPOINT, warm_start
from .yr063_diff_credit import run_diff_episode

EXPERIMENT_ID = "YR-064-bc-init-diff-finetune"
LEVEL = InformationLevel.PRE_ADVICE

REUSE_ROWS = {
    "BC": ("outputs/reports/yr061_imitation/test_results.json", "IMITATE"),
    "DIFF_SCRATCH": ("outputs/reports/yr063_diff/test_results.json", "DIFF"),
    "SF_SPT": ("outputs/reports/yr061_imitation/test_results.json", "SF_SPT"),
    "FIFO": ("outputs/reports/yr061_imitation/test_results.json", "FIFO"),
}


@dataclass(frozen=True)
class Yr064Config:
    base: Yr061Config = Yr061Config()
    bc_checkpoint: str = BC_CHECKPOINT
    window_s: float = 600.0                    # YR-063 동결값 승계
    finetune_lrs: tuple[float, ...] = (1e-4, 3e-4)
    epsilon_scale: float = 0.3                 # BC 근방 약한 탐험 (YR-062 동일)
    reuse: bool = True


def quick_yr064_config() -> Yr064Config:
    return Yr064Config(base=quick_yr061_config(), window_s=300.0,
                       finetune_lrs=(1e-4,), reuse=False)


def _load_reused_rows(test_seeds) -> dict:
    out = {}
    for name, (path, key) in REUSE_ROWS.items():
        rows = json.loads(Path(path).read_text(encoding="utf-8"))[key]
        if [r["seed"] for r in rows] != list(test_seeds):
            raise ValueError(f"{name} 재사용 행의 test seed 불일치")
        out[name] = rows
    return out


def _finetune(arm: str, lr: float, cfg: Yr064Config, profile, params, rc, progress):
    base = cfg.base
    learner = warm_start(cfg.bc_checkpoint,
                         LearnerConfig(variant=base.variant, lr=lr, cost_scale=1.0))
    explore = random.Random(64_100)
    curve: list[dict] = []
    snap0 = copy.deepcopy(learner)
    rows0 = _eval(profile, params, base.validation_seeds, snap0)
    mean0, swa0 = fmean(r.total_cost for r in rows0), _swa(rows0)
    curve.append({"arm": arm, "episode": 0, "val_total_cost": mean0,
                  "val_serve_when_available": swa0})
    progress(f"[train:{arm}] ep=0 (BC 원본) val_cost={mean0:.2f} swa={swa0:.2f}")
    best = (mean0, 0, snap0)
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = min(1.0, cfg.epsilon_scale / (ep ** 0.5))
        info = run_diff_episode(_sim(profile, seed, params), learner=learner, rc=rc,
                                window_s=cfg.window_s, epsilon=eps,
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
        if (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    sel = {"arm": arm, "episode": best[1], "val_total_cost": best[0]}
    return curve, sel, best[2]


def run_yr064(out_dir: str = "outputs/reports/yr064_bc_diff",
              cfg: Yr064Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr064Config()
    base = cfg.base
    started = time.time()
    git = _git_state()
    if not base.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-064 run requires a clean committed tree")
    profile = build_integrated_profile()
    params = _params(base)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc = RewardCalculator.assumed_default()
    progress(f"[YR-064] bc={cfg.bc_checkpoint} window={cfg.window_s:g}s")

    curve, selections, results = [], {}, {}
    for lr in cfg.finetune_lrs:
        arm = f"ft{lr:g}"
        acurve, sel, chosen = _finetune(arm, lr, cfg, profile, params, rc, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] {arm} (선택 ep={sel['episode']})")
        results[arm] = _rl_rows(_eval(profile, params, base.test_seeds, chosen),
                                base.test_seeds)
        chosen.save(out / f"model_{arm}.pt")
    if cfg.reuse:
        results.update(_load_reused_rows(base.test_seeds))
        progress("[test] BC/DIFF_SCRATCH/SF_SPT/FIFO 기존 판정 행 재사용")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    if cfg.reuse:
        for t, lr in enumerate(cfg.finetune_lrs, start=1):
            arm = f"ft{lr:g}"
            paired[f"{arm}_vs_BC"] = _paired(results["BC"], results[arm],
                                             base, t * 2)
            paired[f"{arm}_vs_SF_SPT"] = _paired(results["SF_SPT"], results[arm],
                                                 base, t * 2 + 1)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if base.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "BC warm-start + 차분 1-step 미세조정, ep0 포함 val-best",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr064_results.json", payload)
    report = _report(payload, out, name="yr064_report.md",
                     title="YR-064 — BC 초기화 + 차분 미세조정 판정 결과")
    progress(f"[YR-064] 완료 ({payload['manifest']['elapsed_s']:.0f}s) -> {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr064(out_dir=("outputs/reports/yr064_bc_diff_quick" if quick
                       else "outputs/reports/yr064_bc_diff"),
              cfg=quick_yr064_config() if quick else None)
