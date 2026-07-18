"""YR-062 — BC(모방) 초기화 + RL(TD) 미세조정: 성능 경로 겸 신용 희석 2차 검정.

배경 (YR-061 종결, prereg §6 결과): 같은 망·인코딩이 지도신호로는 SF_SPT 근접
(56.25·swa 0.491)을 배우나 TD 는 즉시 퇴화(70.11·swa 0.094) — 병목 = TD 신용 희석.

질문 2개를 한 실험으로 검정한다:
- **성능**: BC 시작점에서 TD 미세조정이 BC 를 넘어 SF_SPT(53.12) 초과까지 가는가.
- **진단**: 미세조정이 BC 를 되망가뜨리면(퇴화 재발) 신용 희석의 2차 확증이다.

알려진 위험 (사전 명시): BC 의 Q 는 cross-entropy 로 학습된 **서수(순위) 값**이라
비용 척도가 아니다. TD 는 Q 를 비용 척도로 재회귀시키므로 초기 대규모 손실이 순위
정보를 파괴할 수 있다("재척도화 파국"). lr 사다리(1e-4/3e-4/1e-3)가 이를 완충하며,
lr 무관 전멸이면 재척도화 자체가 원인일 수 있어 별도 해석(§판정)한다.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

import torch

from ..domain.enums import InformationLevel
from ..integrated import build_integrated_profile
from ..integrated.adapter import capture
from ..integrated.baselines import (FIFOPreference, ServiceFirstSPTPreference)
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig,
                                      run_episode)
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.qnet import QPreference
from ..integrated.resolver import BaselinePreference
from .direct_job_runner import _git_state, _json_dump
from .yr061_reward_redesign import (_agg, _baseline_rows, _eval, _paired,
                                    _params, _report, _rl_rows, _sim, _swa,
                                    Yr061Config, quick_yr061_config)

EXPERIMENT_ID = "YR-062-bc-init-td-finetune"
LEVEL = InformationLevel.PRE_ADVICE
BC_CHECKPOINT = "outputs/reports/yr061_imitation/model_IMITATE.pt"


@dataclass(frozen=True)
class Yr062Config:
    base: Yr061Config = Yr061Config()          # seed 대역·시나리오·판정 기계 승계
    bc_checkpoint: str = BC_CHECKPOINT
    finetune_lrs: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    epsilon_scale: float = 0.3                 # ε = 0.3/√ep — BC 근방 약한 탐험
    cost_scale_fit_episodes: int = 5


def quick_yr062_config() -> Yr062Config:
    return Yr062Config(base=quick_yr061_config(), finetune_lrs=(1e-4,))


def warm_start(path: str, cfg: LearnerConfig) -> CandidateDQNLearner:
    """BC checkpoint 의 가중치로 online·target 을 초기화한 새 학습기.

    optimizer 는 새로 만든다(cfg.lr) — CE 단계의 Adam 모멘트를 승계하지 않는다.
    target=online=BC 동기 상태에서 시작 (첫 sync 전 표적 안정).
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "yard-rl-candidate-dqn-v1":
        raise ValueError("unsupported candidate DQN format")
    learner = CandidateDQNLearner(cfg, tuple(payload["dims"]))
    learner.online.load_state_dict(payload["online"])
    learner.target.load_state_dict(payload["online"])
    return learner


def _finetune(arm: str, lr: float, cost_scale: float, cfg: Yr062Config,
              profile, params, progress):
    base = cfg.base
    learner = warm_start(cfg.bc_checkpoint,
                         LearnerConfig(variant=base.variant, lr=lr,
                                       cost_scale=cost_scale))
    explore = random.Random(62_100)
    curve: list[dict] = []
    best: tuple | None = None
    # ep0 = BC 원본 평가를 선택 후보에 포함 — 미세조정이 순손해면 BC 로 회귀하는 것이
    # 정직한 선택이고, 그 자체가 판정 데이터다.
    snap0 = copy.deepcopy(learner)
    rows0 = _eval(profile, params, base.validation_seeds, snap0)
    mean0, swa0 = fmean(r.total_cost for r in rows0), _swa(rows0)
    curve.append({"arm": arm, "episode": 0, "val_total_cost": mean0,
                  "val_serve_when_available": swa0})
    progress(f"[train:{arm}] ep=0 (BC 원본) val_cost={mean0:.2f} swa={swa0:.2f}")
    best = (mean0, 0, snap0)
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = min(1.0, cfg.epsilon_scale / (ep ** 0.5))
        run_episode(_sim(profile, seed, params), level=LEVEL, preference=QPreference(),
                    learner=learner, epsilon=eps, explore_rng=explore,
                    collect=True, learn=True)
        if ep % base.checkpoint_every and ep != base.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval(profile, params, base.validation_seeds, snap)
        mean, swa = fmean(r.total_cost for r in rows), _swa(rows)
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean,
                      "val_serve_when_available": swa})
        progress(f"[train:{arm}] ep={ep}/{base.train_episodes} "
                 f"val_cost={mean:.2f} swa={swa:.2f}")
        if (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    sel = {"arm": arm, "episode": best[1], "val_total_cost": best[0]}
    return curve, sel, best[2]


def run_yr062(out_dir: str = "outputs/reports/yr062_finetune",
              cfg: Yr062Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr062Config()
    base = cfg.base
    started = time.time()
    git = _git_state()
    if not base.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-062 run requires a clean committed tree")
    profile = build_integrated_profile()
    params = _params(base)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc = RewardCalculator.assumed_default()

    fit = [run_episode(_sim(profile, s, params), level=LEVEL,
                       preference=BaselinePreference())
           for s in base.train_seeds[:cfg.cost_scale_fit_episodes]]
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions) for r in fit))
    progress(f"[YR-062] bc={cfg.bc_checkpoint} cost_scale={cost_scale:.2f}")

    curve, selections, results = [], {}, {}
    # BC 동결 arm (control) — checkpoint 그대로 test 평가 (YR-061 phase-3 재현 겸 무결성)
    bc = warm_start(cfg.bc_checkpoint, LearnerConfig(variant=base.variant))
    results["BC"] = _rl_rows(_eval(profile, params, base.test_seeds, bc),
                             base.test_seeds)
    selections["BC"] = {"arm": "BC", "episode": 0, "val_total_cost": None}
    for lr in cfg.finetune_lrs:
        arm = f"ft{lr:g}"
        acurve, sel, chosen = _finetune(arm, lr, cost_scale, cfg, profile,
                                        params, progress)
        curve.extend(acurve)
        selections[arm] = sel
        progress(f"[test] {arm} (선택 ep={sel['episode']})")
        results[arm] = _rl_rows(_eval(profile, params, base.test_seeds, chosen),
                                base.test_seeds)
        chosen.save(out / f"model_{arm}.pt")
    progress("[test] baselines (SF_SPT·FIFO)")
    results["SF_SPT"] = _baseline_rows(profile, params, base.test_seeds, rc,
                                       ServiceFirstSPTPreference(), "SF_SPT")
    results["FIFO"] = _baseline_rows(profile, params, base.test_seeds, rc,
                                     FIFOPreference(), "FIFO")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    for t, lr in enumerate(cfg.finetune_lrs, start=1):
        arm = f"ft{lr:g}"
        paired[f"{arm}_vs_BC"] = _paired(results["BC"], results[arm], base, t * 2)
        paired[f"{arm}_vs_SF_SPT"] = _paired(results["SF_SPT"], results[arm], base,
                                             t * 2 + 1)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if base.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "BC warm-start TD 미세조정 — ep0(BC) 포함 val-best 선택",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr062_results.json", payload)
    report = _report(payload, out, name="yr062_report.md",
                     title="YR-062 — BC 초기화 + TD 미세조정 판정 결과")
    progress(f"[YR-062] 완료 ({payload['manifest']['elapsed_s']:.0f}s) → {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr062(out_dir=("outputs/reports/yr062_finetune_quick" if quick
                       else "outputs/reports/yr062_finetune"),
              cfg=quick_yr062_config() if quick else None)
