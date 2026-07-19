"""YR-068 — 차분 표적 QMIX 본 시나리오 확전 (prereg 2026-07-19).

승자 처방(YR-013c: D 앵커 + mixer 보정, 창 2400s·λ=1·1-step) **재조정 없이**
규모 축만 변경: 외부 16 진단 → 외부 40 본 시나리오 (YR-045/059 규격),
test 550000~550059 — yr059_qmix_norm_ladder 와 동일 대역 (paired 재사용).
G1: vs INDEP@2000(norm, 80.51) / G2: vs JR(68.26) / G3: vs SF_SPT(신규 평가).
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
from ..integrated.baselines import ServiceFirstSPTPreference
from ..integrated.cost_config import RewardCalculator
from ..integrated.encoding import encoding_dims, encode_observation
from ..integrated.qmix import DiffQmixConfig, DiffQmixLearner
from ..integrated.scenario_gen import TerminalGenParams
from .direct_job_runner import _git_state, _json_dump
from .yr013_diff_qmix import run_diff_qmix_episode
from .yr059_state_norm import fit_state_norm
from .yr061_reward_redesign import (_agg, _baseline_rows, _paired, _rl_rows,
                                    _sim, _swa)
from .yr067_norm_apply import _eval_norm

EXPERIMENT_ID = "YR-068-scale-diff-qmix"
LEVEL = InformationLevel.PRE_ADVICE

REUSE_PATH = "outputs/reports/yr059_qmix_norm_ladder/test_results.json"
REUSE_KEYS = ("INDEP@2000", "JOINT_ROLLOUT", "QMIX@2000")


@dataclass(frozen=True)
class Yr068Config:
    train_episodes: int = 150
    validation_episodes: int = 8
    test_episodes: int = 60
    checkpoint_every: int = 15
    variant: str = "ddqn"
    lr: float = 1e-3
    window_s: float = 2_400.0
    lambda_mix: float = 1.0
    train_seed0: int = 530_000          # ladder 학습 대역 재사용 (test 대역만 신성)
    validation_seed0: int = 540_000
    test_seed0: int = 550_000
    n_external: int = 40
    n_vessels: int = 2
    fit_seeds_n: int = 5
    reuse: bool = True
    bootstrap_seed: int = 75_168
    bootstrap_resamples: int = 10_000
    quick: bool = False

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


def quick_yr068_config() -> Yr068Config:
    return Yr068Config(train_episodes=6, validation_episodes=2, test_episodes=3,
                       checkpoint_every=3, window_s=300.0, n_external=8,
                       n_vessels=1, reuse=False, bootstrap_resamples=200,
                       quick=True)


def _params(cfg: Yr068Config) -> TerminalGenParams:
    if cfg.quick:
        return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels,
                                 vessel_moves=6, horizon_s=7_200.0,
                                 drain_window_s=3_600.0)
    return TerminalGenParams(n_external=cfg.n_external, n_vessels=cfg.n_vessels)


def _agg_any(rows: list[dict]) -> dict:
    """트랙 간 행 포맷 관용 집계 — yr061 형(serve_available 보유)은 _agg,
    yr013/yr059 형(미보유)은 공통 키만 평균 (1차 실행 실측 크래시 정정)."""
    if rows and "serve_available" in rows[0]:
        return _agg(rows)
    keys = ("total_cost", "interference", "mean_wait_min", "p95_wait_min",
            "completion_rate", "backlog", "wait_actions")
    return {k: fmean(float(r[k]) for r in rows) for k in keys if k in rows[0]}


def _report_yr068(payload: dict, out: Path) -> Path:
    """트랙 간 키 차이에 관용적인 자체 리포트 (yr061._report 는 serve_share 요구)."""
    m, p = payload["means"], payload["paired"]
    lines = ["# YR-068 — 차분 표적 QMIX 본 시나리오 확전 판정", "",
             "> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009). 승자 처방(YR-013c) 무조정 확전 —",
             "> G1: vs INDEP@2000(norm) / G2: vs JOINT_ROLLOUT / G3: vs SF_SPT.", ""]
    keys = ("total_cost", "mean_wait_min", "p95_wait_min", "completion_rate", "backlog")
    lines.append("| arm | " + " | ".join(keys) + " |")
    lines.append("|" + "---|" * (len(keys) + 1))
    for name, v in m.items():
        lines.append("| " + name + " | "
                     + " | ".join(f"{v[k]:.3f}" if k in v else "—" for k in keys) + " |")
    lines.append("")
    for tag, d in p.items():
        tc = d["total_cost"]
        lines.append(f"- **{tag}**: Δtotal={tc['difference']:+.2f} "
                     f"[{tc['difference_ci']['lower']:+.2f}, {tc['difference_ci']['upper']:+.2f}]")
    lines.append("")
    lines.append("*원자료: yr068_results.json · test_results.json (seed별)*")
    path = out / "yr068_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_reused_rows(test_seeds) -> dict:
    data = json.loads(Path(REUSE_PATH).read_text(encoding="utf-8"))
    out = {}
    for key in REUSE_KEYS:
        rows = data[key]
        if [r["seed"] for r in rows] != list(test_seeds):
            raise ValueError(f"{key} 재사용 행의 test seed 불일치")
        out[key] = rows
    return out


def run_yr068(out_dir: str = "outputs/reports/yr068_scale_diff_qmix",
              cfg: Yr068Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr068Config()
    started = time.time()
    git = _git_state()
    if not cfg.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-068 run requires a clean committed tree")
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
    norm, detail = fit_state_norm(profile, params,
                                  cfg.train_seeds[:cfg.fit_seeds_n],
                                  progress=progress)
    _json_dump(out / "state_norm.json",
               {"refs": norm.refs, "clip": norm.clip, "basis": norm.basis,
                "fit_seeds": list(cfg.train_seeds[:cfg.fit_seeds_n]),
                "detail": detail})
    n_agents = 2
    learner = DiffQmixLearner(
        DiffQmixConfig(variant=cfg.variant, n_agents=n_agents, lr=cfg.lr,
                       lambda_mix=cfg.lambda_mix), dims, seed=63_000)
    explore = random.Random(63_100)
    arm = "DIFF_QMIX"
    progress(f"[YR-068] dims={dims} window={cfg.window_s:g}s 외부={cfg.n_external} "
             f"lambda_mix={cfg.lambda_mix:g}")

    curve, best = [], None
    for ep, seed in enumerate(cfg.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        info = run_diff_qmix_episode(_sim(profile, seed, params), learner=learner,
                                     rc=rc, window_s=cfg.window_s, epsilon=eps,
                                     explore_rng=explore, learn=True,
                                     state_norm=norm)
        if ep % cfg.checkpoint_every and ep != cfg.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval_norm(profile, params, cfg.validation_seeds, snap, norm)
        mean, swa = fmean(r.total_cost for r in rows), _swa(rows)
        curve.append({"arm": arm, "episode": ep, "val_total_cost": mean,
                      "val_serve_when_available": swa,
                      "credit_mean": info["credit_mean"]})
        progress(f"[train:{arm}] ep={ep}/{cfg.train_episodes} "
                 f"val_cost={mean:.2f} swa={swa:.2f} "
                 f"D_mean={info['credit_mean']:+.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    selections = {arm: {"arm": arm, "episode": best[1],
                        "val_total_cost": best[0]}}
    chosen = best[2]
    progress(f"[test] {arm} (선택 ep={best[1]})")
    results = {arm: _rl_rows(
        _eval_norm(profile, params, cfg.test_seeds, chosen, norm),
        cfg.test_seeds)}
    chosen.save(out / f"model_{arm}.pt")
    progress("[test] SF_SPT (신규 평가 — 이 대역 기존 행 없음, prereg §3)")
    results["SF_SPT"] = _baseline_rows(profile, params, cfg.test_seeds, rc,
                                       ServiceFirstSPTPreference(), "SF_SPT")
    if cfg.reuse:
        results.update(_load_reused_rows(cfg.test_seeds))
        progress("[test] INDEP@2000/JOINT_ROLLOUT/QMIX@2000 기존 판정 행 재사용")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    refs = (["INDEP@2000", "JOINT_ROLLOUT", "SF_SPT"] if cfg.reuse
            else ["SF_SPT"])
    for t, ref in enumerate(refs, start=1):
        paired[f"{arm}_vs_{ref}"] = _paired(results[ref], results[arm], cfg, t)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if cfg.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "승자 처방 무조정 확전 — 외부 40·550k 대역 (prereg §2)",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg_any(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr068_results.json", payload)
    report = _report_yr068(payload, out)
    progress(f"[YR-068] 완료 ({payload['manifest']['elapsed_s']:.0f}s) -> {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr068(out_dir=("outputs/reports/yr068_scale_quick" if quick
                       else "outputs/reports/yr068_scale_diff_qmix"),
              cfg=quick_yr068_config() if quick else None)
