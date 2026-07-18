"""YR-059 — 상태 scale-only 정규화 fit + QMIX 예산사다리 재실행 (적용전략 §4·§6-1).

절차:
1. fit_state_norm — 학습 대역 baseline(BaselinePreference) 에피소드에서 필드별 |값| 분포를
   수집해 P90 을 동결 기준값(norm_ref override)으로 삼는다. 표본 없는 필드는 스키마
   assumed 유지. 출처(basis)·표본수를 JSON 으로 박제 — val/test 재적합 금지.
2. run_yr059 — 같은 사다리 프로토콜(YR-013: 2000ep·tier 500/1000/2000·lr 3e-4·같은 seed
   대역)로 **정규화 ON** 학습을 실행. OFF 대조는 기존 yr013_qmix_ladder 결과를 그대로
   paired 로 읽는다 (같은 seed·같은 프로토콜 — 차이는 정규화뿐). JR 은 인코딩 미사용이라
   선행 run 의 행을 재사용한다.

변인 통제 주: val 은 OFF 사다리와 동일한 20 seed 를 유지한다 — val 확대(YR-057)는
paired 비교를 깨므로 이번 판정에 넣지 않고 별도 작업으로 남긴다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import fmean
from typing import Callable

from ..domain.enums import InformationLevel
from ..integrated import TerminalSimulator, build_integrated_profile
from ..integrated.adapter import capture
from ..integrated.dqn_learner import run_episode
from ..integrated.encoding import StateNorm
from ..integrated.resolver import BaselinePreference, CentralResolver
from ..integrated.scenario_gen import generate_terminal_scenario
from ..contract import SCHEMA
from .direct_job_runner import _json_dump
from .direct_stats import MetricDirection, MetricSpec, paired_bootstrap
from .yr013_qmix_experiment import Yr013Config, run_yr013, _params, _sim

LEVEL = InformationLevel.PRE_ADVICE


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def fit_state_norm(profile, params, seeds, *, percentile: float = 0.90,
                   clip: float = 5.0,
                   progress: Callable[[str], None] = print) -> tuple[StateNorm, dict]:
    """baseline 에피소드의 FeatureVector 에서 필드별 |값| P{percentile} 를 동결 기준으로.

    known=1 값만 표본. |값| 을 쓰는 이유: signed 필드(연착 gap·slack)의 크기 대칭 정규화.
    P90 이 0 인 필드(상수 0)는 스키마 assumed 유지 — 0 나눗셈 방지.
    """
    samples: dict[str, list[float]] = {}

    def _collect(fv):
        for name, v, kn in zip(fv.names, fv.value, fv.known):
            if kn:
                samples.setdefault(f"{fv.group}.{name}", []).append(abs(float(v)))

    for seed in seeds:
        sim = _sim(profile, seed, params)
        sim.info_level = LEVEL
        resolver = CentralResolver(BaselinePreference())
        k = 0
        dp = sim.run_until_decision()
        while dp is not None:
            state, obs, gen_by = capture(sim, dp.crane_ids, LEVEL, "fit", k)
            _collect(state.features)
            for v in state.vessels:
                _collect(v.features)
            for ob in obs:
                _collect(ob.features)
                _collect(ob.candidates.queue_summary)
                for c in ob.candidates.items:
                    _collect(c.features)
            resolver.apply(sim, resolver.resolve(sim, dp, gen_by), gen_by)
            dp = sim.run_until_decision()
            k += 1
        progress(f"[fit] seed {seed}: 결정 {k}, 필드 {len(samples)}")

    refs: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for key, xs in sorted(samples.items()):
        p = _percentile(xs, percentile)
        group, name = key.split(".", 1)
        assumed = SCHEMA.spec(group, name).norm_ref
        used = p if p > 1e-9 else assumed        # 상수 0 필드 → assumed 유지
        refs[key] = used
        detail[key] = {"n": len(xs), "p90_abs": p, "assumed": assumed, "used": used}
    norm = StateNorm(refs=refs, clip=clip, basis=f"fitted_baseline_p{int(percentile*100)}")
    return norm, detail


def run_yr059(out_dir: str = "outputs/reports/yr059_qmix_norm_ladder",
              off_ladder_dir: str = "outputs/reports/yr013_qmix_ladder",
              fit_seeds_n: int = 5,
              progress: Callable[[str], None] = print,
              quick: bool = False) -> Path:
    """정규화 ON 사다리 실행 + 기존 OFF 사다리와 paired 판정."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if quick:
        from .yr013_qmix_experiment import quick_yr013_config
        cfg = quick_yr013_config()
    else:
        cfg = Yr013Config(train_episodes=2000, budget_ladder=(500, 1000, 2000), lr=3e-4)
    profile = build_integrated_profile()
    params = _params(cfg)

    # 1) norm fit (train 대역 선두 seed — val/test 미접촉) + 박제
    norm_p = out / "state_norm.json"
    if norm_p.exists():
        d = json.loads(norm_p.read_text(encoding="utf-8"))
        norm = StateNorm(refs=d["refs"], clip=d["clip"], basis=d["basis"])
        progress(f"[fit] 기존 fit 재사용 ({d['basis']}, {len(d['refs'])} 필드)")
    else:
        norm, detail = fit_state_norm(profile, params, cfg.train_seeds[:fit_seeds_n],
                                      progress=progress)
        _json_dump(norm_p, {"refs": norm.refs, "clip": norm.clip, "basis": norm.basis,
                            "fit_seeds": list(cfg.train_seeds[:fit_seeds_n]),
                            "detail": detail})
        progress(f"[fit] {len(norm.refs)} 필드 동결 ({norm.basis})")

    # 2) 정규화 ON 사다리 — OFF 와 같은 프로토콜·seed. JR 은 OFF run 행 재사용.
    reuse = str(Path(off_ladder_dir) / "test_results.json")
    if not Path(reuse).exists():
        reuse = None
    run_yr013(out_dir=str(out), cfg=cfg, progress=progress, reuse_jr=reuse,
              state_norm=norm)

    # 3) ON vs OFF paired (같은 test seed — 정규화만 차이)
    onr = json.loads((out / "test_results.json").read_text(encoding="utf-8"))
    verdict: dict[str, object] = {}
    if reuse:
        offr = json.loads(Path(reuse).read_text(encoding="utf-8"))
        for name in sorted(onr):
            if name == "JOINT_ROLLOUT" or name not in offr:
                continue
            seeds = [r["seed"] for r in offr[name]]
            entry = {}
            for i, key_ in enumerate(("total_cost", "interference", "mean_wait_min")):
                entry[key_] = paired_bootstrap(
                    [float(r[key_]) for r in offr[name]],
                    [float(r[key_]) for r in onr[name]],
                    metric=MetricSpec(key_, MetricDirection.MINIMIZE), seeds=seeds,
                    seed=75_059 + i, n_resamples=10_000).as_dict()
            verdict[f"{name}: ON_vs_OFF"] = entry
        _json_dump(out / "yr059_on_vs_off.json", verdict)

    # 4) 요약 리포트
    lines = ["# YR-059 — 상태 정규화(scale-only·P90 동결) 후 QMIX 사다리 재실행", "",
             "> OFF 대조 = 기존 yr013_qmix_ladder (같은 seed·프로토콜 — 차이는 정규화뿐).",
             "> ⚠ 합성·가정 조건 (주장 게이트 YR-002/009).", ""]
    means = {n: fmean(float(r["total_cost"]) for r in rows) for n, rows in onr.items()}
    lines.append("| 정책(ON) | test 총비용 평균 |")
    lines.append("|---|---|")
    for n in sorted(means):
        lines.append(f"| {n} | {means[n]:.2f} |")
    lines.append("")
    for tag, e in verdict.items():
        tc = e["total_cost"]
        lines.append(f"- **{tag}**: Δtotal={tc['difference']:+.2f} "
                     f"[{tc['difference_ci']['lower']:+.2f}, "
                     f"{tc['difference_ci']['upper']:+.2f}] (음수 = ON 개선)")
    path = out / "yr059_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    progress(f"[YR-059] → {path}")
    return path


if __name__ == "__main__":
    import sys
    run_yr059(quick="--quick" in sys.argv[1:])
