"""YR-013c — 차분 표적 QMIX: 명시 신용 앵커 + mixer 팀 보정 (prereg 2026-07-19).

가설: 차분 D_i 는 상호작용 교차항을 놓친다 — 단조 mixer 가 팀 창비용 정합을
보정항으로 학습하면 (utilities 는 D 앵커 고정) DIFF2400_NORM(75.38)을 넘는다.
차이는 mixer 항 **하나**: 나머지는 YR-067 DIFF2400_NORM arm 동결값 승계.
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
from ..integrated.baselines import (ResolverPolicy, ServiceFirstSPTPreference,
                                    _rollout_cost, _wait_of)
from ..integrated.candidates import CandidateGenerator
from ..integrated.cost_config import RewardCalculator
from ..integrated.dqn_learner import _FORCE
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.qmix import DiffQmixConfig, DiffQmixLearner, JointDiffSample
from ..integrated.qnet import QPreference
from ..integrated.resolver import CentralResolver
from .direct_job_runner import _git_state, _json_dump
from .yr059_state_norm import fit_state_norm
from .yr061_reward_redesign import (_agg, _paired, _params, _report, _rl_rows,
                                    _sim, _swa, Yr061Config, quick_yr061_config)
from .yr067_norm_apply import _eval_norm

EXPERIMENT_ID = "YR-013c-diff-target-qmix"
LEVEL = InformationLevel.PRE_ADVICE

REUSE_ROWS = {
    "DIFF2400_NORM": ("outputs/reports/yr067_norm/test_results.json", "DIFF2400_NORM"),
    "CONTROL_TD": ("outputs/reports/yr061_reward/test_results.json", "pen0"),
    "SF_SPT": ("outputs/reports/yr061_imitation/test_results.json", "SF_SPT"),
    "FIFO": ("outputs/reports/yr061_imitation/test_results.json", "FIFO"),
}


@dataclass(frozen=True)
class Yr013cConfig:
    base: Yr061Config = Yr061Config()
    window_s: float = 2_400.0            # YR-065 승자 창 (DIFF2400_NORM 동일)
    lambda_mix: float = 1.0              # prereg §2 동결 — knob 탐색 금지
    fit_seeds_n: int = 5
    reuse: bool = True


def quick_yr013c_config() -> Yr013cConfig:
    return Yr013cConfig(base=quick_yr061_config(), window_s=300.0, reuse=False)


def run_diff_qmix_episode(sim, *, learner: DiffQmixLearner, rc, window_s: float,
                          epsilon: float = 0.0,
                          explore_rng: random.Random | None = None,
                          learn: bool = True, state_norm=None) -> dict:
    """yr063.run_diff_episode 동형 — 표본만 결정 단위 JointDiffSample 로 적재.

    D_i 계산·전략 WAIT 제외·탐험·rollout 전부 YR-063 동결 그대로. mixer 표적
    C_W_team 은 D 계산에 이미 쓰는 actual rollout 값 (추가 rollout 0회).
    """
    gen = CandidateGenerator()
    preference = QPreference()
    resolver = CentralResolver(preference)
    base_policy = ResolverPolicy(ServiceFirstSPTPreference(), "BASE")
    rng = explore_rng or random.Random(0)
    sim.info_level = LEVEL
    n_samples, k = 0, 0
    credits: list[float] = []
    dp = sim.run_until_decision()
    sim.cost.cut()
    while dp is not None:
        state, obs, gen_by = capture(sim, dp.crane_ids, LEVEL, "drive", k,
                                     generator=gen)
        encs = {ob.crane_id: encode_observation(state, ob, norm=state_norm)
                for ob in obs}
        scores: dict[tuple[str, int], float] = {}
        for cid, enc in encs.items():
            s = learner.scores_for(enc)
            wait_cid = (enc.candidate_ids[enc.wait_pos]
                        if enc.wait_pos is not None else None)
            if (wait_cid is not None
                    and any(enc.actionable[i] and i != enc.wait_pos
                            for i in range(len(enc.candidate_ids)))):
                s[wait_cid] = -_FORCE              # 전략적 WAIT 제외 (YR-052)
            if epsilon > 0.0 and rng.random() < epsilon:
                pool = [c for i, c in enumerate(enc.candidate_ids)
                        if enc.actionable[i] and c != wait_cid]
                if pool:
                    s[rng.choice(pool)] = _FORCE
            scores.update({(cid, c): v for c, v in s.items()})
        preference.set_scores(scores)
        resn = resolver.resolve(sim, dp, gen_by)
        assign = {}
        for r in resn.resolutions:
            assign[r.crane_id] = (_wait_of(gen_by[r.crane_id])
                                  if r.chosen_candidate_id is None
                                  else gen_by[r.crane_id].items[r.chosen_candidate_id])
        actual_cost, _ = _rollout_cost(sim, assign, rc, horizon_s=window_s,
                                       base_policy=base_policy, generator=gen)
        j_encs, j_pos, j_d = [], [], []
        for r in resn.resolutions:
            enc = encs[r.crane_id]
            if r.chosen_candidate_id is None:
                pos, d = enc.wait_pos, 0.0         # WAIT = 자기 앵커 (D=0)
            else:
                pos = enc.candidate_ids.index(r.chosen_candidate_id)
                cf = dict(assign)
                cf[r.crane_id] = _wait_of(gen_by[r.crane_id])
                cf_cost, _ = _rollout_cost(sim, cf, rc, horizon_s=window_s,
                                           base_policy=base_policy, generator=gen)
                d = actual_cost - cf_cost
            if pos is not None:
                j_encs.append(enc)
                j_pos.append(pos)
                j_d.append(d)
                credits.append(d)
        if j_encs:
            learner.replay.append(JointDiffSample(tuple(j_encs), tuple(j_pos),
                                                  tuple(j_d), actual_cost))
            n_samples += 1
        resolver.apply(sim, resn, gen_by)
        if learn:
            for _ in range(learner.cfg.updates_per_decision):
                learner.learn_step()
        dp = sim.run_until_decision()
        sim.cost.cut()
        k += 1
    return {"n_decisions": k, "n_samples": n_samples,
            "credit_mean": fmean(credits) if credits else 0.0}


def _load_reused_rows(test_seeds) -> dict:
    out = {}
    for name, (path, key) in REUSE_ROWS.items():
        rows = json.loads(Path(path).read_text(encoding="utf-8"))[key]
        if [r["seed"] for r in rows] != list(test_seeds):
            raise ValueError(f"{name} 재사용 행의 test seed 불일치")
        out[name] = rows
    return out


def run_yr013c(out_dir: str = "outputs/reports/yr013c_diff_qmix",
               cfg: Yr013cConfig | None = None,
               progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr013cConfig()
    base = cfg.base
    started = time.time()
    git = _git_state()
    if not base.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-013c run requires a clean committed tree")
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
    norm, detail = fit_state_norm(profile, params,
                                  base.train_seeds[:cfg.fit_seeds_n],
                                  progress=progress)
    _json_dump(out / "state_norm.json",
               {"refs": norm.refs, "clip": norm.clip, "basis": norm.basis,
                "fit_seeds": list(base.train_seeds[:cfg.fit_seeds_n]),
                "detail": detail})
    n_agents = len(list(profile.cranes)) if hasattr(profile, "cranes") else 2
    learner = DiffQmixLearner(
        DiffQmixConfig(variant=base.variant, n_agents=n_agents, lr=base.lr,
                       lambda_mix=cfg.lambda_mix), dims, seed=63_000)
    explore = random.Random(63_100)
    arm = "DIFF_QMIX"
    progress(f"[YR-013c] dims={dims} window={cfg.window_s:g}s "
             f"lambda_mix={cfg.lambda_mix:g} n_agents={n_agents}")

    curve, best = [], None
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        info = run_diff_qmix_episode(_sim(profile, seed, params), learner=learner,
                                     rc=rc, window_s=cfg.window_s, epsilon=eps,
                                     explore_rng=explore, learn=True,
                                     state_norm=norm)
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
    selections = {arm: {"arm": arm, "episode": best[1],
                        "val_total_cost": best[0]}}
    chosen = best[2]
    progress(f"[test] {arm} (선택 ep={best[1]})")
    results = {arm: _rl_rows(
        _eval_norm(profile, params, base.test_seeds, chosen, norm),
        base.test_seeds)}
    chosen.save(out / f"model_{arm}.pt")
    if cfg.reuse:
        results.update(_load_reused_rows(base.test_seeds))
        progress("[test] DIFF2400_NORM/CONTROL_TD/SF_SPT/FIFO 기존 판정 행 재사용")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    if cfg.reuse:
        for t, ref in enumerate(("DIFF2400_NORM", "SF_SPT", "CONTROL_TD"), start=1):
            paired[f"{arm}_vs_{ref}"] = _paired(results[ref], results[arm], base, t)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if base.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "D 앵커 + λ·mixer(팀 창비용) — 1-step·target 망 없음 (prereg)",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr013c_results.json", payload)
    report = _report(payload, out, name="yr013c_report.md",
                     title="YR-013c — 차분 표적 QMIX 판정 결과")
    progress(f"[YR-013c] 완료 ({payload['manifest']['elapsed_s']:.0f}s) -> {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr013c(out_dir=("outputs/reports/yr013c_diff_qmix_quick" if quick
                        else "outputs/reports/yr013c_diff_qmix"),
               cfg=quick_yr013c_config() if quick else None)
