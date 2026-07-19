"""YR-063 — 신용 개별화: 차분(counterfactual WAIT) 귀속 1-step Q.

배경 (YR-061/062): 팀 공유 return 균등귀속 TD 는 퇴화 정책을 만들고(70.11·swa 0.094)
건강한 BC 정책조차 15ep 내 파괴한다 — 병목 = 신용 희석. 잔여 유일 경로가 귀속 교체다.

처방: 크레인 i 의 결정 credit 을 팀 return 이 아니라 **차분(difference reward)** 으로.
    D_i = C_W(내 행동, 상대 행동) − C_W(WAIT_i, 상대 행동)
- C_W = 고정 시간창(window_s) 누적 팀비용 — `_rollout_cost`(JR baseline 기계) 재사용,
  분기 후 base_policy(SF_SPT resolver)로 진행. 상대 행동은 실제 선택으로 고정.
- "내 행동이 팀 비용에 실제로 만든 차이"만 귀속 → 희석 제거. WAIT 선택 시 D=0 (자기 앵커).
- **1-step 표본**(gamma_dt=0·부트스트랩 없음): YR-061 이분 검정이 보인 대로 순위 신호가
  관건이고, TD 부트스트랩은 희석·불안정의 통로였다. Q(s,c) ≈ D 회귀 → argmin Q 실행.
- 학습 시 특권 정보(시뮬레이터 rollout) 사용, 실행 시 Q 만 — CTDE 관례 (QMIX 와 동일 지위).
  평가 경로는 기존 run_episode(QPreference) 그대로.
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
from ..integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig,
                                      Sample, _FORCE)
from ..integrated.encoding import encode_observation, encoding_dims
from ..integrated.qnet import QPreference
from ..integrated.resolver import CentralResolver
from .direct_job_runner import _git_state, _json_dump
from .yr061_reward_redesign import (_agg, _eval, _paired, _params, _report,
                                    _rl_rows, _sim, _swa, Yr061Config,
                                    quick_yr061_config)

EXPERIMENT_ID = "YR-063-difference-credit"
LEVEL = InformationLevel.PRE_ADVICE

# 동일 test 대역(620000~620019)의 기존 판정 행 재사용 (yr013 reuse_jr 관례) —
# 전 정책 비학습·결정론이라 재실행과 동일함이 실증됨 (YR-061 phase-2 g0.95==pen0).
REUSE_ROWS = {
    "CONTROL_TD": ("outputs/reports/yr061_reward/test_results.json", "pen0"),
    "BC": ("outputs/reports/yr061_imitation/test_results.json", "IMITATE"),
    "SF_SPT": ("outputs/reports/yr061_imitation/test_results.json", "SF_SPT"),
    "FIFO": ("outputs/reports/yr061_imitation/test_results.json", "FIFO"),
}


@dataclass(frozen=True)
class Yr063Config:
    base: Yr061Config = Yr061Config()          # seed 대역·시나리오·판정 기계 승계
    window_s: float = 600.0                    # 차분 credit 시간창 (JR 기본과 동일)
    reuse: bool = True                         # 기존 test 행 재사용


def quick_yr063_config() -> Yr063Config:
    return Yr063Config(base=quick_yr061_config(), window_s=300.0, reuse=False)


def run_diff_episode(sim, *, learner: CandidateDQNLearner, rc, window_s: float,
                     epsilon: float = 0.0, explore_rng: random.Random | None = None,
                     learn: bool = True, state_norm=None) -> dict:
    """차분 credit 수집·학습 드라이버 — run_episode 의 score/resolve 골격 +
    결정마다 counterfactual WAIT rollout 으로 D_i 를 계산해 1-step 표본을 쌓는다.
    전략적 WAIT 기본 제외(YR-052)·탐험 강제 방식은 run_episode 와 동일.
    state_norm (YR-059/067): 학습 인코딩 전용 — rollout·resolver 경로 불변."""
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
                d = actual_cost - cf_cost          # 음수 = 내 행동이 팀비용을 줄임
            if pos is not None:
                learner.replay.append(Sample(enc, pos, d, 0.0, None))
                credits.append(d)
                n_samples += 1
        resolver.apply(sim, resn, gen_by)
        if learn:
            for _ in range(learner.cfg.updates_per_decision):
                learner.learn_step()
        dp = sim.run_until_decision()
        sim.cost.cut()
        k += 1
    return {"n_decisions": k, "n_samples": n_samples,
            "credit_mean": fmean(credits) if credits else 0.0,
            "credit_min": min(credits, default=0.0),
            "credit_max": max(credits, default=0.0)}


def _load_reused_rows(test_seeds) -> dict:
    out = {}
    for name, (path, key) in REUSE_ROWS.items():
        rows = json.loads(Path(path).read_text(encoding="utf-8"))[key]
        if [r["seed"] for r in rows] != list(test_seeds):
            raise ValueError(f"{name} 재사용 행의 test seed 불일치")
        out[name] = rows
    return out


def run_yr063(out_dir: str = "outputs/reports/yr063_diff",
              cfg: Yr063Config | None = None,
              progress: Callable[[str], None] = print) -> Path:
    cfg = cfg or Yr063Config()
    base = cfg.base
    started = time.time()
    git = _git_state()
    if not base.quick and (git["commit"] == "unknown" or git["dirty"]):
        raise RuntimeError("full YR-063 run requires a clean committed tree")
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
    learner = CandidateDQNLearner(
        LearnerConfig(variant=base.variant, lr=base.lr, cost_scale=1.0),
        dims, seed=63_000)                          # 차분 credit 은 O(1) — scale 불요
    explore = random.Random(63_100)
    progress(f"[YR-063] dims={dims} window={cfg.window_s:g}s")

    curve: list[dict] = []
    best: tuple | None = None
    for ep, seed in enumerate(base.train_seeds, start=1):
        eps = 1.0 / (ep ** 0.5)
        info = run_diff_episode(_sim(profile, seed, params), learner=learner, rc=rc,
                                window_s=cfg.window_s, epsilon=eps,
                                explore_rng=explore, learn=True)
        if ep % base.checkpoint_every and ep != base.train_episodes:
            continue
        snap = copy.deepcopy(learner)
        snap.replay.clear()
        rows = _eval(profile, params, base.validation_seeds, snap)
        mean, swa = fmean(r.total_cost for r in rows), _swa(rows)
        curve.append({"arm": "DIFF", "episode": ep, "val_total_cost": mean,
                      "val_serve_when_available": swa,
                      "credit_mean": info["credit_mean"]})
        progress(f"[train:DIFF] ep={ep}/{base.train_episodes} val_cost={mean:.2f} "
                 f"swa={swa:.2f} D_mean={info['credit_mean']:+.2f}")
        if best is None or (mean, ep) < (best[0], best[1]):
            best = (mean, ep, snap)
    selections = {"DIFF": {"arm": "DIFF", "episode": best[1],
                           "val_total_cost": best[0]}}
    chosen = best[2]
    progress(f"[test] DIFF (선택 ep={best[1]})")
    results = {"DIFF": _rl_rows(_eval(profile, params, base.test_seeds, chosen),
                                base.test_seeds)}
    chosen.save(out / "model_DIFF.pt")
    if cfg.reuse:
        results.update(_load_reused_rows(base.test_seeds))
        progress("[test] CONTROL_TD/BC/SF_SPT/FIFO 기존 판정 행 재사용")
    _json_dump(out / "checkpoint_curve.json", curve)
    _json_dump(out / "selections.json", selections)
    _json_dump(out / "test_results.json", results)

    paired = {}
    if cfg.reuse:
        for t, ref in enumerate(("CONTROL_TD", "BC", "SF_SPT")):
            paired[f"DIFF_vs_{ref}"] = _paired(results[ref], results["DIFF"],
                                               base, t + 1)
    payload = {
        "manifest": {"schema_version": 1, "strategy_id": EXPERIMENT_ID,
                     "mode": "quick" if base.quick else "full",
                     "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "git": git, "config": asdict(cfg), "info_level": LEVEL.value,
                     "note": "차분 credit 1-step Q — 학습시 특권 rollout·실행시 Q만 (CTDE)",
                     "elapsed_s": time.time() - started},
        "selections": selections, "paired": paired,
        "means": {name: _agg(rows) for name, rows in results.items()},
    }
    _json_dump(out / "yr063_results.json", payload)
    report = _report(payload, out, name="yr063_report.md",
                     title="YR-063 — 차분 귀속 1-step Q 판정 결과")
    progress(f"[YR-063] 완료 ({payload['manifest']['elapsed_s']:.0f}s) → {report}")
    return report


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv[1:]
    run_yr063(out_dir=("outputs/reports/yr063_diff_quick" if quick
                       else "outputs/reports/yr063_diff"),
              cfg=quick_yr063_config() if quick else None)
