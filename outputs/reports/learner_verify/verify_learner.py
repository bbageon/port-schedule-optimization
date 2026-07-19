"""학습부 검증 (DQN 단일학습기) — QMIX 미접촉.

목적: '학습기가 실제로 배우는가'를 baseline 승리와 분리해 판정.
- (A) overfit: 작은 고정 seed 집합에 학습 → 그 집합 비용이 내려가는가 (기계 정상성)
- (B) loss 곡선: Huber loss 가 내려가는가
- (C) 일반화: held-out seed 도 내려가는가 (overfit-only vs 일반화)
- (D) baseline 대조: VESSEL_WAIT·SPT 대비 위치
- (E) 행동분포 이동: 학습 전/후 action_counts 변화 (정책이 실제로 바뀌는가)
결과: JSON 으로 저장 + 요약 print.
"""
import json, random, copy, time
from statistics import fmean, pstdev

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.dqn_learner import CandidateDQNLearner, LearnerConfig, run_episode
from yard_rl.integrated.encoding import encode_observation, encoding_dims
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.resolver import BaselinePreference
from yard_rl.experiments.candidate_dqn_experiment import SPTPreference
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE
OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/learner_verify/verify.json"


def make_sim(profile, seed, params):
    return TerminalSimulator(profile, generate_terminal_scenario(profile, seed, params),
                             check_invariants=True)


def dims_of(profile, params, seed):
    sim = make_sim(profile, seed, params); sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "dims", 0)
    return encoding_dims(encode_observation(state, obs[0]))


def eval_costs(profile, params, seeds, learner=None, pref_factory=None):
    rows = []
    for s in seeds:
        pref = pref_factory() if pref_factory else QPreference()
        r = run_episode(make_sim(profile, s, params), level=LEVEL, preference=pref,
                        learner=learner, epsilon=0.0)
        rows.append(r)
    return rows


def action_mix(rows):
    agg = {}
    for r in rows:
        for k, v in r.extras.get("action_counts", {}).items():
            agg[k] = agg.get(k, 0) + v
    tot = max(1, sum(agg.values()))
    return {k: round(v / tot, 3) for k, v in sorted(agg.items())}


def main():
    t0 = time.time()
    profile = build_integrated_profile()
    params = TerminalGenParams(n_external=24, n_vessels=2)   # 중간 규모 (속도·대표성 절충)
    TRAIN = [300000, 300001, 300002]                          # overfit 표적 (작은 고정집합)
    HELD = [310000, 310001, 310002, 310003, 310004]           # held-out 일반화
    EPOCHS = 150
    CKPT = 15

    dims = dims_of(profile, params, TRAIN[0])

    # cost_scale = train baseline 결정당 비용 (test 미접촉 관례)
    base_tr_rows = eval_costs(profile, params, TRAIN, pref_factory=BaselinePreference)
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions) for r in base_tr_rows))

    cfg = LearnerConfig(variant="ddqn", cost_scale=cost_scale)
    learner = CandidateDQNLearner(cfg, dims, seed=42)
    explore = random.Random(123)

    # loss 후킹
    losses = []
    _orig = learner.learn_step
    def _patched():
        l = _orig()
        if l is not None:
            losses.append(l)
        return l
    learner.learn_step = _patched

    # baseline·untrained 기준선
    base_vw_held = [r.total_cost for r in eval_costs(profile, params, HELD, pref_factory=BaselinePreference)]
    base_spt_held = [r.total_cost for r in eval_costs(profile, params, HELD, pref_factory=SPTPreference)]
    base_vw_train = [r.total_cost for r in base_tr_rows]

    pre_train_rows = eval_costs(profile, params, TRAIN, learner=learner)
    pre_held_rows = eval_costs(profile, params, HELD, learner=learner)
    pre_train = [r.total_cost for r in pre_train_rows]
    pre_held = [r.total_cost for r in pre_held_rows]
    mix_pre = action_mix(pre_train_rows)

    curve = []
    for ep in range(1, EPOCHS + 1):
        seed = TRAIN[(ep - 1) % len(TRAIN)]
        eps = 1.0 / (ep ** 0.5)
        run_episode(make_sim(profile, seed, params), level=LEVEL, preference=QPreference(),
                    learner=learner, epsilon=eps, explore_rng=explore, collect=True, learn=True)
        if ep % CKPT == 0 or ep == EPOCHS:
            snap = copy.deepcopy(learner)
            snap.learn_step = _orig   # 후킹 제거된 순정 (평가엔 무관)
            tr = [r.total_cost for r in eval_costs(profile, params, TRAIN, learner=snap)]
            hd = [r.total_cost for r in eval_costs(profile, params, HELD, learner=snap)]
            curve.append({"ep": ep, "train": fmean(tr), "held": fmean(hd),
                          "replay": len(learner.replay), "grad": learner.grad_steps})
            print(f"ep={ep:3d} train={fmean(tr):8.3f} held={fmean(hd):8.3f} "
                  f"replay={len(learner.replay):4d} grad={learner.grad_steps:4d} "
                  f"loss_last50={fmean(losses[-50:]) if len(losses)>=50 else float('nan'):.4f}",
                  flush=True)

    post_train_rows = eval_costs(profile, params, TRAIN, learner=learner)
    post_held_rows = eval_costs(profile, params, HELD, learner=learner)
    post_train = [r.total_cost for r in post_train_rows]
    post_held = [r.total_cost for r in post_held_rows]
    mix_post = action_mix(post_train_rows)

    def mean(x): return round(fmean(x), 3)
    def sd(x): return round(pstdev(x), 3) if len(x) > 1 else 0.0

    summary = {
        "config": {"variant": cfg.variant, "lr": cfg.lr, "gamma": cfg.gamma,
                   "hidden": cfg.hidden, "target_sync": cfg.target_sync_every,
                   "replay_cap": cfg.replay_capacity, "cost_scale": round(cost_scale, 2),
                   "epochs": EPOCHS, "n_external": params.n_external, "n_vessels": params.n_vessels,
                   "train_seeds": TRAIN, "held_seeds": HELD},
        "baseline_held": {"VESSEL_WAIT": mean(base_vw_held), "SPT": mean(base_spt_held)},
        "baseline_train_VW": mean(base_vw_train),
        "overfit_train": {"pre": mean(pre_train), "post": mean(post_train),
                          "drop": round(fmean(pre_train) - fmean(post_train), 3),
                          "drop_pct": round(100 * (fmean(pre_train) - fmean(post_train)) / max(1e-9, fmean(pre_train)), 1),
                          "vs_baseline_train": round(fmean(post_train) - fmean(base_vw_train), 3)},
        "generalize_held": {"pre": mean(pre_held), "post": mean(post_held),
                            "drop": round(fmean(pre_held) - fmean(post_held), 3),
                            "vs_best_baseline": round(fmean(post_held) - min(fmean(base_vw_held), fmean(base_spt_held)), 3)},
        "loss": {"n": len(losses),
                 "first50": round(fmean(losses[:50]), 4) if len(losses) >= 50 else None,
                 "last50": round(fmean(losses[-50:]), 4) if len(losses) >= 50 else None},
        "action_mix_train": {"pre": mix_pre, "post": mix_post},
        "curve": curve,
        "elapsed_s": round(time.time() - t0, 1),
    }

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 요약 =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"},
                     ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
