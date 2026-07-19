"""3축 종합 검증 매트릭스 (DQN 단일학습기, QMIX 미접촉).

축1 학습부: variant/lr/target_sync
축2 표적·정규화: 상태 scale-only 정규화 on/off, cost_scale(보상 표적 스케일) ×0.5/×2
축3 후보·조정부: block_pre_rehandle, k_max(pruning budget)

공통 진단: overfit(train 비용 하락) · 일반화(held 하락) · baseline 대비 · loss.
전부 '탐색 진단'이며 locked 주장 아님. 결과 JSON 저장.
"""
import json, random, copy, time, dataclasses
from statistics import fmean

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator
import yard_rl.integrated.dqn_learner as dq
from yard_rl.integrated.dqn_learner import CandidateDQNLearner, LearnerConfig, run_episode
from yard_rl.integrated.encoding import encode_observation as ENC_ORIG, encoding_dims
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.resolver import BaselinePreference
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.experiments.candidate_dqn_experiment import SPTPreference
from yard_rl.integrated.scenario_gen import TerminalGenParams, generate_terminal_scenario

LEVEL = InformationLevel.PRE_ADVICE
OUT = "/mnt/c/Users/geonu/Desktop/port_reinforcement/outputs/reports/learner_verify/matrix.json"
EPOCHS, CKPT = 90, 30
CLIP = 5.0
TRAIN = [300000, 300001, 300002]
HELD = [310000, 310001, 310002, 310003]
PARAMS = TerminalGenParams(n_external=16, n_vessels=2)


def make_sim(seed):
    return TerminalSimulator(PROFILE, generate_terminal_scenario(PROFILE, seed, PARAMS),
                             check_invariants=True)


def dims_of(seed):
    sim = make_sim(seed); sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "dims", 0)
    return encoding_dims(ENC_ORIG(state, obs[0]))


def eval_rows(seeds, learner=None, pref_factory=None, generator=None):
    rows = []
    for s in seeds:
        pref = pref_factory() if pref_factory else QPreference()
        rows.append(run_episode(make_sim(s), level=LEVEL, preference=pref,
                                learner=learner, epsilon=0.0, generator=generator))
    return rows


def action_mix(rows):
    agg = {}
    for r in rows:
        for k, v in r.extras.get("action_counts", {}).items():
            agg[k] = agg.get(k, 0) + v
    tot = max(1, sum(agg.values()))
    return {k: round(v / tot, 3) for k, v in sorted(agg.items())}


# ---------------- 상태 scale-only 정규화 (전략문서 §4 프로토타입) ----------------
def fit_norm(seeds):
    """baseline 롤아웃으로 그룹별 per-dim scale(P90 abs of value channel) 산출."""
    buf = {"g": [], "yc": [], "queue": [], "cand": []}
    rec = []
    dq.encode_observation = lambda s, o: (lambda e: (rec.append(e), e)[1])(ENC_ORIG(s, o))
    try:
        for s in seeds:
            run_episode(make_sim(s), level=LEVEL, preference=BaselinePreference(), epsilon=0.0)
    finally:
        dq.encode_observation = ENC_ORIG
    for e in rec:
        buf["g"].append(e.g); buf["yc"].append(e.yc); buf["queue"].append(e.queue)
        for c in e.cand:
            buf["cand"].append(c)

    def p90abs(vecs, L):
        scales = []
        for d in range(L):
            xs = sorted(abs(v[d]) for v in vecs)
            scales.append(max(xs[min(len(xs) - 1, int(len(xs) * 0.9))], 1e-3))
        return scales
    norm = {}
    for grp, vecs in buf.items():
        L = len(vecs[0]) // 2                     # value 채널 길이 (앞 절반)
        norm[grp] = p90abs(vecs, L)
    return norm


def make_norm_enc(norm):
    def scale_vec(vec, scales):
        L = len(scales)
        val = [max(-CLIP, min(CLIP, vec[i] / scales[i])) for i in range(L)]  # 나눗셈+클립, 0→0 보존
        return tuple(val) + tuple(vec[L:])         # known 채널 원본 유지
    def wrapped(state, ob):
        e = ENC_ORIG(state, ob)
        return dataclasses.replace(
            e, g=scale_vec(e.g, norm["g"]), yc=scale_vec(e.yc, norm["yc"]),
            queue=scale_vec(e.queue, norm["queue"]),
            cand=tuple(scale_vec(c, norm["cand"]) for c in e.cand))
    return wrapped


# ---------------------------------- 단일 run ----------------------------------
def run_one(label, *, variant="ddqn", lr=1e-3, target_sync=500, cost_scale_mult=1.0,
            norm=False, block_pre=False, k_max=12):
    gen = CandidateGenerator(k_max=k_max, block_pre_rehandle=block_pre)
    base_tr = eval_rows(TRAIN, pref_factory=BaselinePreference, generator=gen)
    cost_scale = max(1e-6, fmean(r.total_cost / max(1, r.n_decisions) for r in base_tr)) * cost_scale_mult

    cfg = LearnerConfig(variant=variant, lr=lr, target_sync_every=target_sync, cost_scale=cost_scale)
    learner = CandidateDQNLearner(cfg, DIMS, seed=42)
    explore = random.Random(123)
    losses = []
    _orig = learner.learn_step
    def _patched():
        l = _orig()
        if l is not None: losses.append(l)
        return l
    learner.learn_step = _patched

    dq.encode_observation = make_norm_enc(NORM) if norm else ENC_ORIG
    try:
        pre_tr = [r.total_cost for r in eval_rows(TRAIN, learner=learner, generator=gen)]
        pre_hd = [r.total_cost for r in eval_rows(HELD, learner=learner, generator=gen)]
        curve = []
        for ep in range(1, EPOCHS + 1):
            seed = TRAIN[(ep - 1) % len(TRAIN)]
            run_episode(make_sim(seed), level=LEVEL, preference=QPreference(), learner=learner,
                        epsilon=1.0 / (ep ** 0.5), explore_rng=explore, collect=True, learn=True,
                        generator=gen)
            if ep % CKPT == 0 or ep == EPOCHS:
                snap = copy.deepcopy(learner); snap.learn_step = _orig
                tr = fmean(r.total_cost for r in eval_rows(TRAIN, learner=snap, generator=gen))
                hd = fmean(r.total_cost for r in eval_rows(HELD, learner=snap, generator=gen))
                curve.append({"ep": ep, "train": round(tr, 2), "held": round(hd, 2)})
        post_tr_rows = eval_rows(TRAIN, learner=learner, generator=gen)
        post_hd = [r.total_cost for r in eval_rows(HELD, learner=learner, generator=gen)]
        post_tr = [r.total_cost for r in post_tr_rows]
        mix_post = action_mix(post_tr_rows)
    finally:
        dq.encode_observation = ENC_ORIG

    base_vw_hd = fmean(r.total_cost for r in eval_rows(HELD, pref_factory=BaselinePreference, generator=gen))
    base_spt_hd = fmean(r.total_cost for r in eval_rows(HELD, pref_factory=SPTPreference, generator=gen))
    res = {
        "label": label,
        "knobs": {"variant": variant, "lr": lr, "target_sync": target_sync,
                  "cost_scale_mult": cost_scale_mult, "norm": norm,
                  "block_pre": block_pre, "k_max": k_max},
        "overfit": {"pre": round(fmean(pre_tr), 2), "post": round(fmean(post_tr), 2),
                    "drop_pct": round(100 * (fmean(pre_tr) - fmean(post_tr)) / max(1e-9, fmean(pre_tr)), 1)},
        "generalize": {"pre": round(fmean(pre_hd), 2), "post": round(fmean(post_hd), 2),
                       "drop_pct": round(100 * (fmean(pre_hd) - fmean(post_hd)) / max(1e-9, fmean(pre_hd)), 1)},
        "baseline_held": {"VW": round(base_vw_hd, 2), "SPT": round(base_spt_hd, 2)},
        "post_vs_bestbase_held": round(fmean(post_hd) - min(base_vw_hd, base_spt_hd), 2),
        "loss": {"first": round(fmean(losses[:50]), 4) if len(losses) >= 50 else None,
                 "last": round(fmean(losses[-50:]), 4) if len(losses) >= 50 else None},
        "action_mix_post": mix_post, "curve": curve,
    }
    print(f"[{label:22s}] overfit {res['overfit']['pre']:.1f}->{res['overfit']['post']:.1f} "
          f"({res['overfit']['drop_pct']:+.0f}%) held {res['generalize']['pre']:.1f}->"
          f"{res['generalize']['post']:.1f} vs_base {res['post_vs_bestbase_held']:+.2f}", flush=True)
    return res


def main():
    global PROFILE, DIMS, NORM
    t0 = time.time()
    PROFILE = build_integrated_profile()
    DIMS = dims_of(TRAIN[0])
    NORM = fit_norm(TRAIN)
    print(f"dims={DIMS} norm_fit done", flush=True)

    runs = []
    # 축1 학습부
    runs.append(run_one("A1.ddqn_ref"))
    runs.append(run_one("A1.dqn", variant="dqn"))
    runs.append(run_one("A1.dueling", variant="dueling"))
    runs.append(run_one("A1.lr3e-4", lr=3e-4))
    runs.append(run_one("A1.tsync200", target_sync=200))
    # 축2 표적·정규화
    runs.append(run_one("A2.state_norm", norm=True))
    runs.append(run_one("A2.cost_x0.5", cost_scale_mult=0.5))
    runs.append(run_one("A2.cost_x2", cost_scale_mult=2.0))
    # 축3 후보·조정부
    runs.append(run_one("A3.block_pre", block_pre=True))
    runs.append(run_one("A3.kmax8", k_max=8))
    runs.append(run_one("A3.kmax20", k_max=20))

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"elapsed_s": round(time.time() - t0, 1),
                   "setup": {"epochs": EPOCHS, "n_external": PARAMS.n_external,
                             "train": TRAIN, "held": HELD, "clip": CLIP},
                   "runs": runs}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
