"""YR-039 baseline 결함 피드백 검증 진단 (val 대역만 사용 — test 미접촉).

주장: BASELINE_SPT 가 plan.duration_s 를 SERVE·PRE·REPOSITION 전 후보에
일괄 적용해 '가장 짧은 행동'(짧은 재배치)을 반복 선택 → 약한 baseline.
검증: 정책별 선택 행동 종류 분포 + mean_wait/total_cost 비교.
"""
import sys
from collections import Counter, defaultdict
from statistics import fmean

sys.path.insert(0, "src")

from yard_rl.domain.enums import InformationLevel
from yard_rl.experiments.candidate_dqn_experiment import (SPTPreference, _sim)
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.candidates import CandidateGenerator
from yard_rl.contract.schema import CandidateKind
from yard_rl.integrated.dqn_learner import (CandidateDQNLearner,
                                            _max_vessel_risk_state)
from yard_rl.integrated.encoding import encode_observation
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.resolver import BaselinePreference, CentralResolver
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.scenario_gen import TerminalGenParams

POC_MODEL = "outputs/reports/candidate_dqn_poc/model_CandidateDQN[dueling].pt"

LEVEL = InformationLevel.PRE_ADVICE


class ServiceFirstSPT(BaselinePreference):
    """진단 정책 — 서비스 우선, 서비스끼리 SPT, 서비스 없을 때만 PRE/REPO."""

    def rank(self, sim, crane_id, gc):
        dur = gc.plan.duration_s if gc.plan is not None else float("inf")
        serve_first = 0 if gc.kind == CandidateKind.SERVE else 1
        return (serve_first, dur) + super().rank(sim, crane_id, gc)


def run_counted(sim, pref, learner=None):
    """run_episode 동형 루프 + 선택 후보 kind 집계 + 항목별 기여 적산."""
    gen = CandidateGenerator()
    rc = RewardCalculator.assumed_default()
    resolver = CentralResolver(pref)
    sim.info_level = LEVEL
    kinds = Counter()
    costs = []
    terms = defaultdict(float)
    dp = sim.run_until_decision()
    sim.cost.cut()
    k = 0
    while dp is not None:
        state, obs, gen_by = capture(sim, dp.crane_ids, LEVEL, "diag", k)
        if isinstance(pref, QPreference):
            encs = {ob.crane_id: encode_observation(state, ob) for ob in obs}
            scores = {}
            for cid, enc in encs.items():
                scores.update({(cid, c): v
                               for c, v in learner.scores_for(enc).items()})
            pref.set_scores(scores)
        resn = resolver.resolve(sim, dp, gen_by)
        resolver.apply(sim, resn, gen_by)
        for r in resn.resolutions:
            if r.chosen_candidate_id is None:
                kinds["WAIT/none"] += 1
                continue
            gc = next(g for g in gen_by[r.crane_id].items
                      if g.candidate_id == r.chosen_candidate_id)
            kinds[gc.kind.name] += 1
        t_k = dp.time
        dp = sim.run_until_decision()
        raw = sim.cost.cut()
        cost = rc.cost_for(interval_start_s=t_k, interval_end_s=sim.now, raw=raw,
                           risk_max=_max_vessel_risk_state(state))
        costs.append(cost.total_normalized)
        for t, v in cost.contributions().items():
            terms[t] += v
        k += 1
    waits = [w / 60.0 for w in sim.kpis.wait_samples_s]
    jobs = list(sim.jobs.values())
    done = sum(1 for j in jobs if j.status.name == "DONE")
    return {"kinds": kinds, "total_cost": sum(costs), "terms": dict(terms),
            "mean_wait": fmean(waits) if waits else 0.0,
            "completion": done / max(1, len(jobs))}


def main():
    profile = build_integrated_profile()
    params = TerminalGenParams(n_external=40, n_vessels=2)
    dueling = CandidateDQNLearner.load(POC_MODEL)
    policies = {
        "BASELINE_VESSEL_WAIT": (BaselinePreference, None),
        "BASELINE_SPT (YR-039 사용)": (SPTPreference, None),
        "ServiceFirstSPT (진단)": (ServiceFirstSPT, None),
        "DuelingDQN (YR-039 승자)": (QPreference, dueling),
    }
    for label, seeds in (("test seed 320000 (사용자 표 재현)", [320_000]),
                         ("val 310000~310004 (5 seeds 평균)",
                          list(range(310_000, 310_005)))):
        print(f"\n{'=' * 72}\n[{label}]")
        for name, (factory, learner) in policies.items():
            agg_kinds, tc, mw, comp = Counter(), [], [], []
            terms = defaultdict(float)
            for s in seeds:
                r = run_counted(_sim(profile, s, params), factory(), learner)
                agg_kinds += r["kinds"]
                tc.append(r["total_cost"])
                mw.append(r["mean_wait"])
                comp.append(r["completion"])
                for t, v in r["terms"].items():
                    terms[t] += v / len(seeds)
            n = sum(agg_kinds.values())
            total = fmean(tc)
            imb = terms.get("imbalance", 0.0)
            tw = terms.get("truck_wait", 0.0)
            dist = "  ".join(f"{k}={v}({v / n:.0%})"
                             for k, v in agg_kinds.most_common())
            top3 = "  ".join(f"{t}={v:,.1f}" for t, v in
                             sorted(terms.items(), key=lambda kv: -kv[1])[:3])
            print(f"\n{name}")
            print(f"  total={total:,.1f}  imbalance={imb:,.1f}"
                  f"({imb / total:.1%})  truck_wait={tw:,.1f}"
                  f"  ex-imbalance total={total - imb:,.1f}")
            print(f"  mean_wait={fmean(mw):.3f}분  completion={fmean(comp):.1%}"
                  f"  top terms: {top3}")
            print(f"  actions: {dist}")


if __name__ == "__main__":
    main()
