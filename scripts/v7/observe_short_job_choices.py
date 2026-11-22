"""Observe one frozen date at each load, before applying its unchanged dispatch."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from audit_short_job_rule import DAY, LOADS, OUT, REFERENCE, RUNTIME, CandidateKind, RuleResolver, TruckProbe, bundle, json, run_episode, write


class Observer(RuleResolver):
    def __init__(self, seed):
        super().__init__("SF_SPT", "current", "exact", seed)
        self.choices = Counter()

    def resolve(self, sim, decision, by_crane):
        resolution = super().resolve(sim, decision, by_crane)
        if sim.clock > DAY or len(decision.crane_ids) != 1:
            return resolution
        crane, = decision.crane_ids
        items = by_crane[crane].items
        chosen = items[self.last_assignment[crane]]
        if chosen.kind != CandidateKind.SERVE:
            return resolution
        self.choices["single_crane_service"] += 1
        same_tier = [g for g in items if g.feasible and g.kind == CandidateKind.SERVE and g.mandatory == chosen.mandatory]
        if len(same_tier) < 2:
            return resolution
        self.choices["at_least_two_in_tier"] += 1
        durations = [g.plan.duration_s for g in same_tier]
        quickest = min(durations)
        self.choices["different_durations"] += max(durations) > quickest
        self.choices["chosen_is_shortest"] += chosen.plan.duration_s == quickest
        self.choices["tied_shortest"] += durations.count(quickest) > 1
        # The rule with duration removed retains the original secondary keys and admission check.
        def fallback(g):
            key = self._pair_key(sim, crane, g)
            return key[:2]+key[3:]
        for candidate in sorted(same_tier,key=fallback):
            if candidate.candidate_id == chosen.candidate_id:
                break
            if crane in sim.dry_run_commit({crane:candidate.job_ref}).plans:
                self.choices["different_choice_without_duration"] += 1
                break
        flows = {sim.jobs[g.job_ref.job_id].flow.value for g in same_tier}
        if {"GATE_IN","GATE_OUT"} <= flows:
            self.choices["in_and_out_present"] += 1
            self.choices["chosen_"+sim.jobs[chosen.job_ref.job_id].flow.value] += 1
        return resolution


def observe(load):
    kwargs=bundle().case("holdout",0,load)
    resolver=Observer(kwargs["seed"])
    result=run_episode(**kwargs,dispatcher="SF_SPT",joint_resolver=resolver,evaluation_probe=TruckProbe(resolver),
                       vessel_drain_s=5*DAY,**RUNTIME).as_dict()
    ref=json.loads((REFERENCE/f"rule-0-0-{load}.json").read_text(encoding="utf8"))
    assert all(result[k]==ref[k] for k in ("phi_krw","vessel_calls","mean_gate_to_job_done_s","demand_digest"))
    write(OUT/"decisions"/f"{load}.json",dict(load=load,day=0,seed=kwargs["seed"],same_outcome=True,
                                               counts=dict(resolver.choices)))


if __name__ == "__main__":
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(observe,LOADS))
