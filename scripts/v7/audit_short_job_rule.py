"""Paired rule-only counterfactuals on the frozen vessel-turnaround holdout."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from global_policy_inputs import RUNTIME, bundle
from yard_rl.v7.dispatch import make_preference
from yard_rl.v7.schema.lifecycle import censored_gate_to_job_done_s
from yard_rl.v7.stage.episode import run_episode
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.integrated.baselines import ServiceFirstSPTPreference
from yard_rl.v7.world.integrated.resolver import CentralResolver


OUT = Path("outputs/reports/yr332_short_job_audit")
REFERENCE = Path("outputs/reports/yr332_vessel_turnaround/evaluation")
DAY = 86400
LOADS = (3500, 7500, 15000)
RULES = ("SF_SPT", "FCFS_ARRIVAL", "NEAREST", "VESSEL_FCFS", "LWKR")
ARMS = [(rule, guard, "exact") for rule in RULES for guard in ("current", "vessel")]
ARMS += [(rule, "off", "exact") for rule in RULES[:2]]
ARMS += [("SF_SPT", "current", signal) for signal in ("noise_plus", "noise_minus", "hidden")]
ARMS += [("SF_SPT", "off", "hidden")]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


@lru_cache(maxsize=100000)
def error_draw(seed, block, crane, kind, job):
    key = f"{seed}|{block}|{crane}|{kind}|{job}".encode()
    bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return 0.3 * (2 * bits / 2**64 - 1)


class RuleResolver(CentralResolver):
    collect = False

    def __init__(self, rule, guard, signal, seed):
        preference = ServiceFirstSPTPreference() if rule == "SF_SPT" else make_preference(rule, seed=seed)
        super().__init__(preference)
        self.guard, self.signal, self.seed = guard, signal, seed
        self.block_ids = {}
        self.counts = Counter()

    def _pair_key(self, sim, crane, gc, rank=None):
        key = super()._pair_key(sim, crane, gc, rank)
        priority = key[0]
        if self.guard == "off":
            priority = 1
        if self.guard == "vessel" and crane in self.competition and self.is_vessel_service(gc):
            priority = -1
        if self.signal == "exact" or gc.plan is None:
            return (priority,) + key[1:]
        # Only the duration component of ServiceFirstSPTPreference changes.
        duration = 0.
        if self.signal != "hidden":
            delta = error_draw(self.seed, self.block_ids[id(sim)], crane, gc.kind.value, gc.job_ref.job_id)
            duration = key[2] * (1 + delta if self.signal == "noise_plus" else 1 - delta)
        return (priority, key[1], duration) + key[3:]

    @staticmethod
    def is_vessel_service(gc):
        return gc.kind == CandidateKind.SERVE and gc.job_ref.is_vessel

    def resolve(self, sim, decision, gen_by_crane):
        self.competition = set()
        for crane in decision.crane_ids:
            items = gen_by_crane[crane].items
            if (any(g.feasible and g.mandatory for g in items)
                    and any(g.feasible and self.is_vessel_service(g) for g in items)):
                self.competition.add(crane)
        result = super().resolve(sim, decision, gen_by_crane)
        self.last_assignment = {}
        for row in result.resolutions:
            items = gen_by_crane[row.crane_id].items
            chosen = (items[row.chosen_candidate_id] if row.chosen_candidate_id is not None
                      else next(g for g in items if g.kind == CandidateKind.WAIT))
            self.last_assignment[row.crane_id] = chosen.candidate_id
            if sim.clock <= DAY and row.crane_id in self.competition:
                self.counts["opportunities"] += 1
                self.counts["vessel" if self.is_vessel_service(chosen) else
                            "urgent_truck" if chosen.mandatory else "other"] += 1
        return result


class TruckProbe:
    def __init__(self, resolver):
        self.resolver = resolver

    def start(self, built, mbt):
        self.flows = {x["job_id"]: x["flow"] for x in built["schedule"]}
        self.resolver.block_ids = {id(sim): bid for bid, sim in mbt.blocks.items()}

    def decision(self, *args):
        pass

    def finish(self, mbt, records, episode):
        by_flow = defaultdict(list)
        pending = Counter()
        for job, rec in records.items():
            t = censored_gate_to_job_done_s(rec, DAY)
            if t is None or rec.gate_in_s > DAY:
                continue
            flow = self.flows[job]
            by_flow[flow].append(t / 60)
            pending[flow] += rec.job_done_s is None or rec.job_done_s > DAY
        self.metrics = {k: dict(n=len(v), average=mean(v), p90=float(np.quantile(v, .9)),
                                p99=float(np.quantile(v, .99)), maximum=max(v), pending=pending[k])
                        for k, v in by_flow.items()}
        assert sum(map(len, by_flow.values())) == episode.admitted


def run(case):
    rule, guard, signal, day, load = case
    path = OUT/"runs"/f"{rule}-{guard}-{signal}-{day}-{load}.json"
    if path.exists():
        return
    kwargs = bundle().case("holdout", day, load)
    resolver = RuleResolver(rule, guard, signal, kwargs["seed"])
    probe = TruckProbe(resolver)
    result = run_episode(**kwargs, dispatcher=rule, joint_resolver=resolver,
                         evaluation_probe=probe, vessel_drain_s=5*DAY, **RUNTIME)
    row = result.as_dict()
    assert not (row["policy_exceptions"] or row["txn_failed"]) and row["admitted"] == load
    ref = json.loads((REFERENCE/f"rule-0-{day}-{load}.json").read_text(encoding="utf8"))
    for k in ("demand_digest", "quay_manifest_digest", "n_vessels"):
        assert row[k] == ref[k], (case, k)
    if rule == "SF_SPT" and guard == "current" and signal == "exact":
        for k in ("phi_krw","c_wait","c_vessel","c_move","c_rehandle","vessel_calls",
                  "mean_gate_to_job_done_s","n_job_censored","p99_gate_to_job_done_s"):
            assert row[k] == ref[k], (case, k)
    assert math.isclose(sum(v["average"]*v["n"] for v in probe.metrics.values())/load,
                        row["mean_gate_to_job_done_s"]/60)
    assert sum(v["pending"] for v in probe.metrics.values()) == row["n_job_censored"]
    write(path, dict(case=case, seed=kwargs["seed"], trucks=probe.metrics,
                     competing_choices=dict(resolver.counts), result=row))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()
    cases = [(*arm, d, n) for n in (3500, 15000, 7500) for d in range(3) for arm in ARMS]
    if args.smoke:
        run(("SF_SPT", "current", "exact", 0, 3500))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for _ in pool.map(run, cases):
                pass
