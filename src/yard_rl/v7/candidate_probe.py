"""Read-only recall audit against pre-pruning candidates at the same state."""
from collections import Counter
from time import perf_counter

from .evaluation_probe import EvaluationProbe
from .world.contract.schema import CandidateKind


def action_key(gc):
    return gc.kind.value, gc.job_ref


class CandidateRecallProbe(EvaluationProbe):
    def __init__(self):
        super().__init__()
        self.raw = {}
        self.counts = Counter()
        self.by_kind = Counter()
        self.audit_seconds = 0.

    def capture_candidates(self, sim, cid, raw, keep):
        self.raw[(id(sim), cid)] = tuple(raw)

    def before_select(self, sim, dp, generated, pref):
        start = perf_counter()
        best = {}
        for cid in dp.crane_ids:
            raw = self.raw.pop((id(sim), cid), ())
            eligible = [g for g in raw if g.feasible and g.job_ref is not None]
            kept = {action_key(g) for g in generated[cid].items if g.feasible and g.job_ref is not None}
            all_keys = {action_key(g) for g in eligible}
            assert kept <= all_keys
            mandatory = {action_key(g) for g in raw if g.mandatory}
            assert mandatory <= {action_key(g) for g in generated[cid].items}
            self.counts['crane_decisions'] += 1
            self.counts['eligible'] += len(all_keys)
            self.counts['retained'] += len(kept)
            self.counts['mandatory'] += len(mandatory)
            self.counts['max_eligible'] = max(self.counts['max_eligible'], len(all_keys))
            for gc in eligible:
                self.by_kind[(gc.kind.value, 'eligible')] += 1
                self.by_kind[(gc.kind.value, 'retained')] += action_key(gc) in kept
            if eligible:
                pairs = [(cid, g) for g in eligible]
                ranks = pref.rank_many(sim, pairs)
                i = min(range(len(eligible)), key=lambda i:
                        (not eligible[i].mandatory, ranks[i], eligible[i].kind.value))
                top = eligible[i]
                self.counts['with_action'] += 1
                self.counts['top_retained'] += action_key(top) in kept
                best[cid] = top.job_ref
        if len(best) > 1:
            self.counts['joint_checks'] += 1
            projected = sim.dry_run_commit(best)
            self.counts['independent_top_joint_conflicts'] += set(projected.plans) != set(best)
        self.audit_seconds += perf_counter() - start

    def finish(self, mbt, records, episode):
        super().finish(mbt, records, episode)
        self.result['recall'] = dict(self.counts,
            recall=self.counts['retained']/max(1, self.counts['eligible']),
            top_retention=self.counts['top_retained']/max(1, self.counts['with_action']),
            by_kind={f'{k}/{v}': n for (k,v),n in sorted(self.by_kind.items())},
            audit_seconds=self.audit_seconds,
            scope='generated pre-prune feasible actions; independent small-case oracle tested separately',
            joint_scope='conflicts among independently top-ranked crane actions; not pruning misses')
