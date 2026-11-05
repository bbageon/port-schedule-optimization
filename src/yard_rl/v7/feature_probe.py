"""Feature-eight semantic audit; shadow comparison does not change the live action."""
from collections import Counter
import copy

from .crane.features import crane_features
from .crane.policy import CranePolicy
from .evaluation_probe import EvaluationProbe
from .world.contract.schema import CandidateKind
from .world.domain.enums import JobStatus


class FeatureProbe(EvaluationProbe):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.shadow = None

    def before_select(self, sim, dp, gb, pref):
        if self.shadow is None:
            net = copy.deepcopy(pref.net)
            net.feature_mode = 'feasible'
            self.shadow = CranePolicy(net)
        for cid in dp.crane_ids:
            items = [g for g in gb[cid].items if g.feasible and g.job_ref is not None]
            n = sum(g.feasible and g.kind == CandidateKind.SERVE for g in gb[cid].items)
            if not items:
                continue
            old = crane_features(sim, cid, items[0], feature_mode='legacy')
            new = crane_features(sim, cid, items[0], feature_mode='feasible')
            assert items[0].feasible_serve_count == n
            self.counts['crane_decisions'] += 1
            self.counts['raw_count_different'] += n != sim.jobs.count(JobStatus.WAITING)
            self.counts['feature_different'] += old[7] != new[7]
            self.counts['legacy_saturated'] += old[7] == 1.
            self.counts['corrected_saturated'] += new[7] == 1.
            self.counts['max_feasible_serve'] = max(self.counts['max_feasible_serve'], n)
            pairs = [(cid, g) for g in items]
            keys = lambda p: [(not g.mandatory, rank) for (_,g),rank in zip(pairs, p.rank_many(sim,pairs))]
            a, b = keys(pref), keys(self.shadow)
            self.counts['shadow_top_changed'] += min(range(len(a)),key=a.__getitem__) != min(range(len(b)),key=b.__getitem__)

    def finish(self, mbt, records, episode):
        super().finish(mbt, records, episode)
        self.result['feature_audit'] = dict(self.counts,
            scope='legacy policy executed; corrected feature evaluated on the identical state and candidate set')
