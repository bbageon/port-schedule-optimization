"""Training alternatives certified by the unchanged operational resolver.

Search is deliberately bounded to promoting one shared input score. A rejected
alternative is not a proof of global unreachability by every possible network.
No candidate truncation, priority override, or direct assignment is performed.
"""
from __future__ import annotations

import math

from ..world.integrated.baselines import ResolverPolicy
from .features import crane_features
from .policy import float32_key


def candidate_identity(gc):
    ref = gc.job_ref
    return (gc.kind.value, ref.job_id if ref is not None else None,
            getattr(gc, 'defer_until', None))


def assignment_signature(assignment):
    return tuple((cid, *candidate_identity(gc)) for cid, gc in sorted(assignment.items()))


def input_key(pref, sim, crane, gc):
    net = pref.net
    return float32_key(crane_features(sim, crane, gc,
        feature_mode=net.feature_mode, include_flow=net.include_flow,
        count_transform=net.count_transform))


def reachable_alternative(pref, sim, dp, gb, choices, preferred):
    """Try the sampled pair first, then other alternatives at this same decision.

    Restore the live preference even on errors. The only intervention is a
    single shared feature score, so any induced joint change is recorded too.
    """
    if pref.force_once or pref.score_once:
        raise ValueError('teacher search requires an unmodified live preference')
    crane, picked, first = preferred
    same_crane = next(cands for cid, _, cands in choices if cid == crane)
    alternatives = [first] + sorted((g for g in same_crane if g is not first),
        key=lambda g: (pref.rank(sim, crane, g), candidate_identity(g)))
    pairs = [(cid, gc) for cid, batch in sorted(gb.items())
             for gc in batch.items if gc.feasible and gc.job_ref is not None]
    scores = [float(r[1]) for r in pref.rank_many(sim, pairs)]
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError('non-finite candidate scores')
    promoted = min(scores) - max(1.0, max(scores) - min(scores))
    attempts = []
    picked_key = input_key(pref, sim, crane, picked)
    for alt in alternatives:
        key = input_key(pref, sim, crane, alt)
        if key == picked_key:
            attempts.append(dict(candidate=candidate_identity(alt), reason='same_input_as_picked'))
            continue
        try:
            pref.score_once[key] = promoted
            assignment = ResolverPolicy(pref).decide(sim, dp, gb)
        finally:
            pref.score_once.clear()
        selected = assignment.get(crane)
        accepted = selected is not None and candidate_identity(selected) == candidate_identity(alt)
        attempts.append(dict(candidate=candidate_identity(alt),
                             reason='accepted' if accepted else 'not_selected_by_resolver'))
        if accepted:
            return alt, dict(input_key=key, score=promoted,
                             assignment=assignment_signature(assignment)), attempts
    return None, None, attempts
