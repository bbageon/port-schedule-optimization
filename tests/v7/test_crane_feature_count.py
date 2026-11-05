"""Feature 8 counts this crane's feasible work, independently of the truck queue."""
from dataclasses import replace

import pytest
import torch

from yard_rl.v7.crane.features import crane_features
from yard_rl.v7.crane.policy import CraneNet, CranePolicy
from yard_rl.v7.world.contract.schema import CandidateKind
from yard_rl.v7.world.domain.enums import JobStatus
from test_candidate_recall import state, generate


def test_count_tracks_final_per_crane_candidates_and_preserves_other_inputs():
    sim, dp = state()
    gb = generate(sim, dp, 12)
    gc = next(g for g in gb['YC-A'].items if g.kind == CandidateKind.SERVE)
    a = crane_features(sim, 'YC-A', gc)
    b = crane_features(sim, 'YC-A', gc, feature_mode='feasible')
    assert gc.feasible_serve_count == 11
    assert a[7] == 1 and b[7] == 11/18
    assert a[:7] == b[:7] and a[8:] == b[8:]
    # Released vessel-like work must count, even though it is not WAITING.
    sim.jobs['T000'].status = JobStatus.RELEASED
    g = generate(sim, dp, None)['YC-A']
    expected = sum(x.kind == CandidateKind.SERVE and x.feasible for x in g.items)
    assert expected == 30
    assert all(x.feasible_serve_count == expected for x in g.items)


def test_missing_count_does_not_silently_fall_back_to_queue():
    sim, dp = state()
    gc = generate(sim, dp, None)['YC-A'].items[0]
    with pytest.raises(ValueError, match='generator-provided'):
        crane_features(sim, 'YC-A', replace(gc, feasible_serve_count=None), feature_mode='feasible')


def test_scalar_and_batch_use_same_corrected_feature():
    sim, dp = state()
    gb = generate(sim, dp, 12)
    torch.manual_seed(9913501)
    p = CranePolicy(CraneNet(feature_mode='feasible'))
    pairs = [('YC-A', g) for g in gb['YC-A'].items]
    for scalar, batch in zip([p.rank(sim,*x) for x in pairs], p.rank_many(sim,pairs)):
        assert scalar[::2] == batch[::2]
        assert scalar[1] == pytest.approx(batch[1], abs=1e-6)
