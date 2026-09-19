"""Regression checks for feasible work lost behind blocked high-score work."""
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import pytest

from yard_rl.v3.layouts.candidates import FeasibleCandidateGenerator
from yard_rl.v3.world.contract.schema import CandidateKind
from yard_rl.v3.world.domain.enums import InformationLevel
from yard_rl.v3.world.integrated.candidates import CandidateGenerator, GenCandidate
from yard_rl.v3.world.integrated.jobplan import JobRef


def candidate(name, score, *, feasible=True, mandatory=False,
              kind=CandidateKind.SERVE):
    ref = JobRef(name, name, kind, None, None, ("C1",), True, False)
    return GenCandidate(0, kind, ref, None, mandatory, feasible,
                        None if feasible else "CRANE_ROLE", score)


def names(items):
    return [item.job_ref.job_id for item in items]


def test_blocked_high_scores_cannot_hide_available_work_and_legacy_is_unchanged():
    blocked = [candidate(f"blocked-{i:02d}", 100 + i, feasible=False)
               for i in range(11)]
    available = [candidate("available-low", -100), candidate("available-high", -10)]
    raw = blocked + available
    original = list(raw)
    legacy = CandidateGenerator()._prune(raw)
    kept = FeasibleCandidateGenerator()._prune(raw)
    assert len(legacy) == 11 and not any(item.feasible for item in legacy)
    assert names(legacy) == [f"blocked-{i:02d}" for i in range(10, -1, -1)]
    assert len(kept) == 11
    assert names(kept[:2]) == ["available-high", "available-low"]
    assert names(kept[2:]) == names(legacy[:9])
    assert raw == original  # Scores, masks, refs, and input order are untouched.
    assert all(any(item is original_item for original_item in raw) for item in kept)


@pytest.mark.parametrize("mandatory_count", [2, 3, 5])
def test_full_mandatory_budget_keeps_every_mandatory_plus_one_feasible(mandatory_count):
    mandatory = [candidate(f"urgent-{i}", 10, feasible=False, mandatory=True)
                 for i in range(mandatory_count)]
    raw = mandatory + [candidate("best", 2), candidate("other", 1)]
    kept = FeasibleCandidateGenerator(k_max=3)._prune(raw)
    assert kept[:mandatory_count] == mandatory
    assert names(kept[mandatory_count:]) == ["best"]
    assert len(kept) == mandatory_count + 1


def test_feasible_mandatory_overflow_needs_no_extra_candidate():
    mandatory = [candidate("urgent-blocked", 10, feasible=False, mandatory=True),
                 candidate("urgent-ready", -10, mandatory=True),
                 candidate("urgent-blocked-2", 20, feasible=False, mandatory=True)]
    assert FeasibleCandidateGenerator(k_max=3)._prune(
        mandatory + [candidate("optional", 100)]) == mandatory


def test_no_feasible_work_keeps_blocked_diagnostic_candidates():
    raw = [candidate("urgent", -100, feasible=False, mandatory=True),
           candidate("blocked-low", 1, feasible=False),
           candidate("blocked-best", 10, feasible=False),
           candidate("blocked-next", 9, feasible=False)]
    generator = FeasibleCandidateGenerator(k_max=4)
    assert generator._prune(raw) == CandidateGenerator(k_max=4)._prune(raw)
    assert names(generator._prune(raw)) == ["urgent", "blocked-best", "blocked-next"]
    assert all(item.mask_reason == "CRANE_ROLE" for item in generator._prune(raw))
    assert generator._prune([]) == []


def test_membership_keeps_legacy_score_and_canonical_tie_order():
    raw = [candidate("ready-b", 4), candidate("ready-a", 4),
           candidate("ready-top", 5), candidate("blocked", 100, feasible=False)]
    generator = FeasibleCandidateGenerator(k_max=3)
    for ordering in permutations(raw):
        assert names(generator._prune(ordering)) == ["ready-top", "ready-a"]
    all_feasible = [replace(item, feasible=True, mask_reason=None) for item in raw]
    assert generator._prune(all_feasible) == CandidateGenerator(k_max=3)._prune(all_feasible)


def test_generate_retains_canonical_ids_and_wait_after_mandatory_overflow(monkeypatch):
    generator = FeasibleCandidateGenerator(k_max=3, block_pre_rehandle=True)
    raw = [candidate("urgent-b", 20, feasible=False, mandatory=True),
           candidate("urgent-a", 10, feasible=False, mandatory=True),
           candidate("ready", -1)]
    sim = SimpleNamespace(now=123, fleet=SimpleNamespace(
        get=lambda cid: SimpleNamespace(idle=True, yielded=False)))
    monkeypatch.setattr(generator, "_serve", lambda *args: raw)
    monkeypatch.setattr(generator, "_reposition", lambda *args: [])
    wait = GenCandidate(0, CandidateKind.WAIT, None, None, False, True, None, 0)
    monkeypatch.setattr(generator, "_wait", lambda *args: wait)
    generated = generator.generate(sim, "C1", InformationLevel.PRE_ADVICE)
    assert names(generated.items[:-1]) == ["ready", "urgent-a", "urgent-b"]
    assert [item.candidate_id for item in generated.items] == list(range(4))
    assert generated.items[-1] == replace(wait, candidate_id=3)
    assert generated.items[0].feasible
