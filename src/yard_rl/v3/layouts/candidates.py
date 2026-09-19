"""Opt-in candidate pruning for environments with explicit crane roles.

The legacy generator can fill its budget with high-score, infeasible work and
discard every feasible action. Keep the same scores and canonical tie-breaks,
but reserve the nonmandatory budget for feasible work first. The frozen world
and its default generator remain unchanged.
"""
from __future__ import annotations

from ..world.integrated.candidates import CandidateGenerator, GenCandidate


class FeasibleCandidateGenerator(CandidateGenerator):
    """Preserve mandatory work and at least one available feasible candidate."""

    def _prune(self, raw) -> list[GenCandidate]:
        # WAIT is appended by inherited generate(), outside this work budget.
        budget = self.k_max - 1
        mandatory = [candidate for candidate in raw if candidate.mandatory]
        rest = [candidate for candidate in raw if not candidate.mandatory]
        rest.sort(key=lambda candidate: (
            not candidate.feasible, -candidate.score,
        ) + self._order_key(candidate))
        kept = mandatory + rest[:max(0, budget - len(mandatory))]

        # Mandatory overflow must not hide every executable action. Candidate
        # sets already permit overflow to preserve mandatory work; add just one
        # best feasible action when necessary, without changing scores or masks.
        if not any(candidate.feasible for candidate in kept):
            best_feasible = next((candidate for candidate in rest
                                  if candidate.feasible), None)
            if best_feasible is not None:
                kept.append(best_feasible)
        return kept
