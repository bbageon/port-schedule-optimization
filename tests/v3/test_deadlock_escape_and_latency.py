"""The stall fix (YR-317-k) and the online-latency measurement (YR-317-e).

The stall observed in the 80-run evaluation was not a missing escape action: the
generator produced feasible escape repositions and the candidate budget then threw
them away because infeasible SERVE candidates scored higher. These tests pin that
mechanism, the opt-in wiring, and the fact that the default path is unchanged.
"""
import pytest

from yard_rl.v3.layouts.candidates import (FeasibleCandidateGenerator,
                                           FeasibleFirstCandidateGenerator)
from yard_rl.v3.stage.month import plan_days
from yard_rl.v3.stage.month_run import run_month
from yard_rl.v3.world.contract.schema import CandidateKind
from yard_rl.v3.world.integrated.candidates import CandidateGenerator, GenCandidate
from yard_rl.v3.world.integrated.jobplan import JobRef


def candidate(name, score, *, feasible=True, mandatory=False, kind=CandidateKind.SERVE):
    ref = JobRef(name, name, kind, None, None, ("C1",), True, False)
    return GenCandidate(0, kind, ref, None, mandatory, feasible,
                        None if feasible else "CRANE_INTERFERENCE", score)


def test_layout_alias_is_the_shared_implementation():
    """One implementation serves both yards, and it lives outside the cloned
    engine so `test_world_clone` keeps the copy byte-identical to its origin."""
    assert FeasibleCandidateGenerator is FeasibleFirstCandidateGenerator
    assert FeasibleFirstCandidateGenerator.__module__ == 'yard_rl.v3.layouts.candidates'
    assert issubclass(FeasibleFirstCandidateGenerator, CandidateGenerator)


def test_escape_reposition_survives_a_budget_full_of_blocked_serves():
    """The captured Y17 stall in miniature: every SERVE is blocked by crane
    interference and outscores the three feasible escape repositions."""
    blocked = [candidate(f"serve-{i:02d}", 100 + i, feasible=False) for i in range(174)]
    escapes = [candidate(f"REPO:YC-W:{bay}", -1000.0 + bay, kind=CandidateKind.REPOSITION)
               for bay in (4, 5, 6)]
    raw = blocked + escapes

    legacy = CandidateGenerator()._prune(list(raw))
    assert not any(item.feasible for item in legacy), 'legacy keeps only blocked work'

    kept = FeasibleFirstCandidateGenerator()._prune(list(raw))
    executable = [item for item in kept if item.feasible]
    # Feasible work comes first; inside it the existing score order still decides.
    assert [item.job_ref.job_id for item in executable] == ['REPO:YC-W:6', 'REPO:YC-W:5', 'REPO:YC-W:4']
    assert len(kept) == CandidateGenerator().k_max - 1, 'budget size is unchanged'
    assert all(item in raw for item in kept), 'no candidate is rewritten'


def test_raising_the_budget_alone_also_reveals_them():
    """Confirms the escapes were pruned, not missing: the probe used k_max=200."""
    raw = ([candidate(f"serve-{i:02d}", 100 + i, feasible=False) for i in range(174)]
           + [candidate("REPO:YC-W:4", -996.0, kind=CandidateKind.REPOSITION)])
    assert not any(c.feasible for c in CandidateGenerator()._prune(list(raw)))
    assert any(c.feasible for c in CandidateGenerator(k_max=200)._prune(list(raw)))


def test_run_month_rejects_an_unknown_pruning_mode():
    with pytest.raises(ValueError, match='candidate_pruning'):
        run_month(seed=99194001, arm='NO_REALLOC', days=plan_days(99194001, (10,)),
                  candidate_pruning='whatever')


def test_default_reproduces_itself_and_the_fix_is_a_declared_engine_change():
    """The default stays byte-identical so the 80-run evidence reproduces. The fix is
    NOT a no-op outside deadlocks: preferring feasible work changes the candidate set
    whenever the budget was full, so results from the two modes are not comparable and
    the opt-in is recorded on the result."""
    days = plan_days(99194001, (12,))
    legacy = run_month(seed=99194001, arm='NO_REALLOC', days=days)
    again = run_month(seed=99194001, arm='NO_REALLOC', days=days)
    fixed = run_month(seed=99194001, arm='NO_REALLOC', days=days,
                      candidate_pruning='feasible_first')
    assert legacy.candidate_pruning == 'legacy' and fixed.candidate_pruning == 'feasible_first'
    assert [d.phi_krw for d in legacy.days] == [d.phi_krw for d in again.days]
    assert legacy.online_latency == {} and fixed.online_latency == {}
    assert fixed.policy_exceptions == 0
    assert fixed.skipped == 0 and fixed.admitted == legacy.admitted


def test_latency_measurement_is_opt_in_and_reports_percentiles():
    days = plan_days(99194001, (12,))
    off = run_month(seed=99194001, arm='RL_TIME', days=days,
                    seller_net=None, buyer_net=None)
    on = run_month(seed=99194001, arm='RL_TIME', days=days,
                   seller_net=None, buyer_net=None, measure_latency=True)
    assert off.online_latency == {}
    summary = on.online_latency
    assert summary['decided_epochs'] > 0
    for key in ('mean_s', 'p50_s', 'p90_s', 'p95_s', 'p99_s', 'max_s'):
        assert summary[key] >= 0.0
    assert summary['p50_s'] <= summary['p95_s'] <= summary['max_s']
    assert summary['over_60s'] == 0
    assert 'simulation advancement' in summary['scope']
    # Timing must not change the simulated outcome.
    assert [d.phi_krw for d in off.days] == [d.phi_krw for d in on.days]
