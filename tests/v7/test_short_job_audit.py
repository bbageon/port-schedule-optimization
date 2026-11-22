"""Isolated ranking changes must preserve the physical plan and existing tie keys."""
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def dispatch(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/v7"))
    import audit_short_job_rule as study
    return study


def test_rule_priority_and_duration_interventions(dispatch):
    sim = NS(cum_wait=lambda _: 1500)
    def candidate(jid, vessel, duration):
        ref = NS(job_id=jid, token=jid, is_vessel=vessel, is_external=not vessel)
        return NS(kind=dispatch.CandidateKind.SERVE, plan=NS(duration_s=duration), job_ref=ref,
                  mandatory=not vessel, candidate_id=int(vessel))
    truck = candidate("truck", False, 200)
    ship = candidate("ship", True, 100)
    def keys(guard, signal, competing=True):
        r = dispatch.RuleResolver("SF_SPT", guard, signal, 17)
        r.competition = {"C"} if competing else set()
        r.block_ids[id(sim)] = "B"
        return [r._pair_key(sim, "C", g) for g in (truck, ship)]
    base = keys("current", "exact")
    assert base[0] < base[1]
    assert keys("vessel", "exact", False) == base
    for guard in ("off", "vessel"):
        a, b = keys(guard, "exact")
        assert a > b and a[1:] == base[0][1:] and b[1:] == base[1][1:]
    plus, minus = keys("current", "noise_plus"), keys("current", "noise_minus")
    for i, hidden in enumerate(keys("current", "hidden")):
        assert hidden[2] == 0 and hidden[:2]+hidden[3:] == base[i][:2]+base[i][3:]
        assert (plus[i][2]+minus[i][2])/2 == pytest.approx(base[i][2])
        assert .7*base[i][2] <= plus[i][2] <= 1.3*base[i][2]
    assert (truck.plan.duration_s, ship.plan.duration_s) == (200, 100)
