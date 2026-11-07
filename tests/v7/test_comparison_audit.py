"""YR-332-t audit arithmetic: censored stage split and exact truck-term sums."""
from pathlib import Path
import math

import pytest

from yard_rl.v7.reward import krw
from yard_rl.v7.reward.phi import terminal_cost_krw
from yard_rl.v7.schema.record import ExecutionRecord


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v7'))
    import audit_comparison_validity as module
    return module


def rec(key, a=None, b=None, s=None, c=None, o=None):
    r = ExecutionRecord(doc_key=key)
    r.gate_in_s, r.block_in_s, r.service_start_s, r.job_done_s, r.gate_out_s = a, b, s, c, o
    return r


def test_complete_truck_splits_into_four_stages(audit):
    s = audit.stage_split(rec('Y01:D-1', 100., 400., 700., 900., 1300.))
    assert (s['travel_in'], s['queue'], s['service'], s['exit']) == (300., 300., 200., 400.)
    assert s['total'] == 1200. and not s['censored'] and not s['gap']


def test_unfinished_truck_is_censored_in_the_stage_in_progress(audit):
    s = audit.stage_split(rec('Y01:D-2', 86000., 86300.))
    assert (s['travel_in'], s['queue'], s['service'], s['exit']) == (300., 100., 0., 0.)
    assert s['total'] == 400. and s['censored'] and not s['gap']
    late = audit.stage_split(rec('Y01:D-3', 86000., 86100., 86200., 86300., 86500.))
    assert (late['exit'], late['total'], late['censored']) == (100., 400., True)


def test_missing_middle_stamp_is_reported_as_gap(audit):
    s = audit.stage_split(rec('Y01:D-4', 0., None, 50., 80., 100.))
    assert s['gap'] and math.isclose(s['total'], 100.)


def test_truck_sums_reproduce_the_official_truck_term(audit):
    records = {'Y01:D-1': rec('Y01:D-1', 0., 300., 900., 1200., 1500.),
               'Y02:D-2': rec('Y02:D-2', 100., 400., 5000., 5400., 7000.),
               'Y03:D-3': rec('Y03:D-3', 80000., 80300.),
               'Y04:D-4': rec('Y04:D-4')}
    flow = {k: 'GATE_OUT' for k in records}
    sums = audit.truck_sums(records, flow)
    official = terminal_cost_krw(records, end_s=audit.END_S).wait
    rebuilt = (krw.KRW_TRUCK_HOUR*sums['sum_tt']+krw.KRW_TRUCK_OVER_HOUR*sums['over_3600'])/3600
    assert math.isclose(official, rebuilt, rel_tol=1e-12)
    assert sums['n'] == 3 and sums['censored'] == 1
    assert math.isclose(sums['over_1800'], (6900.-1800.)+(6400.-1800.))
    assert math.isclose(sum(sums['stages'].values()), sums['sum_tt'])


def test_tape_recorder_is_installed_once(audit):
    import yard_rl.v7.stage.episode as episode
    original = episode._CounterTape
    try:
        first = audit.install_tape_recorder()
        assert audit.install_tape_recorder() is first
        assert episode._CounterTape is first and issubclass(first, original)
    finally:
        episode._CounterTape = original
