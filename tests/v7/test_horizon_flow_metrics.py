from pathlib import Path
import pytest


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts/v7'))
    import horizon_metrics
    return horizon_metrics


def row(flow, origin, finish, ready=None, yard=None):
    return dict(flow=flow, origin_s=origin, finish_s=finish,
                ready_s=ready, yard_done_s=yard)


def test_unreleased_vessel_order_is_not_removed_or_assigned_zero(metrics):
    r = [row('VESSEL_DISCHARGE', 10, 40, 20, 40),
         row('VESSEL_DISCHARGE', 10, None)]
    a = metrics.aggregate(r, 100)['VESSEL_DISCHARGE']
    assert a == dict(n=2, total_s=120., mean_s=60., unfinished=1,
                    not_ready=1, ready_n=1, ready_mean_s=20.)


def test_end_limits_future_exit_and_keeps_jobs_finishing_after_boundary(metrics):
    r = [row('GATE_IN', 10, 150, 20, 140), row('GATE_OUT', 120, None)]
    a = metrics.aggregate(r, 100)
    assert a['TRUCK']['n'] == 1
    assert a['TRUCK']['mean_s'] == 90
    assert a['TRUCK']['unfinished'] == 1
    assert 'GATE_OUT' not in a


def test_overall_is_weighted_by_orders_and_trucks_are_separate(metrics):
    r = [row('GATE_IN', 0, 20), row('GATE_IN', 0, 30),
         row('GATE_OUT', 0, 40), row('VESSEL_LOAD', 0, 90)]
    a = metrics.aggregate(r, 100)
    assert a['ALL_ORDERS']['mean_s'] == 45
    assert a['TRUCK']['mean_s'] == 30
    assert a['GATE_IN']['mean_s'] == 25


def test_negative_duration_fails_instead_of_hiding_timestamp_bug(metrics):
    with pytest.raises(ValueError, match='completion precedes origin'):
        metrics.duration(30, 20, 100)
    assert metrics.duration(None, None, 100) is None
    assert metrics.duration(100, None, 100) == 0
