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
    rebuilt = (krw.KRW_TRUCK_HOUR*sums['sum_work']+krw.KRW_TRUCK_OVER_HOUR*sums['work_over_3600'])/3600
    assert math.isclose(official, rebuilt, rel_tol=1e-12)
    assert sums['n'] == 3 and sums['censored'] == 1
    assert math.isclose(sums['over_1800'], (6900.-1800.)+(6400.-1800.))
    assert math.isclose(sum(sums['stages'].values()), sums['sum_tt'])


@pytest.fixture
def report(audit, monkeypatch):
    import report_comparison_audit as module
    return module


def fake_run(truck_s, over_s, gt_idle, ships=None, yc_s=0., rehandles=0):
    ships = ships or {'M1': dict(gt=100_000., idle_s=gt_idle*3600/100_000.)}
    trucks = dict(sum_tt=truck_s, over_3600=over_s, over_1800=over_s, n=1,
                  stages=dict(travel_in=truck_s, queue=0., service=0., exit=0.))
    audit = dict(trucks=trucks, yc_empty_travel_s=yc_s, rehandles=rehandles,
                 vessel_gt_idle_hours=gt_idle, ships_24h=ships)
    result = dict(c_wait=krw.KRW_TRUCK_HOUR*truck_s/3600+krw.KRW_TRUCK_OVER_HOUR*over_s/3600,
                  c_move=krw.yc_move_krw(yc_s), c_rehandle=krw.rehandle_krw(rehandles),
                  c_vessel=krw.KRW_VESSEL_GT_HOUR*gt_idle)
    result['phi_krw'] = sum(result.values())
    return dict(probe=dict(audit=audit), result=result)


def test_base_rescoring_reproduces_the_official_total(report):
    d = fake_run(7200., 1800., 500., yc_s=900., rehandles=7)
    assert math.isclose(report.cost(report.units(d)), d['result']['phi_krw'], rel_tol=1e-12)
    linear = report.cost(report.units(d), over=0.)
    assert math.isclose(d['result']['phi_krw']-linear, krw.KRW_TRUCK_OVER_HOUR*0.5)
    by_class = report.cost(report.units(d), vessel='by_class')
    assert math.isclose(by_class, d['result']['phi_krw'])   # medium ship uses the median rate


def test_break_even_matches_the_analytic_flip(report):
    a = report.units(fake_run(3600., 0., 2000.))   # 40,000 + 35.88 x 2,000 GT-h
    b = report.units(fake_run(7200., 0., 0.))      # 80,000
    found = report.break_even([a]*9, [b]*9, 'vessel')
    expected = 40_000./(krw.KRW_VESSEL_GT_HOUR*2000.)
    assert not found['reference']['better']
    assert math.isclose(found['nearest']['mean'], expected, rel_tol=1e-6)
    assert math.isclose(found['nearest']['majority'], expected, rel_tol=1e-6)


def test_term_split_adds_up_to_the_total_change(report):
    a, b = fake_run(9000., 1000., 300., yc_s=600.), fake_run(7200., 0., 400., yc_s=900.)
    split = report.term_split(a, b)
    total = 100.*(a['result']['phi_krw']/b['result']['phi_krw']-1.)
    assert math.isclose(sum(split.values()), total, rel_tol=1e-12)


def test_inventory_symbols_exist_in_code(report):
    check = report.check_inventory(report.inventory())
    assert check['entries'] >= 30 and check['symbols_missing'] == []


def test_tape_recorder_is_installed_once(audit):
    import yard_rl.v7.stage.episode as episode
    original = episode._CounterTape
    try:
        first = audit.install_tape_recorder()
        assert audit.install_tape_recorder() is first
        assert episode._CounterTape is first and issubclass(first, original)
    finally:
        episode._CounterTape = original
