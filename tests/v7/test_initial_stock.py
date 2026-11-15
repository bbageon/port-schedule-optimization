"""Preserve cargo requests and physical starting stock, including failed repairs."""
import copy
from collections import Counter

import pytest

from yard_rl.v7.stage.initial_stock import InitialStockError, preserve_initial_stock
from yard_rl.v7.stage.vessels import build_diurnal_v3
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.world.integrated.terminal_stream import TerminalStreamParams


def build(seed, mode='preserve'):
    return build_diurnal_v3(build_h21_profile(), seed, load=15000,
        params=TerminalStreamParams(load_4h=15000, gate_out_share=.60), initial_stock_mode=mode)


def check(built):
    for bid, scn in built['scenarios'].items():
        targets = [e['target'] for e in built['schedule'] if e['block'] == bid and e['target']]
        targets += [j.target_container for j in scn.jobs if j.target_container]
        assert len(targets) == len(set(targets))
        assert set(targets) <= set(scn.containers)
        piles = {}
        for c in scn.containers.values():
            piles.setdefault((c.bay, c.row), []).append(c)
        for members in piles.values():
            assert sorted(c.tier for c in members) == list(range(1, len(members) + 1))
            assert len({c.size for c in members}) == 1
            assert len(members) <= build_h21_profile().block.tier_max
    assert all(e['flow'] == e['requested_flow'] for e in built['schedule'])


@pytest.mark.parametrize('seed', [9934102, 9934112, 9934118])
def test_repaired_supply_preserves_inventory_requests_and_vessels(seed):
    old, new = build(seed, 'legacy'), build(seed)
    check(new)
    assert new['flow_fallbacks_total'] == 0
    assert new['initial_stock']['repaired'] == old['flow_fallbacks_total']
    assert Counter(c.size for s in old['scenarios'].values() for c in s.containers.values()) == Counter(
        c.size for s in new['scenarios'].values() for c in s.containers.values())
    for b, s in old['scenarios'].items():
        assert s.jobs == new['scenarios'][b].jobs
        assert s.vessels == new['scenarios'][b].vessels
        protected = {j.target_container for j in s.jobs if j.target_container}
        protected |= {e['target'] for e in old['schedule'] if e['block'] == b and e['target']}
        assert all(s.containers[c] == new['scenarios'][b].containers[c] for c in protected)
    for a, b in zip(old['schedule'], new['schedule']):
        assert {k: v for k, v in a.items() if k not in ('flow','target','fallback_reason')} == {
            k: v for k, v in b.items() if k not in ('flow','target','fallback_reason')}


def test_no_shortage_is_identical_and_repair_is_repeatable():
    p = build_h21_profile()
    a = build_diurnal_v3(p, 9934100, load=3500, initial_stock_mode='legacy')
    b = build_diurnal_v3(p, 9934100, load=3500)
    for key in ('schedule', 'scenarios', 'fleet', 'vessel_schedule'):
        assert a[key] == b[key]
    assert build(9934118) == build(9934118)


def test_infeasible_repair_is_explicit_and_atomic():
    from dataclasses import replace
    from yard_rl.v7.world.integrated.scenario_gen import generate_terminal_scenario, TerminalGenParams
    p = build_h21_profile()
    scenario = generate_terminal_scenario(p, 17, TerminalGenParams(n_external=1, n_vessels=0))
    scns = {'only': replace(scenario, containers={}, jobs=[])}
    rows = [dict(job_id='request', block='only', flow='GATE_IN', requested_flow='GATE_OUT',
                 target=None, fallback_reason='no_free_target')]
    before = copy.deepcopy((scns, rows))
    with pytest.raises(InitialStockError, match='No uncommitted') as caught:
        preserve_initial_stock(scns, rows, p, 7)
    assert caught.value.report['unresolved'] == 1
    assert (scns, rows) == before
    with pytest.raises(InitialStockError, match='exceed'):
        preserve_initial_stock(scns, rows, p, 7, mode='strict')
    with pytest.raises(ValueError, match='initial_stock_mode'):
        preserve_initial_stock(scns, rows, p, 7, mode='typo')
