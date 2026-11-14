"""YR-332-al: multi-container truck visits linked onto calibrated orders.

Input: hourly counts and every order attribute kept; orders of one visit share one gate time;
chassis capacity (two same-flow orders are both 20 ft); inbound stops before outbound stops.
Engine: stop 0 arrives at gate + travel; stop k at the previous stop's last completion + travel;
one gate exit for the visit; no release lands in a block's past; teacher branches replay.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest

from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.visit_link import validate_visit_link
from yard_rl.v7.world.integrated.multiblock import MultiBlockTerminal
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.world.integrated.yard_layout import terminal_layout

AK = Path('outputs/reports/yr332_visit_arrivals')
LINKS = Path('outputs/reports/yr332_visit_links/link-definition.json')
INFO = Path('outputs/reports/yr332_real_order_information/profile.json')
TIME_FIELDS = {'arrival_s', 'lead_s', 'reserved_s', 'eta_offset_s', 'gate_in_s'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def observed():
    return read(AK / 'demand-profiles.json')['profiles']['observed_hours_visit_minutes']


def linked():
    return {**observed(), 'visit_link': read(LINKS)['spec']}


def spec(**over):
    base = dict(method='observed_visit_patterns', size_rule='chassis_two_teu', order_rule='inbound_first',
                patterns=[dict(stops=[[0, 1]], count=5), dict(stops=[[1, 0]], count=5),
                          dict(stops=[[1, 1]], count=2)])
    return {**base, **over}


@pytest.mark.parametrize('bad', [
    dict(method='other'), dict(size_rule='none'), dict(order_rule='any'),
    dict(patterns=[dict(stops=[[0, 1]], count=1)]),
    dict(patterns=[dict(stops=[[0, 1]], count=1), dict(stops=[[1, 0]], count=1), dict(stops=[[0, 3]], count=1)]),
    dict(patterns=[dict(stops=[[0, 1]], count=1), dict(stops=[[1, 0]], count=1),
                   dict(stops=[[1, 0], [0, 1]], count=1)]),
    dict(patterns=[dict(stops=[[0, 1]], count=1), dict(stops=[[1, 0]], count=0)])])
def test_invalid_link_specs_fail(bad):
    with pytest.raises(ValueError):
        validate_visit_link(spec(**bad))
    validate_visit_link(spec())


def build(load=1600, seed=9920995, profile=None):
    return build_stage(load=load, seed=seed, profile=build_h21_profile(),
                       information_profile=read(INFO), demand_profile=profile or linked())


def test_input_keeps_orders_and_obeys_chassis_and_order_rules():
    base, new = build(profile=observed()), build()
    assert new['scenarios'] == base['scenarios'] and new['vessel_schedule'] == base['vessel_schedule']
    hours = lambda b: Counter(int(e['arrival_s'] // 3600) for e in b['schedule'])
    assert hours(new) == hours(base)
    visits = defaultdict(list)
    for a, b in zip(base['schedule'], new['schedule']):
        assert {k: v for k, v in a.items() if k not in TIME_FIELDS} == \
               {k: v for k, v in b.items() if k not in TIME_FIELDS | {'visit'}}
        assert int(a['arrival_s'] // 3600) == int(b['arrival_s'] // 3600)
        assert b['gate_in_s'] == b['arrival_s'] and 0 <= b['lead_s'] <= b['arrival_s']
        if 'visit' in b:
            visits[b['visit']['id']].append(b)
    meta = new['visit_links']
    assert meta['orders'] == len(new['schedule']) and sum(map(len, visits.values())) == meta['orders_in_multi_visits']
    assert .2 < meta['multi_order_share'] < .45 and meta['within_hour']['cycled_hours'] == 0
    sizes = {e['job_id']: e['size_ft40'] for e in new['schedule']}
    for vid, rows in visits.items():
        v = rows[0]['visit']
        plan = lambda d: {k: x for k, x in d.items() if k != 'stop'}
        assert len({e['arrival_s'] for e in rows}) == 1 and all(plan(e['visit']) == plan(v) for e in rows)
        flows = [s[1] for s in v['stops']]
        assert flows == sorted(flows, key=lambda f: f != 'GATE_IN')          # inbound stops first
        assert len(v['travel_s']) == len(v['stops']) and v['exit_travel_s'] > 0
        by_id = {e['job_id']: e for e in rows}
        for k, (blk, flow, ids) in enumerate(v['stops']):
            assert all(by_id[i]['block'] == blk and by_id[i]['flow'] == flow and by_id[i]['visit']['stop'] == k
                       for i in ids)
        for k in range(1, len(v['stops'])):
            a, b = v['stops'][k - 1][0], v['stops'][k][0]
            assert v['travel_s'][k] == (0.0 if a == b else round(terminal_layout().block_to_block_s(a, b), 3))
        n_in = [e for e in rows if e['flow'] == 'GATE_IN']
        if len(n_in) == 2:
            assert not any(sizes[e['job_id']] for e in n_in)                  # two 20 ft boxes
        assert len(n_in) <= 2 and len(rows) - len(n_in) <= 2
    assert build()['schedule'] == new['schedule']                               # deterministic


def test_two_outbound_orders_take_two_20ft_targets():
    from yard_rl.v7.world.domain.enums import ContainerSize
    built = build_stage(load=1600, seed=9920995, profile=build_h21_profile(),
                        information_profile=read(INFO), demand_profile=linked())
    boxes = {}
    for scn in built['scenarios'].values():
        boxes.update({k: c.size for k, c in scn.containers.items()})
    pairs = defaultdict(list)
    for e in built['schedule']:
        if 'visit' in e and e['flow'] == 'GATE_OUT':
            pairs[e['visit']['id']].append(boxes[e['target']])
    two = [p for p in pairs.values() if len(p) == 2]
    assert two and all(s == ContainerSize.FT20 for p in two for s in p)


@pytest.fixture
def captured(monkeypatch):
    seen = []
    original = MultiBlockTerminal.run

    def run(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(MultiBlockTerminal, 'run', run)
    return seen


def test_engine_releases_stops_in_order_and_exits_once(captured):
    from yard_rl.v7.stage.episode import run_episode
    ep = run_episode(load=600, seed=9920994, dispatcher='SF_SPT', information_profile=read(INFO),
                     demand_profile=linked(), candidate_cap=None, preserve_external_demand=True)
    mbt = captured[0]
    assert ep.admitted == 600 and ep.policy_exceptions == 0
    stats = mbt.visit_stats
    assert stats['late_releases'] == stats['late_exits'] == 0 and stats['released'] > 0
    times = {jid: (bid, tt) for bid, sim in mbt.blocks.items() for jid, tt in sim.time_ledger.records.items()}
    checked = exits = 0
    for vid, st in mbt.visits.items():
        done = []
        for k, (blk, _, ids) in enumerate(st['stops']):
            for i in ids:
                bid, tt = times[i]
                assert bid == blk and tt.gate_in == st['gate_in']
                if tt.block_arrival is None:
                    continue
                expect = (st['gate_in'] + st['travel'][0] if k == 0 else
                          max(times[j][1].job_done for j in st['stops'][k - 1][2]) + st['travel'][k])
                assert abs(tt.block_arrival - expect) < 1e-6, (vid, k)
                checked += 1
            done.append(all(times[i][1].job_done is not None for i in ids))
        outs = {times[i][1].gate_out for _, _, ids in st['stops'] for i in ids}
        if all(done):
            last = max(times[i][1].job_done for i in st['stops'][-1][2])
            assert outs == {last + st['exit']}
            exits += 1
        else:
            assert outs <= {None, st['gate_out']}
    assert checked > 0 and exits > 0


def test_unlinked_visits_have_no_terminal_state(captured):
    from yard_rl.v7.stage.episode import run_episode
    ep = run_episode(load=300, seed=9920993, dispatcher='SF_SPT', information_profile=read(INFO),
                     demand_profile=observed(), candidate_cap=None, preserve_external_demand=True)
    assert ep.admitted == 300 and captured[0].visits == {}


def test_teacher_branches_replay_linked_trajectory():
    import torch
    from yard_rl.v7.crane.policy import CraneNet, add_store_input
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.manual_seed(9920992)
    net = add_store_input(CraneNet(feature_mode='feasible'))
    ep = run_episode(load=500, seed=9920992, dispatcher='RL_CRANE', crane_net=net,
                     information_profile=read(INFO), demand_profile=linked(),
                     crane_budget=RolloutBudget(max_labels=4, identity_checks=0), horizon_s=1800.,
                     workers=1, crane_sampling='coverage', crane_alternative_contract='reachable',
                     candidate_cap=None, reuse_rehandle_slots=True, decision_probe_shortcircuit=True,
                     preserve_external_demand=True)
    c = ep.crane_stats
    assert ep.admitted == 500 and ep.policy_exceptions == c['hook_errors'] == c['factual_mismatch'] == 0
    assert c['force_failed'] == 0 and c['labeled'] > 0
