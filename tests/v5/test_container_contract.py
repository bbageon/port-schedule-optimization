"""Identity failures must be visible before any PPO learning, not repaired by substitution."""
import copy
from dataclasses import fields
import json
import os
from types import SimpleNamespace as NS

import pytest

from yard_rl.v5.schema import Order, defer, relocate
from yard_rl.v5.stage.container_contract import (
    ContainerContractError, audit_container_plan, container_no,
    namespace_initial_inventory, require_container_plan,
)
from yard_rl.v5.stage.month import make_retarget
from yard_rl.v5.stage.orders import V3Announcer, orders_from_schedule


def entry(key='pickup', flow='GATE_OUT', target='A', block='B1', arrival=600):
    return dict(job_id=key, flow=flow, target=target, block=block,
                arrival_s=arrival, lead_s=60, travel_s=300, travel_base_s=300,
                exit_travel_s=60, size_ft40=False)


def built(schedule=None):
    return dict(day0=dict(scenarios={'B1': NS(containers={'A': NS(container_id='A')})}),
                schedule=[entry()] if schedule is None else schedule)


def vessel(work='LOAD', targets=('A',), key='ship', moves=1):
    return dict(key=key, block='B1', work=work, targets=targets, moves=moves,
                start_s=1200, cadence_s=120)


def test_v3_six_field_schema_is_preserved():
    from yard_rl.v3.schema import Order as V3Order
    assert [f.name for f in fields(Order)] == [f.name for f in fields(V3Order)]
    assert len(fields(Order)) == 6


@pytest.mark.parametrize('number', [None, '', ' ', 12])
def test_order_requires_nonempty_container_number(number):
    with pytest.raises(ValueError, match='con_no'):
        Order('visit', 0, 0, 60, 'B1', number)


def test_order_number_is_the_actual_retrieve_or_produced_container():
    schedule = [entry(), entry('delivery', 'GATE_IN', None)]
    orders, records = orders_from_schedule(dict(schedule=schedule))
    assert orders['pickup'].con_no == 'A'
    assert orders['delivery'].con_no == 'IN_delivery'
    assert all(r.gate_in_s is None for r in records.values())


@pytest.mark.parametrize('value', [None, '', 'CNpickup', 'B'])
def test_explicit_conflicting_number_is_not_silently_overwritten(value):
    with pytest.raises(ContainerContractError, match='engine container'):
        container_no(entry() | {'con_no': value})


def test_relocation_and_deferral_keep_container_identity():
    order = Order('delivery', 1, 0, 600, 'B1', 'IN_delivery')
    assert relocate(order, 'B2').con_no == order.con_no
    assert defer(order, 900).con_no == order.con_no


def test_block_local_initial_names_get_unique_stable_ids():
    data = dict(scenarios={b: NS(containers={'C1': NS(container_id='C1')},
                                jobs=[NS(target_container='C1')]) for b in ('B1', 'B2')},
                schedule=[entry(target='C1', block=b) for b in ('B1', 'B2')])
    namespace_initial_inventory(data)
    assert [e['target'] for e in data['schedule']] == ['B1-C1', 'B2-C1']
    assert data['scenarios']['B2'].jobs[0].target_container == 'B2-C1'
    assert data['scenarios']['B1'].containers['B1-C1'].container_id == 'B1-C1'
    namespace_initial_inventory(data)
    assert data['schedule'][0]['target'] == 'B1-C1'


def test_day_prefix_is_also_applied_to_the_produced_container():
    from yard_rl.v5.stage.month import DayPlan, DAY_S, build_month
    days = [DayPlan(i, 10, 'unit', 9901306 + i, i * DAY_S, 2) for i in range(2)]
    data = build_month(9900306, days=days)
    orders, _ = orders_from_schedule(data)
    for e in data['schedule']:
        assert orders[e['job_id']].con_no == container_no(e)
        if e['flow'] == 'GATE_IN':
            assert e['con_no'] == 'IN_' + e['job_id']
    names = [c for scn in data['day0']['scenarios'].values() for c in scn.containers]
    assert len(names) == len(set(names))


def test_valid_fixed_plan_is_read_only_and_deterministic():
    data = built([entry('delivery', 'GATE_IN', None, arrival=300),
                  entry(target='IN_delivery', arrival=600)])
    ships = {0: [vessel()]}
    before = copy.deepcopy(data)
    report = audit_container_plan(data, ships)
    require_container_plan(report)
    assert report['passed'] and not report['violations']
    assert report == audit_container_plan(data, ships) and data == before


def test_same_box_cannot_be_exited_twice_even_under_different_order_names():
    report = audit_container_plan(built([entry(), entry('second')]), {})
    assert report['violations']['duplicate_exits'] == 1
    with pytest.raises(ContainerContractError) as error:
        require_container_plan(report)
    assert error.value.report == report


def test_truck_and_ship_cannot_claim_the_same_box():
    report = audit_container_plan(built(), {0: [vessel()]})
    assert report['violations']['duplicate_exits'] == 1


@pytest.mark.parametrize('target,reason', [('UNKNOWN', 'missing_sources'),
                                         ('IN_late', 'exit_before_possible_arrival')])
def test_exit_requires_a_known_prior_source(target, reason):
    report = audit_container_plan(built([entry(target=target),
        entry('late', 'GATE_IN', None, arrival=2000)]), {})
    assert report['violations'][reason] == 1


def test_vessel_unload_supplies_a_fixed_named_box():
    discharge = vessel('DISCHARGE', key='unload')
    report = audit_container_plan(built([entry(target='IN_B1:J-unload-0000', arrival=2000)]),
                                  {0: [discharge]})
    assert report['passed']


def test_vessel_load_without_manifest_cannot_pass():
    report = audit_container_plan(built([]), {0: [vessel(targets=None, moves=3)]})
    assert report['violations'] == {'unbound_vessel_load_moves': 3,
                                    'unbound_vessel_load_streams': 1}


def test_missing_and_reserved_targets_are_never_replaced_by_another_box():
    sim = NS(stacks=NS(containers={'A': object(), 'B': object()}), jobs={})
    mbt = NS(blocks={'B1': sim})
    pick = make_retarget(1)
    assert pick(mbt, 'B1', entry()) == 'A'
    sim.jobs['owner'] = NS(target_container='A')
    assert pick(mbt, 'B1', entry()) is None
    sim.stacks.containers.pop('A')
    assert pick(mbt, 'B1', entry()) is None


def test_custom_retarget_hook_cannot_change_the_order():
    e = entry()
    ann = V3Announcer([e], retarget=lambda *args: 'B')
    ann.review(NS(), 540)
    assert ann.n_admitted == 0 and ann.n_retargeted == 0 and ann.n_skipped == 1
    assert ann.skips[0]['reason'] == 'CONTAINER_ID_CHANGED'
    assert e['target'] == 'A'


def test_real_default_input_is_rejected_before_runtime_bind(monkeypatch, tmp_path):
    from yard_rl.v5.ppo import continuous
    from yard_rl.v5.ppo.runtime import PPORuntime
    monkeypatch.setattr(continuous, 'code_stamp', lambda: {'pid': os.getpid(), 'test_only': True})
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid input must not create/bind the PPO world')
    monkeypatch.setattr(PPORuntime, 'bind', forbidden)
    out = tmp_path / 'invalid-input'
    with pytest.raises(ContainerContractError):
        continuous.run_continuous(output=out, n_days=3, load=10)
    report = json.loads((out / 'container_contract.json').read_text())
    status = json.loads((out / 'status.json').read_text())
    assert not report['passed'] and report['violations']['unbound_vessel_load_streams'] > 0
    assert status['updates'] == 0 and status['completed_days'] == 0
    assert status['snapshot_errors'] == []
    assert not (out / 'final.pt').exists()
