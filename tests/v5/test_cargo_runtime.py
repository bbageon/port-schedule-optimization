"""Actual fixed cargo admission, readiness, cross-block routing and accounting."""
from types import SimpleNamespace as NS

import pytest

from yard_rl.v5.stage.cargo_input import restore_input
from yard_rl.v5.stage.cargo_runtime import CargoBlock, CargoTerminal
from yard_rl.v5.stage.container_contract import ContainerContractError
from yard_rl.v5.stage.episode import _rule_policy, INFO_LEVEL
from yard_rl.v5.stage.fixed_seed import build_fixed_seed
from yard_rl.v5.stage.month import DayPlan
from yard_rl.v5.stage.month_engine import inject_vessel
from yard_rl.v5.stage.orders import V3Announcer, orders_from_schedule
from yard_rl.v5.world.domain.enums import JobStatus
from yard_rl.v5.world.integrated.profiles import build_h21_profile
from yard_rl.v5.world.integrated.terminal_stream import ensure_time_ledger
from yard_rl.v5.world.integrated.yard_layout import terminal_layout


def entry(jid, flow, at, bid='Y01'):
    return dict(job_id=jid, flow=flow, target=None if flow=='GATE_IN' else 'OLD',
        con_no='IN_'+jid if flow=='GATE_IN' else 'OLD', block=bid,
        arrival_s=at, lead_s=at, travel_s=300, travel_base_s=300,
        size_ft40=False, exit_travel_s=60, day=0)


def make_world(*, ship=False, incoming=300, outgoing=60):
    days = [DayPlan(0,2,'unit',9901306,0,1)]
    schedule = [entry('inbound','GATE_IN',incoming)]
    if not ship:
        schedule.append(entry('outbound','GATE_OUT',outgoing))
    data = dict(days=days, month_end_s=86400, lead_mode='DIST', schedule=schedule,
        day0=dict(scenarios={b:NS(containers={}) for b in ('Y01','Y02')}))
    vessels = {0:[dict(key='ship', work='LOAD', block='Y01', moves=1,
                       start_s=60, cadence_s=120)]} if ship else {}
    document,_ = build_fixed_seed(data, vessels, seed=9900306)
    restored = restore_input(document, seed=9900306, days=days, lead_mode='DIST')
    profile = build_h21_profile()
    blocks = {}
    for bid, scenario in restored['day0']['scenarios'].items():
        scenario.horizon_s, scenario.drain_window_s = 3600, 120
        sim = ensure_time_ledger(CargoBlock(profile, scenario))
        sim.info_level = INFO_LEVEL
        blocks[bid] = sim
    terminal = CargoTerminal(blocks, document=document, layout=terminal_layout(),
                              extra_review_epochs=tuple(range(0,3601,60)))
    terminal.orders, terminal.records = orders_from_schedule(restored)
    announcer = V3Announcer(restored['schedule'], resolve_entry=terminal.resolve_entry)
    return terminal, announcer, document, days


@pytest.mark.parametrize('ship', [False, True])
def test_pending_cargo_served_only_after_physical_storage(ship):
    terminal, announcer, document, _ = make_world(ship=ship)
    policy, errors = _rule_policy('SF_SPT', seed=9900306)
    def review(m,t):
        announcer.review(m,t)
        if ship and t == 0:
            inject_vessel(m,'Y01',document['vessels_by_day']['0'][0], key='ship', size_seed='unit')
    terminal.run(policy, review)
    cid = 'IN_inbound'
    assert terminal.ready_times[cid] < terminal.exit_times[cid]
    jid = terminal.exit_jobs[cid]
    job = terminal.blocks['Y01'].jobs[jid]
    assert job.status == JobStatus.DONE and job.service_start >= terminal.ready_times[cid]
    assert announcer.n_skipped == 0 and errors['n'] == 0
    assert terminal.cargo_report()['physical_inventory'] == 0
    if ship:
        vessel = terminal.blocks['Y01'].vessels['ship']
        assert vessel.done and vessel.truth.actual_completion_s > job.service_end
    else:
        times = terminal.blocks['Y01'].time_ledger.records['outbound']
        assert times.gate_in == 60 and times.block_arrival == 360
        assert times.service_start > times.block_arrival


@pytest.mark.parametrize('ship', [False, True])
@pytest.mark.parametrize('when', [60, 420])
def test_follow_reallocated_inbound_preserves_cargo_and_elapsed_wait(ship, when):
    terminal, announcer, document, _ = make_world(ship=ship, incoming=600)
    policy, errors = _rule_policy('SF_SPT', seed=9900306)
    def review(m,t):
        announcer.review(m,t)
        if ship and t == 0:
            inject_vessel(m,'Y01',document['vessels_by_day']['0'][0], key='ship', size_seed='unit')
        if t == when:
            assert m.try_pre_gate_transfer('inbound','Y02', travel_s=201,
                                           route_delta_s=11)
    terminal.run(policy, review)
    jid = terminal.exit_jobs['IN_inbound']
    assert terminal.ledger.records[jid].owner == 'Y02'
    job = terminal.blocks['Y02'].jobs[jid]
    assert job.status == JobStatus.DONE and job.target_container == 'IN_inbound'
    assert terminal.ready_times['IN_inbound'] <= job.service_start
    assert terminal.cargo_report()['physical_inventory'] == 0
    assert errors['n'] == announcer.n_skipped == 0
    if ship:
        assert terminal.remote_handoffs == 1
        assert terminal.blocks['Y01'].vessels['ship'].done
        assert not terminal.blocks['Y02'].vessels
    else:
        tt = terminal.blocks['Y02'].time_ledger.records[jid]
        assert tt.gate_in == 60
        if when == 420:
            assert tt.block_arrival == 360
            assert terminal.relocations[0]['road_ready_s'] > when
        total = sum(s.time_ledger.terminal_area_s for s in terminal.blocks.values())
        raw = sum(tt.gate_out-tt.gate_in for s in terminal.blocks.values()
                  for tt in s.time_ledger.records.values())
        assert total == pytest.approx(raw)


def test_bad_source_rejected_without_replacing_or_injecting_any_box():
    terminal, announcer, document, days = make_world()
    document['schedule'][1]['con_no'] = 'SOMEONE_ELSE'
    with pytest.raises(ContainerContractError):
        restore_input(document, seed=9900306, days=days, lead_mode='DIST')
    assert not terminal.ready_times and not terminal.ledger.records


def test_restore_rejects_wrong_seed_or_day_calendar():
    _,_,document,days = make_world()
    with pytest.raises(ContainerContractError, match='Seed/calendar'):
        restore_input(document, seed=9900307, days=days, lead_mode='DIST')


def test_completion_rejects_premature_exit():
    terminal,_,_,_ = make_world()
    job = NS(job_id='outbound',target_container='IN_inbound')
    with pytest.raises(RuntimeError, match='premature'):
        terminal.check_completion('Y01', job, 100)


def test_not_yet_announced_pickup_uses_actual_stored_block():
    terminal,_,document,_ = make_world(incoming=300, outgoing=1800)
    document['schedule'][1]['lead_s'] = 60
    announcer = V3Announcer(document['schedule'], resolve_entry=terminal.resolve_entry)
    policy,_ = _rule_policy('SF_SPT', seed=9900306)
    def review(m,t):
        announcer.review(m,t)
        if t == 60:
            assert m.try_pre_gate_transfer('inbound','Y02', travel_s=201, route_delta_s=11)
            assert 'outbound' not in m.ledger.records
    terminal.run(policy, review)
    assert terminal.orders['outbound'].con_loc == 'Y02'
    assert terminal.blocks['Y02'].jobs['outbound'].status == JobStatus.DONE
    assert terminal.relocations[0]['phase'] == 'at_notice'


def test_failed_inbound_transfer_does_not_move_its_dependent():
    terminal,announcer,_,_ = make_world()
    announcer.review(terminal,0)
    before = dict(terminal.locations)
    assert not terminal.try_pre_gate_transfer('inbound','UNKNOWN',travel_s=200)
    assert terminal.locations == before and not terminal.relocations
    assert terminal.ledger.records['outbound'].owner == 'Y01'


def test_waiting_pickup_routes_after_daily_accounting_prune():
    from yard_rl.v5.stage.month import prune_completed
    terminal,announcer,_,_ = make_world(incoming=600)
    policy,_ = _rule_policy('SF_SPT', seed=9900306)
    def review(m,t):
        announcer.review(m,t)
        if t == 360:
            assert prune_completed(m,t)['a_sorted'] == 1
        if t == 420:
            assert m.try_pre_gate_transfer('inbound','Y02',travel_s=201)
    terminal.run(policy,review)
    assert terminal.blocks['Y02'].jobs['outbound'].status == JobStatus.DONE
    assert terminal.ready_times['IN_inbound'] < terminal.exit_times['IN_inbound']
    total = sum(s.time_ledger.terminal_area_s for s in terminal.blocks.values())
    raw = sum(tt.gate_out-tt.gate_in for s in terminal.blocks.values()
              for tt in s.time_ledger.records.values())
    assert total == pytest.approx(raw)


def test_dependent_accounting_failure_does_not_partially_commit_inbound():
    terminal,announcer,_,_ = make_world()
    announcer.review(terminal,0)
    terminal.blocks['Y01'].time_ledger.records.pop('outbound')
    before = dict(terminal.locations)
    assert not terminal.try_pre_gate_transfer('inbound','Y02',travel_s=201)
    assert terminal.locations == before and not terminal.relocations
    assert terminal.ledger.records['inbound'].owner == 'Y01'
    assert not terminal._open_txn and terminal._reserved_inbound['Y02'] == 0
