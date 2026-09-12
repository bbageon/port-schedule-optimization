"""Valid demand survives capacity pressure; only physical storage must wait."""
from types import SimpleNamespace as NS

import pytest

from yard_rl.v5.ppo.model import BlockPolicy
from yard_rl.v5.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v5.reward.krw import truck_wait_krw
from yard_rl.v5.stage.bridge import MarketBridge
from yard_rl.v5.stage.cargo_input import restore_input
from yard_rl.v5.stage.cargo_runtime import CargoBlock, CargoTerminal
from yard_rl.v5.stage.episode import INFO_LEVEL, _rule_policy
from yard_rl.v5.stage.fixed_seed import build_fixed_seed
from yard_rl.v5.stage.month import DayPlan
from yard_rl.v5.stage.orders import V3Announcer, orders_from_schedule
from yard_rl.v5.world.domain.enums import ContainerSize, JobStatus, LoadStatus
from yard_rl.v5.world.domain.models import Container
from yard_rl.v5.world.integrated.multiblock import TransferError
from yard_rl.v5.world.integrated.profiles import build_h21_profile
from yard_rl.v5.world.integrated.terminal_stream import ensure_time_ledger
from yard_rl.v5.world.integrated.yard_layout import terminal_layout


def world(*, full=True, pickup=False, initial_count=None, reserved_moves=0):
    profile, layout = build_h21_profile(), terminal_layout()
    g = profile.block
    initial = {}
    if full:
        for bay in range(1,g.bay_count+1):
            for row in range(1,g.row_count+1):
                for tier in range(1,g.tier_max+1):
                    if initial_count is not None and len(initial)>=initial_count:
                        continue
                    cid = f'C{bay}-{row}-{tier}'
                    initial[cid] = Container(cid,ContainerSize.FT20,LoadStatus.FULL,
                                             'Y01',bay,row,tier)
    def entry(jid, flow, at):
        return dict(job_id=jid,flow=flow,target=None,con_no='IN_'+jid,block='Y01',
                    arrival_s=at,lead_s=at,travel_s=300,travel_base_s=300,
                    size_ft40=False,exit_travel_s=60,day=0)
    schedule = [entry('incoming','GATE_IN',60)]
    if pickup:
        schedule.append(entry('pickup','GATE_OUT',900))
    days = [DayPlan(0,len(schedule),'capacity-unit',9901306,0,1)]
    built = dict(days=days,month_end_s=86400,lead_mode='DIST',schedule=schedule,
                 day0=dict(scenarios={'Y01':NS(containers=initial)}))
    vessels = {0:[dict(key='reserved',work='DISCHARGE',block='Y01',moves=reserved_moves,
                       start_s=1800,cadence_s=120)]} if reserved_moves else {}
    doc,_ = build_fixed_seed(built,vessels,seed=9900306)
    if pickup:
        # Place the already-selected fixed pickup at an accessible top tier.
        # This fixture has no rehandle space. No live cargo is moved or replaced.
        rows = doc['initial_inventory']['Y01']
        target = next(c for c in rows if c['container_id']==doc['schedule'][1]['target'])
        top = next(c for c in rows if c['bay']==1 and c['row']==1 and c['tier']==g.tier_max)
        for field in ('bay','row','tier'):
            target[field],top[field] = top[field],target[field]
    restored = restore_input(doc,seed=9900306,days=days,lead_mode='DIST')
    scenario = restored['day0']['scenarios']['Y01']
    scenario.horizon_s, scenario.drain_window_s = 2400,120
    sim = ensure_time_ledger(CargoBlock(profile,scenario))
    sim.info_level = INFO_LEVEL
    terminal = CargoTerminal({'Y01':sim},document=doc,layout=layout,
                              extra_review_epochs=tuple(range(0,2401,60)))
    terminal.orders,terminal.records = orders_from_schedule(restored)
    ann = V3Announcer(doc['schedule'],resolve_entry=terminal.resolve_entry)
    bridge = MarketBridge(None,layout,orders=terminal.orders,records=terminal.records,end_s=2520)
    rt = PPORuntime(BlockPolicy(),config=PPOConfig(rollout_intervals=10000))
    rt.bind(terminal,bridge,{}, {})
    if reserved_moves:
        from yard_rl.v5.stage.month_engine import inject_vessel
        inject_vessel(terminal,'Y01',doc['vessels_by_day']['0'][0],key='reserved',size_seed='unit')
    return terminal,sim,ann,bridge,rt


def run(terminal,ann,bridge,rt,check=lambda m,t:None):
    policy,errors = _rule_policy('SF_SPT',seed=9900306)
    def review(m,t):
        ann.review(m,t)
        bridge._sync(m,t)
        rt.boundary(t)
        check(m,t)
    terminal.run(policy,review)
    bridge._sync(terminal,2520)
    rt.finish(2520,terminated=False)
    assert errors['n']==ann.n_skipped==0


def test_full_yard_retains_waiting_order_and_continues_charging_ppo_reward(record_property):
    terminal,sim,ann,bridge,rt = world()
    size = len(sim.stacks.containers)
    costs = {}
    def check(m,t):
        assert len(sim.stacks.containers)==size
        if t>=360:
            assert sim.jobs['incoming'].status==JobStatus.WAITING
            assert sim.jobs['incoming'].service_start is None
            assert not any(sim._dispatchable(sim.jobs['incoming'],c.crane_id)
                           for c in sim.profile.cranes)
        costs[t] = rt.cost_breakdown['c_wait']
    run(terminal,ann,bridge,rt,check)
    assert ann.n_admitted==1 and 'incoming' in terminal.ledger.records
    assert 'IN_incoming' not in sim.stacks.containers
    assert terminal.orders['incoming'].con_no=='IN_incoming'
    tt = sim.time_ledger.records['incoming']
    assert (tt.gate_in,tt.block_arrival,tt.job_done)==(60,360,None)
    assert costs[0]==0
    assert costs[600]==pytest.approx(truck_wait_krw(540))
    assert costs[1200]==pytest.approx(truck_wait_krw(1140))
    assert costs[1200]>costs[600]>0
    assert rt.cost_krw==pytest.approx(truck_wait_krw(2460))
    assert rt.total_reward==pytest.approx(-rt.cost_krw/rt.config.reward_scale_krw)
    assert sim.time_ledger.terminal_area_s==pytest.approx(2460)
    sim.check_invariants()
    record_property('wait_cost_at_600_s',costs[600])
    record_property('wait_cost_at_1200_s',costs[1200])
    record_property('final_cost_krw',rt.cost_krw)
    record_property('physical_inventory',len(sim.stacks.containers))


def test_actual_pickup_frees_space_then_same_waiting_inbound_is_stored():
    terminal,sim,ann,bridge,rt = world(pickup=True)
    size = len(sim.stacks.containers)
    def check(m,t):
        assert len(sim.stacks.containers)<=size
        if t==600:
            assert sim.jobs['incoming'].status==JobStatus.WAITING
    run(terminal,ann,bridge,rt,check)
    incoming,outgoing = sim.jobs['incoming'],sim.jobs['pickup']
    assert ann.n_admitted==2 and incoming.status==outgoing.status==JobStatus.DONE
    assert incoming.service_start>=outgoing.service_end
    assert 'IN_incoming' in sim.stacks.containers
    assert len(sim.stacks.containers)==size
    assert rt.cost_breakdown['c_wait']==pytest.approx(sum(
        truck_wait_krw(t.gate_out-t.gate_in) for t in sim.time_ledger.records.values()))
    assert (incoming.actual_gate_in,incoming.actual_block_arrival)==(60,360)


def test_negative_reservation_balance_does_not_reject_demand(monkeypatch):
    terminal,sim,ann,bridge,rt = world(full=False)
    monkeypatch.setattr(terminal,'free_slots',lambda bid:-36)
    run(terminal,ann,bridge,rt)
    assert ann.n_admitted==1 and sim.jobs['incoming'].status==JobStatus.DONE


def test_actual_934_boxes_and_future_discharge_reservations_keep_new_order():
    # Reconstruct the failed balance, not the unavailable full day-22 world.
    terminal,sim,ann,_,_ = world(initial_count=934,reserved_moves=542)
    assert len(sim.stacks.containers)==934 and terminal.free_slots('Y01')==-36
    ann.review(terminal,0)
    assert ann.n_admitted==1 and ann.n_skipped==0
    assert sim.jobs['incoming'].status==JobStatus.PLANNED
    assert terminal.free_slots('Y01')==-37


@pytest.mark.parametrize('invalid',['duplicate','unknown_source','past_time','no_ledger'])
def test_capacity_change_does_not_relax_identity_time_or_ledger_checks(invalid):
    from yard_rl.v5.world.integrated.terminal_stream import _job_from_entry
    terminal,sim,ann,_,_ = world()
    e = dict(next(iter(ann.by_epoch.values()))[0])
    if invalid=='duplicate':
        ann.review(terminal,0)
    elif invalid=='unknown_source':
        e['job_id'],e['con_no'] = 'alien','IN_alien'
    elif invalid=='past_time':
        e['arrival_s'] = -1
    else:
        sim.time_ledger = None
    count = len(terminal.ledger.records)
    with pytest.raises(TransferError):
        terminal.admit_external_job('Y01',_job_from_entry(e,e['arrival_s']),
                                    gate_in_s=e['arrival_s'],travel_s=e['travel_s'])
    assert len(terminal.ledger.records)==count
