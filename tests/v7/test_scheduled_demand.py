"""Fixed arrivals survive full storage in the production daily engine."""
import copy

import pytest

from test_cargo_capacity_wait import world, run
from yard_rl.v7.stage.orders import V3Announcer
from yard_rl.v7.world.domain.enums import JobStatus
from yard_rl.v7.world.integrated.multiblock import TransferError
from yard_rl.v7.world.integrated.terminal_stream import _job_from_entry


def test_full_yard_keeps_fixed_arrival_and_charges_all_waiting_time():
    terminal, sim, ann, bridge, rt = world(terminal_type='scheduled')
    inventory = len(sim.stacks.containers)
    run(terminal, ann, bridge, rt)
    assert ann.n_admitted == 1 and ann.n_skipped == 0
    assert sim.jobs['incoming'].status == JobStatus.WAITING
    assert sim.time_ledger.terminal_area_s == pytest.approx(2520 - 60)
    assert rt.physical['truck_s'] == pytest.approx(2520 - 60)
    assert rt.total_reward < 0
    assert bridge.records['incoming'].gate_in_s == 60
    assert bridge.records['incoming'].gate_out_s is None
    assert len(sim.stacks.containers) == inventory
    sim.check_invariants()


def test_waiting_demand_runs_only_after_pickup_frees_real_space():
    terminal, sim, ann, bridge, rt = world(terminal_type='scheduled', pickup=True)
    inventory = len(sim.stacks.containers)
    run(terminal, ann, bridge, rt,
        check=lambda m, t: sim.check_invariants())
    incoming, pickup = sim.jobs['incoming'], sim.jobs['pickup']
    assert incoming.status == pickup.status == JobStatus.DONE
    assert incoming.service_start >= pickup.service_end
    assert (incoming.actual_gate_in, incoming.actual_block_arrival) == (60, 360)
    assert len(sim.stacks.containers) == inventory
    assert 'IN_incoming' in sim.stacks.containers


def test_future_reservations_keep_demand_and_clone_preserves_contract():
    terminal, sim, ann, _, _ = world(terminal_type='scheduled', full=False)
    terminal._reserved_inbound['Y01'] = 2000
    assert terminal.free_slots('Y01') < 0
    clone = copy.deepcopy(terminal)
    assert clone.preserve_external_demand is True
    ann.clone_fresh().review(clone, 0)
    assert 'incoming' in clone.blocks['Y01'].jobs
    assert 'incoming' not in sim.jobs
    assert clone._reserved_inbound == terminal._reserved_inbound
    # Transfer capacity is still unavailable; registering demand did not erase it.
    assert clone.free_slots('Y01') == terminal.free_slots('Y01') - 1


def test_legacy_admission_still_rejects_full_yard():
    terminal, sim, ann, _, _ = world(terminal_type='scheduled')
    terminal.preserve_external_demand = False
    ann.review(terminal, 0)
    assert ann.n_admitted == 0 and ann.n_skipped == 1
    assert 'incoming' not in sim.jobs
    assert '용량 부족' in ann.skips[0]['reason']


@pytest.mark.parametrize('invalid', ['duplicate', 'past_time', 'no_ledger', 'missing_target'])
def test_demand_registration_preserves_identity_time_and_accounting_checks(invalid):
    terminal, sim, ann, _, _ = world(terminal_type='scheduled')
    entry = dict(next(iter(ann.by_epoch.values()))[0])
    if invalid == 'duplicate':
        ann.review(terminal, 0)
    elif invalid == 'past_time':
        entry['arrival_s'] = -1
    elif invalid == 'no_ledger':
        sim.time_ledger = None
    else:
        entry.update(flow='GATE_OUT', target='absent', con_no='absent')
    count = len(terminal.ledger.records)
    with pytest.raises(TransferError):
        terminal.admit_external_job('Y01', _job_from_entry(entry, entry['arrival_s']),
            gate_in_s=entry['arrival_s'], travel_s=entry['travel_s'])
    assert len(terminal.ledger.records) == count


def test_observed_gate_entry_is_not_backdated_to_notice():
    terminal, sim, ann, bridge, _ = world(terminal_type='scheduled', full=False)
    ann.review(terminal, 0)
    bridge._sync(terminal, 0)
    assert bridge.records['incoming'].gate_in_s is None
    bridge._sync(terminal, 59)
    assert bridge.records['incoming'].gate_in_s is None
    bridge._sync(terminal, 60)
    assert bridge.records['incoming'].gate_in_s == 60


def test_clone_after_notice_keeps_waiting_job_without_readmitting():
    terminal, sim, ann, _, _ = world(terminal_type='scheduled')
    ann.review(terminal, 0)
    clone = copy.deepcopy(terminal)
    later = ann.window(60, 180).clone_fresh()
    later.review(clone, 60)
    assert len(clone.ledger.records) == 1
    assert clone.blocks['Y01'].jobs['incoming'].actual_gate_in == 60
    assert later.n_admitted == later.n_skipped == 0
