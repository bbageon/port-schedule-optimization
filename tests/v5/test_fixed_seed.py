"""Generate real fixed cargo plans; never bypass the input audit or run PPO."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from yard_rl.v5.stage.container_contract import ContainerContractError
from yard_rl.v5.stage.fixed_seed import build_fixed_seed, fixed_seed_audit
from yard_rl.v5.stage.month import DayPlan
from yard_rl.v5.stage.seed_bundle import load_seed_bundle, save_seed_bundle
from yard_rl.v5.world.domain.enums import ContainerSize, LoadStatus
from yard_rl.v5.world.domain.models import Container


def entry(key='pickup', flow='GATE_OUT', at=600, block='B1'):
    cid = 'OLD' if flow == 'GATE_OUT' else None
    return dict(job_id=key, flow=flow, target=cid, con_no=cid or 'IN_' + key,
                arrival_s=at, lead_s=60, travel_s=300, travel_base_s=300,
                exit_travel_s=60, size_ft40=False, block=block, day=0)


def data(schedule=None, initial=1):
    boxes = {f'C{i}': Container(f'C{i}', ContainerSize.FT20, LoadStatus.FULL,
                               'B1', i+1, 1, 1) for i in range(initial)}
    return dict(days=[DayPlan(0, 1, 'unit', 9900306, 0, 1)],
                day0=dict(scenarios={'B1': NS(containers=boxes)}),
                schedule=[entry()] if schedule is None else schedule,
                month_end_s=86400, lead_mode='DIST')


def ship(key='load', work='LOAD', at=1200, moves=1):
    return dict(key=key, block='B1', work=work, start_s=at, cadence_s=120, moves=moves)


def test_initial_inventory_needs_no_incoming_order():
    document, audit = build_fixed_seed(data(), {}, seed=9900306)
    assert audit['passed'] and audit['source_exit_counts'] == {'INITIAL->GATE_OUT': 1}
    assert document['orders'][0]['con_no'] == document['schedule'][0]['target'] == 'C0'
    assert len(document['orders'][0]) == 6
    assert audit['empty_execution_records'] and not audit['ready_for_training']


def test_all_source_kinds_have_fixed_unique_exits_and_no_input_mutation():
    original = data([entry('in1', 'GATE_IN', 200), entry('in2', 'GATE_IN', 210)] +
                    [entry(f'out{i}', at=1000+i) for i in range(4)], initial=2)
    vessels = {0: [ship('unload', 'DISCHARGE', 0, 2), ship(moves=2)]}
    before = copy.deepcopy((original, vessels))
    document, audit = build_fixed_seed(original, vessels, seed=9900306)
    assert (original, vessels) == before
    assert audit['passed'] and audit['sources'] == audit['exits'] == 6
    targets = [e['target'] for e in document['schedule'] if e['flow']=='GATE_OUT']
    targets += document['vessels_by_day']['0'][1]['targets']
    assert len(set(targets)) == 6
    assert set(targets) == {s['container'] for s in document['sources']}
    assert audit['ending_unassigned_sources'] == 0


def test_initial_block_label_uses_actual_owner_without_mutating_positions():
    original = data([])
    scenario = original['day0']['scenarios'].pop('B1')
    original['day0']['scenarios']['Y01'] = scenario
    document, audit = build_fixed_seed(original, {}, seed=9900306)
    exported = document['initial_inventory']['Y01'][0]
    assert exported['block'] == 'Y01' and audit['passed']
    box = scenario.containers['C0']
    assert box.block == 'B1'
    assert (exported['bay'], exported['row'], exported['tier']) == (box.bay, box.row, box.tier)


def test_held_initial_cargo_requires_a_release_plan():
    original = data()
    original['day0']['scenarios']['B1'].containers['C0'].work_available = False
    with pytest.raises(ContainerContractError, match='explicit release'):
        build_fixed_seed(original, {}, seed=9900306)


def test_future_delivery_is_reserved_once_without_moving_pickup_time():
    original = data([entry(at=100), entry('late', 'GATE_IN', 2000)], initial=0)
    document, audit = build_fixed_seed(original, {}, seed=9900306)
    assert document['schedule'][0]['target'] == 'IN_late'
    assert document['schedule'][0]['arrival_s'] == 100
    assert audit['passed'] and audit['planned_waiting_exits'] == 1
    assert audit['max_planned_wait_lower_bound_s'] == 1900
    assert audit['contract']['planned_waiting_exits'] == 1


def test_future_source_not_reused_by_a_later_pickup():
    original = data([entry('first', at=100), entry('second', at=200),
                     entry('in1', 'GATE_IN', 2000), entry('in2', 'GATE_IN', 2100)], initial=0)
    document, audit = build_fixed_seed(original, {}, seed=9900306)
    assert [e['target'] for e in document['schedule'][:2]] == ['IN_in1', 'IN_in2']
    assert audit['planned_waiting_exits'] == 2


def test_true_supply_shortage_does_not_create_cargo_or_drop_orders():
    original = data([entry('first'), entry('second')])
    before = copy.deepcopy(original)
    with pytest.raises(ContainerContractError, match='Insufficient') as error:
        build_fixed_seed(original, {}, seed=9900306)
    assert original == before
    assert error.value.report['B1'] == dict(sources=1, exits=2, missing=1)


def test_truck_and_vessel_reserve_different_containers():
    document, audit = build_fixed_seed(data(initial=2), {0: [ship()]}, seed=9900306)
    assert document['schedule'][0]['target'] != document['vessels_by_day']['0'][0]['targets'][0]
    assert audit['passed']


def test_exogenous_assignment_is_repeatable_and_seed_sensitive():
    original = data([entry(f'out{i}', at=1000+i) for i in range(8)], initial=12)
    a, aa = build_fixed_seed(original, {}, seed=9900306)
    b, bb = build_fixed_seed(original, {}, seed=9900306)
    c, _ = build_fixed_seed(original, {}, seed=9900307)
    assert a == b and aa == bb
    assert a['schedule'] != c['schedule']


def test_outbound_size_metadata_follows_the_assigned_container():
    original = data()
    original['day0']['scenarios']['B1'].containers['C0'].size = ContainerSize.FT40
    original['schedule'][0]['size_class'] = ('20', 'GP', None)
    document, audit = build_fixed_seed(original, {}, seed=9900306)
    assert document['schedule'][0]['size_ft40'] is True and audit['passed']
    assert document['schedule'][0]['size_class'] == ['40', 'GP', None]
    document['schedule'][0]['size_class'] = ['20', 'GP', None]
    assert not fixed_seed_audit(document)['passed']
    document['schedule'][0]['size_class'] = ['40', 'GP', None]
    document['schedule'][0]['size_ft40'] = False
    assert not fixed_seed_audit(document)['passed']


@pytest.mark.parametrize('mutation', ['source', 'order', 'duplicate_exit', 'short_manifest'])
def test_mutated_bundle_does_not_pass_independent_validation(mutation):
    document, _ = build_fixed_seed(data(initial=2), {0: [ship()]}, seed=9900306)
    if mutation == 'source':
        document['sources'][0]['planned_source_s'] = 999
    elif mutation == 'order':
        document['orders'][0]['con_no'] = 'WRONG'
    elif mutation == 'duplicate_exit':
        document['vessels_by_day']['0'][0]['targets'] = [document['schedule'][0]['target']]
    else:
        document['vessels_by_day']['0'][0]['targets'] = []
    try:
        result = fixed_seed_audit(document)
    except ContainerContractError:
        pass
    else:
        assert not result['passed']


def test_gzip_is_reproducible_load_checked_and_existing_file_preserved(tmp_path):
    document, audit = build_fixed_seed(data(), {}, seed=9900306)
    a = save_seed_bundle(document, tmp_path / 'a.gz')
    b = save_seed_bundle(document, tmp_path / 'different-name.gz')
    assert a['sha256'] == b['sha256'] and a['content_sha256'] == b['content_sha256']
    assert load_seed_bundle(tmp_path/'a.gz', expected_sha256=a['sha256']) == (document, audit)
    with pytest.raises(FileExistsError):
        save_seed_bundle(document, tmp_path/'a.gz')
    with pytest.raises(ContainerContractError, match='checksum'):
        load_seed_bundle(tmp_path/'a.gz', expected_sha256='0'*64)


def test_vessel_source_size_sequence_matches_the_frozen_engine():
    from yard_rl.v5.stage.month_engine import inject_vessel
    from yard_rl.v5.world.integrated.events import EventQueue
    from yard_rl.v5.world.integrated.profiles import build_h21_profile
    # Real injected jobs; no simulator timeline is run or learned from.
    sim = NS(clock=0, end=86400, vessels={}, jobs={}, queue=EventQueue(),
             profile=build_h21_profile(), _refresh_rates=lambda: None)
    mbt = NS(blocks={'B1': sim}, ledger=NS(register=lambda row: None))
    row = ship('unload', 'DISCHARGE', 0, 30)
    inject_vessel(mbt, 'B1', row, key='unload', size_seed='v3:month:9900306:unload')
    document, _ = build_fixed_seed(data([], initial=0), {0: [row]}, seed=9900306)
    assert {s['source_job']: s['size'] for s in document['sources']} == {
        key: job.inbound_size for key, job in sim.jobs.items()}


def test_real_three_day_generator_and_command_write_valid_fixed_data(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'scripts/v5/generate_fixed_seed.py'
    spec = importlib.util.spec_from_file_location('generate_fixed_seed', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'code_stamp', lambda: {'test_only': True})
    module.generate(tmp_path/'run', 9900306, module.make_plan(9900306, 3, 20), 20)
    report = json.loads((tmp_path/'run/audit.json').read_text())
    assert report['audit']['truck_orders'] == 60 and report['audit']['passed']
    assert all(report['checks'].values())
    assert report['learning_updates'] == report['simulation_steps'] == 0
