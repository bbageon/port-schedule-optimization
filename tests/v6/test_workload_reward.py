"""Conservation, information boundaries, and shaped-return integration."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from yard_rl.v6.ppo.model import BlockPolicy, encode
from yard_rl.v6.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v6.reward.workload import WorkloadConfig, WorkloadMeter, targets, potential
from yard_rl.v6.schema import ExecutionRecord, Order
from yard_rl.v6.world.contract.schema import CandidateKind
from yard_rl.v6.world.domain.enums import JobStatus


def config(eta=1):
    return WorkloadConfig({f: 100.0 for f in ('GATE_IN', 'GATE_OUT', 'VESSEL_LOAD', 'VESSEL_DISCHARGE')},
                          {'L': 1.0, 'W': 1.0}, .8, horizon_s=100, eta=eta)


def meter():
    spec = SimpleNamespace(service_bay_min=1, service_bay_max=10)
    fleet = SimpleNamespace(ids=lambda: ('L', 'W'), spec=lambda c: spec)
    sim = SimpleNamespace(fleet=fleet, jobs={}, profile=SimpleNamespace(block=SimpleNamespace(bay_count=10)),
                          active_plan=lambda c: None)
    o = Order('known', 1, 0, 50, 'A', 'box')
    future = Order('future', 1, 500, 550, 'A', 'future-box')
    bridge = SimpleNamespace(orders={'known': o, 'future': future},
        records={k: ExecutionRecord(k) for k in ('known', 'future')}, market=SimpleNamespace(decided=set()))
    m = WorkloadMeter(config())
    m.bind(SimpleNamespace(blocks={'A': sim, 'B': sim}), bridge)
    return m, bridge, sim


def test_water_fill_respects_low_demand_fixed_work_and_overload():
    assert targets([100, 100], [0, 0], [20, 0], [.8, .8]) == pytest.approx([10, 10])
    assert targets([100, 100], [120, 0], [20, 0], [.8, .8]) == pytest.approx([120, 20])
    assert targets([100, 100], [0, 0], [200, 0], [.8, .8]) == pytest.approx([80, 80])
    assert potential([100, 100], [0, 0], [100, 40], [.8, .8]) > potential([100, 100], [0, 0], [120, 20], [.8, .8])


def test_original_demand_survives_deferral_and_no_future_notice():
    m, bridge, _ = meter()
    m.snapshot(0)
    assert m.last['total_work_s'] == 100
    assert m.last['jobs'] == 1
    bridge.orders['known'] = replace(bridge.orders['known'], in_out_reserve_s=7000)
    m.snapshot(60)
    assert m.last['total_work_s'] == 100
    assert m.last['jobs'] == 1
    assert m.last['conservation_error_s'] < 1e-10
    bridge.records['known'].job_done_s = 90
    m.snapshot(90)
    assert m.last['jobs'] == 0


def test_committed_assignment_and_active_plan_not_shared_twice():
    m, bridge, sim = meter()
    bridge.market.decided.add('known')
    m.snapshot(10)
    assert m.last['fixed_work_s'] == 100
    assert m.last['movable_work_s'] == 0
    sim.jobs['known'] = SimpleNamespace(job_id='known', assigned_crane='L', status=JobStatus.RUNNING,
                                       is_vessel_linked=False)
    sim.active_plan = lambda c: SimpleNamespace(job_id='known', kind=CandidateKind.SERVE,
                                               start_s=10, duration_s=100)
    m.snapshot(60)
    assert m.last['fixed_work_s'] == 50
    assert m.last['rho'] == pytest.approx([.5, 0, 0, 0])


@pytest.mark.parametrize('terminated', [False, True])
def test_runtime_cost_separate_and_truncation_bootstraps_potential(terminated):
    rt = PPORuntime(BlockPolicy(), config=PPOConfig(reward_mode='legacy-krw', rollout_intervals=60))
    rt.bids = ['A']
    rt.states_at = lambda t: encode([[0.0] * 9], 'state')
    rt.read_cost = lambda t: t * 100
    rt.workload = SimpleNamespace(config=SimpleNamespace(eta=2), snapshot=lambda t: -1-t/60)
    rt.boundary(0)
    rt.boundary(60, terminated=terminated)
    expected = 2 * ((0 if terminated else .999 * -2) + 1)
    cost_reward = -6000 / rt.config.reward_scale_krw
    assert rt.buffer[0].reward == pytest.approx(cost_reward + expected)
    assert rt.total_reward == pytest.approx(cost_reward)
    assert rt.learning_shaping_reward == pytest.approx(expected)


def test_discounted_round_trip_cannot_farm_shaping():
    gamma = .999
    path = [-.6, -.1, -.6]
    got = sum(gamma**i * (gamma * path[i+1]-path[i]) for i in range(2))
    assert got == pytest.approx(-path[0] + gamma**2 * path[-1])


def test_shape_disabled_preserves_cost_reward():
    rt = PPORuntime(BlockPolicy(), config=PPOConfig(reward_mode='legacy-krw'))
    rt.bids = ['A']
    rt.states_at = lambda t: encode([[0.0] * 9], 'state')
    rt.read_cost = lambda t: t * 100
    rt.boundary(0)
    rt.boundary(60)
    assert rt.buffer[0].reward == pytest.approx(-6000 / rt.config.reward_scale_krw)
    assert rt.shaping_reward == 0


def test_original_request_audit_does_not_charge_notice_leadtime(tmp_path):
    from yard_rl.v6.ppo.workload_observer import Observer
    o = Order('k', 1, 0, 100, 'A', 'box')
    r = ExecutionRecord('k', gate_in_s=100, gate_out_s=200)
    rt = SimpleNamespace(original_requests={'k': (100, 0)},
                         bridge=SimpleNamespace(orders={'k': o}, records={'k': r}))
    observer = Observer(tmp_path, [])
    normal = observer.request_metrics(rt, 300)
    assert normal['request_to_exit_mean_s'] == 100
    assert normal['original_request_wait_krw'] == normal['actual_gate_wait_krw']
    rt.bridge.orders['k'] = replace(o, in_out_reserve_s=1000)
    r.gate_in_s, r.gate_out_s = None, None
    delayed = observer.request_metrics(rt, 300)
    assert delayed['requested'] == 1 and delayed['unfinished'] == 1
    assert delayed['original_request_wait_krw'] > 0 and delayed['actual_gate_wait_krw'] == 0


def test_meter_runs_on_real_engine_and_zero_eta_matches_baseline(fixed_container_input, tmp_path):
    import torch
    from yard_rl.v6.ppo.runtime import DebugStop
    from yard_rl.v6.stage.month import plan_days
    from yard_rl.v6.stage.month_run import run_month
    from yard_rl.v6.world.integrated.profiles import build_h21_profile
    cfg = replace(config(eta=0), capacity_fraction={c.crane_id: 1 for c in build_h21_profile().cranes})
    results = []
    for meter_arg in (None, WorkloadMeter(cfg)):
        torch.manual_seed(10)
        rt = PPORuntime(BlockPolicy(), seed=10, stop_s=7200, workload=meter_arg)
        with pytest.raises(DebugStop):
            run_month(seed=9909100, days=plan_days(9909100, [60, 60, 60]), ppo=rt)
        results.append(rt)
    assert results[0].cost_krw == results[1].cost_krw
    assert results[0].crane_actions == results[1].crane_actions
    assert results[1].workload.last['conservation_error_s'] < 1e-8
    for key, value in results[0].policy.state_dict().items():
        assert torch.equal(value, results[1].policy.state_dict()[key])
    import gzip
    import json
    from yard_rl.v6.ppo.workload_observer import Observer
    rt = results[1]
    rt.original_requests = {k: (request, notice) for k, (notice, request) in rt.workload.original.items()}
    Observer(tmp_path, []).finish(rt)
    with gzip.open(tmp_path/'execution-records.json.gz', 'rt', encoding='utf-8') as source:
        events = json.load(source)
    assert len(events) == len(rt.bridge.records)
    assert all(e['original_requested_gate_s'] >= e['original_notice_s'] for e in events)
