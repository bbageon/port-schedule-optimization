"""User contract: crane cost stops at completion, even when exit is delayed."""
from dataclasses import replace

import pytest

from yard_rl.v7.reward.krw import truck_wait_krw
from yard_rl.v7.reward.phi import terminal_cost_krw
from yard_rl.v7.reward.operational import physical_totals, reference_config, normalized_loss
from yard_rl.v7.schema.record import ExecutionRecord
from yard_rl.v7.schema.lifecycle import TRUCK_TIME_CONTRACT


def record(*, a=0., c=1800., o=2400.):
    return ExecutionRecord('truck', gate_in_s=a, job_done_s=c, gate_out_s=o)


def physical(r, end=7200.):
    return physical_totals({'truck':r}, end_s=end, vessel_idle={}, empty_s=0, rehandles=0)


@pytest.mark.parametrize('exit_time', [2400., 7000., 9000., None])
def test_exit_delay_cannot_change_objective_or_normalized_reward(exit_time):
    r = record(o=exit_time)
    b = terminal_cost_krw({'truck':r}, end_s=7200.)
    assert b.wait == 20000.
    assert b.mean_gate_to_job_done_s == 1800.
    assert b.n_job_censored == 0 and b.n_job_done == 1
    assert b.n_exit_pending == int(exit_time is None or exit_time > 7200)
    end = min(exit_time,7200.) if exit_time is not None else 7200.
    assert b.mean_turn_time_s == end
    assert b.mean_job_done_to_gate_out_s == end-1800.
    assert b.mean_turn_time_s == b.mean_gate_to_job_done_s+b.post_job_gate_s
    assert physical(r)['truck_s'] == 1800.
    ref = reference_config()
    assert normalized_loss(physical(r),ref) == normalized_loss(physical(record()),ref)
    assert ref['truck_time_contract'] == TRUCK_TIME_CONTRACT
    assert ref['reference_truck_time_contract'] == 'gate-in-to-gate-out-v1'


@pytest.mark.parametrize('done,end,seconds,unfinished', [
    (1800.,7200.,1800.,0), (5400.,7200.,5400.,0),
    (9000.,7200.,7200.,1), (None,7200.,7200.,1), (7200.,7200.,7200.,0)])
def test_completion_and_horizon_define_cost(done,end,seconds,unfinished):
    r = record(c=done, o=None)
    b = terminal_cost_krw({'truck':r},end_s=end)
    assert b.wait == truck_wait_krw(seconds)
    assert b.mean_gate_to_job_done_s == seconds and b.n_job_censored == unfinished
    assert physical(r,end)['truck_s'] == seconds


def test_entered_cohort_and_completed_cohort_have_explicit_denominators():
    records = {'done':record(c=30.,o=60.), 'exit_pending':record(c=80.,o=None),
               'unfinished':record(a=50.,c=None,o=None),
               'boundary':record(a=100.,c=None,o=None),
               'future':record(a=101.,c=None,o=None),
               'absent':record(a=None,c=None,o=None)}
    b = terminal_cost_krw(records,end_s=100.)
    assert (b.n_trucks,b.n_job_done,b.n_job_censored,b.n_censored,b.n_exit_pending)==(4,2,2,3,1)
    assert b.mean_gate_to_job_done_s == 40.  # (30+80+50+0)/4
    assert b.mean_job_done_to_gate_out_s == 25.  # (30+20)/2 completed
    assert b.post_job_gate_s == 50.
    assert b.mean_turn_time_s*4 == b.mean_gate_to_job_done_s*4+b.post_job_gate_s


def test_job_completed_but_no_exit_does_not_accrue_future_interval_cost():
    r = record(c=1800.,o=None)
    assert physical(r,7200.) == physical(r,1800.)
    assert terminal_cost_krw({'t':r},end_s=7200.).wait == terminal_cost_krw({'t':r},end_s=1800.).wait


def test_cpu_and_jitted_array_match_for_distinct_three_events():
    import jax
    jax.config.update('jax_enable_x64',True)
    from yard_rl.v7.gpu.phi import orders_from_records, terminal_cost_krw as array_cost
    records = {'1':record(c=1800.,o=7000.),'2':record(c=3000.,o=None),
               '3':record(a=50.,c=None,o=None),'future':record(a=8000.,c=None,o=None)}
    for end in (100.,3000.,7200.):
        cpu = terminal_cost_krw(records,end_s=end).as_dict()
        gpu = jax.jit(array_cost)(orders_from_records(records),end).as_dict()
        assert gpu == cpu


def test_legacy_checkpoint_and_label_contract_cannot_resume_training(tmp_path):
    from yard_rl.v7.ppo.model import BlockPolicy
    from yard_rl.v7.ppo.runtime import PPORuntime
    from yard_rl.v7.crane.policy import CraneNet
    from yard_rl.v7.crane.fit import CraneTrainer
    from yard_rl.v7.crane.teacher import CraneLabelSet
    from yard_rl.v7.crane.train import CraneTrainState
    import torch
    policy = BlockPolicy()
    policy.checkpoint_reward_mode = 'operational'
    policy.checkpoint_reward_contract = {k:v for k,v in reference_config().items() if k!='truck_time_contract'}
    with pytest.raises(ValueError,match='truck time contract'):
        PPORuntime(policy)
    net=CraneNet(); trainer=CraneTrainer(net)
    with pytest.raises(ValueError,match='Label truck time contract'):
        trainer.fit(CraneLabelSet(truck_time_contract='gate-in-to-gate-out-v1'))
    old=tmp_path/'old.pt'; torch.save({'crane':net.state_dict(),'it':1},old)
    with pytest.raises(ValueError,match='truck time contract'):
        CraneTrainState.load(old,net,trainer)
    CraneTrainState(net,trainer).save(tmp_path/'new',2)
    assert CraneTrainState.load(tmp_path/'new/crane_002.pt',net,trainer)==2


def test_array_runtime_and_day_adapters_carry_completion_without_future_leak():
    from types import SimpleNamespace as NS
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from yard_rl.v7.gpu.state import empty_orders
    from yard_rl.v7.gpu.train import boundary_orders
    from yard_rl.v7.gpu.month import day_orders
    from yard_rl.v7.gpu.phi import terminal_cost_krw as array_cost
    o=empty_orders(1)._replace(block=jnp.array([0]),gate_in_s=jnp.array([0.]),
                              done_s=jnp.array([100.]),gate_out_s=jnp.array([300.]))
    orders=jax.tree_util.tree_map(lambda x:x[None,...],o)
    run=NS(tw=NS(blocks=NS(orders=orders),sched=NS(s=1),
        ledger=NS(registered=jnp.array([True]),a_gate_in=jnp.array([0.]),
                  owner=jnp.array([0]),row=jnp.array([0]))))
    for t,expected in ((50.,50.),(100.,100.),(200.,100.),(400.,100.)):
        runtime=boundary_orders(run,t)
        daily,mask=day_orders(run,NS(truck_day=(0,)),0,sync_t=t)
        for adapted in (runtime,daily):
            assert float(array_cost(adapted,t).mean_gate_to_job_done_s)==expected
            assert bool(jnp.isfinite(adapted.done_s[0]))==(t>=100)
            assert bool(jnp.isfinite(adapted.gate_out_s[0]))==(t>=300)
