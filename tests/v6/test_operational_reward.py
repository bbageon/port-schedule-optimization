"""Physical units, unfinished work, currency independence, and actual learner integration."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yard_rl.v6.ppo.model import BlockPolicy, encode
from yard_rl.v6.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v6.ppo.checkpoint import save_checkpoint, load_policy
from yard_rl.v6.reward.operational import KEYS, fit_reference, reference_config, normalized_loss, physical_totals


def rows():
    return [[0,0,0,0,0], [60,2,4,6,8], [120,6,12,18,24], [180,12,24,36,48], [240,20,40,60,80]]


def test_physical_ledger_counts_unfinished_but_no_future_trucks_and_no_gt_price():
    records = dict(done=SimpleNamespace(gate_in_s=100, gate_out_s=160),
        active=SimpleNamespace(gate_in_s=90, gate_out_s=None),
        future=SimpleNamespace(gate_in_s=210, gate_out_s=None),
        absent=SimpleNamespace(gate_in_s=None, gate_out_s=None))
    actual = physical_totals(records, end_s=200, vessel_idle={'v': (999999999,20)}, empty_s=5, rehandles=2)
    assert actual == dict(truck_s=170, vessel_s=20, empty_s=5, rehandles=2)


def test_each_physical_channel_has_the_independently_computed_scale():
    fit = fit_reference(rows(), gamma=.5, start_s=120, end_s=240)
    assert tuple(fit['scales'].values()) == (1.875,3.75,5.625,7.5)
    ref = reference_config()
    assert ref['samples'] == 2880 and ref['currency_used'] is False
    assert PPOConfig().reward_scale_krw is None and PPOConfig().reward_mode == 'operational'


def test_changing_seconds_to_minutes_leaves_rewards_unchanged():
    raw = rows()
    minutes = [[row[0]]+[x/60 for x in row[1:4]]+[row[4]] for row in raw]
    a = fit_reference(raw, gamma=.5, start_s=120, end_s=240)
    b = fit_reference(minutes, gamma=.5, start_s=120, end_s=240)
    for ref in (a,b): ref['weights'] = dict.fromkeys(KEYS,.25)
    assert normalized_loss(dict(zip(KEYS,raw[-1][1:])),a) == pytest.approx(
        normalized_loss(dict(zip(KEYS,minutes[-1][1:])),b))


@pytest.mark.parametrize('data', [[], [[0,0,0,0,0],[60,0,1,1,1],[120,0,2,2,2]],
    [[0,0,0,0,0],[60,2,2,2,2],[120,1,3,3,3]], [[0,0,0,0,0],[60,float('nan'),2,2,2]]])
def test_missing_or_invalid_scale_cannot_fall_back_to_a_money_constant(data):
    with pytest.raises(ValueError): fit_reference(data,start_s=0,end_s=120)


def test_money_diagnostic_cannot_change_rewards_or_learned_weights():
    snapshots, rewards = [], []
    for money_multiplier in (1., 1_000_000.):
        torch.manual_seed(77)
        rt = PPORuntime(BlockPolicy(), config=PPOConfig(rollout_intervals=2,epochs=1), seed=77)
        rt.bids, rt.index = ['b'], {'b':0}
        rt.states_at = lambda t: encode([[t/3600]],'state')
        rt.read_cost = lambda t: money_multiplier*t
        rt.read_operational = lambda t: {k:(i+1)*t for i,k in enumerate(KEYS)}
        for t in (0,60,120):
            rt.boundary(t)
            if t < 120: rt.select('seller','b',t,[[1,0],[0,1]])
        assert len(rt.updates)==1 and rt.updates[0]['minibatches'] > 0
        snapshots.append({k:v.clone() for k,v in rt.policy.state_dict().items()})
        rewards.append(rt.total_reward)
    assert rewards[0] == rewards[1]
    assert all(torch.equal(snapshots[0][k],snapshots[1][k]) for k in snapshots[0])


def test_operational_checkpoint_survives_reload_and_cannot_be_reinterpreted(tmp_path):
    rt = PPORuntime(BlockPolicy())
    save_checkpoint(tmp_path/'p.pt',rt)
    loaded = load_policy(tmp_path/'p.pt')
    assert PPORuntime(loaded).reward_contract == rt.reward_contract
    with pytest.raises(ValueError,match='mode differs'):
        PPORuntime(loaded,config=PPOConfig(reward_mode='legacy-krw'))
    with pytest.raises(ValueError,match='reference/weights'):
        PPORuntime(loaded,config=PPOConfig(gamma=.99))


def test_cpu_array_boundaries_use_physical_increments_without_extra_division():
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from yard_rl.v6.gpu import ppo_runtime as pr
    from yard_rl.v6.gpu.train import TrainConfig
    ref = reference_config()
    scales = np.asarray([ref['scales'][k] for k in KEYS])
    cfg = TrainConfig(n_blocks=1,cmax=2,amax=2,training=False).runtime()
    assert cfg.reward_mode == 'operational' and cfg.reward_scale_krw is None
    state = pr.new_state(cfg)
    states, values = jnp.zeros((1,37)), jnp.zeros(1)
    step = jax.jit(lambda s,t,p: pr.boundary(s,cfg,t,t*99999999,states,values,physical=p))
    state,_ = step(state,0.,scales)
    state,out = step(state,60.,scales*101)
    assert float(out.reward) == pytest.approx(-100.)  # No clipping or second normalization.
    state,_ = step(state,60.,scales*101)
    assert float(state.total_reward) == pytest.approx(-100.)
    with pytest.raises(ValueError,match='requires physical'):
        pr.boundary(state,cfg,120.,0.,states,values)


def test_partial_completion_and_each_ledger_are_preserved_at_boundaries():
    rt = PPORuntime(BlockPolicy(),training=False)
    rt.bids=['b']
    rt.states_at=lambda t: encode([[0]],'state')
    rt.read_cost=lambda t: 0.
    rt.read_operational=lambda t: {k:t*(i+1) for i,k in enumerate(KEYS)}
    for t in (0,60,60,120): rt.boundary(t)
    assert rt.total_reward == pytest.approx(-normalized_loss(rt.physical,rt.reward_contract))
    rt.read_operational=lambda t: dict.fromkeys(KEYS,0.)
    with pytest.raises(RuntimeError,match='physical ledger decreased'): rt.boundary(180)


def test_array_adapter_still_accepts_an_actual_v5_config():
    from yard_rl.v5.ppo.runtime import PPOConfig as V5Config
    from yard_rl.v6.gpu.ppo_runtime import RuntimeConfig
    old = V5Config()
    adapted = RuntimeConfig.from_ppo_config(old)
    assert adapted.reward_mode == 'legacy-krw' and adapted.reward_scale_krw == old.reward_scale_krw


def test_array_rejects_corrupt_negative_initial_physical_ledger():
    import jax.numpy as jnp
    from yard_rl.v6.gpu import ppo_runtime as pr
    cfg = pr.RuntimeConfig(n_blocks=1,cmax=2,amax=2,reward_mode='operational')
    state,_ = pr.boundary(pr.new_state(cfg),cfg,0.,0.,jnp.zeros((1,37)),jnp.zeros(1),
                          physical=jnp.array([-1.,0.,0.,0.]))
    with pytest.raises(FloatingPointError): pr.raise_on_flags(state.flags)
