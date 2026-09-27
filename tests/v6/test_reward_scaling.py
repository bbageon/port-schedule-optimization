"""Reward units must follow measured data, preserve money, and survive reloads."""
import json

import numpy as np
import pytest
import torch

from yard_rl.v6.ppo.checkpoint import load_policy, save_checkpoint
from yard_rl.v6.ppo.model import BlockPolicy, encode
from yard_rl.v6.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v6.reward.scaling import REFERENCE, fit_reference_scale, reference_scaling


def test_reference_carries_warmup_but_fits_only_registered_window():
    # z = -2, -5, -8.5, -12.25; last two have mean -10.375, std 1.875.
    fit = fit_reference_scale([(0, 0), (60, 2), (120, 6), (180, 12), (240, 20)],
                              gamma=.5, start_s=120, end_s=240)
    assert fit['samples'] == 2
    assert fit['scale_krw'] == 1.875
    assert fit['mean_discounted_return_krw'] == -10.375


def test_currency_change_does_not_change_normalized_rewards():
    rows = [(0, 0), (60, 2), (120, 6), (180, 12), (240, 20)]
    reference = fit_reference_scale(rows, gamma=.5, start_s=120, end_s=240)
    scaled = fit_reference_scale([(t, c*1000) for t,c in rows],
                                 gamma=.5, start_s=120, end_s=240)
    np.testing.assert_allclose(-np.diff([r[1] for r in rows])/reference['scale_krw'],
                              -np.diff([r[1]*1000 for r in rows])/scaled['scale_krw'])


@pytest.mark.parametrize('rows', [
    [(0,0),(60,0),(120,0)], [(0,0),(60,2),(120,1)],
    [(0,0),(60,2),(60,3)], [(0,0),(60,float('nan')),(120,3)],
    [(0,0),(60,2)], [(60,0),(120,2)],
])
def test_bad_reference_never_falls_back_to_an_arbitrary_money_constant(rows):
    with pytest.raises(ValueError):
        fit_reference_scale(rows, start_s=0, end_s=120)


def test_actual_reference_reproduces_scale_and_honors_discount():
    trace = json.loads(REFERENCE.read_text(encoding='utf-8'))
    fit = reference_scaling()
    assert fit['samples'] == 2880 and fit['scale_krw'] != 1_000_000
    raw = np.asarray(trace['rows'])
    acc, values = 0., []
    for n in range(1, len(raw)):
        acc = .999*acc - (raw[n,1]-raw[n-1,1])
        if raw[n,0] > 86400:
            values.append(acc)
    assert fit['scale_krw'] == pytest.approx(np.std(values), rel=1e-12)
    assert PPOConfig().reward_scale_krw == fit['scale_krw']
    assert PPOConfig(gamma=.99).reward_scale_krw == reference_scaling(gamma=.99)['scale_krw']
    assert PPOConfig(gamma=.99).reward_scale_krw != fit['scale_krw']


def test_cpu_reward_preserves_currency_and_does_not_clip_large_losses():
    rt = PPORuntime(BlockPolicy(), training=False)
    rt.bids = ['b']
    rt.states_at = lambda t: encode([[0]], 'state')
    costs = {0: 25., 60: 25.+rt.config.reward_scale_krw*100, 120: 25.+rt.config.reward_scale_krw*300}
    rt.read_cost = costs.__getitem__
    for t in costs:
        rt.boundary(t)
    assert rt.total_reward == pytest.approx(-300)
    assert -rt.total_reward*rt.config.reward_scale_krw == pytest.approx(rt.cost_krw-rt.initial_cost)


def test_checkpoint_units_are_explicit_and_mismatched_reuse_is_rejected(tmp_path):
    old = PPORuntime(BlockPolicy(), config=PPOConfig(reward_scale_krw=1_000_000))
    save_checkpoint(tmp_path/'old.pt', old)
    loaded = load_policy(tmp_path/'old.pt')
    with pytest.raises(ValueError, match='Checkpoint reward scale differs'):
        PPORuntime(loaded)
    replay = PPORuntime(loaded, config=PPOConfig(reward_scale_krw=1_000_000))
    assert replay.config.reward_scale_krw == 1_000_000
    new = PPORuntime(BlockPolicy())
    save_checkpoint(tmp_path/'new.pt', new)
    reloaded = PPORuntime(load_policy(tmp_path/'new.pt'))
    for key, value in new.policy.state_dict().items():
        assert torch.equal(value, reloaded.policy.state_dict()[key])
    data = torch.load(tmp_path/'new.pt', weights_only=True)
    assert data['reward_normalization']['reference_sha256'] == reference_scaling()['reference_sha256']


def test_array_and_cpu_use_the_same_reference_at_real_boundaries():
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from yard_rl.v6.gpu import ppo_runtime as pr
    from yard_rl.v6.gpu.train import TrainConfig
    cpu = PPOConfig()
    cfg = pr.RuntimeConfig(n_blocks=1, cmax=2, amax=2, training=False)
    assert cfg.reward_scale_krw == cpu.reward_scale_krw == TrainConfig().reward_scale_krw
    assert TrainConfig(gamma=.99).reward_scale_krw == PPOConfig(gamma=.99).reward_scale_krw
    st = pr.new_state(cfg)
    states, values = jnp.zeros((1,37)), jnp.zeros(1)
    step = jax.jit(lambda s,t,c: pr.boundary(s,cfg,t,c,states,values))
    st, _ = step(st, 0., 25.)
    st, out = step(st, 60., 25.+cpu.reward_scale_krw*3)
    assert float(out.reward) == pytest.approx(-3., rel=1e-12)
    assert float(st.total_reward) == pytest.approx(-3., rel=1e-12)


def test_array_loader_cannot_silently_reinterpret_old_critic(tmp_path):
    import runpy
    from pathlib import Path
    load_net = runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/v6/train_v6.py'))['load_net']
    old = PPORuntime(BlockPolicy(), config=PPOConfig(reward_scale_krw=1_000_000))
    path = tmp_path/'old.pt'
    save_checkpoint(path, old)
    with pytest.raises(ValueError, match='Checkpoint reward scale differs'):
        load_net(str(path), net_seed=1, hidden=64)
    net, _ = load_net(str(path), net_seed=1, hidden=64, reward_scale_krw=1_000_000)
    assert all(np.isfinite(np.asarray(p)).all() for p in net)


def test_old_preregistration_cannot_silently_receive_new_reward_units(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from yard_rl.v6.ppo import workload_experiment as experiment
    monkeypatch.setattr(experiment, 'code_stamp', lambda: {'test_only': True})
    prereg = tmp_path/'old-prereg.json'
    prereg.write_text(json.dumps({'schema': 'yr331.pilot.v1'}), encoding='utf-8')
    args = SimpleNamespace(prereg=prereg, output=tmp_path/'run', seed=9900628,
                            loads=[7500]*3, reward_scale_krw=None)
    with pytest.raises(ValueError, match='historical scale'):
        experiment.run(args)
    assert not args.output.exists()
