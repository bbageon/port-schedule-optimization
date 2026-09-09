"""Boundary and failure-reporting regressions found during the complete v5 audit."""
import copy

import numpy as np
import pytest
import torch

from yard_rl.v5.ppo.buffer import Choice, Interval, gae
from yard_rl.v5.ppo.model import BlockPolicy, encode
from yard_rl.v5.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v5.ppo.update import update


@pytest.mark.parametrize("field", ["rollout_intervals", "epochs", "minibatch_size"])
@pytest.mark.parametrize("value", [1.5, True])
def test_count_settings_reject_non_integers_before_running(field, value):
    with pytest.raises(ValueError, match="integer"):
        PPOConfig(**{field: value})


def test_early_stop_reports_the_divergence_that_actually_stopped_training():
    torch.manual_seed(303)
    p = BlockPolicy()
    config = PPOConfig(epochs=2)
    opt = torch.optim.Adam(p.parameters(), lr=config.learning_rate)
    states = encode([[0] * 8], "state")
    rows = encode([[0], [1]], "seller")
    mask = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        logp = float(p.distribution(rows, mask).log_prob(torch.tensor(0)))
        values = p.value(states).numpy().copy()
    choice = Choice("seller", 0, rows, mask, 0, logp - 1)
    batch = [Interval(0, 60, states, values, [[choice]], -1)]
    before = copy.deepcopy(p.state_dict())
    rep = update(p, opt, batch, np.zeros(1), config, np.random.default_rng(303))
    assert rep["minibatches"] == 0
    assert rep["max_kl"] > config.target_kl
    assert rep["early_stopped"] is True
    assert all(torch.equal(before[k], v) for k, v in p.state_dict().items())


def test_gae_rejects_discontinuous_collection_instead_of_joining_unrelated_states():
    states = encode([[0] * 8], "state")
    batch = [Interval(0, 60, states, np.zeros(1), [[]], -1),
             Interval(120, 180, states, np.zeros(1), [[]], -2)]
    with pytest.raises(ValueError, match="contiguous"):
        gae(batch, [0], gamma=1, lam=1, time_unit_s=60)


@pytest.mark.parametrize("t", [float("nan"), float("inf"), -1.0])
def test_invalid_review_time_is_rejected_before_reading_world(t):
    rt = PPORuntime(BlockPolicy())
    def must_not_read(t):
        pytest.fail("Invalid clock was passed into state collection")
    rt.states_at = must_not_read
    with pytest.raises(ValueError, match="time"):
        rt.boundary(t)


@pytest.mark.parametrize("unit", [float("nan"), float("inf")])
def test_nonfinite_discount_time_unit_is_rejected(unit):
    row = Interval(0, 60, encode([[0]], "state"), np.zeros(1), [[]], -1)
    with pytest.raises(ValueError):
        gae([row], [0], gamma=.9, lam=.9, time_unit_s=unit)


def test_gae_does_not_broadcast_one_blocks_value_across_two_blocks():
    row = Interval(0, 60, encode([[0], [0]], "state"), np.zeros(1), [[], []], -1)
    with pytest.raises(ValueError, match="block dimensions"):
        gae([row], [0, 0], gamma=1, lam=1, time_unit_s=60)


def test_true_terminal_allows_next_episode_but_cuts_future_rewards():
    states = encode([[0]], "state")
    first = Interval(0, 60, states, np.zeros(1), [[]], -1, terminated=True)
    next_episode = Interval(0, 60, states, np.zeros(1), [[]], -10)
    _, returns = gae([first, next_episode], [2], gamma=1, lam=1, time_unit_s=60)
    assert returns[:, 0].tolist() == [-1, -8]
