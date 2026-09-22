from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yard_rl.v5.ppo.buffer import Interval, gae
from yard_rl.v5.ppo.checkpoint import load_policy, save_checkpoint
from yard_rl.v5.ppo.crane import joint_mask
from yard_rl.v5.ppo.model import BlockPolicy, encode
from yard_rl.v5.ppo.runtime import PPOConfig, PPORuntime
from yard_rl.v5.ppo.update import clipped_surrogate, update
from yard_rl.v5.world.contract.schema import CandidateKind as Kind


def test_mask_is_respected_and_replay_logprob_identical():
    torch.manual_seed(42)
    p = BlockPolicy()
    rows = encode([[1, 2], [3, 4], [5, 6]], "crane")
    d = p.distribution(rows, [True, False, True])
    assert d.probs[1] == 0
    assert 1 not in d.sample((200,)).tolist()
    assert torch.equal(d.log_prob(torch.tensor(2)),
                       p.distribution(rows, [True, False, True]).log_prob(torch.tensor(2)))
    with pytest.raises(ValueError):
        p.distribution(rows, [False] * 3)


def test_buyer_actions_distinct_even_for_zero_offer():
    x = encode([[0] * 16, [0] * 16], "buyer")
    assert not torch.equal(x[0], x[1])


def row(start, end, reward, value=0.0, terminated=False):
    return Interval(start, end, encode([[0] * 8], "state"), np.array([value]),
                    [[]], reward, terminated)


def test_gae_time_discount_and_truncation_bootstrap():
    a, r = gae([row(0, 120, -2, 3)], [10], gamma=0.9, lam=1, time_unit_s=60)
    assert a[0, 0] == pytest.approx(-2 + .81 * 10 - 3)
    assert r[0, 0] == pytest.approx(-2 + .81 * 10)
    a, r = gae([row(0, 120, -2, 3, True)], [10], gamma=.9, lam=1, time_unit_s=60)
    assert r[0, 0] == -2


def test_gae_full_return_and_true_terminal_cuts_trace():
    _, r = gae([row(0, 60, -2), row(60, 120, -3, terminated=True)], [100],
               gamma=1, lam=1, time_unit_s=60)
    assert r[:, 0].tolist() == [-5, -3]


def test_clipping_both_advantage_signs():
    log_ratio = torch.log(torch.tensor([2.0, .5]))
    assert clipped_surrogate(log_ratio, torch.tensor([1., -1.]), .2).tolist() == pytest.approx([1.2, -.8])


def test_joint_mask_rejects_duplicate_tokens_and_corridor_conflict():
    def candidate(token, feasible=True, wait=False):
        return SimpleNamespace(feasible=feasible, kind=Kind.WAIT if wait else Kind.SERVE,
                               job_ref=None if wait else SimpleNamespace(token=token))
    selected = {"a": candidate("one")}
    def dry(trial):
        return SimpleNamespace(plans={} if trial["b"].token == "collision" else trial)
    sim = SimpleNamespace(dry_run_commit=dry)
    items = [("b", candidate(x)) for x in ("one", "collision", "free")]
    items.append(("b", candidate(None, wait=True)))
    assert joint_mask(sim, items, selected).tolist() == [False, False, True, True]


@pytest.mark.parametrize("change", [{"gamma": 0}, {"reward_scale_krw": 0},
                                   {"epochs": 0}, {"learning_rate": float("nan")}])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        PPOConfig(**change)


def test_checkpoint_roundtrip_and_v4_rejected(tmp_path):
    rt = PPORuntime(BlockPolicy())
    target = tmp_path / "policy.pt"
    save_checkpoint(target, rt)
    loaded = load_policy(target)
    for a, b in zip(rt.policy.parameters(), loaded.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(FileExistsError):
        save_checkpoint(target, rt)
    torch.save({"seller": {}}, tmp_path / "legacy.pt")
    with pytest.raises(ValueError):
        load_policy(tmp_path / "legacy.pt")


def test_sample_actions_defaults_to_training_and_can_be_split():
    """★YR-319 — 고르는 방식이 학습 여부와 **따로** 켜져야 한다.

    YR-306 비교는 `training=False` 하나로 **가중치 고정**과 **최고점 선택**을
    한꺼번에 바꿔 172배 격차의 귀속을 막았다. 기본값은 예전 그대로여야 하고
    (기존 실행 재현), 명시하면 갈려야 한다.
    """
    assert PPORuntime(BlockPolicy()).sample_actions is True
    assert PPORuntime(BlockPolicy(), training=False).sample_actions is False
    frozen_sampling = PPORuntime(BlockPolicy(), training=False, sample_actions=True)
    assert frozen_sampling.training is False and frozen_sampling.sample_actions is True


def test_frozen_sampling_explores_while_argmax_repeats_one_action():
    """같은 가중치라도 추첨은 여러 행동을 내고 최고점 선택은 늘 한 가지만 낸다."""
    def picks(**kwargs):
        torch.manual_seed(5)
        rt = PPORuntime(BlockPolicy(), training=False, **kwargs)
        rt.time_s, rt.index, rt.pending = 0.0, {"b": 0}, [[]]
        rows = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        return {rt.select("crane", "b", 0.0, rows) for _ in range(60)}

    assert len(picks(sample_actions=True)) > 1
    assert len(picks(sample_actions=False)) == 1
    # 갱신도 수집도 없다 — 가중치는 여전히 얼어 있다.
    assert not PPORuntime(BlockPolicy(), training=False,
                          sample_actions=True).collecting_at(0.0)


def test_boundary_telescopes_cost_and_preserves_zero_time_choices():
    rt = PPORuntime(BlockPolicy(), config=PPOConfig(rollout_intervals=99), training=False)
    rt.bids = ["b"]
    rt.states_at = lambda t: encode([[t / 3600]], "state")
    rt.read_cost = lambda t: 100 + 2 * t
    rt.boundary(0)
    rt.boundary(60)
    rt.pending[0].append("sentinel")
    rt.boundary(60)
    assert rt.pending[0] == ["sentinel"]
    rt.boundary(120)
    assert rt.intervals == 2
    assert rt.total_reward == pytest.approx(-240 / rt.config.reward_scale_krw)


def test_shared_policy_updates_from_recorded_action():
    torch.manual_seed(17)
    rt = PPORuntime(BlockPolicy(), config=PPOConfig(epochs=2))
    rt.time_s, rt.index, rt.pending = 0, {"b": 0}, [[]]
    before = [p.detach().clone() for p in rt.policy.parameters()]
    rt.select("seller", "b", 0, [[0, 1], [2, 3]])
    states = encode([[0] * 8], "state")
    values = rt.policy.value(states).detach().numpy()
    interval = Interval(0, 60, states, values, rt.pending, -1)
    rep = update(rt.policy, rt.optimizer, [interval], np.zeros(1), rt.config, rt.rng)
    assert rep["minibatches"] > 0 and np.isfinite(rep["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, rt.policy.parameters()))
