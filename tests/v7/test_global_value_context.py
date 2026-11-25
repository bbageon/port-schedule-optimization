from dataclasses import replace
from types import SimpleNamespace as NS
import sys
from pathlib import Path

import pytest
import torch

from yard_rl.v7.crane.actor_critic import GraphState, pack_states
from yard_rl.v7.crane.global_context import QueueSnapshot, snapshot_terminal, expand_value_net
from yard_rl.v7.crane.on_policy import DecisionSample, OnPolicyResolver
from yard_rl.v7.crane.value_normalization import NormalizedValueNet, normalized_ppo_update

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v7'))
from global_value_samples import encode_sample, decode_sample


def state():
    return GraphState(torch.randn(4, 10), torch.arange(4), torch.arange(4) % 2, .5)


def terminal(clock=60.):
    return NS(blocks={b:NS(clock=clock, kpis=NS(waiting_count=lambda n=n:n))
                      for b, n in [('A', 2), ('B', 3)]}, _terminal=set())


def test_snapshot_rejects_async_future_and_wrong_sum():
    t = terminal(); snap = snapshot_terminal(t, 60.)
    assert snap.total == 5 and snap.observation(71., 'B')['age_s'] == 11
    t.blocks['A'].clock = 61.
    with pytest.raises(ValueError, match='future'): snapshot_terminal(t, 60.)
    t.blocks['A'].clock = 59.
    with pytest.raises(ValueError, match='synchronized'): snapshot_terminal(t, 60.)
    with pytest.raises(ValueError, match='sum'): QueueSnapshot(60., (('A', 2),), 3)
    with pytest.raises(ValueError, match='future'): snap.observation(59., 'A')
    with pytest.raises(ValueError, match='missing'): snap.observation(60., 'C')
    assert QueueSnapshot(0., (('A', 21100),), 21100).feature > 1


@pytest.mark.parametrize('kind', ['mlp', 'transformer'])
def test_extended_zero_column_and_actor_only_local(kind, tmp_path):
    torch.manual_seed(21)
    old = NormalizedValueNet(kind=kind).eval(); new = expand_value_net(old)
    s = state(); g = replace(s, global_waiting=2.)
    with torch.no_grad():
        lo, vo = old(pack_states([s])); ln, vn = new(pack_states([g]))
        assert torch.equal(lo, ln) and torch.allclose(vo, vn, atol=2e-6)
        new.critic[0].weight[:, -1].fill_(.2)
        low = new(pack_states([replace(s, global_waiting=0.)]))
        high = new(pack_states([g]))
        assert torch.equal(low[0], high[0])
        assert not torch.allclose(low[1], high[1])
    new.save(tmp_path/'net.pt')
    restored = NormalizedValueNet.load(tmp_path/'net.pt').eval()
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(high, restored(pack_states([g]))))
    with pytest.raises(ValueError, match='missing'): new(pack_states([s]))
    with pytest.raises(ValueError, match='mix'): pack_states([s, g])


def test_resolver_requires_observation_and_uses_real_block_identity():
    net = expand_value_net(NormalizedValueNet())
    r = OnPolicyResolver(net, global_context_mode='global')
    t = terminal()
    with pytest.raises(ValueError, match='Missing'): r.global_state_context(t.blocks['A'])
    r.observe_global_epoch(t, 60.)
    v, obs = r.global_state_context(t.blocks['B'])
    assert obs['decision_block'] == 'B' and v == obs['feature']
    z = OnPolicyResolver(net, global_context_mode='zero'); z.observe_global_epoch(t, 60.)
    v, obs = z.global_state_context(t.blocks['B'])
    assert v == 0 and obs['total'] == 5
    t.blocks['B'].clock = 59.
    with pytest.raises(ValueError, match='future'): r.global_state_context(t.blocks['B'])


def test_context_roundtrip_is_strict_and_normalized_update_replays():
    torch.manual_seed(45)
    net = expand_value_net(NormalizedValueNet()).eval()
    obs = QueueSnapshot(43160., (('A', 5),), 5).observation(43200., 'A')
    st = replace(state(), global_waiting=obs['feature'])
    with torch.no_grad():
        logits, value = net(pack_states([st]))
        lp = torch.log_softmax(logits[0], 0)
    sample = DecisionSample(st, [[0,1,2,3]], [0], float(lp[0]), float(value[0]), 0., -1., obs)
    recovered = decode_sample(encode_sample(sample))
    assert recovered.global_observation == obs and recovered.state.global_waiting == st.global_waiting
    damaged = encode_sample(sample); damaged['global_observation'] = dict(obs, age_s=0.)
    with pytest.raises(ValueError, match='inconsistent'): decode_sample(damaged)
    corpus = [replace(sample, target=-float(i)) for i in range(8)]
    result = normalized_ppo_update(net, torch.optim.Adam(net.parameters(), lr=.0003),
                                  corpus, seed=31, max_samples=8, epochs=1, batch_size=4)
    assert result['replay_logp_max_error'] < 2e-5
    assert result['normalization']['logit_preservation_max_error'] == 0


def test_value_head_restart_learns_a_new_unit_without_moving_action_logits():
    from yard_rl.v7.crane.value_normalization import fit_value_head, restart_value_head
    torch.manual_seed(5)
    net = expand_value_net(NormalizedValueNet().eval())
    net.update_statistics(torch.tensor([1e9, 3e9]))  # statistics left by an earlier reward unit
    samples = [DecisionSample(replace(state(), time_fraction=i/96, global_waiting=.3), [], [], 0., 0., 0.,
                              target=-30.*(1-i/96)) for i in range(96)]
    batch = pack_states([s.state for s in samples])
    with torch.no_grad():
        logits = net(batch)[0]
    rng = torch.get_rng_state()
    restart_value_head(net, seed=7)
    assert torch.equal(rng, torch.get_rng_state()) and net.target_count.item() == 0
    info = fit_value_head(net, samples, seed=8, epochs=200, batch_size=32)
    with torch.no_grad():
        after, value = net(batch)
    assert torch.equal(logits, after)
    assert info['normalized_rmse_after'] < .5 * info['normalized_rmse_before']
    assert net.target_count.item() == 96 and abs(net.target_mean.item() + 15.15625) < 1e-9
    assert float((value - torch.tensor([s.target for s in samples])).abs().mean()) < 4.5
    with pytest.raises(ValueError):
        fit_value_head(net, [], seed=1)


def test_quay_work_is_a_critic_only_column_after_the_global_context():
    torch.manual_seed(3)
    net = NormalizedValueNet(global_value_context=True, vessel_work_context=True).eval()
    base = replace(state(), global_waiting=.4)
    low, high = replace(base, vessel_work=0.), replace(base, vessel_work=2.)
    with torch.no_grad():
        net.critic[0].weight[:, -1].fill_(.3)
        a, b = net(pack_states([low])), net(pack_states([high]))
    assert torch.equal(a[0], b[0]) and not torch.allclose(a[1], b[1])
    with pytest.raises(ValueError, match='mix'): pack_states([low, base])
    with pytest.raises(ValueError, match='beside'): pack_states([replace(state(), vessel_work=1.)])
    with pytest.raises(ValueError, match='Invalid'): pack_states([replace(base, vessel_work=-1.)])
    with pytest.raises(ValueError, match='missing'): net(pack_states([base]))
    with pytest.raises(ValueError, match='requires'): NormalizedValueNet(vessel_work_context=True)
    old = NormalizedValueNet(global_value_context=True)
    assert 'vessel_work_context' not in old.config
    with pytest.raises(ValueError): old(pack_states([low]))


def test_resolver_needs_a_quay_work_source_for_a_quay_work_model():
    net = NormalizedValueNet(global_value_context=True, vessel_work_context=True)
    with pytest.raises(ValueError, match='quay work'):
        OnPolicyResolver(net, global_context_mode='global')

    class Tracking(OnPolicyResolver):
        def critic_vessel_work(self, sim):
            return .5
    assert Tracking(net, global_context_mode='global').critic_vessel_work(None) == .5
    assert OnPolicyResolver(NormalizedValueNet(global_value_context=True),
                            global_context_mode='global').critic_vessel_work(None) is None


def test_update_threads_restores_the_previous_count_even_on_error():
    from yard_rl.v7.crane.update_threads import update_threads
    before = torch.get_num_threads()
    with update_threads(2) as n:
        assert n == 2 and torch.get_num_threads() == 2
    assert torch.get_num_threads() == before
    with pytest.raises(RuntimeError):
        with update_threads(3):
            raise RuntimeError("boom")
    assert torch.get_num_threads() == before
    with pytest.raises(ValueError):
        with update_threads(0):
            pass
