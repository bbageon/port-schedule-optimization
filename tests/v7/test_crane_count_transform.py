"""Candidate-count information survives inference, labels and checkpoint restore."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from yard_rl.v7.crane.features import crane_features, encode_candidate_count
from yard_rl.v7.crane.policy import CraneNet, CranePolicy, add_store_input
from test_candidate_recall import state, generate


@pytest.mark.parametrize('transform', ['linear', 'log1p'])
def test_high_load_is_distinguishable_without_changing_other_inputs(transform):
    sim, dp = state()
    gc = generate(sim, dp, None)['YC-A'].items[0]
    rows = []
    for n in (18, 30, 200):
        g = replace(gc, feasible_serve_count=n)
        old = crane_features(sim, 'YC-A', g, feature_mode='feasible')
        new = crane_features(sim, 'YC-A', g, feature_mode='feasible', count_transform=transform)
        assert old[7] == 1.0
        assert new[:7]+new[8:] == old[:7]+old[8:]
        rows.append(new[7])
    assert rows[0] == 1.0 < rows[1] < rows[2]


@pytest.mark.parametrize('transform', ['clipped', 'linear', 'log1p'])
def test_scalar_batch_and_checkpoint_agree_at_uncapped_candidate_count(transform, tmp_path):
    sim, dp = state()
    gb = generate(sim, dp, None)
    torch.manual_seed(9913501)
    net = add_store_input(CraneNet(feature_mode='feasible', count_transform=transform))
    policy = CranePolicy(net)
    pairs = [('YC-A', gc) for gc in gb['YC-A'].items]
    scalar = [policy.rank(sim, *p) for p in pairs]
    batch = policy.rank_many(sim, pairs)
    for a, b in zip(scalar, batch):
        assert a[::2] == b[::2]
        assert a[1] == pytest.approx(b[1], abs=1e-6)
    p = tmp_path/'model.pt'
    torch.save(dict(crane=net.state_dict(), model_config=net.model_config()), p)
    restored = CraneNet.from_checkpoint(torch.load(p, weights_only=True))
    assert restored.model_config() == net.model_config()
    assert CranePolicy(restored).rank_many(sim, pairs) == batch


def test_old_checkpoint_keeps_original_input_semantics():
    net = CraneNet(feature_mode='feasible')
    old_config = net.model_config()
    old_config.pop('count_transform')
    for ck in (dict(crane=net.state_dict(), model_config=old_config),
               dict(crane=net.state_dict(), feature_mode='feasible')):
        restored = CraneNet.from_checkpoint(ck)
        assert restored.count_transform == 'clipped'
        assert all(torch.equal(v, restored.state_dict()[k]) for k,v in net.state_dict().items())


@pytest.mark.parametrize('count', [-1, float('nan'), float('inf')])
def test_invalid_count_is_not_silently_encoded(count):
    with pytest.raises(ValueError):
        encode_candidate_count(count, 'linear')


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError):
        CraneNet(count_transform='typo')


@pytest.mark.parametrize('transform', ['linear', 'log1p'])
def test_collector_uses_same_transform_as_policy_for_both_rows(transform):
    from yard_rl.v7.crane.collect import CraneCollector
    from yard_rl.v7.stage.rollout import RolloutBudget
    from yard_rl.v7.world.contract.schema import CandidateKind
    sim, dp = state()
    # One truck has already waited longer. Fully identical network inputs are
    # correctly rejected by the reachable teacher and cannot test a label pair.
    sim.jobs['T001'].actual_block_arrival = -60.
    gb = generate(sim, dp, None)
    chosen = next(g for g in gb['YC-A'].items if g.kind == CandidateKind.SERVE)
    collector = CraneCollector(SimpleNamespace(), budget=RolloutBudget(max_labels=1,stride=1),
                               horizon_s=10800)
    net = CraneNet(feature_mode='feasible',count_transform=transform)
    collector.attach(CranePolicy(net))
    captured=[]
    collector.pool.submit=captured.append
    hook=collector.hook(SimpleNamespace(blocks={'Y01':sim}),SimpleNamespace(decided=set()),{},{})
    hook(sim,dp,gb,{'YC-A':chosen})
    assert len(captured)==1
    rows=next(iter(collector.rows.values()))
    assert all(r[7]==encode_candidate_count(chosen.feasible_serve_count,transform)>1 for r in rows)
    collector.close()
