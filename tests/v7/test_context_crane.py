import pytest
import torch

from yard_rl.v7.crane.context import ContextCraneNet
from yard_rl.v7.crane.policy import CraneNet


@pytest.mark.parametrize('kind',['mean_context','star_attention'])
def test_context_order_groups_checkpoint_and_gradients(kind):
    torch.manual_seed(123)
    net=ContextCraneNet(kind)
    x=torch.randn(33,10);g=torch.tensor([0]*17+[1]*16)
    score=net(x,g)
    permutation=torch.randperm(len(x))
    assert torch.allclose(net(x[permutation],g[permutation]),score[permutation],atol=1e-6)
    assert torch.allclose(net(x[:17],g[:17]),score[:17],atol=1e-6)
    changed=x.clone();changed[17:]+=100
    assert torch.allclose(net(changed,g)[:17],score[:17],atol=1e-6)
    restored=CraneNet.from_checkpoint(dict(crane=net.state_dict(),model_config=net.model_config()))
    assert torch.equal(restored(x,g),score)
    (score[0]-score[1]).square().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    with pytest.raises(ValueError): net(x)


def test_all_candidates_above_previous_cap():
    net=ContextCraneNet('star_attention')
    x=torch.randn(1200,10);g=torch.zeros(1200,dtype=torch.long)
    assert net(x,g).shape==(1200,)


def test_context_policy_runs_the_actual_resolver():
    from yard_rl.v7.stage.episode import run_episode
    torch.set_num_threads(1);torch.manual_seed(81)
    ep=run_episode(load=100,seed=9949001,dispatcher='RL_CRANE',
        crane_net=ContextCraneNet('star_attention'),candidate_cap=None,
        reuse_rehandle_slots=True,decision_probe_shortcircuit=True)
    assert ep.admitted==100 and ep.policy_exceptions==0
    assert ep.phi_krw>0 and ep.rollout_worlds==0


def test_context_capture_matches_compared_features():
    from yard_rl.v7.crane.policy import add_store_input
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    torch.set_num_threads(1);torch.manual_seed(81)
    net=add_store_input(CraneNet(feature_mode='feasible'));net.capture_context=True
    ep=run_episode(load=100,seed=9949002,dispatcher='RL_CRANE',crane_net=net,
        candidate_cap=None,reuse_rehandle_slots=True,decision_probe_shortcircuit=True,
        crane_budget=RolloutBudget(max_labels=6,identity_checks=0),horizon_s=1800.,workers=1)
    contexts=ep.crane_stats['candidate_contexts']
    assert len(contexts)>0 and 2*len(contexts)==len(ep.crane_labels)
    assert ep.policy_exceptions==ep.crane_stats['hook_errors']==0
    for i,c in enumerate(contexts):
        assert c['rows'][c['pair'][0]]==ep.crane_labels[2*i].row
        assert c['rows'][c['pair'][1]]==ep.crane_labels[2*i+1].row
