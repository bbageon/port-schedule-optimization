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
