from types import SimpleNamespace

import pytest
import torch

from yard_rl.v7.crane.on_policy import action_statistics, batched_action_statistics
from yard_rl.v7.crane.value_normalization import NormalizedValueNet, restart_value_head


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA runtime unavailable"))])
def test_batched_legal_choices_and_gradients(device):
    samples = [SimpleNamespace(masks=[[0, 2, 1], [1]], choices=[2, 1]),
               SimpleNamespace(masks=[[3, 1]], choices=[1])]
    logits = torch.tensor([[2., -6., 2.4, -torch.inf], [9., 2., -torch.inf, 1.]], requires_grad=True)
    expected = action_statistics(logits, samples)
    sum(x.sum() for x in expected).backward()
    scores = logits.detach().to(device).requires_grad_()
    result = batched_action_statistics(scores, samples)
    sum(x.sum() for x in result).backward()
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual.cpu(), reference)
    torch.testing.assert_close(scores.grad.cpu(), logits.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_restart_critic_from_same_cpu_random_draws_on_cuda():
    cpu, gpu = NormalizedValueNet(), NormalizedValueNet().cuda()
    original = torch.get_rng_state()
    for net in (cpu, gpu):
        restart_value_head(net, seed=140)
        assert torch.equal(original, torch.get_rng_state())
    for a, b in zip(cpu.critic.parameters(), gpu.critic.parameters()):
        torch.testing.assert_close(a, b.cpu(), atol=0, rtol=0)
