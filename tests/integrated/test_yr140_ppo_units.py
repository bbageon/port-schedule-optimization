"""YR-140 — PPO 단위 계약 고정 (14차 피드백): GAE 단위 일관 + 2행동 학습 방향."""
from __future__ import annotations

import random

import pytest
import torch

from yard_rl.experiments.yr090_dense_vessel import SCALE
from yard_rl.experiments.yr139_blockq_v4_ppo import Critic, _gae, ppo_update
from yard_rl.integrated.joint_distill import JointPairNet


def test_gae_zero_advantage_when_value_perfect():
    """가치가 (scaled) 미래수익을 정확히 맞히면 advantage ≈ 0 — 단위 혼합이면 실패한다."""
    rewards = [-10.0, -30.0, -20.0]                     # 원 단위 보상 (비용)
    rtg = [sum(rewards[i:]) / SCALE for i in range(3)]  # scaled return-to-go = 가치 정답
    trans = [([[0.0]], 0, 0.0, rtg[i], rewards[i]) for i in range(3)]
    adv, ret = _gae(trans)
    assert all(abs(a) < 1e-9 for a in adv)              # 구판(원 단위 r)은 크게 어긋남
    assert ret[0] == pytest.approx(sum(rewards) / SCALE)


def test_two_action_one_update_increases_better_prob():
    """비용 낮은 행동으로 1회 학습 → 그 행동의 선택확률 증가 (14차 지정 단위 테스트)."""
    torch.manual_seed(7)
    actor, critic = JointPairNet(250), Critic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    rng0 = random.Random(0)
    rows = [[rng0.uniform(-1, 1) for _ in range(250)] for _ in range(2)]
    x = torch.tensor(rows, dtype=torch.float32)

    def prob0():
        with torch.no_grad():
            logits, _ = actor(x)
        return float(torch.softmax(-logits, dim=0)[0])

    p_before = prob0()
    with torch.no_grad():
        logits, _ = actor(x)
        dist = torch.distributions.Categorical(logits=-logits)
        lp = [float(dist.log_prob(torch.tensor(i))) for i in (0, 1)]
    batch = []
    for _ in range(8):                                   # 행동0 = 비용 2 / 행동1 = 비용 40
        for act, r in ((0, -2.0), (1, -40.0)):
            adv, ret = _gae([(rows, act, lp[act], 0.0, r)])
            batch.append((rows, act, lp[act], adv[0], ret[0]))
    ppo_update(actor, critic, opt, batch, random.Random(1))
    assert prob0() > p_before + 1e-4                     # 더 싼 행동의 확률이 오른다
