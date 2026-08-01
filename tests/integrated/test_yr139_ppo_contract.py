"""YR-139 — PPO 보상 계약 고정: Σ 구간비용 = 평가 총비용 (등식·검열 일관)."""
from __future__ import annotations

import random

import pytest
import torch

from yard_rl.experiments.yr090_dense_vessel import BASE
from yard_rl.experiments.yr139_blockq_v4_ppo import Critic, phi_v2, run_episode
from yard_rl.integrated.encoding import StateNorm
from yard_rl.integrated.joint_distill import JointPairNet


@pytest.fixture(scope="module")
def setup():
    ck0 = torch.load("outputs/reports/yr125_diff_credit/diff1_s99000/rl_net.pt",
                     map_location="cpu")
    norm = StateNorm(refs=ck0["norm_refs"])
    torch.manual_seed(0)
    return JointPairNet(250), Critic(), norm


def test_reward_identity_equals_eval_total(setup):
    """핵심 등식: −Σ r_k = Φ(end) − Φ(0) = 에피소드 평가 총비용 (텔레스코핑 정확)."""
    actor, critic, norm = setup
    trans, st = run_episode(actor, critic, norm, "high-tight", BASE["high-tight"],
                            random.Random(0), sample=False)
    assert trans, "결정이 없으면 계약 검증 불가"
    total_reward = sum(t[4] for t in trans)
    assert -total_reward == pytest.approx(st["total"], abs=1e-6)


def test_phi_starts_near_zero_and_monotone(setup):
    from yard_rl.experiments.yr136_softplus_contract import _sim_contract
    s = _sim_contract("mid-loose", BASE["mid-loose"])
    p0 = phi_v2(s)
    assert p0 == pytest.approx(0.0, abs=1e-6)      # t0: 관측 트럭 체류 0
    s.run_until_decision()
    p1 = phi_v2(s)
    assert p1 >= p0 - 1e-9                          # 단조 비감소 (검열 일관)


def test_determinism_argmax(setup):
    actor, critic, norm = setup
    _, s1 = run_episode(actor, critic, norm, "mid-tight", BASE["mid-tight"],
                        random.Random(1), sample=False)
    _, s2 = run_episode(actor, critic, norm, "mid-tight", BASE["mid-tight"],
                        random.Random(2), sample=False)
    assert s1["total"] == pytest.approx(s2["total"])   # argmax = rng 무관 결정론
