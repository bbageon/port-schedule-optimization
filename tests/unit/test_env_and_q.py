"""YardEnv·Q-learning 단위 테스트 — 결정론·SMDP 할인·정보누출 가드."""
import pytest

from yard_rl.domain.enums import InformationLevel, JobFlow, PriorityRule
from yard_rl.domain.models import Job
from yard_rl.envs.info_filter import assert_no_leakage, is_visible
from yard_rl.envs.yard_env import YardEnv
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams, generate
from yard_rl.policies.baselines import FixedRulePolicy
from yard_rl.policies.q_learning import QLearningAgent, QLearningConfig
from yard_rl.experiments.runner import run_episode

P = load_profile("configs/terminals/poc_single_crane.yaml")
SMALL = GenParams(n_external=30, n_vessel=3, fill_ratio=0.35)


def test_visibility_levels():
    j = Job(job_id="J", flow=JobFlow.GATE_OUT, release_time=0.0,
            actual_gate_in=100.0, actual_block_arrival=500.0, target_container="C")
    assert not is_visible(j, 400.0, InformationLevel.BLOCK_ARRIVAL)
    assert is_visible(j, 500.0, InformationLevel.BLOCK_ARRIVAL)
    assert is_visible(j, 100.0, InformationLevel.GATE_IN)      # Exp-2: 게이트부터
    assert not is_visible(j, 99.0, InformationLevel.GATE_IN)
    assert is_visible(j, 0.0, InformationLevel.PRE_ADVICE)      # Exp-3: 사전정보
    with pytest.raises(RuntimeError, match="정보 누출"):
        assert_no_leakage([j], 400.0, InformationLevel.BLOCK_ARRIVAL)


def test_env_episode_deterministic():
    sc = generate(P, seed=9, params=SMALL)
    outs = []
    for _ in range(2):
        env = YardEnv(P, info_level=InformationLevel.BLOCK_ARRIVAL)
        r = run_episode(FixedRulePolicy(PriorityRule.NEAREST_JOB), env, sc)
        outs.append((r.metrics["queue_area_h"], r.metrics["rehandles"],
                     r.metrics["travel_km"], env.n_steps))
    assert outs[0] == outs[1]


def test_env_rejects_masked_action():
    sc = generate(P, seed=9, params=SMALL)
    env = YardEnv(P)
    state, info = env.reset(sc)
    assert not info.action_mask[PriorityRule.EARLIEST_PROVIDED_ARRIVAL]  # Exp-1 mask
    with pytest.raises(ValueError, match="mask 위반"):
        env.step(int(PriorityRule.EARLIEST_PROVIDED_ARRIVAL))


def test_smdp_discount_and_update():
    cfg = QLearningConfig(alpha=0.5, gamma_ref=0.9, ref_s=60.0)
    ag = QLearningAgent(cfg, seed=1)
    s, s2 = (0,) * 7, (1,) * 7
    mask = [True] * 9
    ag.table.row(s2)[int(PriorityRule.FIFO)] = 2.0
    ag.update(s, 0, r=-1.0, s2=s2, mask2=mask, elapsed_s=120.0, done=False)
    # gamma_eff = 0.9^(120/60) = 0.81 → target = -1 + 0.81*2 = 0.62 → Q = 0.5*0.62
    assert abs(ag.table.q[s][0] - 0.31) < 1e-9
    # done 이면 bootstrap 없음
    ag2 = QLearningAgent(cfg, seed=1)
    ag2.update(s, 0, r=-1.0, s2=None, mask2=[False] * 9, elapsed_s=60.0, done=True)
    assert abs(ag2.table.q[s][0] - (-0.5)) < 1e-9


def test_unvisited_state_fallback_is_longest_wait():
    ag = QLearningAgent(QLearningConfig(), seed=0)
    mask = [True] * 9
    assert ag.act((9, 9, 9, 9, 9, 9, 9), mask) == int(PriorityRule.LONGEST_WAIT)
    mask2 = [False] * 9
    mask2[int(PriorityRule.FIFO)] = True
    assert ag.act((9,) * 7, mask2) == int(PriorityRule.FIFO)


def test_greedy_prefers_learned_value():
    ag = QLearningAgent(QLearningConfig(), seed=0)
    s = (0,) * 7
    row = ag.table.row(s)
    row[int(PriorityRule.NEAREST_JOB)] = 1.5
    row[int(PriorityRule.FIFO)] = 0.2
    mask = [True] * 9
    assert ag.act(s, mask) == int(PriorityRule.NEAREST_JOB)
    # 최고값 action 이 mask 되면 차선
    mask[int(PriorityRule.NEAREST_JOB)] = False
    assert ag.act(s, mask) == int(PriorityRule.FIFO)
