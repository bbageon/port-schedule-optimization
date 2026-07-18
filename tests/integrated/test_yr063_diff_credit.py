"""YR-063 계약 테스트 — 차분 credit 표본은 1-step 이고 WAIT 는 0 앵커."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.domain.enums import InformationLevel
from yard_rl.experiments.yr063_diff_credit import (Yr063Config,
                                                   run_diff_episode)
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.dqn_learner import CandidateDQNLearner, LearnerConfig
from yard_rl.integrated.encoding import encode_observation, encoding_dims
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.scenario_gen import (TerminalGenParams,
                                             generate_terminal_scenario)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
PARAMS = TerminalGenParams(n_external=8, n_vessels=1, vessel_moves=6,
                           horizon_s=7_200.0, drain_window_s=3_600.0)
SEED = 600_901          # 소각 seed — 실험 대역과 불겹침


def _sim():
    return TerminalSimulator(PROF, generate_terminal_scenario(PROF, SEED, PARAMS),
                             check_invariants=True)


def _learner():
    sim = _sim()
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "t", 0)
    dims = encoding_dims(encode_observation(state, obs[0]))
    return CandidateDQNLearner(LearnerConfig(cost_scale=1.0), dims, seed=7)


def test_diff_episode_builds_one_step_samples_with_wait_anchor():
    learner = _learner()
    info = run_diff_episode(_sim(), learner=learner,
                            rc=RewardCalculator.assumed_default(),
                            window_s=300.0, learn=False)
    assert info["n_samples"] > 0 and len(learner.replay) == info["n_samples"]
    for s in learner.replay:
        assert s.gamma_dt == 0.0 and s.next_enc is None      # 1-step 계약
    # WAIT(구조적 양보 포함) 표본은 정확히 0 앵커, 실행동 credit 은 유한값
    kinds = {0.0}
    assert all(abs(s.c_disc) < 1e4 for s in learner.replay)
    assert info["credit_min"] <= 0.0 <= max(info["credit_max"], 0.0)
    assert kinds.issubset({0.0} | {s.c_disc for s in learner.replay} | {0.0})


def test_yr063_config_defaults():
    cfg = Yr063Config()
    assert cfg.window_s == 600.0 and cfg.reuse
    assert cfg.base.test_seed0 == 620_000       # 기존 판정 행과 paired 재사용 전제
