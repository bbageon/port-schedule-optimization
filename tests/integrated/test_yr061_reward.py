"""YR-061 계약 테스트 — 미완료 잔존 페널티는 학습 표적 전용, 평가·기본 거동 불변."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.domain.enums import InformationLevel
from yard_rl.experiments.yr061_reward_redesign import Yr061Config
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig,
                                            run_episode)
from yard_rl.integrated.encoding import encode_observation, encoding_dims
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.scenario_gen import (TerminalGenParams,
                                             generate_terminal_scenario)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
# drain 120s: 도착구간 말미 트럭을 물리적으로 못 끝내는 구성 — backlog>0 보장
# (미학습 정책 실측 backlog=2, seed 600900). 페널티 경로 검증에 미완료가 필요하다.
PARAMS = TerminalGenParams(n_external=24, n_vessels=1, vessel_moves=6,
                           horizon_s=3_600.0, drain_window_s=120.0)
SEED = 600_900          # YR-061 인접 소각 seed (실험 대역 600000~600149 와 불겹침)


def _sim(seed=SEED):
    return TerminalSimulator(PROF, generate_terminal_scenario(PROF, seed, PARAMS),
                             check_invariants=True)


def _dims():
    sim = _sim()
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "t", 0)
    return encoding_dims(encode_observation(state, obs[0]))


def _collect(penalty: float):
    """동일 초기화(seed 고정)·ε=0·learn=False — 궤적 결정론, 차이는 표본 표적뿐."""
    learner = CandidateDQNLearner(
        LearnerConfig(unserved_terminal_cost=penalty, cost_scale=2.0), _dims(), seed=7)
    res = run_episode(_sim(), level=LEVEL, preference=QPreference(),
                      learner=learner, collect=True, learn=False)
    return res


def test_config_rejects_negative_penalty():
    with pytest.raises(ValueError):
        LearnerConfig(unserved_terminal_cost=-0.1)


def test_penalty_zero_is_default_and_checkpoint_roundtrip(tmp_path):
    assert LearnerConfig().unserved_terminal_cost == 0.0
    learner = CandidateDQNLearner(LearnerConfig(unserved_terminal_cost=3.0),
                                  _dims(), seed=1)
    p = tmp_path / "m.pt"
    learner.save(p)
    assert CandidateDQNLearner.load(p).cfg.unserved_terminal_cost == 3.0


def test_penalty_touches_only_terminal_window_samples_not_eval():
    """계약: 평가 total_cost 불변 + 표본은 '마지막 구간을 덮는 창'만 증가."""
    base, pen = _collect(0.0), _collect(5.0)
    assert pen.total_cost == pytest.approx(base.total_cost)   # 평가 지표 불변
    assert pen.backlog == base.backlog and pen.n_decisions == base.n_decisions
    assert base.backlog > 0, "페널티 경로 검증에는 미완료가 있는 시나리오가 필요"
    assert len(base.samples) == len(pen.samples)
    diffs = [(b, q) for b, q in zip(base.samples, pen.samples)
             if q.c_disc != pytest.approx(b.c_disc)]
    assert diffs, "backlog>0 이면 종결 표본 표적이 증가해야 함"
    for b, q in diffs:
        assert q.c_disc > b.c_disc                    # 페널티는 비용 가산(부호 양)
        assert q.gamma_dt == b.gamma_dt and q.action_pos == b.action_pos
    # 마지막 구간을 덮는 표본은 크레인당 정확히 1개 (각 크레인의 말단 창)
    assert all(b.gamma_dt == 0.0 for b, _ in diffs)


def test_yr061_config_guards():
    with pytest.raises(ValueError):
        Yr061Config(penalties=(2.0, 5.0))             # control(0.0) 필수
    with pytest.raises(ValueError):
        Yr061Config(penalties=(0.0, 5.0, 2.0))        # 오름차순 위반
    with pytest.raises(ValueError):
        Yr061Config(train_seed0=300_000)              # 기사용 대역 금지
    assert Yr061Config().penalties[0] == 0.0
    with pytest.raises(ValueError):
        Yr061Config(gammas=(0.99, 1.0))               # control(0.95) 필수
    with pytest.raises(ValueError):
        Yr061Config(gammas=(0.95, 1.0, 0.99))         # 오름차순 위반
    assert Yr061Config(gammas=(0.95, 0.99, 1.0)).gammas[-1] == 1.0
