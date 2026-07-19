"""YR-013c 차분 표적 QMIX 계약 테스트 — 앵커 회귀·mixer 보정·수집 배관."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.cost_config import RewardCalculator
from yard_rl.integrated.encoding import encode_observation, encoding_dims
from yard_rl.integrated.qmix import (DiffQmixConfig, DiffQmixLearner,
                                     JointDiffSample)
from yard_rl.integrated.scenario_gen import (TerminalGenParams,
                                             generate_terminal_scenario)
from yard_rl.experiments.yr013_diff_qmix import run_diff_qmix_episode

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
PARAMS = TerminalGenParams(n_external=8, n_vessels=1, vessel_moves=6,
                           horizon_s=7_200.0, drain_window_s=3_600.0)


def _sim(seed=600_900):
    return TerminalSimulator(PROF, generate_terminal_scenario(PROF, seed, PARAMS),
                             check_invariants=True)


def _enc(seed=600_900):
    sim = _sim(seed)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "t", 0)
    return encode_observation(state, obs[0])


def test_learn_step_regresses_anchor_and_team():
    """반복 학습으로 loss 감소 — D 앵커·mixer 팀 표적 동시 회귀."""
    enc = _enc()
    dims = encoding_dims(enc)
    lr = DiffQmixLearner(DiffQmixConfig(min_replay=1, batch_size=1), dims, seed=1)
    pos = enc.selectable.index(True)
    lr.replay.append(JointDiffSample((enc, enc), (pos, pos), (-2.0, +1.0), 7.5))
    losses = [lr.learn_step() for _ in range(150)]
    assert losses[0] is not None and losses[-1] < losses[0]


def test_lambda_zero_is_pure_anchor():
    """λ_mix=0 이면 mixer 파라미터가 변하지 않는다 (앵커 전용 퇴화)."""
    enc = _enc()
    dims = encoding_dims(enc)
    lr = DiffQmixLearner(DiffQmixConfig(min_replay=1, batch_size=1,
                                        lambda_mix=0.0), dims, seed=2)
    pos = enc.selectable.index(True)
    lr.replay.append(JointDiffSample((enc,), (pos,), (-1.0,), 3.0))
    before = [p.detach().clone() for p in lr.mixer.parameters()]
    for _ in range(5):
        lr.learn_step()
    after = list(lr.mixer.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_episode_collects_joint_samples_and_completes():
    """수집 드라이버: 결정 단위 표본 적재·WAIT 앵커 0 포함·팀비용 유한."""
    lr = DiffQmixLearner(DiffQmixConfig(min_replay=10_000),
                         encoding_dims(_enc()), seed=3)
    info = run_diff_qmix_episode(_sim(), learner=lr,
                                 rc=RewardCalculator.assumed_default(),
                                 window_s=300.0, epsilon=0.3, learn=False)
    assert info["n_samples"] > 0 and len(lr.replay) == info["n_samples"]
    for s in lr.replay:
        assert len(s.encs) == len(s.action_pos) == len(s.d_targets) >= 1
        assert s.team_cost == s.team_cost         # 유한 (NaN 아님)
        for d in s.d_targets:
            assert abs(d) < 1e6


def test_save_load_roundtrip(tmp_path):
    enc = _enc()
    lr = DiffQmixLearner(DiffQmixConfig(), encoding_dims(enc), seed=4)
    p = tmp_path / "dq.pt"
    lr.save(p)
    lr2 = DiffQmixLearner.load(p)
    s1, s2 = lr.scores_for(enc), lr2.scores_for(enc)
    assert s1.keys() == s2.keys()
    for k in s1:
        assert s1[k] == pytest.approx(s2[k], abs=1e-5)
