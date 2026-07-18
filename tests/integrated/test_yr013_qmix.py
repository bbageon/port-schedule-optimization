"""YR-013 QMIX 계약 테스트 — 06 §11 대응 (단조성·퇴화·표적·스티칭·e2e)."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.domain.enums import InformationLevel
from yard_rl.integrated import TerminalSimulator, build_integrated_profile
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.dqn_learner import run_episode
from yard_rl.integrated.encoding import encode_observation, encoding_dims
from yard_rl.integrated.qmix import (JointSample, MonotonicMixer, QmixConfig,
                                     QmixLearner, stitch_joint_samples)
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.scenario_gen import (TerminalGenParams,
                                             generate_terminal_scenario)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
PARAMS = TerminalGenParams(n_external=8, n_vessels=1, vessel_moves=6,
                           horizon_s=7_200.0, drain_window_s=3_600.0)


def _sim(seed=530_900):
    return TerminalSimulator(PROF, generate_terminal_scenario(PROF, seed, PARAMS),
                             check_invariants=True)


def _enc(seed=530_900):
    sim = _sim(seed)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "t", 0)
    return encode_observation(state, obs[0])


def test_mixer_is_monotone_in_agent_utilities():
    """∂Q_tot/∂q_i ≥ 0 — q_i 를 올리면 Q_tot 이 감소하지 않는다 (IGM 전제)."""
    torch.manual_seed(0)
    mixer = MonotonicMixer(n_agents=2, g_dim=10, embed=16)
    g = torch.randn(64, 10)
    pres = torch.ones(64, 2)
    q = torch.randn(64, 2)
    base = mixer(q, pres, g)
    for slot in (0, 1):
        bumped = q.clone()
        bumped[:, slot] += 0.7
        assert (mixer(bumped, pres, g) - base >= -1e-6).all()


def test_mixer_absent_slot_is_inert():
    """presence=0 슬롯은 값이 무엇이든 Q_tot 불변 (1-크레인 결정 퇴화)."""
    torch.manual_seed(1)
    mixer = MonotonicMixer(n_agents=2, g_dim=6, embed=8)
    g = torch.randn(16, 6)
    pres = torch.tensor([[1.0, 0.0]]).repeat(16, 1)
    q1 = torch.randn(16, 2)
    q2 = q1.clone()
    q2[:, 1] = 999.0
    assert torch.allclose(mixer(q1, pres, g), mixer(q2, pres, g), atol=1e-5)


def test_stitch_joint_windows_and_terminal():
    """결정 k→k+1 창: c=costs[k], γ_dt=γ^{Δt/ref}; 말단은 terminal."""
    enc = _enc()
    times = [0.0, 90.0, 240.0]
    costs = [1.5, 2.5, 4.0]
    events = [(0, (("YC-A", enc, 1), ("YC-B", enc, 2))),
              (1, (("YC-A", enc, 0),)),
              (2, (("YC-B", enc, None),))]   # pos None → 예측 제외 → 표본 생략
    s = stitch_joint_samples(times, costs, events, gamma=0.5, ref_s=60.0)
    assert len(s) == 2
    assert s[0].action_pos == (1, 2) and s[0].c_disc == 1.5
    assert s[0].gamma_dt == pytest.approx(0.5 ** (90.0 / 60.0))
    assert s[0].next_encs is not None and len(s[0].next_encs) == 1
    # 두 번째 표본의 bootstrap 상태는 (예측 불가라도) 다음 결정 encs 를 그대로 사용
    assert s[1].action_pos == (0,) and s[1].next_encs is not None


def test_terminal_target_equals_cost():
    """terminal 표본: y=c — 예측이 c 와 같으면 loss 0."""
    enc = _enc()
    dims = encoding_dims(enc)
    lr = QmixLearner(QmixConfig(min_replay=1, batch_size=1), dims, seed=2)
    pos = enc.selectable.index(True)
    lr.replay.append(JointSample((enc,), (pos,), 3.3, 0.0, None))
    # 반복 학습 시 loss 가 감소 (표적 c=3.3 으로 회귀)
    losses = [lr.learn_step() for _ in range(120)]
    assert losses[-1] is not None and losses[-1] < losses[0]


def test_qmix_e2e_quick_training_and_eval():
    """run_episode duck-type: 수집(joint_sink)→absorb→학습→greedy 평가 완주."""
    enc = _enc()
    dims = encoding_dims(enc)
    lr = QmixLearner(QmixConfig(min_replay=8, batch_size=8, cost_scale=2.0),
                     dims, seed=3)
    import random as _r
    explore = _r.Random(7)
    for ep, seed in enumerate((530_900, 530_901, 530_902)):
        sink: dict = {}
        r = run_episode(_sim(seed), level=LEVEL, preference=QPreference(),
                        learner=lr, epsilon=1.0 / (ep + 1), explore_rng=explore,
                        learn=True, joint_sink=sink)
        n = lr.absorb_joint(sink)
        assert r.completion_rate == 1.0 and n > 0
    assert len(lr.replay) >= 8
    loss = lr.learn_step()
    assert loss is not None and loss == loss  # 유한 (NaN 아님)
    r_eval = run_episode(_sim(530_903), level=LEVEL, preference=QPreference(),
                         learner=lr)
    assert r_eval.completion_rate == 1.0 and r_eval.invariants_ok


def test_qmix_save_load_roundtrip(tmp_path):
    enc = _enc()
    dims = encoding_dims(enc)
    lr = QmixLearner(QmixConfig(), dims, seed=4)
    p = tmp_path / "qmix.pt"
    lr.save(p)
    lr2 = QmixLearner.load(p)
    s = lr.scores_for(enc)
    s2 = lr2.scores_for(enc)
    assert s.keys() == s2.keys()
    for k in s:
        assert s[k] == pytest.approx(s2[k], abs=1e-5)
