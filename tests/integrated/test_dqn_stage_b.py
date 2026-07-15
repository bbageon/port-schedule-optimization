"""YR-039 Stage B 계약 테스트 — SMDP 스티칭·DDQN 표적·드라이버·quick e2e."""
import pytest

torch = pytest.importorskip("torch")

from yard_rl.domain.enums import InformationLevel
from yard_rl.experiments.candidate_dqn_experiment import (
    CandidateDqnConfig, quick_candidate_dqn_config, run_candidate_dqn)
from yard_rl.integrated import TerminalSimulator
from yard_rl.integrated.adapter import capture
from yard_rl.integrated.dqn_learner import (CandidateDQNLearner, LearnerConfig,
                                            Sample, run_episode, stitch_samples)
from yard_rl.integrated.encoding import encode_observation, encoding_dims
from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.qnet import QPreference
from yard_rl.integrated.resolver import BaselinePreference
from yard_rl.integrated.scenario_gen import (TerminalGenParams,
                                             generate_terminal_scenario)

PROF = build_integrated_profile()
LEVEL = InformationLevel.PRE_ADVICE
PARAMS = TerminalGenParams(n_external=8, n_vessels=1, vessel_moves=6,
                           horizon_s=7_200.0, drain_window_s=3_600.0)


def _enc(seed=300_101):
    sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, seed, PARAMS),
                            check_invariants=True)
    sim.info_level = LEVEL
    dp = sim.run_until_decision()
    state, obs, _ = capture(sim, dp.crane_ids, LEVEL, "t", 0)
    return encode_observation(state, obs[0])


def test_stitch_discounted_costs_and_windows():
    """매핑 §3 수식: C = Σ γ^{(t_j−t_k)/ref}·c_j · WAIT 는 창에 흡수·말단 terminal."""
    enc = _enc()
    times = [0.0, 60.0, 120.0, 240.0]
    costs = [1.0, 2.0, 4.0, 8.0]
    events = {"YC-A": [(0, enc, 1), (2, enc, None), (3, enc, 0)]}  # k2 는 WAIT
    s = stitch_samples(times, costs, events, gamma=0.5, ref_s=60.0)
    assert len(s) == 2                       # WAIT 결정은 표본 아님
    # 표본 1: k0→k3 창 (WAIT k2 흡수): 1 + 0.5*2 + 0.25*4 = 3.0, γ_dt = 0.5^(240/60)
    assert s[0].c_disc == pytest.approx(3.0)
    assert s[0].gamma_dt == pytest.approx(0.5 ** 4)
    assert s[0].next_enc is not None
    # 표본 2: k3 말단 terminal
    assert s[1].c_disc == pytest.approx(8.0)
    assert s[1].gamma_dt == 0.0 and s[1].next_enc is None


def test_ddqn_target_uses_online_argmin_target_value():
    """DDQN: y = C + γ·Q_target(s', argmin Q_online) — 수기 대조."""
    enc = _enc()
    dims = encoding_dims(enc)
    learner = CandidateDQNLearner(
        LearnerConfig(variant="ddqn", min_replay=1, batch_size=1,
                      target_sync_every=10_000), dims, seed=3)
    # online 만 학습시켜 online/target 분화
    for _ in range(30):
        learner.replay.append(Sample(enc, enc.selectable.index(True), 1.7, 0.0, None))
        learner.learn_step()
    sample = Sample(enc, enc.selectable.index(True), 0.5, 0.8, enc)
    learner.replay.clear()
    learner.replay.append(sample)
    g, yc, qs, cand, sel = learner._tensors([enc])
    with torch.no_grad():
        q_on = learner.online(g, yc, qs, cand, sel)
        q_tg = learner.target(g, yc, qs, cand, sel)
        a_star = int(q_on.masked_fill(~sel, float("inf")).min(dim=1).indices[0])
        y_expect = 0.5 + 0.8 * float(q_tg[0, a_star])
        pred_before = float(learner.online(g, yc, qs, cand, sel)[0, sample.action_pos])
    loss = learner.learn_step()
    # smooth_l1(pred, y) 의 관측 loss 가 수기 y 와 일치하는 방향인지 확인
    diff = abs(pred_before - y_expect)
    expect_loss = 0.5 * diff ** 2 if diff < 1.0 else diff - 0.5
    assert loss == pytest.approx(expect_loss, rel=1e-4)


def test_driver_untrained_equals_baseline_total_cost():
    """드라이버 경유 미학습 QPreference ≡ BaselinePreference (총비용·결정수 일치)."""
    def run(pref, learner=None):
        sim = TerminalSimulator(PROF, generate_terminal_scenario(PROF, 300_102, PARAMS),
                                check_invariants=True)
        return run_episode(sim, level=LEVEL, preference=pref, learner=learner)

    enc = _enc()
    learner = CandidateDQNLearner(LearnerConfig(), encoding_dims(enc), seed=0)
    base = run(BaselinePreference())
    q = run(QPreference(), learner=learner)
    assert q.total_cost == pytest.approx(base.total_cost)
    assert q.n_decisions == base.n_decisions
    assert q.completion_rate == base.completion_rate == 1.0


def test_seed_band_guard():
    with pytest.raises(ValueError):
        CandidateDqnConfig(train_seed0=160_000)   # 단일야드 대역 재사용 금지


def test_quick_run_end_to_end(tmp_path):
    report = run_candidate_dqn(out_dir=str(tmp_path / "out"),
                               cfg=quick_candidate_dqn_config(),
                               progress=lambda _m: None)
    text = report.read_text(encoding="utf-8")
    assert "CandidateDQN[ddqn]" in text and "guardrail" in text
    assert (tmp_path / "out" / "candidate_dqn_results.json").exists()
    assert (tmp_path / "out" / "model_CandidateDQN[ddqn].pt").exists()
