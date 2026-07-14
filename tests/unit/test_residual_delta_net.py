"""YR-012 — 잔차 연속-feature Δ 학습 계약 테스트 (사전등록 §1~§3 대응)."""
import pytest

torch = pytest.importorskip("torch")  # optional [rl] — 미설치 환경은 skip

from yard_rl.envs.direct_job_env import DirectJobEnv, SLAMode
from yard_rl.experiments.residual_delta_experiment import (DeltaExpConfig,
                                                           quick_delta_config,
                                                           run_delta_experiment)
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams, generate
from yard_rl.policies.direct_baselines import DirectJobRulePolicy, DirectRule
from yard_rl.policies.residual_delta_net import (DeltaNetConfig, FeatureScaler,
                                                 N_FEATURES,
                                                 ResidualDeltaNetAgent,
                                                 extract_features)

PROFILE = "configs/terminals/hjnc_armg.yaml"


def _scenario(profile, seed=140_001, n=10):
    return generate(profile, seed, GenParams(n_external=n, n_vessel=0,
                                             drain_window_s=86_400.0))


def _identity_scaler():
    return FeatureScaler((0.0,) * N_FEATURES, (1.0,) * N_FEATURES, fitted=True)


def test_untrained_net_equals_greedy_exactly():
    """§1: 출력층 zero-init → 미학습 정책 ≡ IMMEDIATE_COST_GREEDY (에피소드)."""
    profile = load_profile(PROFILE)
    agent = ResidualDeltaNetAgent(DeltaNetConfig(), scaler=_identity_scaler(),
                                  seed=0)
    greedy = DirectJobRulePolicy(DirectRule.IMMEDIATE_COST_GREEDY)
    env_a = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                         expected_n_config=10)
    env_b = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                         expected_n_config=10)
    sa, ia = env_a.reset(_scenario(profile))
    sb, ib = env_b.reset(_scenario(profile))
    while sa is not None:
        pick_a = agent.act(sa, ia.candidates)
        pick_b = greedy.act(sb, ib.candidates)
        assert pick_a.job_id == pick_b.job_id
        sa, _c, _d, ia = env_a.step(pick_a)
        sb, _c2, _d2, ib = env_b.step(pick_b)
    assert sb is None


def test_features_are_continuous_and_attached():
    """§2: 14차원 연속 feature — bucket 아닌 원값 (서비스 s 그대로)."""
    profile = load_profile(PROFILE)
    env = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                       expected_n_config=10)
    _state, info = env.reset(_scenario(profile))
    for c in info.candidates:
        x = extract_features(c)
        assert len(x) == N_FEATURES
        assert x[8] == pytest.approx(c.estimated_service_s)  # bucket 화 안 됨
        assert 0.0 <= x[0] <= 1.0 and 0.0 <= x[1] <= 1.0 and 0.0 <= x[12] <= 1.0


def test_scaler_fit_freeze_roundtrip(tmp_path):
    rows = [[float(i + d) for d in range(N_FEATURES)] for i in range(20)]
    sc = FeatureScaler.fit(rows)
    assert sc.fitted
    assert sc.mean[0] == 0.0 and sc.std[0] == 1.0  # [0,1] 필드 passthrough
    assert sc.std[8] > 0.0                          # z-score 필드
    p = tmp_path / "scaler.json"
    sc.save(p)
    assert FeatureScaler.load(p) == sc


def test_update_regresses_toward_residual_target():
    """§1 학습식: 반복 update 로 Δθ(x) → (Y − G). 음수 목표 허용."""
    from tests.unit.test_residual_costq import _C
    agent = ResidualDeltaNetAgent(DeltaNetConfig(gamma=1.0, lr=1e-2),
                                  scaler=_identity_scaler(), seed=0)
    c = _C("A", prior=3.0)
    c.global_raw = (0.5, 0.5, 3.0, 100.0, 0.0)
    c.future_raw = (2.0, 400.0, 0.5, 3.0)
    c.transfer_direction = "YARD_TO_TRUCK"
    c.reach_s, c.blocker_count = 50.0, 1
    # 종료 스텝: Y = c = 0.5 → 목표 Y_Δ = −2.5
    for _ in range(400):
        agent.update("s", c, 0.5, None, [], done=True)
    delta = agent.q_totals([c])[0] - 3.0
    assert delta == pytest.approx(-2.5, abs=0.15)


def test_save_load_roundtrip(tmp_path):
    from tests.unit.test_residual_costq import _C
    agent = ResidualDeltaNetAgent(DeltaNetConfig(), scaler=_identity_scaler(),
                                  seed=7)
    c = _C("A", prior=1.0)
    c.global_raw = (0.1, 0.2, 2.0, 50.0, 0.0)
    c.future_raw = (1.0, 200.0, 0.3, 2.0)
    c.transfer_direction = "TRUCK_TO_YARD"
    c.reach_s, c.blocker_count = 30.0, 0
    agent.update("s", c, 0.2, None, [], done=True)
    p = tmp_path / "model.pt"
    agent.save(p)
    loaded = ResidualDeltaNetAgent.load(p)
    assert loaded.q_totals([c])[0] == pytest.approx(agent.q_totals([c])[0])


def test_seed_band_guard():
    with pytest.raises(ValueError):
        DeltaExpConfig(train_seed0=110_000)  # YR-030-c band 재사용 금지


def test_quick_run_end_to_end(tmp_path):
    report = run_delta_experiment(out_dir=str(tmp_path / "out"),
                                  cfg=quick_delta_config(),
                                  progress=lambda _m: None)
    text = report.read_text(encoding="utf-8")
    assert "ResidualDeltaNet" in text and "guardrail" in text
    assert (tmp_path / "out" / "feature_scaler.json").exists()
    assert (tmp_path / "out" / "model_ResidualDeltaNet.pt").exists()
