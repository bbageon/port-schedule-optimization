"""YR-012-c — 집합 맥락 feature·22입력 계약 테스트."""
import json

import pytest

torch = pytest.importorskip("torch")

from yard_rl.envs.direct_job_env import DirectJobEnv, SLAMode  # noqa: E402
from yard_rl.experiments.coverage_ablation import _gen_params  # noqa: E402
from yard_rl.experiments.direct_job_runner import _scenario  # noqa: E402
from yard_rl.experiments.oracle_pattern import _set_aggregates  # noqa: E402
from yard_rl.experiments.residual_setfeat_experiment import (SetFeatConfig,  # noqa: E402
                                                             quick_setfeat_config,
                                                             run_setfeat_experiment)
from yard_rl.io.profile_loader import load_profile  # noqa: E402
from yard_rl.policies.residual_delta_net import (DeltaNetConfig,  # noqa: E402
                                                 FeatureScaler, N_FEATURES,
                                                 N_FEATURES_SET,
                                                 ResidualDeltaNetAgent,
                                                 extract_features,
                                                 extract_features_with_set)

PROFILE = "configs/terminals/hjnc_armg.yaml"


class _Shim:
    n_external = 10
    drain_window_s = 86_400.0


def _first_decision():
    profile = load_profile(PROFILE)
    scenario = _scenario(profile, 200_000, _gen_params(_Shim()), 10)
    env = DirectJobEnv(profile, sla_mode=SLAMode.OFF, expected_n_config=10,
                       state_schema="v1_final")
    _s, info = env.reset(scenario)
    return info.feasible_candidates


def test_env_attaches_8_set_features_shared_and_matches_oracle_def():
    cands = _first_decision()
    assert all(len(c.set_raw) == 8 for c in cands)
    # 결정 시점 1회 산출 → 전 후보 동일 값
    assert len({c.set_raw for c in cands}) == 1
    # oracle_pattern._set_aggregates 와 비트 단위 동일 (fmean 정밀합 — H-A 신호 재현)
    expected = tuple(_set_aggregates(list(cands)))
    assert cands[0].set_raw == expected   # 정확 일치 (approx 아님 — 계약 강제)


def test_extract_22_extends_14_and_dims_consistent():
    c = _first_decision()[0]
    base = extract_features(c)
    ext = extract_features_with_set(c)
    assert len(base) == N_FEATURES == 14
    assert len(ext) == N_FEATURES_SET == 22
    assert ext[:14] == base and ext[14:] == [float(v) for v in c.set_raw]


def test_scaler_selects_zscore_dims_by_length():
    rows14 = [[float(i + r) for i in range(14)] for r in range(4)]
    rows22 = [[float(i + r) for i in range(22)] for r in range(4)]
    s14, s22 = FeatureScaler.fit(rows14), FeatureScaler.fit(rows22)
    assert len(s14.mean) == 14 and len(s22.mean) == 22
    # 비율 2개(20,21)는 passthrough (std=1,mean=0), 집합 z-score(14~19)는 std!=1
    assert s22.std[20] == 1.0 and s22.mean[20] == 0.0
    assert s22.std[16] != 1.0


def test_setfeat_agent_untrained_is_greedy_equivalent():
    scaler = FeatureScaler.fit([[float(i + r) for i in range(22)] for r in range(4)])
    agent = ResidualDeltaNetAgent(DeltaNetConfig(use_set_context=True),
                                  scaler=scaler, seed=0)
    assert len(agent.scaler.mean) == 22

    class C:
        def __init__(self, jid, prior):
            self.job_id = jid
            self.transfer_direction = "YARD_TO_TRUCK"
            self.wait_s = 1.0
            self.reach_s = 1.0
            self.estimated_service_s = 1.0
            self.blocker_count = 0
            self.block_entry_s = 0.0
            self.prior_cost = prior
            self.global_raw = (0.1, 0.5, 3.0, 100.0, 0.0)
            self.future_raw = (2.0, 400.0, 0.5, 3.0)
            self.set_raw = (3.0, 50.0, 100.0, 200.0, 5.0, 10.0, 0.6, 0.5)

    a, b = C("a", 2.0), C("b", 5.0)
    assert agent.act(None, [a, b]).job_id == "a"          # zero-init → greedy
    assert agent.q_totals([a, b]) == pytest.approx([2.0, 5.0])


def test_dim_mismatch_rejected():
    scaler14 = FeatureScaler.fit([[float(i + r) for i in range(14)] for r in range(4)])
    with pytest.raises(ValueError, match="scaler 차원"):
        ResidualDeltaNetAgent(DeltaNetConfig(use_set_context=True),
                              scaler=scaler14, seed=0)   # 14 scaler + 22 모드


def test_setfeat_config_rejects_prior_bands():
    with pytest.raises(ValueError):
        SetFeatConfig(train_seed0=160_000)   # YR-031 band 재사용 금지


def test_setfeat_quick_smoke(tmp_path):
    out = tmp_path / "setfeat"
    report = run_setfeat_experiment(PROFILE, str(out), quick_setfeat_config(),
                                    progress=lambda _msg: None)
    text = report.read_text(encoding="utf-8")
    assert "14" in text and "22" in text and "locked test" in text
    payload = json.loads((out / "setfeat_results.json").read_text(encoding="utf-8"))
    assert "SetFeatDeltaNet[22]" in payload["paired"]
    assert len(json.loads((out / "feature_scaler.json").read_text())["mean"]) == 22
