"""YR-030-b — v1_final 스키마·greedy-prior·γ 계약 테스트."""
import json

import pytest

from yard_rl.envs.direct_job_env import (DirectJobBucketConfig, DirectJobEnv,
                                         JobState, SLAMode, YardState)
from yard_rl.experiments.state_v1_final import (V1FinalConfig,
                                                quick_v1final_config,
                                                run_v1_final_experiment)
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams, generate
from yard_rl.policies.cost_q import CostQAgent, CostQConfig

PROFILE = "configs/terminals/hjnc_armg.yaml"


def _scenario(profile, seed=70_001, n=10):
    return generate(profile, seed, GenParams(n_external=n, n_vessel=0,
                                             drain_window_s=86_400.0))


def test_v1_final_schema_shapes_and_consistency_rules():
    profile = load_profile(PROFILE)
    env = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_final",
                       expected_n_config=10)
    state, info = env.reset(_scenario(profile))
    assert isinstance(state, YardState)
    c = info.candidates[0]
    assert isinstance(c.feature, JobState)
    assert c.feature.job_type in ("TRUCK_TO_YARD", "YARD_TO_TRUCK")
    assert c.prior_cost >= 0.0
    # 일관성 규칙: 후보 대기 ≤ 최장 대기, 30분 초과 수 ≤ 대기 수 (관측마다 강제)
    raw = info.raw_global
    assert all(x.wait_s <= raw.longest_wait_s + 1e-6 for x in info.candidates)
    assert raw.over_30min_truck_count <= raw.waiting_truck_count
    # 에피소드 전체가 규칙 위반 없이 완주
    while state is not None:
        state, _c, _d, info = env.step(info.candidates[0])


def test_v1_final_bucket_granularity():
    # truck_wait 4단계 (edge 3: 3분위 2 + SLA), crane_travel 3단계 (edge 2)
    cfg = DirectJobBucketConfig.fit(
        queue_lengths=[1, 2, 5], service_times_s=[100, 200, 900],
        oldest_waits_s=[0, 100, 2000], own_waits_s=[0, 50, 100, 400, 900, 2500],
        reaches_s=[10, 50, 100, 200], sla_s=1800.0)
    assert len(cfg.truck_wait_s) == 3 and cfg.truck_wait_s[-1] == 1800.0
    assert len(cfg.crane_travel_s) == 2


def test_greedy_prior_no_fallback_and_gamma_backup():
    agent = CostQAgent(CostQConfig(learning_rate_power=1.0, gamma=0.95,
                                   use_greedy_prior=True), seed=0)

    class C:
        def __init__(self, jid, service, prior):
            self.job_id = jid
            self.feature = ("YARD_TO_TRUCK", 1, 1, 1, 0)
            self.wait_s = 0.0
            self.estimated_service_s = service
            self.block_entry_s = 0.0
            self.prior_cost = prior

    s = YardState(0, 0, 0, 0, 0)
    a, b = C("a", 100.0, 2.0), C("b", 200.0, 5.0)
    # 미방문인데도 fallback 없이 prior argmin 으로 선택
    pick = agent.act(s, [a, b])
    assert pick.job_id == "a" and agent.diagnostics.fallback_count == 0
    # γ=0.95 backup: 첫 update(α₁=1) → Q = cost + 0.95 * min(next prior)
    agent.update(s, a, cost=1.0, next_global_state=s, next_candidates=[b],
                 done=False)
    assert agent.table.value(agent.key(s, a)) == pytest.approx(1.0 + 0.95 * 5.0)
    # 방문된 키는 학습값이 prior 를 대체
    assert agent._value_or_prior(s, a) == pytest.approx(5.75)


def test_gamma_validation_and_default_frozen():
    with pytest.raises(ValueError):
        CostQConfig(gamma=1.5)
    with pytest.raises(ValueError):
        CostQConfig(gamma=0.0)
    assert CostQConfig().gamma == 1.0 and CostQConfig().use_greedy_prior is False


def test_v1final_config_rejects_prior_bands():
    with pytest.raises(ValueError):
        V1FinalConfig(train_seed0=40_000)


def test_v1final_quick_smoke(tmp_path):
    out = tmp_path / "v1final"
    report = run_v1_final_experiment(PROFILE, str(out), quick_v1final_config(),
                                     progress=lambda _msg: None)
    text = report.read_text(encoding="utf-8")
    assert "γ grid" in text and "locked test" in text
    payload = json.loads((out / "v1final_results.json").read_text(encoding="utf-8"))
    assert any(name.startswith("CostQ[v1_final|prior|g") for name in payload["paired"])
    assert isinstance(payload["gamma_improved_vs_baseline"], list)
