"""YR-028 — v1_rich 상태 복원·bucket 확장·ablation 스모크 계약 테스트."""
import json

import pytest

from yard_rl.envs.direct_job_env import (DirectJobBucketConfig, DirectJobEnv,
                                         SLAMode, YardState)
from yard_rl.experiments.coverage_ablation import (AblationConfig,
                                                   quick_ablation_config,
                                                   run_coverage_ablation)
from yard_rl.io.profile_loader import load_profile
from yard_rl.io.scenario_gen import GenParams, generate
from yard_rl.policies.cost_q import CostQAgent

PROFILE = "configs/terminals/hjnc_armg.yaml"


def _scenario(profile, seed=40_001, n=10):
    return generate(profile, seed, GenParams(n_external=n, n_vessel=0,
                                             drain_window_s=86_400.0))


def test_v1_rich_schema_shapes_and_v2_default_unchanged(tmp_path):
    profile = load_profile(PROFILE)
    scenario = _scenario(profile)
    env1 = DirectJobEnv(profile, sla_mode=SLAMode.OFF, state_schema="v1_rich",
                        expected_n_config=10)
    state1, info1 = env1.reset(scenario)
    assert isinstance(state1, YardState)
    assert state1._fields == (
        "work_phase", "crane_area", "waiting_truck_level",
        "longest_wait_level", "over_30min_truck_count",
    )
    assert len(state1) == 5 and all(isinstance(x, int) for x in state1)
    # NamedTuple 도 기존 tuple key와 동일 비교·hash — 저장된 Q-table 호환 계약.
    assert state1 == tuple(state1) and hash(state1) == hash(tuple(state1))
    feat = info1.candidates[0].feature
    assert len(feat) == 5 and feat[0] in ("TRUCK_TO_YARD", "YARD_TO_TRUCK")
    assert info1.raw_global.crane_position_bay > 0
    agent = CostQAgent()
    key = agent.key(state1, info1.candidates[0])
    agent.table.q[key], agent.table.n[key] = 1.25, 1
    saved = tmp_path / "named-state-agent.json"
    agent.save(saved)
    loaded = CostQAgent.load(saved)
    assert loaded.table.is_visited(loaded.key(state1, info1.candidates[0]))

    env2 = DirectJobEnv(profile, sla_mode=SLAMode.OFF, expected_n_config=10)
    state2, info2 = env2.reset(scenario)
    assert len(state2) == 2 and state2[0] in ("OPERATING", "CLEAR_OUT")
    assert len(info2.candidates[0].feature) == 3

    with pytest.raises(ValueError):
        DirectJobEnv(profile, state_schema="v3_unknown")


def test_bucket_config_backward_compatible_load(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"queue_len": [1, 3, 6], "service_s": [120, 300, 600],
                               "fitted": True}), encoding="utf-8")
    cfg = DirectJobBucketConfig.load(old)  # v2 시절 파일 — 신규 필드 없음
    assert cfg.fitted and cfg.own_wait_s == (300.0, 900.0, 1800.0)
    fitted = DirectJobBucketConfig.fit(
        queue_lengths=[1, 2, 5], service_times_s=[100, 200, 900],
        oldest_waits_s=[0, 100, 2000], own_waits_s=[0, 50, 2500],
        reaches_s=[10, 50, 200], sla_s=1800.0)
    assert 1800.0 in fitted.own_wait_s and 1800.0 in fitted.oldest_wait_s
    fitted.save(tmp_path / "new.json")
    assert DirectJobBucketConfig.load(tmp_path / "new.json") == fitted


def test_ablation_config_rejects_yr027_bands():
    with pytest.raises(ValueError):
        AblationConfig(train_seed0=10_000)


def test_ablation_quick_smoke(tmp_path):
    out = tmp_path / "ablation"
    report = run_coverage_ablation(PROFILE, str(out), quick_ablation_config(),
                                   progress=lambda _msg: None)
    text = report.read_text(encoding="utf-8")
    assert "primary_cause" in text and "checkpoint 곡선" in text
    payload = json.loads((out / "ablation_results.json").read_text(encoding="utf-8"))
    assert payload["verdict"]["primary_cause"] in (
        "STATE_SPACE", "CHECKPOINT_RULE", "BUDGET", "NONE_REPRODUCED")
    names = set(payload["selections"])
    assert any(name.startswith("CostQ[v1_rich|R1_min_wait@") for name in names)
    assert any(name.startswith("CostQ[v2_minimal|") for name in names)
    curve = json.loads((out / "checkpoint_curve.json").read_text(encoding="utf-8"))
    assert {row["schema"] for row in curve} == {"v1_rich", "v2_minimal"}
