"""YR-027 dedicated runner smoke and frozen-contract tests."""
import json

import pytest

from yard_rl.experiments.direct_job_runner import (
    POLICY_COST_Q,
    DirectExperimentConfig,
    run_direct_job_experiment,
)
import yard_rl.experiments.direct_job_runner as direct_runner


def _smoke_config() -> DirectExperimentConfig:
    return DirectExperimentConfig(
        train_episodes=2,
        validation_episodes=2,
        test_episodes=2,
        checkpoint_every=1,
        n_external=8,
        learning_rate_powers=(0.6,),
        bootstrap_resamples=50,
        quick=True,
    )


def test_frozen_default_has_disjoint_seed_bands_and_is_claim_eligible():
    cfg = DirectExperimentConfig()
    assert len(cfg.train_seeds) == 1_000
    assert len(cfg.validation_seeds) == 30
    assert len(cfg.test_seeds) == 100
    assert not (set(cfg.train_seeds) & set(cfg.validation_seeds))
    assert not (set(cfg.train_seeds) & set(cfg.test_seeds))
    assert cfg.strategy_compliant


def test_runner_rejects_overlapping_seed_splits():
    with pytest.raises(ValueError, match="disjoint"):
        DirectExperimentConfig(
            train_episodes=2,
            validation_episodes=2,
            test_episodes=2,
            train_seed0=10,
            validation_seed0=11,
            test_seed0=30,
        )


def test_full_run_rejects_dirty_source_before_generating(monkeypatch, tmp_path):
    cfg = DirectExperimentConfig(
        train_episodes=2, validation_episodes=2, test_episodes=2,
        checkpoint_every=1, n_external=8, learning_rate_powers=(0.6,),
        bootstrap_resamples=10,
    )
    monkeypatch.setattr(
        direct_runner, "_git_state",
        lambda: {"commit": "abc123", "dirty": True},
    )
    with pytest.raises(RuntimeError, match="clean committed source"):
        run_direct_job_experiment(
            "configs/terminals/poc_single_crane.yaml", str(tmp_path), cfg,
            progress=lambda _message: None,
        )
    assert not any(tmp_path.iterdir())


def test_dedicated_quick_pipeline_writes_no_claim_paired_artifacts(tmp_path):
    messages = []
    report = run_direct_job_experiment(
        "configs/terminals/poc_single_crane.yaml",
        str(tmp_path),
        _smoke_config(),
        progress=messages.append,
    )
    payload = json.loads((tmp_path / "exp1_direct_results.json").read_text())

    assert report.exists() and report.parent == tmp_path
    assert payload["manifest"]["n_vessel"] == 0
    assert payload["manifest"]["information_boundary"] == "BLOCK_ENTRY"
    assert payload["manifest"]["strategy_id"] == "YR-027-v2-minimal-state"
    assert payload["manifest"]["global_state_schema"] == [
        "operation_phase", "queue_length_bucket"
    ]
    assert payload["manifest"]["candidate_feature_schema"] == [
        "transfer_direction", "estimated_service_time_bucket", "end_crane_zone"
    ]
    buckets = json.loads((tmp_path / "direct_buckets.json").read_text())
    # YR-028: bucket 스키마 확장 (v1_rich 필드) — v2 인코딩 불변
    assert set(buckets) == {"fitted", "queue_len", "service_s",
                            "oldest_wait_s", "own_wait_s", "reach_s"}
    assert set(payload["summary"]) == {"SLA_OFF", "SLA_ON"}
    assert POLICY_COST_Q in payload["summary"]["SLA_OFF"]
    assert payload["acceptance"]["overall"] is None
    assert payload["acceptance"]["decision"] == "NO_CLAIM_NONCOMPLIANT"
    assert payload["paired_statistics"]["SLA_OFF"]["mean_wait"]["n"] == 2
    assert payload["paired_statistics"]["SLA_OFF"]["alternative"] == POLICY_COST_Q
    assert (tmp_path / "agent_SLA_OFF.json").exists()
    assert (tmp_path / "agent_SLA_ON.json").exists()
    text = report.read_text(encoding="utf-8")
    assert "selected baseline `" in text
    assert "Quick run" in text
    assert "Mean wait 차이: —" not in text
    assert len(text.splitlines()) <= 200
    assert any("completed" in message for message in messages)
