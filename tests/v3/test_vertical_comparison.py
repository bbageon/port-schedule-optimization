"""Frozen-input, evidence-preservation checks for a single vertical case."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v3"))

from vertical_comparison_io import (CHECKPOINT_SHA, claim_output, operational_outcomes,
                                    pair_summary, recording_checks, validate_config)
from yard_rl.v3.layouts import synthetic_vertical_spec
from yard_rl.v3.world.integrated.profiles import build_h21_profile


@pytest.fixture
def configs():
    relative = "outputs/reports/yr317_v3_independent_eval/config.json"
    base = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    spec = synthetic_vertical_spec(build_h21_profile(), n_blocks=21)
    cfg = dict(schema="yr317.vertical-comparison.v1", seed=20_000_000,
        arms=["NO_REALLOC", "RL_TIME"], base_config=relative,
        base_config_sha256="b" * 64, source_commit="a" * 40,
        prereg="outputs/reports/yr317_v3_vertical_transfer/prereg.md",
        prereg_sha256="c" * 64, checkpoint=base["checkpoint"],
        checkpoint_sha256=base["checkpoint_sha256"],
        environment_spec=deepcopy(spec), new_training_runs=0,
        frozen_model=True, claim_eligible=False)
    return cfg, base, spec


def test_fixed_first_seed_and_declared_environment_are_accepted(configs):
    cfg, base, spec = configs
    before = deepcopy(configs)
    validate_config(cfg, base, spec)
    assert configs == before


@pytest.mark.parametrize("problem", ["another_seed", "reordered_first_seed",
    "full_policy", "duplicate_policy", "changed_geometry", "changed_pipeline"])
def test_config_rejects_case_selection_or_environment_drift(configs, problem):
    cfg, base, spec = configs
    if problem == "another_seed":
        cfg["seed"] = base["seeds"][1]
    elif problem == "reordered_first_seed":
        base["seeds"][0], base["seeds"][1] = base["seeds"][1], base["seeds"][0]
    elif problem == "full_policy":
        cfg["arms"][1] = "RL"
    elif problem == "duplicate_policy":
        cfg["arms"][1] = "NO_REALLOC"
    elif problem == "changed_geometry":
        cfg["environment_spec"]["geometry"]["speed_mps"] *= 2
    else:
        cfg["environment_spec"]["discharge_pipeline_capacity"] += 1
    with pytest.raises(ValueError):
        validate_config(cfg, base, spec)


def test_claim_output_does_not_overwrite_prior_evidence(tmp_path):
    out = tmp_path / "run"
    first = claim_output(out, "NO_REALLOC")
    evidence = first / "result.json"
    evidence.write_bytes(b'{"unfinished": 7}\n')
    with pytest.raises((FileExistsError, ValueError)):
        claim_output(out, "NO_REALLOC")
    assert evidence.read_bytes() == b'{"unfinished": 7}\n'
    second = claim_output(out, "RL_TIME")
    assert second != first and second.is_dir()
    assert evidence.read_bytes() == b'{"unfinished": 7}\n'


def _recording_checks(result, **overrides):
    kwargs = dict(expected_input=result["expected_input"],
        environment_spec_sha256=result["environment_spec_sha256"],
        weights_unchanged=True, checkpoint_unchanged=True,
        runtime_unchanged=True, rollout_count=0)
    kwargs.update(overrides)
    return recording_checks(result, **kwargs)


@pytest.fixture
def results():
    """A loss with unfinished work, using tiny synthetic audit records only."""
    inputs = dict(schedule_sha256="a" * 64,
        initial_scenarios_sha256="b" * 64, vessels_sha256="c" * 64)
    plan = [dict(index=i, load=2, label="fixture", seed=20_001_000 + i,
                 t0=i * 86400.0, n_days=30) for i in range(30)]
    baseline = dict(arm="NO_REALLOC", seed=20_000_000, plan=plan,
        days=[dict(index=i, train=0 < i < 29,
            phi_krw=1000.0 if i == 0 else 2000.0 if i == 29 else 100.0,
            c_wait=10.0, operational=dict(day_index=i, cohort={})) for i in range(30)],
        expected_input=inputs, requested_identity_sha256="d" * 64,
        environment_spec_sha256="e" * 64, pair_contract_sha256="f" * 64,
        source_commit="a" * 40, config_sha256="1" * 64,
        base_config_sha256="2" * 64, prereg_sha256="3" * 64,
        checkpoint_sha256=CHECKPOINT_SHA,
        network_identities=[dict(state_sha256="seller"), dict(state_sha256="buyer")],
        runtime=dict(engine_sha256="fixed", python="3.12"),
        environment_manifest=dict(kind="VERTICAL_END", canonical_input=deepcopy(inputs),
            runtime_schedule_sha256="4" * 64, candidate_pruning="feasible first"),
        request_summary=dict(requested=60, states=dict(COMPLETED=57, CENSORED=3),
            unbound_jobs_at_end=2, recording_ok=True, announcer_counts_match=True,
            all_requests_admitted=True, accounted_wait_krw=300.0,
            physical_invariants_enabled=True),
        vessel_work_summary=dict(recording_ok=True, unadmitted_moves=0,
            requested_moves=8, completed_yard_jobs=3, outstanding_yard_jobs=5,
            all_requested_work_completed=False),
        container_flow_summary=dict(passed=True), policy_exceptions=0,
        space=0, time=0, rollout_calls=0, new_training_runs=0, claim_eligible=False,
        vessel_admissions=[dict(structural_min_overrun_s=3600.0)])
    baseline["recording_checks"] = _recording_checks(baseline)
    proposed = deepcopy(baseline)
    proposed["arm"] = "RL_TIME"
    proposed["time"] = 7
    proposed["request_summary"]["states"] = dict(COMPLETED=53, CENSORED=7)
    for day in proposed["days"]:
        if day["train"]:
            day["phi_krw"] = 110.0
    return [baseline, proposed]


def test_unfinished_work_and_impossible_deadlines_are_preserved_not_recording_errors(results):
    before = deepcopy(results)
    for result in results:
        assert all(_recording_checks(result).values())
        outcomes = operational_outcomes(result)
        assert outcomes["vessel_unfinished"] == 5
        assert outcomes["unbound_at_cutoff"] == 2
        assert outcomes["structurally_late_vessel_streams"] == 1
        assert outcomes["maximum_structural_min_overrun_s"] == 3600.0
        assert not outcomes["all_trucks_completed"] and not outcomes["all_vessels_completed"]
    assert results == before


@pytest.mark.parametrize("problem", ["request_recording", "announcer", "lost_truck",
    "lost_vessel", "inventory_chain", "wait_cost", "physical_checks", "policy_exception",
    "spatial_action", "canonical_input", "environment_identity", "missing_day"])
def test_recording_errors_fail_even_when_result_has_valid_unfinished_work(results, problem):
    result = results[0]
    if problem == "request_recording":
        result["request_summary"]["recording_ok"] = False
    elif problem == "announcer":
        result["request_summary"]["announcer_counts_match"] = False
    elif problem == "lost_truck":
        result["request_summary"]["all_requests_admitted"] = False
    elif problem == "lost_vessel":
        result["vessel_work_summary"]["unadmitted_moves"] = 1
    elif problem == "inventory_chain":
        result["container_flow_summary"]["passed"] = False
    elif problem == "wait_cost":
        result["request_summary"]["accounted_wait_krw"] += 1
    elif problem == "physical_checks":
        result["request_summary"]["physical_invariants_enabled"] = False
    elif problem == "policy_exception":
        result["policy_exceptions"] = 1
    elif problem == "spatial_action":
        result["space"] = 1
    elif problem == "canonical_input":
        result["environment_manifest"]["canonical_input"]["schedule_sha256"] = "changed"
    elif problem == "environment_identity":
        assert not all(_recording_checks(result, environment_spec_sha256="changed").values())
        return
    else:
        result["days"].pop()
        result["request_summary"]["accounted_wait_krw"] -= 10.0
    assert not all(_recording_checks(result).values())


@pytest.mark.parametrize("guard,value", [("weights_unchanged", False),
    ("checkpoint_unchanged", False), ("runtime_unchanged", False), ("rollout_count", 1)])
def test_frozen_model_and_zero_training_guards(results, guard, value):
    assert not all(_recording_checks(results[0], **{guard: value}).values())


def test_pair_keeps_losses_backlog_and_only_one_descriptive_sample(results):
    before = deepcopy(results)
    summary = pair_summary(list(reversed(results)))
    assert summary["passed"]
    assert summary["independent_sample_count"] == 1
    assert not summary["statistical_inference"] and not summary["claim_eligible"]
    measurement = summary["costs"]["measurement_days"]
    assert measurement == dict(baseline_cost_krw=2800.0, time_only_cost_krw=3080.0,
                               saving_krw=-280.0, saving_pct=-10.0)
    assert summary["costs"]["all_days"]["baseline_cost_krw"] == 5800.0
    assert summary["costs"]["all_days"]["time_only_cost_krw"] == 6080.0
    assert summary["operational_outcomes"]["NO_REALLOC"]["truck_unfinished"] == 3
    assert summary["operational_outcomes"]["RL_TIME"]["truck_unfinished"] == 7
    assert results == before


@pytest.mark.parametrize("problem", ["missing_arm", "duplicate_arm", "seed",
    "canonical_input", "runtime_input", "runtime_source", "checkpoint", "network",
    "pair_contract", "recording_failure", "teacher_calls", "training_runs",
    "claim_eligible", "missing_day", "duplicate_day", "measurement_period",
    "nan_cost", "infinite_cost"])
def test_pair_rejects_unmatched_or_corrupt_evidence(results, problem):
    changed = results[1]
    if problem == "missing_arm":
        results.pop()
    elif problem == "duplicate_arm":
        changed["arm"] = "NO_REALLOC"
    elif problem == "seed":
        changed["seed"] += 100_000
    elif problem == "canonical_input":
        changed["expected_input"]["schedule_sha256"] = "changed"
    elif problem == "runtime_input":
        changed["environment_manifest"]["runtime_schedule_sha256"] = "changed"
    elif problem == "runtime_source":
        changed["runtime"]["engine_sha256"] = "changed"
    elif problem == "checkpoint":
        changed["checkpoint_sha256"] = "changed"
    elif problem == "network":
        changed["network_identities"][0]["state_sha256"] = "changed"
    elif problem == "pair_contract":
        changed["pair_contract_sha256"] = "changed"
    elif problem == "recording_failure":
        changed["recording_checks"]["request_recording"] = False
    elif problem == "teacher_calls":
        changed["rollout_calls"] = 1
    elif problem == "training_runs":
        changed["new_training_runs"] = 1
    elif problem == "claim_eligible":
        changed["claim_eligible"] = True
    elif problem == "missing_day":
        changed["days"].pop(15)
    elif problem == "duplicate_day":
        changed["days"][15] = deepcopy(changed["days"][14])
    elif problem == "measurement_period":
        changed["days"][15]["train"] = False
    else:
        changed["days"][15]["phi_krw"] = float("nan" if problem == "nan_cost" else "inf")
    with pytest.raises(ValueError):
        pair_summary(results)


@pytest.mark.parametrize("problem", ["same_wrong_measurement", "same_shortened_main_case"])
def test_matching_but_unregistered_measurement_scope_is_rejected(results, problem):
    for result in results:
        if problem == "same_wrong_measurement":
            result["days"][0]["train"] = True
        else:
            result["plan"] = result["plan"][:2]
            result["days"] = result["days"][:2]
            for day in result["days"]:
                day["train"] = False
    with pytest.raises(ValueError):
        pair_summary(results)
