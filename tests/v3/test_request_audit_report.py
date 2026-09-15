"""The independent report must reject corrupted evidence, not just trust passed flags."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("request_audit_report",
    ROOT / "scripts/v3/analyze_request_audit.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
PILOT = ROOT / "outputs/reports/yr317_v3_request_audit/run-3279b7f/pilot_RL"


@pytest.fixture
def evidence():
    result = REPORT.read_json(PILOT / "result.json")
    manifest = REPORT.read_json(PILOT / "manifest.json")
    with gzip.open(PILOT / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    live = [json.loads(line) for line in (PILOT / "days.jsonl").read_text(encoding="utf-8").splitlines()]
    return result, manifest, rows, live


def test_real_completed_pilot_reconciles(evidence):
    audit = REPORT.reconcile(*evidence)
    assert audit["passed"], audit["issues"]
    assert audit["requested"] == audit["states"]["COMPLETED"] == 300
    assert audit["full_truck_demand_completed"]
    assert not audit["vessel_admissions"]["completion_validated"]


@pytest.mark.parametrize("corruption,expected", [
    ("missing_stage", "missing_prior_stage"),
    ("changed_cost", "request_cost_mismatch"),
    ("moved_cohort", "cohort_request_count"),
    ("other_request_event", "admission_identity"),
    ("missing_guard", "runner_checks"),
    ("duplicate_request", "duplicate_request"),
    ("skipped_with_events", "events_on_nonadmitted_request"),
])
def test_corrupt_evidence_fails_despite_runner_passed_flags(evidence, corruption, expected):
    result, manifest, rows, live = deepcopy(evidence)
    if corruption == "missing_stage":
        rows[0]["block_in_s"] = None
    elif corruption == "changed_cost":
        rows[0]["accounted_wait_krw"] += 1000
    elif corruption == "moved_cohort":
        rows[0]["requested_day"] = 7
    elif corruption == "other_request_event":
        rows[0]["admission_events"][0]["job_id"] = "other"
    elif corruption == "missing_guard":
        del result["recording_checks"]["no_policy_exceptions"]
    elif corruption == "duplicate_request":
        rows.append(deepcopy(rows[0]))
    elif corruption == "skipped_with_events":
        rows[0]["admission_events"][0].update(outcome="SKIPPED", reason="NO_TARGET")
    audit = REPORT.reconcile(result, manifest, rows, live)
    assert not audit["passed"]
    assert audit["issues"][expected] > 0


def test_a_partial_day_file_never_becomes_a_completed_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(REPORT, "ROOT", tmp_path)
    run = tmp_path / "run"
    arm = run / "NO_REALLOC"
    arm.mkdir(parents=True)
    (arm / "manifest.json").write_text("{}", encoding="utf-8")
    (arm / "days.jsonl").write_text('{"index": 0}\n{"index":', encoding="utf-8")
    audit = REPORT.analyze(run)
    assert audit["completed_arms"] == {}
    assert audit["pending_arms"]["NO_REALLOC"]["completed_day_count"] == 1
    assert not audit["all_main_arms_complete"]
    assert audit["same_requested_work_across_all_main_arms"] is None
    assert not audit["completed_artifacts_valid"]


def test_runner_failure_is_preserved_as_a_report_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(REPORT, "ROOT", tmp_path)
    (tmp_path / "failure.json").write_text("{}", encoding="utf-8")
    audit = REPORT.analyze(tmp_path)
    assert audit["problems"]
    assert not audit["completed_artifacts_valid"]
