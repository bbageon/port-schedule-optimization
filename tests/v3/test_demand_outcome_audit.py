"""Saved evidence must not turn retained-but-unfinished work into completion."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v3"))
from audit_demand_outcomes import inspect


def evidence():
    vessel = dict(key="ship", block="B", work="LOAD", start_s=10, asked=2,
        admitted=2, completed_yard_jobs=2, outstanding_yard_jobs=0,
        unadmitted_moves=0, unaccounted_yard_jobs=0, attempted=True,
        sts_done=True, sts_remaining=0, sts_completion_s=90)
    result = dict(plan=[{}], request_summary=dict(end_s=100, late_target_bindings=1,
        unbound_jobs_at_end=0), vessel_work_ledger=[vessel], vessel_work_summary=dict(
        streams=1, requested_moves=2, admitted_moves=2, completed_yard_jobs=2,
        outstanding_yard_jobs=0, unadmitted_moves=0, unaccounted_yard_jobs=0,
        all_requested_work_completed=True), demand_bindings=[dict(
        job_id="truck", at_s=25, target="box", block="B", flow="GATE_OUT")])
    request = dict(job_id="truck", flow="GATE_OUT", requested_arrival_s=10,
        gate_in_s=20, block_in_s=25, job_done_s=35, gate_out_s=40, state="COMPLETED")
    return result, [request]


def test_completed_work_and_unpriced_time_are_separate():
    result, rows = evidence()
    before = deepcopy((result, rows))
    audit = inspect(result, rows)
    assert audit["passed"] and audit["all_trucks_completed"] and audit["all_vessel_work_completed"]
    assert audit["arrival_shift"]["total_hours"] == 10 / 3600
    assert (result, rows) == before
    assert not audit["claim_eligible"]


def test_ship_incomplete_after_all_yard_work_is_not_full_completion():
    result, rows = evidence()
    result["vessel_work_ledger"][0].update(sts_done=False, sts_remaining=1, sts_completion_s=None)
    result["vessel_work_summary"]["all_requested_work_completed"] = False
    audit = inspect(result, rows)
    assert audit["passed"] and not audit["all_vessel_work_completed"]
    assert len(audit["vessels_with_unfinished_work"]) == 1


def test_corrupt_vessel_totals_fail_even_if_saved_recording_flag_passed():
    result, rows = evidence()
    result["vessel_work_summary"]["completed_yard_jobs"] = 3
    result["vessel_work_summary"]["recording_ok"] = True
    assert not inspect(result, rows)["passed"]


def test_missing_and_unfinished_work_cannot_be_relabelled_completed():
    result, rows = evidence()
    result["vessel_work_ledger"][0]["outstanding_yard_jobs"] = 1
    audit = inspect(result, rows)
    assert not audit["passed"] and not audit["all_vessel_work_completed"]


def test_early_or_duplicate_binding_fails():
    result, rows = evidence()
    result["demand_bindings"][0]["at_s"] = 24
    assert not inspect(result, rows)["passed"]
    result, rows = evidence()
    result["demand_bindings"].append(deepcopy(result["demand_bindings"][0]))
    assert not inspect(result, rows)["passed"]


def test_no_gate_event_is_unfinished_not_zero_delay_completion():
    result, rows = evidence()
    result["demand_bindings"] = []
    result["request_summary"]["late_target_bindings"] = 0
    rows[0].update(gate_in_s=None, state="ADMITTED_PRE_GATE")
    audit = inspect(result, rows)
    assert audit["passed"] and not audit["all_trucks_completed"]
    assert audit["arrival_shift"]["later_gate_count"] == 0
    assert audit["unfinished_trucks_by_state"] == {"ADMITTED_PRE_GATE": 1}
