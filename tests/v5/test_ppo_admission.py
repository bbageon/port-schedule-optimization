"""Injected failures test diagnostics, NOT the unexplained original 30-day loss."""
import json
import os
from types import SimpleNamespace

import pytest

from yard_rl.v5.ppo import continuous
from yard_rl.v5.ppo.journal import AdmissionFailure, RunJournal
from yard_rl.v5.ppo.runtime import PPORuntime
from yard_rl.v5.stage import month_run
from yard_rl.v5.stage.month_run import MonthResult


@pytest.fixture
def test_stamp(monkeypatch):
    monkeypatch.setattr(continuous, "code_stamp", lambda: {"pid": os.getpid(), "test_only": True})


def read(out, name):
    return json.loads((out / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("failure", ["capacity", "no_target", "tail"])
def test_real_truck_admission_stops_before_policy_boundary(test_stamp, monkeypatch, tmp_path, failure):
    build = month_run.build_month
    captured = {}
    def bad_schedule(*args, **kwargs):
        built = build(*args, **kwargs)
        # Only this injected first event fails; original distributions are unchanged in production.
        e = dict(built["schedule"][0], arrival_s=60.0, lead_s=60.0)
        e["flow"] = "GATE_OUT" if failure == "no_target" else "GATE_IN"
        if failure == "tail":
            e["travel_s"] = 4 * 86400.0
        built["schedule"] = [e]
        captured.update(e)
        return built
    monkeypatch.setattr(month_run, "build_month", bad_schedule)
    if failure == "no_target":
        monkeypatch.setattr(month_run, "make_retarget", lambda seed: lambda *args: None)
    if failure == "capacity":
        bind = PPORuntime.bind
        def full(self, *args):
            bind(self, *args)
            self.mbt.capacity_margin = 100000
        monkeypatch.setattr(PPORuntime, "bind", full)
    def forbidden(*args, **kwargs):
        raise AssertionError("Policy must not advance after this first admission failure")
    monkeypatch.setattr(PPORuntime, "boundary", forbidden)
    out = tmp_path / failure
    with pytest.raises(AdmissionFailure):
        continuous.run_continuous(output=out, n_days=3, load=1)
    status, admissions = read(out, "status.json"), read(out, "admissions.json")
    assert status["state"] == "failed" and status["time_s"] == 0
    assert status["last_policy_boundary_s"] is None and status["updates"] == 0
    assert admissions["admitted"] == 0 and admissions["skipped"] == 1
    row = admissions["truck_failures"][0]
    assert row["job_id"] == captured["job_id"] and row["block"] == captured["block"]
    assert row["flow"] == captured["flow"] and row["block_snapshot"]["time_s"] == 0
    assert ("용량 부족" in row["reason"] if failure == "capacity"
            else row["reason"] == {"no_target": "NO_TARGET", "tail": "TAIL"}[failure])
    assert (out / "failed-policy.pt").exists() and not (out / "final.pt").exists()
    assert status["snapshot_errors"] == []


def test_real_vessel_failure_is_saved_with_identity(test_stamp, monkeypatch, tmp_path):
    plan = month_run.plan_month_vessels
    captured = {}
    def bad_plan(*args, **kwargs):
        rows = plan(*args, **kwargs)
        rows[0][0] = dict(rows[0][0], moves=0, work="DISCHARGE")
        captured.update(rows[0][0])
        return rows
    monkeypatch.setattr(month_run, "plan_month_vessels", bad_plan)
    out = tmp_path / "vessel"
    with pytest.raises(AdmissionFailure, match="물량이 0"):
        continuous.run_continuous(output=out, n_days=3, load=1)
    status, admissions = read(out, "status.json"), read(out, "admissions.json")
    assert status["time_s"] == 0 and status["updates"] == 0
    assert admissions["vessel_failed"] == 1
    row = admissions["vessels"][-1]
    assert not row["ok"] and row["key"] == captured["key"] and row["block"] == captured["block"]
    assert row["asked"] == row["moves"] == 0
    assert read(out, "partial_report.json")["state"] == "failed"
    assert (out / "failed-policy.pt").exists() and not (out / "final.pt").exists()


def test_result_is_saved_before_final_validation(test_stamp, monkeypatch, tmp_path):
    result = MonthResult(skipped=1, truck_skips=[{"job_id": "missing", "reason": "injected"}])
    def finished(**kwargs):
        kwargs["ppo"].mbt = SimpleNamespace(blocks={})
        return result
    monkeypatch.setattr(continuous, "run_month", finished)
    out = tmp_path / "validation"
    with pytest.raises(RuntimeError, match="boundaries"):
        continuous.run_continuous(output=out, n_days=3, load=1)
    assert read(out, "month_result.json")["truck_skips"] == result.truck_skips
    assert (out / "cohort_reports.json").exists() and not (out / "final.pt").exists()
    assert read(out, "status.json")["state"] == "failed"
    assert read(out, "status.json")["skipped"] == 1


def test_snapshot_error_does_not_hide_original_failure(test_stamp, monkeypatch, tmp_path):
    def failed(**kwargs):
        raise RuntimeError("original world failure")
    checkpoint = RunJournal.checkpoint
    def disk_failure(self, name, runtime):
        if name == "failed-policy.pt":
            raise OSError("injected disk failure")
        return checkpoint(self, name, runtime)
    monkeypatch.setattr(continuous, "run_month", failed)
    monkeypatch.setattr(RunJournal, "checkpoint", disk_failure)
    out = tmp_path / "disk"
    with pytest.raises(RuntimeError, match="original world failure"):
        continuous.run_continuous(output=out, n_days=3, load=1)
    status = read(out, "status.json")
    assert "original world failure" in status["error"]
    assert status["snapshot_errors"] == ["failed_checkpoint: injected disk failure"]


def test_unwritable_failure_journal_preserves_original_exception(test_stamp, monkeypatch, tmp_path):
    def failed(**kwargs):
        raise RuntimeError("original world failure")
    def failed_journal(*args):
        raise OSError("injected journal failure")
    monkeypatch.setattr(continuous, "run_month", failed)
    monkeypatch.setattr(RunJournal, "fail", failed_journal)
    with pytest.raises(RuntimeError, match="original world failure") as raised:
        continuous.run_continuous(output=tmp_path / "unwritable", n_days=3, load=1)
    assert any("injected journal failure" in note for note in raised.value.__notes__)


def test_partial_vessel_clipping_is_recorded_not_newly_rejected(tmp_path):
    journal = RunJournal(tmp_path / "journal", [], {"code": {"pid": 0}, "ppo": {}})
    journal.admission({"kind": "vessel", "time_s": 0, "key": "v", "block": "b",
                       "ok": True, "asked": 10, "moves": 8, "why": "stock clipped"})
    journal.save_admissions()
    data = read(journal.output, "admissions.json")
    assert data["vessel_failed"] == 0 and data["vessels"][0]["moves"] == 8
