from types import SimpleNamespace

import pytest

from yard_rl.v3.schema.record import ExecutionRecord
from yard_rl.v3.stage import orders as mod
from yard_rl.v3.stage.request_audit import request_ledger


def entry(key, **kw):
    return {"job_id": key, "day": 0, "flow": "GATE_IN", "block": "Y01",
            "arrival_s": 120.0, "lead_s": 120.0, "travel_s": 30.0, **kw}


def test_observer_keeps_the_same_admission_decisions_and_own_branch_records(monkeypatch):
    schedule = [entry("ok"), entry("tail", travel_s=200), entry("no_target", flow="GATE_OUT"), entry("full")]
    monkeypatch.setattr(mod, "_job_from_entry", lambda e, t: SimpleNamespace(job_id=e["job_id"]))
    def admit(block, job, **kw):
        if job.job_id == "full":
            raise mod.TransferError("capacity test")
    terminal = SimpleNamespace(admit_external_job=admit)
    results = []
    for capture in (False, True):
        ann = mod.V3Announcer(schedule, end_s=300, retarget=lambda *_: None, record_admissions=capture)
        ann.review(terminal, 0)
        results.append((ann.n_admitted, ann.n_skipped, ann.skips))
        assert len(ann.admission_events) == (4 if capture else 0)
    assert results[0] == results[1]
    assert (ann.n_admitted, ann.n_skipped) == (1, 3)
    assert [e["reason"] for e in ann.admission_events] == [None, "TAIL", "NO_TARGET", "capacity test"]
    clone = ann.clone_fresh()
    window = ann.window(0, 60)
    assert clone.record_admissions and window.record_admissions
    assert clone.admission_events == window.admission_events == []
    clone.review(terminal, 0)
    assert len(ann.admission_events) == len(clone.admission_events) == 4


def test_ledger_preserves_complete_censored_skipped_and_unprocessed_requests():
    schedule = [entry(k) for k in ("done", "pending", "skip", "missing")]
    events = [{"job_id": k, "outcome": status, "reason": reason}
              for k, status, reason in [("done", "ADMITTED", None),
                ("pending", "ADMITTED", None), ("skip", "SKIPPED", "NO_TARGET")]]
    records = {
        "done": ExecutionRecord("done", gate_in_s=120, block_in_s=130, service_start_s=140, job_done_s=150, gate_out_s=160),
        "pending": ExecutionRecord("pending", gate_in_s=120),
        "skip": ExecutionRecord("skip"), "missing": ExecutionRecord("missing")}
    rows, summary = request_ledger(schedule, events, records, {}, end_s=200)
    assert [r["state"] for r in rows] == ["COMPLETED", "CENSORED", "SKIPPED", "UNPROCESSED"]
    assert [r["accounted_turn_time_s"] for r in rows] == [40, 80, None, None]
    assert summary["requested"] == summary["admitted"] + summary["skipped"] + summary["unprocessed"] == 4
    assert not summary["recording_ok"]  # Missing admission record is not silently inferred.
    assert not summary["all_requests_admitted"]
    assert summary["skip_reasons"] == {"NO_TARGET": 1}


@pytest.mark.parametrize("problem", ["duplicate", "unknown", "contradiction", "time_order"])
def test_inconsistent_records_cannot_pass(problem):
    schedule = [entry("a")]
    events = [{"job_id": "a", "outcome": "ADMITTED", "reason": None}]
    rec = ExecutionRecord("a", gate_in_s=120, block_in_s=130, service_start_s=140, job_done_s=150, gate_out_s=160)
    if problem == "duplicate":
        events.append(events[0].copy())
    elif problem == "unknown":
        events.append({"job_id": "other", "outcome": "ADMITTED", "reason": None})
    elif problem == "contradiction":
        events[0]["outcome"] = "SKIPPED"
    else:
        rec.gate_out_s = 100
    _, summary = request_ledger(schedule, events, {"a": rec}, {}, end_s=200)
    assert not summary["recording_ok"]


def test_zero_time_and_request_day_are_preserved():
    rows, summary = request_ledger([entry("zero", day=2, arrival_s=0)],
        [{"job_id": "zero", "outcome": "ADMITTED", "reason": None}],
        {"zero": ExecutionRecord("zero", gate_in_s=0, block_in_s=0, service_start_s=0, job_done_s=0, gate_out_s=0)},
        {}, end_s=200)
    assert rows[0]["state"] == "COMPLETED" and rows[0]["requested_day"] == 2
    assert rows[0]["accounted_wait_krw"] == 0
    assert summary["recording_ok"] and summary["all_requests_admitted"]
