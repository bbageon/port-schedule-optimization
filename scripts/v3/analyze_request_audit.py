"""Read-only reconciliation of the frozen YR-317-g diagnostic, including partial runs.

Uses only the standard library; never imports or reruns the simulator. Cost constants
below reproduce the frozen experiment's formula, not a newly introduced charge.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN_COMMIT = "3279b7fc41b3c9ae1a5c44ed92327b6d648160c1"
ARMS = ("NO_REALLOC", "RL", "RL_NOVETO")
IDENTITY = ("job_id", "requested_day", "flow", "requested_block",
            "requested_arrival_s", "lead_s", "requested_target", "travel_s")
TIMES = ("gate_in_s", "block_in_s", "service_start_s", "job_done_s", "gate_out_s")
ADMITTED = ("COMPLETED", "CENSORED", "ADMITTED_PRE_GATE")
COSTS = ("c_wait", "c_move", "c_rehandle", "c_vessel")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-4)


def reconcile(result, manifest, rows, live_days):
    """Independently check raw events, cohort totals, and the existing cost formula."""
    issues = Counter()
    examples = defaultdict(list)

    def check(ok, kind, key="summary"):
        if not ok:
            issues[kind] += 1
            if len(examples[kind]) < 5:
                examples[kind].append(key)

    summary = result["request_summary"]
    end = summary["end_s"]
    states, reasons, cohorts, skip_groups = Counter(), Counter(), {}, Counter()
    seen, identity = set(), []
    wait_total = skipped_cost = 0.0
    for row in rows:
        key = row["job_id"]
        check(key not in seen, "duplicate_request", key)
        seen.add(key)
        identity.append({k: row[k] for k in IDENTITY})
        times = [row[k] for k in TIMES]
        observed = [v for v in times if v is not None]
        check(all(math.isfinite(v) and 0 <= v <= end + 1e-6 for v in observed),
              "invalid_event_time", key)
        check(all(b >= a - 1e-6 for a, b in zip(observed, observed[1:])),
              "event_time_reversal", key)
        # Service start is optional; the four externally observed stages may not skip.
        required = [row[k] for k in (TIMES[0], TIMES[1], TIMES[3], TIMES[4])]
        check(not any(v is not None and None in required[:i] for i, v in enumerate(required)),
              "missing_prior_stage", key)
        check(times[2] is None or times[1] is not None, "service_without_block_entry", key)
        events = row["admission_events"]
        check(len(events) == 1, "admission_event_count", key)
        event = events[-1] if events else {}
        outcome = event.get("outcome", "UNPROCESSED")
        check(outcome in ("ADMITTED", "SKIPPED", "UNPROCESSED"), "unknown_outcome", key)
        if events:
            check(event["job_id"] == key and event["flow"] == row["flow"]
                  and event["block"] == row["requested_block"], "admission_identity", key)
            check(close(event["planned_gate_in_s"], row["requested_arrival_s"]),
                  "original_arrival_changed", key)
            check(0 <= event["t"] <= end and
                  (times[0] is None or event["t"] <= times[0] + 1e-6),
                  "admission_time", key)
        if outcome == "ADMITTED":
            state = ("ADMITTED_PRE_GATE" if times[0] is None or times[0] > end else
                     "COMPLETED" if times[-1] is not None and times[-1] <= end else "CENSORED")
        else:
            state = outcome
            check(not observed, "events_on_nonadmitted_request", key)
        check(state == row["state"], "state_mismatch", key)
        states[state] += 1
        if outcome == "SKIPPED":
            reason = event.get("reason")
            check(bool(reason), "missing_skip_reason", key)
            reasons[reason] += 1
            skip_groups[(row["requested_day"], row["requested_block"], row["flow"], reason)] += 1
            skipped_cost += row["accounted_wait_krw"]
        tt = None if times[0] is None else max(0.0, min(end, times[-1] if times[-1] is not None else end) - times[0])
        cost = 0.0 if tt is None else 40000.0 * (tt + max(0.0, tt - 3600.0)) / 3600.0
        recorded_tt = row["accounted_turn_time_s"]
        check(recorded_tt is None if tt is None else recorded_tt is not None and close(tt, recorded_tt),
              "turn_time_mismatch", key)
        check(close(cost, row["accounted_wait_krw"]), "request_cost_mismatch", key)
        wait_total += cost
        cohort = cohorts.setdefault(row["requested_day"],
            {"requested": 0, "states": Counter(), "accounted_wait_krw": 0.0,
             "gate_entered": 0, "flows": Counter()})
        cohort["requested"] += 1
        cohort["states"][state] += 1
        cohort["flows"][row["flow"]] += 1
        cohort["accounted_wait_krw"] += cost
        cohort["gate_entered"] += times[0] is not None and times[0] <= end

    check(len(rows) == summary["requested"] == sum(d["load"] for d in result["plan"]), "requested_count")
    check(dict(states) == summary["states"], "state_totals")
    check(sum(states[s] for s in ADMITTED) == summary["admitted"], "admitted_total")
    check(states["SKIPPED"] == summary["skipped"], "skipped_total")
    check(states["UNPROCESSED"] == summary["unprocessed"], "unprocessed_total")
    check(dict(reasons) == summary["skip_reasons"], "skip_reason_totals")
    check(close(wait_total, summary["accounted_wait_krw"]), "summary_cost")
    check(digest(identity) == result["requested_identity_sha256"], "request_identity_hash")
    settings = manifest["contract"]["settings"]
    check(settings["days"] == result["plan"] and settings["arm"] == result["arm"]
          and settings["seed"] == result["seed"], "execution_contract")
    check(settings["labels_per_day"] is None and settings["explore"] == 0
          and settings["capture_requests"] is True, "diagnostic_settings")
    check(result["repro"] == manifest["repro"], "repro_manifest")
    check(result["repro"]["code"]["git_head"] == FROZEN_COMMIT
          and result["repro"]["code"]["git_dirty"] is False, "frozen_clean_code")
    check(result["rollout_calls"] == result["policy_exceptions"] == 0, "runtime_guards")
    expected_checks = {"request_recording", "announcer_counts", "wait_cost_reconciles",
                       "no_teacher", "no_policy_exceptions"}
    check(expected_checks <= result["recording_checks"].keys() and
          all(result["recording_checks"].get(k) is True for k in expected_checks), "runner_checks")
    check(summary["recording_ok"] is True and not summary["recording_issues"]
          and summary["announcer_counts_match"] is True, "runner_summary")
    check(result["claim_eligible"] is False, "claim_scope")
    check([d["index"] for d in live_days] == [d["index"] for d in result["plan"]]
          == [d["index"] for d in result["days"]], "day_sequence")
    for day in result["days"]:
        index = day["index"]
        cohort = cohorts.get(index, {})
        check(cohort.get("requested") == day["load"], "cohort_request_count", index)
        check(cohort.get("gate_entered") == day["n_trucks"], "cohort_gate_count", index)
        check(close(cohort.get("accounted_wait_krw", -1), day["c_wait"]), "cohort_cost", index)
        check(cohort.get("states", {}).get("CENSORED", 0) == day["n_censored"], "cohort_censored", index)
        check(close(sum(day[k] for k in COSTS), day["phi_krw"]), "four_cost_total", index)
        check(day["provisional"] is False, "final_day_is_provisional", index)
    live_reasons = Counter()
    for day in live_days:
        live_reasons.update(day["truck_skip_reasons"])
    check(dict(live_reasons) == dict(reasons) and
          sum(d["truck_skipped"] for d in live_days) == summary["skipped"], "daily_skip_totals")
    vessels = result["vessel_admissions"]
    return {"passed": not issues, "issues": dict(issues), "examples": dict(examples),
        "requested": len(rows), "states": dict(states), "skip_reasons": dict(reasons),
        "cohorts": cohorts, "skip_groups": [{"day": d, "block": b, "flow": f, "reason": r, "count": n}
            for (d, b, f, r), n in sorted(skip_groups.items())],
        "accounted_wait_krw": wait_total, "skipped_accounted_wait_krw": skipped_cost,
        "four_cost_totals_krw": {k: sum(d[k] for d in result["days"]) for k in COSTS},
        "vessel_admissions": {"streams": len(vessels), "failed": sum(v["ok"] is False for v in vessels),
            "admitted_moves": sum(v.get("moves", 0) for v in vessels),
            "asked_moves_in_recorded_streams": sum(v.get("asked", 0) for v in vessels),
            "missing_asked_fields": sum("asked" not in v for v in vessels),
            "completion_validated": False},
        "full_truck_demand_completed": states["COMPLETED"] == len(rows) and bool(rows)}


def analyze(run):
    completed, pending, problems = {}, {}, []
    identities = set()
    for label in ("pilot_RL", *ARMS):
        folder = run / label
        if not (folder / "result.json").exists():
            live_path = folder / "days.jsonl"
            # The live writer can be between write calls. Ignore only its unfinished last line.
            data = live_path.read_text(encoding="utf-8") if live_path.exists() else ""
            lines = data.splitlines(keepends=True)
            days = [json.loads(line) for line in lines if line.endswith("\n")]
            pending[label] = {"state": "running" if (folder / "manifest.json").exists() else "not_started",
                "completed_day_count": len(days), "live_rows": days,
                "costs_are_provisional": True}
            continue
        result, manifest = read_json(folder / "result.json"), read_json(folder / "manifest.json")
        with gzip.open(folder / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        live = [json.loads(line) for line in (folder / "days.jsonl").read_text(encoding="utf-8").splitlines()]
        audit = reconcile(result, manifest, rows, live)
        artifacts = {name: sha(folder / name) for name in
                     ("manifest.json", "result.json", "requests.jsonl.gz", "days.jsonl")}
        repro = result["repro"]
        checks = {"ledger_sha256": artifacts["requests.jsonl.gz"] == result["request_ledger_sha256"],
            "checkpoint_sha256": sha(ROOT / "outputs/v3/month-02/ckpt_029.pt") == repro["checkpoint_sha256"],
            "prereg_sha256": sha(run.parent / "prereg.md") == repro["prereg_sha256"],
            "runner_sha256": sha(ROOT / "scripts/v3/run_request_audit.py") == repro["runner_sha256"]}
        audit.update(artifact_sha256=artifacts, artifact_checks=checks, elapsed_s=result["elapsed_s"],
                     requested_identity_sha256=result["requested_identity_sha256"])
        audit["passed"] = audit["passed"] and all(checks.values())
        if label in ARMS:
            identities.add(result["requested_identity_sha256"])
            original = ROOT / "outputs/v3/judge-consent/arms" / f"arm_{label}.json"
            old = {d["index"]: d for d in read_json(original)["days"]}
            fields = ("load", "truck_skipped", "traded", "n_space", "n_time")
            audit["original_prefix_diagnostic"] = {"source_sha256": sha(original),
                "scope": "descriptive counts only; 10-day cutoff differs from original 30-day run",
                "differences": [{"day": d["index"], "field": k, "original": old[d["index"]][k], "replay": d[k]}
                    for d in result["days"] for k in fields if old[d["index"]][k] != d[k]]}
        if not audit["passed"]:
            problems.append(f"{label}: raw-record reconciliation failed")
        completed[label] = audit
    if len(identities) > 1:
        problems.append("Completed main policies have different requested work.")
    if (run / "failure.json").exists():
        problems.append("Runner failure.json exists; inspect the preserved traceback.")
    return {"schema": "yr317.request-audit.reconciliation.v1",
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run.relative_to(ROOT)), "analyzer_sha256": sha(__file__),
        "completed_arms": completed, "pending_arms": pending, "problems": problems,
        "completed_artifacts_valid": bool(completed) and not problems,
        "all_main_arms_complete": all(arm in completed for arm in ARMS),
        "same_requested_work_across_all_main_arms": len(identities) == 1 if all(arm in completed for arm in ARMS) else None,
        "claim_eligible": False, "independent_confirmatory_runs": 0,
        "interpretation": {"NO_TARGET": "No unclaimed container available in the requested block at announcement; not proof of an empty yard or a congestion cause.",
            "cost": "Existing gate-entry cost excludes skipped requests; zero recorded cost is not evidence of zero operational burden.",
            "days": "Live-day skip counters follow announcement intervals; ledger cohorts follow original requested day.",
            "scope": "Fixed, reused-seed 10-day diagnostic; no new training or performance inference."}}


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository containing the input evidence")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New snapshot path; never overwrite evidence")
    args = parser.parse_args()
    ROOT = args.repo.resolve()
    report = analyze(args.run.resolve())
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": list(report["completed_arms"]),
        "pending": {k: v["completed_day_count"] for k, v in report["pending_arms"].items()},
        "completed_artifacts_valid": report["completed_artifacts_valid"],
        "all_main_arms_complete": report["all_main_arms_complete"], "problems": report["problems"]}))
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
