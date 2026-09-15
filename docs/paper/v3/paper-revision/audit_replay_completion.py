"""Verify saved replay results without running a new simulation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/v3"))
from analyze_admission_replay import summarize_arm, sha


def main():
    base = ROOT / "outputs/reports/yr317_v3_request_audit"
    run = base / "replay-1b601c0"
    saved = json.loads((run / "full/summary.json").read_text(encoding="utf-8"))
    checks, inputs, reported = {}, {}, {}
    for arm in ("NO_REALLOC", "RL", "RL_NOVETO"):
        computed = summarize_arm(run / "full" / arm, base / "run-3279b7f",
            "1b601c04a0ec5f779a875921dd37ee116b5cfa9a")
        expected = dict(saved["arms"][arm])
        expected.pop("saving_percent")
        # JSON object keys are strings. Normalize only serialization types, not numbers.
        checks[arm + "_summary_matches_recalculation"] = json.loads(json.dumps(computed)) == expected
        checks[arm + "_request_cost_reconciliation"] = computed["ledger_reconciliation"]["passed"]
        prior = computed["prior_run_comparison"]
        checks[arm + "_prior_behavior_unchanged"] = (prior["same_day_count"]
            and prior["same_requested_identity"] and not prior["differences"])
        v = computed["vessel_work"]
        checks[arm + "_vessel_balance"] = (
            v["requested_moves"] == v["admitted_moves"] + v["unadmitted_moves"]
            and v["admitted_moves"] == v["completed_yard_jobs"]
            + v["outstanding_yard_jobs"] + v["unaccounted_yard_jobs"])
        reported[arm] = {"cost_krw": computed["total_cost_krw"],
            "truck_completed": computed["requests"]["states"]["COMPLETED"],
            "truck_skipped": computed["requests"]["skipped"],
            "failure_kinds": computed["failure_kinds"], "vessel_work": v}
        for name in ("manifest.json", "days.jsonl", "requests.jsonl.gz", "result.json", "progress.json"):
            path = run / "full" / arm / name
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ("progress.json", "full/summary.json", "full/results.md"):
        path = run / name
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    payload = {"date": "2026-09-16", "scope": "completed reused-seed replay, read-only verification",
        "checks": checks, "passed": all(checks.values()), "reported": reported,
        "artifact_sha256": inputs, "new_training_runs": 0, "independent_confirmation": False,
        "claim_eligible": False, "gate_reassessment_pending": True,
        "initial_comparison": {"record": "replay-completion-validation.json",
            "issue": "Direct Python-vs-JSON comparison treated integer day keys as different from their serialized string keys.",
            "resolution": "Normalize JSON representation only; retain the first receipt, with no changes to source results."}}
    (base / "replay-completion-verified.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "checks": checks}, ensure_ascii=True))
    return int(not payload["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
