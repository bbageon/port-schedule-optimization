"""Record the completed diagnostic and observer validation without promoting performance."""
import hashlib
import json
from pathlib import Path
import subprocess

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, audit_dashboard, judge_claim_alignment)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, encoding="utf-8").strip()
    remote = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "@{u}"],
                                     cwd=ROOT, encoding="utf-8").strip()
    names = ["auto-b0fe052/reconciliation-4.json", "completed-diagnostic.md", "observer-tests.json",
             "admission-probe-prereg.md", "probe-09611b3-r2/comparison.json",
             "probe-09611b3-r2/off.json", "probe-09611b3-r2/on.json"]
    names += [f"run-3279b7f/{arm}/{name}" for arm in ("NO_REALLOC", "RL", "RL_NOVETO")
              for name in ("result.json", "requests.jsonl.gz", "manifest.json", "days.jsonl")]
    paths = [(OUT / n).relative_to(ROOT).as_posix() for n in names]
    snapshot = read(OUT / names[0])
    probe = read(OUT / "probe-09611b3-r2/comparison.json")
    if not (snapshot["completed_artifacts_valid"] and snapshot["all_main_arms_complete"] and
            snapshot["same_requested_work_across_all_main_arms"] and probe["passed"] and
            all(probe["checks"].values())):
        raise SystemExit("The saved diagnostic evidence did not pass its recording checks.")
    raw = {}
    for arm in ("NO_REALLOC", "RL", "RL_NOVETO"):
        result = read(OUT / f"run-3279b7f/{arm}/result.json")
        raw[f"{arm}.completed"] = result["request_summary"]["states"]["COMPLETED"]
        raw[f"{arm}.skipped"] = result["request_summary"]["skipped"]
        raw[f"{arm}.vessel_reduced"] = sum(v["asked"] - v["moves"] for v in result["vessel_admissions"])
    expected = {f"{arm}.{key}": value for arm, values in {
        "NO_REALLOC": (68830, 170, 928), "RL": (68820, 180, 1027),
        "RL_NOVETO": (68864, 136, 1055)}.items()
        for key, value in zip(("completed", "skipped", "vessel_reduced"), values)}
    alignment = judge_claim_alignment(expected, raw)
    board = audit_dashboard(ROOT, task_id="YR-317-g", expected_state="in-progress",
        spec_path=".claude/docs/dashboard-task-specs/YR-317-g-v3-review-demand-accounting.md",
        evidence_paths=paths, evidence_commits=[commit], remote_ref=remote, pin_commit=commit)
    if board.status is not GateStatus.PASS or alignment.status is not GateStatus.PASS:
        raise SystemExit(json.dumps({"board": board.as_dict(), "counts": alignment.as_dict()}, ensure_ascii=False))
    evidence = {"scope": "completed reused-seed 10-day diagnosis plus a 300-request observer check",
        "board_commit": commit, "execution_commit": "3279b7fc41b3c9ae1a5c44ed92327b6d648160c1",
        "observer_execution_commit": probe["modes"]["on"]["repro"]["code"]["git_head"],
        "artifact_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "dashboard": board.as_dict(), "claim_alignment": alignment.as_dict(),
        "completed_record_reconciliation": True, "observer_checks": probe["checks"],
        "full_workload_counts": raw, "observer_vessel_counts": probe["modes"]["on"]["vessel_work_summary"]}
    gates = ResearchGateReport(
        GateOutcome("performance", GateStatus.INCONCLUSIVE,
            "No independent performance confirmation has been run.",
            ("One reused-seed shortened run and an observer equivalence check cannot establish general benefit.",), evidence),
        GateOutcome("reliability", GateStatus.FAIL,
            "Saved diagnostic records reconcile, but the full failure-state evidence is incomplete.",
            ("The original ten-day artifacts lack vessel completion and inventory-at-failure observations.",
             "RL policy counts differ from the historical run from its first day; exact replay is not established.",
             "The new observer is validated on a small workload, not on the ten-day failure states."), evidence),
        GateOutcome("scenario_validity", GateStatus.FAIL,
            "Requested workload is reduced by the current admission behavior.",
            ("Truck requests skipped: 170, 180, 136; corresponding vessel moves not admitted: 928, 1027, 1055.",
             "Adding observation does not fix admission losses or validate realistic terminal operation."), evidence))
    payload = {"schema": "yr317.v3.completed-diagnostic-gates.v1", "common_gates": gates.as_dict(),
        "claim_eligible": False, "observer_validation_passed": True, "new_training_runs": 0,
        "independent_confirmatory_runs": 0}
    with (OUT / "completed_gate.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"dashboard": board.status.value, "claim_alignment": alignment.status.value,
        "observer_validation": "PASS", "unresolved_research_gates": gates.unresolved,
        "claim_eligible": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
