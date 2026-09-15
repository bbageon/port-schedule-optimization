"""Preserve partial gates after the completed baseline; never promote a pending run."""
import hashlib
import json
from pathlib import Path
import subprocess

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, audit_dashboard)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (
        OUT / "auto-b0fe052/reconciliation-2.json", OUT / "run-3279b7f/NO_REALLOC/result.json",
        OUT / "run-3279b7f/NO_REALLOC/requests.jsonl.gz", OUT / "reconciliation-tests.xml")]
    snapshot = json.loads((ROOT / paths[0]).read_text(encoding="utf-8"))
    baseline = snapshot["completed_arms"]["NO_REALLOC"]
    if not snapshot["completed_artifacts_valid"] or not baseline["passed"]:
        raise SystemExit("The completed baseline did not reconcile; preserve the failed evidence.")
    board = audit_dashboard(ROOT, task_id="YR-317-g", expected_state="in-progress",
        spec_path=".claude/docs/dashboard-task-specs/YR-317-g-v3-review-demand-accounting.md",
        evidence_paths=paths, evidence_commits=[commit],
        remote_ref="origin/강화학습-판매", pin_commit=commit)
    if board.status is not GateStatus.PASS:
        print(json.dumps(board.as_dict(), ensure_ascii=False))
        raise SystemExit("Dashboard evidence does not reconcile.")
    evidence = {"scope": "completed NO_REALLOC baseline only; other two policies pending at snapshot",
        "board_commit": commit, "execution_commit": "3279b7fc41b3c9ae1a5c44ed92327b6d648160c1",
        "artifact_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "dashboard": board.as_dict(), "requested_trucks": baseline["requested"],
        "truck_states": baseline["states"], "vessel_admissions": baseline["vessel_admissions"],
        "record_reconciliation_passed": True, "claim_eligible": False}
    gates = ResearchGateReport(
        GateOutcome("performance", GateStatus.INCONCLUSIVE,
            "No independent comparison is available; two diagnostic policies remain pending.",
            ("Reused-seed ten-day diagnostic does not establish a general cost benefit.",), evidence),
        GateOutcome("reliability", GateStatus.FAIL,
            "Baseline records reconcile, but the complete review evidence remains unresolved.",
            ("Two main policy ledgers remain pending at the saved snapshot.",
             "Submitted manuscript identity and text-to-evidence corrections remain separate pending work."), evidence),
        GateOutcome("scenario_validity", GateStatus.FAIL,
            "The requested workload is not fully admitted in the diagnostic baseline.",
            ("170 requested trucks were skipped before gate entry and have zero cost under the existing formula.",
             "Recorded vessel requests contain 44,322 moves, but only 43,394 moves were admitted.",
             "Full vessel completion and the cause of unavailable outbound targets remain unverified."), evidence))
    payload = {"schema": "yr317.v3.partial-audit-gates.v1", "common_gates": gates.as_dict(),
        "claim_eligible": False, "snapshot_at": snapshot["snapshot_at"]}
    (OUT / "baseline_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dashboard": board.status.value, "research_unresolved": gates.unresolved,
                      "claim_eligible": False}))


if __name__ == "__main__":
    main()
