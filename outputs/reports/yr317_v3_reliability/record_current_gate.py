"""Record remaining v3 review debt after the tested code correction."""
import hashlib
import json
import subprocess
from pathlib import Path

from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, audit_dashboard)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = ["outputs/reports/yr317_v3_reliability/validation.json",
             "outputs/reports/yr317_v3_reliability/tests.xml"]
    audit = audit_dashboard(ROOT, task_id="YR-317-a", expected_state="in-progress",
        spec_path=".claude/docs/dashboard-task-specs/YR-317-a-v3-review-evidence-audit.md",
        evidence_paths=paths, evidence_commits=[commit],
        remote_ref="origin/강화학습-판매", pin_commit=commit)
    (OUT / "dashboard_audit.json").write_text(
        json.dumps(audit.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = json.loads((ROOT / paths[0]).read_text(encoding="utf-8"))
    if audit.status is not GateStatus.PASS or validation["failures"]:
        raise SystemExit("Record validation failed; do not advance.")
    evidence = {"code_commit": "b1145f075f0827ce12d44039a96ad4b7b47bcdd6",
        "board_commit": commit, "selected_tests": validation["tests"],
        "artifact_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "dashboard": audit.as_dict(),
        "resolved": ["seeded initialization and zero-update checkpoint for new training",
                     "evaluation cache contract and mandatory recorded guards",
                     "monthly summary output no longer reports daily significance"]}
    report = ResearchGateReport(
        GateOutcome("performance", GateStatus.INCONCLUSIVE,
            "No new independent monthly confirmation has been run.",
            ("Code checks do not establish a cost improvement.",), evidence),
        GateOutcome("reliability", GateStatus.FAIL,
            "Code remediation passed; manuscript-to-evidence alignment remains incomplete.",
            ("The submitted PDF/source identity is unresolved (YR-298).",
             "Manuscript, figure and analysis-script corrections are drafted but not applied.",), evidence),
        GateOutcome("scenario_validity", GateStatus.INCONCLUSIVE,
            "Policy-dependent skipped demand still needs a conserved request ledger.",
            ("Existing arm totals do not establish equal completed work.",), evidence))
    result = {"schema": "yr317.v3.audit-gates.v1", "common_gates": report.as_dict()}
    (OUT / "current_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dashboard": audit.status.value,
                     "research_unresolved": report.unresolved, "claim_eligible": False}))


if __name__ == "__main__":
    main()
