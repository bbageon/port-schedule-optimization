"""Reissue unresolved v3 gates from the committed read-only audit; no simulation."""
import hashlib
import json
import subprocess
from pathlib import Path

from yard_rl.experiments.gate_harness import GateOutcome, GateStatus, ResearchGateReport

ROOT = Path(__file__).resolve().parents[3]
AUDIT = "outputs/reports/yr317_v3_review_strategy/refinement-audit.json"


def main():
    evidence = {
        "audit_path": AUDIT,
        "audit_sha256": hashlib.sha256((ROOT / AUDIT).read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "v3 manuscript review; read-only audit; no new performance experiment",
    }
    report = ResearchGateReport(
        performance=GateOutcome("performance", GateStatus.INCONCLUSIVE,
            "Independent monthly replication has not been established.",
            ("Daily observations within one continuous run are dependent.",), evidence),
        reliability=GateOutcome("reliability", GateStatus.FAIL,
            "The initialization claim and result reuse contract do not match v3 code.",
            ("ckpt_000 follows day-zero fitting; true initialization was not saved.",
             "Cached arm files do not validate seed, configuration or network weights.",
             "Missing legacy guard fields default to zero.",
             "The submitted PDF/source identity is unresolved (YR-298)."), evidence),
        scenario_validity=GateOutcome("scenario_validity", GateStatus.INCONCLUSIVE,
            "Existing truck admission accounting needs verification.",
            ("Skipped demand differs across policies; per-request reasons are absent from arm files.",
             "No claim of real-terminal feasibility is authorized by these synthetic results."), evidence),
    )
    output = Path(__file__).with_name("bootstrap_gate.json")
    output.write_text(json.dumps({"schema": "yr317.v3.audit-gates.v1",
        "common_gates": report.as_dict()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output.relative_to(ROOT)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "unresolved": report.unresolved}))


if __name__ == "__main__":
    main()
