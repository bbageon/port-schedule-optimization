"""Independently reconcile the completed real pilot; never mark pending arms complete."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = OUT / "run-3279b7f"


def main():
    pilot = RUN / "pilot_RL"
    result = json.loads((pilot / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    path = pilot / "requests.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    code = result["repro"]["code"]
    checks = {
        "correct_frozen_commit": code["git_head"] == "3279b7fc41b3c9ae1a5c44ed92327b6d648160c1",
        "clean_execution_code": code["git_dirty"] is False,
        "ledger_hash": hashlib.sha256(path.read_bytes()).hexdigest() == result["request_ledger_sha256"],
        "requested_count": len(rows) == result["request_summary"]["requested"] == 300,
        "unique_requests": len({r["job_id"] for r in rows}) == 300,
        "one_admission_event_each": all(len(r["admission_events"]) == 1 for r in rows),
        "all_completed": all(r["state"] == "COMPLETED" for r in rows),
        "cost_reconciles": math.isclose(sum(r["accounted_wait_krw"] for r in rows),
            sum(d["c_wait"] for d in result["days"]), rel_tol=1e-10, abs_tol=1e-4),
        "unchanged_checkpoint": hashlib.sha256((ROOT / "outputs/v3/month-02/ckpt_029.pt").read_bytes()).hexdigest()
            == result["repro"]["checkpoint_sha256"],
        "no_training": manifest["contract"]["settings"]["labels_per_day"] is None,
        "no_teacher": result["rollout_calls"] == 0,
        "no_policy_exceptions": result["policy_exceptions"] == 0,
        "no_performance_claim": result["claim_eligible"] is False,
    }
    tests = ET.parse(OUT / "tests.xml").getroot().find("testsuite")
    checks["regression_tests"] = tests.get("tests") == "59" and tests.get("failures") == tests.get("errors") == "0"
    payload = {"scope": "completed one-day RL recording pilot only", "checks": checks,
        "passed": all(checks.values()), "elapsed_s": result["elapsed_s"],
        "completed_trucks": 300, "independent_confirmatory_runs": 0,
        "main_run": "three 10-day diagnostic replays; consult run progress for completion"}
    (OUT / "pilot_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(not payload["passed"])


if __name__ == "__main__":
    main()
