"""Read-only comparison of the fixed three-day smoke against its original run."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(old, new, evidence):
    before, after = read(old / "report.json"), read(new / "report.json")
    days, admissions = read(new / "days.json"), read(new / "admissions.json")
    manifests = [read(p / "manifest.json") for p in (old, new)]
    keys = ("roles", "crane_actions", "cost_krw", "team_reward", "learning_reward",
            "learning_intervals", "updates", "admitted", "skipped", "traded_edges")
    identical = {key: before[key] == after[key] for key in keys}
    checkpoint_equal = {}
    for name in ("initial.pt", "day_01.pt", "day_02.pt", "day_03.pt", "final.pt"):
        a, b = [torch.load(p / name, map_location="cpu", weights_only=True) for p in (old, new)]
        checkpoint_equal[name] = (all(torch.equal(a["policy"][k], b["policy"][k]) for k in a["policy"])
                                  and torch.equal(a["action_rng"], b["action_rng"]))
    config_keys = ("seed", "days", "ppo", "learning_window_s", "initial_occupancy", "action_mode")
    settings_equal = {k: manifests[0][k] == manifests[1][k] for k in config_keys}
    tests = {}
    for name in ("unit-tests.xml", "continuous-tests.xml"):
        suites = list(ET.parse(evidence / name).getroot().iter("testsuite"))
        tests[name] = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                       for key in ("tests", "failures", "errors", "skipped")}
    checks = {
        "normal_results_unchanged": all(identical.values()),
        "policy_and_action_rng_unchanged": all(checkpoint_equal.values()),
        "input_and_learning_settings_unchanged": all(settings_equal.values()),
        "clean_pinned_source": after["code"]["git_dirty"] is False,
        "completed": read(new / "status.json")["state"] == after["state"] == "completed",
        "updates_0_24_0": [d["updates"] for d in days] == [0, 24, 0],
        "no_counterfactuals": after["counterfactual_worlds"] == 0,
        "all_180_trucks_admitted": admissions["admitted"] == 180 and admissions["skipped"] == 0,
        "no_vessel_failures": admissions["vessel_failed"] == 0,
        "physical_invariants": all(d["physical_invariants_checked"] for d in days),
        "live_cohorts_saved": len(read(new / "cohort_live.json")) == 3,
        "tests_passed": all(r["failures"] == r["errors"] == r["skipped"] == 0 for r in tests.values()),
        "all_tests_present": sum(r["tests"] for r in tests.values()) == 157}
    result = {"claim_scope": "NO_PERFORMANCE_CLAIM", "checks": checks,
              "code": after["code"], "seed": manifests[1]["seed"], "ppo": manifests[1]["ppo"],
              "old_run": str(old), "new_run": str(new), "tests": tests,
              "identical_report_fields": identical, "identical_checkpoints": checkpoint_equal,
              "identical_settings": settings_equal, "updates_by_day": [d["updates"] for d in days],
              "admitted": admissions["admitted"], "vessel_admissions": len(admissions["vessels"]),
              "cost_krw": after["cost_krw"], "roles": after["roles"],
              "original_30_day_failure_resolved": False,
              "artifacts": {str(p): sha(p) for base in (old, new)
                            for p in sorted(base.iterdir()) if p.is_file() and p.suffix in (".json", ".pt")}}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not all(checks.values()):
        raise SystemExit("Debug verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    audit(args.old, args.new, args.evidence)
