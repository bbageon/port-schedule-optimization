"""Reconcile frozen v3 inputs without running policy simulations or training."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "outputs/reports/yr317_v3_seed_bank"
RUN = BASE / "run-3292645"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def main():
    spec = read(BASE / "seed-bank-prereg.json")
    manifest, summary = read(RUN / "manifest.json"), read(RUN / "summary.json")
    rows, artifacts = [], {}
    checks = {"fixed_twenty_seeds": spec["seeds"] == [20_000_000 + 100_000 * i for i in range(20)],
        "preregistered_contract": manifest["spec"] == spec and manifest["spec_sha256"] == sha(BASE / "seed-bank-prereg.json"),
        "clean_source_and_cpu_budget": manifest["code_dirty"] is False and manifest["cpus"] == list(range(14)),
        "existing_curve_and_load_parameters": spec["generator_contract"]["night_fraction"] == .38
            and spec["generator_contract"]["peaks"] == [[10.0, 1.5, .317], [15.0, 2.5, .633], [21.0, 1.0, .05]]
            and spec["generator_contract"]["load_weights"] == [[3500, .3, "원활"], [5000, .3, "보통"],
                [7500, .25, "혼잡시작"], [12500, .1, "혼잡"], [15000, .05, "초혼잡"]],
        "original_generator_source_matches": True,
        "disjoint_seed_families": True, "bundle_hashes_and_counts": True,
        "daily_counts_and_hourly_arrivals": True, "component_hashes": True,
        "fresh_initial_state_per_run": True, "unfiltered_twenty_months": True,
        "regeneration_and_curve_checks": summary["first_month_regeneration_matches"] and summary["all_curves_preserved"],
        "no_policy_results_claimed": summary["performance_evaluations"] == 0 and summary["new_training_runs"] == 0
            and summary["claim_eligible"] is False and summary["scenario_completion_validated"] is False}
    seen = set(spec["prior_reserved_numeric_seeds"])
    for domain in spec["seed_domains"]:
        numbers = {domain["root"], *domain["days"], *domain["background_numeric_seeds"]}
        checks["disjoint_seed_families"] &= not bool(numbers & seen)
        seen.update(numbers)
    for name, expected in spec["source_contract"].items():
        raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        checks["original_generator_source_matches"] &= hashlib.sha256(raw).hexdigest() == expected
    histogram, hourly = Counter(), [0] * 24
    for metrics in summary["rows"]:
        path = RUN / metrics["bundle"]
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        requests, days = payload["schedule"], payload["days"]
        checks["bundle_hashes_and_counts"] &= (sha(path) == metrics["bundle_sha256"]
            and digest(payload) == metrics["payload_sha256"]
            and payload["seed"] == metrics["seed"] and len(days) == 30
            and len(requests) == metrics["requested_trucks"]
            and len({r["job_id"] for r in requests}) == len(requests))
        for key in ("schedule", "initial_scenarios", "vessels"):
            checks["component_hashes"] &= digest(payload[key]) == metrics[key + "_sha256"]
        grouped = {d["index"]: [] for d in days}
        for request in requests:
            grouped[request["day"]].append(request)
        fallback = sum(r.get("fallback_reason") is not None for r in requests)
        checks["bundle_hashes_and_counts"] &= fallback == metrics["flow_fallbacks"]
        for day, detail in zip(days, metrics["days_detail"], strict=True):
            histogram[day["load"]] += 1
            daily = grouped[day["index"]]
            hours = Counter(min(23, int((r["arrival_s"] - day["t0"]) / 3600)) for r in daily)
            hours = [hours[i] for i in range(24)]
            checks["daily_counts_and_hourly_arrivals"] &= (len(daily) == day["load"] == detail["load"]
                and detail["hourly_requests"] == hours
                and digest([r["arrival_s"] for r in daily]) == detail["arrival_sha256"]
                and day["t0"] == day["index"] * 86400)
            hourly = [a + b for a, b in zip(hourly, hours)]
        initial = payload["initial_scenarios"]
        checks["fresh_initial_state_per_run"] &= (len(initial) == 21
            and all(not s["jobs"] and not s["vessels"] and s["horizon_s"] == 30 * 86400
                for s in initial.values()))
        rows.append({"seed": payload["seed"], "requests": len(requests), "flow_fallbacks": fallback,
            "loads": [d["load"] for d in days], "congested_days": sum(d["load"] >= 12500 for d in days)})
    checks["fresh_initial_state_per_run"] &= len({m["initial_scenarios_sha256"] for m in summary["rows"]}) == 20
    checks["unfiltered_twenty_months"] &= ([r["seed"] for r in rows] == spec["seeds"]
        and sum(r["requests"] for r in rows) == summary["total_requests"]
        and len({m["schedule_sha256"] for m in summary["rows"]}) == 20)
    for path in (BASE / "seed-bank-prereg.json", RUN / "manifest.json", RUN / "summary.json",
            *sorted(RUN.glob("*.summary.json"))):
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    result = {"schema": "yr317.seed-input-reconciliation.v1", "date": "2026-09-16",
        "checks": checks, "passed": all(checks.values()), "rows": rows,
        "months": len(rows), "days": sum(histogram.values()), "load_days": dict(histogram),
        "total_requests": sum(r["requests"] for r in rows), "hourly_requests": hourly,
        "flow_fallbacks": sum(r["flow_fallbacks"] for r in rows), "artifact_sha256": artifacts,
        "policy_simulations": 0, "new_training_runs": 0, "claim_eligible": False,
        "scenario_completion_validated": False,
        "initial_audit": {"record": "input-verification-initial.json",
            "issue": "The auditor expected load pairs, but the unchanged generator stores load/probability/label triples.",
            "resolution": "Correct the expected schema, preserving the initial audit and all generated inputs."}}
    (BASE / "input-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "months", "days", "total_requests", "flow_fallbacks", "load_days")}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
