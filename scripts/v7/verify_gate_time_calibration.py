"""Independent source join and original-calibrator replay for YR-332-ba."""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parents[2]
DATA = BASE/"오더데이터"
OUT = BASE/"outputs/reports/yr332_gate_time_provenance"
PRIVATE = BASE/"outputs/private/yr332_gate_time_provenance"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def time(value):
    if not value or value in ("null", "None"):
        return None
    return datetime.strptime(value, "%Y%m%d%H%M%S") if len(value) == 14 and value.isdigit() else datetime.fromisoformat(value)


def field(line, name):
    match = re.search(r"\b"+name+r"\s*=\s*([^,}\]]*)", line)
    return match[1].strip() if match else ""


def source_keys():
    explicit, log_only = set(), set()
    for path in sorted((DATA/"기사/_raw").glob("*.gz")):
        with gzip.open(path,"rt",encoding="utf-8") as handle:
            for line in handle:
                if not re.search(r"GateInResult REQUEST TO TSS|/api/vbs/terminal/GateIn\s*:",line):
                    continue
                flow = field(line,"inOut")
                flow = {"2":"IN","TO":"IN","1":"OUT","FR":"OUT"}.get(flow,flow)
                at = time(field(line,"gateInTime"))
                key = (field(line,"conNo"),field(line,"truckNo"),at or time(line[:19]),flow)
                (explicit if at else log_only).add(key)
    return explicit,log_only


def row_key(row):
    return row["con_no"],row["truck_no"],time(row["gate_in_time"]),row["in_out"]


def run():
    explicit,log_only = source_keys()
    visits = []
    for path in sorted((DATA/"기사/방문").glob("*.csv")):
        with path.open(encoding="utf-8-sig",newline="") as handle:
            visits.extend(csv.DictReader(handle))
    counts,uncertain = Counter(),{}
    for row in visits:
        key = row_key(row)
        if key[2] is None:
            status = "no_gate_time"
        elif key in explicit:
            status = "confirmed_other_message" if key in log_only else "confirmed_explicit"
        else:
            status = "unconfirmed_log_only" if key in log_only else "unconfirmed_no_request_match"
            uncertain[row["visit_id"]] = dict(status=status,source_row=row)
        counts[status] += 1

    # Reuse the ORIGINAL booking selector, with an independently constructed
    # gate-time inclusion set. Do not call the new pair/profile builder.
    import calibrate_order_information as legacy
    original_read,original_out = legacy.read,legacy.OUT
    independent_out = PRIVATE/"independent-original-calibrator"
    if (independent_out/"profile.json").exists():
        raise FileExistsError("Independent verification output already exists")

    def confirmed_read(paths):
        paths = list(paths)
        rows = original_read(paths)
        return [r for r in rows if row_key(r) in explicit] if paths and paths[0].parent.name == "방문" else rows

    try:
        legacy.read,legacy.OUT = confirmed_read,independent_out
        with redirect_stdout(io.StringIO()):
            legacy.audit(DATA)
    finally:
        legacy.read,legacy.OUT = original_read,original_out

    independent = read(independent_out/"profile.json")
    target = read(OUT/"profile-confirmed.json")
    summary = read(OUT/"provenance-summary.json")
    comparison = read(OUT/"calibration-comparison.json")
    impact = read(OUT/"arrival-impacts.json")
    inputs = read(OUT/"input-check.json")
    build = read(OUT/"build-record.json")
    quarantine = PRIVATE/"unconfirmed-visits.jsonl"
    private_rows = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    private_map = {r["source_row"]["visit_id"]:r for r in private_rows}
    manifest = read(OUT/"source-manifest.json")
    replay_names = ("profile-confirmed.json","calibration-comparison.json","arrival-impacts.json",
                    "source-manifest.json","visit-arrival-templates-confirmed.json","visit-patterns-confirmed.json","input-bundle.json")
    replay = {name:sha(OUT/name) == sha(OUT/"replay"/name) for name in replay_names}
    independent_audit = read(independent_out/"audit.json")
    cohort_counts = {k:v["error_s"]["n"] for k,v in independent_audit["cohorts"].items()}
    public_counts = {k:v["confirmed"]["n"] for k,v in comparison["cohorts"].items()}
    old_profile = BASE/"outputs/reports/yr332_real_order_information/profile.json"
    frozen = read(BASE/"outputs/reports/yr332_transport_queue_recheck/audit.json")
    runtime_unchanged = all(sha(BASE/p) == h for p,h in frozen["runtime_file_sha256"].items())
    ignored = subprocess.run(["git","check-ignore","--quiet",str(quarantine)],cwd=BASE).returncode == 0
    tracked = subprocess.run(["git","ls-files","--",str(PRIVATE)],cwd=BASE,capture_output=True,text=True).stdout.strip()
    checks = dict(
        raw_request_join_independently_matches=all(counts[k] == n for k,n in summary["status_counts"].items()),
        independent_booking_histogram_exact=independent["flows"] == target["flows"],
        independent_split_counts=cohort_counts == public_counts,
        fixed_date_split=target["calibration_dates"] == ["2026-10-28","2026-11-05"] and target["holdout_dates"] == ["2026-11-06","2026-11-09"],
        no_holdout_fitting=not comparison["heldout_used_in_fit"],
        quarantine_exact_rows=private_map == uncertain and len(private_rows) == len(uncertain),
        quarantine_hash=sha(quarantine) == summary["private_quarantine"]["sha256"],
        private_original_rows_ignored_and_untracked=ignored and not tracked,
        original_600_files_unchanged=len(manifest) == 600 and all(sha(DATA/m["path"]) == m["sha256"] for m in manifest),
        original_timing_profile_unchanged=sha(old_profile) == target["source_event_contract"]["legacy_profile_sha256"] == build["old_profile_sha256"],
        original_runtime_unchanged=runtime_unchanged,
        new_profile_hash=sha(OUT/"profile-confirmed.json") == build["new_profile_sha256"],
        build_source_hashes=all(sha(BASE/p) == h for p,h in build["source_sha256"].items()),
        build_and_inputs_clean=not build["repro"]["code"]["git_dirty"] and not inputs["repro"]["code"]["git_dirty"],
        exact_calibration_replay=all(replay.values()),
        original_demand_calibrations_reproduced=all(impact["checks"].values()),
        input_cases_complete=inputs["status"] == "PASS" and len(inputs["cases"]) == 18 and len(inputs["pairs"]) == 9,
        physical_inputs_preserved=all(all(p["unchanged_physics"].values()) for p in inputs["pairs"]),
        only_information_fields_changed=all(set(p["changed_fields"]) <= {"reserved_s","lead_s","eta_offset_s"} for p in inputs["pairs"]),
        input_replays_exact=len(inputs["repeats"]) == 2 and all(r["exact"] for r in inputs["repeats"]),
        no_policy_training_or_evaluation=not inputs["configuration"]["policy_execution"] and not build["configuration"]["performance_evaluation"],
    )
    result = dict(status="PASS" if all(checks.values()) else "FAIL",checks=checks,
        independent_counts=dict(counts),independent_cohort_counts=cohort_counts,replay_matches=replay,
        independent_profile_sha256=sha(independent_out/"profile.json"),verification_script_sha256=sha(Path(__file__)),
        limitations=["Timestamp provenance does not establish physical truth or supplier error semantics.",
                     "Input correctness is not scheduling performance or complete simulator realism."])
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return all(checks.values())


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
