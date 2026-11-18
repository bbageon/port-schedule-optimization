"""Independent counts, raw gate timestamp check and exact audit replay."""
from collections import Counter
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "outputs/reports/yr332_transport_queue_recheck"
DATA = BASE / "오더데이터"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def field(line, name):
    match = re.search(r"\b" + name + r"\s*=\s*([^,}\]]*)", line)
    return match[1].strip() if match else ""


def time(value):
    if not value:
        return None
    if len(value) == 14 and value.isdigit():
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    return datetime.fromisoformat(value)


def run():
    audit, queue, overlap = read("audit.json"), read("queue-reconciliation.json"), read("calibration-overlap.json")
    explicit, raw_queue_rows = set(), 0
    for path in sorted((DATA / "기사/_raw").glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                match = re.search(r"\bblcWaitingCarCount\s*=\s*([0-9]+)(?:,|})", line)
                raw_queue_rows += bool(match)
                if not re.search(r"GateInResult REQUEST TO TSS|/api/vbs/terminal/GateIn\s*:", line):
                    continue
                at = time(field(line, "gateInTime"))
                if at:
                    di = field(line, "inOut")
                    di = "IN" if di in ("2", "TO") else "OUT" if di in ("1", "FR") else di
                    explicit.add((field(line, "conNo"), field(line, "truckNo"), at, di))
    event_counts, queue_counts, unconfirmed = Counter(), Counter(), set()
    for path in sorted((DATA / "기사/이벤트").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                event_counts[row["event"]] += 1
                queue_counts[row["event"]] += bool(row["block_waiting_cars"])
                if row["event"] == "GateIn":
                    key = (row["con_no"], row["truck_no"], time(row["event_time"]), row["in_out"])
                    if key not in explicit:
                        unconfirmed.add(key)
    visit_unconfirmed, no_followup = 0, 0
    for path in sorted((DATA / "기사/방문").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["con_no"], row["truck_no"], time(row["gate_in_time"]), row["in_out"])
                if key in unconfirmed:
                    visit_unconfirmed += 1
                    no_followup += not any(row[f] for f in ("enter_block_time", "job_done_time", "gate_out_time"))
    filenames = ("source-manifest.json", "source-comparison.json", "queue-reconciliation.json",
                 "vessel-reconciliation.json", "calibration-overlap.json", "audit.json")
    replay = {name:sha(OUT/name) == sha(OUT/"replay"/name) for name in filenames}
    manifest = read("source-manifest.json")
    sources = all(sha(DATA/item["path"]) == item["sha256"] for item in manifest)
    runtime = all(sha(BASE/path) == h for path,h in audit["runtime_file_sha256"].items())
    helpers = all(sha(BASE/path) == h for path,h in audit["helper_sha256"].items())
    original_profile = BASE/"outputs/reports/yr332_real_order_information/profile.json"
    diff = subprocess.run(["git", "diff", "--quiet", audit["base_commit"], "--", "src",
        "outputs/reports/yr332_real_order_information/profile.json"], cwd=BASE)
    claims = dict(
        gate_events=event_counts["GateIn"], queue_values=sum(queue_counts.values()),
        raw_queue_records=raw_queue_rows, unconfirmed_gate_timestamps=len(unconfirmed),
        affected_visit_rows=visit_unconfirmed, affected_rows_without_followup=no_followup,
        affected_calibration=sum(v["affected"] for k,v in overlap["cohorts"].items() if k.startswith("calibration_")),
        affected_holdout=sum(v["affected"] for k,v in overlap["cohorts"].items() if k.startswith("holdout_")),
        retained_uncertain_visits=overlap["visit_implications"]["physical_visits_retaining_affected_rows"])
    expected = dict(gate_events=34304, queue_values=48620, raw_queue_records=54245,
        unconfirmed_gate_timestamps=601, affected_visit_rows=601, affected_rows_without_followup=601,
        affected_calibration=344, affected_holdout=143, retained_uncertain_visits=23)
    checks = dict(all_audit_checks=all(audit["checks"].values()), six_replay_files_exact=all(replay.values()),
        original_files_unchanged=sources and len(manifest) == 600,
        raw_gate_timestamp_check_independent=len(unconfirmed) == queue["data_quality"]["all_gate_event_times_without_explicit_request"],
        event_queue_counts_independent=all(queue_counts[k] == v.get("queue_present",0) for k,v in queue["event_counts"].items()),
        report_claim_counts=claims == expected, runtime_and_helper_hashes_unchanged=runtime and helpers,
        simulator_and_saved_profile_unmodified=diff.returncode == 0,
        saved_profile_hash_unchanged=sha(original_profile) == overlap["saved_profile_sha256"],
        final_audit_script_hash=sha(BASE/"scripts/v7/recheck_transport_queue.py") == audit["script_sha256"])
    result = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, claims=claims,
        replay_file_sha256={n:sha(OUT/n) for n in filenames}, replay_matches=replay,
        verification_script_sha256=sha(Path(__file__)),
        limitations=["PASS verifies the audit and its reported limitations, not physical gate-entry truth or simulator realism.",
                     "No PPO or scheduling performance comparison was performed."])
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return all(checks.values())


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
