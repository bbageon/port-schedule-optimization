"""Read-only provenance audit for YR-332-i/f; outputs contain no order IDs.

This checks data transmission and timestamp intervals, not the physical meaning
of the fields. It does not calibrate a simulator or train/evaluate a policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


BASE = Path(__file__).resolve().parents[2]
DEFAULT_OUT = BASE / "outputs/reports/yr332_transport_queue_recheck"
TIME_FIELDS = {
    "GateIn": "gateInTime", "EnterBlock": "blockInTime",
    "JobDone": "jobDoneTime", "ChangeConLoc": "conLocChangedTime",
}
RAW_ASSIGNMENT = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,}\]]*)")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def stamp(value: str | None) -> datetime | None:
    if not value or value.strip() in ("null", "None"):
        return None
    value = value.strip()
    if re.fullmatch(r"\d{14}", value):
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    return datetime.fromisoformat(value)


def direction(value: str) -> str:
    return {"1": "OUT", "FR": "OUT", "2": "IN", "TO": "IN"}.get(value, value)


def summary(values) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    return dict(n=len(values), min=values[0], median=statistics.median(values),
                p90_nearest=values[round((len(values)-1)*.9)], max=values[-1],
                mean=statistics.mean(values), negative=sum(v < 0 for v in values),
                zero=sum(v == 0 for v in values))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def inventory(root: Path):
    manifest = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
                for p in sorted(root.rglob("*")) if p.is_file()]
    previous = {}
    old_dir = BASE / "outputs/reports/yr332_real_order_information"
    for name in ("source-manifest.json", "original-source-manifest.json"):
        for entry in json.loads((old_dir / name).read_text(encoding="utf-8-sig")):
            key = entry["path"]
            if key in previous:
                assert previous[key] == entry["sha256"], "conflicting earlier manifests"
            previous[key] = entry["sha256"]
    now = {m["path"]: m["sha256"] for m in manifest}
    comparison = dict(files=len(manifest), bytes=sum(m["bytes"] for m in manifest),
        extensions=dict(Counter(Path(m["path"]).suffix for m in manifest)),
        unchanged=sum(previous.get(k) == v for k, v in now.items()),
        changed=sorted(k for k, v in now.items() if k in previous and previous[k] != v),
        new=sorted(set(now)-set(previous)), missing=sorted(set(previous)-set(now)))
    return manifest, comparison


def queue_audit(root: Path):
    raw = defaultdict(Counter)
    raw_observations = defaultdict(list)
    explicit_gate_times = set()
    raw_classes, raw_fields, raw_lags = Counter(), Counter(), defaultdict(list)
    raw_line_count = queue_line_count = unknown_class = missing_key = missing_time = 0
    raw_log_time_index = set()
    for path in sorted((root / "기사/_raw").glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw_line_count += 1
                fields = dict(RAW_ASSIGNMENT.findall(line))
                raw_fields.update(fields.keys())
                if "GateInResult REQUEST TO TSS" in line or "/api/vbs/terminal/GateIn :" in line:
                    gate_time = stamp(fields.get("gateInTime"))
                    if gate_time:
                        explicit_gate_times.add(("GateIn", fields.get("conNo", "").strip(),
                            fields.get("truckNo", "").strip(), gate_time,
                            direction(fields.get("inOut", "").strip())))
                value = fields.get("blcWaitingCarCount", "").strip()
                if not value or value == "null":
                    continue
                queue_line_count += 1
                tss = re.search(r"(\w+)Result REQUEST TO TSS", line)
                vbs = re.search(r"/api/vbs/terminal/(\w+)\s*:", line)
                if tss:
                    kind, source = tss[1], "TSS"
                elif vbs:
                    kind, source = vbs[1], "VBS"
                else:
                    unknown_class += 1
                    continue
                raw_classes[f"{source}:{kind}"] += 1
                time = stamp(fields.get(TIME_FIELDS.get(kind, "")))
                container, truck = fields.get("conNo", "").strip(), fields.get("truckNo", "").strip()
                if not container or not truck:
                    missing_key += 1
                    continue
                log_time = stamp(line[:19])
                imputed = time is None
                missing_time += imputed
                # Keep this explicit: matching a CSV log-time substitution does
                # not establish the physical arrival time.
                time = time or log_time
                key = (kind, container, truck, time, direction(fields.get("inOut", "").strip()))
                value = int(value)
                raw[key][value] += 1
                raw_log_time_index.add((*key, value, log_time))
                raw_observations[(key, value)].append((log_time, imputed))
                if not imputed:
                    raw_lags[f"{source}:{kind}"].append((log_time-time).total_seconds())
    counts, observed_values, csv_lags = defaultdict(Counter), defaultdict(list), defaultdict(list)
    event_index, event_queues = defaultdict(set), defaultdict(set)
    headers = set()
    fallback_keys = set()
    unconfirmed_gate_keys = set()
    unconfirmed_gate_flags = Counter()
    merged_offsets = defaultdict(list)
    fallback_flags = Counter()
    queue_rows = matched_rows = same_log_rows = 0
    for path in sorted((root / "기사/이벤트").glob("*.csv")):
        for row in rows(path):
            headers.update(row)
            kind = row["event"]
            count = counts[kind]
            count["rows"] += 1
            key = (kind, row["con_no"], row["truck_no"], stamp(row["event_time"]), row["in_out"])
            event_index[kind].add(key)
            if kind == "GateIn" and key not in explicit_gate_times:
                unconfirmed_gate_keys.add(key)
                unconfirmed_gate_flags["error_true"] += row["error"].lower() == "true"
                unconfirmed_gate_flags["error_false"] += row["error"].lower() == "false"
            if not row["block_waiting_cars"]:
                count["queue_missing"] += 1
                continue
            value = int(row["block_waiting_cars"])
            queue_rows += 1
            count["queue_present"] += 1
            matched = value in raw.get(key, {})
            matched_rows += matched
            count["raw_same_key_and_value"] += matched
            same_log = (*key, value, stamp(row["log_time"])) in raw_log_time_index
            same_log_rows += same_log
            count["raw_same_log_time"] += same_log
            observations = raw_observations.get((key, value), [])
            if observations and all(imputed for _, imputed in observations):
                fallback_keys.add(key)
                count["requires_log_time_as_event_time"] += 1
                fallback_flags["error_true"] += row["error"].lower() == "true"
                fallback_flags["error_false"] += row["error"].lower() == "false"
                fallback_flags["error_message_present"] += bool(row["error_message"])
                fallback_flags["explicit_gate_time_elsewhere_in_request_logs"] += key in explicit_gate_times
            if observations and not same_log:
                merged_offsets[kind].append(min((lt-stamp(row["log_time"])).total_seconds() for lt, _ in observations))
            observed_values[kind].append(value)
            event_queues[key].add(value)
            csv_lags[kind].append((stamp(row["log_time"])-key[3]).total_seconds())
    visit = Counter()
    visit_headers = set()
    for path in sorted((root / "기사/방문").glob("*.csv")):
        for row in rows(path):
            visit["rows"] += 1
            visit_headers.update(row)
            if not row["gate_in_time"]:
                continue
            visit["gate_in_present"] += 1
            key = ("GateIn", row["con_no"], row["truck_no"], stamp(row["gate_in_time"]), row["in_out"])
            visit["same_event_key"] += key in event_index["GateIn"]
            if row["gate_in_block_waiting_cars"]:
                visit["queue_present"] += 1
                visit["same_event_queue"] += int(row["gate_in_block_waiting_cars"]) in event_queues.get(key, set())
    checks = dict(raw_classes_complete=unknown_class == 0, raw_join_fields_complete=missing_key == 0,
        all_event_queue_values_trace_to_raw=matched_rows == queue_rows,
        all_visit_gate_events_match=visit["gate_in_present"] == visit["same_event_key"],
        all_visit_queue_values_match=visit["queue_present"] == visit["same_event_queue"])
    return dict(checks=checks, raw_lines=raw_line_count, raw_queue_lines=queue_line_count,
        raw_classes=dict(raw_classes), unknown_raw_classes=unknown_class, missing_raw_join_keys=missing_key,
        raw_queue_records_missing_event_time=missing_time,
        raw_queue_unique_keys=len(raw), raw_keys_with_multiple_queue_values=sum(len(v) > 1 for v in raw.values()),
        event_rows=sum(c["rows"] for c in counts.values()), event_queue_rows=queue_rows,
        matched_event_queue_rows=matched_rows, event_counts={k:dict(v) for k,v in counts.items()},
        queue_values_by_event={k:summary(v) for k,v in observed_values.items()},
        raw_log_minus_event_seconds={k:summary(v) for k,v in raw_lags.items()},
        csv_log_minus_event_seconds={k:summary(v) for k,v in csv_lags.items()},
        queue_source_log_minus_csv_log_seconds={k:summary(v) for k,v in merged_offsets.items()},
        data_quality=dict(event_times_require_log_substitution=len(fallback_keys),
            event_times_without_explicit_gate_request=len(fallback_keys-explicit_gate_times),
            all_gate_event_times_without_explicit_request=len(unconfirmed_gate_keys),
            unconfirmed_gate_flags=dict(unconfirmed_gate_flags),
            substituted_event_flags=dict(fallback_flags),
            csv_log_time_not_same_as_queue_source=queue_rows-same_log_rows,
            interpretation="Values trace; substituted event times and merged record clocks are not physical-time validation."),
        visit_gate_reconciliation=dict(visit), event_columns=sorted(headers), visit_columns=sorted(visit_headers),
        raw_assignment_fields=dict(sorted(raw_fields.items())),
        limits=["Key joins use container/truck IDs only in memory; no identifiers are published.",
                "Raw TSS and VBS messages may duplicate one event: this is an existence/value check, not a bijection.",
                "Log time minus event time measures record timing, not necessarily queue measurement age.",
                "No service-start event or authoritative counting-population definition was supplied.",
                "Physical trucks, visits, orders, in-service and in-transit populations cannot be equated from the field name."]), unconfirmed_gate_keys


def calibration_overlap(root, fallback_keys):
    """Recompute the already-frozen selection, without rewriting its profile."""
    from calibrate_order_information import stamp as booking_stamp
    from audit_visit_units import gate_records, physical_visits, MERGE_S
    by_doc = defaultdict(list)
    for path in sorted((root / "기사/오더").glob("*.csv")):
        for row in rows(path):
            notice, reserved = booking_stamp(row["log_time"]), booking_stamp(row["in_out_reserve_time"])
            if row["copino_err_type"] == "NM" and notice and reserved and reserved.year >= 2000:
                by_doc[row["doc_key"]].append((notice, reserved, row["in_out"]))
    entered, cohorts, affected, quality = [], defaultdict(list), defaultdict(list), Counter()
    def key(row):
        return ("GateIn", row["con_no"], row["truck_no"], stamp(row["gate_in_time"]), row["in_out"])
    for path in sorted((root / "기사/방문").glob("*.csv")):
        for row in rows(path):
            arrival = booking_stamp(row["gate_in_time"])
            if arrival is None:
                continue
            entered.append(row)
            uncertain = key(row) in fallback_keys
            if uncertain:
                quality["affected_visit_rows"] += 1
                quality["with_later_block_entry"] += bool(row["enter_block_time"])
                quality["with_later_job_completion"] += bool(row["job_done_time"])
                quality["with_later_gate_exit"] += bool(row["gate_out_time"])
            first = booking_stamp(row["copino_first_time"])
            candidates = sorted({item for doc in re.split(r"[|;,]", row["doc_keys"])
                for item in by_doc.get(doc.strip(), []) if item[0] <= arrival and item[2] == row["in_out"]})
            if not candidates or first is None or candidates[0][0] != first:
                continue
            notice, reserved, flow = candidates[0]
            group = ("calibration" if arrival.date().isoformat() <= "2026-11-05" else "holdout", flow)
            pair = (int((arrival-reserved).total_seconds()), int((arrival-notice).total_seconds()))
            cohorts[group].append(pair)
            if uncertain:
                affected[group].append(pair)
    profile_path = BASE/"outputs/reports/yr332_real_order_information/profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    matched = {}
    for flow, profile_key in (("IN", "GATE_IN"), ("OUT", "GATE_OUT")):
        observed = Counter(cohorts[("calibration", flow)])
        saved = Counter({(e, lead): n for e, lead, n in profile["flows"][profile_key]["samples"]})
        matched[flow] = observed == saved
    visits = physical_visits(gate_records(entered), MERGE_S)
    quality["physical_visits_using_existing_merge"] = len(visits)
    quality["affected_rows_in_superseded_gate_records"] = sum(key(r) in fallback_keys
        for v in visits for _, rs in v["superseded"] for r in rs)
    quality["affected_rows_retained_as_physical_visit"] = sum(key(r) in fallback_keys for v in visits for r in v["rows"])
    quality["physical_visits_retaining_affected_rows"] = sum(any(key(r) in fallback_keys for r in v["rows"]) for v in visits)
    return dict(checks=dict(saved_calibration_histograms_reproduced=all(matched.values())),
        saved_profile_sha256=digest(profile_path), selection_matches_saved_by_flow=matched,
        affected_definition="No TSS/VBS GateIn request supplies the CSV physical timestamp; measured across all GateIn events, including events without a queue value.",
        cohorts={f"{split}_{flow}":dict(n=len(pairs), affected=len(affected[(split,flow)]),
            eta_median_seconds=statistics.median(p[0] for p in pairs),
            sensitivity_excluding_unconfirmed=dict(
                n=len(pairs)-len(affected[(split,flow)]),
                eta_median_seconds=statistics.median(p[0] for p in (Counter(pairs)-Counter(affected[(split,flow)])).elements()),
                notice_lead_median_seconds=statistics.median(p[1] for p in (Counter(pairs)-Counter(affected[(split,flow)])).elements())))
            for (split,flow),pairs in cohorts.items()},
        visit_implications=dict(quality), existing_merge_seconds=MERGE_S,
        limits=["No sample or parameter was removed/replaced in the saved calibration.",
                "Error flags and absent subsequent events require source confirmation before interpreting a row as an actual entry.",
                "Physical visit grouping is the existing 30-minute rule, not independent confirmation of entry."])


def vessel_audit(root: Path):
    fields = ("loadDiscGbn", "conNo", "atw", "firstYardInTime", "lastYardInTime")
    counts, gaps, precision = defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    pairs = matched = total = 0
    for path in sorted((root / "선박/양적하목록").glob("*.csv")):
        converted = list(rows(path))
        original = json.loads(path.with_suffix(".json").read_text(encoding="utf-8-sig"))["response"]["data"]["conList"]
        def key(row):
            return tuple(str(row.get(f) or "") for f in fields)
        pairs += 1
        matched += Counter(map(key, converted)) == Counter(map(key, original))
        total += len(converted)
        for row in converted:
            flow = row["loadDiscGbn"]
            count = counts[flow]
            count["rows"] += 1
            times = {f:stamp(row.get(f)) for f in fields[2:]}
            for field, time in times.items():
                if time:
                    precision[field]["present"] += 1
                    precision[field]["nonzero_seconds"] += bool(time.second or time.microsecond)
            atw = times["atw"]
            count["atw_present"] += atw is not None
            other = times["firstYardInTime" if flow == "D" else "lastYardInTime"]
            if atw and other:
                seconds = (other-atw).total_seconds() if flow == "D" else (atw-other).total_seconds()
                gaps[flow].append(seconds/60)
    return dict(checks=dict(all_csv_json_pairs_equal=pairs == matched), csv_json_pairs=pairs,
        matched_pairs=matched, rows=total, compared_fields=list(fields),
        by_flow={k:dict(counts=dict(v), gap_minutes=summary(gaps[k]),
                        gap_definition="firstYardInTime - atw" if k == "D" else "atw - lastYardInTime")
                 for k,v in counts.items()}, timestamp_precision={k:dict(v) for k,v in precision.items()},
        limits=["Exact CSV/JSON agreement establishes transformation integrity, not event semantics.",
                "Gaps may include waiting, travel and handling; they are not measured 470-second cycles.",
                "Final snapshots do not establish when the operational planner first knew these times."])


def run(root: Path, out: Path):
    if root.resolve() == out.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError("Output must not be inside the original data directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest, comparison = inventory(root)
    queue, fallback_keys = queue_audit(root)
    vessel = vessel_audit(root)
    overlap = calibration_overlap(root, fallback_keys)
    checks = {"original_600_files_unchanged": comparison["files"] == 600 and comparison["unchanged"] == 600
              and not (comparison["new"] or comparison["changed"] or comparison["missing"]),
              **queue["checks"], **vessel["checks"], **overlap["checks"]}
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE, text=True).strip()
    runtime_paths = ["src/yard_rl/v7/world/integrated/transfer.py", "src/yard_rl/v7/world/integrated/profiles.py",
                     "src/yard_rl/v7/world/sim/kpis.py", "src/yard_rl/v7/features/block.py"]
    result = dict(schema="yard_rl.transport_queue_recheck.v1", kind="read_only_data_audit",
        base_commit=code, script_sha256=digest(Path(__file__)), checks=checks,
        status="TRACEABLE_WITH_TIME_LIMITATIONS" if all(checks.values()) else "FAIL", stochastic_execution=False,
        runtime_file_sha256={p:digest(BASE/p) for p in runtime_paths},
        helper_sha256={p:digest(BASE/p) for p in ("scripts/v7/calibrate_order_information.py", "scripts/v7/audit_visit_units.py")},
        interpretation="Provenance checked; substituted timestamps, counting definitions and transport realism remain unresolved.")
    for name, data in (("source-manifest.json",manifest), ("source-comparison.json",comparison),
                       ("queue-reconciliation.json",queue), ("vessel-reconciliation.json",vessel),
                       ("calibration-overlap.json",overlap),
                       ("audit.json",result)):
        save(out/name, data)
    print(json.dumps(dict(status=result["status"], checks=checks, files=comparison["files"],
        queue_events=queue["event_queue_rows"], queue_matched=queue["matched_event_queue_rows"],
        vessel_rows=vessel["rows"]), ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE/"오더데이터")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.root, args.out)
    raise SystemExit(2 if result["status"] == "FAIL" else 0)
