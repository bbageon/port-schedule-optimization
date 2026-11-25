"""Compare raw-confirmed truck events with the same event boundaries in BP's 30-day test."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from math import isclose
from pathlib import Path
import re

from audit_visit_units import MERGE_S, gate_records, physical_visits
from gate_time_provenance import direction, parse_time
from inbound_realism_common import RAW, read, sha, stats
from inbound_realism_records import EVENTS
from recheck_transport_queue import RAW_ASSIGNMENT, inventory, save


REPORTS = Path("outputs/reports")
PREVIOUS = REPORTS / "yr332_inbound_realism/real-records.json"
BP = REPORTS / "yr332_short_window/evaluation/retrained"
OUT = REPORTS / "yr332_gate_age_curve/month-field-times.json"
REQUEST = re.compile(r"(\w+)Result REQUEST TO TSS|/api/vbs/terminal/(\w+)\s*:")


def summary(durations):
    return {**stats(durations), "over_7_min": sum(t > 7 + 1e-8 for t in durations)}


def observed():
    manifest, integrity = inventory(RAW)
    archived = read(PREVIOUS)
    known = {r["path"]: r["sha256"] for r in archived["raw_files"]}
    current = {r["path"]: r["sha256"] for r in manifest}
    if any(current.get(p) != digest for p, digest in known.items()):
        raise ValueError("Original field data is missing or changed")
    extra = sorted(current.keys() - known.keys())
    if any(p.startswith("기사/") for p in extra):
        raise ValueError("New driver records need a separate observation period")
    explicit = defaultdict(set)
    event_types = Counter()
    for path in sorted((RAW / "기사/_raw").glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf8") as f:
            for line in f:
                match = REQUEST.search(line)
                if not match:
                    continue
                event = match[1] or match[2]
                event_types[event] += 1
                if event not in EVENTS:
                    continue
                fields = dict(RAW_ASSIGNMENT.findall(line))
                stamp = parse_time(fields.get(EVENTS[event][0]))
                flow = fields.get("inOut", "").strip()
                if stamp is not None and flow in ("1", "2", "IN", "OUT", "TO", "FR"):
                    explicit[event].add((fields["conNo"].strip(), fields["truckNo"].strip(), direction(flow), stamp))

    groups = defaultdict(lambda: defaultdict(list))
    entered = []
    paired_gate_minutes = {}
    gates = Counter()
    days = set()
    for path in sorted((RAW / "기사/방문").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                flow = row["in_out"]
                if row["gate_in_time"]:
                    entered.append(row)
                key = (row["con_no"], row["truck_no"], flow)
                t = {event: parse_time(row[col]) for event, (_, col) in EVENTS.items()}
                if (*key, t["GateIn"]) not in explicit["GateIn"]:
                    continue
                gates[flow] += 1
                days.add(t["GateIn"].date())
                for name, start, end in (("gate_to_block", "GateIn", "EnterBlock"),
                                         ("block_to_done", "EnterBlock", "JobDone")):
                    if (*key, t[start]) in explicit[start] and (*key, t[end]) in explicit[end] and t[start] <= t[end]:
                        elapsed = (t[end] - t[start]).total_seconds() / 60
                        groups[flow][name].append(elapsed)
                        if name == "gate_to_block":
                            paired_gate_minutes[row["visit_id"]] = elapsed
    for flow, values in groups.items():
        previous = archived["groups"][flow]
        if gates[flow] != previous["confirmed_gate_orders"]:
            raise ValueError("Confirmed gate cohort changed")
        for key, durations in values.items():
            if stats(durations) != previous["observed_intervals"][key]:
                raise ValueError(f"Raw event pairs differ from archived {flow}/{key}")
    single_order = defaultdict(list)
    for visit in physical_visits(gate_records(entered), MERGE_S):
        if len(visit["rows"]) == 1 and not visit["superseded"]:
            row = visit["rows"][0]
            if row["visit_id"] in paired_gate_minutes:
                single_order[row["in_out"]].append(paired_gate_minutes[row["visit_id"]])
    return dict(original_files_verified=len(known), extra_files=extra, dated_gate_days=len(days),
                inventory_count=integrity["files"], raw_request_names=sorted(event_types),
                confirmed_gate_orders=dict(gates),
                gate_to_block_single_order_no_repeat={fl: summary(x) for fl, x in single_order.items()},
                groups={fl: {k: summary(x) for k, x in values.items()} for fl, values in groups.items()})


def simulated():
    result = read(BP / "result.json")
    test = read(REPORTS / "yr332_joint_turnaround/test-input.json")
    manifest = Path("build/yr332_joint_turnaround") / test["path"]
    if sha(manifest) != result["manifest_sha256"] or result["manifest_sha256"] != test["sha256"]:
        raise ValueError("Evaluation and registered test input differ")
    jobs = {r["job_id"]: r for r in read(manifest)["schedule"]}
    ledger = BP / "trucks.jsonl.gz"
    groups = defaultdict(lambda: defaultdict(list))
    with gzip.open(ledger, "rt", encoding="utf8") as f:
        for row in map(json.loads, f):
            job = jobs.pop(row["job"])
            a, b, s, c = (row[k] for k in ("gate", "block", "start", "finish"))
            if not (a <= b <= s <= c and row["flow"] == job["flow"] and a == job["arrival_s"]
                    and isclose(b - a, job["travel_s"], abs_tol=1e-6)):
                raise ValueError("Event times or input mismatch")
            group = groups[{"GATE_IN": "IN", "GATE_OUT": "OUT"}[row["flow"]]]
            for name, seconds in (("gate_to_block", b-a), ("start_to_done", c-s), ("block_to_done", c-b)):
                group[name].append(seconds / 60)
    count = sum(len(g["gate_to_block"]) for g in groups.values())
    if jobs or count != test["orders"] or count != result["windows"]["all"]["trucks"]:
        raise ValueError("Incomplete cohort")
    return dict(seed=result["seed"], days=result["days"], orders=count, manifest_sha256=test["sha256"],
                ledger_sha256=sha(ledger),
                groups={fl: {k: summary(x) for k, x in values.items()} for fl, values in groups.items()})


if __name__ == "__main__":
    save(OUT, dict(unit="minutes per external truck order", observed=observed(), simulated=simulated(),
                   percentile="p90 = sorted values at round(0.9 * (n-1)), zero based",
                   scope="Descriptive comparison of different periods and loads; no field service-start event"))
