"""Reconstruct block visits from explicit driver events. Publish aggregate counts only."""
import csv
import gzip
from collections import Counter, defaultdict
from datetime import datetime
from itertools import groupby, pairwise
from math import isclose
from pathlib import Path

from audit_visit_units import BLOCK, MERGE_S, gate_records, physical_visits
from gate_time_provenance import direction, parse_time
from inbound_realism_common import RAW, read, sha, stats
from recheck_transport_queue import RAW_ASSIGNMENT, inventory, save
from verify_month_field_times import BP, EVENTS, OUT as BASELINE, PREVIOUS, REPORTS, REQUEST, summary


def block(location):
    match = BLOCK.match(location)
    return match[0] if match else ""


def minutes(end, start):
    return (end - start).total_seconds() / 60


def source_events():
    manifest, _ = inventory(RAW)
    expected = {r["path"]: r["sha256"] for r in read(PREVIOUS)["raw_files"]}
    current = {r["path"]: r["sha256"] for r in manifest}
    if not expected.items() <= current.items() or sorted(current.keys() - expected.keys()) != read(BASELINE)["observed"]["extra_files"]:
        raise ValueError("Field data differs from the registered source inventory")
    events = defaultdict(set)
    entries = defaultdict(set)
    completions = defaultdict(set)
    fields_by_event = defaultdict(set)
    first_yard = Counter()
    changes = defaultdict(set)
    logs = sorted((RAW / "기사/_raw").glob("*.gz"))
    for path in logs:
        with gzip.open(path, "rt", encoding="utf8") as stream:
            for line in stream:
                match = REQUEST.search(line)
                if not match:
                    continue
                event = match[1] or match[2]
                fields = dict(RAW_ASSIGNMENT.findall(line))
                fields_by_event[event].update(k for k in fields if any(
                    term in k.lower() for term in ("time", "start", "crane", "equipment", "rehandle")))
                flow = fields.get("inOut", "").strip()
                if flow not in ("IN", "OUT", "1", "2", "TO", "FR"):
                    continue
                key = (fields.get("conNo", "").strip(), fields.get("truckNo", "").strip(), direction(flow))
                if not all(key):
                    continue
                if event == "ChangeConLoc":
                    at = parse_time(fields.get("conLocChangedTime"))
                    if at:
                        changes[key].add(at)
                if event not in EVENTS:
                    continue
                at = parse_time(fields.get(EVENTS[event][0]))
                if at is None:
                    continue
                events[event].add((*key, at))
                if event == "EnterBlock":
                    entries[key].add((at, block(fields.get("conLoc", "").strip())))
                if event == "JobDone":
                    completions[key].add(at)
                if event == "GateIn":
                    yard = parse_time(fields.get("firstYardInTime"))
                    if yard is not None:
                        first_yard["before_gate" if yard < at else "at_or_after_gate"] += 1
    inventory_note = dict(original_files=len(expected), driver_logs=len(logs),
                          structured_fields={e: sorted(v) for e, v in fields_by_event.items()},
                          first_yard_time_in_gate_messages=dict(first_yard))
    csv_timing_fields = {}
    for group in ("기사/방문", "기사/오더", "기사/이벤트", "선박/양적하목록"):
        files = sorted((RAW / group).glob("*.csv"))
        names = set()
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                names.update(next(csv.reader(stream)))
        csv_timing_fields[group] = dict(files=len(files), keys=sorted(k for k in names if any(
            term in k.lower() for term in ("time", "start", "crane", "equipment", "atw", "work_min"))))
    inventory_note["table_timing_fields"] = csv_timing_fields
    return events, entries, completions, changes, inventory_note


def order_record(row, gate, visit_end, events, entries, completions, changes):
    key = (row["con_no"], row["truck_no"], row["in_out"])
    def confirmed(event):
        at = parse_time(row[EVENTS[event][1]])
        return at if (*key, at) in events[event] else None
    gated = confirmed("GateIn") == gate
    arrival, done = confirmed("EnterBlock"), confirmed("JobDone")
    elapsed = minutes(arrival, gate) if gated and arrival and gate <= arrival else None
    location = block(row["enter_block_loc"])
    exit_at = confirmed("GateOut")
    def in_visit(t):
        return gate <= t < visit_end and (exit_at is None or t <= exit_at)
    observed = {(t, b) for t, b in entries.get(key, ()) if in_visit(t)}
    stamps = sorted(t for t, b in observed)
    located = (elapsed is not None and bool(location) and (arrival, location) in observed
               and {b for _, b in observed} == {location})
    observed_done = sorted(t for t in completions.get(key, ()) if in_visit(t))
    done_ok = done if located and done in observed_done and arrival <= done else None
    moves_in_visit = [t for t in changes.get(key, ()) if gate <= t < visit_end]
    return dict(flow=key[2], elapsed=elapsed, arrival=arrival, block=location, located=located,
                earliest=stamps[0] if located else None, latest=stamps[-1] if located else None,
                done=done_ok, done_earliest=min(observed_done) if done_ok else None,
                done_latest=max(observed_done) if done_ok else None,
                location_change_inside=bool(done_ok and any(arrival <= t <= done_ok for t in moves_in_visit)))


def stops(records):
    if not all(r["located"] for r in records):
        return "incomplete_or_conflicting", []
    ordered = sorted(records, key=lambda r: r["arrival"])
    for _, tied in groupby(ordered, key=lambda r: r["arrival"]):
        if len({r["block"] for r in tied}) > 1:
            return "simultaneous_distinct_blocks", []
    grouped = [list(items) for _, items in groupby(ordered, key=lambda r: r["block"])]
    if any(max(r["latest"] for r in left) >= min(r["earliest"] for r in right) for left, right in pairwise(grouped)):
        return "distinct_block_report_windows_overlap", []
    return ("one_block" if len(grouped) == 1 else "ordered_blocks"), grouped


def month_visit_links():
    root = Path("build/yr332_joint_turnaround")
    result = {}
    for split, replay in (("test", BP / "result.json"), ("train", root / "short-window/wave-0-0/configuration.json")):
        registered = read(REPORTS / f"yr332_joint_turnaround/{split}-input.json")
        path = root / registered["path"]
        digest = sha(path)
        if digest != registered["sha256"] or digest != read(replay)["manifest_sha256"]:
            raise ValueError("Registered month and replay differ")
        schedule = read(path)["schedule"]
        if len(schedule) != registered["orders"]:
            raise ValueError("Incomplete month schedule")
        result[split] = dict(seed=registered["seed"], orders=len(schedule), manifest_sha256=digest,
                             orders_with_visit_link=sum(row.get("visit") is not None for row in schedule))
    return result


def main():
    events, entries, completions, changes, source = source_events()
    entered = []
    for path in sorted((RAW / "기사/방문").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            entered.extend(row for row in csv.DictReader(stream) if row["gate_in_time"])
    visits = sorted(physical_visits(gate_records(entered), MERGE_S), key=lambda v: (v["truck"], v["arrival"]))
    counts, multi, rescans, directions, links = (Counter() for _ in range(5))
    out_groups, legs = defaultdict(list), defaultdict(list)
    proxy = Counter()
    same_block_pair = Counter()
    scan_gaps = []
    for _, truck_visits in groupby(visits, key=lambda v: v["truck"]):
        for visit, nxt in pairwise([*truck_visits, None]):
            gate = visit["arrival"]
            end = nxt["first_scan"] if nxt else datetime.max
            rows = visit["rows"]
            records = [order_record(r, gate, end, events, entries, completions, changes) for r in rows]
            is_multi, repeated = len(rows) > 1, bool(visit["superseded"])
            counts["visits"] += 1
            category, grouped = stops(records)
            if repeated:
                rescans["visits"] += 1
                rescans["superseded_gate_records"] += len(visit["superseded"])
                rescans["multi" if is_multi else "single"] += 1
                prior_rows = visit["superseded"][0][1]
                if any((r["con_no"], r["truck_no"], r["in_out"], visit["first_scan"]) in events["GateIn"] for r in prior_rows) and any(
                        (r["con_no"], r["truck_no"], r["in_out"], gate) in events["GateIn"] for r in rows):
                    scan_gaps.append(minutes(gate, visit["first_scan"]))
            if is_multi:
                multi[category] += 1
                if len(grouped) > 1:
                    for left, right in pairwise(grouped):
                        links[(left[0]["block"], right[0]["block"])] += 1
            if category == "one_block" and len(records) == 2 and {r["flow"] for r in records} == {"IN", "OUT"}:
                inbound, outbound = sorted(records, key=lambda r: r["flow"])
                if not inbound["done"] or not outbound["done"]:
                    label = "missing_valid_completion"
                elif inbound["done_latest"] < outbound["done_earliest"]:
                    label = "inbound_completed_first"
                elif outbound["done_latest"] < inbound["done_earliest"]:
                    label = "outbound_completed_first"
                else:
                    label = "tied_or_overlapping"
                same_block_pair[label] += 1
            for row in records:
                if row["done"]:
                    proxy["located_order_with_done"] += 1
                    proxy["change_between_entry_and_done"] += row["location_change_inside"]
                if row["flow"] != "OUT" or row["elapsed"] is None:
                    continue
                if not is_multi:
                    label = "single_rescan" if repeated else "single_no_repeat"
                elif len(grouped) <= 1:
                    label = category
                else:
                    index = next(i for i, group in enumerate(grouped) if any(r is row for r in group))
                    label = "first_of_distinct_blocks" if index == 0 else "later_of_distinct_blocks"
                out_groups[label].append(row["elapsed"])
            if len(records) == 2 and len(grouped) == 2:
                first, second = (group[0] for group in grouped)
                complete = first["done"] is not None and first["done_latest"] <= second["earliest"]
                pattern = first["flow"] + "_to_" + second["flow"]
                directions[(pattern, "first_done_by_second_entry" if complete else "completion_unresolved_or_overlap")] += 1
                if complete:
                    a = first["elapsed"]
                    b = minutes(first["done"], first["arrival"])
                    c = minutes(second["arrival"], first["done"])
                    total = second["elapsed"]
                    if not isclose(a+b+c, total, abs_tol=1e-8):
                        raise ValueError("Two-stop timing ledger does not close")
                    legs[pattern].append((a, b, c, total))

    baseline = read(BASELINE)["observed"]
    all_out = [v for values in out_groups.values() for v in values]
    if (summary(all_out) != baseline["groups"]["OUT"]["gate_to_block"]
            or summary(out_groups["single_no_repeat"]) != baseline["gate_to_block_single_order_no_repeat"]["OUT"]):
        raise ValueError("Outgoing cases do not reconcile with the prior raw-confirmed cohort")
    if sum(multi.values()) != sum(len(v["rows"]) > 1 for v in visits):
        raise ValueError("Multi-visit classification is not exhaustive")
    if sum(v for (_, status), v in directions.items() if status == "first_done_by_second_entry") != sum(map(len, legs.values())):
        raise ValueError("Two-order completion records do not reconcile")
    result = dict(field_source=source, visits=counts["visits"], multi_visits=dict(multi), rescans=dict(rescans),
                  confirmed_first_to_effective_scan_min=stats(scan_gaps), proxy_coverage=dict(proxy),
                  outbound_gate_to_block_min={k: summary(v) for k,v in out_groups.items()},
                  two_order_directions={"|".join(k): v for k,v in directions.items()},
                  same_block_two_mixed_orders=dict(same_block_pair), months=month_visit_links(),
                  frequent_block_transitions=[dict(origin=a, destination=b, visits_or_legs=n)
                                              for (a,b),n in links.most_common(5)],
                  closed_legs_min={p: dict(zip(("gate_to_first", "first_block_to_first_done",
                                                "first_done_to_second_block", "gate_to_second"),
                                               map(stats, zip(*v)))) for p,v in legs.items()},
                  source_audit_sha256=sha(BASELINE))
    save(BASELINE.with_name("visit-sequence.json"), result)


if __name__ == "__main__":
    main()
