"""Pair the two shortest-job ledgers and separate truck waiting from cargo readiness."""
import gzip
import json
from collections import Counter, defaultdict
from math import ceil, fsum, isclose
from pathlib import Path
from statistics import median

from joint_turnaround_preflight import CACHE, OUT as MONTH, read, save
from yard_rl.v7.stage.seed_bundle import file_sha256
from yard_rl.v7.stage.seed_inventory import enumerate_inventory
from yard_rl.v7.reward.joint_turnaround import VESSEL_TIME_WEIGHT


CLASSIC = Path("outputs/reports/yr332_unguarded_spt/evaluation")
GUARDED = MONTH / "evaluation/SF_SPT"
OUT = Path("outputs/reports/yr332_short_job_audit/month-rule-waits.json")


def ledger(folder, manifest_sha, entries):
    result = read(folder / "result.json")
    window = result["windows"]["all"]
    if (result["manifest_sha256"] != manifest_sha or result["cargo"]["dependent_reroutes"]
        or window["unfinished_trucks"] or window["unfinished_ships"]):
        raise ValueError(f"Incomplete or different month in {folder}")
    rows = {}
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        for r in map(json.loads, stream):
            j = r["job"]
            e = entries[j]
            if (j in rows or r["flow"] != e["flow"] or r["gate"] != e["arrival_s"] or
                not isclose(r["block"], e["arrival_s"] + e["travel_s"], abs_tol=1e-6) or
                not r["gate"] <= r["block"] <= r["start"] <= r["finish"]):
                raise ValueError(f"Clock, identity or flow mismatch for {j} in {folder}")
            rows[j] = r
    if rows.keys() != entries.keys() or len(rows) != window["trucks"]:
        raise ValueError(f"Incomplete order coverage in {folder}")
    return rows, window


def block_wait(r):
    return (r["start"] - r["block"]) / 60


def waits(rows):
    values = sorted(map(block_wait, rows))
    return dict(count=len(values), mean_min=fsum(values)/len(values), median_min=median(values),
                p95_min=values[ceil(.95*len(values))-1], max_min=values[-1],
                over_24min=sum(v > 24 for v in values), at_least_24h=sum(v >= 1440 for v in values))


def peak_queue(rows, entries):
    changes = defaultdict(Counter)
    for j, r in rows.items():
        events = changes[entries[j]["block"]]
        events[r["block"]] += 1
        events[r["start"]] -= 1
    area = 0.
    peak = dict(orders=0)
    for block, events in changes.items():
        previous, count = 0., 0
        for t, change in sorted(events.items()):
            area += count * (t - previous)
            previous = t
            count += change
            if count > peak["orders"]:
                peak = dict(orders=count, block=block, time_s=t)
        if count:
            raise ValueError(f"Queue did not empty in {block}")
    if not isclose(area, fsum(r["start"]-r["block"] for r in rows.values()), rel_tol=1e-10):
        raise ValueError("Integrated queues and individual waits disagree")
    return peak


def readiness(rows, entries, sources):
    counts = Counter()
    longest_ready = None
    for j, r in rows.items():
        if r["flow"] != "GATE_OUT":
            continue
        src = sources[entries[j]["target"]]
        kind = src["source_kind"]
        if kind == "VESSEL_DISCHARGE":
            counts["individual_discharge_finish_unrecorded"] += 1
            continue
        if kind not in ("INITIAL", "GATE_IN"):
            raise ValueError(f"Unknown cargo source {kind}")
        ready = 0. if kind == "INITIAL" else rows[src["source_job"]]["finish"]
        if ready > r["start"] + 1e-6:
            raise ValueError(f"Truck service preceded cargo readiness for {j}")
        if ready > r["block"]:
            counts["arrived_before_cargo_ready"] += 1
        else:
            counts["cargo_ready_by_block_arrival"] += 1
            if longest_ready is None or block_wait(r) > longest_ready["block_wait_min"]:
                longest_ready = dict(job=j, source_kind=kind, source_job=src["source_job"],
                                     cargo_ready_s=ready, block_s=r["block"], start_s=r["start"],
                                     block=src["block"], block_wait_min=block_wait(r))
            if block_wait(r) >= 1440:
                counts["waited_at_least_24h_with_cargo_already_in_yard"] += 1
    return dict(counts=counts, longest_with_known_ready_cargo=longest_ready)


def main():
    info = read(MONTH / "test-input.json")
    manifest = CACHE / info["path"]
    if file_sha256(manifest) != info["sha256"]:
        raise ValueError("Unexpected manifest digest")
    with gzip.open(manifest, "rt", encoding="utf8") as f:
        doc = json.load(f)
    entries = {e["job_id"]: e for e in doc["schedule"]}
    if len(entries) != len(doc["schedule"]):
        raise ValueError("Duplicate input job")
    listed, _ = enumerate_inventory(doc["initial_inventory"], doc["schedule"], doc["vessels_by_day"], doc["seed"])
    sources = {s["container"]: s for s in listed}
    classic, cw = ledger(CLASSIC, info["sha256"], entries)
    guarded, gw = ledger(GUARDED, info["sha256"], entries)
    paired = Counter()
    for j, r in classic.items():
        if r["block"] != guarded[j]["block"]:
            raise ValueError(f"Arrival differs between policies for {j}")
        diff = block_wait(guarded[j]) - block_wait(r)
        key = "equal" if abs(diff) < 1e-6 else "longer" if diff > 0 else "shorter"
        paired[key] += 1
    data = {}
    for label, rows in (("classic", classic), ("guarded", guarded)):
        data[label] = {flow: waits(r for r in rows.values() if r["flow"] == flow)
                       for flow in ("GATE_IN", "GATE_OUT")}
        data[label].update(all=waits(rows.values()), cargo=readiness(rows, entries, sources),
                           peak_simultaneous_orders_at_one_block=peak_queue(rows, entries))
    calls = [{r["ship"]: r for r in read(folder/"ships.json")} for folder in (CLASSIC, GUARDED)]
    if ({j: r["start"] for j, r in calls[0].items()} != {j: r["start"] for j, r in calls[1].items()}
        or any(len(c) != w["ships"] for c,w in zip(calls, (cw,gw)))):
        raise ValueError("Port call cohorts differ")
    truck_saved = fsum((r["finish"]-guarded[j]["finish"])/60 for j,r in classic.items())
    vessel_added = fsum((calls[1][j]["finish"]-r["finish"])/60 for j,r in calls[0].items())
    for rows, ships, window in zip((classic, guarded), calls, (cw, gw)):
        if not (isclose(fsum((r["finish"]-r["gate"])/60 for r in rows.values())/len(rows), window["truck_minutes"])
                and isclose(fsum((r["finish"]-r["start"])/3600 for r in ships.values())/len(ships), window["ship_hours"])):
            raise ValueError("Turnaround totals disagree with the stored month")
    hourly = Counter(int(r["gate"]//3600) for r in classic.values())
    save(OUT, dict(manifest_sha256=info["sha256"], seed=doc["seed"], guarded_vs_classic=paired,
                   block_wait=data, gate_peak_orders_per_hour=max(hourly.values()),
                   hours_over_245_orders=sum(n>245 for n in hourly.values()),
                   truck_turnaround_min_saved=truck_saved, port_call_min_added=vessel_added,
                   vessel_weight_break_even=truck_saved/vessel_added if vessel_added else None,
                   registered_vessel_weight=VESSEL_TIME_WEIGHT))


if __name__ == "__main__":
    main()
