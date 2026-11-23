"""Replay the proposed time objective and truck-rate proxy on the saved diagnostic month."""
import argparse
from collections import defaultdict
import gzip
import json
from math import fsum, isclose
from pathlib import Path


RUNS = Path("outputs/reports/yr332_month_cranes/runs")
POLICIES = ("SF_SPT", "trained", "LWKR")
DAY = 86400


def times(rows, identity, start, lower, upper):
    cohort = {}
    for row in rows:
        if lower <= row[start] < upper:
            key = row[identity]
            end = row["finish"]
            if key in cohort or end is None or end < row[start]:
                raise ValueError(f"Duplicate, unfinished, or reversed interval at {key}")
            cohort[key] = (row[start], end)
    if not cohort:
        raise ValueError("Expected a nonempty cohort")
    return cohort


def elapsed_seconds(cohort):
    # A job or port call counts once while open, including during service.
    events = defaultdict(int)
    for start, end in cohort.values():
        events[start] += 1
        events[end] -= 1
    previous, active = 0.0, 0
    increments = []
    for instant in sorted(events):
        increments.append(active * (instant - previous))
        active += events[instant]
        previous = instant
    area = fsum(increments)
    direct = fsum(end - start for start, end in cohort.values())
    if active or not isclose(area, direct, rel_tol=1e-12):
        raise ValueError("Event-time reward disagrees with completed durations")
    return area


def review(lower_day, upper_day):
    result = {}
    reference_inputs = None
    for name in POLICIES:
        folder = RUNS / name
        window = (lower_day * DAY, upper_day * DAY)
        with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf-8") as stream:
            trucks = times(map(json.loads, stream), "job", "gate", *window)
        rows = json.loads((folder / "ships.json").read_text(encoding="utf-8"))
        for ship in rows:
            finishes = ship["stream_finishes"]
            if not finishes or None in finishes or ship["finish"] != max(finishes):
                raise ValueError(f"Unverified final quay move for {ship['ship']}")
        vessels = times(rows, "ship", "start", *window)
        inputs = [{key: start for key, (start, _) in cohort.items()}
                  for cohort in (trucks, vessels)]
        if reference_inputs is None:
            reference_inputs = inputs
        elif inputs != reference_inputs:
            raise ValueError("The policies do not share the same cohorts and entry times")
        truck_seconds, vessel_seconds = elapsed_seconds(trucks), elapsed_seconds(vessels)
        exposure = fsum(max(end - start - 3600, 0) for start, end in trucks.values())
        result[name] = dict(truck_orders=len(trucks), port_calls=len(vessels),
                            truck_minutes=truck_seconds / len(trucks) / 60,
                            vessel_hours=vessel_seconds / len(vessels) / 3600,
                            truck_proxy_krw=exposure * 20000 / 1800)
    rule = result["SF_SPT"]
    for row in result.values():
        row["time_score"] = 0.5 * (row["truck_minutes"] / rule["truck_minutes"]
                                   + row["vessel_hours"] / rule["vessel_hours"])
        row["truck_proxy_saving_vs_rule_krw"] = rule["truck_proxy_krw"] - row["truck_proxy_krw"]
    equivalent = (rule["truck_orders"] * rule["truck_minutes"]
                  / (rule["port_calls"] * rule["vessel_hours"] * 60))
    return dict(policies=result, truck_minutes_equivalent_to_one_vessel_minute=equivalent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the diagnostic JSON")
    args = parser.parse_args()
    report = dict(reference=POLICIES[0], source=str(RUNS.as_posix()),
                  primary_days_2_29=review(1, 29), all_days_1_30=review(0, 30))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")
