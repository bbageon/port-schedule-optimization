"""Prepare visit-linked copies of registered cargo inputs and verify a frozen replay."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from math import fsum, isclose
from pathlib import Path
from tempfile import TemporaryDirectory

from joint_turnaround_preflight import CACHE, OUT as REGISTERED, read, save
from inbound_realism_common import stats
from month_crane_input import MANIFEST, OUT as COMMON, build
from yard_rl.v7.reward.gate_age import GateAgeTruckClock, StopGateAgeClock, truck_points
from yard_rl.v7.stage.fixed_seed import as_audit_input, sha
from yard_rl.v7.stage.month_visits import stop_intervals, visit_plans
from yard_rl.v7.stage.orders import orders_from_schedule
from yard_rl.v7.stage.seed_bundle import file_sha256, save_seed_bundle
from yard_rl.v7.stage.seed_inventory import enumerate_inventory
from yard_rl.v7.visit_link import link_fixed_month
from yard_rl.v7.world.integrated.yard_layout import terminal_layout

OUT = Path("outputs/reports/yr332_gate_age_curve/linked-visits")
ROOT = CACHE / "linked-visits"
PROFILE = Path("outputs/reports/yr332_confirmed_demand/confirmed/visit-link.json")


def assignment(schedule):
    return sha([(e["job_id"], int(e["arrival_s"] // 3600), e["block"], e["flow"],
                 e["con_no"], e["target"], e["size_ft40"]) for e in schedule])


def ledger_digest(path):
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(split):
    ROOT.mkdir(parents=True, exist_ok=True)
    if split == "busy":
        from yard_rl.v7.ppo.continuous import make_plan
        seed = 9900307
        days = make_plan(seed, 3, 3500)
        with TemporaryDirectory(dir=ROOT) as scratch:
            path = Path(scratch) / "busy-original.json.gz"
            parent = dict(days=[d.as_dict() for d in days], **build(seed, days, path))
            with gzip.open(path, "rt", encoding="utf8") as stream:
                doc = json.load(stream)
        return prepare_document(split, parent, doc)
    parent = read(COMMON / "smoke-input.json" if split == "smoke" else REGISTERED / f"{split}-input.json")
    path = (MANIFEST.parent if split == "smoke" else CACHE) / parent["path"]
    if file_sha256(path) != parent["sha256"]:
        raise ValueError("Original registered input differs")
    with gzip.open(path, "rt", encoding="utf8") as stream:
        doc = json.load(stream)
    return prepare_document(split, parent, doc)


def prepare_document(split, parent, doc):
    schedule = doc["schedule"]
    identity = assignment(schedule)
    link_fixed_month(doc["seed"], schedule, read(PROFILE), terminal_layout())
    if assignment(schedule) != identity:
        raise ValueError("Cargo assignment, hourly orders or order identity changed")
    schedule.sort(key=lambda e: e["arrival_s"])
    plans = visit_plans(schedule)
    sizes = Counter(sum(len(js) for _, _, js in p["stops"]) for p in plans.values())
    linked = sum(n * count for n, count in sizes.items())
    sizes[1] = len(schedule) - linked
    doc["orders"] = [asdict(order) for order in orders_from_schedule(as_audit_input(doc))[0].values()]
    doc["sources"] = enumerate_inventory(doc["initial_inventory"], schedule, doc["vessels_by_day"], doc["seed"])[0]
    OUT.mkdir(parents=True, exist_ok=True)
    target = ROOT / f"{split}.json.gz"
    with TemporaryDirectory(dir=ROOT) as scratch:
        scratch_file = Path(scratch) / target.name
        info = save_seed_bundle(doc, scratch_file)
        if target.exists():
            if file_sha256(target) != info["sha256"]:
                raise ValueError("Stored visit-linked input differs from rebuild")
        else:
            scratch_file.replace(target)
    record = dict(split=split, seed=doc["seed"], days=parent["days"], orders=len(schedule),
                  original_sha256=parent["sha256"], visit_profile_sha256=file_sha256(PROFILE),
                  cargo_assignment_sha256=identity, visits=sum(sizes.values()), sizes=dict(sizes),
                  multi_visits=len(plans), linked_orders=linked, **info)
    save(OUT / f"{split}-input.json", record)
    print(split, record["orders"], record["visits"], record["multi_visits"], flush=True)


def verify(split):
    import torch
    from month_crane_replay import run
    torch.set_num_threads(1)
    info_path = OUT / f"{split}-input.json"
    folder = ROOT / f"{split}-replay"
    result = run("SPT_UNGUARDED", False, info_path=info_path, manifest_dir=ROOT, folder=folder)
    if not result["complete"]:
        raise ValueError("Incomplete visit accounting replay")
    check_ledger(split)


def check_ledger(split):
    """Independently regroup saved order events using the input and reconcile actual engine output."""
    info = read(OUT / f"{split}-input.json")
    folder = ROOT / f"{split}-replay"
    meta = read(folder / "result.json")
    if meta["manifest_sha256"] != info["sha256"] or file_sha256(ROOT / info["path"]) != info["sha256"]:
        raise ValueError("Ledger and input use different manifests")
    with gzip.open(ROOT / info["path"], "rt", encoding="utf8") as stream:
        schedule = json.load(stream)["schedule"]
    groups = defaultdict(list)
    for e in schedule:
        groups[e["visit"]["id"] if e.get("visit") else e["job_id"]].append(e)
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        orders = {r["job"]: r for r in map(json.loads, stream)}
    with gzip.open(folder / "visits.jsonl.gz", "rt", encoding="utf8") as stream:
        visits = list(map(json.loads, stream))
    if len(orders) != info["orders"] or len(visits) != len(groups) or {v["visit"] for v in visits} != groups.keys():
        raise ValueError("Missing or duplicate visits in the replay")
    intervals, two_stop_pairs = [], defaultdict(list)
    followups = cross_block = multi_job_stops = cross_midnight = 0
    for v in visits:
        entries = groups[v["visit"]]
        records = [orders[e["job_id"]] for e in entries]
        gate, finish = v["gate"], max(r["finish"] for r in records)
        if ({r["gate"] for r in records} != {gate} or any(e["arrival_s"] != gate for e in entries)
                or finish != v["finish"]
                or v["seconds"] != finish - gate or len(records) != v["orders"]):
            raise ValueError("Visit does not match original gate and final completion")
        intervals.append((gate, finish))
        cross_midnight += gate // 86400 != finish // 86400
        plan = entries[0].get("visit")
        if plan:
            prior = gate
            for k, (bid, _, jobs) in enumerate(plan["stops"]):
                expected = prior + plan["travel_s"][k]
                if any(not isclose(orders[j]["block"], expected, rel_tol=0, abs_tol=1e-6) for j in jobs):
                    raise ValueError("Stop arrived before or after its physical release")
                prior = max(orders[j]["finish"] for j in jobs)
                followups += k > 0
                cross_block += k > 0 and bid != plan["stops"][k-1][0]
                multi_job_stops += len(jobs) > 1
            if len(entries) == len(plan["stops"]) == 2 and plan["stops"][0][0] != plan["stops"][1][0]:
                first, second = [orders[s[2][0]] for s in plan["stops"]]
                pattern = first["flow"][5:] + "_to_" + second["flow"][5:]
                two_stop_pairs[pattern].append((second["block"] - first["finish"])/60)
    direct = fsum((f-g)/60 for g, f in intervals)
    clock = GateAgeTruckClock(intervals)
    curved = fsum(truck_points((f-g)/60) for g, f in intervals)
    segments = stop_intervals(visit_plans(schedule), orders.values(), {e["job_id"]: e["block"] for e in schedule})
    by_block = [StopGateAgeClock(seg) for seg in segments.values()]
    hour_checks = [abs(clock.minutes_at(t) - fsum(b.minutes_at(t) for b in by_block))
                   for t in range(0, int(max(f for _, f in intervals)) + 3600, 3600)]
    events = meta["visit_events"]
    all_days = meta["windows"]["all"]
    if not (followups and cross_block and events["exits"] == info["multi_visits"]
            and events["late_releases"] == events["late_exits"] == 0
            and (split != "busy" or (multi_job_stops and cross_midnight))
            and all_days["trucks"] == len(visits) and all_days["truck_orders"] == len(orders)
            and all_days["unfinished_trucks"] == all_days["unfinished_ships"] == meta["skipped"] == 0
            and isclose(all_days["truck_minutes"], direct/len(visits), abs_tol=1e-8)
            and isclose(clock.total_minutes, curved, rel_tol=1e-10) and max(hour_checks) < 1e-5):
        raise ValueError("Visit events, count, average or score failed reconciliation")
    save(OUT / f"{split}-accounting.json", dict(input_sha256=info["sha256"], code=meta["code"],
         rule=meta["arm"], days=meta["days"], orders=len(orders), visits=len(visits),
         multi_visits=info["multi_visits"], subsequent_stops=followups, different_block_transitions=cross_block,
         multi_job_stops=multi_job_stops, visits_spanning_midnight=cross_midnight, visit_events=events,
         first_done_to_second_block_min={key: stats(v) for key, v in two_stop_pairs.items()},
         visit_minutes=direct, per_order_minutes=fsum(r["seconds"]/60 for r in orders.values()),
         curved_visit_points=curved, clock_points=clock.total_minutes, windows=meta["windows"],
         hourly_block_checks=len(hour_checks), max_block_points_error=max(hour_checks),
         unfinished_orders=0, skipped=meta["skipped"], cargo=meta["cargo"],
         ledger_content_sha256={p: ledger_digest(folder / p) for p in ("visits.jsonl.gz","trucks.jsonl.gz","ships.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {"prepare": prepare, "verify": verify, "check-ledger": check_ledger}
    parser.add_argument("action", choices=actions)
    parser.add_argument("split", choices=("smoke", "busy", "train", "test"))
    args = parser.parse_args()
    actions[args.action](args.split)
