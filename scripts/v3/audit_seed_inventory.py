"""Read all locked input months and check original-assignment container balances.

No policy runs, new demand, filtering, resampling, or modifications to the bank.
Day-end counts assume every job planned for that day could finish immediately;
they do not certify time-of-day availability, crane capacity, or service success.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {3})
    manifest = json.loads((args.bank / "manifest.json").read_text(encoding="utf-8"))
    months = []
    for seed in manifest["spec"]["seeds"]:
        path = args.bank / f"seed-{seed}.json.gz"
        summary = json.loads(path.with_name(f"seed-{seed}.summary.json").read_text(encoding="utf-8"))
        bundle_sha = sha(path)
        assert bundle_sha == summary["bundle_sha256"]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        assert data["seed"] == seed and len(data["schedule"]) == summary["requested_trucks"]
        stock = {b: len(s["containers"]) for b, s in data["initial_scenarios"].items()}
        initial = dict(stock)
        daily = {d["index"]: Counter() for d in data["days"]}
        truck_net, vessel_net = Counter(), Counter()
        for r in data["schedule"]:
            assert r["flow"] in ("GATE_IN", "GATE_OUT")
            sign = 1 if r["flow"] == "GATE_IN" else -1
            daily[r["day"]][r["block"]] += sign
            truck_net[r["block"]] += sign
        for day, vessels in data["vessels"].items():
            for v in vessels:
                assert v["work"] in ("DISCHARGE", "LOAD")
                moves = v["moves"] if v["work"] == "DISCHARGE" else -v["moves"]
                daily[int(day)][v["block"]] += moves
                vessel_net[v["block"]] += moves
        negative_days = []
        for day in sorted(daily):
            for b, n in daily[day].items():
                stock[b] += n
            deficits = {b: -n for b, n in stock.items() if n < 0}
            if deficits:
                negative_days.append({"day_index": day, "shortfall_by_block": deficits})
        assert all(stock[b] == initial[b] + truck_net[b] + vessel_net[b] for b in stock)
        months.append({"seed": seed, "requested_trucks": len(data["schedule"]),
            "bundle_sha256": bundle_sha, "initial_boxes": initial,
            "planned_end_balance_by_block": stock,
            "month_end_shortfall": {b: -n for b, n in stock.items() if n < 0},
            "negative_planned_day_balances": negative_days})
        print(json.dumps({"seed": seed, "month_end_shortfall": months[-1]["month_end_shortfall"]}), flush=True)
    result = {"schema": "yr317.input-inventory-accounting.v1", "months": months,
        "same_locked_seed_list": [m["seed"] for m in months] == manifest["spec"]["seeds"],
        "source_manifest_sha256": sha(args.bank / "manifest.json"), "analyzer_sha256": sha(Path(__file__)),
        "requested_trucks": sum(m["requested_trucks"] for m in months),
        "months_with_negative_end_balance": sum(bool(m["month_end_shortfall"]) for m in months),
        "months_with_negative_day_balance": sum(bool(m["negative_planned_day_balances"]) for m in months),
        "new_simulations": 0, "new_training_runs": 0, "bank_modified": False,
        "limits": "Count sufficiency is necessary but not sufficient for timely physical service. No seed is replaced or discarded.",
        "claim_eligible": False}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
