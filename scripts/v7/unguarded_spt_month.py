"""Replay the unguarded shortest-job rule on BP's published test month and compare ledgers."""
import argparse
import gzip
import json
from math import ceil
from pathlib import Path

import torch

from joint_turnaround_preflight import CACHE, OUT as BL, read, save
from month_crane_replay import run
from planned_time_comparison import read_month


OUT = Path("outputs/reports/yr332_unguarded_spt")
ARMS = {
    "SPT_UNGUARDED": OUT / "evaluation",
    "SF_SPT": BL / "evaluation" / "SF_SPT",
    "bp": Path("outputs/reports/yr332_short_window/evaluation/retrained"),
}


def compare():
    manifest = read(BL / "test-input.json")["sha256"]
    metrics = {}
    for name, folder in ARMS.items():
        result = read(folder / "result.json")
        if result["manifest_sha256"] != manifest or result["days"] != 30:
            raise ValueError(f"Unexpected evaluation input for {name}")
        window = result["windows"]["all"]
        row = {k: window[k] for k in ("trucks", "ships", "unfinished_trucks", "unfinished_ships")}
        metrics[name] = row
        if row["unfinished_trucks"] or row["unfinished_ships"]:
            continue
        row.update(read_month(folder, manifest))
        with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as ledger:
            waits = sorted((r["start"] - r["block"]) / 60 for r in map(json.loads, ledger))
        row["block_wait_p95_minutes"] = waits[ceil(.95 * len(waits)) - 1]
        row["block_wait_max_minutes"] = waits[-1]
    complete = all("cohort" in row for row in metrics.values())
    if complete:
        if len({row["cohort"] for row in metrics.values()}) != 1:
            raise ValueError("Truck and ship identities or entry times differ")
        baseline = metrics["SPT_UNGUARDED"]["weighted_minutes"]
        for row in metrics.values():
            row["weighted_percent_vs_unguarded"] = 100 * (row["weighted_minutes"] / baseline - 1)
    save(OUT / "comparison.json", dict(manifest_sha256=manifest, complete=complete, arms=metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "compare"))
    args = parser.parse_args()
    if args.action == "replay":
        torch.set_num_threads(1)
        run("SPT_UNGUARDED", False, info_path=BL / "test-input.json", manifest_dir=CACHE,
            folder=ARMS["SPT_UNGUARDED"])
    compare()
