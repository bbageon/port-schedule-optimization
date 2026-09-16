"""Explain unfinished demand using container-count identities, without simulation.

Regenerates the frozen diagnostic input only. This is an accounting bound, not a
service-capacity test: a negative balance precludes completing every removal in
that block with the recorded assignments, even if service were instantaneous.
"""
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from yard_rl.v3.eval.seed_bank import create_month_payload, digest
from yard_rl.v3.stage.month import plan_month

FIELDS = ("job_id", "requested_day", "flow", "requested_block",
          "requested_arrival_s", "lead_s", "requested_target", "travel_s")


def balances(initial, requests, vessels, *, use_final_blocks):
    blocks = {b: Counter(initial_boxes=n) for b, n in initial.items()}
    for r in requests:
        block = r["final_block"] if use_final_blocks else r["requested_block"]
        item = blocks[block]
        inward = r["flow"] == "GATE_IN"
        item["truck_in_requested" if inward else "truck_out_requested"] += 1
        if not use_final_blocks:
            continue
        if r["job_done_s"] is not None:
            item["truck_in_completed" if inward else "truck_out_completed"] += 1
        else:
            item["truck_in_unfinished" if inward else "truck_out_unfinished"] += 1
    for v in vessels:
        item = blocks[v["block"]]
        inward = v["work"] == "DISCHARGE"
        prefix = "vessel_in" if inward else "vessel_out"
        item[prefix + "_requested"] += v["asked"]
        if use_final_blocks:
            item[prefix + "_completed"] += v["completed_yard_jobs"]
            item[prefix + "_unfinished"] += v["asked"] - v["completed_yard_jobs"]
    rows = {}
    for b, item in blocks.items():
        planned = (item["initial_boxes"] + item["truck_in_requested"] + item["vessel_in_requested"]
                   - item["truck_out_requested"] - item["vessel_out_requested"])
        current = (item["initial_boxes"] + item["truck_in_completed"] + item["vessel_in_completed"]
                   - item["truck_out_completed"] - item["vessel_out_completed"])
        rows[b] = {**item, "balance_if_all_requested_work_completed": planned,
                   "minimum_unfinished_removals": max(0, -planned),
                   "end_stock_from_completed_flows": current if use_final_blocks else None}
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    days = [replace(d, n_days=10) for d in plan_month(9_500_000)[:10]]
    payload = create_month_payload(9_500_000, days=days)
    initial = {b: len(s["containers"]) for b, s in payload["initial_scenarios"].items()}
    expected = [{"job_id": e["job_id"], "requested_day": e["day"], "flow": e["flow"],
        "requested_block": e["block"], "requested_arrival_s": e["arrival_s"],
        "lead_s": e["lead_s"], "requested_target": e.get("target"), "travel_s": e["travel_s"]}
        for e in payload["schedule"]]
    file_sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {"scope": "fixed 10-day diagnostic inventory accounting; no simulation or training",
              "input_payload_sha256": digest(payload), "arms": {}, "claim_eligible": False,
              "generator_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "analyzer_sha256": file_sha(__file__), "artifact_sha256": {}}
    for arm in ("NO_REALLOC", "RL", "RL_NOVETO"):
        folder = args.run / arm
        if not (folder / "result.json").exists():
            continue
        raw = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        with gzip.open(folder / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
            requests = [json.loads(line) for line in stream]
        assert [{k: r[k] for k in FIELDS} for r in requests] == expected
        assert raw["seed"] == 9_500_000 and raw["request_summary"]["skipped"] == 0
        for name in ("result.json", "requests.jsonl.gz", "manifest.json"):
            result["artifact_sha256"][str(folder / name)] = file_sha(folder / name)
        original = balances(initial, requests, raw["vessel_work_ledger"], use_final_blocks=False)
        final = balances(initial, requests, raw["vessel_work_ledger"], use_final_blocks=True)
        result["arms"][arm] = {"original_assignments": original, "final_assignments": final,
            "negative_balance_original": {b: r for b, r in original.items()
                if r["minimum_unfinished_removals"]},
            "negative_balance_final": {b: r for b, r in final.items()
                if r["minimum_unfinished_removals"]},
            "end_stock_total": sum(r["end_stock_from_completed_flows"] for r in final.values()),
            "recorded_end_stock_total": raw["days"][-1]["load_after"]["boxes"]}
        info = result["arms"][arm]
        assert all(r["end_stock_from_completed_flows"] >= 0 for r in final.values())
        assert info["end_stock_total"] == info["recorded_end_stock_total"]
    result["limits"] = ["Negative planned balance is a lower bound on unfinished removals, not the cause of every unfinished job.",
        "Original-assignment balances describe requested work only, not an executed policy.",
        "Positive balance does not establish service feasibility or timely availability; no new inventory or demand is introduced."]
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for arm, v in result["arms"].items():
        print(json.dumps({"arm": arm, "original_deficit": {b: r["minimum_unfinished_removals"]
            for b, r in v["negative_balance_original"].items()},
            "final_deficit": {b: r["minimum_unfinished_removals"] for b, r in v["negative_balance_final"].items()},
            "end_stock_total": v["end_stock_total"]}))


if __name__ == "__main__":
    main()
