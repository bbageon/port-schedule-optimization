"""Independently verify matched port calls and aggregate the paired final trials."""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from vessel_turnaround_experiment import LOADS, OUT, SEEDS, save


def checked_metrics(row):
    calls = row["vessel_calls"]
    assert row["vessel_complete"] and len(calls) == row["n_vessels"] > 0
    assert len({v["vessel_id"] for v in calls}) == len(calls)
    for v in calls:
        assert v["remaining"] == 0 and v["finish_s"] == max(v["stream_finishes"])
        hours = (v["finish_s"] - v["start_s"]) / 3600
        assert hours > 0 and math.isclose(hours, v["hours"])
        assert math.isclose(hours * v["gt"] * 35.88, v["cost"])
    assert math.isclose(sum(v["cost"] for v in calls), row["c_vessel"])
    assert math.isclose(sum(row[k] for k in ("c_wait","c_move","c_rehandle","c_vessel")),row["phi_krw"])
    assert row["n_trucks"] == row["admitted"] == row["load"]
    return dict(cost=row["phi_krw"], ship_hours=mean(v["hours"] for v in calls),
                truck_minutes=row["mean_gate_to_job_done_s"]/60, unfinished_trucks_24h=row["n_job_censored"],
                vessel_cost=row["c_vessel"], truck_cost=row["c_wait"], move_cost=row["c_move"],
                rehandle_cost=row["c_rehandle"], ships=len(calls))


def averages(rows):
    return {k:mean(r[k] for r in rows) for k in rows[0]}


def main():
    rows = defaultdict(list)
    paired = defaultdict(set)
    for arm, seeds in (("rule",(0,)), ("previous",SEEDS), ("trained",SEEDS)):
        for index in range(3):
            for load in LOADS:
                for seed in seeds:
                    path = OUT/"evaluation"/f"{arm}-{seed}-{index}-{load}.json"
                    row = json.loads(path.read_text(encoding="utf8"))
                    assert (row["arm"],row["index"],row["load"],row["split"]) == (arm,index,load,"holdout")
                    paired[index,load].add((row["input_seed"],row["demand_digest"],row["quay_manifest_digest"]))
                    rows[arm,index,load].append(checked_metrics(row))
    assert len(paired) == 9 and all(len(s) == 1 for s in paired.values())
    cases = {arm:[dict(index=i,load=l,**averages(rows[arm,i,l])) for i in range(3) for l in LOADS]
             for arm in ("rule","previous","trained")}
    groups = {arm:averages([averages(rows[arm,i,l]) for i in range(3) for l in LOADS]) for arm in cases}
    changes = {}
    for ref in ("previous","rule"):
        changes[ref] = {k:100*(groups["trained"][k]/groups[ref][k]-1)
                        for k in ("cost","ship_hours","truck_minutes","vessel_cost")}
        changes[ref]["wins"] = sum(t["cost"] < r["cost"] for t,r in zip(cases["trained"],cases[ref]))
    for seed in SEEDS:
        for i in range(12):
            row=json.loads((OUT/"models"/str(seed)/f"day-{i}.json").read_text(encoding="utf8"))
            checked_metrics(row)
            assert row["reward_check"]["final_cost"] == row["phi_krw"]
        for i in range(6):
            row=json.loads((OUT/"models"/str(seed)/f"fit-{i}.json").read_text(encoding="utf8"))
            assert row["selected"] == 2048 and row["optimizer_steps"] == 128
    save(OUT/"summary.json",dict(groups=groups, changes=changes, cases=cases,
         verification=dict(heldout_trials=63, matched_inputs=9, trained_days=36, updates=18),
         scope="Fixed synthetic daily cohort, established holdout reused; not continuous port operations"))
    print(json.dumps(dict(groups=groups,changes=changes),indent=2))


if __name__ == "__main__":
    main()
