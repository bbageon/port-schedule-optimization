"""Independently verify matched port calls and aggregate the paired final trials."""
import json
import math
from collections import defaultdict
from statistics import mean

import numpy as np

from vessel_turnaround_experiment import LOADS, OUT, SEEDS, save


def checked_metrics(row):
    calls = row["vessel_calls"]
    assert len(calls) == row["n_vessels"] > 0
    assert row["vessel_complete"] == all(v["finish_s"] is not None for v in calls)
    assert len({v["vessel_id"] for v in calls}) == len(calls)
    for v in calls:
        if v["finish_s"] is None:
            assert None in v["stream_finishes"] and v["remaining"] > 0
            end = 144 * 3600
        else:
            assert v["remaining"] == 0 and v["finish_s"] == max(v["stream_finishes"])
            end = v["finish_s"]
        hours = (end - v["start_s"]) / 3600
        assert hours > 0 and math.isclose(hours, v["hours"])
        assert math.isclose(hours * v["gt"] * 35.88, v["cost"])
    assert math.isclose(sum(v["cost"] for v in calls), row["c_vessel"])
    assert math.isclose(sum(row[k] for k in ("c_wait","c_move","c_rehandle","c_vessel")),row["phi_krw"])
    assert row["n_trucks"] == row["admitted"] == row["load"]
    return dict(cost=row["phi_krw"], ship_hours=mean(v["hours"] for v in calls),
                truck_minutes=row["mean_gate_to_job_done_s"]/60, unfinished_trucks_24h=row["n_job_censored"],
                vessel_cost=row["c_vessel"], truck_cost=row["c_wait"], move_cost=row["c_move"],
                rehandle_cost=row["c_rehandle"], ships=len(calls), complete=int(row["vessel_complete"]),
                completed_ships=sum(v["finish_s"] is not None for v in calls),
                remaining_quay_moves=sum(v["remaining"] for v in calls))


def averages(rows):
    return {k:mean(r[k] for r in rows) for k in rows[0]}


def paired_cost_interval(rows, reference):
    # Keep the three load levels together within a synthetic date; pair model initializations.
    def grid(arm):
        return np.array([[mean(rows[arm, day, load][0 if arm == "rule" else seed]["cost"]
                               for load in LOADS) for seed in range(3)] for day in range(3)])
    rng = np.random.default_rng(10089900)
    days = rng.integers(3, size=(10000,3,1))
    seeds = rng.integers(3, size=(10000,1,3))
    base = grid(reference)[days,seeds].mean(axis=(1,2))
    model = grid("trained")[days,seeds].mean(axis=(1,2))
    return dict(change_pct_ci95=np.quantile(100*(model/base-1),[.025,.975]).tolist(),
                cost_difference_ci95=np.quantile(model-base,[.025,.975]).tolist(),
                bootstrap_seed=10089900,repeats=10000)


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
    eligible = [(i,l) for i in range(3) for l in LOADS
                if all(row["complete"] for arm in cases for row in rows[arm,i,l])]
    groups = {arm:averages([averages(rows[arm,i,l]) for i,l in eligible])
              for arm in cases} if eligible else {}
    changes = {}
    for ref in (("previous","rule") if eligible else ()):
        changes[ref] = {k:100*(groups["trained"][k]/groups[ref][k]-1)
                        for k in ("cost","ship_hours","truck_minutes","vessel_cost")}
        changes[ref]["wins"] = sum(averages(rows["trained",i,l])["cost"] < averages(rows[ref,i,l])["cost"]
                                   for i,l in eligible)
        if len(eligible) == 9:
            changes[ref].update(paired_cost_interval(rows,ref))
    training = {}
    for seed in SEEDS:
        complete_days = 0
        steps = 0
        for i in range(12):
            row=json.loads((OUT/"models"/str(seed)/f"day-{i}.json").read_text(encoding="utf8"))
            checked_metrics(row)
            if row["vessel_complete"]:
                assert row["reward_check"]["final_cost"] == row["phi_krw"]
                complete_days += 1
        for i in range(6):
            row=json.loads((OUT/"models"/str(seed)/f"fit-{i}.json").read_text(encoding="utf8"))
            assert ((row["selected"],row["optimizer_steps"]) == (0,0)
                    or (row["selected"],row["optimizer_steps"]) == (2048,128))
            steps += row["optimizer_steps"]
        training[str(seed)] = dict(complete_days=complete_days,optimizer_steps=steps)
    save(OUT/"summary.json",dict(groups=groups, changes=changes, cases=cases,
         comparable_inputs=eligible, training=training,
         verification=dict(heldout_trials=63, matched_inputs=9, training_days_attempted=36),
         all_input_means_including_cost_lower_bounds={arm:averages([averages(rows[arm,i,l]) for i in range(3)
             for l in LOADS]) for arm in cases},
         scope="Fixed synthetic daily cohort, established holdout reused; not continuous port operations"))
    print(json.dumps(dict(groups=groups,changes=changes),indent=2))


if __name__ == "__main__":
    main()
