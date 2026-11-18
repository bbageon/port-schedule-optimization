"""Validate the new timing profile on the frozen 18 synthetic input cases."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_day_shapes import check_stage
from build_gate_time_calibration import BASE, OUT, OLD_PROFILE, PREREG, save
from evaluate_visit_links import rule_checks
from global_value_inputs import demand
from recheck_transport_queue import digest as file_digest
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.evaluation_probe import digest
from yard_rl.v7.information_contract import notice_time
from yard_rl.v7.stage.orders import build_stage, orders_from_schedule
from yard_rl.v7.world.integrated.profiles import build_h21_profile

ALLOWED = frozenset(("reserved_s", "lead_s", "eta_offset_s"))
SEEDS = (9975100,9975101,9975102)
LOADS = (3500,7500,15000)


def one(case, info, p):
    built = build_stage(load=case["load"], seed=case["seed"], profile=build_h21_profile(),
        information_profile=info, demand_profile=p, initial_stock_mode="preserve")
    stage = check_stage(built, case["load"])
    rules = rule_checks(built)
    if any(rules.values()):
        raise AssertionError(f"Input constraints failed: {rules}")
    assert all(0 <= notice_time(e) <= e["arrival_s"] for e in built["schedule"])
    assert all(e["eta_offset_s"] == info["flows"][e["flow"]]["eta_offset_s"] for e in built["schedule"])
    # Exercise the actual job/record adapter, not only profile deserialization.
    orders, records = orders_from_schedule(built)
    assert len(orders) == len(records) == case["load"]
    physical = [{k:v for k,v in row.items() if k not in ALLOWED} for row in built["schedule"]]
    result = dict(case=case, stage=stage, rules=rules, physical_schedule_sha256=digest(physical),
        scenarios_sha256=digest(built["scenarios"]), vessels_sha256=digest(built["vessel_schedule"]),
        pre_window_notices=built["reservation_contract"]["pre_window_notices"],
        information_contract=built["reservation_contract"])
    return result, built["schedule"]


def run(calibration_dir, out):
    target = out/"input-check.json"
    if target.exists():
        raise FileExistsError("Input check is immutable; choose a new --out")
    old = json.loads(OLD_PROFILE.read_text(encoding="utf-8"))
    new = json.loads((calibration_dir/"profile-confirmed.json").read_text(encoding="utf-8"))
    assert new["source_event_contract"]["legacy_profile_sha256"] == file_digest(OLD_PROFILE)
    config = dict(seeds=list(SEEDS), loads=list(LOADS), curve_split="train",curve_indices=[0,1,2],
        allowed_changed_schedule_fields=sorted(ALLOWED), new_profile_sha256=file_digest(calibration_dir/"profile-confirmed.json"),
        legacy_profile_sha256=file_digest(OLD_PROFILE), policy_execution=False)
    repro = repro_stamp(experiment="YR-332-ba-inputs", seeds=dict(input=list(SEEDS)), params=config, prereg=PREREG)
    rows, pairs, repeats = [], [], []
    for index,seed in enumerate(SEEDS):
        p = demand("train",index)
        for load in LOADS:
            case = dict(index=index,seed=seed,load=load)
            before, old_schedule = one(dict(case,arm="legacy"),old,p)
            after, new_schedule = one(dict(case,arm="confirmed"),new,p)
            checks = {k:before[k] == after[k] for k in ("physical_schedule_sha256","scenarios_sha256","vessels_sha256")}
            if not all(checks.values()):
                raise AssertionError(f"Non-information inputs changed: {case} {checks}")
            changed = Counter()
            for a,b in zip(old_schedule,new_schedule):
                for key in set(a)|set(b):
                    if a.get(key) != b.get(key):
                        changed[key] += 1
            assert set(changed) <= ALLOWED
            rows.extend((before,after))
            pairs.append(dict(case=case, unchanged_physics=checks, changed_fields=dict(changed)))
            if index == 0 and load == 3500:
                for arm,info,reference in (("legacy",old,before),("confirmed",new,after)):
                    replay,_ = one(dict(case,arm=arm),info,p)
                    repeats.append(dict(arm=arm,exact=replay == reference))
            save(out/"input-progress.json", dict(finished=len(rows),total=18))
            print(f"inputs {len(rows)}/18", flush=True)
    assert all(r["exact"] for r in repeats)
    save(target, dict(status="PASS", configuration=config, repro=repro, cases=rows, pairs=pairs, repeats=repeats,
        generated_orders=sum(r["case"]["load"] for r in rows),
        interpretation="Input/adapter correctness only; no scheduling policy performance or PPO training."))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir",type=Path,default=OUT)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args();run(a.calibration_dir,a.out)
