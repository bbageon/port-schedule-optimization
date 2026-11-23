"""Frozen crane decisions in one continuous fixed-cargo world, without market trades."""
import argparse
from collections import defaultdict
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from month_crane_input import MANIFEST, OUT
from value_stability import restore
from vessel_turnaround_experiment import OUT as TRAINING, save
from yard_rl.v7.crane.on_policy import OnPolicyResolver
from yard_rl.v7.information_contract import announcement_time
from yard_rl.v7.reward.krw import rehandle_krw, truck_wait_krw, vessel_krw_per_hour, yc_move_krw
from yard_rl.v7.stage.cargo_input import restore_input
from yard_rl.v7.stage.cargo_runtime import CargoBlock, CargoTerminal
from yard_rl.v7.stage.episode import INFO_LEVEL, _rule_policy, rehandles_of, yc_empty_travel_s
from yard_rl.v7.stage.month import DAY_S, DayPlan, prune_completed, retire_done_vessels
from yard_rl.v7.stage.month_engine import VESSEL_DEADLINE_MULT, inject_vessel
from yard_rl.v7.stage.orders import V3Announcer
from yard_rl.v7.stage.seed_bundle import file_sha256
from yard_rl.v7.world.integrated.profiles import build_h21_profile
from yard_rl.v7.world.integrated.terminal_stream import ensure_time_ledger
from yard_rl.v7.world.integrated.yard_layout import terminal_layout


def initial_world(document, days, end):
    built = restore_input(document, seed=document["seed"], days=days, lead_mode=document["lead_mode"])
    profile = build_h21_profile()
    horizon = days[-1].t1
    blocks = {}
    for bid, original in built["day0"]["scenarios"].items():
        scenario = dataclasses.replace(original, horizon_s=horizon, drain_window_s=end-horizon)
        sim = ensure_time_ledger(CargoBlock(profile, scenario, yard_handover_cap=profile.transfer.n_units))
        sim.info_level = INFO_LEVEL
        sim.dispatch_cache.rehandle_slots = True
        sim.decision_probe_shortcircuit = True
        blocks[bid] = sim
    epochs = set(range(0, int(end)+1, 60))
    epochs.update(announcement_time(e) for e in document["schedule"])
    terminal = CargoTerminal(blocks, document=document, layout=terminal_layout(),
                             preserve_external_demand=True, extra_review_epochs=tuple(sorted(epochs)))
    return terminal


def write_lines(path, rows):
    with gzip.open(path, "wt", encoding="utf8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False)+"\n")


def run(arm, smoke, *, output_root=OUT, check_turnaround_targets=False):
    started = time.monotonic()
    info = json.loads((OUT/("smoke-input.json" if smoke else "input.json")).read_text("utf8"))
    manifest = MANIFEST.with_name(info["path"])
    if file_sha256(manifest) != info["sha256"]:
        raise ValueError("Manifest checksum mismatch")
    with gzip.open(manifest, "rt", encoding="utf8") as stream:
        document = json.load(stream)
    days = [DayPlan(**{k:v for k,v in d.items() if k != "train"}) for d in info["days"]]
    n_days, seed = len(days), document["seed"]
    end = (n_days+5)*DAY_S
    folder = output_root/("smoke" if smoke else "runs")/arm
    folder.mkdir(parents=True, exist_ok=True)
    terminal = initial_world(document, days, end)
    ann = V3Announcer(document["schedule"], end_s=end, resolve_entry=terminal.resolve_entry)
    streams, trucks, checkpoints, daily = {}, {}, {}, []
    checkpoint = TRAINING/"models"/"9951101"/"update-5.pt"
    resolver, net = None, None
    if arm == "trained":
        net = restore(checkpoint)[0].eval()
        resolver = OnPolicyResolver(net, stochastic=False, collect=check_turnaround_targets, horizon=DAY_S,
                                    collect_until=end, global_context_mode="global")
        resolver.observe_global_epoch(terminal, 0.)
    policy, errors = _rule_policy("RL_CRANE" if net is not None else arm, seed=seed,
        crane_net=net, candidate_cap=None, reuse_plans=True, joint_resolver=resolver)
    meta = dict(arm=arm, seed=seed, manifest_sha256=info["sha256"], days=n_days,
                checkpoint_sha256=file_sha256(checkpoint) if net is not None else None,
                code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    save(folder/"configuration.json", meta)
    groups = defaultdict(list)
    for rows in document["vessels_by_day"].values():
        for row in rows:
            groups[row["ship"]].append(row)
    for ship, rows in groups.items():
        if len({(r["gt"],r["start_s"],r["day"]) for r in rows}) != 1 or len(rows) != rows[0]["sts"]:
            raise ValueError(f"Incomplete ship manifest for {ship}")
    for e in document["schedule"]:
        if e["job_id"] in trucks:
            raise ValueError("Duplicate external order")
        trucks[e["job_id"]] = None

    def snapshot(t):
        for sim in terminal.blocks.values():
            for jid, tt in sim.time_ledger.records.items():
                if jid in trucks:
                    trucks[jid] = tt
        checkpoints[t] = (yc_empty_travel_s(terminal), rehandles_of(terminal))
        active = [v for v in trucks.values() if v and v.gate_in <= t]
        completed = sum(v.job_done is not None and v.job_done <= t for v in active)
        cargo = terminal.cargo_report()
        if cargo["physical_inventory"] != cargo["expected_inventory"]:
            raise ValueError("Daily physical inventory mismatch")
        row = dict(day=int(t//DAY_S), elapsed_s=time.monotonic()-started, admitted=ann.n_admitted,
            entered=len(active), done=completed, unfinished=len(active)-completed,
            quay_streams_open=len(streams), quay_streams_done=sum(
                v.truth.actual_completion_s is not None for v in streams.values()),
            cargo=cargo, blocks={b:len(s.stacks.containers) for b,s in terminal.blocks.items()})
        daily.append(row)
        save(folder/"progress.json", dict(**meta, daily=daily))
        print(arm,row["day"],"days",row["done"],"completed",round(row["elapsed_s"]),flush=True)

    def review(m,t):
        ann.review(m,t)
        if ann.n_skipped:
            raise ValueError(f"Missing external demand: {ann.skips[:2]}")
        if t % DAY_S == 0:
            d = int(t//DAY_S)
            snapshot(t)
            if t:
                prune_completed(m,t)
                retire_done_vessels(m,{},t=t)
            for r in document["vessels_by_day"].get(str(d), []):
                admission = inject_vessel(m,r["block"],r,key=r["key"],
                    deadline_mult=VESSEL_DEADLINE_MULT,size_seed=f"v3:month:{seed}:{r['key']}")
                if admission.moves != r["moves"]:
                    raise ValueError("Truncated quay work")
                streams[r["key"]] = m.blocks[r["block"]].vessels[r["key"]]
        if resolver is not None:
            resolver.observe_global_epoch(m,t)
            # Only the latest synchronized snapshot is consumed by frozen inference.
            del resolver.global_snapshots[:-1]
            resolver.global_decisions.clear()
            resolver.latencies.clear()

    with torch.inference_mode():
        terminal.run(policy,review_fn=review)
    if end not in checkpoints:
        snapshot(end)
    if errors["n"] or ann.n_admitted != len(trucks) or any(v is None for v in trucks.values()):
        raise ValueError("Incomplete admission, event records, or failed dispatch")
    for sim in terminal.blocks.values():
        sim.check_invariants()
    truck_rows = []
    for e in document["schedule"]:
        tt = trucks[e["job_id"]]
        if tt.gate_in != e["arrival_s"] or tt.job_done is not None and not (
                tt.gate_in <= tt.block_arrival <= tt.service_start <= tt.job_done <= end):
            raise ValueError(f"Invalid clock for {e['job_id']}")
        duration = (tt.job_done if tt.job_done is not None else end)-tt.gate_in
        truck_rows.append(dict(job=e["job_id"], day=int(tt.gate_in//DAY_S), flow=e["flow"],
            gate=tt.gate_in,block=tt.block_arrival,start=tt.service_start,finish=tt.job_done,
            seconds=duration,cost=truck_wait_krw(duration)))
    ships = []
    for ship, rows in groups.items():
        finishes = [streams[r["key"]].truth.actual_completion_s for r in rows]
        finish = max(finishes) if all(t is not None for t in finishes) else None
        r = rows[0]
        seconds = (finish if finish is not None else end)-r["start_s"]
        ships.append(dict(ship=ship, day=r["day"], start=r["start_s"], gt=r["gt"], finish=finish,
            stream_finishes=finishes,seconds=seconds,cost=seconds/3600*vessel_krw_per_hour(r["gt"])))
    write_lines(folder/"trucks.jsonl.gz",truck_rows)
    save(folder/"ships.json",ships)
    if check_turnaround_targets:
        from math import fsum, isclose
        from yard_rl.v7.reward.joint_turnaround import JointTurnaroundClock, VESSEL_TIME_WEIGHT
        clock = JointTurnaroundClock(((r["gate"], r["finish"]) for r in truck_rows),
                                     ((r["start"], r["finish"]) for r in ships))
        check = resolver.finish_turnaround(clock, scale=1.0)
        comparisons = []
        for sample in resolver.samples[::max(1, len(resolver.samples)//8)]:
            t = sample.state.time_fraction * resolver.horizon
            remaining = fsum(max(0, r["finish"] - max(r["gate"], t)) for r in truck_rows)
            remaining += VESSEL_TIME_WEIGHT * fsum(max(0, r["finish"] - max(r["start"], t)) for r in ships)
            if not isclose(sample.target, -remaining / 60, rel_tol=1e-11, abs_tol=1e-7):
                raise ValueError("A decision target disagrees with the remaining elapsed time")
            comparisons.append(dict(t=t, target=sample.target, expected=-remaining / 60))
        check["sampled_decisions"] = comparisons
        save(folder/"target-check.json", check)
    if not smoke:
        future = json.loads(Path("outputs/reports/yr332_month_supply/future-supply.json").read_text("utf8"))
        write_lines(folder/"cargo-waits.jsonl.gz",(dict(container=e["container"], job=e["demand"]["job"],
            ready=terminal.ready_times.get(e["container"]),out=terminal.exit_times.get(e["container"]),
            block=trucks[e["demand"]["job"]].block_arrival) for e in future))
    result = {}
    for name,lo,hi in (("primary",1,n_days-1),("all",0,n_days)):
        a = [r for r in truck_rows if lo <= r["day"] < hi]
        b = [r for r in ships if lo <= r["day"] < hi]
        counters = [y-x for x,y in zip(checkpoints[lo*DAY_S],checkpoints[hi*DAY_S])]
        costs = dict(truck=sum(r["cost"] for r in a),ship=sum(r["cost"] for r in b),
                     empty=yc_move_krw(counters[0]),rehandles=rehandle_krw(counters[1]))
        result[name] = dict(trucks=len(a),ships=len(b),unfinished_trucks=sum(r["finish"] is None for r in a),
            unfinished_ships=sum(r["finish"] is None for r in b),costs=costs, total=sum(costs.values()),
            truck_minutes=sum(r["seconds"] for r in a)/60/len(a),
            ship_hours=sum(r["seconds"] for r in b)/3600/len(b))
    save(folder/"result.json",dict(**meta,windows=result, seconds=time.monotonic()-started,
        cargo=terminal.cargo_report(), admitted=ann.n_admitted,skipped=ann.n_skipped,
        model_decisions=resolver.decisions if resolver else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm",choices=["trained","SF_SPT","LWKR"])
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.arm,args.smoke)
