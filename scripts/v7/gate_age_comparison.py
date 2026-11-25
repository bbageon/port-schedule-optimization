"""Score the gate-age experiment on common ledgers. Wiring replays keep the initial policy fixed."""
import argparse
import gzip
import json
from math import ceil, fsum, isclose
from pathlib import Path

import torch

import short_window_train as bp
from after_bp_planned_times import save
from joint_turnaround_preflight import read
from planned_time_comparison import read_month
from short_window_samples import anchors_from
from yard_rl.v7.reward.gate_age import GateAgeTruckClock, MAX_SURCHARGE, truck_points
from yard_rl.v7.stage.seed_bundle import file_sha256


OUT = Path("outputs/reports/yr332_gate_age_curve")
CACHE = bp.CACHE / "gate-age"
BASE = Path("outputs/reports/yr332_short_window")
ARMS = ("flat", "curve")


def accounting_rollout(arm):
    bp.configure_gate_age(arm)
    torch.set_num_threads(1)
    archived = read(BASE / "smoke/runs/smoke-0.json")
    if archived["checkpoint_sha256"] != file_sha256(bp.START):
        raise ValueError("Starting policy differs from the reference")
    bp.rollout("accounting", bp.START, archived["sampling_seed"], archived["anchors"], smoke=True)


def check_update(arm):
    bp.configure_gate_age(arm)
    torch.set_num_threads(1)
    anchors = bp.load("accounting", smoke=True)
    if not anchors or any(not isclose(sample.target, -sample.cost / bp.SCALE) for sample in anchors):
        raise ValueError("The learner received a different target")
    model = bp.update(0, ["accounting"], len(anchors), smoke=True)
    evaluated = bp.evaluate(model, smoke=True)
    counts = evaluated["windows"]["all"]
    if evaluated["checkpoint_sha256"] != file_sha256(model) or counts["unfinished_trucks"] or counts["unfinished_ships"]:
        raise ValueError("Evaluation did not finish with the updated policy")
    save(bp.OUT / "smoke/update-check.json",
         dict(arm=arm, samples=len(anchors), checkpoint=evaluated["checkpoint_sha256"],
              trucks=counts["trucks"], ships=counts["ships"], complete=True,
              purpose="Target, optimizer and dispatch wiring on three synthetic days"))


def check():
    dirs = [CACHE / arm / "accounting" for arm in ARMS]
    reference = read(BASE / "smoke/runs/smoke-0.json")
    records = [read(OUT / arm / "smoke/runs/accounting.json") for arm in ARMS]
    for arm, r in zip(ARMS, records):
        if (r["gate_age_arm"] != arm or r["wait_promotion"] or
                any(r[k] != reference[k] for k in ("anchors", "sampling_seed", "checkpoint_sha256"))):
            raise ValueError("Replay settings differ")
    original = bp.CACHE / "short-window/smoke-0"
    manifests = [read(p / "configuration.json")["manifest_sha256"] for p in [original, *dirs]]
    if len(set(manifests)) != 1:
        raise ValueError("Smoke inputs differ")
    metrics = [read_month(p, manifests[0]) for p in [original, *dirs]]
    if len({m["cohort"] for m in metrics}) != 1 or metrics[1] != metrics[2]:
        raise ValueError("Frozen no-promotion policies must replay the same demand and outcomes")
    increases = []
    streams = (anchors_from(p / "decisions.json.gz") for p in dirs)
    for flat, curve in zip(*streams, strict=True):
        delta = curve.pop("cost") - flat.pop("cost")
        if curve != flat or delta < -1e-7:
            raise ValueError("The paired actions changed or the surcharge was negative")
        increases.append(delta)
    if len(increases) != reference["anchors"] or not any(v > 1e-7 for v in increases):
        raise ValueError("The replays must exercise the gate-age surcharge")
    train_file = bp.CACHE / "short-window/wave-0-0/trucks.jsonl.gz"
    with gzip.open(train_file, "rt", encoding="utf8") as stream:
        intervals = []
        for r in map(json.loads, stream):
            if not r["gate"] <= r["block"] <= r["start"] <= r["finish"]:
                raise ValueError("Invalid event order in calibration ledger")
            intervals.append((r["gate"], r["finish"]))
    base = fsum((f-g)/60 for g, f in intervals)
    direct = fsum(truck_points((f-g)/60) for g, f in intervals)
    month_clock = GateAgeTruckClock(intervals)
    if not (isclose(month_clock.total_minutes, direct, rel_tol=1e-10)
            and isclose(direct / base, 1.10, rel_tol=1e-10)):
        raise ValueError("Full training-month clock differs from the calibrated visit totals")
    example = dict(gate_to_block_minutes=50, block_to_finish_minutes=20,
                   total_points=truck_points(70), points_after_block=truck_points(70)-truck_points(50))
    evidence = dict(purpose="Matched frozen-policy wiring checks", manifest=manifests[0],
                    checkpoint=reference["checkpoint_sha256"], arms=["bp", *ARMS], frozen_metrics=metrics,
                    anchors=len(increases), positive_surcharge_windows=sum(v > 1e-7 for v in increases),
                    max_window_surcharge=max(increases), direct_ledger_checks_per_arm=min(64, len(increases)),
                    trucks=records[0]["month"]["trucks"], ships=records[0]["month"]["ships"], example=example,
                    training_month=dict(trucks=len(intervals), sha256=file_sha256(train_file), base_points=base,
                                        curved_points=direct, clock_minus_ledger=month_clock.total_minutes-direct))
    save(OUT / "accounting-check.json", evidence)
    print(json.dumps(evidence, indent=2))


def comparison(reference_only=False):
    manifest = read(bp.setting(False)["test"])["sha256"]
    paths = {"bp": BASE / "evaluation/retrained", "spt": Path("outputs/reports/yr332_unguarded_spt/evaluation")}
    if not reference_only:
        paths.update({arm: OUT / arm / "evaluation/retrained" for arm in ARMS})
    rows, durations = {}, {}
    for name, path in paths.items():
        if name in ARMS:
            result = read(path / "result.json")
            if result.get("gate_age_arm") != name or result.get("wait_promotion") is not False:
                raise ValueError(f"Evaluation settings disagree with {name}")
        score = read_month(path, manifest)
        times, by_flow = {}, {}
        with gzip.open(path / "trucks.jsonl.gz", "rt", encoding="utf8") as f:
            for r in map(json.loads, f):
                minutes = (r["finish"] - r["gate"]) / 60
                times[r["job"]] = minutes
                by_flow.setdefault(r["flow"], []).append(minutes)
        if len(times) != sum(map(len, by_flow.values())):
            raise ValueError("Duplicate truck order")
        surcharge = fsum(truck_points(t) - t for t in times.values())
        score.update(surcharge=surcharge, curved_total=score["weighted_minutes"] + surcharge,
                     fraction_over_60=sum(t > 60 for t in times.values()) / len(times),
                     flow_p95={flow: sorted(values)[ceil(.95*len(values))-1] for flow, values in by_flow.items()})
        rows[name], durations[name] = score, times
    if len({r["cohort"] for r in rows.values()}) != 1:
        raise ValueError("Truck and vessel cohorts differ")
    for name, score in rows.items():
        score["paired"] = {
            ref: dict(fraction_trucks_slower=sum(t > durations[ref][j] for j, t in durations[name].items())
                                             / len(durations[name]),
                      improves_both=all(score[k] < rows[ref][k] for k in ("truck_minutes", "ship_hours")))
            for ref in rows if ref != name}
    name = "reference-scores.json" if reference_only else "comparison.json"
    save(OUT / name, dict(manifest=manifest, max_surcharge=MAX_SURCHARGE, arms=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay-flat", "replay-curve", "update-flat", "update-curve",
                                          "check", "reference", "compare"))
    action = parser.parse_args().action
    if action.startswith("replay-"):
        accounting_rollout(action.removeprefix("replay-"))
    elif action.startswith("update-"):
        check_update(action.removeprefix("update-"))
    elif action == "check":
        check()
    else:
        comparison(reference_only=action == "reference")
