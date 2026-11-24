"""Held-out month and paired first-action audit for the planned-time policies."""
import gzip
import hashlib
import json
import math
from pathlib import Path

import month_crane_replay
from joint_turnaround_preflight import CACHE, OUT as BL, read
from joint_turnaround_train import setting
from short_window_train import save
from yard_rl.v7.crane.on_policy import OnPolicyResolver
from yard_rl.v7.crane.planned_times import ARMS
from yard_rl.v7.reward.joint_turnaround import VESSEL_TIME_WEIGHT
from yard_rl.v7.stage.seed_bundle import file_sha256
from yard_rl.v7.world.integrated.baselines import ServiceFirstSPTPreference, _wait_of
from yard_rl.v7.world.integrated.resolver import CentralResolver


OUT = Path("outputs/reports/yr332_planned_times")
BP = Path("outputs/reports/yr332_short_window")
DAYS = (3, 10, 17, 24)   # 1-based entry days, fixed before running


def read_month(folder, manifest):
    run = read(folder / "result.json")
    r = run["windows"]["all"]
    if run["manifest_sha256"] != manifest or r["unfinished_ships"] or r["unfinished_trucks"]:
        raise ValueError(f"Input mismatch or incomplete test replay in {folder}")
    digest = hashlib.sha256()
    durations = []
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as f:
        for line in f:
            a = json.loads(line)
            digest.update(json.dumps([a["job"], a["gate"]]).encode() + b"\n")
            durations.append((a["finish"] - a["gate"]) / 60)
    ships = read(folder / "ships.json")
    for s in sorted(ships, key=lambda v: v["ship"]):
        digest.update(json.dumps([s["ship"], s["start"]]).encode() + b"\n")
    ship_minutes = math.fsum((s["finish"] - s["start"]) / 60 for s in ships)
    if (len(durations) != r["trucks"] or len(ships) != r["ships"] or
        not math.isclose(math.fsum(durations) / len(durations), r["truck_minutes"]) or
        not math.isclose(ship_minutes / len(ships) / 60, r["ship_hours"])):
        raise ValueError(f"Ledger totals disagree with {folder}")
    return dict(cohort=digest.hexdigest(), truck_minutes=r["truck_minutes"], ship_hours=r["ship_hours"],
                truck_p95=sorted(durations)[math.ceil(.95*len(durations))-1],
                weighted_minutes=math.fsum(durations) + VESSEL_TIME_WEIGHT * ship_minutes,
                coverage=run.get("planned_time_coverage"))


def compare():
    manifest = read(BL / "test-input.json")["sha256"]
    paths = {"rule": BL / "evaluation" / "SF_SPT", "bp": BP / "evaluation" / "retrained"}
    paths.update({arm: OUT / arm / "evaluation" / "retrained" for arm in ARMS})
    rows = {arm: read_month(p, manifest) for arm, p in paths.items()}
    if len({r["cohort"] for r in rows.values()}) != 1:
        raise ValueError("Arms did not receive identical jobs and gate/start times")
    for arm in ARMS:
        r = rows[arm]
        r["beats_rule_on_both"] = all(r[k] < rows["rule"][k] for k in ("truck_minutes", "ship_hours"))
        r["weighted_delta_vs_bp"] = r["weighted_minutes"] - rows["bp"]["weighted_minutes"]
        references = {"eta": (), "etw": (), "both": ("eta", "etw"), "both_need": ("both",)}
        r["delta_vs"] = {other: {k: r[k] - rows[other][k] for k in
                                 ("truck_minutes", "ship_hours", "truck_p95", "weighted_minutes")}
                         for other in ("bp", "rule", *references[arm])}
    save(OUT / "comparison.json", rows)
    return rows


class PairedResolver(OnPolicyResolver):
    """Same deterministic model throughout, with at most one rule action substituted."""
    def __init__(self, *args, days=DAYS, intervention=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.days, self.intervention = days, intervention
        self.events = {}
        self.prefix = hashlib.sha256()
        self.rule = CentralResolver(ServiceFirstSPTPreference())

    def resolve(self, sim, decision, generated):
        chosen = super().resolve(sim, decision, generated)
        day = int(sim.clock // 86400) + 1
        signature = [self.global_block_ids[id(sim)], sim.clock, sorted(self.last_assignment.items())]
        self.prefix.update(json.dumps(signature).encode() + b"\n")
        if day not in self.days or day in self.events:
            return chosen
        alternative = self.rule.resolve(sim, decision, generated)
        rule_ids = {r.crane_id: (r.chosen_candidate_id if r.chosen_candidate_id is not None
                                 else _wait_of(generated[r.crane_id]).candidate_id)
                    for r in alternative.resolutions}
        if rule_ids != self.last_assignment:
            self.events[day] = dict(block=signature[0], time=sim.clock, prefix=self.prefix.hexdigest(),
                                    model=dict(self.last_assignment), rule=rule_ids)
            if day == self.intervention:
                self.last_assignment = rule_ids
                return alternative
        return chosen


def paired(smoke=False):
    from functools import partial
    cfg = setting(smoke)
    arm = "both"
    cache = CACHE / "planned-times" / arm
    model = cache / ("smoke-models/update-0.pt" if smoke else "models/update-3.pt")
    days = (1,) if smoke else DAYS
    info = read(cfg["test"])
    checkpoint = file_sha256(model)
    folder = (cache / "smoke-pairs") if smoke else OUT / "paired"

    def replay(day):
        p = folder / ("model" if day is None else f"day-{day}")
        record = p / "pair.json"
        if record.exists():
            saved = read(record)
            if saved["checkpoint"] != checkpoint or saved["manifest"] != info["sha256"]:
                raise ValueError(f"Different paired replay inputs in {p}")
            return saved
        factory = partial(PairedResolver, days=days if day is None else (day,), intervention=day)
        result = month_crane_replay.run("trained", False, info_path=cfg["test"], manifest_dir=cfg["manifests"],
                                        checkpoint=model, folder=p, horizon=cfg["horizon"], resolver_factory=factory)
        metrics = read_month(p, info["sha256"])
        saved = dict(checkpoint=checkpoint, manifest=info["sha256"], metrics=metrics,
                     events={str(k): v for k, v in result["resolver"].events.items()})
        save(record, saved)
        return saved

    control = replay(None)
    fixed = read_month(cache / "smoke-evaluation" if smoke else OUT / arm / "evaluation/retrained",
                       info["sha256"])
    if control["metrics"] != fixed:
        raise ValueError("The paired control differs from the fixed policy evaluation")
    rows = {}
    for day in days:
        key = str(day)
        if key not in control["events"]:
            rows[key] = dict(disagreement=False)
            continue
        other = replay(day)
        if (other["events"].get(key) != control["events"][key] or
                other["metrics"]["cohort"] != control["metrics"]["cohort"]):
            raise ValueError("The replay did not branch from the same decision and demand")
        rows[key] = dict(disagreement=True, event=control["events"][key],
                         rule_minus_model={k: other["metrics"][k] - control["metrics"][k]
                                           for k in ("truck_minutes", "ship_hours", "weighted_minutes")})
    save(folder / "summary.json", dict(days=rows, checkpoint=checkpoint, manifest=info["sha256"]))
