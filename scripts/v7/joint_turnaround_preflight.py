"""Verify reproducible, disjoint monthly inputs and replay a frozen checkpoint before training."""
import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from month_crane_input import OUT as PREVIOUS_MONTH, SEED as PREVIOUS_SEED, build
from yard_rl.v7.stage.month import plan_month
from yard_rl.v7.stage.seed_bundle import file_sha256


OUT = Path("outputs/reports/yr332_joint_turnaround")
CACHE = Path("build/yr332_joint_turnaround")
MONTHS = {"train": 12104101, "test": 12204201}


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def prepare(split):
    seed = MONTHS[split]
    days = plan_month(seed)
    own_seeds = {day.seed for day in days}
    for other in {PREVIOUS_SEED, *MONTHS.values()} - {seed}:
        if own_seeds.intersection(day.seed for day in plan_month(other)):
            raise ValueError("Overlapping daily seeds")
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{split}.json.gz"
    with TemporaryDirectory(dir=CACHE, prefix=split + "-") as scratch:
        a, b = (Path(scratch) / name / target.name for name in ("first", "second"))
        first, second = build(seed, days, a), build(seed, days, b)
        if first != second or (target.exists() and file_sha256(target) != first["sha256"]):
            raise ValueError("Independent month generation disagrees")
        if not target.exists():
            a.replace(target)
    info = dict(split=split, seed=seed, days=[day.as_dict() for day in days],
                orders=sum(day.load for day in days), independent_builds=2, **first)
    save(OUT / f"{split}-input.json", info)
    print(split, info["orders"], first["content_sha256"], flush=True)


def replay():
    import torch
    from month_crane_replay import run
    torch.set_num_threads(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    reference = PREVIOUS_MONTH / "smoke" / "trained"
    with TemporaryDirectory(dir=CACHE, prefix="smoke-") as scratch:
        run("trained", True, output_root=Path(scratch))
        replayed = Path(scratch) / "smoke" / "trained"
        with gzip.open(reference / "trucks.jsonl.gz", "rb") as left, gzip.open(replayed / "trucks.jsonl.gz", "rb") as right:
            if left.read() != right.read():
                raise ValueError("Frozen replay changed a truck event")
        ships = read(replayed / "ships.json")
        if ships != read(reference / "ships.json"):
            raise ValueError("Frozen replay changed a quay event")
        new, old = (read(folder / "result.json") for folder in (replayed, reference))
        keys = ("admitted", "skipped", "model_decisions", "windows", "cargo",
                "seed", "manifest_sha256", "checkpoint_sha256")
        if any(new[key] != old[key] for key in keys):
            raise ValueError("Frozen replay metadata or results changed")
        save(OUT / "smoke-replay.json", dict(source=reference.as_posix(), trucks=new["admitted"],
             ships=len(ships), decisions=new["model_decisions"], seed=new["seed"],
             manifest_sha256=new["manifest_sha256"], checkpoint_sha256=new["checkpoint_sha256"],
             matched_fields=list(keys), truck_events_equal=True, vessel_events_equal=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[*MONTHS, "replay"])
    action = parser.parse_args().action
    OUT.mkdir(parents=True, exist_ok=True)
    if action == "replay":
        replay()
    else:
        prepare(action)
