"""YR-332-bo: record each port call's remaining quay work beside unchanged frozen decisions.

month_crane_replay.run builds its resolver from the module-level name OnPolicyResolver. This runner substitutes
a subclass that makes exactly the same decisions, reads (never changes) quay crane progress at every
synchronized epoch, and attaches one number to each decision that had alternatives:

    hours of quay work left on every unfinished port call that has a part at the deciding block,
    each call counted by its part with the most moves left (remaining moves x quay crane interval).

Only the current state is read. Ground-truth completion fields are not used.
"""
import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch

import month_crane_replay
from joint_turnaround_preflight import CACHE, read, save as save_json
from joint_turnaround_train import MODEL_SEED, SCALE, START, encode, setting
from month_crane_input import MANIFEST as SMOKE_MANIFEST, OUT as SMOKE_INFO
from yard_rl.v7.crane.on_policy import OnPolicyResolver
from yard_rl.v7.stage.seed_bundle import file_sha256


OUT = Path("outputs/reports/yr332_vessel_progress")
ROOT = CACHE / "vessel-progress"
MONTHS, KEEP, EPOCH_S = 4, 8192, 60
SEEDS = [MODEL_SEED + 6000 + k for k in range(MONTHS)]
SOURCES = ("scripts/v7/vessel_progress_record.py", "scripts/v7/month_crane_replay.py",
           "src/yard_rl/v7/crane/on_policy.py", "src/yard_rl/v7/world/integrated/vessel.py")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, data)


class Calls:
    """Port calls and the blocks that share each one, read from the cargo manifest."""

    def __init__(self, document):
        self.call_of, self.blocks_of = {}, defaultdict(set)
        for rows in document["vessels_by_day"].values():
            for row in rows:
                self.call_of[row["key"]] = row["ship"]
                self.blocks_of[row["ship"]].add(row["block"])


CALLS = None


class RecordingResolver(OnPolicyResolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.open_streams, self.work_left_h, self.epochs = {}, {}, []

    def observe_global_epoch(self, terminal, t):
        super().observe_global_epoch(terminal, t)
        if t % EPOCH_S == 0:   # calls are added at midnight, which is also such an epoch
            for sim in terminal.blocks.values():
                for key, process in sim.vessels.items():
                    if not process.done:
                        self.open_streams.setdefault(key, process)
        left = {}
        for key, process in list(self.open_streams.items()):
            if process.done:
                del self.open_streams[key]
                continue
            call, hours = CALLS.call_of[key], process.remaining_service_time_s() / 3600
            if call not in left or hours > left[call][0]:
                left[call] = (hours, key)
        per_block = defaultdict(float)
        for call, (hours, _) in left.items():
            for block in CALLS.blocks_of[call]:
                per_block[block] += hours
        self.work_left_h = per_block
        if t % EPOCH_S == 0:
            self.epochs.append([t, sorted([call, hours, key] for call, (hours, key) in left.items())])

    def resolve(self, sim, decision, gen_by_crane):
        before = len(self.samples)
        assignment = super().resolve(sim, decision, gen_by_crane)
        if len(self.samples) > before:
            self.samples[-1].work_left_h = self.work_left_h.get(self.global_block_ids[id(sim)], 0.0)
        return assignment


def install(manifest):
    global CALLS
    with gzip.open(manifest, "rt", encoding="utf8") as stream:
        CALLS = Calls(json.load(stream))
    month_crane_replay.OnPolicyResolver = RecordingResolver


def smoke():
    """The recorder must leave every decision of the archived three-day replay unchanged."""
    info = read(SMOKE_INFO / "smoke-input.json")
    install(SMOKE_MANIFEST.with_name(info["path"]))
    reference = SMOKE_INFO / "smoke" / "trained"
    CACHE.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix="bo-smoke-") as scratch:
        out = month_crane_replay.run("trained", True, output_root=Path(scratch), check_turnaround_targets=True)
        replayed = Path(scratch) / "smoke" / "trained"
        with gzip.open(reference / "trucks.jsonl.gz", "rb") as a, gzip.open(replayed / "trucks.jsonl.gz", "rb") as b:
            trucks = a.read() == b.read()
        ships = read(replayed / "ships.json") == read(reference / "ships.json")
    resolver = out["resolver"]
    values = [getattr(s, "work_left_h", None) for s in resolver.samples]
    attached = all(v is not None and math.isfinite(v) and v >= 0 for v in values)
    ended = resolver.epochs[-1][1] == []
    if not (trucks and ships and attached and ended):
        raise RuntimeError(dict(trucks=trucks, ships=ships, attached=attached, ended=ended))
    save(OUT / "smoke-check.json", dict(
        purpose="The recording resolver reproduces the archived three-day frozen replay event for event",
        reference=reference.as_posix(), truck_events_equal=trucks, quay_events_equal=ships,
        decisions_with_number=len(values), positive=sum(v > 0 for v in values), max_hours=max(values),
        epochs=len(resolver.epochs), all_calls_finished_at_end=ended,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    print("smoke check passed", len(values), max(values), flush=True)


def rollout(index):
    started = time.monotonic()
    cfg = setting(False)
    info = read(cfg["train"])
    install(cfg["manifests"] / info["path"])
    folder, seed = ROOT / str(index), SEEDS[index]
    out = month_crane_replay.run("trained", False, info_path=cfg["train"], manifest_dir=cfg["manifests"],
                                 checkpoint=START, folder=folder, stochastic=True, sample_seed=seed,
                                 horizon=cfg["horizon"], target_scale=SCALE)
    resolver = out["resolver"]
    samples = resolver.samples
    if any(not hasattr(s, "work_left_h") for s in samples):
        raise ValueError("A decision has no recorded quay work")
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + 500000))[:KEEP].tolist()
    kept = folder / "samples.json.gz"
    with gzip.open(kept, "wt", encoding="utf8", newline="\n") as stream:
        json.dump(dict(horizon=cfg["horizon"], keep=[dict(**encode(samples[i], cfg["horizon"]),
                                                          work_left_h=samples[i].work_left_h) for i in order]), stream)
    work = folder / "work-left.json.gz"
    with gzip.open(work, "wt", encoding="utf8", newline="\n") as stream:
        json.dump(dict(epoch_s=EPOCH_S, epochs=resolver.epochs), stream)
    save(OUT / "rollouts" / f"rollout-{index}.json", dict(
        index=index, sampling_seed=seed, complete=out["complete"], decisions=resolver.decisions,
        samples=len(samples), kept=len(order), samples_sha256=file_sha256(kept), work_left_sha256=file_sha256(work),
        checkpoint_sha256=file_sha256(START), month=out["windows"]["all"], seconds=time.monotonic() - started))


def campaign():
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Commit the code before a registered run")
    cfg = setting(False)
    save(OUT / "campaign.json", dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources={p: file_sha256(p) for p in SOURCES}, start_model=file_sha256(START),
        train_input=read(cfg["train"])["sha256"], months=MONTHS, sampling_seeds=SEEDS, kept_per_month=KEEP,
        epoch_s=EPOCH_S, horizon_s=cfg["horizon"], collection="stochastic, no weight update"))
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = {k: subprocess.Popen([sys.executable, "-X", "utf8", "scripts/v7/vessel_progress_record.py", "rollout",
                                    "--index", str(k)], stdout=(logs / f"rollout-{k}.log").open("w", encoding="utf8"),
                                   stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": "src"})
               for k in range(MONTHS) if not (OUT / "rollouts" / f"rollout-{k}.json").exists()}
    print("started", sorted(running), flush=True)
    failed = [k for k, process in running.items() if process.wait() != 0]
    if failed:
        raise RuntimeError(f"Failed rollouts {failed}; logs under {logs}")
    print("campaign complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "rollout", "campaign"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"smoke": smoke, "campaign": campaign}.get(args.action, lambda: rollout(args.index))()
