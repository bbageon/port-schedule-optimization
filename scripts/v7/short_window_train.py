"""YR-332-bp: learn from thirty minutes of actual block time only, with as many decisions per update as the noise needs.

Target of a decision at block b and time t = -(block time in (t, t + 30 min]) / 10^3. Block time is YR-332-bm's:
the block's trucks (gate to finish) + 67 x port calls whose part at b is unfinished at t, until the whole call
finishes. Nothing after the window is added. Decisions per update come from the YR-332-bm noise measurement.

  smoke   three short days: target checked against the ledger, one update, one evaluation run
  train   four updates on the registered training month, then the unseen month
"""
import argparse
from collections import defaultdict
import gzip
import json
from math import ceil, fsum, isclose
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

import month_crane_replay
from joint_turnaround_preflight import CACHE, read, save as save_json
from joint_turnaround_train import MODEL_SEED, START, decode, encode, setting
from local_return_scope import port_calls_in_window, time_share, window
from short_window_samples import anchors_from
from value_stability import restore
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.value_normalization import fit_value_head, normalized_ppo_update, restart_value_head
from yard_rl.v7.reward.joint_turnaround import JointTurnaroundClock, VESSEL_TIME_WEIGHT
from yard_rl.v7.stage.seed_bundle import file_sha256


OUT = Path("outputs/reports/yr332_short_window")
ROOT = CACHE / "short-window"
BM = Path("outputs/reports/yr332_local_return_scope/summary.json")
# Thirty-minute block time is a few thousand weighted minutes. In units of 10^6 the targets are ~0.002 and the
# output-preserving normalization right after the value head restart inflates the head by hundreds (wiring check).
WINDOW_H, SCALE = .5, 1e3
MONTHS, UPDATES = 4, 4
SOURCES = ("scripts/v7/short_window_train.py", "scripts/v7/short_window_samples.py",
           "scripts/v7/local_return_scope.py", "scripts/v7/month_crane_replay.py",
           "src/yard_rl/v7/crane/on_policy.py", "src/yard_rl/v7/crane/actor_critic.py",
           "src/yard_rl/v7/crane/value_normalization.py", str(BM).replace("\\", "/"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, data)


def decisions_per_update():
    measured = read(BM)["noise"][f"block/{WINDOW_H:g}h"]["decisions_for_equal_signal"]
    return max(2048, ceil(measured / 256) * 256)


def seed_of(update, index):
    return MODEL_SEED + 10000 + 16 * update + index


def model_path(update, smoke=False):
    return START if update == 0 else ROOT / ("smoke-models" if smoke else "models") / f"update-{update - 1}.pt"


def calls_and_jobs(manifest):
    with gzip.open(manifest, "rt", encoding="utf8") as stream:
        document = json.load(stream)
    order = defaultdict(list)
    for rows in document["vessels_by_day"].values():   # the replay lists stream finishes in this order
        for row in rows:
            order[row["ship"]].append(row["block"])
    return {e["job_id"]: e["block"] for e in document["schedule"]}, order


def rollout(tag, checkpoint_path, seed, keep, smoke=False):
    started = time.monotonic()
    cfg = setting(smoke)
    jobs, order = calls_and_jobs(cfg["manifests"] / read(cfg["train"])["path"])
    folder = ROOT / tag
    out = month_crane_replay.run("trained", False, info_path=cfg["train"], manifest_dir=cfg["manifests"],
                                 checkpoint=checkpoint_path, folder=folder, stochastic=True, sample_seed=seed,
                                 horizon=cfg["horizon"], target_scale=SCALE)
    if not out["complete"]:
        raise RuntimeError("A training month left work unfinished")
    trucks, ships = defaultdict(list), defaultdict(list)
    for r in out["trucks"]:
        trucks[jobs[r["job"]]].append((r["gate"], r["finish"]))
    for r in out["ships"]:
        where = order[r["ship"]]
        if len(where) != len(r["stream_finishes"]) or r["finish"] != max(r["stream_finishes"]):
            raise ValueError(f"Stream ledger mismatch for {r['ship']}")
        own = defaultdict(float)
        for b, f in zip(where, r["stream_finishes"]):
            own[b] = max(own[b], f)
        for b, f in own.items():
            ships[b].append((r["start"], r["finish"], f, f == r["finish"]))
    clocks = {b: JointTurnaroundClock(rows, []) for b, rows in trucks.items()}
    empty = JointTurnaroundClock([], [])
    samples = out["resolver"].samples
    anchors = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + 500000))[:keep].tolist()
    rows = []
    for i in anchors:
        obs = samples[i].global_observation
        b, t = obs["decision_block"], obs["decision_time"]
        cost = window(clocks.get(b, empty), t, WINDOW_H) + port_calls_in_window(ships[b], t, WINDOW_H)
        rows.append((i, b, t, cost))
    for i, b, t, cost in random.Random(seed).sample(rows, min(64, len(rows))):   # direct sums, independent of the clock
        end = t + 3600 * WINDOW_H
        direct = fsum(max(0., min(f, end) - max(g, t)) for g, f in trucks[b])
        direct += VESSEL_TIME_WEIGHT * fsum(max(0., min(f, end) - max(g, t)) for g, f, own, _ in ships[b] if own > t)
        if not isclose(direct / 60, cost, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Block time in the window disagrees with the ledger")
    path = folder / "decisions.json.gz"
    with gzip.open(path, "wt", encoding="utf8", newline="\n") as stream:
        json.dump(dict(horizon=cfg["horizon"], window_h=WINDOW_H, scale=SCALE,
                       anchors=[dict(**encode(samples[i], cfg["horizon"]), block=b, cost=cost) for i, b, _, cost in rows]),
                  stream)
    record = dict(tag=tag, sampling_seed=seed, checkpoint=Path(checkpoint_path).as_posix(),
                  checkpoint_sha256=file_sha256(checkpoint_path), decisions=out["resolver"].decisions,
                  nontrivial=len(samples), anchors=len(rows), decisions_sha256=file_sha256(path),
                  month=out["windows"]["all"], seconds=time.monotonic() - started)
    save((OUT / "smoke" if smoke else OUT) / "runs" / f"{tag}.json", record)
    return record


def load(tag, smoke=False):
    record = read((OUT / "smoke" if smoke else OUT) / "runs" / f"{tag}.json")
    path = ROOT / tag / "decisions.json.gz"
    if file_sha256(path) != record["decisions_sha256"]:
        raise ValueError(f"Stored decisions of {tag} changed")
    anchors = []
    for x in anchors_from(path):
        s = decode(x)
        # The critic input and decision time are already in the state. The full audit
        # snapshot stays in the immutable gzip; PPO does not consume its per-block map.
        s.global_observation = None
        s.block, s.cost = x["block"], x["cost"]
        s.target = -s.cost / SCALE
        anchors.append(s)
    if len(anchors) != record["anchors"]:
        raise ValueError(f"Incomplete anchors for {tag}")
    return anchors


def values(net, samples):
    out = []
    with torch.no_grad():
        for b in range(0, len(samples), 64):
            out += net(pack_states([s.state for s in samples[b:b+64]]))[1].tolist()
    return out


def note(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def update(u, tags, needed, smoke=False):
    note(f"update {u}: loading decisions")
    anchors = [s for tag in tags for s in load(tag, smoke)]
    note(f"update {u}: {len(anchors)} decisions loaded")
    net, optimizer, _ = restore(model_path(u, smoke))
    report = dict(update=u, tags=list(tags), anchors=len(anchors))
    if u == 0:
        # The old value head predicted another quantity: start it again, fit it alone, keep actions fixed.
        probe = anchors[:256]
        with torch.no_grad():
            before = net(pack_states([s.state for s in probe]))[0]
        optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
        restart_value_head(net, seed=MODEL_SEED + 3000)
        report["value_fit"] = fit_value_head(net, anchors, seed=MODEL_SEED + 3001)
        note(f"update {u}: value head fitted")
        with torch.no_grad():
            if not torch.equal(before, net(pack_states([s.state for s in probe]))[0]):
                raise ValueError("Fitting the value head moved an action score")
        for s, v in zip(anchors, values(net, anchors)):
            s.old_value = v
    advantage = np.array([s.target - s.old_value for s in anchors]) * SCALE
    horizon = setting(smoke)["horizon"]
    times = np.array([s.state.time_fraction * horizon for s in anchors])
    report["noise_weighted_minutes"] = float(np.sqrt(np.mean(advantage ** 2)))
    report["time_share"] = time_share(advantage, times)[0]
    note(f"update {u}: policy update starts")
    report["fit"] = normalized_ppo_update(net, optimizer, anchors, seed=MODEL_SEED + 1100 + u,
                                          max_samples=needed, epochs=4, batch_size=64)
    note(f"update {u}: policy update done")
    path = ROOT / ("smoke-models" if smoke else "models") / f"update-{u}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model_config=net.config, state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), arm="short_window", seed=MODEL_SEED, iteration=u, return_scale=SCALE,
                    window_h=WINDOW_H, return_unit=f"block truck minutes + {VESSEL_TIME_WEIGHT} x port-call minutes"), tmp)
    tmp.replace(path)
    report["checkpoint_sha256"] = file_sha256(path)
    save((OUT / "smoke" if smoke else OUT) / "training" / f"fit-{u}.json", report)
    return path


def evaluate(checkpoint_path, smoke=False):
    cfg = setting(smoke)
    folder = (ROOT / "smoke-evaluation") if smoke else (OUT / "evaluation" / "retrained")
    month_crane_replay.run("trained", False, info_path=cfg["test"], manifest_dir=cfg["manifests"],
                           checkpoint=checkpoint_path, folder=folder, horizon=cfg["horizon"])
    return read(folder / "result.json")


def smoke():
    tags = ["smoke-0", "smoke-1"]
    records = [rollout(tag, START, MODEL_SEED + 7800 + k, 1024, smoke=True) for k, tag in enumerate(tags)]
    final = update(0, tags, 1024, smoke=True)
    evaluated = evaluate(final, smoke=True)
    save(OUT / "smoke" / "smoke-check.json", dict(
        purpose="Wiring check on three short days; numbers are not results", rollouts=records,
        update=read(OUT / "smoke" / "training" / "fit-0.json"),
        evaluation=dict(complete=evaluated["windows"]["all"]["unfinished_trucks"] == 0, windows=evaluated["windows"]),
        decisions_per_update=decisions_per_update(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    print("smoke check passed", flush=True)


def parallel(jobs):
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = {name: subprocess.Popen([sys.executable, "-X", "utf8", "scripts/v7/short_window_train.py", *arguments],
                                      stdout=(logs / f"{name}.log").open("w", encoding="utf8"), stderr=subprocess.STDOUT,
                                      env={**os.environ, "PYTHONPATH": "src"})
               for name, (done, arguments) in jobs.items() if not done.exists()}
    print(time.strftime("%H:%M:%S"), "started", sorted(running), flush=True)
    failed = [name for name, process in running.items() if process.wait() != 0]
    if failed:
        raise RuntimeError(f"Failed {failed}; logs under {logs}")
    print(time.strftime("%H:%M:%S"), "finished", sorted(running), flush=True)


def train_campaign():
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Commit the code before a registered run")
    needed = decisions_per_update()
    first = OUT / "train-campaign.json"
    save(first if not first.exists() else OUT / f"train-campaign-resumed-{time.strftime('%m%d-%H%M')}.json", dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources={p: file_sha256(p) for p in SOURCES}, start_model=file_sha256(START), decisions_per_update=needed,
        anchors_per_month=needed // MONTHS, window_h=WINDOW_H, updates=UPDATES, months_per_update=MONTHS))
    for u in range(UPDATES):
        source = model_path(u)
        tags = [f"wave-{u}-{k}" for k in range(MONTHS)]
        parallel({tag: (OUT / "runs" / f"{tag}.json",
                        ["rollout", "--tag", tag, "--checkpoint", Path(source).as_posix(), "--seed", str(seed_of(u, k)),
                         "--keep", str(needed // MONTHS)])
                  for k, tag in enumerate(tags)})
        if not (ROOT / "models" / f"update-{u}.pt").exists():
            update(u, tags, needed)
            print(time.strftime("%H:%M:%S"), f"update {u + 1}/{UPDATES} applied", flush=True)
    evaluate(ROOT / "models" / f"update-{UPDATES - 1}.pt")
    print(time.strftime("%H:%M:%S"), "campaign complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train-campaign", "rollout"))
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "rollout":
        rollout(args.tag, Path(args.checkpoint), args.seed, args.keep)
    else:
        {"smoke": smoke, "train-campaign": train_campaign}[args.action]()
