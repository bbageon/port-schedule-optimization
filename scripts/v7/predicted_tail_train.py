"""YR-332-bn: score each decision by thirty minutes of actual block time plus the value model's estimate of the rest.

Block time = the deciding block's trucks (gate to finish) + 67 x every port call that has a part at the block, until the
whole call finishes. No discount. The value model also reads log(1 + quay work left) on those calls (YR-332-bo).

  smoke     three short days: decisions unchanged, targets match the ledgers, one update and one evaluation run
  measure   four training months with the start model: noise of the new learning signal and decisions per update
  train     four updates, four training months each, then the unseen month (only after `measure` passed)
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
import gzip
import json
import math
from math import ceil, fsum, isclose, sqrt
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

import month_crane_replay
from joint_turnaround_preflight import CACHE, read, save as save_json
from joint_turnaround_train import (MODEL_SEED, START, decode as decode_global, encode as encode_global, setting)
from local_return_scope import gain_reference, time_share
from month_crane_input import OUT as SMOKE_INFO
from value_stability import restore
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.on_policy import OnPolicyResolver
from yard_rl.v7.crane.value_normalization import NormalizedValueNet, fit_value_head, normalized_ppo_update, restart_value_head
from yard_rl.v7.reward.joint_turnaround import JointTurnaroundClock, VESSEL_TIME_WEIGHT
from yard_rl.v7.stage.seed_bundle import file_sha256


OUT = Path("outputs/reports/yr332_predicted_tail")
ROOT = CACHE / "predicted-tail"
WINDOW_S, SCALE, EPOCH_S = 1800, 1e6, 60
MONTHS, UPDATES = 4, 4
MEASURE_SEEDS = [MODEL_SEED + 8000 + k for k in range(MONTHS)]
MEASURE_KEEP = 16384
MIN_DECISIONS, ROUND, MAX_TIME_SHARE = 2048, 256, .20
SOURCES = ("scripts/v7/predicted_tail_train.py", "scripts/v7/month_crane_replay.py",
           "src/yard_rl/v7/crane/on_policy.py", "src/yard_rl/v7/crane/actor_critic.py",
           "src/yard_rl/v7/crane/value_normalization.py", "src/yard_rl/v7/reward/joint_turnaround.py")
PLAIN_RESOLVER = month_crane_replay.OnPolicyResolver


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, data)


def train_seed(update, index):
    return MODEL_SEED + 9000 + 16 * update + index


# ---------------------------------------------------------------- quay work, read at synchronized epochs
class Calls:
    def __init__(self, document):
        self.call_of, self.blocks_of = {}, defaultdict(set)
        for rows in document["vessels_by_day"].values():
            for row in rows:
                self.call_of[row["key"]] = row["ship"]
                self.blocks_of[row["ship"]].add(row["block"])
        self.block_of_job = {e["job_id"]: e["block"] for e in document["schedule"]}


CALLS = None


class TailResolver(OnPolicyResolver):
    """Same decisions as OnPolicyResolver; additionally feeds the critic the quay work left at the block."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.open_streams, self.hours_by_block = {}, {}

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
            left[call] = max(left.get(call, 0.), hours)
        per_block = defaultdict(float)
        for call, hours in left.items():
            for block in CALLS.blocks_of[call]:
                per_block[block] += hours
        self.hours_by_block = per_block

    def critic_vessel_work(self, sim):
        if not getattr(self.net, "vessel_work_context", False):
            return None
        return math.log1p(self.hours_by_block.get(self.global_block_ids[id(sim)], 0.))


def install(manifest):
    global CALLS
    with gzip.open(manifest, "rt", encoding="utf8") as stream:
        CALLS = Calls(json.load(stream))
    month_crane_replay.OnPolicyResolver = TailResolver


# ---------------------------------------------------------------- model
def checkpoint(path, net, optimizer, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model_config=net.config, state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), return_scale=SCALE, window_s=WINDOW_S,
                    return_unit=f"block truck minutes + {VESSEL_TIME_WEIGHT} x port-call minutes", **meta), tmp)
    tmp.replace(path)


def widen():
    """Start model with one more critic-only input; actor and encoder copied, value head started again."""
    path = ROOT / "models" / "start.pt"
    if path.exists():
        return path
    old = restore(START)[0]
    net = NormalizedValueNet(**old.config, vessel_work_context=True)
    missing, unexpected = net.load_state_dict({k: v for k, v in old.state_dict().items()
                                               if not k.startswith("critic.")}, strict=False)
    if unexpected or not all(k.startswith("critic.") for k in missing):
        raise ValueError("Only the value head may differ from the start model")
    restart_value_head(net, seed=MODEL_SEED + 3000)
    checkpoint(path, net.eval(), torch.optim.Adam(net.parameters(), lr=.0003), arm="predicted_tail", iteration=-1,
               start_model_sha256=file_sha256(START))
    return path


def values(net, samples):
    out = []
    with torch.no_grad():
        for b in range(0, len(samples), 64):
            out += net(pack_states([s.state for s in samples[b:b+64]]))[1].tolist()
    return out


# ---------------------------------------------------------------- samples
def encode(sample, horizon):
    return dict(**encode_global(sample, horizon), vessel_work=sample.state.vessel_work)


def decode(obj):
    sample = decode_global(obj)
    sample.state.vessel_work = obj["vessel_work"]
    return sample


def block_clocks(trucks, ships):
    by_block, calls = defaultdict(list), defaultdict(list)
    for r in trucks:
        by_block[CALLS.block_of_job[r["job"]]].append((r["gate"], r["finish"]))
    for r in ships:
        for b in CALLS.blocks_of[r["ship"]]:
            calls[b].append((r["start"], r["finish"]))
    return {b: JointTurnaroundClock(by_block[b], calls[b]) for b in set(by_block) | set(calls)}, by_block, calls


def rollout(tag, checkpoint_path, seed, keep, smoke=False):
    started = time.monotonic()
    cfg = setting(smoke)
    install(cfg["manifests"] / read(cfg["train"])["path"])
    folder = ROOT / tag
    out = month_crane_replay.run("trained", False, info_path=cfg["train"], manifest_dir=cfg["manifests"],
                                 checkpoint=checkpoint_path, folder=folder, stochastic=True, sample_seed=seed,
                                 horizon=cfg["horizon"], target_scale=SCALE)
    if not out["complete"]:
        raise RuntimeError("A training month left work unfinished; its decisions cannot be scored")
    samples = out["resolver"].samples
    clocks, trucks, calls = block_clocks(out["trucks"], out["ships"])
    empty = JointTurnaroundClock([], [])
    sequence = defaultdict(list)
    for i, s in enumerate(samples):
        obs = s.global_observation
        sequence[obs["decision_block"]].append((obs["decision_time"], i))
    for rows in sequence.values():
        rows.sort()
    times = {b: [t for t, _ in rows] for b, rows in sequence.items()}
    anchors = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + 500000))[:keep].tolist()
    rows, successors = [], {}
    for i in anchors:
        obs = samples[i].global_observation
        b, t = obs["decision_block"], obs["decision_time"]
        clock = clocks.get(b, empty)
        k = bisect_left(times[b], t + WINDOW_S)
        if k < len(times[b]) and (k == 0 or times[b][k - 1] < t + WINDOW_S) and times[b][k] >= t + WINDOW_S:
            j, until = sequence[b][k][1], times[b][k]
            successors.setdefault(j, len(successors))
            cost = clock.minutes_at(until) - clock.minutes_at(t)
        elif k == len(times[b]):
            j, until, cost = None, None, clock.total_minutes - clock.minutes_at(t)
        else:
            raise ValueError("Successor search broke the block's decision order")
        rows.append(dict(anchor=i, block=b, t=t, until=until, successor=None if j is None else successors[j],
                         cost=cost, month_left=clock.total_minutes - clock.minutes_at(t)))
    rng = random.Random(seed)
    for r in rng.sample(rows, min(64, len(rows))):   # direct overlap sums, independent of the event clock
        end = r["until"] if r["until"] is not None else float("inf")
        direct = fsum(max(0., min(f, end) - max(g, r["t"])) for g, f in trucks[r["block"]])
        direct += VESSEL_TIME_WEIGHT * fsum(max(0., min(f, end) - max(g, r["t"])) for g, f in calls[r["block"]])
        if not isclose(direct / 60, r["cost"], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Block time in the window disagrees with the ledger")
    order = sorted(successors, key=successors.get)
    path = folder / "decisions.json.gz"
    with gzip.open(path, "wt", encoding="utf8", newline="\n") as stream:
        json.dump(dict(horizon=cfg["horizon"], window_s=WINDOW_S, scale=SCALE,
                       anchors=[dict(**encode(samples[r["anchor"]], cfg["horizon"]),
                                     **{k: r[k] for k in ("block", "until", "successor", "cost", "month_left")})
                                for r in rows],
                       successors=[encode(samples[j], cfg["horizon"]) for j in order]), stream)
    record = dict(tag=tag, sampling_seed=seed, checkpoint=Path(checkpoint_path).as_posix(),
                  checkpoint_sha256=file_sha256(checkpoint_path), decisions=out["resolver"].decisions,
                  nontrivial=len(samples), anchors=len(rows), successors=len(order),
                  without_successor=sum(r["successor"] is None for r in rows), decisions_sha256=file_sha256(path),
                  month=out["windows"]["all"], seconds=time.monotonic() - started)
    save((OUT / "smoke" if smoke else OUT) / "runs" / f"{tag}.json", record)
    return record


def load(tag, smoke=False):
    record = read((OUT / "smoke" if smoke else OUT) / "runs" / f"{tag}.json")
    path = ROOT / tag / "decisions.json.gz"
    if file_sha256(path) != record["decisions_sha256"]:
        raise ValueError(f"Stored decisions of {tag} changed")
    with gzip.open(path, "rt", encoding="utf8") as stream:
        data = json.load(stream)
    successors = [decode(x) for x in data["successors"]]
    anchors = []
    for x in data["anchors"]:
        s = decode(x)
        s.block, s.cost, s.month_left = x["block"], x["cost"], x["month_left"]
        s.next = None if x["successor"] is None else successors[x["successor"]]
        anchors.append(s)
    return anchors, successors, record


def bootstrapped(anchors):
    for s in anchors:
        s.target = -s.cost / SCALE + (s.next.old_value if s.next is not None else 0.)


# ---------------------------------------------------------------- step 2: measure
def measure(tags, smoke=False):
    fit, held = [], []
    nontrivial = []
    for k, tag in enumerate(tags):
        anchors, successors, record = load(tag, smoke)
        nontrivial.append(record["nontrivial"])
        (fit if k < len(tags) // 2 else held).append((anchors, successors))
    net = restore(widen())[0]
    fitting = [s for anchors, _ in fit for s in anchors]
    for s in fitting:
        s.target = -s.month_left / SCALE
    fit_info = fit_value_head(net, fitting, seed=MODEL_SEED + 3001)
    anchors = [s for a, _ in held for s in a]
    for a, successors in held:
        for s, v in zip(successors, values(net, successors)):
            s.old_value = v
    for s, v in zip(anchors, values(net, anchors)):
        s.old_value = v
    bootstrapped(anchors)
    advantage = np.array([s.target - s.old_value for s in anchors]) * SCALE
    monte_carlo = np.array([-s.month_left / SCALE - s.old_value for s in anchors]) * SCALE
    times = np.array([s.state.time_fraction * read_horizon(smoke) for s in anchors])
    gain = gain_reference()
    noise = float(np.sqrt(np.mean(advantage ** 2)))
    ratio = noise / gain["per_decision"]
    needed = max(MIN_DECISIONS, ceil(ratio ** 2 / ROUND) * ROUND)
    share, bins = time_share(advantage, times)
    collected = MONTHS * min(nontrivial)
    checks = dict(time_share=share <= MAX_TIME_SHARE, decisions=needed <= collected)
    result = dict(
        purpose="Noise of the 30-minute-plus-estimate signal with the start model, before any retraining",
        tags=list(tags), fit_decisions=len(fitting), held_out_decisions=len(anchors), value_fit=fit_info,
        noise_weighted_minutes=noise, monte_carlo_noise_weighted_minutes=float(np.sqrt(np.mean(monte_carlo ** 2))),
        actual_part_mean=float(np.mean([s.cost for s in anchors])), time_share=share, chance_share=(bins - 1) / (len(anchors) - 1),
        gain_per_decision=gain["per_decision"], noise_over_gain=ratio, decisions_per_update=needed,
        collected_per_wave=collected, without_successor=sum(s.next is None for s in anchors), checks=checks,
        outcome="retrain" if all(checks.values()) else "close without retraining",
        thresholds=dict(max_time_share=MAX_TIME_SHARE, min_decisions=MIN_DECISIONS, round_to=ROUND))
    save((OUT / "smoke" if smoke else OUT) / "measure.json", result)
    return result


def read_horizon(smoke):
    return setting(smoke)["horizon"]


# ---------------------------------------------------------------- step 3: updates
def model_path(update):
    return widen() if update == 0 else ROOT / "models" / f"update-{update - 1}.pt"


def update(u, tags, needed, smoke=False):
    anchors, successors = [], []
    for tag in tags:
        a, s, _ = load(tag, smoke)
        anchors += a
        successors += s
    net, optimizer, _ = restore(model_path(u))
    report = dict(update=u, tags=list(tags), anchors=len(anchors), successors=len(successors))
    if u == 0:
        # The restarted value head first learns the month-end block time; actions must not move.
        probe = anchors[:256]
        with torch.no_grad():
            before = net(pack_states([s.state for s in probe]))[0]
        for s in anchors:
            s.target = -s.month_left / SCALE
        report["value_fit"] = fit_value_head(net, anchors, seed=MODEL_SEED + 3001)
        with torch.no_grad():
            if not torch.equal(before, net(pack_states([s.state for s in probe]))[0]):
                raise ValueError("Fitting the value head moved an action score")
        for group in (anchors, successors):
            for s, v in zip(group, values(net, group)):
                s.old_value = v
    bootstrapped(anchors)
    advantage = np.array([s.target - s.old_value for s in anchors]) * SCALE
    times = np.array([s.state.time_fraction * read_horizon(smoke) for s in anchors])
    report["noise_weighted_minutes"] = float(np.sqrt(np.mean(advantage ** 2)))
    report["time_share"] = time_share(advantage, times)[0]
    report["fit"] = normalized_ppo_update(net, optimizer, anchors, seed=MODEL_SEED + 1100 + u,
                                          max_samples=needed, epochs=4, batch_size=64)
    path = ROOT / ("smoke-models" if smoke else "models") / f"update-{u}.pt"
    checkpoint(path, net, optimizer, arm="predicted_tail", iteration=u)
    report["checkpoint_sha256"] = file_sha256(path)
    save((OUT / "smoke" if smoke else OUT) / "training" / f"fit-{u}.json", report)
    return path


def evaluate(checkpoint_path, smoke=False):
    cfg = setting(smoke)
    install(cfg["manifests"] / read(cfg["test"])["path"])
    folder = (ROOT / "smoke-evaluation") if smoke else (OUT / "evaluation" / "retrained")
    month_crane_replay.run("trained", False, info_path=cfg["test"], manifest_dir=cfg["manifests"],
                           checkpoint=checkpoint_path, folder=folder, horizon=cfg["horizon"])
    return read(folder / "result.json")


# ---------------------------------------------------------------- step 1: smoke
def smoke():
    month_crane_replay.OnPolicyResolver = PLAIN_RESOLVER
    reference = SMOKE_INFO / "smoke" / "trained"
    CACHE.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix="bn-smoke-") as scratch:
        month_crane_replay.run("trained", True, output_root=Path(scratch), check_turnaround_targets=True)
        with gzip.open(reference / "trucks.jsonl.gz", "rb") as a, \
                gzip.open(Path(scratch) / "smoke" / "trained" / "trucks.jsonl.gz", "rb") as b:
            unchanged = a.read() == b.read()
    if not unchanged:
        raise RuntimeError("The plain resolver no longer reproduces the archived replay")
    start = widen()
    old, new = restore(START)[0], restore(start)[0]
    tags = ["smoke-0", "smoke-1"]
    records = [rollout(tag, start, MODEL_SEED + 7700 + k, 1024, smoke=True) for k, tag in enumerate(tags)]
    anchors, _, _ = load(tags[0], smoke=True)
    with torch.no_grad():
        plain = [replace_work(s.state) for s in anchors[:128]]
        same_actor = torch.equal(old(pack_states(plain))[0], new(pack_states([s.state for s in anchors[:128]]))[0])
    if not same_actor:
        raise RuntimeError("The widened model changed action scores")
    measured = measure(tags, smoke=True)
    final = update(0, tags, 1024, smoke=True)
    evaluated = evaluate(final, smoke=True)
    save(OUT / "smoke" / "smoke-check.json", dict(
        purpose="Wiring check on three short days; numbers are not results",
        plain_resolver_reproduces_archive=unchanged, widened_actor_unchanged=same_actor,
        rollouts=records, measure=measured, update=read(OUT / "smoke" / "training" / "fit-0.json"),
        evaluation=dict(complete=evaluated["windows"]["all"]["unfinished_trucks"] == 0, windows=evaluated["windows"]),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    print("smoke check passed", flush=True)


def replace_work(state):
    from dataclasses import replace
    return replace(state, vessel_work=None)


# ---------------------------------------------------------------- campaigns (detached)
def frozen(stage):
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Commit the code before a registered run")
    save(OUT / f"{stage}-campaign.json", dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources={p: file_sha256(p) for p in SOURCES}, start_model=file_sha256(START)))


def parallel(jobs):
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = {name: subprocess.Popen([sys.executable, "-X", "utf8", "scripts/v7/predicted_tail_train.py", *arguments],
                                      stdout=(logs / f"{name}.log").open("w", encoding="utf8"), stderr=subprocess.STDOUT,
                                      env={**os.environ, "PYTHONPATH": "src"})
               for name, (done, arguments) in jobs.items() if not done.exists()}
    print(time.strftime("%H:%M:%S"), "started", sorted(running), flush=True)
    failed = [name for name, process in running.items() if process.wait() != 0]
    if failed:
        raise RuntimeError(f"Failed {failed}; logs under {logs}")
    print(time.strftime("%H:%M:%S"), "finished", sorted(running), flush=True)


def measure_campaign():
    frozen("measure")
    start = widen()
    tags = [f"measure-{k}" for k in range(MONTHS)]
    parallel({tag: (OUT / "runs" / f"{tag}.json",
                    ["rollout", "--tag", tag, "--checkpoint", start.as_posix(), "--seed", str(seed), "--keep", str(MEASURE_KEEP)])
              for tag, seed in zip(tags, MEASURE_SEEDS)})
    result = measure(tags)
    print(json.dumps({k: v for k, v in result.items() if k != "value_fit"}, indent=1), flush=True)


def train_campaign():
    plan = read(OUT / "measure.json")
    if plan["outcome"] != "retrain":
        raise RuntimeError("Step 2 did not pass; retraining is not registered")
    frozen("train")
    needed = plan["decisions_per_update"]
    for u in range(UPDATES):
        source = model_path(u)
        tags = [f"wave-{u}-{k}" for k in range(MONTHS)]
        parallel({tag: (OUT / "runs" / f"{tag}.json",
                        ["rollout", "--tag", tag, "--checkpoint", source.as_posix(), "--seed", str(train_seed(u, k)),
                         "--keep", str(needed // MONTHS)])
                  for k, tag in enumerate(tags)})
        if not (ROOT / "models" / f"update-{u}.pt").exists():
            update(u, tags, needed)
            print(time.strftime("%H:%M:%S"), f"update {u + 1}/{UPDATES} applied", flush=True)
    evaluate(ROOT / "models" / f"update-{UPDATES - 1}.pt")
    print(time.strftime("%H:%M:%S"), "campaign complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "measure-campaign", "train-campaign", "rollout"))
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "rollout":
        rollout(args.tag, Path(args.checkpoint), args.seed, args.keep)
    else:
        {"smoke": smoke, "measure-campaign": measure_campaign, "train-campaign": train_campaign}[args.action]()
