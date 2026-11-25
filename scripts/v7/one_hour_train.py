"""YR-332-br: the same short-horizon learning as YR-332-bp with one change, the learning target.

bp scored a decision by the block's next 30 minutes only. br discounts block time with a one-hour time constant and
lets the block's first decision at least 30 minutes later carry the rest with its value estimate:
    target = -(discounted block time from t to t') / 10^3 + exp(-(t' - t) / 1 h) x value estimate at t'
Block time uses the YR-332-bn scope (the block's trucks + 67 x every port call with a part at the block, until the whole
call finishes), so the value is the same quantity at t and at t'. YR-332-bq measured this learning signal.
Everything else is bp's: start model, sampling and anchor seeds, decisions per update, value-head restart, PPO, test
month. Update steps use 16 CPU threads (YR-332-bu). Files that YR-332-bt froze are imported, never changed.

  smoke      three short days: archived replay reproduced, targets match the ledgers, one update, one evaluation
  register   fingerprint every project file and input the smoke used; the campaign refuses to run if one changes
  campaign   four updates on the training month, then the unseen month
  wait-bt    wait until YR-332-bt has finished, then run `campaign`
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
import gzip
import json
import math
from math import ceil, fsum, isclose
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
from joint_turnaround_preflight import CACHE, read
from joint_turnaround_train import MODEL_SEED, START, decode, encode, setting
from local_return_scope import time_share
from month_crane_input import OUT as SMOKE_INFO
from predicted_tail_train import Calls, save
from value_horizon import Discounted
from value_stability import restore
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.update_threads import UPDATE_THREADS, update_threads
from yard_rl.v7.crane.value_normalization import fit_value_head, normalized_ppo_update, restart_value_head
from yard_rl.v7.reward.joint_turnaround import JointTurnaroundClock, VESSEL_TIME_WEIGHT
from yard_rl.v7.stage.seed_bundle import file_sha256


REPO = Path(__file__).resolve().parents[2]
OUT = Path("outputs/reports/yr332_one_hour")
ROOT = CACHE / "one-hour"
BP = Path("outputs/reports/yr332_short_window")
BQ = Path("outputs/reports/yr332_value_horizon/summary.json")
BT_STATUS = Path("outputs/reports/yr332_planned_times/queue-status.json")
REGISTRATION = OUT / "registration.json"
HOURS, WINDOW_S, SCALE = 1, 1800, 1e3
MONTHS, UPDATES = 4, 4


def note(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def decisions_per_update():
    """bp's registered count, provided it covers what bq measured for the one-hour signal."""
    registered = read(BP / "train-campaign.json")["decisions_per_update"]
    needed = ceil(read(BQ)["rows"][f"{HOURS}h"]["decisions_for_equal_signal"] / 256) * 256
    if registered < needed:
        raise ValueError("bp's decisions per update do not cover the one-hour learning signal")
    return registered


def seed_of(update, index):
    return MODEL_SEED + 10000 + 16 * update + index   # bp's sampling seeds


def model_path(update, smoke=False):
    return START if update == 0 else ROOT / ("smoke-models" if smoke else "models") / f"update-{update - 1}.pt"


def checkpoint(path, net, optimizer, update):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model_config=net.config, state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), arm="one_hour", seed=MODEL_SEED, iteration=update,
                    return_scale=SCALE, discount_hours=HOURS, successor_after_s=WINDOW_S,
                    return_unit=f"discounted block truck minutes + {VESSEL_TIME_WEIGHT} x port-call minutes"), tmp)
    tmp.replace(path)


def write_lines(path, rows):
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    tmp.replace(path)


def lines(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for line in stream:
            yield json.loads(line)


def values(net, states):
    with torch.no_grad():
        return net(pack_states(states))[1].tolist()


# ---------------------------------------------------------------- collecting one training month
def block_ledgers(calls, trucks, ships):
    """Per block: its trucks (gate, finish) and every port call with a part there (start, finish)."""
    by_block, held = defaultdict(list), defaultdict(list)
    for r in trucks:
        by_block[calls.block_of_job[r["job"]]].append((r["gate"], r["finish"]))
    for r in ships:
        for b in calls.blocks_of[r["ship"]]:
            held[b].append((r["start"], r["finish"]))
    return by_block, held


def rollout(tag, checkpoint_path, seed, keep, smoke=False):
    started = time.monotonic()
    cfg = setting(smoke)
    with gzip.open(cfg["manifests"] / read(cfg["train"])["path"], "rt", encoding="utf8") as stream:
        calls = Calls(json.load(stream))
    folder = ROOT / tag
    out = month_crane_replay.run("trained", False, info_path=cfg["train"], manifest_dir=cfg["manifests"],
                                 checkpoint=checkpoint_path, folder=folder, stochastic=True, sample_seed=seed,
                                 horizon=cfg["horizon"], target_scale=SCALE)
    if not out["complete"]:
        raise RuntimeError("A training month left work unfinished; its decisions cannot be scored")
    samples = out["resolver"].samples
    trucks, held = block_ledgers(calls, out["trucks"], out["ships"])
    value = {b: Discounted(JointTurnaroundClock(trucks[b], held[b]), HOURS) for b in set(trucks) | set(held)}
    nothing = Discounted(JointTurnaroundClock([], []), HOURS)
    sequence = defaultdict(list)
    for i, s in enumerate(samples):
        obs = s.global_observation
        sequence[obs["decision_block"]].append((obs["decision_time"], i))
    for rows in sequence.values():
        rows.sort()
    times = {b: [t for t, _ in rows] for b, rows in sequence.items()}
    # bp's anchor rule: with bp's model and seed the first wave picks the same decisions
    anchors = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + 500000))[:keep].tolist()
    tc, rows, successors = 3600. * HOURS, [], {}
    for i in anchors:
        obs = samples[i].global_observation
        b, t = obs["decision_block"], obs["decision_time"]
        d = value.get(b, nothing)
        now = d.at(t)
        k = bisect_left(times[b], t + WINDOW_S)
        if k < len(times[b]):
            j, until = sequence[b][k][1], times[b][k]
            successors.setdefault(j, len(successors))
            decay = math.exp(-(until - t) / tc)
            cost = now - decay * d.at(until)
        else:   # no later decision at the block: the whole discounted remainder is actual time
            j, until, decay, cost = None, None, 0., now
        rows.append(dict(anchor=i, block=b, t=t, until=until, successor=None if j is None else successors[j],
                         decay=decay, cost=cost, value_now=now))

    def direct(intervals, t, end):
        return fsum(tc * (math.exp(-(max(g, t) - t) / tc) - math.exp(-(min(f, end) - t) / tc))
                    for g, f in intervals if min(f, end) > max(g, t))

    for r in random.Random(seed).sample(rows, min(64, len(rows))):   # direct sums over the ledgers, no event clock
        for end, stored in ((math.inf if r["until"] is None else r["until"], r["cost"]), (math.inf, r["value_now"])):
            whole = (direct(trucks[r["block"]], r["t"], end)
                     + VESSEL_TIME_WEIGHT * direct(held[r["block"]], r["t"], end)) / 60
            if not isclose(whole, stored, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError("Discounted block time disagrees with the ledger")
    order = sorted(successors, key=successors.get)
    keys = ("block", "successor", "decay", "cost", "value_now")
    write_lines(folder / "anchors.jsonl.gz",
                (dict(**encode(samples[r["anchor"]], cfg["horizon"]), **{k: r[k] for k in keys}) for r in rows))
    write_lines(folder / "successors.jsonl.gz", (encode(samples[j], cfg["horizon"]) for j in order))
    record = dict(tag=tag, sampling_seed=seed, checkpoint=Path(checkpoint_path).as_posix(),
                  checkpoint_sha256=file_sha256(checkpoint_path), decisions=out["resolver"].decisions,
                  nontrivial=len(samples), anchors=len(rows), successors=len(order),
                  without_successor=sum(r["successor"] is None for r in rows),
                  anchors_sha256=file_sha256(folder / "anchors.jsonl.gz"),
                  successors_sha256=file_sha256(folder / "successors.jsonl.gz"),
                  month=out["windows"]["all"], seconds=time.monotonic() - started)
    save((OUT / "smoke" if smoke else OUT) / "runs" / f"{tag}.json", record)
    return record


# ---------------------------------------------------------------- one update
def load(tag, smoke=False):
    """Stream one month's anchors. The audit snapshot stays in the file; PPO does not read it."""
    record = read((OUT / "smoke" if smoke else OUT) / "runs" / f"{tag}.json")
    for name in ("anchors", "successors"):
        if file_sha256(ROOT / tag / f"{name}.jsonl.gz") != record[f"{name}_sha256"]:
            raise ValueError(f"Stored {name} of {tag} changed")
    anchors = []
    for x in lines(ROOT / tag / "anchors.jsonl.gz"):
        s = decode(x)
        s.global_observation = None
        s.tag, s.block, s.successor = tag, x["block"], x["successor"]
        s.cost, s.decay, s.value_now = x["cost"], x["decay"], x["value_now"]
        anchors.append(s)
    if len(anchors) != record["anchors"]:
        raise ValueError(f"Incomplete anchors for {tag}")
    return anchors, record


def successor_values(tag, record, net=None):
    """Value estimates at the successors: as stored while collecting with this same model, or recomputed by `net`."""
    out, batch = [], []
    for x in lines(ROOT / tag / "successors.jsonl.gz"):
        if net is None:
            out.append(x["sample"]["old_value"])
            continue
        batch.append(decode(x).state)
        if len(batch) == 64:
            out += values(net, batch)
            batch = []
    if batch:
        out += values(net, batch)
    if len(out) != record["successors"]:
        raise ValueError(f"Incomplete successors for {tag}")
    return out


def update(u, tags, needed, smoke=False):
    note(f"update {u}: loading decisions")
    loaded = {tag: load(tag, smoke) for tag in tags}
    anchors = [s for tag in tags for s in loaded[tag][0]]
    note(f"update {u}: {len(anchors):,} decisions loaded")
    net, optimizer, _ = restore(model_path(u, smoke))
    report = dict(update=u, tags=list(tags), anchors=len(anchors), update_threads=UPDATE_THREADS)
    with update_threads():
        if u == 0:
            # As in bp: the start model's value head predicted another quantity. Start it again on the discounted
            # block time, fit it alone, and keep every action score fixed.
            probe = [s.state for s in anchors[:256]]
            with torch.no_grad():
                before = net(pack_states(probe))[0]
            optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
            restart_value_head(net, seed=MODEL_SEED + 3000)
            for s in anchors:
                s.target = -s.value_now / SCALE
            report["value_fit"] = fit_value_head(net, anchors, seed=MODEL_SEED + 3001)
            note(f"update {u}: value head fitted")
            with torch.no_grad():
                if not torch.equal(before, net(pack_states(probe))[0]):
                    raise ValueError("Fitting the value head moved an action score")
            for b in range(0, len(anchors), 64):
                group = anchors[b:b + 64]
                for s, v in zip(group, values(net, [x.state for x in group])):
                    s.old_value = v
        following = {tag: successor_values(tag, loaded[tag][1], net if u == 0 else None) for tag in tags}
        note(f"update {u}: successor values ready")
        for s in anchors:
            s.target = -s.cost / SCALE + (s.decay * following[s.tag][s.successor] if s.successor is not None else 0.)
        advantage = np.array([s.target - s.old_value for s in anchors]) * SCALE
        horizon = setting(smoke)["horizon"]
        report["noise_weighted_minutes"] = float(np.sqrt(np.mean(advantage ** 2)))
        report["time_share"] = time_share(advantage, np.array([s.state.time_fraction * horizon for s in anchors]))[0]
        report["value_error_weighted_minutes"] = float(np.sqrt(np.mean(
            np.square([s.old_value * SCALE + s.value_now for s in anchors]))))
        report["without_successor"] = sum(s.successor is None for s in anchors)
        note(f"update {u}: policy update starts")
        report["fit"] = normalized_ppo_update(net, optimizer, anchors, seed=MODEL_SEED + 1100 + u,
                                              max_samples=needed, epochs=4, batch_size=64)
    note(f"update {u}: policy update done")
    path = ROOT / ("smoke-models" if smoke else "models") / f"update-{u}.pt"
    checkpoint(path, net, optimizer, u)
    report["checkpoint_sha256"] = file_sha256(path)
    save((OUT / "smoke" if smoke else OUT) / "training" / f"fit-{u}.json", report)
    return path


def evaluate(checkpoint_path, smoke=False):
    cfg = setting(smoke)
    folder = (ROOT / "smoke-evaluation") if smoke else (OUT / "evaluation" / "retrained")
    month_crane_replay.run("trained", False, info_path=cfg["test"], manifest_dir=cfg["manifests"],
                           checkpoint=checkpoint_path, folder=folder, horizon=cfg["horizon"])
    return read(folder / "result.json")


# ---------------------------------------------------------------- wiring check and registration
def imported_sources():
    found = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            relative = Path(path).resolve().relative_to(REPO)
        except ValueError:
            continue
        if relative.parts[0] in ("scripts", "src"):
            found.add(relative.as_posix())
    return sorted(found)


def smoke():
    reference = SMOKE_INFO / "smoke" / "trained" / "trucks.jsonl.gz"
    CACHE.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix="br-smoke-") as scratch:
        month_crane_replay.run("trained", True, output_root=Path(scratch), check_turnaround_targets=True)
        with gzip.open(reference, "rb") as a, gzip.open(Path(scratch) / "smoke" / "trained" / "trucks.jsonl.gz", "rb") as b:
            reproduced = a.read() == b.read()
    if not reproduced:
        raise RuntimeError("The current code no longer reproduces the archived three-day replay")
    note("archived three-day replay reproduced")
    tags = ["smoke-0", "smoke-1"]
    records = [rollout(tag, START, MODEL_SEED + 7900 + k, 1024, smoke=True) for k, tag in enumerate(tags)]
    # From the second update on, targets use the value estimates stored while collecting. They must equal what the
    # collecting model gives for the stored states.
    collecting, gaps = restore(START)[0], []
    for tag, record in zip(tags, records):
        anchors, _ = load(tag, smoke=True)
        recomputed = [v for b in range(0, len(anchors), 64)
                      for v in values(collecting, [s.state for s in anchors[b:b + 64]])]
        gaps.append(np.abs(np.subtract([s.old_value for s in anchors], recomputed)).max())
        gaps.append(np.abs(np.subtract(successor_values(tag, record), successor_values(tag, record, collecting))).max())
    stored_value_gap = float(max(gaps))
    if stored_value_gap > 1e-5:
        raise RuntimeError("Stored value estimates differ from the collecting model")
    final = update(0, tags, 2048, smoke=True)
    fit = read(OUT / "smoke" / "training" / "fit-0.json")
    evaluated = evaluate(final, smoke=True)
    complete = evaluated["windows"]["all"]["unfinished_trucks"] == 0
    imported = imported_sources()
    save(OUT / "smoke" / "smoke-check.json", dict(
        purpose="Wiring check on three short days; numbers are not results",
        archived_replay_reproduced=reproduced, stored_value_max_gap=stored_value_gap, rollouts=records, update=fit,
        evaluation=dict(complete=complete, windows=evaluated["windows"]),
        decisions_per_update=decisions_per_update(), imported_sources=imported, commit=git("rev-parse", "HEAD"),
        sources_committed=not git("status", "--porcelain", "--", *imported)))
    if fit["value_fit"]["normalized_rmse_after"] >= 1:
        raise RuntimeError("The value head did not learn the one-hour targets")
    if not complete:
        raise RuntimeError("The three-day evaluation left work unfinished")
    print("smoke check passed", flush=True)


def register():
    if REGISTRATION.exists():
        raise RuntimeError("Refusing to overwrite an existing registration")
    checked = read(OUT / "smoke" / "smoke-check.json")
    head = git("rev-parse", "HEAD")
    if checked["commit"] != head or not checked["sources_committed"]:
        raise RuntimeError("Run the three-day check from the committed code first")
    bp = read(BP / "train-campaign.json")
    if bp["start_model"] != file_sha256(START):
        raise ValueError("bp started from another model")
    if any(read(BP / "runs" / f"wave-{u}-{k}.json")["sampling_seed"] != seed_of(u, k)
           for u in range(UPDATES) for k in range(MONTHS)):
        raise ValueError("bp used other sampling seeds")
    cfg = setting(False)
    inputs = [cfg["train"], cfg["test"], cfg["manifests"] / read(cfg["train"])["path"],
              cfg["manifests"] / read(cfg["test"])["path"], START, BQ, BP / "train-campaign.json"]
    files = sorted(set(checked["imported_sources"]) | {Path(p).as_posix() for p in inputs})
    needed = decisions_per_update()
    save(REGISTRATION, dict(
        task="YR-332-br", commit=head, sources={p: file_sha256(p) for p in files},
        settings=dict(discount_hours=HOURS, successor_after_s=WINDOW_S, learning_unit_minutes=SCALE,
                      block_scope="block trucks + 67 x port calls with a part at the block, until the whole call finishes",
                      decisions_per_update=needed, anchors_per_month=needed // MONTHS, updates=UPDATES,
                      months_per_update=MONTHS, start_model_sha256=file_sha256(START),
                      sampling_seeds=[[seed_of(u, k) for k in range(MONTHS)] for u in range(UPDATES)],
                      anchor_seed="sampling seed + 500000", value_head_seeds=[MODEL_SEED + 3000, MODEL_SEED + 3001],
                      ppo_seeds=[MODEL_SEED + 1100 + u for u in range(UPDATES)],
                      ppo=dict(learning_rate=3e-4, epochs=4, batch_size=64), update_threads=UPDATE_THREADS),
        same_as_bp="start model, sampling and anchor seeds, decisions per update, value-head restart, PPO, test month",
        changed_from_bp="learning target only: one-hour discount, the value estimate at the block's first decision "
                        "30 minutes later carries the rest, bn's time-consistent block scope"))
    print(f"registered {len(files)} files", flush=True)


def verify():
    registration = read(REGISTRATION)
    changed = [p for p, h in registration["sources"].items() if not Path(p).is_file() or file_sha256(p) != h]
    if changed:
        raise RuntimeError(f"Registered files changed: {changed}")
    return registration


# ---------------------------------------------------------------- registered run
def status(stage, **details):
    save(OUT / "queue-status.json", dict(stage=stage, pid=os.getpid(),
                                         at=time.strftime("%Y-%m-%d %H:%M:%S %z"), **details))


@contextmanager
def single_runner():
    """One waiter or campaign at a time; the lock goes with the process."""
    path = ROOT / "queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def parallel(jobs):
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = {}
    for name, (done, arguments) in jobs.items():
        if done.exists():
            continue
        log = (logs / f"{name}.log").open("w", encoding="utf8")
        running[name] = (subprocess.Popen([sys.executable, "-X", "utf8", "scripts/v7/one_hour_train.py", *arguments],
                                          stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "PYTHONPATH": "src"}), log)
    note(f"started {sorted(running)}")
    failed = []
    for name, (process, log) in running.items():
        if process.wait() != 0:
            failed.append(name)
        log.close()
    if failed:
        raise RuntimeError(f"Failed {failed}; logs under {logs}")
    note(f"finished {sorted(running)}")


def compare_first_wave():
    """bp's first wave used the same model and seeds, so each month should come out identical."""
    rows = []
    for k in range(MONTHS):
        mine, theirs = read(OUT / "runs" / f"wave-0-{k}.json"), read(BP / "runs" / f"wave-0-{k}.json")
        rows.append(dict(month=k, identical=mine["month"] == theirs["month"],
                         decisions=[mine["decisions"], theirs["decisions"]],
                         truck_minutes=[mine["month"]["truck_minutes"], theirs["month"]["truck_minutes"]]))
    save(OUT / "first-wave-vs-bp.json", dict(identical=all(r["identical"] for r in rows), months=rows))


def campaign():
    registration = verify()
    needed = registration["settings"]["decisions_per_update"]
    first = OUT / "campaign.json"
    save(first if not first.exists() else OUT / f"campaign-resumed-{time.strftime('%m%d-%H%M')}.json", dict(
        commit=git("rev-parse", "HEAD"), registration_sha256=file_sha256(REGISTRATION), pid=os.getpid(),
        started=time.strftime("%Y-%m-%d %H:%M:%S %z"), update_threads=UPDATE_THREADS))
    for u in range(UPDATES):
        tags = [f"wave-{u}-{k}" for k in range(MONTHS)]
        verify()
        status("training", update=u, step="collecting")
        parallel({tag: (OUT / "runs" / f"{tag}.json",
                        ["rollout", "--tag", tag, "--checkpoint", Path(model_path(u)).as_posix(),
                         "--seed", str(seed_of(u, k)), "--keep", str(needed // MONTHS)])
                  for k, tag in enumerate(tags)})
        if u == 0:
            compare_first_wave()
        if not (ROOT / "models" / f"update-{u}.pt").exists():
            verify()
            status("training", update=u, step="updating")
            update(u, tags, needed)
            note(f"update {u + 1}/{UPDATES} applied")
    verify()
    status("evaluating")
    evaluate(ROOT / "models" / f"update-{UPDATES - 1}.pt")
    note("campaign complete")


def bt_processes():
    """Command lines of running YR-332-bt processes, or None when the process list could not be read."""
    listed = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
                             "Select-Object -ExpandProperty CommandLine"],
                            capture_output=True, text=True, encoding="utf8", errors="replace")
    rows = listed.stdout.splitlines()
    if listed.returncode != 0 or not any("one_hour_train.py" in row for row in rows):
        return None   # the list must at least show this waiter
    return [row for row in rows if "one_hour_train.py" not in row
            and ("planned-times" in row or "after_bp_planned_times" in row)]


def wait_bt(interval=300, grace_h=2.):
    """Start the campaign when YR-332-bt reports completion and its processes are gone, or once they have been
    gone for `grace_h` hours whatever its last status (failed, stopped, or a stale record)."""
    with single_runner():
        try:
            verify()
            idle_since = None
            while True:
                bt = read(BT_STATUS) if BT_STATUS.exists() else {}
                running = bt_processes()
                status("waiting_for_bt", bt_stage=bt.get("stage"), bt_arm=bt.get("arm"),
                       bt_processes=None if running is None else len(running))
                if running == []:
                    if bt.get("stage") == "complete":
                        break
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since >= 3600 * grace_h:
                        break
                else:
                    idle_since = None
                time.sleep(interval)
            note(f"YR-332-bt finished (stage {bt.get('stage')}); starting the registered campaign")
            campaign()
            status("complete")
        except BaseException as exc:
            status("failed", error=repr(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "register", "campaign", "wait-bt", "rollout"))
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "rollout":
        rollout(args.tag, Path(args.checkpoint), args.seed, args.keep)
    elif args.action == "campaign":
        with single_runner():
            campaign()
    else:
        {"smoke": smoke, "register": register, "wait-bt": wait_bt}[args.action]()
