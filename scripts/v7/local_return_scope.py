"""YR-332-bm step 1: recompute narrower learning targets on the stored YR-332-bl training months.

No simulation and no weight update. For each scope and window this measures how much of a decision's
effect the window keeps (capture) and how much unrelated variation it keeps (noise left after a value fit).
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from math import fsum, isclose, sqrt
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn

from joint_turnaround_train import START, decode, restore
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.reward.joint_turnaround import JointTurnaroundClock, VESSEL_TIME_WEIGHT


SOURCE = Path("build/yr332_joint_turnaround")
PREVIOUS = Path("outputs/reports/yr332_joint_turnaround")
OUT = Path("outputs/reports/yr332_local_return_scope")
HOURS = (0.5, 1, 2, 3, 4, 6, 8, 12, 24, 48, None)   # None: until every truck and port call of the month is done
SCOPES = ("block", "terminal", "block_trucks")      # block_trucks drops the 67 term: comparison only
FIT_WAVES, HELD_OUT_WAVES = (0, 1), (2, 3)
BIN_S = 6 * 3600
SEED = 9951101 + 4000
MAX_TIME_SHARE, MAX_NOISE_RATIO, MIN_SHIP_CAPTURE = .20, sqrt(2048), .50
SOURCES = ("scripts/v7/local_return_scope.py", "src/yard_rl/v7/reward/joint_turnaround.py",
           "src/yard_rl/v7/crane/actor_critic.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf8", newline="\n")
    tmp.replace(path)


def label(hours):
    return "month" if hours is None else f"{hours:g}h"


def manifest():
    info = read(PREVIOUS / "train-input.json")
    path = SOURCE / info["path"]
    if sha(path) != info["sha256"]:
        raise ValueError("Training month differs from the registered input")
    with gzip.open(path, "rt", encoding="utf8") as stream:
        document = json.load(stream)
    blocks = {e["job_id"]: e["block"] for e in document["schedule"]}
    streams = defaultdict(list)
    for rows in document["vessels_by_day"].values():   # the replay lists stream finishes in this order
        for row in rows:
            streams[row["ship"]].append(row["block"])
    return blocks, streams


def gain_reference():
    """Average improvement per decision on the unseen month, split into its truck and port-call parts."""
    arms = read(PREVIOUS / "summary.json")["evaluation"]
    old, rule = arms["previous"], arms["SF_SPT"]
    truck = (old["truck_minutes"] - rule["truck_minutes"]) * old["trucks"]
    ship = VESSEL_TIME_WEIGHT * 60 * (old["ship_hours"] - rule["ship_hours"]) * old["ships"]
    if not isclose(truck + ship, old["weighted_minutes"] - rule["weighted_minutes"], rel_tol=1e-9):
        raise ValueError("The gap does not split into truck and port-call parts")
    return dict(per_decision=(truck + ship) / old["decisions"], truck_share=truck / (truck + ship),
                ship_share=ship / (truck + ship), source=(PREVIOUS / "summary.json").as_posix())


def window(clock, t, hours):
    end = clock.total_minutes if hours is None else clock.minutes_at(t + 3600 * hours)
    return end - clock.minutes_at(t)


def port_calls_in_window(rows, t, hours):
    """67 x the whole port call's open minutes, for calls whose part at this block is still unfinished."""
    end = float("inf") if hours is None else t + 3600 * hours
    return VESSEL_TIME_WEIGHT / 60 * fsum(max(0., min(finish, end) - max(start, t))
                                          for start, finish, own, _ in rows if own > t)


def month(update, index, blocks, streams):
    folder = SOURCE / "training" / "rollouts" / f"{update}-{index}"
    record = read(PREVIOUS / "training" / f"rollout-{update}-{index}.json")
    if not record["complete"] or sha(folder / "samples.json.gz") != record["samples_sha256"]:
        raise ValueError("Stored decisions differ from the registered rollout")
    cargo = read(folder / "result.json")["cargo"]
    if cargo["dependent_reroutes"] or cargo["remote_vessel_handoffs"]:
        raise ValueError("A job moved away from its listed block")
    trucks, every = defaultdict(list), []
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        for row in map(json.loads, stream):
            trucks[blocks[row["job"]]].append((row["gate"], row["finish"]))
            every.append((row["gate"], row["finish"]))
    ships, calls = defaultdict(list), []
    for row in read(folder / "ships.json"):
        where = streams[row["ship"]]
        if len(where) != len(row["stream_finishes"]) or row["finish"] != max(row["stream_finishes"]):
            raise ValueError(f"Stream ledger mismatch for {row['ship']}")
        own = defaultdict(float)
        for b, f in zip(where, row["stream_finishes"]):
            own[b] = max(own[b], f)
        for b, f in own.items():
            ships[b].append((row["start"], row["finish"], f, f == row["finish"]))
        calls.append((row["start"], row["finish"]))
    terminal = JointTurnaroundClock(every, calls)
    if not isclose(terminal.total_minutes, record["weighted_minutes"], rel_tol=1e-12):
        raise ValueError("Monthly total differs from the registered rollout")
    local = {b: JointTurnaroundClock(rows, []) for b, rows in trucks.items()}
    arrays = {b: (np.array([g for g, _ in rows]), np.array([f for _, f in rows])) for b, rows in trucks.items()}
    with gzip.open(folder / "samples.json.gz", "rt", encoding="utf8") as stream:
        document = json.load(stream)
    rows, worst = [], 0.
    for kind in ("policy", "value"):
        for item in document[kind]:
            sample, obs = decode(item), item["global_observation"]
            t, b = obs["decision_time"], obs["decision_block"]
            stored = -sample.target * document["scale"]
            recomputed = window(terminal, t, None)
            worst = max(worst, abs(stored - recomputed))
            if not isclose(stored, recomputed, rel_tol=1e-9, abs_tol=1e-3):
                raise ValueError("Recomputed month-end target differs from the stored target")
            targets = {}
            for hours in HOURS:
                truck = window(local[b], t, hours) if b in local else 0.
                ship = port_calls_in_window(ships[b], t, hours)
                targets[label(hours)] = dict(block=truck + ship, block_trucks=truck, terminal=window(terminal, t, hours))
            gate, finish = arrays.get(b, (np.empty(0), np.empty(0)))
            remaining = np.sort(finish[(gate <= t) & (finish > t)] - t)
            controlled = sorted(finish_s - t for start, finish_s, own, last in ships[b] if last and start <= t < own)
            rows.append(dict(state=sample.state, wave=update, month=index, kind=kind, t=t, block=b, targets=targets,
                trucks=len(remaining), ships=len(controlled),
                truck_done={label(h): len(remaining) if h is None else int(np.searchsorted(remaining, 3600 * h, "right"))
                            for h in HOURS},
                ship_done={label(h): len(controlled) if h is None else sum(r <= 3600 * h for r in controlled)
                           for h in HOURS},
                truck_p50=float(np.percentile(remaining, 50)) / 3600 if len(remaining) else None,
                truck_p90=float(np.percentile(remaining, 90)) / 3600 if len(remaining) else None,
                ship_p50=float(np.percentile(controlled, 50)) / 3600 if controlled else None))
    check = dict(wave=update, month=index, decisions=len(rows), weighted_minutes=terminal.total_minutes,
                 month_end_target_max_abs_error=worst)
    return rows, check


def contexts(rows):
    net = restore(START)[0].eval()
    with torch.no_grad():
        return torch.cat([net.actor_and_critic_input(pack_states([r["state"] for r in rows[b:b+64]]))[1]
                          for b in range(0, len(rows), 64)])


def residual(context, target, fit, test, seed):
    """Fit only a fresh value head on fixed features; return the error on decisions it never saw."""
    y = torch.as_tensor(target[fit], dtype=torch.float64)
    mean, scale = float(y.mean()), float(y.std(unbiased=False))
    if scale == 0:
        return target[test] - mean
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(context.shape[1], 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    x, z = context[fit], ((y - mean) / scale).float()
    order_rng = torch.Generator().manual_seed(seed + 1)
    for _ in range(30):
        order = torch.randperm(len(x), generator=order_rng)
        for begin in range(0, len(order), 256):
            ix = order[begin:begin+256]
            loss = .5 * (head(x[ix]).squeeze(-1) - z[ix]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        prediction = head(context[test]).squeeze(-1).double().numpy() * scale + mean
    return target[test] - prediction


def time_share(values, times):
    """Fraction of the variance that the six-hour bin of the month alone reproduces."""
    bins = (times // BIN_S).astype(int)
    keys = np.unique(bins)
    within = sum(values[bins == k].var() * (bins == k).sum() for k in keys) / len(values)
    return float(1 - within / values.var()), len(keys)


def measure(rows, gain):
    fit = np.array([r["wave"] in FIT_WAVES for r in rows])
    test = np.array([r["wave"] in HELD_OUT_WAVES for r in rows])
    times = np.array([r["t"] for r in rows])
    context = contexts(rows)
    capture = {}
    for hours in HOURS:
        key = label(hours)
        trucks = sum(r["truck_done"][key] for r in rows) / sum(r["trucks"] for r in rows)
        ships = sum(r["ship_done"][key] for r in rows) / sum(r["ships"] for r in rows)
        capture[key] = dict(truck=trucks, ship=ships,
                            signal=gain["per_decision"] * (gain["truck_share"] * trucks + gain["ship_share"] * ships),
                            truck_signal=gain["per_decision"] * gain["truck_share"] * trucks)
    table = {}
    for s, scope in enumerate(SCOPES):
        for h, hours in enumerate(HOURS):
            key = label(hours)
            target = np.array([r["targets"][key][scope] for r in rows])
            left = residual(context, target, fit, test, SEED + 100 * s + h)
            rmse = float(np.sqrt(np.mean(left ** 2)))
            share, bins = time_share(left, times[test])
            raw_share, _ = time_share(target[test], times[test])
            signal = capture[key]["truck_signal" if scope == "block_trucks" else "signal"]
            table[f"{scope}/{key}"] = dict(scope=scope, window=key, target_mean=float(target[test].mean()),
                target_sd=float(target[test].std()), raw_time_share=raw_share, noise=rmse, time_share=share,
                chance_share=(bins - 1) / (int(test.sum()) - 1), signal=signal,
                noise_over_signal=rmse / signal if signal else None,
                # Descriptive, added after the two-month wiring run; not a decision criterion.
                decisions_for_equal_signal=(rmse / signal) ** 2 if signal else None)
            print(scope, key, f"noise={rmse:,.0f}", f"time_share={share:.3f}",
                  f"ratio={rmse / signal if signal else float('nan'):,.1f}", flush=True)
    return capture, table, dict(fit=int(fit.sum()), held_out=int(test.sum()))


def decide(capture, table):
    scored = [(table[f"block/{label(h)}"]["signal"] / table[f"block/{label(h)}"]["noise"], label(h)) for h in HOURS]
    best = max(scored)[1]
    row, ship = table[f"block/{best}"], capture[best]["ship"]
    checks = dict(time_share=row["time_share"] <= MAX_TIME_SHARE,
                  noise=row["noise_over_signal"] <= MAX_NOISE_RATIO,
                  ship_capture=ship >= MIN_SHIP_CAPTURE)
    if all(checks.values()):
        outcome = "retrain"
    elif checks["time_share"] and checks["noise"]:
        outcome = "register the predicted-tail follow-up"
    else:
        outcome = "close without retraining"

    def first(condition):
        return next((label(h) for h in HOURS if condition(label(h))), None)
    return dict(recommended_window=best, checks=checks, outcome=outcome,
                signal_over_noise={k: v for v, k in scored},
                shortest_window=dict(
                    ship_capture=first(lambda k: capture[k]["ship"] >= MIN_SHIP_CAPTURE),
                    truck_capture_90=first(lambda k: capture[k]["truck"] >= .9),
                    time_share=first(lambda k: table[f"block/{k}"]["time_share"] <= MAX_TIME_SHARE),
                    noise=first(lambda k: table[f"block/{k}"]["noise_over_signal"] <= MAX_NOISE_RATIO)),
                thresholds=dict(max_time_share=MAX_TIME_SHARE, max_noise_over_signal=MAX_NOISE_RATIO,
                                min_ship_capture=MIN_SHIP_CAPTURE))


def main(smoke):
    started = time.monotonic()
    blocks, streams = manifest()
    months = [(0, 0), (2, 0)] if smoke else [(u, k) for u in range(4) for k in range(4)]
    rows, checks = [], []
    for update, index in months:
        found, check = month(update, index, blocks, streams)
        rows += found
        checks.append(check)
        print("month", update, index, len(found), f"max error {check['month_end_target_max_abs_error']:.2e}", flush=True)
    gain = gain_reference()
    capture, table, split = measure(rows, gain)
    pick = lambda key: [r[key] for r in rows if r[key] is not None]
    summary = dict(
        purpose="Recompute narrower learning targets from stored decisions; no simulation, no weight update",
        smoke=smoke, months=checks, split=split, gain=gain, windows=[label(h) for h in HOURS], scopes=SCOPES,
        capture=capture, noise=table, decision=decide(capture, table),
        collected_per_wave=[sum(read(PREVIOUS / "training" / f"rollout-{u}-{k}.json")["samples"] for k in range(4))
                            for u in range(4)],
        used_per_update=2048,
        queue_hours=dict(truck_p50_median=float(np.median(pick("truck_p50"))),
                         truck_p90_median=float(np.median(pick("truck_p90"))),
                         controlled_ship_p50_median=float(np.median(pick("ship_p50"))),
                         decisions_with_controlled_ship=sum(r["ships"] > 0 for r in rows) / len(rows)),
        provenance=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                           text=True).strip()),
                        sources={p: sha(p) for p in SOURCES}, start_model=sha(START), python=platform.python_version(),
                        torch=torch.__version__, seconds=time.monotonic() - started))
    save((SOURCE / "local-scope-smoke.json") if smoke else (OUT / "summary.json"), summary)
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="two months only, written outside the evidence folder")
    torch.set_num_threads(1)
    main(parser.parse_args().smoke)
