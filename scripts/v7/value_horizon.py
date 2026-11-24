"""YR-332-bq: how far should the value estimate look ahead? Read-only on the four YR-332-bn measurement months.

For a discount time constant T, the block value is D_T(t) = integral from t of exp(-(s - t) / T) x block rate(s),
with bn's block scope (the block's trucks + 67 x port calls that have a part at the block, until the call finishes).
The learning signal of a decision is the discounted 30-minute block time plus the discounted estimate at the block's
first decision 30 minutes later, minus the estimate now. For each T this measures
  - its noise (value head on fixed features, fitted on months 0-1, scored on months 2-3) and time-of-month share
  - how much of a decision's effect survives the discount: trucks present at the block and port calls the block
    holds back, each weighted by exp(-(time until it finishes) / T)
  - the decisions per update at which the surviving effect equals the noise
"""
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from joint_turnaround_train import MODEL_SEED, setting
from local_return_scope import gain_reference, time_share
from predicted_tail_train import ROOT as BN_ROOT, load, restore, widen
from vessel_progress_record import save
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.reward.joint_turnaround import JointTurnaroundClock, VESSEL_TIME_WEIGHT


OUT = Path("outputs/reports/yr332_value_horizon")
TAGS = [f"measure-{k}" for k in range(4)]
HOURS = (1, 3, 6, 12, 24, 48, None)            # None: no discount, the bn definition
SCALE = 1e3
SEED = MODEL_SEED + 4101
MAX_TIME_SHARE, MIN_SHIP_KEEP = .20, .25


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


class Discounted:
    """Exact discounted remaining block time from a piecewise-constant event clock."""

    def __init__(self, clock, hours):
        self.clock, self.t = clock, None if hours is None else 3600. * hours
        if self.t is None:
            return
        s, w = clock.times, clock.active_weights
        self.after = [0.] * (len(s) + 1)          # G[k] = discounted time from s[k] onward
        for k in range(len(s) - 1, -1, -1):
            gap = s[k + 1] - s[k] if k + 1 < len(s) else math.inf
            decay = math.exp(-gap / self.t) if gap != math.inf else 0.
            self.after[k] = w[k] * self.t * (1 - decay) + decay * self.after[k + 1]

    def at(self, t):
        if self.t is None:
            return self.clock.total_minutes - self.clock.minutes_at(t)
        s, w = self.clock.times, self.clock.active_weights
        k = bisect_right(s, t) - 1
        if k < 0:
            return (math.exp(-(s[0] - t) / self.t) * self.after[0] if s else 0.) / 60
        if k + 1 < len(s):
            decay = math.exp(-(s[k + 1] - t) / self.t)
            return (w[k] * self.t * (1 - decay) + decay * self.after[k + 1]) / 60
        return w[k] * self.t / 60


def ledger(tag, jobs, order):
    folder = BN_ROOT / tag
    trucks, calls, held = defaultdict(list), defaultdict(list), defaultdict(list)
    with gzip.open(folder / "trucks.jsonl.gz", "rt", encoding="utf8") as stream:
        for r in map(json.loads, stream):
            trucks[jobs[r["job"]]].append((r["gate"], r["finish"]))
    for r in read(folder / "ships.json"):
        where = order[r["ship"]]
        own = defaultdict(float)
        for b, f in zip(where, r["stream_finishes"]):
            own[b] = max(own[b], f)
        for b, f in own.items():
            calls[b].append((r["start"], r["finish"]))
            if f == r["finish"]:
                held[b].append((r["start"], r["finish"]))   # the block's part finished last: it held the call back
    clocks = {b: JointTurnaroundClock(trucks[b], calls[b]) for b in set(trucks) | set(calls)}
    arrays = {b: (np.array([g for g, _ in v]), np.array([f for _, f in v])) for b, v in trucks.items()}
    return clocks, arrays, held


def manifest():
    cfg = setting(False)
    with gzip.open(cfg["manifests"] / read(cfg["train"])["path"], "rt", encoding="utf8") as stream:
        document = json.load(stream)
    order = defaultdict(list)
    for rows in document["vessels_by_day"].values():
        for row in rows:
            order[row["ship"]].append(row["block"])
    return {e["job_id"]: e["block"] for e in document["schedule"]}, order


def contexts(net, samples):
    with torch.no_grad():
        return torch.cat([net.actor_and_critic_input(pack_states([s.state for s in samples[b:b+64]]))[1]
                          for b in range(0, len(samples), 64)])


def head_fit(x, y, seed):
    mean, scale = float(y.mean()), float(y.std()) or 1.
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(x.shape[1], 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    z = torch.as_tensor((y - mean) / scale, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed + 1)
    for _ in range(30):
        order = torch.randperm(len(x), generator=rng)
        for b in range(0, len(order), 256):
            ix = order[b:b+256]
            loss = .5 * (head(x[ix]).squeeze(-1) - z[ix]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return lambda q: head(q).squeeze(-1).detach().double().numpy() * scale + mean


def main():
    jobs, order = manifest()
    months = []
    for tag in TAGS:
        anchors, successors, record = load(tag)
        months.append((anchors, record, *ledger(tag, jobs, order)))
    net = restore(widen())[0]
    fit = [(s, m) for m in months[:2] for s in m[0]]
    held = [(s, m) for m in months[2:] for s in m[0]]
    nexts = [(s, m) for s, m in held if s.next is not None]   # anchor and its month; the successor shares the block
    has = np.array([s.next is not None for s, _ in held])
    c_fit, c_held = contexts(net, [s for s, _ in fit]), contexts(net, [s for s, _ in held])
    c_next = contexts(net, [s.next for s, _ in nexts])
    horizon = setting(False)["horizon"]
    t_held = np.array([s.state.time_fraction * horizon for s, _ in held])
    t_next = np.array([s.next.state.time_fraction * horizon for s, _ in nexts])
    gain = gain_reference()
    rows = {}
    for hours in HOURS:
        key = "none" if hours is None else f"{hours}h"
        value = {}
        def remaining(s, m, t):
            b = s.block
            if (id(m), b) not in value:
                value[id(m), b] = Discounted(m[2].get(b, JointTurnaroundClock([], [])), hours)
            return value[id(m), b].at(t)
        y_fit = np.array([-remaining(s, m, s.state.time_fraction * horizon) for s, m in fit])
        truth_i = np.array([remaining(s, m, t) for (s, m), t in zip(held, t_held)])
        truth_j = np.zeros(len(held))
        truth_j[has] = [remaining(s, m, t) for (s, m), t in zip(nexts, t_next)]
        if hours is None:   # the bn definition must come back exactly
            stored = np.array([s.month_left for s, _ in held])
            if not np.allclose(truth_i, stored, rtol=1e-9, atol=1e-6):
                raise ValueError("Undiscounted block time differs from the stored bn value")
        gap = np.zeros(len(held))
        gap[has] = t_next - t_held[has]
        decay = np.ones(len(held)) if hours is None else np.exp(-gap / (3600. * hours))
        decay[~has] = 0.
        actual = truth_i - decay * truth_j                    # discounted block time until the successor
        predict = head_fit(c_fit, y_fit / SCALE, SEED)
        v_i = predict(c_held) * SCALE
        v_j = np.zeros(len(held))
        v_j[has] = predict(c_next) * SCALE
        advantage = -actual + decay * v_j - v_i
        if hours is None:
            stored = np.array([s.cost for s, _ in held])
            if not np.allclose(actual, stored, rtol=1e-9, atol=1e-6):
                raise ValueError("Undiscounted 30-minute block time differs from the stored bn value")
        noise = float(np.sqrt(np.mean(advantage ** 2)))
        truck_keep, ship_keep = [0., 0], [0., 0]
        for (s, m), t in zip(held, t_held):
            gate, finish = m[3].get(s.block, (np.empty(0), np.empty(0)))
            left = finish[(gate <= t) & (finish > t)] - t
            truck_keep[0] += float(np.sum(np.ones_like(left) if hours is None else np.exp(-left / (3600. * hours))))
            truck_keep[1] += len(left)
            for start, end in m[4].get(s.block, []):
                if start <= t < end:
                    ship_keep[0] += 1. if hours is None else math.exp(-(end - t) / (3600. * hours))
                    ship_keep[1] += 1
        keep_t, keep_s = truck_keep[0] / truck_keep[1], ship_keep[0] / ship_keep[1]
        signal = gain["per_decision"] * (gain["truck_share"] * keep_t + gain["ship_share"] * keep_s)
        rows[key] = dict(hours=hours, noise_weighted_minutes=noise, time_share=time_share(advantage, t_held)[0],
                         truck_effect_kept=keep_t, ship_effect_kept=keep_s, signal=signal,
                         noise_over_signal=noise / signal, decisions_for_equal_signal=(noise / signal) ** 2,
                         value_error_rms=float(np.sqrt(np.mean((v_i + truth_i) ** 2))))
        print(key, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[key].items()}), flush=True)
    collected = 4 * min(m[1]["nontrivial"] for m in months)
    feasible = [k for k, r in rows.items() if r["decisions_for_equal_signal"] <= collected and r["time_share"] <= MAX_TIME_SHARE]
    best = max(feasible, key=lambda k: rows[k]["ship_effect_kept"]) if feasible else None
    outcome = ("register a retraining candidate (separate confirmation)" if best and rows[best]["ship_effect_kept"] >= MIN_SHIP_KEEP
               else "no length keeps the port-call effect; close")
    summary = dict(purpose="Discount length versus noise and surviving effect, on stored bn months; no simulation",
                   tags=TAGS, fit_decisions=len(fit), held_out_decisions=len(held), scale=SCALE, gain=gain,
                   collected_per_wave=collected, rows=rows, feasible=feasible, chosen=best, outcome=outcome,
                   thresholds=dict(max_time_share=MAX_TIME_SHARE, min_ship_effect_kept=MIN_SHIP_KEEP))
    save(OUT / "summary.json", summary)
    print(json.dumps(dict(feasible=feasible, chosen=best, outcome=outcome), indent=1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
