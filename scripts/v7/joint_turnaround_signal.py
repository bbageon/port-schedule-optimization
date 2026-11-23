"""What moved the policy: how much of each update's advantage is decided by the clock alone."""
import gzip
import json
from math import isclose

import torch

from joint_turnaround_train import (MAX_SAMPLES, MODEL_SEED, OUT, START, decode, model_path, read, restore, save,
                                    setting, value_quality)
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.value_normalization import fit_value_head, restart_value_head


BIN_S = 6 * 3600


def stored(cfg, update):
    """The same decisions, in the same order, that the registered update consumed."""
    rows = [read(cfg["evidence"] / "training" / f"rollout-{update}-{k}.json") for k in range(cfg["rollouts"])]
    complete = [r for r in rows if r["complete"]]
    each = -(-MAX_SAMPLES // len(complete))
    policy, value, month = [], [], []
    for r in complete:
        with gzip.open(cfg["root"] / "rollouts" / f"{update}-{r['index']}" / "samples.json.gz", "rt", encoding="utf8") as f:
            document = json.load(f)
        chosen = [decode(x) for x in document["policy"][:each]]
        policy += chosen
        month += [r["index"]] * len(chosen)
        value += [decode(x) for x in document["value"]]
    return policy[:MAX_SAMPLES], value, month[:MAX_SAMPLES]


def share(values, groups):
    """Fraction of the variance of `values` that the group means reproduce."""
    values = torch.as_tensor(values, dtype=torch.float64)
    total = values.var(unbiased=False)
    cells = {}
    for i, g in enumerate(groups):
        cells.setdefault(g, []).append(i)
    within = sum(len(ix) * values[ix].var(unbiased=False) for ix in cells.values()) / len(values)
    return float(1 - within / total), len(cells)


def diagnose(update, cfg):
    policy, value, month = stored(cfg, update)
    fit = read(cfg["evidence"] / "training" / f"fit-{update}.json")
    if update == 0:  # rebuild the restarted value head exactly as the registered update did
        net = restore(model_path(cfg, 0))[0]
        restart_value_head(net, seed=MODEL_SEED + 3000)
        fit_value_head(net, value, seed=MODEL_SEED + 3001)
        with torch.no_grad():
            for b in range(0, len(policy), 64):
                for s, v in zip(policy[b:b+64], net(pack_states([s.state for s in policy[b:b+64]]))[1].tolist()):
                    s.old_value = v
    quality = value_quality(policy)
    if not all(isclose(quality[k], fit["value_before_update"][k], rel_tol=1e-9, abs_tol=1e-12) for k in quality):
        raise ValueError("Rebuilt advantages differ from the registered update")
    advantage = [s.target - s.old_value for s in policy]
    time = [s.state.time_fraction * cfg["horizon"] for s in policy]
    clock = [int(t // BIN_S) for t in time]
    by_clock, bins = share(advantage, clock)
    spread = float(torch.tensor(advantage, dtype=torch.float64).std(unbiased=False))
    return dict(update=update, decisions=len(policy), advantage_sd_million=spread,
                share_from_clock_alone=by_clock, clock_bins=bins, bin_hours=BIN_S / 3600,
                share_expected_by_chance=(bins - 1) / (len(policy) - 1),
                share_from_month_alone=share(advantage, month)[0],
                share_from_clock_and_month=share(advantage, list(zip(clock, month)))[0],
                left_within_clock_and_month_million=spread * (1 - share(advantage, list(zip(clock, month)))[0]) ** .5)


def top_choices(path, samples):
    """Highest-scoring first choice per stored decision, with how flat the probabilities are."""
    net = restore(path)[0]
    choices, gaps = [], []
    with torch.no_grad():
        for b in range(0, len(samples), 64):
            batch = samples[b:b+64]
            for row, s in zip(net(pack_states([s.state for s in batch]))[0], batch):
                legal = next(m for m in s.masks if len(m) > 1)   # earlier steps were forced, so this set is fixed
                probability = torch.softmax(row[legal], 0)
                choices.append(legal[int(probability.argmax())])
                best = torch.topk(probability, 2).values
                gaps.append(float(best[0] - best[1]))
    return choices, sum(gaps) / len(gaps)


def choice_change(cfg):
    """Added after the results were seen: how many fixed (highest-score) choices each update overturned."""
    policy, value, _ = stored(cfg, 0)
    samples = policy + value        # first-wave decisions, all reached by the start model
    start, gap = top_choices(START, samples)
    rows = []
    for u in range(cfg["updates"]):
        chosen, after = top_choices(cfg["root"] / "models" / f"update-{u}.pt", samples)
        rows.append(dict(after_update=u + 1, top_choice_changed=sum(a != b for a, b in zip(start, chosen)) / len(start),
                         mean_gap_between_two_best_probabilities=after))
    return dict(decisions=len(samples), start_mean_gap_between_two_best_probabilities=gap, updates=rows,
                registered=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    cfg = setting(False)
    done = [u for u in range(cfg["updates"]) if (cfg["evidence"] / "training" / f"fit-{u}.json").exists()]
    rows = [diagnose(u, cfg) for u in done]
    save(OUT / "signal-diagnostic.json", dict(
        question="Is the difference between the realized return and the predicted return set by when a decision was made?",
        note="Unrelated numbers sorted into the same bins would reproduce share_expected_by_chance of their variance.",
        updates=rows, top_choice_change=choice_change(cfg) if len(done) == cfg["updates"] else None))
    for row in rows:
        print(json.dumps(row))
