"""YR-332-bn, added after the registered step-2 result: where does the 30-minute-plus-estimate noise come from?

Read-only on the four measurement months. The registered verdict (close without retraining) does not change.
  1. split the noise into the actual 30-minute part and the change of the value estimate between the two decisions
  2. how alike the value errors at the two decisions are, and what the decisions without a successor add
  3. a value head that also knows which block decides (one of 21), fitted and scored exactly like one without it
"""
import numpy as np
import torch
from torch import nn

from local_return_scope import time_share
from predicted_tail_train import (MODEL_SEED, OUT, SCALE, fit_value_head, load, read_horizon, restore, values, widen)
from yard_rl.v7.crane.actor_critic import pack_states
from vessel_progress_record import save


TAGS = [f"measure-{k}" for k in range(4)]


def contexts(net, samples):
    with torch.no_grad():
        return torch.cat([net.actor_and_critic_input(pack_states([s.state for s in samples[b:b+64]]))[1]
                          for b in range(0, len(samples), 64)])


def head_fit(x, y, seed, epochs=30):
    mean, scale = float(y.mean()), float(y.std())
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(x.shape[1], 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    z = torch.as_tensor((y - mean) / scale, dtype=torch.float32)
    rng = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=rng)
        for b in range(0, len(order), 256):
            ix = order[b:b+256]
            loss = .5 * (head(x[ix]).squeeze(-1) - z[ix]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return lambda q: head(q).squeeze(-1).detach().double().numpy() * scale + mean


def main():
    months = [load(tag) for tag in TAGS]
    fit = [s for anchors, _, _ in months[:2] for s in anchors]
    held = [s for anchors, _, _ in months[2:] for s in anchors]
    net = restore(widen())[0]
    for s in fit:
        s.target = -s.month_left / SCALE
    fit_value_head(net, fit, seed=MODEL_SEED + 3001)   # the registered fit, same seed
    v_i = np.array(values(net, held)) * SCALE
    nexts = [s.next for s in held]
    has = np.array([n is not None for n in nexts])
    v_j = np.zeros(len(held))
    v_j[has] = np.array(values(net, [n for n in nexts if n is not None])) * SCALE
    cost = np.array([s.cost for s in held])
    left = np.array([s.month_left for s in held])
    # Values are negative remaining time, so the signal is target - value = (-cost + v_j) - v_i. In one realized
    # month the remaining time at the successor is left - cost, hence signal = (error at successor) - (error at anchor).
    advantage = -cost + v_j - v_i
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    err_i, err_j = v_i + left, v_j + (left - cost)
    gap = float(np.max(np.abs(advantage[has] - (err_j - err_i)[has])))
    if gap > 1e-3:
        raise ValueError("The signal is not the change of the value error")
    times = np.array([s.state.time_fraction * read_horizon(False) for s in held])
    parts = dict(
        advantage_rms=rms(advantage), advantage_median_abs=float(np.median(np.abs(advantage))),
        advantage_p90_abs=float(np.percentile(np.abs(advantage), 90)), time_share=time_share(advantage, times)[0],
        actual_part_mean=float(cost[has].mean()), actual_part_sd=float(cost[has].std()),
        value_error_rms_anchor=rms(err_i[has]), value_error_rms_successor=rms(err_j[has]),
        value_error_correlation=float(np.corrcoef(err_i[has], err_j[has])[0, 1]), identity_max_gap=gap,
        without_successor=int((~has).sum()), advantage_rms_with_successor_only=rms(advantage[has]))
    # Value head with and without the deciding block, on the same fixed features and targets.
    blocks = sorted({s.block for s in fit + held})
    def onehot(samples):
        out = torch.zeros(len(samples), len(blocks))
        for r, s in enumerate(samples):
            out[r, blocks.index(s.block)] = 1.
        return out
    succ = [n for n in nexts if n is not None]
    c_fit, c_held, c_succ = contexts(net, fit), contexts(net, held), contexts(net, succ)
    y = -np.array([s.month_left for s in fit])
    variants = {}
    for name, extra in (("without_block", None), ("with_block", onehot)):
        x_fit = c_fit if extra is None else torch.cat([c_fit, extra(fit)], 1)
        x_held = c_held if extra is None else torch.cat([c_held, extra(held)], 1)
        x_succ = c_succ if extra is None else torch.cat([c_succ, extra([s for s, h in zip(held, has) if h])], 1)
        predict = head_fit(x_fit, y, MODEL_SEED + 4001)
        a = predict(x_held)
        b = np.zeros(len(held))
        b[has] = predict(x_succ)
        adv = -cost + b - a
        variants[name] = dict(value_error_rms=rms(a + left), advantage_rms=rms(adv),
                              advantage_rms_with_successor_only=rms(adv[has]),
                              decisions_for_equal_signal=(rms(adv) / 24.747435702432035) ** 2)
    summary = dict(purpose="Added after the registered step-2 result; explains the noise and does not change the verdict",
                   registered=read_measure(), parts=parts, value_head_variants=variants)
    save(OUT / "noise-diagnostic.json", summary)
    import json
    print(json.dumps(dict(parts=parts, variants=variants), indent=1))


def read_measure():
    import json
    return json.loads((OUT / "measure.json").read_text(encoding="utf8"))["noise_weighted_minutes"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
