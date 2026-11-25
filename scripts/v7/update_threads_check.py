"""YR-332-bu: time one registered PPO update on stored BP decisions with 1, 8 and 16 threads.

Same decisions, model, seed and settings in every run; only the thread count changes. Records the time, the
built-in checks of the update, how far the resulting weights drift from the one-thread run, and whether a repeated
run with the same thread count gives identical weights.
"""
from pathlib import Path
import itertools
import json
import subprocess
import time

import torch

from joint_turnaround_train import MODEL_SEED, decode
from short_window_samples import anchors_from
from short_window_train import ROOT, SCALE
from value_stability import restore
from vessel_progress_record import save
from yard_rl.v7.crane.update_threads import update_threads
from yard_rl.v7.crane.value_normalization import normalized_ppo_update


OUT = Path("outputs/reports/yr332_update_threads")
SAMPLES = 8192
MODEL = ROOT / "models" / "update-2.pt"        # the model that collected wave 3
DECISIONS = ROOT / "wave-3-0" / "decisions.json.gz"


def anchors():
    rows = []
    for x in itertools.islice(anchors_from(DECISIONS), SAMPLES):
        s = decode(x)
        s.global_observation = None
        s.target = -x["cost"] / SCALE
        rows.append(s)
    return rows


def run(n, data):
    net, optimizer, _ = restore(MODEL)
    with update_threads(n):
        started = time.perf_counter()
        fit = normalized_ppo_update(net, optimizer, data, seed=MODEL_SEED + 1103, max_samples=SAMPLES,
                                    epochs=4, batch_size=64)
        seconds = time.perf_counter() - started
    return seconds, fit, {k: v.detach().clone() for k, v in net.state_dict().items()}


def drift(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a if a[k].is_floating_point())


def main():
    torch.set_num_threads(1)
    data = anchors()
    rows, weights = {}, {}
    for label, n in (("1", 1), ("8", 8), ("16", 16), ("16-again", 16)):
        seconds, fit, w = run(n, data)
        weights[label] = w
        rows[label] = dict(threads=n, seconds=seconds, replay_logp_max_error=fit["replay_logp_max_error"],
                           optimizer_steps=fit["optimizer_steps"], kl=fit["means"]["approx_kl"])
        print(label, round(seconds, 1), fit["replay_logp_max_error"], flush=True)
    for label in rows:
        rows[label]["max_weight_drift_from_one_thread"] = drift(weights[label], weights["1"])
    summary = dict(
        purpose="Thread count for the update step; same decisions, model and seed",
        decisions=len(data), model=MODEL.as_posix(), source=DECISIONS.as_posix(), runs=rows,
        speedup_8=rows["1"]["seconds"] / rows["8"]["seconds"], speedup_16=rows["1"]["seconds"] / rows["16"]["seconds"],
        repeat_identical_16=drift(weights["16"], weights["16-again"]) == 0.,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    save(OUT / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
