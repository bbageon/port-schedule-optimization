"""Time the BP update and its one-decision inference on identical saved training data."""
import argparse
from itertools import islice
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import torch

from crane_devices import device_for
from joint_turnaround_train import decode
from short_window_samples import anchors_from
from short_window_train import ROOT, SCALE, save
from value_stability import restore
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.value_normalization import normalized_ppo_update
from yard_rl.v7.crane.update_threads import update_threads
from yard_rl.v7.stage.seed_bundle import file_sha256


def benchmark(count):
    torch.set_num_threads(1)
    device_for("train", "cuda")
    checkpoint = ROOT / "models/update-2.pt"
    archive = ROOT / "wave-3-0/decisions.json.gz"
    samples = []
    for row in islice(anchors_from(archive), count):
        sample = decode(row)
        sample.target = -row["cost"] / SCALE
        samples.append(sample)
    if len(samples) != count:
        raise ValueError("Incomplete benchmark sample")
    rows, choices = {}, {}
    cpu_parameters = None
    for label, device, threads in (("cpu", "cpu", 1), ("cpu16", "cpu", 16), ("cuda", "cuda", 1)):
        net = restore(checkpoint, device=device)[0]
        timings, actions = [], []
        with torch.inference_mode():
            for i, sample in enumerate(samples[:256]):
                start = perf_counter()
                logits, value = net(pack_states([sample.state]))
                scores = logits[0].cpu()
                value.item()
                actions.append([m[int(scores[m].argmax())] for m in sample.masks])
                if i >= 16:
                    timings.append(perf_counter() - start)
        choices[label] = actions
        update_times = []
        repeat_drift = 0.
        for _ in range(3):
            net, optim, _ = restore(checkpoint, device=device)
            with update_threads(threads):
                start = perf_counter()
                fit = normalized_ppo_update(net, optim, samples, seed=123,
                                            max_samples=count, epochs=4, batch_size=64)
                if device == "cuda":
                    torch.cuda.synchronize()
                update_times.append(perf_counter() - start)
            params = torch.nn.utils.parameters_to_vector(net.parameters()).detach().cpu()
            if len(update_times) == 1:
                first_parameters = params
            else:
                repeat_drift = max(repeat_drift, (params - first_parameters).abs().max().item())
        if label == "cpu":
            cpu_parameters = first_parameters
        rows[label] = dict(update_threads=threads, update_seconds=update_times, median_update_seconds=median(update_times),
                            single_decision_median_ms=1000 * median(timings),
                            max_parameter_difference_vs_cpu=(first_parameters - cpu_parameters).abs().max().item(),
                            max_parameter_difference_on_repeat=repeat_drift,
                            replay_logp_max_error=fit["replay_logp_max_error"],
                            final_run=fit["means"])
        print(label, rows[label], flush=True)
    return dict(devices=rows, matched_inference=sum(a == b for a, b in zip(choices["cpu"], choices["cuda"])),
                compared_inference=len(choices["cpu"]), samples=count, epochs=4, batch_size=64,
                checkpoint_sha256=file_sha256(checkpoint), archive=archive.as_posix(),
                pytorch=torch.__version__, gpu=torch.cuda.get_device_name(), model=net.config,
                parameters=sum(p.numel() for p in net.parameters()),
                code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    save(args.output, benchmark(args.samples))
