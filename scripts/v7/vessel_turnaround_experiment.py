"""Re-run a fixed daily cohort through the last quay move, under the new objective."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from global_policy_inputs import OUT as PREVIOUS, RUNTIME, SCALE, SEEDS, bundle
from value_stability import checkpoint, restore
from yard_rl.v7.crane.on_policy import OnPolicyResolver
from yard_rl.v7.crane.value_normalization import normalized_ppo_update
from yard_rl.v7.stage.episode import run_episode


OUT = Path("outputs/reports/yr332_vessel_turnaround")
DAY = 86400
DRAIN = 5 * DAY
LOADS = (3500, 7500, 15000)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def previous(seed):
    return PREVIOUS / "policy" / f"global-{seed}" / "final.pt"


def simulate(split, index, load, net, *, stochastic=False, seed=0):
    kw = bundle().case(split, index, load)
    resolver = None if net is None else OnPolicyResolver(net, stochastic=stochastic, seed=seed,
        collect=stochastic, horizon=DAY, collect_until=DAY+DRAIN, global_context_mode="global")
    started = time.perf_counter()
    ep = run_episode(**kw, dispatcher="SF_SPT" if net is None else "RL_CRANE",
                     joint_resolver=resolver, crane_net=net, vessel_drain_s=DRAIN, **RUNTIME)
    data = ep.as_dict()
    if ep.admitted != load or ep.policy_exceptions or ep.txn_failed:
        raise RuntimeError("Admission or dispatch contract failed")
    assert math.isclose(sum(data[k] for k in ("c_wait","c_move","c_rehandle","c_vessel")),
                        ep.phi_krw, abs_tol=1e-5)
    data.update(split=split, index=index, load=load, input_seed=kw["seed"], seconds=time.perf_counter()-started)
    if stochastic and data["vessel_complete"]:
        data["reward_check"] = resolver.finish(ep.phi_krw, SCALE)
        data["tail_samples"] = sum(s.state.time_fraction >= 1 for s in resolver.samples)
    return data, resolver


def train(seed):
    base = OUT/"models"/str(seed)
    net, optimizer, _ = restore(previous(seed))
    for iteration in range(6):
        path = base/f"update-{iteration}.pt"
        if path.exists():
            net, optimizer, _ = restore(path)
            continue
        samples = []
        for index in (2*iteration, 2*iteration+1):
            load = LOADS[index % len(LOADS)]
            row, resolver = simulate("train", index, load, net, stochastic=True, seed=seed+900+index)
            save(base/f"day-{index}.json", row)
            if row["vessel_complete"]:
                samples.extend(resolver.samples)
            print(f"seed {seed} training day {index}: complete={row['vessel_complete']}", flush=True)
        fit = (normalized_ppo_update(net, optimizer, samples, seed=seed+1100+iteration,
                                     max_samples=2048, epochs=4, batch_size=64) if samples else
               dict(selected=0, optimizer_steps=0, reason="no fully observed training trajectory"))
        save(base/f"fit-{iteration}.json", fit)
        base.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        checkpoint(tmp, net, optimizer, arm="vessel_turnaround", seed=seed, iteration=iteration)
        tmp.replace(path)
        print(f"training {seed} update {iteration+1}/6", flush=True)


def evaluate(arm, seed, index, load, *, smoke=False):
    path = previous(seed) if arm == "previous" else OUT/"models"/str(seed)/"update-5.pt"
    net = None if arm == "rule" else restore(path)[0]
    split = "train" if smoke else "holdout"
    row, _ = simulate(split, index, load, net)
    row.update(arm=arm, model_seed=seed if net is not None else None,
               checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if net is not None else None)
    folder = "smoke" if smoke else "evaluation"
    save(OUT/folder/f"{arm}-{seed}-{index}-{load}.json", row)
    print(arm, index, load, row["vessel_complete"], row["seconds"], flush=True)
    if smoke and not row["vessel_complete"]:
        raise RuntimeError("Incomplete ship cohort; see saved stream completion times and remaining moves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train","evaluate","smoke"))
    parser.add_argument("--arm", choices=("rule","previous","trained"), default="rule")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--index", type=int, default=12)
    parser.add_argument("--load", type=int, default=3500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "train":
        train(args.seed)
    else:
        evaluate(args.arm, args.seed, args.index, args.load, smoke=args.action == "smoke")
