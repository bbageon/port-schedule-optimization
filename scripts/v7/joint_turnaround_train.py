"""Retrain crane decisions on truck minutes + 67 x port-call minutes, then replay an unseen month."""
import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from joint_turnaround_preflight import CACHE, OUT, read, save
from month_crane_input import MANIFEST as SMOKE_MANIFEST, OUT as SMOKE_INFO
from month_crane_replay import run
from resched_ppo import decode_sample as decode_state, encode_sample as encode_state
from value_stability import restore
from vessel_turnaround_experiment import OUT as PREVIOUS
from yard_rl.v7.crane.actor_critic import pack_states
from yard_rl.v7.crane.global_context import validate_observation
from yard_rl.v7.crane.value_normalization import fit_value_head, normalized_ppo_update, restart_value_head
from yard_rl.v7.reward.joint_turnaround import VESSEL_TIME_WEIGHT
from yard_rl.v7.stage.month import DAY_S
from yard_rl.v7.stage.seed_bundle import file_sha256


MODEL_SEED = 9951101
START = PREVIOUS / "models" / str(MODEL_SEED) / "update-5.pt"
UPDATES, ROLLOUTS = 4, 4
MAX_SAMPLES, VALUE_SAMPLES = 2048, 4096      # per update / per first-wave month, disjoint
SCALE = 1e6                                  # learning targets are million weighted minutes
BASELINES = {0: "SF_SPT", 1: "LWKR", 2: "previous"}   # replayed beside the training waves
SOURCES = ("scripts/v7/joint_turnaround_train.py", "scripts/v7/month_crane_replay.py",
           "src/yard_rl/v7/crane/on_policy.py", "src/yard_rl/v7/crane/value_normalization.py",
           "src/yard_rl/v7/crane/actor_critic.py", "src/yard_rl/v7/reward/joint_turnaround.py")


def setting(smoke):
    if smoke:  # three short days from an unregistered seed: wiring only, never a result
        root, info = CACHE / "smoke-training", SMOKE_INFO / "smoke-input.json"
        cfg = dict(root=root, evidence=root / "evidence", train=info, test=info,
                   manifests=SMOKE_MANIFEST.parent, updates=2, rollouts=2)
    else:
        cfg = dict(root=CACHE / "training", evidence=OUT, train=OUT / "train-input.json",
                   test=OUT / "test-input.json", manifests=CACHE, updates=UPDATES, rollouts=ROLLOUTS)
    cfg["horizon"] = len(read(cfg["train"])["days"]) * DAY_S   # the clock input stays near [0, 1.2]
    return cfg


def model_path(cfg, update):
    return START if update == 0 else cfg["root"] / "models" / f"update-{update-1}.pt"


def encode(sample, horizon):
    obs = sample.global_observation
    validate_observation(obs)
    if (abs(sample.state.time_fraction * horizon - obs["decision_time"]) > 1e-6
            or sample.state.global_waiting != obs["feature"]):
        raise ValueError("Decision state and synchronized observation differ")
    return dict(sample=encode_state(sample), global_waiting=sample.state.global_waiting, global_observation=obs)


def decode(obj):
    sample = decode_state(obj["sample"])
    sample.state.global_waiting = obj["global_waiting"]
    sample.global_observation = obj["global_observation"]
    return sample


def value_quality(samples):
    target = torch.tensor([s.target for s in samples], dtype=torch.float64)
    error = target - torch.tensor([s.old_value for s in samples], dtype=torch.float64)
    spread = float(target.std(unbiased=False))
    rmse = float(error.square().mean().sqrt())
    return dict(target_mean=float(target.mean()), target_sd=spread, value_rmse=rmse,
                explained=1 - (rmse / spread) ** 2 if spread > 0 else None)


def rollout(update, index, smoke):
    cfg = setting(smoke)
    started = time.monotonic()
    folder = cfg["root"] / "rollouts" / f"{update}-{index}"
    seed = MODEL_SEED + 2000 + 16 * update + index
    source = model_path(cfg, update)
    out = run("trained", False, info_path=cfg["train"], manifest_dir=cfg["manifests"], checkpoint=source,
              folder=folder, stochastic=True, sample_seed=seed, horizon=cfg["horizon"], target_scale=SCALE)
    resolver = out["resolver"]
    row = dict(update=update, index=index, sampling_seed=seed, checkpoint_sha256=file_sha256(source),
               complete=out["complete"], month=out["windows"]["all"], decisions=resolver.decisions)
    if out["complete"]:
        samples = resolver.samples
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed + 500000)).tolist()
        value = order[MAX_SAMPLES:MAX_SAMPLES + VALUE_SAMPLES] if update == 0 else []
        path = folder / "samples.json.gz"
        with gzip.open(path, "wt", encoding="utf8", newline="\n") as stream:
            json.dump(dict(horizon=cfg["horizon"], scale=SCALE,
                           policy=[encode(samples[i], cfg["horizon"]) for i in order[:MAX_SAMPLES]],
                           value=[encode(samples[i], cfg["horizon"]) for i in value]), stream)
        days = {}
        for s in samples:
            days.setdefault(int(s.state.time_fraction * cfg["horizon"] // DAY_S), []).append(s)
        row.update(samples_sha256=file_sha256(path), weighted_minutes=read(folder / "target-check.json")["final_cost"],
                   **value_quality(samples), samples=len(samples),
                   daily=[dict(day=d, decisions=len(rows), target=sum(s.target for s in rows) / len(rows),
                               value=sum(s.old_value for s in rows) / len(rows)) for d, rows in sorted(days.items())])
    row["seconds"] = time.monotonic() - started
    save(cfg["evidence"] / "training" / f"rollout-{update}-{index}.json", row)


def update(index, smoke):
    cfg = setting(smoke)
    rows = [read(cfg["evidence"] / "training" / f"rollout-{index}-{k}.json") for k in range(cfg["rollouts"])]
    complete = [r for r in rows if r["complete"]]
    if not complete:
        raise RuntimeError("No training month finished all work; the policy is left unchanged")
    each = -(-MAX_SAMPLES // len(complete))
    policy, value = [], []
    for r in complete:
        path = cfg["root"] / "rollouts" / f"{index}-{r['index']}" / "samples.json.gz"
        if file_sha256(path) != r["samples_sha256"]:
            raise ValueError("Stored decisions changed after the rollout")
        with gzip.open(path, "rt", encoding="utf8") as stream:
            document = json.load(stream)
        policy += [decode(x) for x in document["policy"][:each]]
        value += [decode(x) for x in document["value"]]
    policy = policy[:MAX_SAMPLES]
    net, optimizer, _ = restore(model_path(cfg, index))
    report = dict(update=index, months=[r["index"] for r in complete], policy_samples=len(policy))
    if index == 0:
        # The earlier value head predicted KRW over one day. Start it again for month-long minutes and
        # fit it alone before any action probability moves; Adam restarts for the same reason.
        optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
        with torch.no_grad():
            before = [net(pack_states([s.state for s in policy[b:b+64]]))[0] for b in range(0, len(policy), 64)]
        restart_value_head(net, seed=MODEL_SEED + 3000)
        report["value_restart"] = fit_value_head(net, value, seed=MODEL_SEED + 3001)
        with torch.no_grad():
            for j, b in enumerate(range(0, len(policy), 64)):
                logits, values = net(pack_states([s.state for s in policy[b:b+64]]))
                if not torch.equal(logits, before[j]):
                    raise ValueError("Refitting the value head changed an action logit")
                for s, v in zip(policy[b:b+64], values.tolist()):
                    s.old_value = v
    report["value_before_update"] = value_quality(policy)   # on decisions the value head was not fitted to
    report["fit"] = normalized_ppo_update(net, optimizer, policy, seed=MODEL_SEED + 1100 + index,
                                          max_samples=MAX_SAMPLES, epochs=4, batch_size=64)
    path = cfg["root"] / "models" / f"update-{index}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model_config=net.config, state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), arm="joint_turnaround", seed=MODEL_SEED, iteration=index,
                    return_scale=SCALE, return_unit=f"truck minutes + {VESSEL_TIME_WEIGHT} x port-call minutes",
                    horizon_s=cfg["horizon"]), tmp)
    tmp.replace(path)
    report["checkpoint_sha256"] = file_sha256(path)
    save(cfg["evidence"] / "training" / f"fit-{index}.json", report)


def evaluate(arm, smoke):
    cfg = setting(smoke)
    kw = dict(info_path=cfg["test"], manifest_dir=cfg["manifests"], horizon=cfg["horizon"],
              folder=cfg["evidence"] / "evaluation" / arm)
    if arm == "retrained":
        run("trained", False, checkpoint=cfg["root"] / "models" / f"update-{cfg['updates']-1}.pt", **kw)
    elif arm == "previous":
        run("trained", False, checkpoint=START, **kw)
    else:
        run(arm, False, **kw)


def note(text):
    print(datetime.now().strftime("%m-%d %H:%M:%S"), text, flush=True)


def wave(cfg, jobs, flag):
    """Run the unfinished jobs side by side; one failure stops the campaign with its log kept."""
    running = {}
    for name, (done, arguments) in jobs.items():
        if done.exists():
            continue
        log = cfg["root"] / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        running[name] = subprocess.Popen(
            [sys.executable, "-X", "utf8", "scripts/v7/joint_turnaround_train.py", *arguments, *flag],
            stdout=log.open("w", encoding="utf8"), stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": "src"})
    note(f"started {sorted(running)}")
    failed = [name for name, process in running.items() if process.wait() != 0]
    if failed:
        raise RuntimeError(f"Failed {failed}; logs under {cfg['root'] / 'logs'}")
    note(f"finished {sorted(running)}")


def campaign(smoke):
    cfg = setting(smoke)
    flag = ["--smoke"] if smoke else []
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    frozen = dict(
        configuration=dict(model_seed=MODEL_SEED, updates=cfg["updates"], months_per_update=cfg["rollouts"],
            max_samples=MAX_SAMPLES, value_samples_per_first_month=VALUE_SAMPLES, epochs=4, batch_size=64,
            lr=.0003, clip=.2, value_coefficient=.5, entropy_coefficient=.01, gradient_clip=1., discount=1.,
            return_scale=SCALE, vessel_time_weight=VESSEL_TIME_WEIGHT, horizon_s=cfg["horizon"],
            collection="stochastic", evaluation="fixed argmax; last update only, no checkpoint selection",
            baselines=BASELINES, train_input=read(cfg["train"])["sha256"], test_input=read(cfg["test"])["sha256"]),
        sources={**{p: file_sha256(p) for p in SOURCES}, START.as_posix(): file_sha256(START)},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    path = cfg["evidence"] / "training" / "campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path)["sources"] != frozen["sources"] or read(path)["configuration"] != json.loads(json.dumps(frozen["configuration"])):
            raise RuntimeError("Code, start model or settings changed after the campaign began")
    else:
        if dirty and not smoke:
            raise RuntimeError("Commit the code before a registered run")
        save(path, frozen)
    for u in range(cfg["updates"]):
        jobs = {f"rollout-{u}-{k}": (cfg["evidence"] / "training" / f"rollout-{u}-{k}.json",
                                    ["rollout", "--update", str(u), "--index", str(k)])
                for k in range(cfg["rollouts"])}
        if u in BASELINES:
            jobs[f"evaluate-{BASELINES[u]}"] = (cfg["evidence"] / "evaluation" / BASELINES[u] / "result.json",
                                                ["evaluate", "--arm", BASELINES[u]])
        wave(cfg, jobs, flag)
        if not (cfg["root"] / "models" / f"update-{u}.pt").exists():
            update(u, smoke)
            note(f"update {u+1}/{cfg['updates']} applied")
    wave(cfg, {f"evaluate-{arm}": (cfg["evidence"] / "evaluation" / arm / "result.json", ["evaluate", "--arm", arm])
               for arm in ("retrained", *BASELINES.values())}, flag)
    note("campaign complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("campaign", "rollout", "update", "evaluate"))
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--arm", choices=("retrained", *BASELINES.values()), default="retrained")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "campaign":
        campaign(args.smoke)
    elif args.action == "rollout":
        rollout(args.update, args.index, args.smoke)
    elif args.action == "update":
        update(args.update, args.smoke)
    else:
        evaluate(args.arm, args.smoke)
