"""Queue the registered ETA/ETW comparison after the four BP updates and held-out month."""
import argparse
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BP = Path("outputs/reports/yr332_short_window")
BL = Path("outputs/reports/yr332_joint_turnaround")
OUT = Path("outputs/reports/yr332_planned_times")
CACHE = Path("build/yr332_joint_turnaround")
ARMS = ("eta", "etw", "both", "both_need")
SOURCE = Path(__file__).resolve()
SOURCES = (SOURCE.name, "planned_time_comparison.py", "short_window_train.py", "short_window_samples.py",
           "crane_devices.py", "value_stability.py",
           "month_crane_replay.py", "joint_turnaround_train.py", "local_return_scope.py")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(p)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@contextmanager
def lock(path, wait=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if not wait or exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(10)
        yield  # closing the file also releases the process lock after an exception


def finished(out, root):
    result = out / "evaluation/retrained/result.json"
    if not result.is_file():
        return False
    manifest = read(BL / "test-input.json")
    r = read(result)
    for u in range(4):
        ck = root / "models" / f"update-{u}.pt"
        fit = read(out / "training" / f"fit-{u}.json")
        if fit["update"] != u or sha(ck) != fit["checkpoint_sha256"]:
            raise ValueError(f"Invalid trained checkpoint {ck}")
    window = r["windows"]["all"]
    if (r["checkpoint_sha256"] != fit["checkpoint_sha256"] or r["manifest_sha256"] != manifest["sha256"]
        or r["days"] != 30 or window["unfinished_trucks"] or window["unfinished_ships"]
        or window["trucks"] != manifest["orders"] or window["ships"] < 1):
        raise ValueError(f"Invalid prerequisite evaluation in {result}")
    return True


def contract():
    paths = [Path("scripts/v7") / name for name in SOURCES]
    paths += [Path("src/yard_rl/v7") / p for p in (
        "crane/planned_times.py", "crane/actor_critic.py", "crane/on_policy.py", "crane/value_normalization.py",
        "reward/joint_turnaround.py", "stage/orders.py")]
    paths += [BL / "train-input.json", BL / "test-input.json", BP / "train-campaign.json",
              Path(".claude/docs/dashboard-task-specs/YR-332-bt-planned-times.md")]
    return {str(p.as_posix()): sha(p) for p in paths}


def status(stage, **details):
    save(OUT / "queue-status.json", dict(stage=stage, pid=os.getpid(),
         at=time.strftime("%Y-%m-%d %H:%M:%S %z"), **details))


def queued(interval, registry):
    with lock(CACHE / "planned-times/queue.lock"):
        try:
            registration = read(registry)
            if registration["sources"] != contract():
                raise ValueError("Registered experiment sources or inputs changed")
            for phase, device in registration.get("devices", dict(train="cpu", inference="cpu")).items():
                os.environ[f"YARD_{phase.upper()}_DEVICE"] = device
            status("waiting_for_bp")
            while not finished(BP, CACHE / "short-window"):
                time.sleep(interval)
            if registration["sources"] != contract():
                raise ValueError("Registered experiment sources or inputs changed during wait")
            from planned_time_comparison import compare, paired, read_month
            import torch
            torch.set_num_threads(1)
            manifest = read(BL / "test-input.json")["sha256"]
            baseline = read_month(BP / "evaluation/retrained", manifest)
            rule = read_month(BL / "evaluation/SF_SPT", manifest)
            if baseline["cohort"] != rule["cohort"]:
                raise ValueError("BP and the rule have different held-out demand")
            smoke_record = OUT / "both/smoke/smoke-check.json"
            if not smoke_record.is_file():
                status("checking_three_day_replay")
                subprocess.run([sys.executable, "-u", "scripts/v7/short_window_train.py", "smoke",
                                "--planned-times", "both"], check=True)
            if not read(smoke_record)["evaluation"]["complete"]:
                raise ValueError("The three-day check left work unfinished")
            paired(smoke=True)
            for arm in ARMS:
                if registration["sources"] != contract():
                    raise ValueError("Registered experiment changed during wait")
                if finished(OUT / arm, CACHE / "planned-times" / arm):
                    continue
                status("training", arm=arm)
                subprocess.run([sys.executable, "-u", "scripts/v7/short_window_train.py", "train-campaign",
                                "--planned-times", arm], check=True)
                if not finished(OUT / arm, CACHE / "planned-times" / arm):
                    raise ValueError(f"Missing final evaluation for {arm}")
            compare()
            status("paired_first_action")
            paired()
            status("complete")
        except BaseException as exc:
            status("failed", error=repr(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "wait", "status", "smoke-pair"))
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--train-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    devices = dict(train=args.train_device, inference=args.inference_device)
    registry = OUT / ("registration.json" if set(devices.values()) == {"cpu"} else
                      f"registration-{args.train_device}-{args.inference_device}.json")
    if args.action == "register":
        sources = contract()
        dest = registry
        if dest.exists() and read(dest)["sources"] != sources:
            raise ValueError("Refusing to overwrite an existing registration")
        if not dest.exists():
            if "cuda" in devices.values() and any((OUT / arm).exists() for arm in ARMS):
                raise ValueError("Device change requires a fresh planned-time campaign")
            save(dest, dict(sources=sources, arms=ARMS, devices=devices, baseline="completed BP, identical start and seeds",
                            commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()))
    elif args.action == "wait":
        if args.interval < 1:
            parser.error("interval must be positive")
        queued(args.interval, registry)
    elif args.action == "smoke-pair":
        os.environ["YARD_INFERENCE_DEVICE"] = args.inference_device
        import torch
        torch.set_num_threads(1)
        from planned_time_comparison import paired
        paired(smoke=True)
    else:
        print((OUT / "queue-status.json").read_text(encoding="utf8"))
