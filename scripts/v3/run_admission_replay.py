"""Run three isolated admission-observer replays, preserving each continuous history."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_request_audit import ROOT, now, run_one, sha

ARMS = ("NO_REALLOC", "RL", "RL_NOVETO")


def save(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def parse_cpus(value):
    cpus = tuple(int(v) for v in value.split(","))
    if len(cpus) != 3 or len(set(cpus)) != 3 or min(cpus) < 0:
        raise ValueError("Exactly three distinct nonnegative CPU IDs are required.")
    return cpus


def validate_cpus(cpus):
    if not hasattr(os, "sched_getaffinity") or not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError("Run under Linux/WSL with all three requested CPUs available.")


def source_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def child(args):
    import torch
    from yard_rl.v3.stage.month import plan_days, plan_month
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = args.out / args.arm
    if folder.exists():
        raise FileExistsError(folder)
    days = plan_days(9_900_700, (300,)) if args.phase == "smoke" else [
        replace(d, n_days=10) for d in plan_month(9_500_000)[:10]]
    try:
        args.diagnose_admissions, args.isolated_progress = True, True
        result = run_one(args.arm, args.arm, 9_900_700 if args.phase == "smoke" else 9_500_000,
                         days, args, sha(args.checkpoint))
        save(folder / "progress.json", {"state": "completed", "arm": args.arm,
            "phase": args.phase, "at": now(), "pid": os.getpid(), "cpu": args.cpu,
            "days_completed": len(days), "cost_krw": sum(d["phi_krw"] for d in result["days"]),
            "requests": result["request_summary"], "vessel_work": result["vessel_work_summary"]})
    except BaseException:
        folder.mkdir(parents=True, exist_ok=True)
        failure = {"state": "failed", "at": now(), "arm": args.arm, "traceback": traceback.format_exc()}
        save(folder / "failure.json", failure)
        save(folder / "progress.json", failure)
        raise


def run_phase(args, phase):
    out = args.out / phase
    out.mkdir(exist_ok=False)
    workers = {}
    try:
        for arm, cpu in zip(ARMS, args.cpus):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(out),
                "--checkpoint", args.checkpoint, "--prereg", args.prereg,
                "--arm", arm, "--cpu", str(cpu), "--phase", phase]
            with (out / f"{arm}.log").open("xb") as stream:
                workers[arm] = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        save(out / "launch.json", {"started_at": now(), "source_commit": source_commit(),
            "workers": {arm: {"pid": p.pid, "cpu": cpu} for (arm, p), cpu in zip(workers.items(), args.cpus)}})
        while True:
            statuses = {arm: p.poll() for arm, p in workers.items()}
            save(args.out / "progress.json", {"state": "running", "phase": phase, "at": now(),
                "exit_codes": statuses, "workers": {a: p.pid for a, p in workers.items()},
                "progress_files": {a: str(out / a / "progress.json") for a in ARMS}})
            if any(code not in (None, 0) for code in statuses.values()):
                raise RuntimeError(f"Worker failure; partial evidence retained: {statuses}")
            if all(code == 0 for code in statuses.values()):
                break
            time.sleep(5)
    finally:
        # Only stop child processes created by this supervisor, never other experiments.
        for process in workers.values():
            if process.poll() is None:
                process.terminate()
        for process in workers.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    from analyze_admission_replay import summarize
    summary = summarize(out, expected_commit=source_commit(),
                        previous=Path(args.previous) if phase == "full" else None)
    if not summary["recording_passed"]:
        raise RuntimeError("Independent ledger reconciliation failed; see summary.json.")
    if phase == "smoke" and any(r["requests"]["states"] != {"COMPLETED": 300} or
        not r["vessel_work"]["all_requested_work_completed"] for r in summary["arms"].values()):
        raise RuntimeError("Smoke workload did not fully complete; do not launch the full replay.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--previous")
    parser.add_argument("--cpus", type=parse_cpus, default=(0, 1, 2))
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--phase", choices=("smoke", "full"))
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out = args.out.resolve()
    args.checkpoint, args.prereg = str(Path(args.checkpoint).resolve()), str(Path(args.prereg).resolve())
    if args.arm:
        child(args)
        return
    validate_cpus(args.cpus)
    from yard_rl.integrated.repro import code_dirty
    if code_dirty() is not False:
        raise RuntimeError("Use a clean committed checkout.")
    if not args.previous or not all((Path(args.previous) / arm / "result.json").exists() for arm in ARMS):
        raise RuntimeError("All three prior diagnostic results are required.")
    if args.launch:
        args.out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(args.out),
            "--checkpoint", args.checkpoint, "--prereg", args.prereg, "--previous", args.previous,
            "--cpus", ",".join(map(str, args.cpus))]
        with (args.out / "supervisor.log").open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        launch = {"pid": process.pid, "at": now(), "source_commit": source_commit(),
            "source_checkout": str(ROOT), "cpus": args.cpus, "command": command,
            "checkpoint_sha256": sha(args.checkpoint), "prereg_sha256": sha(args.prereg),
            "runner_sha256": sha(__file__), "user_cpu_limit": 14}
        save(args.out / "launch.json", launch)
        print(json.dumps(launch, ensure_ascii=False))
        return
    try:
        run_phase(args, "smoke")
        run_phase(args, "full")
        save(args.out / "progress.json", {"state": "completed", "phase": "full", "at": now(),
            "result": str(args.out / "full/summary.json"), "claim_eligible": False})
    except BaseException:
        failure = {"state": "failed", "at": now(), "traceback": traceback.format_exc()}
        save(args.out / "failure.json", failure)
        save(args.out / "progress.json", failure)
        raise


if __name__ == "__main__":
    main()
