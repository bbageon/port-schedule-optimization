"""Fixed request-preservation diagnosis, with smoke checks before the known 10-day failure."""
import argparse
from dataclasses import replace
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
from run_request_audit import ROOT, now, run_one, sha
from run_admission_replay import save, source_commit

ARMS = ("NO_REALLOC", "RL", "RL_NOVETO")


def child(args):
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    import torch
    torch.set_num_threads(1)
    from yard_rl.v3.stage.month import plan_days, plan_month
    seed = 9_900_700 if args.phase == "smoke" else 9_500_000
    days = plan_days(seed, (300,)) if args.phase == "smoke" else [
        replace(d, n_days=10) for d in plan_month(seed)[:10]]
    args.diagnose_admissions, args.isolated_progress = True, True
    folder = args.out / args.arm
    try:
        result = run_one(args.arm, args.arm, seed, days, args, sha(args.checkpoint))
        save(folder / "progress.json", {"state": "completed", "at": now(),
            "pid": os.getpid(), "cpu": args.cpu, "phase": args.phase,
            "requests": result["request_summary"], "vessels": result["vessel_work_summary"]})
    except BaseException:
        folder.mkdir(parents=True, exist_ok=True)
        failure = {"state": "failed", "at": now(), "traceback": traceback.format_exc()}
        save(folder / "failure.json", failure)
        save(folder / "progress.json", failure)
        raise


def phase(args, name):
    folder = args.out / name
    folder.mkdir(exist_ok=False)
    workers = {}
    try:
        for cpu, arm in enumerate(ARMS):
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--out", str(folder), "--checkpoint", args.checkpoint, "--prereg", args.prereg,
                "--phase", name, "--arm", arm, "--cpu", str(cpu), "--admission-mode", "PRESERVE"]
            with (folder / f"{arm}.log").open("xb") as log:
                workers[arm] = subprocess.Popen(command, cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        save(folder / "launch.json", {"at": now(), "source_commit": source_commit(),
            "processes": {arm: {"pid": p.pid, "cpu": i} for i, (arm, p) in enumerate(workers.items())}})
        while True:
            codes = {arm: p.poll() for arm, p in workers.items()}
            save(args.out / "progress.json", {"state": "running", "phase": name,
                "at": now(), "exit_codes": codes})
            if any(c not in (None, 0) for c in codes.values()):
                raise RuntimeError(f"Child failed; preserve partial evidence: {codes}")
            if all(c == 0 for c in codes.values()):
                break
            time.sleep(5)
    finally:
        for p in workers.values():
            if p.poll() is None:
                p.terminate()
        for p in workers.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    results = {a: json.loads((folder / a / "result.json").read_text()) for a in ARMS}
    checks = {"same_requests": len({r["requested_identity_sha256"] for r in results.values()}) == 1,
        "no_request_loss": all(r["request_summary"]["all_requests_admitted"] and
            r["request_summary"]["unprocessed"] == 0 for r in results.values()),
        "full_vessel_requests_admitted": all(r["vessel_work_summary"]["unadmitted_moves"] == 0 for r in results.values()),
        "recording": all(all(r["recording_checks"].values()) for r in results.values())}
    summary = {"phase": name, "at": now(), "checks": checks, "passed": all(checks.values()),
        "arms": {a: {"total_cost_krw": sum(d["phi_krw"] for d in r["days"]),
            "requests": r["request_summary"], "vessels": r["vessel_work_summary"],
            "elapsed_s": r["elapsed_s"]} for a, r in results.items()},
        "performance_confirmed": False, "new_training_runs": 0}
    save(folder / "summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Preservation checks failed; do not advance")
    if name == "smoke" and any(r["request_summary"]["states"] != {"COMPLETED": 300}
            or not r["vessel_work_summary"]["all_requested_work_completed"] for r in results.values()):
        raise RuntimeError("Short connection test did not complete all requests; do not run full diagnosis")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--phase", choices=("smoke", "full"))
    parser.add_argument("--admission-mode", default="PRESERVE", choices=("PRESERVE",))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.arm:
        return child(args)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Use a clean frozen checkout")
    if args.launch:
        checkpoint_hash, prereg_hash = sha(args.checkpoint), sha(args.prereg)
        available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
        if 3 * 3 * 1024**2 > .8 * available:
            raise RuntimeError("Three policy workers exceed the 80% memory budget")
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out / "supervisor.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--out", str(args.out), "--checkpoint", args.checkpoint, "--prereg", args.prereg],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(args.out / "launch.json", {"at": now(), "pid": process.pid,
            "source_commit": source_commit(), "user_cpu_limit": 20, "active_policy_workers": 3,
            "cpus": [0, 1, 2], "checkpoint_sha256": checkpoint_hash, "prereg_sha256": prereg_hash})
        print(json.dumps({"pid": process.pid, "out": str(args.out)}))
        return
    os.sched_setaffinity(0, set(range(20)))
    try:
        phase(args, "smoke")
        phase(args, "full")
        save(args.out / "progress.json", {"state": "completed", "at": now(),
            "performance_confirmed": False, "new_training_runs": 0})
    except BaseException:
        save(args.out / "failure.json", {"at": now(), "traceback": traceback.format_exc()})
        save(args.out / "progress.json", {"state": "failed", "at": now()})
        raise


if __name__ == "__main__":
    main()
