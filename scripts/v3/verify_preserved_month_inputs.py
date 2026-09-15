"""Rebuild fixed inputs in a frozen checkout, comparing hashes without policy outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"


def init_worker(source, cpus):
    sys.path.insert(0, str(Path(source) / "src"))
    os.sched_setaffinity(0, {cpus.get(timeout=30)})
    os.nice(10)


def verify(row):
    from yard_rl.v3.eval.seed_bank import create_month_payload, digest
    payload = create_month_payload(row["seed"])
    actual = digest(payload)
    return {"seed": row["seed"], "requests": len(payload["schedule"]),
        "expected_payload_sha256": row["payload_sha256"], "actual_payload_sha256": actual,
        "identical": actual == row["payload_sha256"], "cpu": next(iter(os.sched_getaffinity(0)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.source, text=True).strip():
        raise RuntimeError("Frozen generator checkout is dirty")
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    if 17 * 1.5 * 1024**2 > .8 * available:
        raise RuntimeError("Input verifier exceeds available memory budget")
    expected = json.loads(args.expected.read_text())
    ctx = multiprocessing.get_context("spawn")
    cpu_queue = ctx.Queue()
    for cpu in range(3, 20):
        cpu_queue.put(cpu)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=17, mp_context=ctx, initializer=init_worker,
            initargs=(str(args.source), cpu_queue)) as pool:
        rows = list(pool.map(verify, expected["rows"]))
    result = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=args.source, text=True).strip(), "code_dirty": False,
        "expected_file": str(args.expected), "expected_sha256": hashlib.sha256(args.expected.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows, "passed": len(rows) == 20 and all(r["identical"] for r in rows),
        "months": len(rows), "requests": sum(r["requests"] for r in rows),
        "elapsed_s": time.monotonic() - start, "cpus": list(range(3, 20)),
        "policy_simulations": 0, "new_training_runs": 0}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "months", "requests", "elapsed_s")}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
