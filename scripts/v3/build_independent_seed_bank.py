"""Build and verify independent v3 input bundles, preserving the existing curve."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(ROOT / "src"))
from yard_rl.v3.eval.seed_bank import (canonical_bytes, create_month_payload, digest,
    file_sha, generator_contract, load_bundle, validate_payload, validate_seeds, write_bundle)


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_sha(path):
    raw = path.read_bytes()
    if path.suffix in (".py", ".yaml", ".yml", ".json", ".csv", ".md"):
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def prepare_spec(path, *, base=20_000_000, count=20, created_date='2026-09-16'):
    if path.exists():
        raise FileExistsError(path)
    prior, inputs, skipped = set(), {}, []
    patterns = [ROOT / "outputs/v3", ROOT / "outputs/reports/yr317_v3_request_audit"]
    regex = re.compile(r'"(?:seed|seed_base|base_seed|environment_seed|init_seed)"\s*:\s*(\d+)')
    for folder in patterns:            # `base` 는 대역 시작값이므로 이름을 가리면 안 된다
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or source.suffix not in (".json", ".jsonl"):
                continue
            if source.stat().st_size > 20_000_000:
                skipped.append(source.relative_to(ROOT).as_posix())
                continue
            raw = source.read_text(encoding="utf-8")
            found = {int(n) for n in regex.findall(raw)}
            if found:
                prior.update(found)
                inputs[source.relative_to(ROOT).as_posix()] = file_sha(source)
    seeds = [base + 100_000 * i for i in range(count)]
    # Conservatively reserve every observed seed's daily/background offset family.
    reserved = sorted({s + offset * 1000 for s in prior for offset in range(52)})
    domains = validate_seeds(seeds, prior_seeds=reserved)
    source_hashes = {p.relative_to(ROOT).as_posix(): source_sha(p)
        for p in sorted((ROOT / "src/yard_rl/v3").rglob("*.py"))}
    for p in sorted((ROOT / "configs").rglob("*")):
        if p.is_file():
            source_hashes[p.relative_to(ROOT).as_posix()] = source_sha(p)
    spec = {"schema": "yard_rl.v3.independent-seed-bank-spec.v1", "created_date": created_date,
        "seeds": seeds, "seed_domains": domains, "generator_contract": generator_contract(),
        "prior_observed_seeds": sorted(prior), "prior_reserved_numeric_seeds": reserved,
        "prior_input_sha256": inputs, "prior_files_too_large_to_scan": skipped,
        "source_contract": source_hashes, "max_workers": 14,
        "selection_rule": "fixed arithmetic sequence; no rejection/replacement by load or performance",
        "scope": f"{count} input months; final policy evaluation budget is a separate preregistration",
        "band_base": base,
        "performance_evaluation_started": False, "new_training_runs": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, spec)
    print(json.dumps({"spec": str(path), "months": len(seeds), "prior_seeds": len(prior),
        "skipped_files": skipped, "contract": spec["generator_contract"]}, ensure_ascii=True))


def worker_init(cpu_queue):
    import torch
    os.sched_setaffinity(0, {cpu_queue.get(timeout=30)})
    os.nice(10)
    torch.set_num_threads(1)


def build_one(seed, folder):
    cpu = next(iter(os.sched_getaffinity(0)))
    start = time.monotonic()
    payload = create_month_payload(seed)
    metrics = validate_payload(payload)
    path = Path(folder) / f"seed-{seed}.json.gz"
    packed_sha = write_bundle(path, payload)
    restored = load_bundle(path, expected_sha256=packed_sha)
    if digest(restored) != metrics["payload_sha256"]:
        raise AssertionError("Serialized input changed")
    metrics.update(bundle=path.name, bundle_sha256=packed_sha,
        elapsed_s=time.monotonic() - start, pid=os.getpid(), cpu=cpu)
    save(path.with_suffix("").with_suffix(".summary.json"), metrics)
    return metrics


def build_bank(args):
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Use Linux/WSL for explicit CPU allocation")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec["generator_contract"] != generator_contract():
        raise ValueError("Generator contract drift")
    validate_seeds(spec["seeds"], prior_seeds=spec["prior_reserved_numeric_seeds"])
    for name, expected in spec["source_contract"].items():
        p = ROOT / name
        actual = source_sha(p)
        if actual != expected:
            raise ValueError(f"Source changed after input preregistration: {name}")
    if not 1 <= args.workers <= spec["max_workers"]:
        raise ValueError("Worker budget exceeded")
    # 호출자가 taskset 으로 고른 코어를 존중한다. 예전에는 0..workers-1 을 고정으로
    # 요구해 다른 작업이 그 코어를 쓰고 있으면 실행 자체가 불가능했다. 예산(workers)은
    # 그대로 지킨다.
    cpus = sorted(os.sched_getaffinity(0))[:args.workers]
    if len(cpus) < args.workers:
        raise ValueError("Requested CPU IDs unavailable")
    os.sched_setaffinity(0, set(cpus))
    available_kib = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    if args.workers * 1.5 * 1024**2 > 0.8 * available_kib:
        raise ValueError("Worker memory budget exceeds 80% of available memory")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Commit code and use a clean execution checkout")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "manifest.json", {"source_commit": commit, "code_dirty": False,
        "spec_sha256": file_sha(args.spec), "spec": spec, "workers": args.workers,
        "cpus": cpus, "available_memory_kib": available_kib, "new_training_runs": 0})
    start, rows = time.monotonic(), []
    context = multiprocessing.get_context("spawn")
    cpu_queue = context.Queue()
    for cpu in cpus:
        cpu_queue.put(cpu)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
            initializer=worker_init, initargs=(cpu_queue,)) as pool:
        futures = {pool.submit(build_one, s, args.out): s for s in spec["seeds"]}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except BaseException as exc:
                save(args.out / "failure.json", {"seed": futures[future], "error": repr(exc)})
                raise
            save(args.out / "progress.json", {"state": "generating", "completed": len(rows),
                "total": len(futures), "elapsed_s": time.monotonic() - start})
            print(json.dumps({"completed": len(rows), "total": len(futures),
                "seed": rows[-1]["seed"], "requests": rows[-1]["requested_trucks"]}), flush=True)
    rows.sort(key=lambda row: row["seed"])
    # Rebuild the first fixed month to verify input reproducibility, never to select it.
    repeated = create_month_payload(spec["seeds"][0])
    repeat_matches = digest(repeated) == rows[0]["payload_sha256"]
    unique = len({row["schedule_sha256"] for row in rows}) == len(rows)
    summary = {"schema": "yard_rl.v3.independent-seed-bank-result.v1", "months": len(rows),
        "days_per_month": 30, "total_requests": sum(r["requested_trucks"] for r in rows),
        "all_curves_preserved": all(r["curve_exactly_preserved"] for r in rows),
        "unique_requests_per_seed": unique, "first_month_regeneration_matches": repeat_matches,
        "passed": unique and repeat_matches and len(rows) == len(spec["seeds"]),
        "rows": rows, "elapsed_s": time.monotonic() - start,
        "performance_evaluations": 0, "new_training_runs": 0,
        "scenario_completion_validated": False, "claim_eligible": False}
    save(args.out / "summary.json", summary)
    save(args.out / "progress.json", {"state": "completed", "months": len(rows), "passed": summary["passed"]})
    if not summary["passed"]:
        raise AssertionError("Input bank verification failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--prepare-spec", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--base", type=int, default=20_000_000,
                   help="시드 대역의 시작값. 기존 판정 대역은 20,000,000")
    p.add_argument("--count", type=int, default=20, help="달 수")
    p.add_argument("--created-date", default="2026-09-16")
    args = p.parse_args()
    if args.prepare_spec:
        prepare_spec(args.spec, base=args.base, count=args.count,
                     created_date=args.created_date)
    else:
        if args.out is None:
            p.error("--out is required for generation")
        build_bank(args)


if __name__ == "__main__":
    main()
