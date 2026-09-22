"""YR-317-g: fixed pilot, then three policies on the existing consent-run prefix.

Run from a clean frozen checkout. All outputs go to an explicit external directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def run_one(label, arm, seed, days, args, checkpoint_hash):
    from yard_rl.integrated.repro import repro_stamp, code_dirty
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import arm_contract, digest, network_identity, runtime_identity, write_json
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
    from yard_rl.v3.stage.month_run import run_month

    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=False)
    if sha(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during the experiment.")
    seller, buyer, _ = _load_nets(args.checkpoint)
    weights_before = [network_identity(net) for net in (seller, buyer)]
    job = dict(_label=label, arm=arm, seed=seed, days=days,
               seller_net=seller, buyer_net=buyer, capture_requests=True,
               capture_daily=getattr(args, "capture_daily", True),
               daily_sample_s=getattr(args, "daily_sample_s", 300.0))
    if getattr(args, "expected_input", None) is not None:
        job["expected_input"] = args.expected_input
    if getattr(args, "admission_mode", "LEGACY") != "LEGACY":
        job["admission_mode"] = args.admission_mode
    if getattr(args, "supply_mode", "ORIGINAL") != "ORIGINAL":
        job["supply_mode"] = args.supply_mode
    if getattr(args, "candidate_pruning", "legacy") != "legacy":
        job["candidate_pruning"] = args.candidate_pruning
    if getattr(args, "measure_latency", False):
        job["measure_latency"] = True
    diagnostics = getattr(args, "diagnose_admissions", False)
    if diagnostics:
        job["diagnose_admissions"] = True
    contract = arm_contract(job, runtime_identity(), run_month)
    stamp = repro_stamp(experiment=getattr(args, "experiment", "YR-317-g"), seeds={"base": [seed],
        "days": [d.seed for d in days]}, params={"run": contract["settings"]},
        prereg=args.prereg, extra={"checkpoint_sha256": checkpoint_hash,
            "prereg_sha256": sha(args.prereg), "runner_sha256": sha(__file__)})
    if stamp["code"]["git_dirty"] is not False or code_dirty() is not False:
        raise RuntimeError("Experiment checkout is not verified clean.")
    start = time.monotonic()
    write_json(out / "manifest.json", {"started_at": now(), "contract": contract,
        "repro": stamp, "checkpoint": args.checkpoint, "purpose": getattr(args, "purpose",
            "request-record diagnosis; no confirmatory inference")})
    print(json.dumps({"event": "started", "arm": label, "days": len(days), "seed": seed}, ensure_ascii=False), flush=True)

    def on_day(day):
        row = {"at": now(), "elapsed_s": time.monotonic() - start, **day.as_dict()}
        with (out / "days.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress = {"at": now(), "state": "running", "arm": label,
                    "day_index_completed": day.index, "days_planned": len(days),
                    "elapsed_s": row["elapsed_s"], "truck_skipped": day.truck_skipped,
                    "skip_reasons": day.truck_skip_reasons}
        progress_root = out if getattr(args, "isolated_progress", False) else Path(args.out)
        write_json(progress_root / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    reset_rollout_calls()
    kwargs = {k: v for k, v in job.items() if k != "_label"}
    state_path = out / "operating-state.jsonl.gz"
    if job['capture_daily']:
        with gzip.open(state_path, 'wt', encoding='utf-8') as stream:
            def on_state(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                if row['boundary']:
                    stream.flush()
            res = run_month(**kwargs, on_day=on_day, on_observation=on_state)
    else:
        res = run_month(**kwargs, on_day=on_day)
    ledger_path = out / "requests.jsonl.gz"
    with gzip.open(ledger_path, "wt", encoding="utf-8") as stream:
        for row in res.request_ledger:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    requested_identity = [{k: row[k] for k in ("job_id", "requested_day", "flow",
        "requested_block", "requested_arrival_s", "lead_s", "requested_target", "travel_s")}
        for row in res.request_ledger]
    checks = {"request_recording": res.request_summary["recording_ok"],
              "announcer_counts": res.request_summary["announcer_counts_match"],
              "wait_cost_reconciles": math.isclose(res.request_summary["accounted_wait_krw"],
                  sum(d.c_wait for d in res.days), rel_tol=1e-10, abs_tol=1e-4),
              "no_teacher": rollout_calls() == 0, "no_policy_exceptions": res.policy_exceptions == 0,
              "frozen_networks": weights_before == [network_identity(net) for net in (seller, buyer)],
              "checkpoint_unchanged": sha(args.checkpoint) == checkpoint_hash}
    result = {"arm": arm, "label": label, "seed": seed, "finished_at": now(),
        "elapsed_s": time.monotonic() - start, "plan": [asdict(d) for d in days],
        "days": [d.as_dict() for d in res.days], "traded": res.traded_edges,
        "space": res.n_space, "time": res.n_time, "txn_failed": res.txn_failed,
        "policy_exceptions": res.policy_exceptions, "rollout_calls": rollout_calls(),
        "request_summary": res.request_summary, "vessel_admissions": res.vessel_admissions,
        "requested_identity_sha256": digest(requested_identity),
        "request_ledger_sha256": sha(ledger_path), "recording_checks": checks,
        "claim_eligible": False, "repro": stamp,
        "candidate_pruning": res.candidate_pruning, "online_latency": res.online_latency,
        "supply_plan_audit": res.supply_plan_audit}
    if job['capture_daily']:
        daily_path = out / 'daily-final.jsonl'
        with daily_path.open('w', encoding='utf-8') as stream:
            for day in res.days:
                stream.write(json.dumps(dict(seed=seed, arm=arm, policy_label=label, **day.as_dict()),
                                        ensure_ascii=False, allow_nan=False) + '\n')
        result['daily_observation'] = res.daily_observation
        result['daily_artifacts'] = {
            'operating-state.jsonl.gz': sha(state_path), 'daily-final.jsonl': sha(daily_path)}
        checks['daily_observation_complete'] = all(
            d.operational.get('day_index') == d.index and 'cohort' in d.operational for d in res.days)
    if diagnostics:
        checks["vessel_recording"] = res.vessel_work_summary["recording_ok"]
        checks["container_identity_chain"] = res.container_flow_summary["passed"]
        link_path = out / "container-links.jsonl.gz"
        with gzip.open(link_path, "wt", encoding="utf-8") as stream:
            for row in res.container_links:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        result["container_flow_summary"] = res.container_flow_summary
        result["container_links_sha256"] = sha(link_path)
        checks["failure_inventory_recorded"] = all(
            e.get("reason") == "TAIL" or "inventory_at_failure" in e
            for row in res.request_ledger for e in row["admission_events"]
            if e["outcome"] == "SKIPPED")
        result.update(vessel_work_ledger=res.vessel_work_ledger,
                      vessel_work_summary=res.vessel_work_summary)
    if getattr(args, "admission_mode", "LEGACY") == "PRESERVE":
        result["demand_bindings"] = res.demand_bindings
        checks["physical_invariants_enabled"] = res.request_summary["physical_invariants_enabled"]
    write_json(out / "result.json", result)
    print(json.dumps({"event": "finished", "arm": label, "elapsed_s": result["elapsed_s"],
        "requests": res.request_summary, "checks": checks}, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"Recording checks failed for {label}; evidence preserved.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.out).resolve()
    args.out = str(out)
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.prereg = str(Path(args.prereg).resolve())
    if args.launch:
        out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", args.out,
            "--checkpoint", args.checkpoint, "--prereg", args.prereg]
        with (out / "run.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        info = {"pid": process.pid, "launched_at": now(), "command": command,
                "source_checkout": str(ROOT), "log": str(out / "run.log")}
        (out / "launch.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(info, ensure_ascii=False))
        return 0

    import torch
    from yard_rl.v3.eval.contracts import write_json
    from yard_rl.v3.stage.month import plan_days, plan_month
    torch.set_num_threads(1)
    if hasattr(os, "nice"):
        os.nice(10)
    write_json(out / "progress.json", {"at": now(), "state": "starting", "pid": os.getpid()})
    try:
        checkpoint_hash = sha(args.checkpoint)
        pilot = run_one("pilot_RL", "RL", 9_900_700, plan_days(9_900_700, (300,)), args, checkpoint_hash)
        days = [replace(day, n_days=10) for day in plan_month(9_500_000)[:10]]
        results = []
        for arm in ("NO_REALLOC", "RL", "RL_NOVETO"):
            result = run_one(arm, arm, 9_500_000, days, args, checkpoint_hash)
            results.append(result)
            if len({r["requested_identity_sha256"] for r in results}) != 1:
                raise RuntimeError("Policies received different requested work.")
        write_json(out / "summary.json", {"completed_at": now(), "claim_eligible": False,
            "purpose": "diagnostic replay of an already-used seed, shortened from 30 to 10 days",
            "pilot_recording_checks": pilot["recording_checks"],
            "arms": {r["arm"]: {"requests": r["request_summary"],
                "elapsed_s": r["elapsed_s"], "recording_checks": r["recording_checks"]} for r in results},
            "common_gates": {"performance": {"status": "INCONCLUSIVE", "reason": "no independent confirmation"},
                "reliability": {"status": "INCONCLUSIVE", "reason": "recording checks passed; manuscript alignment remains separate"},
                "scenario_validity": {"status": "FAIL" if any(not r["request_summary"]["all_requests_admitted"] for r in results)
                                      else "INCONCLUSIVE", "reason": "full demand and vessel-flow validation is still required"}}})
        write_json(out / "progress.json", {"at": now(), "state": "completed", "claim_eligible": False})
    except BaseException:
        write_json(out / "failure.json", {"at": now(), "traceback": traceback.format_exc()})
        write_json(out / "progress.json", {"at": now(), "state": "failed", "claim_eligible": False})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
