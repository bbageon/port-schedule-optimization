"""A fixed 300-request on/off check of the read-only admission observer (YR-317-g)."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior(result):
    """Exclude only the new observational fields, keeping all existing outcomes."""
    rows = []
    for row in result.request_ledger:
        item = dict(row)
        item["admission_events"] = [{k: v for k, v in event.items()
            if k != "inventory_at_failure"} for event in row["admission_events"]]
        rows.append(item)
    return {"days": [asdict(d) for d in result.days], "requests": rows,
        "request_summary": result.request_summary,
        "vessels": [{k: v for k, v in r.items() if k != "inventory_before_admission"}
                    for r in result.vessel_admissions],
        **{k: getattr(result, k) for k in ("admitted", "skipped", "traded_edges", "n_space",
            "n_time", "txn_failed", "decisions", "policy_exceptions", "retargeted")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from yard_rl.integrated.repro import code_dirty, repro_stamp
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.eval.contracts import arm_contract, runtime_identity
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls
    from yard_rl.v3.stage.month import plan_days
    from yard_rl.v3.stage.month_run import run_month

    torch.set_num_threads(1)
    if code_dirty() is not False:
        raise RuntimeError("Use a clean, committed source checkout.")
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha(args.checkpoint)
    seed = 9_900_700
    days = plan_days(seed, (300,))
    results, modes = {}, {}

    def save(name, data):
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                allow_nan=False) + "\n", encoding="utf-8")

    for mode in ("off", "on"):
        if sha(args.checkpoint) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during the probe.")
        seller, buyer, _ = _load_nets(args.checkpoint)
        kwargs = dict(seed=seed, arm="RL", days=days, seller_net=seller, buyer_net=buyer,
                      capture_requests=True, diagnose_admissions=mode == "on")
        contract = arm_contract(kwargs, runtime_identity(), run_month)
        stamp = repro_stamp(experiment="YR-317-g-admission-observer", seeds={"base": [seed],
            "days": [d.seed for d in days]}, params={"run": contract["settings"]},
            prereg=str(args.prereg), extra={"checkpoint_sha256": checkpoint_hash,
                "prereg_sha256": sha(args.prereg), "runner_sha256": sha(__file__)})
        start = time.monotonic()
        save("progress.json", {"state": "running", "mode": mode,
            "at": datetime.now(timezone.utc).isoformat()})
        reset_rollout_calls()
        result = run_month(**kwargs)
        results[mode] = behavior(result)
        modes[mode] = {"elapsed_s": time.monotonic() - start, "repro": stamp,
            "request_summary": result.request_summary, "teacher_calls": rollout_calls(),
            "vessel_work_ledger": result.vessel_work_ledger,
            "vessel_work_summary": result.vessel_work_summary,
            "policy_exceptions": result.policy_exceptions,
            "behavior_sha256": hashlib.sha256(json.dumps(results[mode], sort_keys=True,
                ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()}
        save(f"{mode}.json", {"behavior": results[mode], **modes[mode]})
        print(json.dumps({"mode": mode, "elapsed_s": modes[mode]["elapsed_s"],
            "requests": result.request_summary["states"],
            "vessel_work": result.vessel_work_summary}, ensure_ascii=False), flush=True)

    checks = {"behavior_unchanged": results["off"] == results["on"],
        "request_recording": all(m["request_summary"]["recording_ok"] and
            m["request_summary"]["announcer_counts_match"] for m in modes.values()),
        "vessel_recording": modes["on"]["vessel_work_summary"]["recording_ok"],
        "no_teacher": all(m["teacher_calls"] == 0 for m in modes.values()),
        "no_policy_exception": all(m["policy_exceptions"] == 0 for m in modes.values())}
    save("comparison.json", {"schema": "yr317.admission-observer-probe.v1", "checks": checks,
        "passed": all(checks.values()), "modes": modes, "new_training_runs": 0,
        "independent_confirmation": False, "performance_claim_eligible": False,
        "scope": "same fixed small workload, comparing diagnostics off/on; not the 10-day failure-state replay"})
    save("progress.json", {"state": "completed", "passed": all(checks.values())})
    return int(not all(checks.values()))


if __name__ == "__main__":
    raise SystemExit(main())
