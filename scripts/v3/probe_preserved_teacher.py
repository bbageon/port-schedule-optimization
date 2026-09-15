"""Check counterfactual wiring in the repaired environment, without updating weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.source, text=True).strip():
        raise RuntimeError("Source must be frozen and clean")
    os.sched_setaffinity(0, {3})
    sys.path.insert(0, str(args.source / "src"))
    import torch
    torch.set_num_threads(1)
    from yard_rl.v3.eval.__main__ import _load_nets
    from yard_rl.v3.stage.month import plan_days
    from yard_rl.v3.stage.month_run import run_month
    seller, buyer, _ = _load_nets(args.checkpoint)
    def weights_sha():
        return hashlib.sha256(b"".join(v.detach().numpy().tobytes()
            for net in (seller, buyer) for v in net.state_dict().values())).hexdigest()
    before, rows, start = weights_sha(), [], time.monotonic()
    def collect(day, labels):
        rows.extend(labels)
        return {}
    res = run_month(seed=9_900_700, days=plan_days(9_900_700, (300,)), arm="RL",
        seller_net=seller, buyer_net=buyer, admission_mode="PRESERVE", labels_per_day=2,
        workers=1, explore=.37, horizon_s=10800, on_fit=collect,
        capture_requests=True, diagnose_admissions=True)
    checks = {"labels_generated": len(rows) > 0,
        "factual_decisions_match": sum(d.identity_ok for d in res.days) > 0
            and sum(d.identity_bad for d in res.days) == 0,
        "weights_unchanged": before == weights_sha(),
        "all_trucks_completed": res.request_summary["states"] == {"COMPLETED": 300},
        "no_policy_exceptions": res.policy_exceptions == 0,
        "records_reconcile": res.request_summary["recording_ok"] and res.vessel_work_summary["recording_ok"]}
    payload = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=args.source, text=True).strip(), "seed": 9_900_700, "load": 300,
        "labels_per_day": 2, "horizon_s": 10800, "explore": .37, "workers": 1, "cpu": 3,
        "checks": checks, "passed": all(checks.values()), "labels": rows,
        "identity_ok": sum(d.identity_ok for d in res.days), "identity_bad": sum(d.identity_bad for d in res.days),
        "worlds": sum(d.worlds for d in res.days), "weights_sha256": before,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "elapsed_s": time.monotonic() - start, "new_training_runs": 0,
        "purpose": "wiring-only counterfactual probe, without optimizer or performance inference"}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("passed", "checks", "worlds", "identity_ok", "identity_bad", "elapsed_s")}))
    return int(not payload["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
