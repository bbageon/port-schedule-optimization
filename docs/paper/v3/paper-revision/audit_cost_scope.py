"""Check existing v3 cost wiring and saved requests; never change a policy or cost."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from yard_rl.v3.reward.phi import terminal_cost_krw
from yard_rl.v3.reward.krw import truck_wait_krw
from yard_rl.v3.schema.record import ExecutionRecord
from yard_rl.v3.world.integrated.time_sell import deferral_ledger


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = 14 * 3600.0
    examples = []
    for gate in (original, original + 7200):
        rec = ExecutionRecord(doc_key="example", copino_notice_s=13 * 3600,
            gate_in_s=gate, block_in_s=gate + 60, service_start_s=gate + 120,
            job_done_s=gate + 540, gate_out_s=gate + 600)
        cost = terminal_cost_krw({rec.doc_key: rec}, end_s=17 * 3600)
        examples.append({"original_appointment_s": original, "gate_in_s": gate,
            "gate_out_s": gate + 600, "cost": cost.as_dict()})
    engine_rec = SimpleNamespace(entry_deferrals=1, owner="B", flow="GATE_OUT",
        entry_deferred_s=7200, a_gate_in=original + 7200)
    job = SimpleNamespace(appointment_gate_time=original)
    mbt = SimpleNamespace(ledger=SimpleNamespace(records={"example": engine_rec}),
        blocks={"B": SimpleNamespace(jobs={"example": job})})
    recorded_deferral = deferral_ledger(mbt)
    pre_gate = terminal_cost_krw({"example": ExecutionRecord(doc_key="example")},
        end_s=15 * 3600).total
    checks = {
        "equal_dwell_equal_cost_despite_two_hour_schedule_shift":
            examples[0]["cost"]["phi_krw"] == examples[1]["cost"]["phi_krw"],
        "engine_audit_can_record_two_hour_external_interval":
            recorded_deferral[0]["driver_outside_wait_s"] == 7200,
        "pre_gate_record_has_zero_current_cost": pre_gate == 0,
    }
    saved = {}
    inputs = {}
    base = ROOT / "outputs/reports/yr317_v3_request_audit/run-3279b7f"
    for arm in ("NO_REALLOC", "RL", "RL_NOVETO"):
        result_path, ledger_path = base / arm / "result.json", base / arm / "requests.jsonl.gz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        costs, shifted, shifted_seconds, count, examples_saved = [], 0, 0.0, 0, []
        all_match = True
        with gzip.open(ledger_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                gate, out = row["gate_in_s"], row["gate_out_s"]
                end = result["request_summary"]["end_s"]
                dwell = None if gate is None else max(0.0, min(out, end) - gate) if out is not None else max(0.0, end - gate)
                cost = truck_wait_krw(dwell)
                all_match &= math.isclose(cost, row["accounted_wait_krw"], rel_tol=1e-12, abs_tol=1e-6)
                costs.append(cost)
                if gate is not None and gate > row["requested_arrival_s"] + 1e-6:
                    shifted += 1
                    shifted_seconds += gate - row["requested_arrival_s"]
                    if len(examples_saved) < 2:
                        examples_saved.append({k: row[k] for k in ("job_id", "requested_arrival_s",
                            "gate_in_s", "gate_out_s", "accounted_turn_time_s", "accounted_wait_krw")})
        wait = math.fsum(costs)
        checks[f"{arm}_all_requests_reconciled"] = count == result["request_summary"]["requested"] and all_match
        checks[f"{arm}_wait_sum_matches_saved_cost"] = math.isclose(wait,
            result["request_summary"]["accounted_wait_krw"], rel_tol=1e-12, abs_tol=1e-3)
        saved[arm] = {"requests": count, "later_gate_count": shifted,
            "later_gate_minus_request_hours": shifted_seconds / 3600,
            "wait_recomputed_krw": wait, "saved_wait_krw": result["request_summary"]["accounted_wait_krw"],
            "examples": examples_saved}
        for path in (result_path, ledger_path):
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    paths = ["src/yard_rl/v3/reward/phi.py", "src/yard_rl/v3/reward/krw.py",
        "src/yard_rl/v3/schema/lifecycle.py", "src/yard_rl/v3/stage/month_run.py",
        "src/yard_rl/v3/stage/rollout.py", "src/yard_rl/v3/stage/bridge.py",
        "src/yard_rl/v3/world/integrated/time_sell.py",
        "src/yard_rl/v3/world/integrated/multiblock.py",
        "src/yard_rl/experiments/yr151_transfer_ppo.py",
        "src/yard_rl/experiments/yr164_cost_decomposition.py"]
    for name in paths:
        inputs[name] = sha(ROOT / name)
    refs = ("9715cc6", "3279b7f", "1b601c0")
    source_comparison = {}
    for ref in refs:
        comparison = {}
        for name in paths[:3]:
            data = subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT)
            # Git stores LF; a Windows working tree may use CRLF.
            comparison[name] = data.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        source_comparison[ref] = comparison
        checks[f"core_cost_source_matches_{ref}"] = all(comparison.values())
    payload = {"schema": "v3.cost_scope_audit.v1", "date": "2026-09-15",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checks": checks, "passed": all(checks.values()), "illustrative_cost_calls": examples,
        "illustrative_engine_deferral_record": recorded_deferral,
        "saved_diagnostic_runs": saved, "core_source_comparison": source_comparison,
        "input_sha256": inputs,
        "conclusion": "Historical phi_terminal adds the external interval; v3 terminal_cost_krw uses actual gate-in dwell and omits that addition in both evaluation and counterfactual labels.",
        "limits": ["The example calls cost and audit functions; it is not a full terminal simulation.",
            "Gate minus requested arrival is an interval, not proof of actual driver idle time.",
            "Source at manuscript preparation is not proof of the exact original experiment executable.",
            "No external-cost rate, break-even calculation, policy change or new training was introduced."],
        "new_simulations": 0, "new_training_runs": 0, "performance_confirmed": False}
    (OUT / "cost-scope-audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "checks": checks, "saved": saved}, ensure_ascii=True))
    return int(not payload["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
