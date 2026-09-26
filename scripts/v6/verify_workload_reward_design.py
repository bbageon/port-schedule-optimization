"""YR-331: executable design arithmetic, NOT the v6 runtime or a port experiment.

Run with standard Python. Reproduces the literature-derived illustration and
checks conservation/discount boundaries on a fully enumerated small task system.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def targets(capacity, fixed, movable, safe, available=None):
    """Water level for ONE mutually compatible task group; not a dispatcher."""
    available = [1.0] * len(capacity) if available is None else available
    rows = list(zip(capacity, fixed, movable, safe, available, strict=True))
    if not rows:
        raise ValueError("empty group")
    if any(not all(math.isfinite(v) for v in row) for row in rows):
        raise ValueError("non-finite input")
    if any(a <= 0 or f < 0 or m < 0 or not 0 < s < 1 or not 0 <= u <= 1
           for a, f, m, s, u in rows):
        raise ValueError("invalid workload/capacity")
    high = [max(f, a * s * u) for a, f, _, s, u in rows]
    assigned = sum(fixed) + min(sum(movable), sum(h - f for h, f in zip(high, fixed)))
    low_level, high_level = 0.0, max(h / a for h, a in zip(high, capacity))
    for _ in range(90):
        level = (low_level + high_level) / 2
        trial = [min(h, max(f, a * level)) for a, f, h in zip(capacity, fixed, high)]
        if sum(trial) < assigned:
            low_level = level
        else:
            high_level = level
    return [min(h, max(f, a * high_level)) for a, f, h in zip(capacity, fixed, high)]


def potential(capacity, fixed, movable, safe, available=None):
    goal = targets(capacity, fixed, movable, safe, available)
    return -sum((f + m - y) ** 2 / a
                for a, f, m, y in zip(capacity, fixed, movable, goal)) / sum(capacity)


def queue_limit(wait_min, service_min, service_cv):
    """M/G/1: stationary Poisson input, iid service, one server, no interruptions."""
    if wait_min <= 0 or service_min <= 0 or service_cv < 0:
        raise ValueError("invalid queue inputs")
    variability = (1 + service_cv ** 2) / 2
    rho = wait_min / (wait_min + variability * service_min)
    return rho, rho * 60 / service_min


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    checks = []

    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    close = lambda a, b: math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    service = 2 / 0.633
    examples = []
    for wait in (2, 5, 10):
        rho, rate = queue_limit(wait, service, 0.42687)
        reconstructed = rho / (1 - rho) * (1 + 0.42687 ** 2) / 2 * service
        check(f"queue_inverse_{wait}_minutes", close(reconstructed, wait))
        examples.append({"assumed_queue_limit_min": wait, "load_limit": rho,
                         "admissions_per_hour": rate})
    check("capacity_is_not_admission_rate", all(x["admissions_per_hour"] < 18.99 for x in examples))

    cap, fixed, safe = [100, 100], [0, 0], [0.8, 0.8]
    p = lambda m: potential(cap, fixed, m, safe)
    before, after = p([120, 20]), p([100, 40])
    gamma = 0.999
    check("helpful_buy_improves_both_sides_total", close(before, -0.25) and close(after, -0.09))
    check("one_minute_shaping_example", close(gamma * after - before, 0.16009))
    check("balanced_target_is_maximum", close(p([70, 70]), 0))
    check("unnecessary_trade_is_worse", p([90, 50]) < p([70, 70]))
    check("overloaded_pair_can_still_benefit_from_buy", p([120, 110]) > p([140, 90]))
    check("no_demand_has_no_fill_incentive", close(p([0, 0]), 0))
    check("low_balanced_demand_has_no_fill_incentive", close(p([10, 10]), 0))
    check("surplus_is_not_erased", p([140, 140]) < p([80, 80]))
    check("different_capacities_split_proportionally", all(close(x, y) for x, y in zip(
        targets([60, 30], [0, 0], [45, 0], safe), [30, 15])))
    check("fixed_vessel_work_cannot_be_transferred", all(close(x, y) for x, y in zip(
        targets([60, 60], [55, 0], [0, 20], safe), [55, 20])))
    check("known_outage_creates_no_new_capacity", all(close(x, y) for x, y in zip(
        targets([60, 60], [0, 0], [30, 0], safe, [0, 1]), [0, 30])))

    # Exhaust all fixed/movable corner cases in a heterogeneous two-crane group.
    allocation_cases = 0
    for f0, f1, m0, m1 in itertools.product((0, 20, 80), repeat=4):
        f, m, a, s = [f0, f1], [m0, m1], [60, 30], [0.7, 0.8]
        y = targets(a, f, m, s)
        h = [max(fi, ai * si) for fi, ai, si in zip(f, a, s)]
        expected = sum(f) + min(sum(m), sum(hi - fi for hi, fi in zip(h, f)))
        if not (close(sum(y), expected) and all(fi - 1e-9 <= yi <= hi + 1e-9
                                              for fi, yi, hi in zip(f, y, h))):
            raise AssertionError("water_level_conservation")
        allocation_cases += 1
    check("water_level_conservation_81_cases", allocation_cases == 81)

    # Exhaust feasible 5-step paths with real completions, transfers and WAIT.
    # Base objective is intentionally simple, with terminal unfinished-work cost.
    # This tests policy ordering algebra, NOT real-world port performance.
    phi = lambda jobs: potential([3, 3], [0, 0], jobs, safe)
    initial = (3, 1)
    paths = [(initial, 0.0, 0.0, ())]
    for step in range(5):
        expanded = []
        for jobs, raw, shaped, history in paths:
            choices = [("WAIT", jobs, 0.0)]
            for i in (0, 1):
                if jobs[i] > 0:
                    served = list(jobs); served[i] -= 1
                    moved = list(jobs); moved[i] -= 1; moved[1 - i] += 1
                    choices.extend([(f"SERVE_{i}", tuple(served), 0.0),
                                    (f"TRANSFER_{i}", tuple(moved), 0.2)])
            for action, nxt, transfer_cost in choices:
                raw_step = -sum(jobs) - transfer_cost
                if step == 4:
                    raw_step -= 10 * sum(nxt)
                next_phi = 0.0 if step == 4 else phi(nxt)
                shaped_step = raw_step + gamma * next_phi - phi(jobs)
                expanded.append((nxt, raw + gamma ** step * raw_step,
                                 shaped + gamma ** step * shaped_step, history + (action,)))
        paths = expanded
    errors = [abs(shaped - raw + phi(initial)) for _, raw, shaped, _ in paths]
    check("all_complete_path_returns_differ_by_same_constant", max(errors) < 1e-10)
    selected = max(paths, key=lambda row: row[2])
    check("shaping_preserves_best_base_policy_in_toy", close(selected[1], max(row[1] for row in paths)))
    check("return_proof_includes_wait_and_transfer_loops",
          any(h == ("TRANSFER_0", "TRANSFER_1", "WAIT", "WAIT", "WAIT") for *_, h in paths))

    # A truncated rollout needs the shifted continuation value, not terminal zero.
    ps, rs, elapsed = [-0.25, -0.09, -0.16], [-0.2, -0.4], [0.5, 2.0]
    discount, raw, shaped = 1.0, 0.0, 0.0
    for i, dt in enumerate(elapsed):
        g = gamma ** dt
        raw += discount * rs[i]
        shaped += discount * (rs[i] + g * ps[i + 1] - ps[i])
        discount *= g
    continuation = -3.7
    check("variable_time_discount_and_truncation_bootstrap",
          close(shaped + discount * (continuation - ps[-1]), raw + discount * continuation - ps[0]))
    check("dropping_terminal_potential_would_change_objective", not close(shaped - raw, -ps[0]))

    for invalid in [([0, 1], [0, 0], [1, 0], safe),
                    ([1, 1], [0, 0], [-1, 0], safe),
                    ([1, 1], [0, 0], [math.inf, 0], safe)]:
        try:
            targets(*invalid)
        except ValueError:
            continue
        raise AssertionError("invalid inputs accepted")
    check("invalid_capacity_workload_nonfinite_rejected", True)

    excerpt_path = ROOT / "docs/research/v6-workload-reward/historical_run_excerpt.json"
    saved = json.loads(excerpt_path.read_text(encoding="utf-8"))
    saved_path = ROOT / saved["original_path"]
    original_verified = False
    if saved_path.exists():
        original = json.loads(saved_path.read_text(encoding="utf-8"))
        if file_hash(saved_path) != saved["original_sha256"] or any(
                original[k] != saved[k] for k in ("traded_edges", "n_space", "n_time", "code")):
            raise AssertionError("historical excerpt does not match local original")
        original_verified = True
    check("historical_trades_are_space_plus_time", saved["traded_edges"] == saved["n_space"] + saved["n_time"] > 0)
    sources = ["src/yard_rl/v6/ppo/runtime.py", "src/yard_rl/v6/reward/phi.py",
               "src/yard_rl/v6/reward/krw.py", "src/yard_rl/v6/actors/buyer.py"]
    return {
        "schema": "yard_rl.design_arithmetic.v1", "task": "YR-331", "date": "2026-09-26",
        "claim_scope": "DESIGN_ARITHMETIC_ONLY_NO_PORT_PERFORMANCE_CLAIM",
        "port_simulations": 0, "training_runs": 0, "runtime_reward_modified": False,
        "checks_passed": len(checks), "checks": checks,
        "literature_input": {"service_containers_per_2min": 0.633, "service_cv": 0.42687,
                             "service_min": service, "service_rate_per_hour": 60 / service},
        "illustrations_not_calibrated_v6_targets": examples,
        "transfer_example": {"potential_before": before, "potential_after": after,
                             "same_time_delta": after - before,
                             "one_minute_shaping_eta_1": gamma * after - before},
        "allocation_cases": allocation_cases,
        "toy_paths": len(paths), "max_return_identity_error": max(errors),
        "historical_v5_report": {"path": str(saved_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": saved["original_sha256"], "code_commit": saved["code"]["git_head"],
            "excerpt_sha256": file_hash(excerpt_path), "original_verified_this_run": original_verified,
            "traded_edges": saved["traded_edges"], "space": saved["n_space"], "time": saved["n_time"],
            "not_v6_gpu_evidence": True},
        "source_sha256": {s: file_hash(ROOT / s) for s in sources},
        "calculator_sha256": file_hash(Path(__file__)),
        "workspace_head_when_checked": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "note": "Checks run with design edits and unrelated GPU edits present; no clean simulation run claimed.",
        "research_gate": {"allowed": False, "reason": "저장된 scenario PASS 재계산 실패",
                          "gate_commit": "7af37f760ab373eb663a6b7a88320c46698ea3bc",
                          "permission_scope": "user-requested literature/design only; no learning experiment"},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/reports/yr331_workload_reward/design_checks.json")
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("checks_passed", "allocation_cases", "toy_paths",
                                           "max_return_identity_error", "claim_scope")}, indent=2))
