"""Read-only code/data audit supporting the revised review strategy."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def main() -> None:
    source_paths = [
        "src/yard_rl/v3/train/labels.py", "src/yard_rl/v3/train/month_loop.py",
        "src/yard_rl/v3/train/loop.py", "src/yard_rl/v3/actors/seller.py",
        "src/yard_rl/v3/actors/buyer.py", "src/yard_rl/v3/actors/resolver.py",
        "src/yard_rl/v3/stage/branchpool.py", "src/yard_rl/v3/stage/rollout.py",
        "src/yard_rl/v3/stage/month_run.py", "src/yard_rl/v3/stage/orders.py",
        "src/yard_rl/v3/stage/month.py", "src/yard_rl/v3/eval/month_judge.py",
        "src/yard_rl/v3/eval/__main__.py", "src/yard_rl/v3/reward/krw.py",
        "outputs/v3/month-02/history.json",
    ]
    modules = {}
    for name in source_paths:
        if name.endswith(".py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))
            modules[name] = [{"name": node.name, "line": node.lineno}
                             for node in ast.walk(tree)
                             if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    data = {}
    for group in ["judge-consent", "env-quiet", "env-mixed", "env-heavy",
                  "veto-quiet", "veto-mixed", "veto-heavy"]:
        arms = {}
        for path in sorted((ROOT / "outputs/v3" / group / "arms").glob("arm_*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            days = [d for d in raw.get("days", []) if 1 <= d["index"] <= 28]
            if not days:
                continue
            source_paths.append(path.relative_to(ROOT).as_posix())
            assert len(days) == 28
            arms[raw["arm"]] = {
                "measured_days": len(days),
                "truck_skipped": sum(d["truck_skipped"] for d in days),
                "n_censored": sum(d["n_censored"] for d in days),
                "n_trucks": sum(d["n_trucks"] for d in days),
                "cost_terms_krw": {k: sum(d[k] for d in days) for k in
                                   ["c_wait", "c_move", "c_rehandle", "c_vessel"]},
            }
        data[group] = arms
    history = json.loads((ROOT / "outputs/v3/month-02/history.json").read_text(encoding="utf-8"))
    first = history[0]
    findings = {
        "early_checkpoint": {
            "file": "outputs/v3/month-02/ckpt_000.pt",
            "interpretation": "saved after the first day's fit, not an untrained initialization",
            "history_first_it": first["it"], "first_day_labels": first["n_labels"],
            "first_day_seller_loss": first["seller_loss"],
            "first_day_buyer_loss": first["buyer_loss"],
            "training_hours_from_history": sum(row["secs"] for row in history) / 3600,
        },
        "ranking_scope": {
            "seller_counterfactual": "KEEP versus one SELL coordinate",
            "sell_target": "(cost(SELL coordinate) - cost(KEEP)) / (2 * ADV_SCALE)",
            "keep_target": "depends on the paired SELL coordinate",
            "noveto": "always BUY but still runs BuyerNet and returns its BUY score",
            "resolver": "sorts accepted offers by predicted_phi then deterministic offer key",
            "arbitrary_coordinate_override": "not provided by current KEEP/SELL forcing API",
        },
        "measurement_gaps": [
            "month judge does not save MonthResult admitted/skipped totals or skip reasons",
            "old arm files can lack all per-day detailed fields",
            "cached arm files are selected by arm label without a run-contract hash check",
            "month base seed + 1000*(day+1) can overlap if month bases differ by 1000",
        ],
    }
    sources = [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
               for p in source_paths]
    result = {
        "kind": "read_only_refinement_audit", "new_simulations": 0,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "findings": findings, "observed_month_data": data,
        "code_locations": modules, "sources": sources,
        "limitations": "Skip reasons and their cost impact cannot be recovered from these aggregate files.",
    }
    (REPORT / "refinement-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Read-only audit complete; new simulations: 0.")
    print("First checkpoint: after fitting", first["n_labels"], "labels.")
    print("Consent skipped:", {k: v["truck_skipped"] for k, v in data["judge-consent"].items()})
    print("Source files:", len(sources))


if __name__ == "__main__":
    main()
