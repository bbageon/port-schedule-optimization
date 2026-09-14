"""Recompute existing v3 evidence for the review strategy; runs no simulation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
ARMS = {"NO_REALLOC", "RL", "RL_TIME", "RL_SPACE", "RL_EARLY", "RL_NOVETO"}
GROUPS = [
    "judge-30d", "judge-consent", "env-quiet", "env-mixed", "env-heavy",
    "window-sweep/w1800", "window-sweep/w3600", "window-sweep/w7200",
]


def main() -> None:
    groups = {}
    sources = []
    for group in GROUPS:
        directory = ROOT / "outputs/v3" / group
        files = sorted(directory.glob("arm_*.json"))
        files += sorted((directory / "arms").glob("arm_*.json"))
        arms = {}
        for path in files:
            arm = path.stem.removeprefix("arm_")
            if arm not in ARMS:
                continue
            assert arm not in arms, (group, arm, "duplicate artifact")
            raw = path.read_bytes()
            values = json.loads(raw)["phi_by_day"]
            assert set(map(int, values)) == set(range(30)), path
            total = sum(float(values[str(day)]) for day in range(1, 29))
            arms[arm] = {"measured_days": 28, "total_krw": total}
            sources.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        assert {"NO_REALLOC", "RL"} <= set(arms), group
        baseline = arms["NO_REALLOC"]["total_krw"]
        for values in arms.values():
            values["saving_pct"] = 100 * (baseline - values["total_krw"]) / baseline
        groups[group] = arms
    main_arms = groups["judge-30d"]
    extra = main_arms["RL"]["saving_pct"] - main_arms["RL_TIME"]["saving_pct"]
    consent = groups["judge-consent"]
    consent_gap = consent["RL_NOVETO"]["total_krw"] - consent["RL"]["total_krw"]
    payload = {
        "kind": "read_only_recalculation_of_existing_results",
        "new_simulations": 0,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "measurement": "day indices 1..28; exclude indices 0 and 29",
        "independent_inference_performed": False,
        "sources": sources,
        "groups": groups,
        "full_minus_time_saving_percentage_points": extra,
        "noveto_minus_full_cost_krw": consent_gap,
    }
    (REPORT / "snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recomputed {len(sources)} existing artifacts in {len(groups)} groups; no simulations.")
    print(f"Full saving={main_arms['RL']['saving_pct']:.6f}%")
    print(f"Time saving={main_arms['RL_TIME']['saving_pct']:.6f}%")
    print(f"Space saving={main_arms['RL_SPACE']['saving_pct']:.6f}%")
    print(f"Full-time={extra:.6f} percentage points; NOVETO-full={consent_gap / 1e8:.6f} eok KRW")


if __name__ == "__main__":
    main()
