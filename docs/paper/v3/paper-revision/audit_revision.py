"""Recompute the copied paper's descriptive evidence and verify the unchanged sources.

No simulation, training, significance test, or paper rewriting occurs here.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = ROOT / "docs/paper/v3/submissionv2"
sys.path.insert(0, str(ROOT / "scripts/v3"))
from figdata import month


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    files = subprocess.check_output(["git", "ls-files", "--", "docs/paper/v3/submissionv2"],
                                   cwd=ROOT, text=True).splitlines()
    copied = []
    for name in files:
        original = ROOT / name
        relative = original.relative_to(SOURCE)
        target = OUT / ("source-README.md" if str(relative) == "README.md" else relative)
        copied.append({"source": name, "copy": str(target.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha(original), "copy_sha256": sha(target),
            "identical": original.read_bytes() == target.read_bytes()})
    evidence = {}
    inputs = {}
    for folder in ("judge-30d", "env-quiet", "env-mixed", "env-heavy"):
        arms = {}
        for arm in ("NO_REALLOC", "RL", "RL_TIME", "RL_SPACE"):
            path = ROOT / "outputs/v3" / folder / "arms" / f"arm_{arm}.json"
            arms[arm] = read(path)
            inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
        recorded_days = arms["NO_REALLOC"].get("days")
        if recorded_days is None:
            recorded_days = [d.as_dict() for d in month().plan_month(9_900_950)]
        days = [d for d in recorded_days if 1 <= d["index"] <= 28]
        totals = {a: sum(float(data["phi_by_day"][str(d["index"])]) for d in days)
                  for a, data in arms.items()}
        savings = {a: totals["NO_REALLOC"] - totals[a] for a in ("RL", "RL_TIME", "RL_SPACE")}
        by_load = []
        for load in sorted({d["load"] for d in days}):
            group = [d["index"] for d in days if d["load"] == load]
            reduced = {a: sum(arms["NO_REALLOC"]["phi_by_day"][str(i)] -
                arms[a]["phi_by_day"][str(i)] for i in group) for a in savings}
            by_load.append({"load": load, "days": len(group),
                "reduction_million_krw": {a: v / 1e6 for a, v in reduced.items()},
                "mean_daily_reduction_million_krw": {a: v / len(group) / 1e6 for a, v in reduced.items()},
                "space_increment_over_time_million_krw": (reduced["RL"] - reduced["RL_TIME"]) / 1e6})
        rl_days = [d for d in arms["RL"].get("days", []) if 1 <= d["index"] <= 28]
        space, timing = arms["RL"]["n_space"], arms["RL"]["n_time"]
        measured_actions = sum(d["n_space"] + d["n_time"] for d in rl_days)
        evidence[folder] = {"compared_days": len(days), "congested_days": sum(d["load"] >= 12500 for d in days),
            "cost_totals_krw": totals, "reduction_krw": savings,
            "reduction_percent": {a: 100 * v / totals["NO_REALLOC"] for a, v in savings.items()},
            "time_only_share_percent": 100 * savings["RL_TIME"] / savings["RL"],
            "spatial_action_percent": 100 * space / (space + timing),
            "action_share_scope": "30-day run totals, as used by the original manuscript checker",
            "middle28_spatial_percent": 100 * sum(d["n_space"] for d in rl_days) / measured_actions
                if measured_actions else None, "by_load": by_load}
    checked_md, broken_links = {}, []
    for path in sorted(OUT.rglob("*.md")):
        body = path.read_text(encoding="utf-8")
        checked_md[str(path.relative_to(OUT))] = len(body.splitlines())
        for target in re.findall(r"\]\(([^)]+)\)", body):
            if target.startswith(("https://", "http://", "#")):
                continue
            local = unquote(target.split("#", 1)[0].strip("<>"))
            if local and not (path.parent / local).exists():
                broken_links.append({"file": str(path.relative_to(OUT)), "target": target})
    payload = {"schema": "yr317.paper-revision-source-audit.v1",
        "source_selection": "latest prepared submission version in repository history; portal-upload identity unverified",
        "source_commit": subprocess.check_output(["git", "log", "-1", "--format=%H", "--",
            "docs/paper/v3/submissionv2"], cwd=ROOT, text=True).strip(),
        "copied_files": copied, "input_sha256": inputs, "descriptive_evidence": evidence,
        "markdown_lines": checked_md, "broken_local_links": broken_links,
        "passed": all(f["identical"] for f in copied) and not broken_links and
                  all(n <= 200 for n in checked_md.values()),
        "new_simulations": 0, "new_training_runs": 0, "performance_confirmed": False}
    (OUT / "source-audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "identical_files": sum(f["identical"] for f in copied),
        "markdown_files": len(checked_md), "broken_links": broken_links,
        "by_load": evidence["judge-30d"]["by_load"],
        "spatial_share": {k: v["spatial_action_percent"] for k, v in evidence.items()}}, ensure_ascii=False))
    return int(not payload["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
