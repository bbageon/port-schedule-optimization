"""Validate the new strategy documents and their local evidence links."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def main() -> None:
    specs = sorted((ROOT / ".claude/docs/dashboard-task-specs").glob("YR-317*.md"))
    new_docs = sorted((ROOT / ".claude/docs/strategy-history").glob("2026-09-14-v3-review-*.md"))
    new_docs += specs + [REPORT / "README.md"]
    files = new_docs + [ROOT / p for p in [
        ".claude/Dashboard/README.md", ".claude/Dashboard/backlog.md",
        ".claude/docs/strategy-history/README.md",
    ]]
    failures = []
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if len(text.splitlines()) > 200:
            failures.append(f"Line limit: {path.name}")
        if path in new_docs:
            for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
                if target.startswith(("https://", "http://")):
                    continue
                if not (path.parent / target.split("#")[0]).exists():
                    failures.append(f"Broken link: {path.name}: {target}")
    boards = {p.name: p.read_text(encoding="utf-8-sig").splitlines()
              for p in (ROOT / ".claude/Dashboard").glob("*.md") if p.name != "README.md"}
    for spec in specs:
        task = re.match(r"YR-317(?:-[a-g](?=-))?", spec.name).group(0)
        matches = [(state, line) for state, lines in boards.items() for line in lines
                   if re.match(r"\| " + re.escape(task) + r" \|", line)]
        if len(matches) != 1 or matches[0][0] != "backlog.md":
            failures.append(f"Board state or uniqueness: {task}")
        elif spec.name not in matches[0][1]:
            failures.append(f"Spec link: {task}")
        if "**상태**: backlog" not in spec.read_text(encoding="utf-8"):
            failures.append(f"Spec state: {task}")
    sources = {}
    for name in ["snapshot.json", "refinement-audit.json"]:
        snapshot = json.loads((REPORT / name).read_text(encoding="utf-8"))
        for source in snapshot["sources"]:
            path, expected = source["path"], source["sha256"]
            if path in sources and sources[path] != expected:
                failures.append(f"Conflicting source versions: {path}")
            sources[path] = expected
    for path, expected in sources.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"Source hash: {path}")
    result = {
        "scope": "YR-317 strategy documents and new rows only",
        "markdown_files_checked": len(files), "row_specs_checked": len(specs),
        "source_hashes_checked": len(sources), "failures": failures,
        "new_simulations": 0,
        "global_dashboard_audit": "not claimed; existing ID/status/length issues documented",
    }
    (REPORT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
