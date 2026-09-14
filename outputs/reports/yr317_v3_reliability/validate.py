"""Validate current YR-317 records while preserving the historical strategy audit."""
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    failures = []
    specs = sorted((ROOT / ".claude/docs/dashboard-task-specs").glob("YR-317*.md"))
    docs = specs + sorted(OUT.glob("*.md")) + [ROOT / ".claude/Dashboard/README.md",
        ROOT / ".claude/Dashboard/backlog.md", ROOT / ".claude/Dashboard/ready.md",
        ROOT / ".claude/Dashboard/in-progress.md",
        ROOT / ".claude/docs/dashboard-task-specs/YR-296-consent-confirm-fresh-band.md"]
    for path in docs:
        content = path.read_text(encoding="utf-8-sig")
        if len(content.splitlines()) > 200:
            failures.append(f"Line limit: {path.relative_to(ROOT)}")
        if path.parent != ROOT / ".claude/Dashboard":
            for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
                if not target.startswith(("http://", "https://")):
                    if not (path.parent / target.split("#")[0]).exists():
                        failures.append(f"Broken link: {path.name}: {target}")
    board = {name: (ROOT / ".claude/Dashboard" / f"{name}.md").read_text(encoding="utf-8")
             for name in ("backlog", "ready", "in-progress", "done", "cancelled")}
    for spec in specs + [docs[-1]]:
        task = re.match(r"YR-\d+(?:-[a-g](?=-))?", spec.name)[0]
        states = [name for name, content in board.items()
                  for _ in re.finditer(rf"^\|\s*{task}\s*\|", content, re.M)]
        state = re.search(r"\*\*상태\*\*:\s*(\S+)", spec.read_text(encoding="utf-8"))[1]
        if states != [state]:
            failures.append(f"State/uniqueness: {task}: {states} vs {state}")
    sources = {}
    changed = []
    # Historical hashes are never replaced by hashes of newly edited source files.
    for name in ("snapshot.json", "refinement-audit.json"):
        payload = json.loads((OUT.parent / "yr317_v3_review_strategy" / name).read_text(encoding="utf-8"))
        commit = payload.get("git_head", payload.get("source_git_head"))
        for src in payload["sources"]:
            path, expected = src["path"], src["sha256"]
            raw = (ROOT / path).read_bytes()
            if hashlib.sha256(raw).hexdigest() == expected:
                sources[path] = "unchanged_current_file"
                continue
            old = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
            # Git stores LF; the audited Windows checkout may have contained CRLF.
            variants = [old, old.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")]
            if not any(hashlib.sha256(v).hexdigest() == expected for v in variants):
                failures.append(f"Historical source hash mismatch: {path}")
            sources[path] = f"historical_commit:{commit}"
            changed.append(path)
    tree = ET.parse(OUT / "tests.xml")
    suites = tree.getroot().findall("testsuite")
    tests = {key: sum(int(s.get(key, 0)) for s in suites)
             for key in ("tests", "failures", "errors", "skipped")}
    if not tests["tests"] or tests["failures"] or tests["errors"]:
        failures.append("Selected regression checks did not pass.")
    result = {"scope": "YR-317-a code and records; not a performance or global Dashboard PASS",
        "tests": tests, "deselected": 1, "markdown_checked": len(docs),
        "row_specs_checked": len(specs) + 1, "historical_source_hashes_checked": len(sources),
        "historical_sources_verified": sources, "current_source_changes": sorted(set(changed)),
        "new_confirmatory_months": 0,
        "full_v3_suite": "interrupted during the first long branch simulation; not claimed passed",
        "remaining": ["submitted PDF identity", "manuscript and figure/script corrections",
                      "current reliability gate re-evaluation", "request conservation audit YR-317-g"],
        "failures": failures}
    (OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: val for key, val in result.items()
                      if key not in {"historical_sources_verified", "current_source_changes"}}, ensure_ascii=False))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
