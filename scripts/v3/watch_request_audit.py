"""Validate finished diagnostic arms automatically; never run or modify a policy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import analyze_request_audit as report


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    repo, run, out = (p.resolve() for p in (args.repo, args.run, args.out))
    report.ROOT = repo
    if args.launch:
        out.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--repo", str(repo), "--run", str(run), "--out", str(out)]
        with (out / "watch.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {"at": now(), "pid": process.pid, "command": command,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], text=True).strip(),
            "watcher_sha256": report.sha(__file__), "analyzer_sha256": report.sha(report.__file__)}
        save(out / "launch.json", launch)
        print(json.dumps(launch, ensure_ascii=False))
        return 0
    if hasattr(os, "nice"):
        os.nice(10)
    previous = None
    try:
        while True:
            finished = tuple(label for label in ("pilot_RL", *report.ARMS)
                             if (run / label / "result.json").exists())
            failed = (run / "failure.json").exists()
            if finished != previous or failed:
                audit = report.analyze(run)
                path = out / f"reconciliation-{len(finished)}{'-failure' if failed else ''}.json"
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                status = {"at": now(), "state": "failed" if audit["problems"] else
                    "completed" if audit["all_main_arms_complete"] else "waiting",
                    "completed": list(finished), "snapshot": path.name,
                    "problems": audit["problems"], "claim_eligible": False}
                save(out / "status.json", status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
                if audit["problems"] or audit["all_main_arms_complete"]:
                    return int(bool(audit["problems"]))
                previous = finished
            # A lost runner must not be mislabeled as an active simulation forever.
            runner = report.read_json(run / "launch.json")["pid"]
            try:
                os.kill(runner, 0)
            except ProcessLookupError:
                raise RuntimeError("Runner exited without all three completed results; do not restart automatically.")
            time.sleep(30)
    except BaseException:
        save(out / "watch_failure.json", {"at": now(), "traceback": traceback.format_exc()})
        save(out / "status.json", {"at": now(), "state": "failed", "claim_eligible": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
