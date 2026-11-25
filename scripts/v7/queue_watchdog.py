"""Hourly watchdog for the registered YR-332 queue: bt (planned-time comparison), then br (one-hour discount).

Why: on 2026-11-25 21:34:32 an Orca pane in the working folder was closed (daemon.log: operator_close,
session-killed) and the bt runner, started inside that pane, died with it. A process created through WMI
(Win32_Process.Create) is not part of any pane, so it survives pane or app closure.

Each run (Windows Task Scheduler, hourly, windowless pythonw) checks every job:
- a process of the job is alive            -> nothing
- the job reported 'complete'              -> nothing
- otherwise                                -> relaunch the job's waiter from its frozen worktree through WMI;
                                              the waiters resume (finished arms, waves and updates are skipped)
At most MAX_RESTARTS per job within WINDOW_H hours; after that it only records the state (a repeated failure
needs a person). When every job is complete it deletes its own scheduled task. Standard library only.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(r"C:\Users\geonu\orca\workspaces\port_reinforcement")
HOME = BASE / "yr332-watchdog"          # state and log, outside every worktree
TASK = "yard-rl-queue-watchdog"
MAX_RESTARTS, WINDOW_H = 3, 6
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def bt_process(cmd):
    return "one_hour_train.py" not in cmd and ("after_bp_planned_times" in cmd or "planned-times" in cmd)


def br_process(cmd):
    return "one_hour_train.py" in cmd


JOBS = (
    dict(name="bt", folder=BASE / "yr332-bt-gpu-9afa8b70", python=BASE / "yr332-cuda" / "Scripts" / "python.exe",
         args="-u scripts/v7/after_bp_planned_times.py wait --train-device cuda --inference-device cpu",
         log=r"build\yr332_joint_turnaround\planned-times\queue-gpu.log",
         status="outputs/reports/yr332_planned_times/queue-status.json", owns=bt_process),
    dict(name="br", folder=BASE / "yr332-br-3c003a80", python=BASE / "yr331-env" / "Scripts" / "python.exe",
         args=r"-X utf8 -u scripts\v7\one_hour_train.py wait-bt",
         log=r"build\yr332_joint_turnaround\one-hour\wait-bt.log",
         status="outputs/reports/yr332_one_hour/queue-status.json", owns=br_process),
)


def decide(alive, stage, recent_restarts):
    """Pure decision for one job: 'alive', 'complete', 'restart' or 'gave_up'."""
    if alive:
        return "alive"
    if stage == "complete":
        return "complete"
    return "restart" if recent_restarts < MAX_RESTARTS else "gave_up"


def powershell(script, **env):
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                           "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + script],
                          capture_output=True, text=True, encoding="utf8", errors="replace",
                          env={**os.environ, **env}, creationflags=NO_WINDOW)


def python_command_lines():
    out = powershell("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
                     "Select-Object -ExpandProperty CommandLine")
    if out.returncode != 0:
        raise RuntimeError(f"process list failed: {out.stderr.strip()[:300]}")
    return [line for line in out.stdout.splitlines() if line.strip()]


def launch(job):
    command = (f'cmd.exe /c "set PYTHONPATH=src&& "{job["python"]}" {job["args"]} '
               f'>> {job["log"]} 2>&1"')
    out = powershell("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
                     "@{CommandLine = $env:WD_CMD; CurrentDirectory = $env:WD_DIR}; "
                     "\"$($r.ReturnValue) $($r.ProcessId)\"", WD_CMD=command, WD_DIR=str(job["folder"]))
    code, _, pid = out.stdout.strip().partition(" ")
    if out.returncode != 0 or code != "0":
        raise RuntimeError(f"launch failed ({out.stdout.strip()} {out.stderr.strip()[:300]})")
    return pid


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return default


def main():
    HOME.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    log = (HOME / "watchdog.log").open("a", encoding="utf8")
    state = read_json(HOME / "state.json", {"restarts": {}})
    try:
        commands = python_command_lines()
        done = 0
        for job in JOBS:
            status = read_json(job["folder"] / job["status"], {})
            stage = status.get("stage")
            alive = any(job["owns"](c) for c in commands)
            since = now - timedelta(hours=WINDOW_H)
            recent = [t for t in state["restarts"].get(job["name"], []) if datetime.fromisoformat(t) > since]
            action = decide(alive, stage, len(recent))
            detail = f"stage={stage} arm={status.get('arm', status.get('bt_arm'))}"
            if action == "complete":
                done += 1
            elif action == "restart":
                if not job["folder"].is_dir():
                    action, detail = "missing", f"{detail} folder {job['folder']} not found"
                else:
                    with (job["folder"] / job["log"]).open("a", encoding="utf8") as f:
                        f.write(f"{now:%m-%d %H:%M:%S} watchdog restart (no live process, {detail})\n")
                    pid = launch(job)
                    state["restarts"].setdefault(job["name"], []).append(now.isoformat(timespec="seconds"))
                    detail += f" -> relaunched pid {pid}"
            log.write(f"{stamp} {job['name']} {action} {detail}\n")
        (HOME / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf8")
        if done == len(JOBS):
            subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"], capture_output=True, creationflags=NO_WINDOW)
            log.write(f"{stamp} all jobs complete; scheduled task {TASK} deleted\n")
    except Exception as exc:  # pythonw has no console: the log is the only trace
        log.write(f"{stamp} watchdog error {exc!r}\n")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        assert decide(True, None, 9) == "alive"
        assert decide(False, "complete", 0) == "complete"
        assert decide(False, "training", 0) == "restart"
        assert decide(False, "failed", 2) == "restart"
        assert decide(False, "waiting_for_bt", 3) == "gave_up"
        assert bt_process("python -u scripts/v7/after_bp_planned_times.py wait --train-device cuda")
        assert bt_process("python -X utf8 scripts/v7/short_window_train.py rollout --tag wave-0-0 --planned-times etw")
        assert not bt_process("python -X utf8 -u scripts\\v7\\one_hour_train.py wait-bt")
        assert br_process("python -X utf8 scripts/v7/one_hour_train.py rollout --tag wave-0-1")
        assert not br_process("python -u scripts/v7/after_bp_planned_times.py wait")
        print("self-test passed")
    else:
        main()
