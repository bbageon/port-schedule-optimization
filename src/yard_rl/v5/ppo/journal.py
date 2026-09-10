"""Durable progress and day-boundary weights; never claims physical-world resume."""
from __future__ import annotations

import json
from pathlib import Path
import time

from ..stage.month import DAY_S
from .checkpoint import save_checkpoint
from .provenance import file_sha256


def write_json(path, data):
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    pending.replace(path)


class RunJournal:
    def __init__(self, output, days, manifest):
        self.output, self.days = Path(output), days
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.last_time = None
        self.daily = []
        self.last_day_cost, self.last_updates, self.last_learning_reward = 0.0, 0, 0.0
        self.last_roles, self.last_cranes = {}, {}
        write_json(self.output / "manifest.json", manifest)
        self.event("start", {"code": manifest["code"], "config": manifest["ppo"],
                             "n_days": len(days), "learning_days": len(days) - 2})
        write_json(self.output / "status.json", {"state": "building_world", "day": 0,
                                                  "time_s": 0, "pid": manifest["code"]["pid"]})

    def event(self, kind, row):
        message = {"event": kind, "wall_seconds": time.perf_counter() - self.started, **row}
        line = json.dumps(message, ensure_ascii=False, allow_nan=False)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(line + "\n")
        print(line, flush=True)

    def update(self, row):
        self.event("update", row)

    def checkpoint(self, name, runtime):
        target = self.output / name
        if target.exists():
            raise FileExistsError(target)
        pending = target.with_name(target.name + ".partial")
        save_checkpoint(pending, runtime)
        pending.replace(target)
        return {"path": name, "sha256": file_sha256(target),
                "scope": "weights-and-optimizer; no physical-world resume",
                "pending_intervals": len(runtime.buffer)}

    def boundary(self, runtime):
        t = runtime.time_s
        if t == self.last_time:
            return
        self.last_time = t
        if t == 0:
            self.last_day_cost = runtime.cost_krw
        if t > 0 and t % DAY_S == 0 and t <= len(self.days) * DAY_S:
            day = self.days[int(t // DAY_S) - 1]
            for sim in runtime.mbt.blocks.values():
                sim.check_invariants()
            row = {"day": day.index + 1, "train": day.is_train, "load": day.load,
                   "time_s": t, "interval_cost_krw": runtime.cost_krw - self.last_day_cost,
                   "cost_krw": runtime.cost_krw, "updates": len(runtime.updates) - self.last_updates,
                   "learning_reward": runtime.learning_reward - self.last_learning_reward,
                   "roles": {k: v - self.last_roles.get(k, 0) for k, v in runtime.role_counts.items()},
                   "crane_actions": {k: v - self.last_cranes.get(k, 0)
                                     for k, v in runtime.crane_actions.items()},
                   "physical_invariants_checked": True,
                   "checkpoint": self.checkpoint(f"day_{day.index + 1:02d}.pt", runtime)}
            self.daily.append(row)
            self.last_day_cost, self.last_updates = runtime.cost_krw, len(runtime.updates)
            self.last_learning_reward = runtime.learning_reward
            self.last_roles, self.last_cranes = dict(runtime.role_counts), dict(runtime.crane_actions)
            write_json(self.output / "days.json", self.daily)
            self.event("day", row)
        if t % 3600 == 0:
            phase = ("warmup" if t < DAY_S else "training" if runtime.collecting_at(t)
                     else "cooldown" if t < len(self.days) * DAY_S else "drain")
            state = {"state": "running", "phase": phase, "time_s": t,
                     "day": min(len(self.days), int(t // DAY_S) + 1),
                     "completed_days": len(self.daily), "updates": len(runtime.updates),
                     "learning_intervals": runtime.learning_intervals,
                     "cost_krw": runtime.cost_krw, "roles": dict(runtime.role_counts),
                     "wall_seconds": time.perf_counter() - self.started}
            write_json(self.output / "status.json", state)
            self.event("progress", state)

    def fail(self, error, runtime):
        row = {"state": "failed", "error": f"{type(error).__name__}: {error}",
               "time_s": runtime.time_s, "completed_days": len(self.daily),
               "restart": "Use the same pinned code/seed in a NEW directory from the beginning"}
        write_json(self.output / "status.json", row)
        self.event("failed", row)
