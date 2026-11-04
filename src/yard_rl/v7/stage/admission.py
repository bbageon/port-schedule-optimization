"""Read-only admission diagnostics, outside the frozen physical engine."""


def truck_admission_event(announcer, terminal, t, previous_skips):
    failures = []
    new = announcer.skips[previous_skips:]
    entries = ({e["job_id"]: e for e in announcer.by_epoch.get(round(t, 6), [])}
               if new else {})
    for skip in new:
        entry = entries[skip["job_id"]]
        bid = entry["block"]
        row = {**skip, "block": bid, "flow": entry["flow"],
               "arrival_s": entry["arrival_s"], "travel_s": entry["travel_s"]}
        if bid in terminal.blocks:
            sim = terminal.blocks[bid]
            row["block_snapshot"] = {
                "scope": "after admission batch, before policy boundary",
                "time_s": t, "clock_s": sim.clock,
                "containers": len(sim.stacks.containers), "jobs": len(sim.jobs),
                "free_slots": terminal.free_slots(bid),
                "capacity_margin": terminal.capacity_margin}
        failures.append(row)
    return {"kind": "trucks", "time_s": t, "admitted": announcer.n_admitted,
            "skipped": announcer.n_skipped, "failures": failures}
