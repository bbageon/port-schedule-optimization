"""Read-only admission diagnostics; never consulted by a policy or an admission rule."""
from collections import Counter, defaultdict

from ..world.domain.enums import JobFlow, JobStatus


def inventory_snapshot(mbt, bid):
    """Separate physical inventory, claimed boxes and reservations at the failure time."""
    if bid not in mbt.blocks:
        return {"block": bid, "available": False, "reason": "UNKNOWN_BLOCK"}
    sim = mbt.blocks[bid]
    containers = set(sim.stacks.containers)
    claimed = {j.target_container for j in sim.jobs.values()
               if j.target_container is not None}
    pending = Counter(j.flow.value for j in sim.jobs.values()
                      if j.status == JobStatus.PLANNED and j.flow in
                      (JobFlow.GATE_IN, JobFlow.VESSEL_DISCHARGE))
    block = sim.profile.block
    capacity = block.bay_count * block.row_count * block.tier_max
    reserved = mbt._reserved_inbound[bid]
    physical_free = capacity - len(containers)
    return {"block": bid, "available": True, "at_s": float(sim.clock), "physical_capacity": capacity,
        "inventory_boxes": len(containers), "physical_free": physical_free,
        "claimed_inventory_boxes": len(containers & claimed),
        "unclaimed_inventory_boxes": len(containers - claimed),
        "planned_inbound_by_flow": dict(pending), "transfer_reservations": reserved,
        "admission_free": mbt.free_slots(bid), "capacity_margin": mbt.capacity_margin,
        "free_formula_matches": physical_free - sum(pending.values()) - reserved == mbt.free_slots(bid)}


class VesselWorkAudit:
    """Retain completed yard-job identities before daily pruning removes them."""

    def __init__(self, plan):
        self.rows = {}
        self.done = defaultdict(set)
        self.active = defaultdict(set)
        self.issues = []
        for row in plan:
            key = row["key"]
            if key in self.rows:
                self.issues.append(f"duplicate vessel plan: {key}")
            self.rows[key] = {"key": key, "block": row["block"], "work": row["work"],
                "start_s": row["start_s"], "asked": int(row["moves"]),
                "attempted": False, "admitted": 0, "sts_done": None,
                "sts_remaining": None, "sts_completion_s": None}

    def admission(self, row):
        key = row["key"]
        if key not in self.rows:
            self.issues.append(f"unplanned vessel admission: {key}")
            return
        item = self.rows[key]
        if item["attempted"]:
            self.issues.append(f"duplicate vessel admission: {key}")
        if row.get("asked") != item["asked"]:
            self.issues.append(f"vessel requested count mismatch: {key}")
        item.update(attempted=True, admitted=row.get("moves", 0),
                    admission_ok=row["ok"], reason=row.get("why", ""))

    def observe(self, mbt):
        active = defaultdict(set)
        for bid, sim in mbt.blocks.items():
            for jid, job in sim.jobs.items():
                key = getattr(job, "vessel_id", None)
                if key is None:
                    continue
                if key not in self.rows:
                    self.issues.append(f"unplanned vessel job: {jid}")
                    continue
                if job.status == JobStatus.DONE:
                    self.done[key].add(jid)
                else:
                    active[key].add(jid)
                    if jid in self.done[key]:
                        self.issues.append(f"completed vessel job became active: {jid}")
            for key, vessel in sim.vessels.items():
                if key not in self.rows:
                    self.issues.append(f"unplanned vessel process: {key}")
                    continue
                item = self.rows[key]
                if bid != item["block"] or vessel.plan.total_moves != item["admitted"]:
                    self.issues.append(f"vessel process admission mismatch: {key}")
                item.update(sts_done=bool(vessel.done),
                    sts_remaining=(vessel.remaining_moves if vessel.started else vessel.plan.total_moves),
                    sts_completion_s=vessel.truth.actual_completion_s)
        self.active = active

    def finish(self, mbt):
        self.observe(mbt)
        rows, issues = [], list(self.issues)
        for key, item in self.rows.items():
            completed, outstanding = len(self.done[key]), len(self.active[key])
            missing = item["admitted"] - completed - outstanding
            row = dict(item, completed_yard_jobs=completed, outstanding_yard_jobs=outstanding,
                       unaccounted_yard_jobs=missing, unadmitted_moves=item["asked"] - item["admitted"])
            rows.append(row)
            if not item["attempted"]:
                issues.append(f"vessel admission not observed: {key}")
            if not 0 <= item["admitted"] <= item["asked"] or missing != 0:
                issues.append(f"vessel work does not reconcile: {key}")
            if item["admitted"] and item["sts_done"] is None:
                issues.append(f"vessel process not observed: {key}")
            if item["sts_done"] and (item["sts_remaining"] != 0 or item["sts_completion_s"] is None):
                issues.append(f"vessel completion record inconsistent: {key}")
        summary = {"streams": len(rows), "requested_moves": sum(r["asked"] for r in rows),
            "admitted_moves": sum(r["admitted"] for r in rows),
            "unadmitted_moves": sum(r["unadmitted_moves"] for r in rows),
            "completed_yard_jobs": sum(r["completed_yard_jobs"] for r in rows),
            "outstanding_yard_jobs": sum(r["outstanding_yard_jobs"] for r in rows),
            "unaccounted_yard_jobs": sum(r["unaccounted_yard_jobs"] for r in rows),
            "recording_ok": not issues, "issues": issues,
            "all_requested_work_completed": bool(rows) and not issues and
                all(r["unadmitted_moves"] == 0 and r["outstanding_yard_jobs"] == 0 and
                    r["sts_done"] is True for r in rows)}
        return rows, summary
