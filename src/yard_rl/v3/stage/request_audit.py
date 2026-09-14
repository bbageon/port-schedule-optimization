"""Observe the existing request and cost contract without altering dispatch (YR-317-g)."""
from collections import Counter, defaultdict

from ..reward.krw import truck_wait_krw
from ..schema.lifecycle import censored_turn_time_s

EVENT_FIELDS = ("gate_in_s", "block_in_s", "service_start_s", "job_done_s", "gate_out_s")


def request_ledger(schedule, events, records, orders, *, end_s):
    by_id = defaultdict(list)
    for event in events:
        by_id[event["job_id"]].append(event)
    rows, issues = [], []
    requested = {e["job_id"] for e in schedule}
    if len(requested) != len(schedule):
        issues.append("duplicate requested job IDs")
    for key in by_id.keys() - requested:
        issues.append(f"unrequested admission event: {key}")
    for entry in schedule:
        key = entry["job_id"]
        attempts = by_id[key]
        if len(attempts) != 1:
            issues.append(f"{key}: admission outcomes={len(attempts)}")
        outcome = attempts[-1]["outcome"] if attempts else "UNPROCESSED"
        rec = records.get(key)
        times = {field: getattr(rec, field, None) for field in EVENT_FIELDS}
        tt = censored_turn_time_s(rec, end_s) if rec is not None else None
        if outcome == "ADMITTED":
            if times["gate_in_s"] is None or times["gate_in_s"] > end_s:
                state = "ADMITTED_PRE_GATE"
            elif times["gate_out_s"] is not None and times["gate_out_s"] <= end_s:
                state = "COMPLETED"
            else:
                state = "CENSORED"
        else:
            state = outcome
            if times["gate_in_s"] is not None:
                issues.append(f"{key}: gate event despite {outcome}")
        observed = [times[k] for k in EVENT_FIELDS if times[k] is not None]
        if observed != sorted(observed):
            issues.append(f"{key}: lifecycle time order")
        order = orders.get(key)
        rows.append({"job_id": key, "requested_day": entry.get("day"),
            "flow": entry["flow"], "requested_block": entry["block"],
            "requested_arrival_s": entry["arrival_s"], "lead_s": entry["lead_s"],
            "requested_target": entry.get("target"), "travel_s": entry["travel_s"],
            "admission_events": attempts, "state": state, **times,
            "final_reserved_arrival_s": getattr(order, "in_out_reserve_s", None),
            "final_block": getattr(order, "con_loc", None),
            "accounted_turn_time_s": tt,
            "accounted_wait_krw": truck_wait_krw(tt) if tt is not None else 0.0})
    states = dict(Counter(row["state"] for row in rows))
    reasons = dict(Counter(e["reason"] for e in events if e["outcome"] == "SKIPPED"))
    summary = {"requested": len(rows), "states": states, "skip_reasons": reasons,
        "admitted": sum(states.get(k, 0) for k in ("COMPLETED", "CENSORED", "ADMITTED_PRE_GATE")),
        "skipped": states.get("SKIPPED", 0), "unprocessed": states.get("UNPROCESSED", 0),
        "accounted_wait_krw": sum(row["accounted_wait_krw"] for row in rows),
        "recording_issues": issues, "recording_ok": not issues,
        "all_requests_admitted": not issues and not states.get("SKIPPED", 0),
        "end_s": end_s,
        "cost_scope": "existing gate-in to gate-out/cutoff cost; no new external-wait or rescheduling fees"}
    return rows, summary
