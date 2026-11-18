"""Gate-entry evidence for offline calibration; never substitutes a log clock.

Identifiers stay in memory/private quarantine. An explicit time may be supplied
by another request for the exact same container, truck, direction and time.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import gzip
from pathlib import Path
import re

REQUEST = re.compile(r"GateInResult REQUEST TO TSS|/api/vbs/terminal/GateIn\s*:")
CONFIRMED = frozenset(("confirmed_explicit", "confirmed_other_message"))


def parse_time(value):
    value = (value or "").strip()
    if value in ("", "null", "None"):
        return None
    result = (datetime.strptime(value, "%Y%m%d%H%M%S")
              if re.fullmatch(r"\d{14}", value) else datetime.fromisoformat(value))
    if result.tzinfo is not None:
        raise ValueError("Timezone-aware data requires a separately declared clock conversion")
    return result


def direction(value):
    value = {"1": "OUT", "FR": "OUT", "2": "IN", "TO": "IN"}.get(value, value)
    if value not in ("IN", "OUT"):
        raise ValueError("Unknown gate direction")
    return value


def field_value(line, name):
    values = {v.strip() for v in re.findall(r"\b" + name + r"\s*=\s*([^,}\]]*)", line)}
    if len(values) > 1:
        raise ValueError(f"Conflicting raw request field: {name}")
    return next(iter(values), "")


def key(container, truck, at, flow):
    if not container or not truck:
        raise ValueError("Missing gate container/truck join key")
    return (container.strip(), truck.strip(), at, direction(flow))


def visit_key(row):
    return key(row.get("con_no"), row.get("truck_no"), parse_time(row.get("gate_in_time")), row["in_out"])


@dataclass
class GateEvidence:
    explicit: set = field(default_factory=set)
    log_only: set = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)

    def consume(self, line):
        self.counts["lines"] += 1
        if not REQUEST.search(line):
            return
        self.counts["gate_requests"] += 1
        container, truck = field_value(line, "conNo"), field_value(line, "truckNo")
        flow = field_value(line, "inOut")
        event_time = parse_time(field_value(line, "gateInTime"))
        log_time = parse_time(line[:19])
        if log_time is None:
            raise ValueError("Gate request lacks a log clock")
        if event_time is not None:
            self.explicit.add(key(container, truck, event_time, flow))
            self.counts["explicit_gate_time_requests"] += 1
        else:
            self.log_only.add(key(container, truck, log_time, flow))
            self.counts["requests_without_gate_time"] += 1

    def classify(self, row):
        at = parse_time(row.get("gate_in_time"))
        if at is None:
            return "no_gate_time"
        k = visit_key(row)
        if k in self.explicit:
            return "confirmed_other_message" if k in self.log_only else "confirmed_explicit"
        return "unconfirmed_log_only" if k in self.log_only else "unconfirmed_no_request_match"


def load_evidence(paths):
    paths = sorted(paths)
    if not paths:
        raise ValueError("Raw gate request logs are required; cannot infer entry from CSV alone")
    proof = GateEvidence()
    for path in paths:
        with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
            for line in handle:
                proof.consume(line)
    if not proof.counts["gate_requests"]:
        raise ValueError("No gate requests found in supplied logs")
    return proof


def partition(visits, proof):
    groups = {name: [] for name in (*sorted(CONFIRMED), "unconfirmed_log_only",
                                   "unconfirmed_no_request_match", "no_gate_time")}
    for row in visits:
        groups[proof.classify(row)].append(row)
    return groups
