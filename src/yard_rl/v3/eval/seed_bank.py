"""Frozen independent inputs using the existing v3 synthetic demand generator.

This module generates inputs, not policy outcomes. It neither changes admissions
nor filters seeds by congestion, completion, cost, or learned-policy performance.
"""
from collections import Counter
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import gzip
import hashlib
import json
import math
from pathlib import Path

from ..stage.month import (DAY_S, LOAD_WEIGHTS, MONTH_FILL_RATIO, N_DAYS, DayPlan,
    build_month, plan_month, plan_month_vessels, truck_net_by_block)
from ..world.integrated.profiles import build_h21_profile
from ..world.integrated.terminal_stream import (DIURNAL_DRAIN_S,
    DIURNAL_NIGHT_FRAC, DIURNAL_PEAKS, diurnal_arrivals)
from ..world.integrated.yard_layout import terminal_layout
from .guards import check_bands


def primitive(value):
    if is_dataclass(value):
        return primitive(asdict(value))
    if isinstance(value, Enum):
        return primitive(value.value)
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(primitive(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [primitive(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported input value: {type(value)}")


def canonical_bytes(value):
    return json.dumps(primitive(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator_contract():
    return primitive({"generator": "yard_rl.v3.stage.month.build_month",
        "arrival_generator": "yard_rl.v3.world.integrated.terminal_stream.diurnal_arrivals",
        "day_s": DAY_S, "n_days": N_DAYS, "measurement_day_indices": list(range(1, 29)),
        "night_fraction": DIURNAL_NIGHT_FRAC, "peaks": DIURNAL_PEAKS,
        "load_weights": LOAD_WEIGHTS, "initial_fill_ratio": MONTH_FILL_RATIO,
        "lead_mode": "DIST", "arrival_cdf_grid_s": 60.0,
        "arrival_sampling": "existing stratified inverse CDF, rounded to milliseconds",
        "within_run_state": "continuous, no daily state reset",
        "between_run_state": "fresh terminal with its own initial inventory",
        "policy_pairing": "same requests, initial scenarios, and planned vessel work",
        "admission_and_flow_fallback": "existing rules unchanged; not validated as loss-free",
        "new_rescheduling_cost": False})


def seed_domains(seed, n_days=N_DAYS):
    days = [seed + 1000 * (i + 1) for i in range(n_days)]
    background = sorted({d + 1000 * (b + 1) for d in days for b in range(21)})
    return {"root": seed, "days": days, "background_numeric_seeds": background,
        "arrival_rngs": [f"h21d:arr:{d}" for d in days]}


def validate_seeds(seeds, *, prior_seeds=()):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Missing or duplicate root seeds")
    if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds):
        raise ValueError("Root seeds must be nonnegative integers")
    used = set()
    old = set(prior_seeds)
    domains = []
    for seed in seeds:
        domain = seed_domains(seed)
        numeric = {seed, *domain["days"], *domain["background_numeric_seeds"]}
        if check_bands(numeric):
            raise ValueError("Used evaluation or diagnostic seed band")
        if numeric & used or numeric & old:
            raise ValueError("Derived seed collision between runs or prior inputs")
        used.update(numeric)
        domains.append(domain)
    return domains


def create_month_payload(seed, *, days=None):
    days = plan_month(seed) if days is None else days
    profile, layout = build_h21_profile(), terminal_layout()
    built = build_month(seed, days=days, n_days=len(days), profile=profile,
        layout=layout, lead_mode="DIST", fill_ratio=MONTH_FILL_RATIO)
    vessels = plan_month_vessels(days, layout,
        truck_net=truck_net_by_block(built["schedule"]))
    # Match run_month: only day-zero initial inventory is used; vessel jobs
    # are injected later from the separate continuous-month vessel plan.
    initial = {b: replace(s, jobs=[], vessels=[], horizon_s=len(days) * DAY_S,
        drain_window_s=DIURNAL_DRAIN_S) for b, s in built["day0"]["scenarios"].items()}
    return primitive({"schema": "yard_rl.v3.independent-month-input.v1", "seed": seed,
        "days": days, "schedule": built["schedule"], "vessels": vessels,
        "initial_scenarios": initial, "profile": profile, "layout": layout.as_dict(),
        "generator_contract": generator_contract()})


def validate_payload(payload, *, require_month=True):
    seed, days, schedule = payload["seed"], payload["days"], payload["schedule"]
    if payload["generator_contract"] != generator_contract():
        raise ValueError("Synthetic generator contract changed")
    if require_month and days != primitive(plan_month(seed)):
        raise ValueError("Monthly load draws or day seeds differ from the existing generator")
    if len({e["job_id"] for e in schedule}) != len(schedule):
        raise ValueError("Duplicate request identity")
    grouped = {d["index"]: [] for d in days}
    for e in schedule:
        grouped[e["day"]].append(e)
    by_day = []
    for d in days:
        rows = grouped[d["index"]]
        expected = [round(t, 3) + d["t0"] for t in diurnal_arrivals(d["seed"], total=d["load"])]
        if [e["arrival_s"] for e in rows] != expected:
            raise ValueError(f"Arrival curve or request count changed on day {d['index']}")
        if any(not math.isfinite(e["lead_s"]) or e["lead_s"] > e["arrival_s"] - d["t0"] + 1e-8 for e in rows):
            raise ValueError("Announcement lead no longer follows the original day-local contract")
        hourly = [0] * 24
        for e in rows:
            hourly[min(23, int((e["arrival_s"] - d["t0"]) / 3600))] += 1
        by_day.append({"index": d["index"], "seed": d["seed"], "load": d["load"],
            "hourly_requests": hourly, "arrival_sha256": digest(expected),
            "flows": dict(Counter(e["flow"] for e in rows)),
            "flow_fallbacks": sum(e.get("fallback_reason") is not None for e in rows)})
    vessels = plan_month_vessels([DayPlan(**d) for d in days], terminal_layout(),
        truck_net=truck_net_by_block(schedule))
    if primitive(vessels) != payload["vessels"]:
        raise ValueError("Vessel plan differs from existing request-based planning")
    return {"seed": seed, "days": len(days), "requested_trucks": len(schedule),
        "schedule_sha256": digest(schedule), "initial_scenarios_sha256": digest(payload["initial_scenarios"]),
        "vessels_sha256": digest(payload["vessels"]), "payload_sha256": digest(payload),
        "days_detail": by_day, "curve_exactly_preserved": True,
        "flow_fallbacks": sum(d["flow_fallbacks"] for d in by_day),
        "performance_evaluated": False, "scenario_completion_validated": False}


def write_bundle(path, payload):
    path = Path(path)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as gz:
            gz.write(canonical_bytes(payload))
    return file_sha(path)


def load_bundle(path, *, expected_sha256):
    if file_sha(path) != expected_sha256:
        raise ValueError("Input bundle hash mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)
