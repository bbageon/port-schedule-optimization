"""TerminalProfile YAML 로더 — 하드코딩 금지 원칙 (구현계획 01 §2)."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..domain.models import BlockGeometry, CraneSpec, TerminalProfile
from ..domain.validators import validate_profile


def load_profile(path: str | Path) -> TerminalProfile:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    b = raw["block"]
    c = raw["crane"]
    ops = raw["operations"]
    profile = TerminalProfile(
        terminal_id=raw["terminal_id"],
        profile_date=str(raw["profile_date"]),
        assumed=bool(raw.get("assumed", True)),  # 명시 없으면 가정값으로 취급
        block=BlockGeometry(
            block_id=b["block_id"],
            bay_count=int(b["bay_count"]),
            row_count=int(b["row_count"]),
            tier_max=int(b["tier_max"]),
            bay_length_m=float(b["bay_length_m"]),
            row_width_m=float(b["row_width_m"]),
            tier_height_m=float(b["tier_height_m"]),
            transfer_row=int(raw.get("handover", {}).get("transfer_row", 0)),
        ),
        crane=CraneSpec(
            crane_id=c["crane_id"],
            service_bay_min=int(c["service_bay_min"]),
            service_bay_max=int(c["service_bay_max"]),
            gantry_speed_mps=float(c["gantry_speed_mps"]),
            trolley_speed_mps=float(c["trolley_speed_mps"]),
            hoist_speed_loaded_mps=float(c["hoist_speed_loaded_mps"]),
            hoist_speed_empty_mps=float(c["hoist_speed_empty_mps"]),
            lock_time_s=float(c["lock_time_s"]),
            unlock_time_s=float(c["unlock_time_s"]),
            truck_positioning_time_s=float(c["truck_positioning_time_s"]),
        ),
        long_wait_sla_s=float(ops["long_wait_sla_s"]),
        decision_horizon_s=float(ops["decision_horizon_s"]),
        gate_travel_estimate_s=float(ops.get("gate_travel_estimate_s", 600.0)),
    )
    validate_profile(profile)
    return profile
