"""Connect a declared vertical geometry to the existing continuous simulator.

Canonical demand is generated BEFORE applying this adapter. Only travel fields
are derived anew; request times, identities, flow, inventory and vessel plans
are conserved. The synthetic defaults are a transparent implementation fixture,
not measured terminal coordinates or a preregistered performance experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from .geometry import VerticalLayoutSpec, VerticalYardLayout, build_vertical_layout
from ..world.integrated.terminal_stream import TerminalStreamParams


def synthetic_vertical_spec(profile, *, n_blocks: int = 21) -> dict:
    """Explicit geometry assumptions in metres; no fitted performance values."""
    g = profile.block
    return {
        "kind": "VERTICAL_END",
        "geometry": asdict(VerticalLayoutSpec(
            n_blocks=n_blocks, bay_count=g.bay_count, row_count=g.row_count,
            tier_max=g.tier_max, bay_length_m=g.bay_length_m,
            row_width_m=g.row_width_m, block_gap_m=13.0,
            land_road_clearance_m=26.0, water_road_clearance_m=26.0,
            gate_approach_m=1000.0, quay_approach_m=1000.0, speed_mps=5.0)),
        "discharge_pipeline_capacity": 3,
        "source_exit_mean_s": TerminalStreamParams.exit_travel_mu_s,
    }


def _positive(name, value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class VerticalEnvironment:
    layout: VerticalYardLayout
    discharge_pipeline_capacity: int
    source_exit_mean_s: float

    @classmethod
    def from_dict(cls, spec: dict, profile, source_layout) -> VerticalEnvironment:
        required = {"kind", "geometry", "discharge_pipeline_capacity", "source_exit_mean_s"}
        if set(spec) != required or spec["kind"] != "VERTICAL_END":
            raise ValueError("A complete VERTICAL_END environment specification is required")
        geometry = VerticalLayoutSpec(**spec["geometry"])
        for key in ("bay_count", "row_count", "tier_max", "bay_length_m", "row_width_m"):
            if getattr(geometry, key) != getattr(profile.block, key):
                raise ValueError(f"Layout/input block mismatch: {key}")
        if geometry.n_blocks != source_layout.n:
            raise ValueError("Layout/input block count mismatch")
        if len(profile.cranes) != 2:
            raise ValueError("Vertical model requires two nonpassing cranes per block")
        gap = _positive("crane safety gap", profile.safety_gap_bay)
        parking_m = math.ceil(gap) * geometry.bay_length_m
        if min(geometry.land_road_clearance_m, geometry.water_road_clearance_m) <= parking_m:
            raise ValueError("Roads must lie beyond the declared crane parking extensions")
        cap = spec["discharge_pipeline_capacity"]
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("discharge_pipeline_capacity must be a positive integer")
        exit_mean = _positive("source_exit_mean_s", spec["source_exit_mean_s"])
        if exit_mean != TerminalStreamParams.exit_travel_mu_s:
            raise ValueError("Exit reference must match the canonical demand generator")
        return cls(build_vertical_layout(geometry, ids=source_layout.ids), cap, exit_mean)

    def adapt_schedule(self, schedule: list[dict]) -> list[dict]:
        """Preserve sampled deviations; never clip back to legacy road bounds.

        The inbound residual is relative to the old block-specific route.
        The exit residual has a DIFFERENT origin: the old generator's fixed
        exit mean (300 s), not its gate-to-block distance. Exit uses the reverse
        of the new bidirectional landside path. Reject impossible durations.
        """
        out = []
        for entry in schedule:
            base = _positive("source travel_base_s", entry["travel_base_s"])
            actual = _positive("source travel_s", entry["travel_s"])
            exit_time = _positive("source exit_travel_s", entry["exit_travel_s"])
            new_base = _positive("route time", self.layout.gate_to_block_s(entry["block"]))
            row = dict(entry)
            row["travel_base_s"] = new_base
            row["travel_s"] = _positive("derived travel_s", new_base + actual - base)
            row["exit_travel_s"] = _positive(
                "derived exit_travel_s", new_base + exit_time - self.source_exit_mean_s)
            out.append(row)
        return out

    def make_sim(self, scenario, profile, block_id: str):
        from .runtime import VerticalTerminalSimulator
        from ..stage.episode import INFO_LEVEL

        # Preserve the existing aggregate transfer-cycle approximation. The
        # cycle now consumes the declared road distance, not the old 180 s.
        cycle = self.layout.yt_round_trip_s(block_id)
        block_profile = replace(profile, transfer=replace(
            profile.transfer, kind="YT", move_time_s=cycle))
        land, row = self.layout.local_interface("LANDSIDE")
        water, _ = self.layout.local_interface("WATERSIDE")
        return VerticalTerminalSimulator(
            block_profile, scenario, landside_bay=land, waterside_bay=water,
            transfer_row=row, yard_handover_cap=self.discharge_pipeline_capacity,
            check_invariants=True, info_level=INFO_LEVEL)

    def manifest(self, *, canonical_input: dict, schedule: list[dict]) -> dict:
        from ..eval.seed_bank import digest
        return {
            "kind": "VERTICAL_END", "synthetic": True,
            "layout": self.layout.as_dict(),
            "canonical_input": canonical_input,
            "runtime_schedule_sha256": digest(schedule),
            "derived_fields": ["travel_base_s", "travel_s", "exit_travel_s"],
            "source_exit_mean_s": self.source_exit_mean_s,
            "vessel_deadlines": "canonical source profile; runtime physical lower bound recorded separately",
            "discharge_pipeline_capacity": self.discharge_pipeline_capacity,
            "crane_roles": "profile crane order: external / vessel; full stock access",
            "crane_parking": "outside each end by ceil(safety_gap_bay); no storage",
            "candidate_pruning": "feasible-first optional candidates; preserve all mandatory candidates and at least one feasible action when available",
            "transfer_cycles_s": {b: self.layout.yt_round_trip_s(b) for b in self.layout.ids},
            "limitations": [
                "Synthetic coordinates and parking extensions, not a measured terminal",
                "Aggregate round trip: delivery and vehicle return are not separate events",
                "No road traffic or intersection queue simulation",
                "Geometry, interfaces, roles and finite pipeline change together",
                "No training or layout-performance validation is implied by implementation",
            ],
        }
