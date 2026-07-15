"""synthetic 통합 터미널 fixture (YR-036).

다중 YC 2기(서비스 구간 겹침)·외부트럭 반입/반출·본선 양하(RISK)+선적(SYMPTOM)·
이송 fleet·레인 2+연결·장애(EquipmentDown)·계획변경(PlanChange)을 담는다. 전 항목 assumed.
"""
from __future__ import annotations

from ..contract.state import LaneGraph
from ..contract.vessel import CompletionBasis
from ..domain.enums import ContainerSize, JobFlow, LoadStatus
from ..domain.models import BlockGeometry, Container, CraneSpec, Job
from .profile import IntegratedProfile, TransferFleetSpec
from .scenario import InjectedEvent, TerminalScenario
from .vessel import VesselPlan, VesselProcess, VesselWorkType

_BLOCK = BlockGeometry(block_id="B1", bay_count=40, row_count=4, tier_max=4,
                       bay_length_m=6.5, row_width_m=2.9, tier_height_m=2.6, transfer_row=0)


def _spec(cid: str) -> CraneSpec:
    return CraneSpec(crane_id=cid, service_bay_min=1, service_bay_max=40,
                     gantry_speed_mps=2.0, trolley_speed_mps=1.0,
                     hoist_speed_loaded_mps=0.5, hoist_speed_empty_mps=0.9,
                     lock_time_s=30.0, unlock_time_s=20.0, truck_positioning_time_s=25.0)


def build_integrated_profile() -> IntegratedProfile:
    return IntegratedProfile(
        terminal_id="POC-MULTI", profile_date="2026-07-15", assumed=True, block=_BLOCK,
        cranes=(_spec("YC-A"), _spec("YC-B")),
        lane_graph=LaneGraph(("L1", "L2"), (("L1", "L2"),)),
        transfer=TransferFleetSpec("TF1", "YT", n_units=2, move_time_s=180.0),
        long_wait_sla_s=1800.0, decision_horizon_s=1800.0, safety_gap_bay=2.0,
        shift_len_s=28800.0, sts_target_moves_per_hour=25.0)


def _c(cid, bay, row, tier, size=ContainerSize.FT40):
    return Container(container_id=cid, size=size, load_status=LoadStatus.FULL,
                     block="B1", bay=bay, row=row, tier=tier)


def _gate_out(jid, target, arrival):
    return Job(job_id=jid, flow=JobFlow.GATE_OUT, release_time=0.0,
               actual_gate_in=max(0.0, arrival - 600.0), actual_block_arrival=arrival,
               target_container=target)


def build_minimal_terminal_scenario() -> TerminalScenario:
    containers = {
        "C-A1": _c("C-A1", 5, 1, 1), "C-A2": _c("C-A2", 5, 1, 2),   # A2 = C-A1 위 blocker
        "C-B1": _c("C-B1", 35, 1, 1),
        "C-VD": _c("C-VD", 10, 2, 1), "C-VL": _c("C-VL", 30, 2, 1),
        "C-F1": _c("C-F1", 15, 3, 1), "C-F2": _c("C-F2", 25, 3, 1),
    }
    jobs = [
        _gate_out("J-OUT-A", "C-A1", 300.0),   # blocker 1 → 재조작 발생
        _gate_out("J-OUT-B", "C-B1", 500.0),
        Job(job_id="J-IN-A", flow=JobFlow.GATE_IN, release_time=0.0,
            actual_gate_in=100.0, actual_block_arrival=700.0,
            inbound_size=ContainerSize.FT40, inbound_load=LoadStatus.FULL),
        Job(job_id="J-VES-D", flow=JobFlow.VESSEL_DISCHARGE, release_time=600.0,
            actual_gate_in=None, actual_block_arrival=None, target_container="C-VD",
            deadline=7200.0, priority_class=1),
        Job(job_id="J-VES-L", flow=JobFlow.VESSEL_LOAD, release_time=1200.0,
            actual_gate_in=None, actual_block_arrival=None, target_container="C-VL",
            deadline=8000.0, priority_class=1),
    ]
    vessels = [
        VesselProcess("V-DISCH", VesselWorkType.DISCHARGE, VesselPlan(
            planned_start_s=600.0, planned_completion_s=7200.0,
            completion_basis=CompletionBasis.PLAN_COMPUTED, etd_s=9000.0,
            total_moves=20, sts_move_interval_s=144.0, quay_buffer_cap=3)),
        VesselProcess("V-LOAD", VesselWorkType.LOAD, VesselPlan(
            planned_start_s=1200.0, planned_completion_s=None, completion_basis=None,
            etd_s=None, total_moves=15, sts_move_interval_s=144.0, quay_buffer_cap=3)),
    ]
    injected = [
        InjectedEvent(2000.0, "EQUIPMENT_DOWN", "YC-B"),
        InjectedEvent(2600.0, "EQUIPMENT_UP", "YC-B"),
        InjectedEvent(1500.0, "PLAN_CHANGE", "V-DISCH",
                      data=(("planned_completion_s", 6800.0),)),
    ]
    return TerminalScenario(scenario_id="min-terminal", seed=0, horizon_s=7200.0,
                            drain_window_s=3600.0, containers=containers, jobs=jobs,
                            vessels=vessels, injected_events=injected)
