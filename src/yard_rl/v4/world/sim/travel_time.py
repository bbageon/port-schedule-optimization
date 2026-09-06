"""크레인 이동·서비스시간 모델 — 구현계획 02 §2.

가정(assumed): 축 순차동작 합산 모델. Gantry/Trolley/Hoist 동시동작 장비라면
실측 보정 시 max/phase 모델로 교체한다. 이동거리 KPI 는 Gantry 축 미터만 집계
(03 §4.2 적재/비적재 Gantry 이동거리).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import BlockGeometry, CraneSpec

Slot = tuple[int, int, int]  # (bay, row, tier) — row 0 = 차선(트럭/인계지점), tier 1 = 차량 섀시 높이


@dataclass(frozen=True)
class MoveResult:
    duration_s: float
    loaded_gantry_m: float
    empty_gantry_m: float
    end_bay: float
    end_row: float


def _clearance_tier(geom: BlockGeometry) -> int:
    return geom.tier_max + 1


def gantry_m(geom: BlockGeometry, from_bay: float, to_bay: float) -> float:
    return abs(from_bay - to_bay) * geom.bay_length_m


def trolley_m(geom: BlockGeometry, from_row: float, to_row: float) -> float:
    return abs(from_row - to_row) * geom.row_width_m


def hoist_leg_s(geom: BlockGeometry, spec: CraneSpec, tier: int, *, loaded: bool) -> float:
    """스프레더가 통과높이(clearance)↔해당 tier 를 한 번 이동하는 시간."""
    dist = (_clearance_tier(geom) - tier) * geom.tier_height_m
    speed = spec.hoist_speed_loaded_mps if loaded else spec.hoist_speed_empty_mps
    return dist / speed


def move_container(spec: CraneSpec, geom: BlockGeometry,
                   start_bay: float, start_row: float,
                   src: Slot, dst: Slot) -> MoveResult:
    """컨테이너 1개를 src → dst 로 옮기는 1 사이클.

    빈 주행(현위치→src) → 하강(빈)·lock·상승(적재) → 적재 주행(src→dst)
    → 하강(적재)·unlock·상승(빈).
    """
    e_gantry = gantry_m(geom, start_bay, src[0])
    e_trolley = trolley_m(geom, start_row, src[1])
    l_gantry = gantry_m(geom, src[0], dst[0])
    l_trolley = trolley_m(geom, src[1], dst[1])
    duration = (
        e_gantry / spec.gantry_speed_mps
        + e_trolley / spec.trolley_speed_mps
        + hoist_leg_s(geom, spec, src[2], loaded=False)   # 하강(빈)
        + spec.lock_time_s
        + hoist_leg_s(geom, spec, src[2], loaded=True)    # 상승(적재)
        + l_gantry / spec.gantry_speed_mps
        + l_trolley / spec.trolley_speed_mps
        + hoist_leg_s(geom, spec, dst[2], loaded=True)    # 하강(적재)
        + spec.unlock_time_s
        + hoist_leg_s(geom, spec, dst[2], loaded=False)   # 상승(빈)
    )
    return MoveResult(duration_s=duration, loaded_gantry_m=l_gantry,
                      empty_gantry_m=e_gantry, end_bay=float(dst[0]), end_row=float(dst[1]))


def estimate_reach_s(spec: CraneSpec, geom: BlockGeometry,
                     from_bay: float, from_row: float, to_bay: float, to_row: float) -> float:
    """빈 크레인이 목표 지점까지 이동하는 예상시간 (NEAREST_JOB 판정 등)."""
    return (gantry_m(geom, from_bay, to_bay) / spec.gantry_speed_mps
            + trolley_m(geom, from_row, to_row) / spec.trolley_speed_mps)
