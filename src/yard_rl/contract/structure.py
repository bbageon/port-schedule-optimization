"""Level 2 터미널 구조계약 (YR-083 §최소 계약) — 블록 물리구조의 기계 표현.

목적: YR-082 Level 2 자료(블록좌표·역할·레인·인계점)가 확보되면 시뮬레이터·후보·resolver 가
**실제로 소비**할 계약을 정의한다. 현재 `IntegratedProfile`(단일 block·역할 없는 대칭 CraneSpec·
방향/용량 없는 LaneGraph·고정 이송시간)이 버리는 구조 차이를 여기서 명시적으로 담는다.

원칙:
- 미확보 값은 None (조용히 0·평균으로 채우지 않는다 — YR-082 증거 계약과 동일).
- Level 2 에서 시간 파라미터(이송·작업시간)는 `assumed` 허용, 실측 보정은 Level 3.
- 이 모듈은 **계약(자료 구조)만** 정의한다. 엔진 소비(mask/resolver)는 YR-083 step 2 (별도).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VehicleType(str, Enum):
    EXTERNAL_TRUCK = "EXTERNAL_TRUCK"   # 외부트럭 (게이트↔블록 육측)
    YT = "YT"                           # 야드트랙터 (내부 이송)
    AGV = "AGV"                         # 무인이송차 (DGT 해측)
    SC = "SC"                           # 스트래들캐리어 (BNCT·BCT)


class CraneSide(str, Enum):
    LANDSIDE = "LANDSIDE"               # 육측 — 외부트럭 인계
    WATERSIDE = "WATERSIDE"             # 해측 — AGV/본선 인계
    SHARED = "SHARED"                   # 역할 미분리 (현 엔진 = 이것만 표현)


class TransferControl(str, Enum):
    MANUAL = "MANUAL"
    TRAFFIC_LIGHT_AND_READY_BUTTON = "TRAFFIC_LIGHT_AND_READY_BUTTON"   # DGT LSTP
    FMS = "FMS"                         # DGT WSTP (AGV 자동배차)


@dataclass(frozen=True)
class RoadSegment:
    """차량 레인 구간 — 방향·중심선 길이·용량·허용차종·충돌그룹.

    공개자료로는 대부분 미확보(길이·방향·용량) → None. 부두·안벽·철도 길이를 레인 길이로
    대체 금지(YR-082). directed=None 이면 방향 미확인(무방향 가정 아님 — 미지).
    """

    seg_id: str
    allowed_vehicles: tuple[VehicleType, ...]
    directed: bool | None = None            # True=일방통행, None=미확인
    centerline_m: float | None = None
    speed_mps: float | None = None
    capacity: int | None = None
    conflict_group: str | None = None       # 교차·합류 상호배제 그룹
    provenance: str = "unresolved"


@dataclass(frozen=True)
class TransferPoint:
    """크레인↔차량 인계점 — 블록·작업면·허용차종·동시용량·제어절차."""

    tp_id: str
    block_id: str
    side: CraneSide
    allowed_vehicles: tuple[VehicleType, ...]
    capacity: int | None = None             # 동시 처리 차량 수 (미확보 None)
    control: TransferControl | None = None
    position_bay: int | None = None
    provenance: str = "unresolved"


@dataclass(frozen=True)
class CraneRole:
    """블록 소속 크레인의 역할 — 육/해/공유·허용작업·담당 Bay·인계점.

    현 엔진의 CraneSpec 은 side/allowed_work/transfer_point_ids 가 없다 (전부 SHARED 취급).
    """

    crane_id: str
    block_id: str
    side: CraneSide
    service_bay_min: int
    service_bay_max: int
    allowed_work: tuple[str, ...] = ()      # JobFlow.value 들 (빈 튜플=제약 미지정=전부)
    transfer_point_ids: tuple[str, ...] = ()
    provenance: str = "assumed"


@dataclass(frozen=True)
class CraneInteraction:
    """동일 블록 크레인 간 상호작용 — 통과 가능·비통과 선후·최소 안전거리."""

    can_cross: bool | None = None           # None=공개 미확인
    non_cross_order: tuple[str, ...] = ()   # 비통과 시 crane_id 선후 (bay 오름차 등)
    min_separation_bay: float | None = None
    provenance: str = "assumed"


@dataclass(frozen=True)
class TransferFleet:
    """이송 fleet — 차종·허용경로·대수·이동시간(Level2 assumed 허용)·provenance."""

    fleet_id: str
    vehicle_type: VehicleType
    n_units: int | None = None
    move_time_s: float | None = None
    allowed_route_ids: tuple[str, ...] = ()
    buffer_capacity: int | None = None
    provenance: str = "assumed"


@dataclass(frozen=True)
class BlockStructure:
    """블록 기하 — Bay/Row/Tier + (Level2) 좌표·진입출구·작업면."""

    block_id: str
    bay_count: int | None
    row_count: int | None
    tier_max: int | None
    orientation: str | None = None          # HORIZONTAL | VERTICAL | None(미확인)
    coordinates: tuple | None = None        # (x,y) 등 Level2 — 미확보 None
    provenance: str = "assumed"


@dataclass(frozen=True)
class StructureContract:
    """터미널 1개의 Level 2 구조계약 — 위 계약들의 조립."""

    terminal_id: str
    archetype: str
    blocks: tuple[BlockStructure, ...]
    crane_roles: tuple[CraneRole, ...]
    crane_interaction: CraneInteraction
    transfer_points: tuple[TransferPoint, ...]
    road_segments: tuple[RoadSegment, ...]
    transfer_fleets: tuple[TransferFleet, ...]
    role_separated: bool | None = None       # 육/해측 분리 여부 (None=미확인)
    provenance: str = "assumed"
    notes: tuple[str, ...] = ()

    @property
    def sides(self) -> frozenset[CraneSide]:
        return frozenset(r.side for r in self.crane_roles)

    @property
    def is_role_split(self) -> bool:
        """실제로 육/해측 역할이 갈렸는가 (SHARED 만이면 False)."""
        return bool(self.sides - {CraneSide.SHARED})

    @property
    def vehicle_types(self) -> frozenset[VehicleType]:
        return frozenset(f.vehicle_type for f in self.transfer_fleets)
