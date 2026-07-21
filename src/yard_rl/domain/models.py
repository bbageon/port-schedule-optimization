"""핵심 도메인 객체 — 구현계획 01 §5.2.

시간 규약(합성 PoC): episode 시작(운영일 00:00)을 0 으로 하는 초 단위 float.
실자료 파이프라인(YR-005)에서는 UTC epoch 로 교체 예정 (01 §6.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (ContainerSize, CraneStatus, JobFlow, JobStatus, LoadStatus,
                    RequesterType, ServiceMode)


@dataclass
class Container:
    container_id: str
    size: ContainerSize
    load_status: LoadStatus
    block: str
    bay: int          # 1-based
    row: int          # 1-based (row_count 까지)
    tier: int         # 1-based (바닥 = 1)
    work_available: bool = True   # Hold·검사 반영
    special_flags: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Job:
    job_id: str
    flow: JobFlow
    # --- 시각 (초, episode 기준) ---
    release_time: float                 # 정책이 선택 가능해지는 최초시각 (내부작업용)
    actual_gate_in: float | None        # 외부트럭: 게이트 진입 실제시각
    actual_block_arrival: float | None  # 외부트럭: 블록 도착 실제시각
    provided_eta: float | None = None   # 부산항 제공 ETA (Exp-3 에서만 공개, 외생 입력)
    deadline: float | None = None       # 본선·내부작업 마감
    # --- 대상 ---
    target_container: str | None = None  # GATE_OUT·VESSEL_*: 야드 내 컨테이너
    inbound_size: ContainerSize | None = None      # 신규 반입 규격 (GATE_IN·양하 STORE)
    inbound_load: LoadStatus | None = None
    priority_class: int = 0             # 0=일반 외부트럭, 1=본선·내부 연계
    vessel_id: str | None = None        # 본선연계 job 의 소속 선박 (YR-080 인과 연결)
    # --- 런타임 상태 ---
    status: JobStatus = JobStatus.PLANNED
    assigned_crane: str | None = None
    service_start: float | None = None
    service_end: float | None = None
    rehandle_count: int = 0

    @property
    def is_external_truck(self) -> bool:
        return self.flow in (JobFlow.GATE_IN, JobFlow.GATE_OUT)

    @property
    def is_vessel_linked(self) -> bool:
        return self.flow in (JobFlow.VESSEL_LOAD, JobFlow.VESSEL_DISCHARGE, JobFlow.TRANSSHIPMENT)

    @property
    def service_mode(self) -> ServiceMode:
        """물리 실행 모드 (YR-080 §1) — **데이터 주도** 판정.

        inbound_size 가 있으면 신규 반입(STORE = 인계점→스택), 없으면 반출(RETRIEVE).
        현재는 GATE_IN 만 STORE 라 flow 분기와 완전 등가(단계 1 등가 리팩터 계약).
        본선 양하가 inbound 로 전환되면(단계 2) 자동으로 STORE 경로를 탄다.
        """
        return (ServiceMode.STORE if self.inbound_size is not None
                else ServiceMode.RETRIEVE)

    @property
    def requester_type(self) -> RequesterType:
        """업무 요청 주체 — 비용(트럭 대기 vs 선석 초과)·통계 구분용 (YR-080 §1)."""
        return (RequesterType.VESSEL if self.is_vessel_linked
                else RequesterType.TRUCK)


@dataclass
class CraneState:
    crane_id: str
    position_bay: float       # 연속 좌표 (bay 단위)
    trolley_row: float        # 연속 좌표 (row 단위; 0 = 차선/인계지점)
    available_at: float = 0.0
    assigned_job: str | None = None
    status: CraneStatus = CraneStatus.IDLE
    service_bay_min: int = 1
    service_bay_max: int = 1
    # 누적 지표
    loaded_travel_m: float = 0.0
    empty_travel_m: float = 0.0


@dataclass
class TruckState:
    truck_id: str
    job_id: str
    gate_in_time: float
    block_arrival_time: float
    service_start_time: float | None = None
    service_end_time: float | None = None


@dataclass(frozen=True)
class BlockGeometry:
    block_id: str
    bay_count: int
    row_count: int
    tier_max: int
    bay_length_m: float
    row_width_m: float
    tier_height_m: float
    transfer_row: int = 0  # 차선(트럭 인계) 위치 — row 좌표 0


@dataclass(frozen=True)
class CraneSpec:
    crane_id: str
    service_bay_min: int
    service_bay_max: int
    gantry_speed_mps: float
    trolley_speed_mps: float
    hoist_speed_loaded_mps: float
    hoist_speed_empty_mps: float
    lock_time_s: float
    unlock_time_s: float
    truck_positioning_time_s: float


@dataclass(frozen=True)
class TerminalProfile:
    """터미널 설정 — 하드코딩 금지 원칙 (구현계획 01 §2).

    assumed=True 인 프로파일의 결과물은 반드시 '가정 프로파일' 표시와 함께 보고한다.
    """

    terminal_id: str
    profile_date: str
    assumed: bool
    block: BlockGeometry
    crane: CraneSpec
    long_wait_sla_s: float
    decision_horizon_s: float
    # Exp-2: 게이트 진입 후 블록 도착예상 = gate_in + 본 추정치 (자체 추정, ETA 아님)
    gate_travel_estimate_s: float = 600.0
