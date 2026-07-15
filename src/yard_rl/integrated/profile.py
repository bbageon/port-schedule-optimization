"""통합 터미널 프로파일 — 다중 YC·레인·이송 fleet 설정 (YR-036).

전 항목 assumed (PoC). 미확보 확률/물리 파라미터를 임의 확정하지 않는다 — 실측 보정은
YR-002/009. 단일 YC TerminalProfile 과 별개로 두어 기존 엔진을 오염시키지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.state import LaneGraph
from ..domain.models import BlockGeometry, CraneSpec


@dataclass(frozen=True)
class TransferFleetSpec:
    fleet_id: str
    kind: str                  # "YT" | "AGV" | "SC"
    n_units: int
    move_time_s: float         # 본선↔야드 왕복 assumed


@dataclass(frozen=True)
class IntegratedProfile:
    terminal_id: str
    profile_date: str
    assumed: bool
    block: BlockGeometry
    cranes: tuple[CraneSpec, ...]        # 다중 YC (service_bay 구간 겹침 허용)
    lane_graph: LaneGraph
    transfer: TransferFleetSpec
    long_wait_sla_s: float
    decision_horizon_s: float
    safety_gap_bay: float = 2.0          # 비통과 크레인 간 최소 bay 간격 (assumed)
    shift_len_s: float = 28800.0         # 8h 교대 (shift_idx 산출)
    gate_travel_estimate_s: float = 600.0
    sts_target_moves_per_hour: float = 25.0   # assumed 목표 생산성 (cadence 산출)

    @property
    def sts_move_interval_s(self) -> float:
        return 3600.0 / self.sts_target_moves_per_hour
