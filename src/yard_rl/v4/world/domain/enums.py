"""도메인 Enum 정의 — 구현계획 01 §5.1.

반입/반출(JobFlow)과 공/적(LoadStatus)을 하나의 Enum 으로 합치지 않는다.
"""
from __future__ import annotations

from enum import Enum, IntEnum


class JobFlow(str, Enum):
    GATE_IN = "GATE_IN"            # 외부트럭 반입 (트럭이 컨테이너를 가져옴)
    GATE_OUT = "GATE_OUT"          # 외부트럭 반출 (야드 컨테이너를 트럭에 상차)
    VESSEL_LOAD = "VESSEL_LOAD"    # 본선 연계 (PoC: 우선순위·대기비용으로만 반영)
    VESSEL_DISCHARGE = "VESSEL_DISCHARGE"
    TRANSSHIPMENT = "TRANSSHIPMENT"
    REHANDLE = "REHANDLE"          # 재조작 (blocker 이동)


class ServiceMode(str, Enum):
    """야드크레인 물리 실행 모드 (YR-080 §1) — 행동 종류(SERVE)는 하나, 물리는 2모드.

    STORE=인계점→스택(트럭 반입·본선 양하), RETRIEVE=스택→인계점(트럭 반출·본선 적하).
    Job.service_mode 가 데이터(inbound_size 유무)로 파생 — flow 분기 하드코딩 대체.
    """

    STORE = "STORE"
    RETRIEVE = "RETRIEVE"


class RequesterType(str, Enum):
    """업무 요청 주체 (YR-080 §1) — 비용·통계·인계 구분용 (물리 실행과 독립)."""

    TRUCK = "TRUCK"
    VESSEL = "VESSEL"


class LoadStatus(str, Enum):
    FULL = "FULL"
    EMPTY = "EMPTY"


class ContainerSize(str, Enum):
    FT20 = "FT20"
    FT40 = "FT40"
    FT45 = "FT45"


class JobStatus(str, Enum):
    PLANNED = "PLANNED"      # 사전정보만 존재 (Exp-3 에서만 공개)
    RELEASED = "RELEASED"    # 정책이 선택 가능해짐
    WAITING = "WAITING"      # 차량이 블록 도착, 서비스 대기
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"


class CraneStatus(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    HANDLING = "HANDLING"
    BLOCKED = "BLOCKED"
    DOWN = "DOWN"


class InformationLevel(str, Enum):
    """정책에 공개되는 차량 정보 시점 — 구현계획 02 §3.1."""

    BLOCK_ARRIVAL = "BLOCK_ARRIVAL"  # Baseline·Exp-1: 블록 도착 이후만
    GATE_IN = "GATE_IN"              # Exp-2: 게이트 진입 이후
    PRE_ADVICE = "PRE_ADVICE"        # Exp-3: 사전 반출입정보 + 제공 ETA


class ControlScope(str, Enum):
    """허용 행동 범위 — 구현계획 02 §3.2."""

    SEQUENCE_ONLY = "SEQUENCE_ONLY"
    PLUS_POSITIONING = "PLUS_POSITIONING"
    PLUS_PRE_REHANDLE = "PLUS_PRE_REHANDLE"


class PriorityRule(IntEnum):
    """행동 = priority rule 선택 — 구현계획 02 §6 (9종)."""

    FIFO = 0
    LONGEST_WAIT = 1
    NEAREST_JOB = 2
    MIN_REHANDLE = 3
    VESSEL_PRIORITY = 4
    EARLIEST_PROVIDED_ARRIVAL = 5  # Exp-3 전용: 제공 ETA 가 가장 가까운 작업
    PRE_REHANDLE = 6               # PLUS_PRE_REHANDLE 에서만
    SAME_BAY_BATCH = 7
    WAIT_YIELD = 8
