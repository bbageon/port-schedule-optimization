"""작업 계획 — deferred commit 단위 (YR-036).

dispatch 시점에 JobPlan 을 미리 계산(스택 미변형)하고 예약만 잡는다. 물리 실현(스택 변형)은
JOB_COMPLETED 에서 moves 를 순차 커밋한다 → 진행 중 이동을 다른 크레인이 관측 불가(누출 0).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contract.schema import CandidateKind
from ..domain.models import Container


@dataclass(frozen=True)
class Move:
    """컨테이너 1개 이동 1사이클 (완료 시 커밋)."""

    container_id: str
    src: tuple[int, int, int]
    dst: tuple[int, int, int]
    loaded_gantry_m: float
    empty_gantry_m: float
    duration_s: float
    inbound: Container | None = None    # GATE_IN 신규 컨테이너 (완료 시 place)
    depart: bool = False                # 반출/선적: dst 는 차선(야드 이탈)


@dataclass(frozen=True)
class JobRef:
    """candidates_for/generator 반환 (계약 Candidate 아님 — 어댑터가 계약으로 변환)."""

    job_id: str
    token: str | None          # SERVE/PRE_REHANDLE=job_id, REPOSITION/WAIT=None
    kind: CandidateKind
    target_container: str | None
    lane_id: str | None
    eligible_crane_ids: tuple[str, ...]
    is_vessel: bool
    is_external: bool
    reposition_target_bay: float | None = None   # REPOSITION 목표 bay (YR-037)


@dataclass(frozen=True)
class JobPlan:
    crane_id: str
    job_id: str
    token: str | None
    kind: CandidateKind
    moves: tuple[Move, ...]
    corridor: tuple[float, float]       # (lo, hi) bay 축 점유
    slots: frozenset                    # 예약 (bay,row) — find_slot exclude 로 주입
    lane_id: str | None
    start_s: float
    duration_s: float
    end_bay: float
    end_row: float
    rehandles: int
    loaded_gantry_m: float
    empty_gantry_m: float
