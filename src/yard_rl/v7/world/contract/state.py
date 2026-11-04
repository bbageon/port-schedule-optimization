"""Global State·Local Observation·Joint Action 계약 — 최종전략 §6·§8.6 (YR-035).

CTDE(중앙집중 학습·분산 실행): 학습 시 GlobalState(특권 전역), 실행 시 각 YC 의
LocalObservation 으로 행동. JointAction 은 중앙 resolver 가 공동제약을 보장한 배정 결과.
"""
from __future__ import annotations

from dataclasses import dataclass

from .candidate import CandidateSet
from .schema import CandidateKind
from .vectors import FeatureVector
from .vessel import VesselUrgency


@dataclass(frozen=True)
class LaneGraph:
    lane_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]   # 무방향 연결 (§7.6 레인 연결 그래프)


@dataclass(frozen=True)
class GlobalState:
    schema_version: str
    episode_id: str
    decision_index: int
    now_s: float
    info_level: str                  # InformationLevel.value (실험 상한)
    control_scope: str               # ControlScope.value
    profile_assumed: bool            # 가정 프로파일 여부 (TerminalProfile.assumed)
    features: FeatureVector           # group="global"
    vessels: tuple[VesselUrgency, ...]
    lane_graph: LaneGraph


@dataclass(frozen=True)
class LocalObservation:
    schema_version: str
    crane_id: str
    now_s: float
    features: FeatureVector           # group="yc"
    candidates: CandidateSet


@dataclass(frozen=True)
class Assignment:
    crane_id: str
    candidate_id: int | None         # None = WAIT/no-op (양보)
    kind: CandidateKind
    resolved_by: str                 # "local_argmin"|"central_resolver"|"yield"


@dataclass(frozen=True)
class JointAction:
    schema_version: str
    now_s: float
    assignments: tuple[Assignment, ...]   # crane_id 오름차순
