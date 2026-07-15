"""통합 전이 계약 (itc-v1) — YR-035.

차량·본선·이송장비·레인·다중 YC 를 처음부터 같은 State/Action/Total Cost 계약으로 다루는
단일 통합정책(최종전략)의 데이터 계약. 이 패키지는 schema·fixture·validation·serialization 만
제공한다 — 값 생성(YR-036)·후보 생성기·resolver(YR-037)·비용 수치(YR-038)·학습(YR-039)은
후속 과제이며 계약은 그 산출물을 저장·검증한다.
"""
from __future__ import annotations

from .candidate import Candidate, CandidateSet, padding_candidate
from .cost import CostBreakdown, make_cost
from .fixtures import build_minimal_transition
from .leakage import LeakageError, field_visible
from .schema import (COST_TERMS, SCHEMA, SCHEMA_VERSION, VESSEL_FAMILY,
                     AblationGroup, CandidateKind, FieldSource, FieldSpec,
                     TimeOfKnowledge, Unit)
from .serialize import dumps, from_dict, loads, to_dict
from .state import (Assignment, GlobalState, JointAction, LaneGraph,
                    LocalObservation)
from .transition import TransitionAudit, TransitionRecord
from .validate import validate_all
from .vectors import FeatureVector, build_feature_vector, zero_feature_vector
from .vessel import (CompletionBasis, VesselUrgency, VesselUrgencyMode,
                     resolve_mode)

__all__ = [
    "SCHEMA_VERSION", "SCHEMA", "COST_TERMS", "VESSEL_FAMILY",
    "FieldSource", "TimeOfKnowledge", "AblationGroup", "Unit", "CandidateKind",
    "FieldSpec", "FeatureVector", "build_feature_vector", "zero_feature_vector",
    "Candidate", "CandidateSet", "padding_candidate",
    "VesselUrgency", "VesselUrgencyMode", "CompletionBasis", "resolve_mode",
    "GlobalState", "LocalObservation", "Assignment", "JointAction", "LaneGraph",
    "CostBreakdown", "make_cost",
    "TransitionRecord", "TransitionAudit",
    "field_visible", "LeakageError", "validate_all",
    "to_dict", "from_dict", "dumps", "loads",
    "build_minimal_transition",
]
