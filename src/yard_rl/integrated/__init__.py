"""통합 터미널 이벤트 시뮬레이터 (YR-036).

차량·본선·이송장비·레인·다중 YC 를 같은 시계에서 처리하는 이벤트 엔진. 단일 YC sim/engine.py
는 동결하고 순수 프리미티브만 재사용한다. 계약(YR-035)의 값을 산출하며 값 생성기·resolver·
비용수치·학습은 YR-037~039. 실측 validation 은 YR-002/009 (전 항목 assumed).
"""
from __future__ import annotations

from .adapter import capture, record_episode
from .audit import (CandidateVerdict, CraneResolution, JointResolution,
                    resolution_stream_hash)
from .candidates import CandidateGenerator, GenCandidate, GeneratedCandidates
from .cost import (ASSUMED_SCALE, ASSUMED_WEIGHT, CostAccumulator,
                   assumed_lambda_vessel)
from .cost_config import (LambdaMode, LambdaVesselPolicy, Provenance, ProvBasis,
                          RewardCalculator, RiskBand, TermCost, TerminalCostConfig,
                          default_assumed_config)
from .dispatcher import ReferenceDispatcher
from .ledger import (CostCause, CostLedger, assert_ledger_identity, build_ledger_report)
from .engine import (CommitProjection, CraneAssignment, TerminalDecision,
                     TerminalSimulator)
from .resolver import (BaselinePreference, CentralResolver, DispatcherPreference)
from .events import EventKind, EventQueue
from .fixtures import build_integrated_profile, build_minimal_terminal_scenario
from .jobplan import JobPlan, JobRef, Move
from .profile import IntegratedProfile, TransferFleetSpec
from .reservation import Corridor, Reservation, ReservationTable
from .scenario import InjectedEvent, TerminalScenario
from .vessel import (VesselPlan, VesselProcess, VesselTruth, VesselWorkType)

__all__ = [
    "TerminalSimulator", "TerminalDecision", "CraneAssignment", "CommitProjection",
    "ReferenceDispatcher", "record_episode", "capture",
    "CandidateGenerator", "GenCandidate", "GeneratedCandidates",
    "CentralResolver", "BaselinePreference", "DispatcherPreference",
    "JointResolution", "CraneResolution", "CandidateVerdict", "resolution_stream_hash",
    "IntegratedProfile", "TransferFleetSpec", "TerminalScenario", "InjectedEvent",
    "VesselProcess", "VesselPlan", "VesselTruth", "VesselWorkType",
    "ReservationTable", "Reservation", "Corridor",
    "JobPlan", "JobRef", "Move", "EventKind", "EventQueue",
    "CostAccumulator", "ASSUMED_SCALE", "ASSUMED_WEIGHT", "assumed_lambda_vessel",
    "TerminalCostConfig", "RewardCalculator", "default_assumed_config", "LambdaVesselPolicy",
    "LambdaMode", "RiskBand", "Provenance", "ProvBasis", "TermCost",
    "CostLedger", "CostCause", "assert_ledger_identity", "build_ledger_report",
    "build_integrated_profile", "build_minimal_terminal_scenario",
]
