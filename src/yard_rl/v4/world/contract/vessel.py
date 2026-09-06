"""본선 위험도↔지연징후 판별 — 최종전략 §7.9·7.10 (YR-035).

계획 본선작업 완료시각을 확보하면 '위험도(risk)'를, 확보하지 못하면 STS·이송장비 대기
기반 '지연 징후 점수(delay_symptom_score)'만 산출한다. 둘을 섞지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .vectors import FeatureVector


class VesselUrgencyMode(str, Enum):
    NONE = "NONE"        # 본선작업 없음/무관
    RISK = "RISK"        # 계획 완료시각 확보 → 정량 위험도
    SYMPTOM = "SYMPTOM"  # 완료시각 결측 → 지연 징후 점수만


class CompletionBasis(str, Enum):
    """계획 완료시각 확보 우선순위 (§7.9). tier3·4 는 가정값(assumed)."""

    TOS_TARGET = "TOS_TARGET"            # 1순위: TOS/본선계획 목표 완료시각
    PLAN_COMPUTED = "PLAN_COMPUTED"      # 2순위: 운영계획상 계산값
    ATD_MINUS_BUFFER = "ATD_MINUS_BUFFER"  # 3순위: 출항예정 - 준비버퍼 (가정)
    OPERATOR_TEMP = "OPERATOR_TEMP"      # 4순위: 운영자 임시 마감 (가정)


_ASSUMED_BASES = frozenset({CompletionBasis.ATD_MINUS_BUFFER, CompletionBasis.OPERATOR_TEMP})


@dataclass(frozen=True)
class VesselUrgency:
    vessel_id: str
    mode: VesselUrgencyMode
    completion_basis: CompletionBasis | None
    assumed: bool                     # 완료시각이 가정값 기반인가
    features: FeatureVector           # group="vessel"


def resolve_mode(planned_completion_s: float | None,
                 basis: CompletionBasis | None) -> tuple[VesselUrgencyMode, bool]:
    """완료시각·근거로 mode·assumed 결정 (§7.10).

    완료시각 또는 근거가 없으면 RISK 를 금지하고 SYMPTOM 으로 강등한다.
    """
    if planned_completion_s is None or basis is None:
        return VesselUrgencyMode.SYMPTOM, False
    return VesselUrgencyMode.RISK, basis in _ASSUMED_BASES
