"""본선 위험도↔지연징후 판별 — 최종전략 §7.10 (YR-035)."""
import pytest
from dataclasses import replace

from yard_rl.contract import (CompletionBasis, VesselUrgencyMode,
                              build_minimal_transition, resolve_mode)
from yard_rl.contract.validate import validate_vessel
from yard_rl.domain.validators import ValidationError


def test_risk_demotion_on_missing_completion():
    """완료시각 결측 → RISK 금지·SYMPTOM 강등."""
    assert resolve_mode(None, None)[0] == VesselUrgencyMode.SYMPTOM
    assert resolve_mode(5000.0, CompletionBasis.TOS_TARGET)[0] == VesselUrgencyMode.RISK


def test_assumed_completion_basis():
    """3·4순위 근거(출항예정-버퍼·운영자임시)는 assumed=True."""
    assert resolve_mode(5000.0, CompletionBasis.ATD_MINUS_BUFFER)[1] is True
    assert resolve_mode(5000.0, CompletionBasis.OPERATOR_TEMP)[1] is True
    assert resolve_mode(5000.0, CompletionBasis.PLAN_COMPUTED)[1] is False


def test_fixture_vessels_valid():
    """fixture: V1=RISK(완료시각 확보), V2=SYMPTOM(결측)."""
    rec = build_minimal_transition()
    v1, v2 = rec.state.vessels
    assert v1.mode == VesselUrgencyMode.RISK
    assert v1.features.known_of("risk") is True
    assert v2.mode == VesselUrgencyMode.SYMPTOM
    assert v2.features.known_of("risk") is False
    assert v2.features.known_of("delay_symptom_score") is True
    validate_vessel("v1", v1)
    validate_vessel("v2", v2)


def _set_known(f, name, value):
    i = f.names.index(name)
    return replace(f, value=f.value[:i] + (value,) + f.value[i + 1:],
                   known=f.known[:i] + (True,) + f.known[i + 1:])


def test_symptom_with_risk_known_rejected():
    """SYMPTOM 인데 risk known=1 이면 위험도·징후 혼용 — VESSEL_MODE."""
    rec = build_minimal_transition()
    v2 = rec.state.vessels[1]
    with pytest.raises(ValidationError, match="VESSEL_MODE"):
        validate_vessel("v2", replace(v2, features=_set_known(v2.features, "risk", 0.5)))


def test_risk_with_symptom_known_rejected():
    """RISK 인데 delay_symptom_score known=1 → 혼용 VESSEL_MODE (SYMPTOM 분기와 대칭)."""
    rec = build_minimal_transition()
    v1 = rec.state.vessels[0]
    bad = _set_known(v1.features, "delay_symptom_score", 0.4)
    with pytest.raises(ValidationError, match="VESSEL_MODE"):
        validate_vessel("v1", replace(v1, features=bad))


def test_risk_assumed_flag_mismatch_rejected():
    """RISK 인데 assumed 플래그가 완료근거와 불일치 → VESSEL_MODE."""
    rec = build_minimal_transition()
    v1 = rec.state.vessels[0]   # basis=PLAN_COMPUTED(비가정)인데 assumed=True 로 오염
    with pytest.raises(ValidationError, match="VESSEL_MODE"):
        validate_vessel("v1", replace(v1, assumed=True))


def test_none_mode_with_known_channel_rejected():
    """NONE 모드인데 위험/징후 채널 known=1 → VESSEL_MODE."""
    rec = build_minimal_transition()
    v2 = rec.state.vessels[1]
    none_v = replace(v2, mode=VesselUrgencyMode.NONE, completion_basis=None,
                     features=_set_known(v2.features, "risk", 0.3))
    with pytest.raises(ValidationError, match="VESSEL_MODE"):
        validate_vessel("v", none_v)
