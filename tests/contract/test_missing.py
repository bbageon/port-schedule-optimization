"""결측·가정 표기 검증 — known=0 중화, 필수/가정 근거 (YR-035)."""
import pytest

from yard_rl.contract import (SCHEMA, FeatureVector, build_feature_vector)
from yard_rl.contract.validate import (validate_assumed_fv, validate_missing_fv)
from yard_rl.domain.enums import InformationLevel
from yard_rl.domain.validators import ValidationError

_CAND = SCHEMA.names("candidate")


def _fv(known_vals: dict, assumed: set = frozenset()):
    """{name: value} 로 candidate FeatureVector 수동 구성 (검증 음성 케이스용)."""
    value, kn, asm = [], [], []
    for nm in _CAND:
        if nm in known_vals:
            value.append(float(known_vals[nm]))
            kn.append(True)
        else:
            value.append(0.0)
            kn.append(False)
        asm.append(nm in assumed)
    return FeatureVector(SCHEMA.version, "candidate", _CAND,
                         tuple(value), tuple(kn), tuple(asm))


def test_missing_value_zeroed():
    """known=0 인데 value≠0 이면 STALE_MISSING."""
    fv = _fv({})
    bad = FeatureVector(fv.schema_version, fv.group, fv.names,
                        (5.0,) + fv.value[1:], fv.known, fv.assumed)
    with pytest.raises(ValidationError, match="STALE_MISSING"):
        validate_missing_fv("x", bad)


def test_required_field_present():
    """nullable=False 필드 결측 → REQUIRED_MISSING (reach_s 는 필수)."""
    raw = {"reach_s": None, "is_external": 0.0, "is_vessel": 0.0, "action_kind_idx": 0.0,
           "expected_service_time_s": 10.0, "expected_handling_count": 1.0,
           "blocker_count": 0.0, "expected_rehandle_time_s": 0.0, "end_bay": 1.0,
           "lane_congestion_local": 0.0, "interference_penalty_s": 0.0, "resequence_count": 0.0}
    fv = build_feature_vector("candidate", raw, now=0.0,
                              info_level=InformationLevel.PRE_ADVICE)
    assert fv.known_of("reach_s") is False
    with pytest.raises(ValidationError, match="REQUIRED_MISSING"):
        validate_missing_fv("x", fv)


def test_assumed_requires_basis():
    """assumed=1 인데 assumed_default 없는 필드면 ASSUMED_NO_BASIS (reach_s)."""
    fv = _fv({"reach_s": 50.0}, assumed={"reach_s"})
    with pytest.raises(ValidationError, match="ASSUMED_NO_BASIS"):
        validate_assumed_fv("x", fv)


def test_assumed_without_known_rejected():
    """assumed=1 인데 known=0 → ASSUMED_NO_BASIS (첫 분기)."""
    fv = _fv({}, assumed={"reach_s"})   # reach_s known=0 이지만 assumed=1
    with pytest.raises(ValidationError, match="ASSUMED_NO_BASIS"):
        validate_assumed_fv("x", fv)


def test_imputed_field_is_assumed():
    """assumed_default 있는 필드는 결측 시 imputed + assumed=1 (vessel remaining_service_time_s)."""
    fv = build_feature_vector("vessel", {"remaining_service_time_s": None,
                                         "remaining_moves": 10.0, "sts_wait_s": 0.0,
                                         "transfer_wait_s": 0.0, "delay_symptom_score": 0.3},
                              now=0.0, info_level=InformationLevel.PRE_ADVICE)
    v, kn, asm = fv.channel("remaining_service_time_s")
    assert kn is True and asm is True and v == 1200.0
