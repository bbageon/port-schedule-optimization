"""정보누출 차단 — 미래정보·상한초과·금지원천 (YR-035)."""
import pytest
from dataclasses import replace

from yard_rl.contract import (SCHEMA, build_feature_vector,
                              build_minimal_transition, validate_all)
from yard_rl.contract.leakage import LeakageError
from yard_rl.contract.schema import FieldSource, TimeOfKnowledge
from yard_rl.domain.enums import InformationLevel

_RAW = {"predicted_arrival_gap_s": 600.0, "eta_confidence": 0.8, "reach_s": 50.0,
        "is_external": 1.0, "is_vessel": 0.0, "action_kind_idx": 0.0,
        "expected_service_time_s": 100.0, "expected_handling_count": 1.0,
        "blocker_count": 0.0, "expected_rehandle_time_s": 0.0, "end_bay": 5.0,
        "lane_congestion_local": 0.3, "interference_penalty_s": 0.0, "resequence_count": 0.0}


def test_no_forbidden_field_in_registry():
    """NEVER·GROUND_TRUTH 필드는 스키마에 물리적으로 부재 (1차 방어선)."""
    for sp in SCHEMA.specs:
        assert sp.tok != TimeOfKnowledge.NEVER
        assert sp.source != FieldSource.GROUND_TRUTH


def test_future_info_masked():
    """realized_at 이 미래면 known=0·value=0 (미도착 트럭 누적대기)."""
    fv = build_feature_vector("candidate", {**_RAW, "cum_wait_s": 999.0},
                              now=100.0, info_level=InformationLevel.PRE_ADVICE,
                              realized_at={"cum_wait_s": 700.0})
    assert fv.known_of("cum_wait_s") is False
    assert fv.value_of("cum_wait_s") == 0.0


def test_info_level_source_gating():
    """BLOCK_ARRIVAL 레벨은 ETA 마스크, PRE_ADVICE 는 공개."""
    ba = build_feature_vector("candidate", _RAW, now=100.0,
                              info_level=InformationLevel.BLOCK_ARRIVAL)
    pa = build_feature_vector("candidate", _RAW, now=100.0,
                              info_level=InformationLevel.PRE_ADVICE)
    assert ba.known_of("predicted_arrival_gap_s") is False
    assert ba.value_of("predicted_arrival_gap_s") == 0.0
    assert pa.known_of("predicted_arrival_gap_s") is True


def test_ablation_off_masks_group():
    off = build_feature_vector("candidate", _RAW, now=100.0,
                               info_level=InformationLevel.PRE_ADVICE,
                               ablation_off={"ETA"})
    assert off.known_of("predicted_arrival_gap_s") is False


def test_record_leakage_caught_on_level_downgrade():
    """PRE_ADVICE 로 만든 ETA-known 레코드를 BLOCK_ARRIVAL 로 표기하면 누출 검출."""
    rec = build_minimal_transition()
    downgraded = replace(rec, state=replace(rec.state,
                                            info_level=InformationLevel.BLOCK_ARRIVAL.value))
    with pytest.raises(LeakageError):
        validate_all(downgraded)
