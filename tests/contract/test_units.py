"""단위 일관성 — 초/미터/비율/원 혼용 방지 (YR-035)."""
import pytest
from dataclasses import replace

from yard_rl.contract import SCHEMA, build_minimal_transition
from yard_rl.contract.validate import validate_units_fv
from yard_rl.domain.validators import ValidationError


def test_fixture_units_consistent():
    """fixture 전 벡터가 단위·범위·clip 를 만족."""
    rec = build_minimal_transition()
    validate_units_fv("state", rec.state.features)
    for v in rec.state.vessels:
        validate_units_fv("vessel", v.features)
    for o in rec.observations:
        validate_units_fv("yc", o.features)
        validate_units_fv("queue", o.candidates.queue_summary)
        for c, pad in zip(o.candidates.items, o.candidates.pad_mask):
            if pad:
                validate_units_fv("cand", c.features)


def test_ratio_out_of_range_caught():
    """ratio_0_1 필드가 1 초과면 UNIT_RANGE."""
    rec = build_minimal_transition()
    fv = rec.state.features
    i = fv.names.index("lane_congestion_mean")
    tampered = replace(fv, value=fv.value[:i] + (1.5,) + fv.value[i + 1:])
    with pytest.raises(ValidationError, match="UNIT_RANGE"):
        validate_units_fv("state", tampered)


def test_bool01_enforced():
    rec = build_minimal_transition()
    fv = rec.observations[0].features
    i = fv.names.index("is_loaded")
    tampered = replace(fv, value=fv.value[:i] + (0.5,) + fv.value[i + 1:])
    with pytest.raises(ValidationError, match="UNIT_RANGE"):
        validate_units_fv("yc", tampered)


def test_negative_slack_allowed():
    """slack_s 는 음수 허용 (§7.9) — clip 없음."""
    v = [sp for sp in SCHEMA.group_specs("vessel") if sp.name == "slack_s"][0]
    assert v.clip_lo is None
