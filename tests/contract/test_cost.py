"""정규화 구간 비용 항등식 — 최종전략 §10 (YR-035)."""
import pytest
from dataclasses import replace

from yard_rl.contract import VESSEL_FAMILY, build_minimal_transition
from yard_rl.contract.validate import validate_cost
from yard_rl.domain.validators import ValidationError


def test_cost_identity():
    """Σcontributions == total, reward == -total, 항목 중복계상 0."""
    rec = build_minimal_transition()
    c = rec.cost
    validate_cost(c)
    assert abs(sum(c.contributions().values()) - c.total_normalized) < 1e-6
    assert abs(c.reward + c.total_normalized) < 1e-6


def test_lambda_only_on_vessel_family():
    """λ_vessel 은 본선 계열 항에만 곱 (§10.6)."""
    rec = build_minimal_transition()
    c = rec.cost
    contrib = c.contributions()
    for k in VESSEL_FAMILY:
        expected = c.lambda_vessel * c.weight[k] * c.raw[k] / c.scale[k]
        assert abs(contrib[k] - expected) < 1e-6
    # 비본선 항에는 λ 미적용
    k = "truck_wait"
    assert abs(contrib[k] - c.weight[k] * c.raw[k] / c.scale[k]) < 1e-6


def test_broken_identity_caught():
    rec = build_minimal_transition()
    bad = replace(rec.cost, total_normalized=rec.cost.total_normalized + 1.0)
    with pytest.raises(ValidationError, match="COST_IDENTITY"):
        validate_cost(bad)


def test_zero_scale_caught():
    rec = build_minimal_transition()
    bad_scale = dict(rec.cost.scale)
    bad_scale["truck_wait"] = 0.0
    with pytest.raises(ValidationError, match="ZERO_SCALE"):
        validate_cost(replace(rec.cost, scale=bad_scale))


def test_missing_cost_term_caught():
    rec = build_minimal_transition()
    bad_raw = dict(rec.cost.raw)
    del bad_raw["imbalance"]
    with pytest.raises(ValidationError, match="COST_TERMS"):
        validate_cost(replace(rec.cost, raw=bad_raw))


def test_nonfinite_raw_caught():
    """raw 에 inf/nan → COST_NONFINITE (항등식보다 먼저; merge 키충돌 회귀 방지)."""
    import math
    rec = build_minimal_transition()
    for bad_val in (math.inf, math.nan):
        bad_raw = dict(rec.cost.raw)
        bad_raw["truck_wait"] = bad_val
        with pytest.raises(ValidationError, match="COST_NONFINITE"):
            validate_cost(replace(rec.cost, raw=bad_raw))
    bad_scale = dict(rec.cost.scale)
    bad_scale["rehandle"] = math.inf
    with pytest.raises(ValidationError, match="COST_NONFINITE"):
        validate_cost(replace(rec.cost, scale=bad_scale))


def test_full_precision_cost_roundtrips():
    """6자리 초과 정밀 cost 도 loads(dumps)==rec·validate 통과 (make_cost 양자화)."""
    from yard_rl.contract import COST_TERMS, dumps, loads, make_cost, validate_all
    raw = {k: 0.12345678 for k in COST_TERMS}
    scale = {k: 1.0 for k in COST_TERMS}
    weight = {k: 1.0 for k in COST_TERMS}
    c = make_cost(interval_start_s=0.0, interval_end_s=300.0, raw=raw, scale=scale,
                  weight=weight, lambda_vessel=2.7777779)
    rec = replace(build_minimal_transition(), cost=c)
    validate_all(rec)
    back = loads(dumps(rec))
    assert back == rec
    validate_all(back)
