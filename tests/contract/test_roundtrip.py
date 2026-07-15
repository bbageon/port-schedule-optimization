"""전이 계약 직렬화 왕복 — YR-035 수용기준 핵심."""
import math

import pytest
from dataclasses import replace

from yard_rl.contract import build_minimal_transition, dumps, loads
from yard_rl.contract.vectors import _canon


def test_transition_roundtrip():
    """loads(dumps(rec)) == rec — 모든 필드·마스크·ID 감사필드 무손실."""
    rec = build_minimal_transition()
    assert loads(dumps(rec)) == rec


def test_roundtrip_idempotent():
    """dumps(loads(dumps(rec))) == dumps(rec) — 정렬·float 정규화 bit-안정."""
    rec = build_minimal_transition()
    s = dumps(rec)
    assert dumps(loads(s)) == s


def test_id_audit_fields_preserved():
    """candidate_id → ref_job_id·resolver_token·lane_id 역참조 복원 가능."""
    rec = loads(dumps(build_minimal_transition()))
    c0 = rec.observations[0].candidates.items[0]
    assert c0.ref_job_id == "J-OUT-1"
    assert c0.resolver_token == "JOUT1"
    assert c0.lane_id == "L1"
    shared = rec.observations[0].candidates.items[2]
    assert shared.eligible_crane_ids == ("YC-A", "YC-B")


def test_no_inf_nan_rejected():
    """inf/nan 은 값이 아니라 결측 — _canon 이 거부하고 dumps 가 전파."""
    with pytest.raises(ValueError):
        _canon(math.inf)
    with pytest.raises(ValueError):
        _canon(math.nan)
    rec = build_minimal_transition()
    bad = replace(rec, cost=replace(rec.cost, lambda_vessel=math.inf))
    with pytest.raises(ValueError):
        dumps(bad)


def test_schema_mismatch_rejected():
    from yard_rl.contract import from_dict, to_dict
    from yard_rl.domain.validators import ValidationError
    d = to_dict(build_minimal_transition())
    d["schema_version"] = "itc-v999"
    with pytest.raises(ValidationError, match="SCHEMA_MISMATCH"):
        from_dict(d)


def test_terminal_transition_roundtrip():
    """terminal(next_state=None) 분기 왕복·검증 — serialize/from_dict/validate None 경로 동결."""
    from yard_rl.contract import validate_all
    rec = build_minimal_transition()
    term = replace(rec, next_state=None, next_observations=(), terminal=True)
    validate_all(term)
    back = loads(dumps(term))
    assert back == term
    assert back.next_state is None
    assert back.next_observations == ()
    assert dumps(loads(dumps(term))) == dumps(term)
