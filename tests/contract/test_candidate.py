"""동적 후보·마스크 계약 — 최종전략 §8 (YR-035)."""
import pytest
from dataclasses import replace

from yard_rl.contract import (SCHEMA, CandidateKind, build_minimal_transition)
from yard_rl.contract.validate import validate_candidates
from yard_rl.domain.validators import ValidationError


def test_fixture_candidates_valid():
    rec = build_minimal_transition()
    for o in rec.observations:
        validate_candidates(o.candidates)


def test_pad_feasible_subset():
    """feasible ⊆ pad, 패딩 자리 zero, mask_reason 일관 (YC-A 는 패딩 슬롯 보유)."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    assert cs.pad_mask[-1] is False           # 마지막은 패딩
    assert cs.feasible_mask[-1] is False
    assert cs.mask_reason[-1] == "PAD"
    assert not any(cs.items[-1].features.known)  # 패딩 자리 전 채널 known=0
    # feasible ⊆ pad
    for feas, pad in zip(cs.feasible_mask, cs.pad_mask):
        assert not (feas and not pad)


def test_pad_nonzero_rejected():
    """패딩 자리에 known 채널이 있으면 PAD_NONZERO."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    real_fv = rec.observations[0].candidates.items[0].features
    bad_pad = replace(cs.items[-1], features=real_fv)
    bad_cs = replace(cs, items=cs.items[:-1] + (bad_pad,))
    with pytest.raises(ValidationError, match="PAD_NONZERO"):
        validate_candidates(bad_cs)


def test_infeasible_requires_reason():
    """feasible=False ⇔ mask_reason≠None."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    bad = replace(cs, mask_reason=cs.mask_reason[:6] + (None,) + cs.mask_reason[7:])
    with pytest.raises(ValidationError, match="MASK_REASON"):
        validate_candidates(bad)


def test_pad_value_nonzero_rejected():
    """패딩 슬롯 value≠0(known=0)도 zero-padding 불변식 위반 → PAD_NONZERO."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    pad = cs.items[-1]
    bad_f = replace(pad.features, value=(999.0,) + pad.features.value[1:])
    bad = replace(cs, items=cs.items[:-1] + (replace(pad, features=bad_f),))
    with pytest.raises(ValidationError, match="PAD_NONZERO"):
        validate_candidates(bad)


def test_feasible_pad_subset_enforced():
    """feasible=True 인데 pad=False → INFEASIBLE_PAD."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    bad = replace(cs, feasible_mask=cs.feasible_mask[:-1] + (True,))  # 패딩 자리 feasible
    with pytest.raises(ValidationError, match="INFEASIBLE_PAD"):
        validate_candidates(bad)


def test_candidate_id_position_enforced():
    """candidate_id 는 items 위치와 일치해야 함 → CANDIDATE_ID."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    bad = replace(cs, items=(replace(cs.items[0], candidate_id=5),) + cs.items[1:])
    with pytest.raises(ValidationError, match="CANDIDATE_ID"):
        validate_candidates(bad)


def test_mask_length_mismatch():
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates
    bad = replace(cs, pad_mask=cs.pad_mask[:-1])   # 길이 불일치
    with pytest.raises(ValidationError, match="LENGTH_MISMATCH"):
        validate_candidates(bad)


def test_mandatory_present():
    """SLA 임박 mandatory 후보가 보존 (§8.2 pruning 금지 대상)."""
    rec = build_minimal_transition()
    mand = [c for c in rec.observations[0].candidates.real_items if c.mandatory]
    assert len(mand) >= 1
    assert mand[0].kind == CandidateKind.SERVE


def test_no_job_id_channel():
    """candidate FeatureVector 에 Job-ID 파생 채널 없음 (신원 암기 금지, YR-039 비목표)."""
    names = SCHEMA.names("candidate")
    assert "ref_job_id" not in names
    assert not any(nm.endswith("_id") or "job" in nm for nm in names)


def test_all_four_kinds_representable():
    rec = build_minimal_transition()
    kinds = {c.kind for o in rec.observations for c in o.candidates.real_items}
    assert {CandidateKind.SERVE, CandidateKind.PRE_REHANDLE,
            CandidateKind.REPOSITION, CandidateKind.WAIT} <= kinds
