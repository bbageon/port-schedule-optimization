"""공동 Action 제약 — 최종전략 §8.6 (YR-035)."""
import pytest
from dataclasses import replace

from yard_rl.contract import (SCHEMA_VERSION, Assignment, CandidateKind,
                              JointAction, build_minimal_transition)
from yard_rl.contract.validate import validate_joint
from yard_rl.domain.validators import ValidationError


def _joint(rec, pairs):
    return replace(rec, joint_action=JointAction(
        SCHEMA_VERSION, rec.state.now_s,
        tuple(Assignment(cid, idx, kind, "test") for cid, idx, kind in pairs)))


def test_fixture_joint_valid():
    validate_joint(build_minimal_transition())


def test_no_dup_job():
    """두 크레인이 동일 resolver_token(공유 본선 후보) 선택 → DUP_JOB."""
    rec = build_minimal_transition()
    dup = _joint(rec, [("YC-A", 2, CandidateKind.SERVE),
                       ("YC-B", 0, CandidateKind.SERVE)])
    with pytest.raises(ValidationError, match="DUP_JOB"):
        validate_joint(dup)


def test_lane_conflict():
    """두 크레인이 동일 lane(L2) 후보 동시 선택 → LANE_CONFLICT."""
    rec = build_minimal_transition()
    lane = _joint(rec, [("YC-A", 1, CandidateKind.SERVE),   # lane L2
                        ("YC-B", 1, CandidateKind.SERVE)])   # lane L2
    with pytest.raises(ValidationError, match="LANE_CONFLICT"):
        validate_joint(lane)


def test_infeasible_selection():
    """feasible=False 후보(YC-A idx6) 선택 금지."""
    rec = build_minimal_transition()
    bad = _joint(rec, [("YC-A", 6, CandidateKind.SERVE)])
    with pytest.raises(ValidationError, match="INFEASIBLE_SELECTION"):
        validate_joint(bad)


def test_kind_mismatch():
    rec = build_minimal_transition()
    bad = _joint(rec, [("YC-A", 0, CandidateKind.WAIT)])   # idx0 은 SERVE
    with pytest.raises(ValidationError, match="KIND_MISMATCH"):
        validate_joint(bad)


def test_wait_noop_excluded():
    """candidate_id=None(WAIT/no-op)은 중복·레인 검사 제외 — 둘 다 대기 가능."""
    rec = build_minimal_transition()
    both_wait = replace(rec, joint_action=JointAction(SCHEMA_VERSION, rec.state.now_s, (
        Assignment("YC-A", None, CandidateKind.WAIT, "yield"),
        Assignment("YC-B", None, CandidateKind.WAIT, "yield"))))
    validate_joint(both_wait)   # 예외 없음


def test_dup_crane_rejected():
    """§8.6 크레인당 1동작 — 같은 크레인 두 배정 → DUP_CRANE."""
    rec = build_minimal_transition()
    bad = _joint(rec, [("YC-A", 0, CandidateKind.SERVE),
                       ("YC-A", 2, CandidateKind.SERVE)])
    with pytest.raises(ValidationError, match="DUP_CRANE"):
        validate_joint(bad)


def test_none_candidate_must_be_wait():
    """candidate_id=None 인데 kind≠WAIT → KIND_MISMATCH (None⟺WAIT 불변식)."""
    rec = build_minimal_transition()
    bad = replace(rec, joint_action=JointAction(SCHEMA_VERSION, rec.state.now_s, (
        Assignment("YC-A", None, CandidateKind.SERVE, "yield"),)))
    with pytest.raises(ValidationError, match="KIND_MISMATCH"):
        validate_joint(bad)


def test_ghost_crane_rejected():
    """observations 에 없는 크레인 배정 → BAD_ASSIGN (None·비None 모두)."""
    rec = build_minimal_transition()
    ghost_wait = replace(rec, joint_action=JointAction(SCHEMA_VERSION, rec.state.now_s, (
        Assignment("GHOST-YC", None, CandidateKind.WAIT, "yield"),)))
    with pytest.raises(ValidationError, match="BAD_ASSIGN"):
        validate_joint(ghost_wait)


def test_bad_candidate_index():
    """candidate_id 가 items 범위 밖 → BAD_ASSIGN."""
    rec = build_minimal_transition()
    bad = _joint(rec, [("YC-A", 99, CandidateKind.SERVE)])
    with pytest.raises(ValidationError, match="BAD_ASSIGN"):
        validate_joint(bad)


def test_ineligible_crane():
    """자격 밖 크레인이 eligible 제한 후보 선택 → INELIGIBLE (§8.6 수행자격)."""
    rec = build_minimal_transition()
    cs = rec.observations[0].candidates            # YC-A
    shared = replace(cs.items[2], eligible_crane_ids=("YC-B",))  # YC-A 제외
    obs_a = replace(rec.observations[0],
                    candidates=replace(cs, items=cs.items[:2] + (shared,) + cs.items[3:]))
    rec2 = replace(rec, observations=(obs_a, rec.observations[1]))
    bad = _joint(rec2, [("YC-A", 2, CandidateKind.SERVE)])
    with pytest.raises(ValidationError, match="INELIGIBLE"):
        validate_joint(bad)
