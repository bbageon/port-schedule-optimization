"""도메인 모델·프로파일 검증 단위 테스트 (05 §1.1)."""
import pytest

from yard_rl.domain.enums import ContainerSize, JobFlow, LoadStatus
from yard_rl.domain.models import Container, Job
from yard_rl.domain.validators import ValidationError, validate_job, validate_scenario
from yard_rl.io.profile_loader import load_profile

PROFILE = "configs/terminals/poc_single_crane.yaml"


def _container(cid="C1", bay=1, row=1, tier=1):
    return Container(container_id=cid, size=ContainerSize.FT40,
                     load_status=LoadStatus.FULL, block="B1", bay=bay, row=row, tier=tier)


def _gate_out(job_id="J1", target="C1", gate_in=0.0, arrival=600.0):
    return Job(job_id=job_id, flow=JobFlow.GATE_OUT, release_time=0.0,
               actual_gate_in=gate_in, actual_block_arrival=arrival, target_container=target)


def test_profile_loads_and_is_assumed():
    p = load_profile(PROFILE)
    assert p.assumed is True
    assert p.block.bay_count == 24
    assert p.crane.service_bay_max <= p.block.bay_count


def test_job_arrival_order_enforced():
    bad = _gate_out(gate_in=700.0, arrival=600.0)
    with pytest.raises(ValidationError, match="NEGATIVE_DURATION"):
        validate_job(bad)


def test_gate_out_requires_target():
    j = _gate_out(target=None)
    j.target_container = None
    with pytest.raises(ValidationError, match="UNMATCHED_JOB"):
        validate_job(j)


def test_scenario_rejects_duplicate_slot():
    p = load_profile(PROFILE)
    cs = {"C1": _container("C1"), "C2": _container("C2")}  # 같은 슬롯
    with pytest.raises(ValidationError, match="DUPLICATE_EVENT"):
        validate_scenario([], cs, p)


def test_scenario_rejects_floating_container():
    p = load_profile(PROFILE)
    cs = {"C1": _container("C1", tier=2)}  # tier 1 없이 tier 2
    with pytest.raises(ValidationError, match="FLOATING_CONTAINER"):
        validate_scenario([], cs, p)


def test_scenario_rejects_missing_target():
    p = load_profile(PROFILE)
    cs = {"C1": _container("C1")}
    with pytest.raises(ValidationError, match="UNMATCHED_JOB"):
        validate_scenario([_gate_out(target="NOPE")], cs, p)
