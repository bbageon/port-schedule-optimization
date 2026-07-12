"""도메인·프로파일 검증 — 오류를 조용히 통과시키지 않는다 (01 §6.3)."""
from __future__ import annotations

from .enums import JobFlow
from .models import Container, Job, TerminalProfile


class ValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def validate_profile(p: TerminalProfile) -> None:
    b, c = p.block, p.crane
    if b.bay_count < 1 or b.row_count < 1 or b.tier_max < 1:
        raise ValidationError("PROFILE_GEOMETRY", "bay/row/tier 는 1 이상이어야 함")
    if min(b.bay_length_m, b.row_width_m, b.tier_height_m) <= 0:
        raise ValidationError("PROFILE_GEOMETRY", "블록 치수는 양수여야 함")
    speeds = (c.gantry_speed_mps, c.trolley_speed_mps,
              c.hoist_speed_loaded_mps, c.hoist_speed_empty_mps)
    if min(speeds) <= 0:
        raise ValidationError("PROFILE_SPEED", "크레인 속도는 양수여야 함")
    if min(c.lock_time_s, c.unlock_time_s, c.truck_positioning_time_s) < 0:
        raise ValidationError("PROFILE_TIME", "취급시간은 음수 불가")
    if not (1 <= c.service_bay_min <= c.service_bay_max <= b.bay_count):
        raise ValidationError("PROFILE_RANGE", "service range 가 블록 범위를 벗어남")
    if p.long_wait_sla_s <= 0 or p.decision_horizon_s <= 0:
        raise ValidationError("PROFILE_OPS", "SLA·horizon 은 양수여야 함")


def validate_container(c: Container, profile: TerminalProfile) -> None:
    b = profile.block
    if not (1 <= c.bay <= b.bay_count and 1 <= c.row <= b.row_count and 1 <= c.tier <= b.tier_max):
        raise ValidationError("INVALID_SLOT", f"{c.container_id}: 슬롯 ({c.bay},{c.row},{c.tier}) 범위 밖")


def validate_job(j: Job) -> None:
    if j.is_external_truck:
        if j.actual_gate_in is None or j.actual_block_arrival is None:
            raise ValidationError("UNMATCHED_JOB", f"{j.job_id}: 외부트럭 작업에 도착시각 없음")
        if j.actual_block_arrival < j.actual_gate_in:
            raise ValidationError("NEGATIVE_DURATION", f"{j.job_id}: 블록도착 < 게이트진입")
    if j.flow == JobFlow.GATE_OUT and j.target_container is None:
        raise ValidationError("UNMATCHED_JOB", f"{j.job_id}: GATE_OUT 인데 대상 컨테이너 없음")
    if j.flow == JobFlow.GATE_IN and (j.inbound_size is None or j.inbound_load is None):
        raise ValidationError("UNMATCHED_JOB", f"{j.job_id}: GATE_IN 인데 반입 규격 없음")
    if j.is_vessel_linked and j.deadline is None:
        raise ValidationError("UNMATCHED_JOB", f"{j.job_id}: 본선·내부 작업에 deadline 없음")
    if j.release_time < 0:
        raise ValidationError("NEGATIVE_DURATION", f"{j.job_id}: release_time 음수")


def validate_scenario(jobs: list[Job], containers: dict[str, Container],
                      profile: TerminalProfile) -> None:
    """시나리오 일관성: 대상 컨테이너 존재·중복 슬롯·시각 순서."""
    seen_slots: set[tuple[int, int, int]] = set()
    for c in containers.values():
        validate_container(c, profile)
        slot = (c.bay, c.row, c.tier)
        if slot in seen_slots:
            raise ValidationError("DUPLICATE_EVENT", f"슬롯 중복 점유 {slot}")
        seen_slots.add(slot)
    # tier 연속성 (공중 적재 금지)
    occupied = {(c.bay, c.row, c.tier) for c in containers.values()}
    for (bay, row, tier) in occupied:
        if tier > 1 and (bay, row, tier - 1) not in occupied:
            raise ValidationError("FLOATING_CONTAINER", f"({bay},{row},{tier}) 아래가 비어 있음")
    seen_ids: set[str] = set()
    for j in jobs:
        if j.job_id in seen_ids:
            raise ValidationError("DUPLICATE_EVENT", f"작업 ID 중복 {j.job_id}")
        seen_ids.add(j.job_id)
        validate_job(j)
        if j.target_container is not None and j.target_container not in containers:
            raise ValidationError("UNMATCHED_JOB",
                                  f"{j.job_id}: 대상 컨테이너 {j.target_container} 가 야드에 없음")
