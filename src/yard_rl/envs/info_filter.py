"""InformationFilter — 실험별 정보 공개시점 제어 (구현계획 02 §3.1).

동일한 진실 이벤트를 쓰되 정책에 '보이는' 작업만 제한한다.
Baseline·Exp-1 은 같은 정보수준(블록 도착 이후)을 공유하며, 차이는 정책뿐이다.
"""
from __future__ import annotations

from ..domain.enums import InformationLevel
from ..domain.models import Job


def is_visible(job: Job, now: float, level: InformationLevel) -> bool:
    if not job.is_external_truck:
        # 야드 내부(본선·이송) 작업정보는 TOS 에 상시 존재 — release 이후 공개
        return job.release_time <= now
    if level == InformationLevel.BLOCK_ARRIVAL:
        return job.actual_block_arrival is not None and job.actual_block_arrival <= now
    if level == InformationLevel.GATE_IN:
        return job.actual_gate_in is not None and job.actual_gate_in <= now
    if level == InformationLevel.PRE_ADVICE:
        return True  # 사전 반출입정보 — Exp-3 (YR-011-c) 에서 ETA 필드와 함께 사용
    raise ValueError(f"미지원 정보수준 {level}")


def assert_no_leakage(jobs: list[Job], now: float, level: InformationLevel) -> None:
    """후보·관측에 미래정보가 새지 않았는지 자동검사 (02 §3, 05 §1.2)."""
    for j in jobs:
        if not is_visible(j, now, level):
            raise RuntimeError(f"정보 누출: {j.job_id} 는 level={level.value}, t={now} 에서 비공개여야 함")


def predicted_arrival(job: Job, level: InformationLevel, gate_travel_estimate_s: float) -> float | None:
    """정책이 사용할 수 있는 (수준별) 도착예상.

    - BLOCK_ARRIVAL: 예측정보 없음 (이미 도착한 작업만 보임)
    - GATE_IN: 게이트 진입시각 + 자체 소요추정 — 실제 도착과 오차 존재 (Exp-2)
    - PRE_ADVICE: 부산항 제공 ETA (외생 입력, Exp-3)
    실제 actual_block_arrival 은 절대 정책에 노출하지 않는다.
    """
    if not job.is_external_truck:
        return None
    if level == InformationLevel.GATE_IN:
        if job.actual_gate_in is None:
            return None
        return job.actual_gate_in + gate_travel_estimate_s
    if level == InformationLevel.PRE_ADVICE:
        return job.provided_eta
    return None
