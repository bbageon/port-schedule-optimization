"""본선 프로세스 — STS cadence·Slack·지연 (YR-036, 최종전략 §7.8·7.9·7.10).

STS 가 결정론 cadence 로 move 를 처리하며 remaining_moves 감소, 본선연계 job 을 야드로
발생시킨다. 대기(STS blocked)는 자원경합(버퍼 만재/staged 없음)에서 창발한다. 완료시각
결측 ⟺ SYMPTOM (계약 vessel.resolve_mode 와 정합). 미확보 분포는 assumed cadence 로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..contract.vessel import CompletionBasis


class VesselWorkType(str, Enum):
    DISCHARGE = "DISCHARGE"   # 양하 (본선→야드)
    LOAD = "LOAD"             # 선적 (야드→본선)


@dataclass(frozen=True)
class VesselPlan:
    """PLANNED — 정책 가시. completion 결측 가능(→SYMPTOM)."""

    planned_start_s: float
    planned_completion_s: float | None
    completion_basis: CompletionBasis | None
    etd_s: float | None
    total_moves: int
    sts_move_interval_s: float          # assumed cadence = 3600/목표생산성
    quay_buffer_cap: int = 3            # assumed STS 홀딩 버퍼
    # YR-106-b 게이트 A: **정책 무관 최소 완료시각** (YC→YT→STS 전체 사슬). 생성기가
    # `vessel_deadline_achievable=True` 일 때만 채운다. None = 구계약(STS 단독으로 추정).
    phys_min_completion_s: float | None = None


@dataclass
class VesselTruth:
    """GROUND_TRUTH/NEVER — 비용정산 전용, feature 진입 금지."""

    actual_completion_s: float | None = None


@dataclass
class VesselProcess:
    vessel_id: str
    work_type: VesselWorkType
    plan: VesselPlan
    truth: VesselTruth = field(default_factory=VesselTruth)
    started: bool = False
    remaining_moves: int = -1            # -1 = 미개시
    buffer_level: int = 0                # 안벽 버퍼 점유 (DISCHARGE)
    sts_blocked_since_s: float | None = None
    sts_wait_accum_s: float = 0.0
    done: bool = False

    def remaining_service_time_s(self) -> float:
        rem = max(0, self.remaining_moves) if self.started else self.plan.total_moves
        return rem * self.plan.sts_move_interval_s

    def slack_s(self, now: float) -> float | None:
        pc = self.plan.planned_completion_s
        if pc is None:
            return None
        return pc - now - self.remaining_service_time_s()

    def expected_delay_s(self, now: float) -> float | None:
        pc = self.plan.planned_completion_s
        if pc is None:
            return None
        return max(0.0, now + self.remaining_service_time_s() - pc)

    def structural_min_overrun_s(self) -> float:
        """YR-109 진단 — **정책과 무관한** 최소 선석초과 (야드가 무한히 빨라도 남는 몫).

        계획완료가 물리 최소완료보다 앞서면 그 차이는 어떤 야드 정책으로도 못 줄이는
        상수다. 이 상수가 vessel_delay 를 총비용의 ~70% 로 밀어올려 판정 분산을
        지배했다(설계감사 2026-07-27).

        **YR-106-b 게이트 A**: 하한은 생성기가 계산한 `phys_min_completion_s`
        (YC→YT→STS 전체 사슬)를 쓴다. 없으면(구계약 시나리오) STS 단독 하한으로
        후퇴하는데, 이는 적하(LOAD)에서 **과소평가**임을 알고 쓰는 값이다.
        """
        pc = self.plan.planned_completion_s
        if pc is None:
            return 0.0
        phys_min = self.plan.phys_min_completion_s
        if phys_min is None:
            phys_min = (self.plan.planned_start_s
                        + self.plan.total_moves * self.plan.sts_move_interval_s)
        return max(0.0, phys_min - pc)

    def is_symptom(self) -> bool:
        return self.plan.planned_completion_s is None or self.plan.completion_basis is None

    @property
    def sts_blocked(self) -> bool:
        return self.sts_blocked_since_s is not None
