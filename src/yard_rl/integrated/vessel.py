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


# ══════════════════════════════════════════════════════════════════════════════
# 선급 3종 — v3 무대 구성 (사용자 결정 2026-08-20)
# 정본: `.claude/docs/architecture/02b-본선.md`
#
# ★여기는 **무대(물리)** 다. 원화 단가는 v3 의 `reward/krw.py` 가 갖는다.
#   무대는 "얼마나 큰 배가 몇 개를 몇 시간 동안 처리하나" 만 정하고,
#   그걸 돈으로 바꾸는 일은 세대별 보상 축이 한다.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VesselClass:
    """선급 하나 — 크기(GT)·선복(TEU)·안벽 크레인 수."""

    name: str
    gt: int
    teu: int
    sts: int                      # STS 대수 = 동시 스트림 수
    stream_moves_per_h: float = 27.5   # 스트림당 25~30 의 중앙값

    def moves_range(self) -> tuple[int, int]:
        """기항 물량 범위 = 선복 × 15~30%. 만선으로 오지 않는다."""
        return (round(self.teu * 0.15), round(self.teu * 0.30))

    def work_time_s(self, moves: int) -> float:
        """순수 하역 시간 = 물량 ÷ (STS × 스트림 생산성)."""
        return moves / (self.sts * self.stream_moves_per_h) * 3600.0


#: 선급 3종 — (이름, GT, TEU, STS). v3 `reward/krw.py` 의 표와 **같아야** 한다.
VESSEL_CLASSES: tuple[VesselClass, ...] = (
    VesselClass("SMALL", 50_000, 3_000, sts=2),
    VesselClass("MEDIUM", 100_000, 7_500, sts=4),
    VesselClass("LARGE", 150_000, 14_000, sts=6),
)

#: 스트림(STS) 한 대에 붙는 YT 대수 — 사용자 결정 2026-08-20
YT_PER_STREAM = 6

#: 한국 평균 Port Time 실측표 (moves 상한, 시간). **유도하지 않고 그대로 쓴다.**
#: 접안 시간 ⊃ 작업 시간 이고, 차이가 유휴(정박 대기·조선·접이안·검역·교대)다.
PORT_TIME_TABLE: tuple[tuple[int, float], ...] = (
    (500, 15.8), (1_000, 20.4), (1_500, 25.1), (2_000, 27.7), (2_500, 31.1),
    (3_000, 34.3), (4_000, 38.6), (6_000, 47.5),
)
PORT_TIME_ABOVE_H = 62.6          # 6,000 moves 초과


def port_time_s(moves: int) -> float:
    """기항 물량 → 접안 시간(초). 실측표 lookup — 계단식이다."""
    for hi, hours in PORT_TIME_TABLE:
        if moves <= hi:
            return hours * 3600.0
    return PORT_TIME_ABOVE_H * 3600.0


def sample_vessel_moves(cls: VesselClass, u: float) -> int:
    """`moves ~ TEU × U(0.15, 0.30)` — `u` 는 [0,1) 균등 난수다.

    난수를 안에서 뽑지 않고 **받는다**: 시드에서 미리 추출해야 재현된다.
    """
    lo, hi = cls.moves_range()
    return int(round(lo + (hi - lo) * float(u)))


def vessel_class_by_name(name: str) -> VesselClass:
    for c in VESSEL_CLASSES:
        if c.name == name:
            return c
    raise KeyError(f"알 수 없는 선급: {name}")
