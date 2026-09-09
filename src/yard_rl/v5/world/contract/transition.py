"""전이 레코드 — round-trip 직렬화의 최상위 단위 (YR-035).

이벤트 기반 SMDP(§5): dt_s 는 결정 사이 실제 경과시간이며 effective_gamma 는 학습기가
파생(γ_base 는 하이퍼파라미터, YR-039). next_observations 를 함께 담는 이유는 Double DQN
target Q(s', argmin a') 이 다음 상태의 후보블록·feasible_mask 를 요구하기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cost import CostBreakdown
from .state import GlobalState, JointAction, LocalObservation


@dataclass(frozen=True)
class TransitionAudit:
    built_at_now_s: float
    info_level: str
    ablation_off: tuple[str, ...]        # AblationGroup.value 목록
    missing_fields: tuple[str, ...]      # known=0 경로 ("candidate[3].eta_confidence" 등)
    assumed_fields: tuple[str, ...]      # assumed=1 경로
    forbidden_touched: tuple[str, ...]   # NEVER/GROUND_TRUTH 접근 시도 (정상=())
    event_stream_hash: str               # 시나리오 이벤트열 지문 (재현성)


@dataclass(frozen=True)
class TransitionRecord:
    schema_version: str
    episode_id: str
    decision_index: int
    dt_s: float                          # 경과시간 (SMDP)
    state: GlobalState                    # s_t
    observations: tuple[LocalObservation, ...]   # crane_id 순
    joint_action: JointAction             # a_t
    cost: CostBreakdown
    next_state: GlobalState | None        # terminal 이면 None
    next_observations: tuple[LocalObservation, ...]  # Double DQN target 용 (YR-039)
    terminal: bool
    audit: TransitionAudit
