"""통합 터미널 시나리오 — 한 운영구간 episode 입력 (YR-036).

외부트럭·본선·컨테이너에 더해 injected_events(장애·계획변경)를 담는다. RNG 는 시나리오
생성에만 존재하고 엔진은 이 결정론적 입력만 소비한다 (동일 시나리오 → 동일 trace).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.models import Container, Job
from .vessel import VesselProcess


@dataclass(frozen=True)
class InjectedEvent:
    """결정론적 외란 — 엔진이 시각순으로 소비 (장애·계획변경)."""

    time: float
    kind: str                     # "EQUIPMENT_DOWN" | "EQUIPMENT_UP" | "PLAN_CHANGE"
    target: str                   # crane_id | vessel_id
    data: tuple = ()              # PLAN_CHANGE: (("planned_completion_s", 값), ...)


@dataclass
class TerminalScenario:
    scenario_id: str
    seed: int
    horizon_s: float
    drain_window_s: float
    containers: dict[str, Container]
    jobs: list[Job]                       # 외부트럭·본선연계·내부작업
    vessels: list[VesselProcess]
    injected_events: list[InjectedEvent] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.horizon_s + self.drain_window_s
