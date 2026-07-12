"""시나리오 컨테이너 — 시뮬레이터 입력 단위 (한 운영일 episode)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Container, Job


@dataclass
class Scenario:
    scenario_id: str
    seed: int
    horizon_s: float          # 신규 도착 마감 시각 (episode 종료시각)
    drain_window_s: float     # clear-out 허용 구간 (03 §2.2)
    jobs: list[Job]
    containers: dict[str, Container]
    meta: dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.horizon_s + self.drain_window_s
