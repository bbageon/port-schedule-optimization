"""다중 YC 런타임 상태 (YR-036).

domain.CraneState 를 래핑해 down/yielded 등 통합 전용 상태를 더한다. 순회는 항상
crane_id 정렬 — set/dict 반복순서 의존 금지(결정론).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.enums import CraneStatus
from ..domain.models import CraneSpec, CraneState


@dataclass
class YcRuntime:
    state: CraneState
    down: bool = False
    down_pending: bool = False        # 작업 중 EquipmentDown → 완료 후 DOWN (비선점)
    yielded: bool = False             # WAIT 후 다음 상태변경까지 결정 제외 (무한루프 방지)
    recent_yield_count: int = 0       # 경합 패배(LOST_CONTENTION) 양보 누적 (COORD, YR-056)
    recent_completions: int = 0
    served_count: int = 0             # 부하 불균형 산출
    recent_empty_travel_s: float = 0.0
    last_move_dir: float = 0.0        # -1/0/1
    is_loaded: bool = False

    @property
    def crane_id(self) -> str:
        return self.state.crane_id

    @property
    def idle(self) -> bool:
        return self.state.assigned_job is None and not self.down


def make_runtime(spec: CraneSpec, transfer_row: float) -> YcRuntime:
    st = CraneState(
        crane_id=spec.crane_id, position_bay=float(spec.service_bay_min),
        trolley_row=float(transfer_row), service_bay_min=spec.service_bay_min,
        service_bay_max=spec.service_bay_max, status=CraneStatus.IDLE)
    return YcRuntime(state=st)


@dataclass
class CraneFleet:
    _cranes: dict[str, YcRuntime] = field(default_factory=dict)
    specs: dict[str, CraneSpec] = field(default_factory=dict)

    def add(self, spec: CraneSpec, transfer_row: float) -> None:
        self._cranes[spec.crane_id] = make_runtime(spec, transfer_row)
        self.specs[spec.crane_id] = spec

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cranes))

    def get(self, crane_id: str) -> YcRuntime:
        return self._cranes[crane_id]

    def spec(self, crane_id: str) -> CraneSpec:
        return self.specs[crane_id]

    def all(self) -> list[YcRuntime]:
        return [self._cranes[c] for c in self.ids()]

    def __len__(self) -> int:
        return len(self._cranes)
