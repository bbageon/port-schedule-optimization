"""연결 레인 네트워크 — 혼잡·간섭 적분 (YR-036, 최종전략 §7.6).

배타 점유(lock)는 ReservationTable 이 authoritative. 여기서는 연결 그래프와 혼잡/대기의
시간적분만 담당한다(경계 분리). LaneGraph.edges 는 무방향 '연결' — 인접 레인이 서로 영향.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.state import LaneGraph


@dataclass
class LaneNetwork:
    graph: LaneGraph
    cong_area_s: float = 0.0
    _adj: dict[str, frozenset] = field(default_factory=dict)

    def __post_init__(self):
        adj: dict[str, set] = {lid: set() for lid in self.graph.lane_ids}
        for a, b in self.graph.edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        self._adj = {k: frozenset(v) for k, v in adj.items()}

    def lane_ids(self) -> tuple[str, ...]:
        return self.graph.lane_ids

    def neighbors(self, lane_id: str) -> frozenset:
        return self._adj.get(lane_id, frozenset())

    def conflict_set(self, lane_id: str) -> frozenset:
        """자기 + 연결 인접 (비통과 영향 범위)."""
        return frozenset({lane_id}) | self.neighbors(lane_id)

    def occupancy(self, occupied: frozenset) -> tuple[float, float]:
        """점유 레인 집합 → (평균 혼잡률, 최대 혼잡률). 인접 점유가 혼잡을 가중.

        각 레인 혼잡 = (자신 점유 + 연결 인접 점유 수) / (1 + 연결차수), [0,1] 정규화.
        """
        ids = self.graph.lane_ids
        if not ids:
            return 0.0, 0.0
        vals = []
        for lid in ids:
            deg = len(self._adj.get(lid, frozenset()))
            load = (1.0 if lid in occupied else 0.0) + sum(
                1.0 for n in self._adj.get(lid, frozenset()) if n in occupied)
            vals.append(load / (1.0 + deg))
        return sum(vals) / len(vals), max(vals)

    def integrate(self, t0: float, t1: float, occupied: frozenset) -> None:
        dt = t1 - t0
        if dt > 0:
            mean, _ = self.occupancy(occupied)
            self.cong_area_s += mean * dt
