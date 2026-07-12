"""야드 장치상태(stack) 관리 — 슬롯 점유·blocker·합법 슬롯 탐색.

PoC 적재 규칙(assumed): 빈 바닥 또는 **같은 규격** 컨테이너 위에만 적재.
reefer/DG 구역·중량규칙은 실측 프로파일 확보 후 추가 (02 §8.2).
"""
from __future__ import annotations

from ..domain.enums import ContainerSize
from ..domain.models import BlockGeometry, Container, CraneSpec
from .travel_time import gantry_m, trolley_m


class YardStacks:
    def __init__(self, geom: BlockGeometry, containers: dict[str, Container]):
        self.geom = geom
        self.containers: dict[str, Container] = {}
        self._stacks: dict[tuple[int, int], list[str]] = {}
        for cid in sorted(containers):  # 결정론적 구축
            c = containers[cid]
            self._stacks.setdefault((c.bay, c.row), [])
        # tier 순으로 적재 (validator 가 연속성 보장)
        for (bay, row), pile in self._stacks.items():
            members = [c for c in containers.values() if (c.bay, c.row) == (bay, row)]
            for c in sorted(members, key=lambda x: x.tier):
                pile.append(c.container_id)
                self.containers[c.container_id] = c

    # --- 조회 ---
    def stack(self, bay: int, row: int) -> list[str]:
        return self._stacks.get((bay, row), [])

    def top_tier(self, bay: int, row: int) -> int:
        return len(self.stack(bay, row))

    def blockers_above(self, container_id: str) -> list[str]:
        """대상 위 컨테이너 (위에서부터 제거 순서로)."""
        c = self.containers[container_id]
        pile = self.stack(c.bay, c.row)
        idx = pile.index(container_id)
        return list(reversed(pile[idx + 1:]))

    def stack_size_ok(self, bay: int, row: int, size: ContainerSize) -> bool:
        pile = self.stack(bay, row)
        if not pile:
            return True
        top = self.containers[pile[-1]]
        return top.size == size

    # --- 변형 ---
    def remove(self, container_id: str) -> tuple[int, int, int]:
        """스택 최상단에서 제거. 최상단이 아니면 오류 (blocker 미처리 버그 검출)."""
        c = self.containers[container_id]
        pile = self._stacks[(c.bay, c.row)]
        if not pile or pile[-1] != container_id:
            raise RuntimeError(f"{container_id} 는 최상단이 아님 — blocker 미처리")
        pile.pop()
        slot = (c.bay, c.row, c.tier)
        del self.containers[container_id]
        return slot

    def place(self, container: Container, bay: int, row: int) -> tuple[int, int, int]:
        pile = self._stacks.setdefault((bay, row), [])
        if len(pile) >= self.geom.tier_max:
            raise RuntimeError(f"({bay},{row}) tier 초과 적재 시도")
        if not self.stack_size_ok(bay, row, container.size):
            raise RuntimeError(f"({bay},{row}) 규격 불일치 적재 시도")
        pile.append(container.container_id)
        container.bay, container.row, container.tier = bay, row, len(pile)
        self.containers[container.container_id] = container
        return (bay, row, len(pile))

    # --- 합법 슬롯 탐색 (결정론적) ---
    def find_slot(self, size: ContainerSize, spec: CraneSpec,
                  near_bay: float, near_row: float,
                  exclude: set[tuple[int, int]] = frozenset()) -> tuple[int, int] | None:
        """규격·tier·service range 를 만족하는 최근접 스택.

        비용 = 기준점으로부터의 이동거리 + 적층높이 패널티(낮은 스택 선호).
        동률은 (bay, row) 오름차순 — 항상 같은 입력이면 같은 슬롯.
        """
        best: tuple[float, int, int] | None = None
        for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
            for row in range(1, self.geom.row_count + 1):
                if (bay, row) in exclude:
                    continue
                top = self.top_tier(bay, row)
                if top >= self.geom.tier_max:
                    continue
                if not self.stack_size_ok(bay, row, size):
                    continue
                cost = (gantry_m(self.geom, near_bay, bay)
                        + trolley_m(self.geom, near_row, row)
                        + top * self.geom.tier_height_m)  # 높은 스택 회피(미래 blocker 위험)
                key = (cost, bay, row)
                if best is None or key < best:
                    best = key
        return None if best is None else (best[1], best[2])
