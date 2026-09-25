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

    def rehandle_capacity_ok(self, target_id: str, spec: CraneSpec) -> bool:
        """재조작 슬롯 존재 보장 (1차 차단용 — 만재 야드 크래시 방지).

        스택은 항상 단일 규격(place 가 강제)이므로 blocker 전부 같은 규격 s.
        s 를 받을 수 있는 스택(빈 바닥 또는 top==s, tier 여유)의 잔여용량 합이
        blocker 수 이상이면 순차 배치가 항상 성공한다 (배치가 호환성을 보존).
        """
        blockers = self.blockers_above(target_id)
        if not blockers:
            return True
        c = self.containers[target_id]
        size = self.containers[blockers[0]].size
        src = (c.bay, c.row)
        capacity = 0
        for bay in range(spec.service_bay_min, spec.service_bay_max + 1):
            for row in range(1, self.geom.row_count + 1):
                if (bay, row) == src:
                    continue
                top = self.top_tier(bay, row)
                if top >= self.geom.tier_max:
                    continue
                if top > 0 and not self.stack_size_ok(bay, row, size):
                    continue
                capacity += self.geom.tier_max - top
                if capacity >= len(blockers):
                    return True
        return False

    # --- 합법 슬롯 탐색 (결정론적) ---
    def find_slot(self, size: ContainerSize, spec: CraneSpec,
                  near_bay: float, near_row: float,
                  exclude: set[tuple[int, int]] = frozenset()) -> tuple[int, int] | None:
        """규격·tier·service range 를 만족하는 최근접 스택.

        비용 = 기준점으로부터의 이동거리 + 적층높이 패널티(낮은 스택 선호).
        동률은 (bay, row) 오름차순 — 항상 같은 입력이면 같은 슬롯.

        ■ 왜 손으로 편 모양인가 ([[YR-300]] · 2026-09-07 · **v4 전용**)
          이 함수가 시뮬레이션 시간의 **84%** 를 먹는다 (2일치 프로파일 누적 775초/925초).
          호출 280만 회 × 칸 240개라, 칸마다 `top_tier()`·`stack_size_ok()` 를 부르면
          그 안에서 `stack()` 이 또 불려 **`stack()` 만 12억 회**가 된다.

          그래서 셋을 폈다 — **계산식은 한 글자도 안 바뀐다.**
            ① 칸마다 `_stacks` 를 **한 번만** 꺼내 top·규격 양쪽에 쓴다
               (원래는 `top_tier` 가 한 번, `stack_size_ok` 가 또 한 번 꺼냈다)
            ② `gantry` 는 bay 에만 의존하므로 **row 루프 밖**으로 뺐다 (10분의 1)
            ③ `geom` 속성·딕셔너리를 지역변수로 받는다 (frozen dataclass 라 안전)
            ④ **가까운 bay 부터 보다가, 거리 하한이 이미 찾은 최선을 넘으면 멈춘다**
               (240칸 전수 → 실측 41.7칸 · 17.4%)

          부동소수도 **비트 단위로 같다** — 피연산자·결합순서를 안 바꿨다
          (`(A+B)+C` 유지). 그래서 동률 tie-break 가 갈릴 여지가 없다.
          실측 **2.95배** (281.0 → 95.4초 · 2일치 전체 런 · 결과 digest 동일).
          ①②③만으로 2.28배, ④ 를 더해 2.95배다.

        ■ ⚠️ `stack_size_ok` 와 **같은 규칙이 두 곳에 생겼다**
          나중에 reefer·위험물·중량 규칙이 `stack_size_ok` 에 붙으면 여기가 조용히
          뒤처진다. 신규 적재(STORE) 경로에는 후조건 가드가 없어 틀린 슬롯이 그대로
          나간다. `tests/v5/test_find_slot_equiv.py` 가 두 구현의 일치를 지킨다.
        """
        stacks, conts, geom = self._stacks, self.containers, self.geom
        tier_max, tier_h = geom.tier_max, geom.tier_height_m
        bay_len, row_w = geom.bay_length_m, geom.row_width_m
        rows = range(1, geom.row_count + 1)
        best: tuple[float, int, int] | None = None
        #: ④ **가까운 bay 부터** 본다 — 아래 조기 종료의 전제다.
        #:  동률(같은 거리의 좌우 bay)은 bay 번호 오름차순으로 깨서 결정론을 지킨다.
        near_first = sorted(range(spec.service_bay_min, spec.service_bay_max + 1),
                            key=lambda b: (abs(near_bay - b), b))
        for bay in near_first:
            g = abs(near_bay - bay) * bay_len          # ② row 와 무관 — 밖으로
            #: ④ 조기 종료 — 남은 bay 는 **전부 열등**하므로 안 본다.
            #:  비용 = g + trolley + top·tier_h 이고 **뒤 두 항이 항상 0 이상**이라
            #:  g 는 그 bay 가 낼 수 있는 비용의 하한이다. 하한이 이미 찾은 최선보다
            #:  크면 그 bay 도, 더 먼 bay 도 최선을 못 넘는다.
            #:  ⚠️ 이 논리는 **뒤 두 항이 비음수**일 때만 성립한다 — 비용식에 음수 항을
            #:     더하면 여기를 먼저 고쳐야 한다.
            #:  동률 안전: 잘리는 후보는 비용이 **엄격히** 크므로(> best[0]) 사전식
            #:  tie-break 와 무관하다. 실측 21,000 대조에서 불일치 0.
            if best is not None and g > best[0]:
                break
            for row in rows:
                if (bay, row) in exclude:
                    continue
                pile = stacks.get((bay, row))          # ① 한 번만 꺼낸다
                top = len(pile) if pile else 0
                # ★tier 검사를 `if pile` **밖**에 둔다 — 원본과 완전 동치.
                #   안으로 넣으면 tier_max <= 0 에서 빈 칸 처리가 갈린다.
                if top >= tier_max:
                    continue
                if pile and conts[pile[-1]].size != size:   # == not stack_size_ok(...)
                    continue
                cost = g + abs(near_row - row) * row_w + top * tier_h
                key = (cost, bay, row)
                if best is None or key < best:
                    best = key
        return None if best is None else (best[1], best[2])
