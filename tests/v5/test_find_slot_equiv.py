"""`find_slot` 의 인라인 사본이 `stack_size_ok` 와 어긋나지 않게 지킨다 ([[YR-300]]).

■ 왜 이 시험이 필요한가
  `find_slot` 은 성능 때문에 `top_tier`·`stack_size_ok` 를 **손으로 펴 넣었다**
  (시뮬레이션 시간의 84% 를 먹던 자리 · 2.28배 단축). 그 대가로 적재 규칙이
  **두 곳**에 있게 됐다.

  나중에 reefer·위험물·중량 규칙이 `stack_size_ok` 에 붙으면 `find_slot` 안의 사본이
  조용히 뒤처진다. 그런데 신규 적재(STORE) 경로에는 후조건 가드가 없어
  **틀린 슬롯이 그대로 나간다.** 그래서 여기서 둘을 같은 입력으로 대조한다 —
  규칙이 갈리는 순간 이 시험이 먼저 깨진다.
"""
from __future__ import annotations

import random

import pytest

from yard_rl.v5.world.domain.enums import ContainerSize, LoadStatus
from yard_rl.v5.world.domain.models import Container
from yard_rl.v5.world.integrated.profiles import build_h21_profile
from yard_rl.v5.world.sim.stack import YardStacks


def _yard(seed: int, fill: float) -> tuple:
    """무작위로 채운 야드. **적재 규칙을 지켜** 쌓는다(같은 규격 위에만·tier 연속)."""
    prof = build_h21_profile()
    geom = prof.block
    rng = random.Random(seed)
    piles: dict[tuple[int, int], list[Container]] = {}
    target = int(geom.bay_count * geom.row_count * geom.tier_max * fill)
    made = 0
    for i in range(target * 3):          # 규칙에 막혀 못 쌓는 몫을 감안해 넉넉히 시도
        if made >= target:
            break
        bay = rng.randint(1, geom.bay_count)
        row = rng.randint(1, geom.row_count)
        size = rng.choice(list(ContainerSize))
        pile = piles.setdefault((bay, row), [])
        if len(pile) >= geom.tier_max:
            continue
        if pile and pile[-1].size != size:      # 같은 규격 위에만
            continue
        made += 1
        pile.append(Container(container_id=f"C{made:05d}", size=size,
                              load_status=LoadStatus.FULL, block="B01",
                              bay=bay, row=row, tier=len(pile) + 1))
    conts = {c.container_id: c for pile in piles.values() for c in pile}
    return prof, YardStacks(geom, conts)


@pytest.mark.parametrize("fill", [0.0, 0.30, 0.45, 0.75, 0.95])
def test_inline_matches_stack_size_ok(fill):
    """★규칙이 갈리면 여기서 잡는다 — 모든 칸에서 두 판정이 같아야 한다."""
    prof, stk = _yard(seed=7 + int(fill * 100), fill=fill)
    geom = prof.block
    for size in list(ContainerSize):
        for bay in range(1, geom.bay_count + 1):
            for row in range(1, geom.row_count + 1):
                pile = stk._stacks.get((bay, row))
                inline = not (pile and stk.containers[pile[-1]].size != size)
                assert inline == stk.stack_size_ok(bay, row, size), (
                    f"적재 규칙 불일치 — bay {bay} row {row} size {size}: "
                    f"find_slot 인라인 {inline} vs stack_size_ok "
                    f"{stk.stack_size_ok(bay, row, size)}")


@pytest.mark.parametrize("fill", [0.30, 0.45, 0.75])
def test_top_matches_top_tier(fill):
    """`len(pile) if pile else 0` 이 `top_tier` 와 같은가."""
    prof, stk = _yard(seed=11 + int(fill * 100), fill=fill)
    geom = prof.block
    for bay in range(1, geom.bay_count + 1):
        for row in range(1, geom.row_count + 1):
            pile = stk._stacks.get((bay, row))
            assert (len(pile) if pile else 0) == stk.top_tier(bay, row)


def test_find_slot_is_deterministic():
    """같은 입력이면 같은 슬롯 — 최적화가 순회 순서를 바꾸지 않았는지."""
    prof, stk = _yard(seed=99, fill=0.45)
    spec = prof.cranes[0]
    rng = random.Random(5)
    for _ in range(300):
        size = rng.choice(list(ContainerSize))
        nb, nr = rng.uniform(0, 25), rng.uniform(0, 11)
        assert stk.find_slot(size, spec, nb, nr) == stk.find_slot(size, spec, nb, nr)
