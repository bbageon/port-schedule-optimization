"""[[YR-300]] — v3 원본과 v4 최적화판이 **같은 슬롯·같은 비트**를 내는지 직접 대조.

    PYTHONPATH=src python scripts/v5/verify_find_slot.py

같은 야드 상태를 두 구현에 똑같이 넣고 `find_slot` 을 수만 번 부른다.
슬롯이 하나라도 다르면 최적화가 결과를 바꾼 것이므로 즉시 되돌려야 한다.
"""
from __future__ import annotations

import random
import struct
import sys

sys.path.insert(0, "src")

from yard_rl.v3.world.domain.enums import ContainerSize, LoadStatus   # noqa: E402
from yard_rl.v3.world.domain.models import Container as C3            # noqa: E402
from yard_rl.v3.world.integrated.profiles import build_h21_profile as p3  # noqa: E402
from yard_rl.v3.world.sim.stack import YardStacks as S3               # noqa: E402
from yard_rl.v5.world.domain.models import Container as C4            # noqa: E402
from yard_rl.v5.world.integrated.profiles import build_h21_profile as p4  # noqa: E402
from yard_rl.v5.world.sim.stack import YardStacks as S4               # noqa: E402


def build(seed: int, fill: float):
    """같은 씨앗으로 v3·v4 야드를 하나씩 — 내용이 같아야 대조가 성립한다."""
    prof = p3()
    g = prof.block
    rng = random.Random(seed)
    piles: dict = {}
    target = int(g.bay_count * g.row_count * g.tier_max * fill)
    made = 0
    rows = []
    for _ in range(target * 4):
        if made >= target:
            break
        bay, row = rng.randint(1, g.bay_count), rng.randint(1, g.row_count)
        size = rng.choice(list(ContainerSize))
        pile = piles.setdefault((bay, row), [])
        if len(pile) >= g.tier_max or (pile and pile[-1][1] != size):
            continue
        made += 1
        pile.append((f"C{made:05d}", size, bay, row, len(pile) + 1))
        rows.append(pile[-1])
    mk = lambda K: {r[0]: K(container_id=r[0], size=r[1], load_status=LoadStatus.FULL,
                            block="B01", bay=r[2], row=r[3], tier=r[4]) for r in rows}
    return prof, S3(g, mk(C3)), S4(p4().block, mk(C4))


def main() -> int:
    bits = lambda x: struct.pack("<d", x)
    total = diff = 0
    print("=" * 62)
    print("YR-300 · v3 원본 vs v4 최적화 — find_slot 결과 대조")
    print("=" * 62)
    for fill in (0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 0.95):
        prof, a, b = build(seed=int(fill * 1000) + 3, fill=fill)
        spec = prof.cranes[0]
        rng = random.Random(int(fill * 777))
        n = m = 0
        for _ in range(3000):
            size = rng.choice(list(ContainerSize))
            nb, nr = rng.uniform(-2, 27), rng.uniform(-2, 13)
            ex = frozenset((rng.randint(1, 24), rng.randint(1, 10))
                           for _ in range(rng.randint(0, 5)))
            ra, rb = a.find_slot(size, spec, nb, nr, ex), b.find_slot(size, spec, nb, nr, ex)
            n += 1
            if ra != rb:
                m += 1
                if m == 1:
                    print(f"  ★불일치 fill={fill} size={size} near=({nb:.3f},{nr:.3f})")
                    print(f"     v3 {ra}  vs  v4 {rb}")
        total += n
        diff += m
        print(f"  적재율 {fill:>4.0%}  호출 {n:,}  불일치 {m}")
    print("=" * 62)
    print(f"총 {total:,} 호출 · 불일치 {diff}건")
    print("판정 — " + ("**동일. 최적화가 결과를 바꾸지 않는다.**" if diff == 0
                      else "★다르다. 되돌려야 한다."))
    print("=" * 62)
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
