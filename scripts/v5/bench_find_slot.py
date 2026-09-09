"""[[YR-300]] — `find_slot` 자체 속도. v3 원본 vs v4 최적화, 같은 야드에서."""
from __future__ import annotations
import random, sys, time
sys.path.insert(0, "src")
from yard_rl.v3.world.domain.enums import ContainerSize, LoadStatus
from yard_rl.v3.world.domain.models import Container as C3
from yard_rl.v3.world.integrated.profiles import build_h21_profile as p3
from yard_rl.v3.world.sim.stack import YardStacks as S3
from yard_rl.v5.world.domain.models import Container as C4
from yard_rl.v5.world.integrated.profiles import build_h21_profile as p4
from yard_rl.v5.world.sim.stack import YardStacks as S4

def build(seed, fill):
    g = p3().block; rng = random.Random(seed)
    piles, rows, made = {}, [], 0
    target = int(g.bay_count * g.row_count * g.tier_max * fill)
    for _ in range(target * 4):
        if made >= target: break
        b, r = rng.randint(1, g.bay_count), rng.randint(1, g.row_count)
        sz = rng.choice(list(ContainerSize)); pile = piles.setdefault((b, r), [])
        if len(pile) >= g.tier_max or (pile and pile[-1][1] != sz): continue
        made += 1; pile.append((f"C{made:05d}", sz, b, r, len(pile) + 1)); rows.append(pile[-1])
    mk = lambda K: {x[0]: K(container_id=x[0], size=x[1], load_status=LoadStatus.FULL,
                            block="B01", bay=x[2], row=x[3], tier=x[4]) for x in rows}
    return S3(g, mk(C3)), S4(p4().block, mk(C4))

spec = p3().cranes[0]
N = 20_000
print("=" * 60)
print(f"YR-300 · find_slot 속도 (호출 {N:,}회 · 같은 야드)")
print("=" * 60)
print(f"  {'적재율':>7}{'v3 원본':>12}{'v4 최적화':>12}{'배수':>8}")
tot3 = tot4 = 0.0
for fill in (0.15, 0.45, 0.75):
    a, b = build(int(fill * 100) + 5, fill)
    rng = random.Random(42)
    args = [(rng.choice(list(ContainerSize)), rng.uniform(0, 25), rng.uniform(0, 11))
            for _ in range(N)]
    for stk in (a, b):                       # 워밍업
        for s, nb, nr in args[:200]: stk.find_slot(s, spec, nb, nr)
    t = time.perf_counter()
    for s, nb, nr in args: a.find_slot(s, spec, nb, nr)
    d3 = time.perf_counter() - t
    t = time.perf_counter()
    for s, nb, nr in args: b.find_slot(s, spec, nb, nr)
    d4 = time.perf_counter() - t
    tot3 += d3; tot4 += d4
    print(f"  {fill:>6.0%}{d3:>11.2f}초{d4:>11.2f}초{d3/d4:>7.2f}배")
print("=" * 60)
print(f"  합계 {tot3:.2f}초 → {tot4:.2f}초   **{tot3/tot4:.2f}배**")
print(f"  find_slot 이 시뮬레이션의 84% 이므로 전체 기대 "
      f"{1/(0.16 + 0.84/(tot3/tot4)):.2f}배")
print("=" * 60)
