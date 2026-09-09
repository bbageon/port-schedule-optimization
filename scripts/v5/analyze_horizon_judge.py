"""YR-299 B 판정 — 창 3h/6h 정책을 **비용으로** 겨룬다 (사용자 지적 2026-09-09).

검증 손실은 *"라벨을 잘 맞추나"* 이고, 알고 싶은 것은 *"비용이 줄었나"* 다.
학습 로그의 Φ 는 탐색이 섞여 못 쓰므로, 탐색 0 으로 **안 배운 달**에서 다시 굴렸다.
"""
from __future__ import annotations
import json, pathlib
from math import comb

R = pathlib.Path("outputs/v5/horizon-judge")


def sp(d):
    n = sum(1 for x in d if abs(x) > 1e-9)
    k = sum(1 for x in d if x > 1e-9)
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0


def load(tag):
    A = {}
    for f in (R / tag / "arms").glob("arm_*.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        A[j["arm"]] = ({int(k): v for k, v in j["phi_by_day"].items()}, j)
    return A


print("=" * 72)
print("YR-299 B 판정 · 창 3h vs 6h — **비용** (시드 9,900,997 · 안 배운 달 · 탐색 0)")
print("=" * 72)

got = {}
for tag in ("h3", "h6"):
    A = load(tag)
    D = [i for i in sorted(A["RL"][0]) if 1 <= i <= 28]
    B = sum(A["NO_REALLOC"][0][i] for i in D)
    Rl = sum(A["RL"][0][i] for i in D)
    d = [A["NO_REALLOC"][0][i] - A["RL"][0][i] for i in D]
    got[tag] = dict(days=D, base=B, rl=Rl, cut=(B - Rl) / B,
                    win=sum(1 for x in d if x > 0), p=sp(d),
                    phi=A["RL"][0], j=A["RL"][1])

print(f"\n  {'창':>6}{'재배치없음':>13}{'학습정책':>12}{'감소':>9}{'낮은날':>9}{'부호p':>9}")
for t, lbl in (("h3", "3시간"), ("h6", "6시간")):
    g = got[t]
    print(f"  {lbl:>6}{g['base']/1e8:>12.2f}억{g['rl']/1e8:>11.2f}억"
          f"{g['cut']:>9.2%}{g['win']:>7}/28{g['p']:>9.4f}")

# ★핵심 — 두 정책 직접 짝비교
a, b = got["h3"], got["h6"]
dd = [b["phi"][i] - a["phi"][i] for i in a["days"]]        # >0 이면 3시간이 쌌다
w3 = sum(1 for x in dd if x > 1e-9)
print(f"\n  ★직접 짝비교 (3시간 vs 6시간)")
print(f"     3시간이 싼 날 {w3}/28  ·  부호검정 p={sp(dd):.4f}")
print(f"     비용 차이 합계 {sum(dd)/1e8:+.2f}억  (양수 = 3시간이 그만큼 쌈)")
print(f"     거래 3시간 {a['j']['traded']:,} / 6시간 {b['j']['traded']:,}"
      f"  ·  공간 {a['j']['n_space']:,} / {b['j']['n_space']:,}"
      f"  ·  시간 {a['j']['n_time']:,} / {b['j']['n_time']:,}")

print("\n" + "=" * 72)
p = sp(dd)
if p < 0.05 and w3 > 14:
    print("판정 — **3시간이 유의하게 싸다.** 6시간 창은 성능도 나쁘다.")
elif p < 0.05:
    print("판정 — **6시간이 유의하게 싸다.** 검증 손실과 반대다 — 손실은 수단일 뿐이다.")
else:
    print(f"판정 — **비용 차이가 유의하지 않다** (p={p:.3f}).")
    print("       검증 손실은 6.8배 나빴지만 **결정은 비슷하다** — 순서만 맞으면")
    print("       예측이 부정확해도 같은 후보를 고르기 때문이다.")
print("=" * 72)
