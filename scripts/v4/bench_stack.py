"""[[YR-300]] — v3 원본 vs v4 최적화 실속도. 같은 달을 굴려 벽시계와 결과를 함께 본다."""
from __future__ import annotations
import sys, time
sys.path.insert(0, "src")

SEED, LOADS = 9_100_777, (3_500, 3_500)

def run(gen: str):
    if gen == "v3":
        from yard_rl.v3.stage.month import plan_days
        from yard_rl.v3.stage.month_run import run_month
    else:
        from yard_rl.v4.stage.month import plan_days
        from yard_rl.v4.stage.month_run import run_month
    t = time.perf_counter()
    res = run_month(seed=SEED, arm="RL", n_days=len(LOADS),
                    days=plan_days(SEED, LOADS))
    dt = time.perf_counter() - t
    obs = [r for r in res.live if r.train]
    return dt, sum(r.phi_krw for r in obs), sum(r.traded for r in obs)

print("=" * 62)
print("YR-300 · find_slot 최적화 실속도 (2일 · 부하 3,500)")
print("=" * 62)
got = {}
for gen in ("v3", "v4"):
    dt, phi, tr = run(gen)
    got[gen] = (dt, phi, tr)
    print(f"  {gen}  {dt:>7.1f}초   Φ {phi/1e8:>7.4f}억   거래 {tr:,}")
a, b = got["v3"], got["v4"]
print("=" * 62)
print(f"  속도 {a[0]/b[0]:.2f}배 빨라짐  ({a[0]:.0f}초 → {b[0]:.0f}초)")
print(f"  결과 {'동일 ✅' if abs(a[1]-b[1]) < 1 and a[2] == b[2] else '★다름 — 되돌릴 것'}")
print("=" * 62)
