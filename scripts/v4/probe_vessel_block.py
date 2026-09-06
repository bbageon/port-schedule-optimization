"""배가 실제로 막히는가 — [[YR-248]] 게이트가 두 번 빗나가서 직접 잰다.

내 이론(유휴 = (공급간격−STS간격)/공급간격)은 왕복 470s 에 16.4% 를 예측했는데
실측은 1.10% 였다. 이론이 틀렸거나, 막히는 경로가 트랙터가 아니다. 어느 쪽인지
**배 단위 실측**으로 가른다.
"""
import sys
sys.path.insert(0, "src")

from yard_rl.v4.stage.month import plan_days                      # noqa: E402
from yard_rl.v4.stage.month_run import run_month                  # noqa: E402
from yard_rl.v4.world.integrated.profiles import build_h21_profile  # noqa: E402

t = build_h21_profile().transfer
print(f"트랙터 {t.n_units}대 · 왕복 {t.move_time_s:.0f}s → 공급 "
      f"{t.move_time_s / t.n_units:.1f}s   vs   STS 130.9s")
print(f"이론 유휴 {max(0, (t.move_time_s / t.n_units - 130.9) / (t.move_time_s / t.n_units)):.1%}\n")

seen = {}
run_month(seed=9_100_555, arm="NO_REALLOC", n_days=3,
          days=plan_days(9_100_555, (3_500, 15_000, 3_500)),
          on_day=lambda r: seen.__setitem__(r.index, r))

for i, r in sorted(seen.items()):
    if not r.train:
        continue
    print(f"  {i}일 부하 {r.load:>6,} · 배 {r.vessels:>2}척 · "
          f"본선 {r.c_vessel / 1e8:>6.3f}억 / Φ {r.phi_krw / 1e8:>6.2f}억 "
          f"= {r.c_vessel / max(r.phi_krw, 1):>6.2%}")
