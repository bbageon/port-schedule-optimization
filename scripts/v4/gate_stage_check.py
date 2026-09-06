"""YR-248 1단계 게이트 — 무대(왕복 400초)가 쓸 만한가. 통과해야 학습으로 간다.

    PYTHONPATH=src python scripts/v4/gate_stage_check.py     # exit 0 = 통과

■ 두 가지만 본다 (기준은 실행 전 고정)
  ① 본선 유휴가 살아났나 — `c_vessel` 비중 ≥ 3%
     v3 는 0.84% 였다. 문헌의 안벽 성능 손실은 10~20% 지만 그건 **시간** 비중이고
     여기 재는 것은 **비용** 비중이라 같은 양이 아니다. 그래서 하한만 둔다:
     3% 를 못 넘으면 배가 여전히 안 굶는 것이고, 크레인이 배울 재료가 없다.
  ② 트럭이 안 무너졌나 — 90분위 체류시간이 v3 대비 2배 미만
     트랙터가 느려지면 트럭도 느려진다. 감당 못 할 수준이면 무대가 트럭 축까지
     망가뜨린 것이라, 본선을 살리려다 전부 잃는다.

■ 왜 v3 와 나란히 재나
  절대값만 보면 "원래 그런 건지 바뀐 건지" 를 못 가른다. 같은 날·같은 시드로
  두 세대를 굴려 **차이**를 본다.
"""
from __future__ import annotations

import statistics as st
import sys

sys.path.insert(0, "src")

LOADS = (3_500, 7_500, 15_000, 3_500)
SEED = 9_100_888
VESSEL_SHARE_MIN = 0.03      # ① 통과선
TURN_RATIO_MAX = 2.0         # ② 통과선


def run(gen: str) -> dict:
    if gen == "v3":
        from yard_rl.v3.stage.month import plan_days
        from yard_rl.v3.stage.month_run import run_month
    else:
        from yard_rl.v4.stage.month import plan_days
        from yard_rl.v4.stage.month_run import run_month
    res = run_month(seed=SEED, arm="NO_REALLOC", n_days=len(LOADS),
                    days=plan_days(SEED, LOADS))
    obs = [r for r in res.live if r.train]
    phi = sum(r.phi_krw for r in obs) or 1.0
    return {"phi": phi,
            "c_vessel": sum(r.c_vessel for r in obs),
            "c_wait": sum(r.c_wait for r in obs),
            "share": sum(r.c_vessel for r in obs) / phi,
            "p90": st.mean([r.p90_turn_time_s for r in obs]) / 3600.0}


def main() -> int:
    print("=" * 70)
    print("YR-248 1단계 게이트 · 야드트랙터 왕복 180s(v3) vs 400s(v4)")
    print("=" * 70)
    g = {k: run(k) for k in ("v3", "v4")}

    print(f"\n  {'항':<14}{'v3 (180s)':>14}{'v4 (400s)':>14}{'변화':>12}")
    for k, lbl in (("phi", "Φ 합계"), ("c_wait", "트럭 대기"),
                   ("c_vessel", "본선 유휴")):
        a, b = g["v3"][k], g["v4"][k]
        print(f"  {lbl:<14}{a / 1e8:>13.2f}억{b / 1e8:>13.2f}억"
              f"{'—' if a == 0 else f'{(b - a) / a:+.1%}':>12}")
    print(f"  {'본선 비중':<14}{g['v3']['share']:>13.2%}{g['v4']['share']:>13.2%}")
    print(f"  {'90분위 체류':<14}{g['v3']['p90']:>13.1f}h{g['v4']['p90']:>13.1f}h"
          f"{g['v4']['p90'] / max(g['v3']['p90'], 1e-9):>11.2f}배")

    ok1 = g["v4"]["share"] >= VESSEL_SHARE_MIN
    ok2 = g["v4"]["p90"] / max(g["v3"]["p90"], 1e-9) < TURN_RATIO_MAX
    print("\n" + "=" * 70)
    print(f"  ① 본선 유휴 ≥ {VESSEL_SHARE_MIN:.0%}   {g['v4']['share']:.2%}  "
          f"{'통과' if ok1 else '✗ 실패 — 배가 여전히 안 굶는다'}")
    print(f"  ② 체류 < {TURN_RATIO_MAX:.0f}배        "
          f"{g['v4']['p90'] / max(g['v3']['p90'], 1e-9):.2f}배  "
          f"{'통과' if ok2 else '✗ 실패 — 트럭 축이 망가졌다'}")
    if ok1 and ok2:
        print("\n판정 — **통과.** 배가 굶기 시작했고 트럭은 견딘다. 학습으로 간다.")
        print("=" * 70)
        return 0
    print("\n판정 — **실패.** 학습을 걸지 않는다.")
    if not ok1:
        print("       400s 로도 부족하다 → 더 조이거나(500s), 안벽 버퍼를 줄인다.")
    if not ok2:
        print("       트럭이 감당 못 한다 → 400s 는 과하다. 300s 부터 다시.")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
