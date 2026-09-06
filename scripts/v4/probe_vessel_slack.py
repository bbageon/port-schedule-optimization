"""YR-248 0단계 — 본선 일정 여유(`slack_s`)가 죽은 칸인 **이유**.

    PYTHONPATH=src python scripts/v4/probe_vessel_slack.py

■ 결론부터: 실험이 필요 없다. 대수로 증명된다.

  계획완료      pc      = start + M·c·d              (`scenario_gen.py:396`)
  남은작업      rem(t)  = (M − done)·c               (`vessel.py:57`)
  일정 여유     slack(t)= pc − t − rem(t)            (`vessel.py:64`)

  계획대로 진행 중이면 진척 f 에서 t = start + f·M·c 이고 done = f·M 이므로

      slack = (start + M·c·d) − (start + f·M·c) − (1−f)·M·c
            = M·c·(d − f − 1 + f)
            = **M·c·(d − 1)**        ← f 가 소거된다

  **진척과 무관한 상수다.** 배가 얼마나 처리됐든, 지금이 몇 시든 같은 값이다.
  `d = vessel_deadline_mult = 2.0` 이므로 slack = M·c = **그 배의 총 작업시간 전체**.

■ 그래서 97.5% 상한 포화가 설명된다
  특징은 `max(-2, min(2, slack/3600))` 로 ±2h 에 자른다(`features/block.py:152`).
  배 하나의 총 작업시간이 2시간을 넘으면 무조건 상한이다 — 선급 3종 전부 그렇다.

■ 무엇이 이 상수를 깨나 (= 실제로 정보가 생기는 유일한 경로)
  위 유도는 **계획대로 진행될 때**다. 실제로 배가 밀리면 t 가 계획보다 앞서 흐르고
  slack 이 줄어든다. 즉 이 칸은 *"이 배가 급한가"* 가 아니라 **"이 배가 계획보다
  얼마나 밀렸나"** 를 재고 있다. 그런데 여유가 총 작업시간만큼(=100%) 깔려 있어서,
  **작업이 두 배로 늘어져야 비로소 0 에 닿는다.** 그 전까지는 상한 뒤에 숨는다.

■ 처방 — 둘 다 필요하다
  ① 클램프 ±2h 는 이 무대에서 무의미하다. 하지만 넓히기만 하면 *"총 작업시간"* 이라는
     **배 크기 상수**가 특징으로 들어간다 — 급함이 아니라 크기를 읽게 된다.
  ② 진짜 처방은 `d` 를 1 에 가깝게 내리는 것이다. `scenario_gen.py:194` 에 이미
     `vessel_deadline_mult: 1.15` 프리셋이 있다 — slack = M·c·0.15 로 15% 여유가 되어
     밀림이 곧바로 신호가 된다.

  ⚠️ `d` 를 바꾸면 **무대가 바뀐다** — v3 판정과 비교 불가. v4 전용 변경으로 못 박고,
     바꾼 뒤 본선 유휴·Φ 구성이 어떻게 달라지는지 함께 보고해야 한다.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from yard_rl.v4.world.integrated.scenario_gen import (            # noqa: E402
    TerminalGenParams)
from yard_rl.v4.stage.vessels import STREAM_MOVES_PER_H           # noqa: E402

CLAMP_H = 2.0


def main() -> int:
    d = TerminalGenParams().vessel_deadline_mult
    cad_h = 1.0 / STREAM_MOVES_PER_H          # move 하나에 걸리는 시간(시간)

    print("=" * 72)
    print("YR-248 0단계 · 본선 일정 여유는 왜 죽은 칸인가 — 대수 증명")
    print("=" * 72)
    print(f"""
  pc    = start + M·c·d          계획완료      (scenario_gen.py:396)
  rem   = (M − done)·c           남은작업      (vessel.py:57)
  slack = pc − t − rem                         (vessel.py:64)

  계획대로면 t = start + f·M·c, done = f·M 이므로
      slack = M·c·(d − f − 1 + f) = M·c·(d − 1)      ← f 가 소거된다

  현재 d = {d} → slack = M·c·{d - 1:.2f} = 그 배의 총 작업시간의 {(d - 1) * 100:.0f}%
  → **진척·시각과 무관한 상수.** 배마다 하나의 값만 갖는다.
""")
    print(f"{'선급 규모(moves)':>18}{'총 작업':>10}{'slack':>9}"
          f"{'±2h 클램프 후':>14}   정보")
    for moves in (120, 240, 360, 480, 720):
        work_h = moves * cad_h
        slack_h = work_h * (d - 1)
        clamped = max(-CLAMP_H, min(CLAMP_H, slack_h))
        info = "없음 (상한)" if abs(clamped) >= CLAMP_H else "있음"
        print(f"{moves:>18,}{work_h:>9.1f}h{slack_h:>8.1f}h"
              f"{clamped:>13.1f}h   {info}")

    print(f"""
{"=" * 72}
판정 — **원값이 평평하다.** 클램프를 넓혀도 정보가 안 생긴다:
       그 자리엔 '급함'이 아니라 **배 크기 상수**가 들어온다.

처방 — `vessel_deadline_mult` {d} → 1.15 (이미 있는 프리셋, scenario_gen.py:194).
       그러면 slack = M·c·0.15 라 계획보다 15%만 밀려도 신호가 살아난다.
       ⚠️ 무대가 바뀌므로 **v4 전용**으로 못 박고, 본선 유휴·Φ 구성 변화를 함께 본다.
{"=" * 72}""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
