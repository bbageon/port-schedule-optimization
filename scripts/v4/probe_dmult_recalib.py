"""YR-248 1단계 — 본선 여유 배수 재보정. 신호가 살아나는가, 무대가 망가지지 않는가.

    PYTHONPATH=src python scripts/v4/probe_dmult_recalib.py [일수]

■ 무엇을 바꾸나
  `vessel_deadline_mult` 2.0 → 1.15. 0단계가 증명했듯 `slack = M·c·(d−1)` 이므로
  d=2.0 이면 slack 이 **배 크기 상수**이고(총 작업시간 전체가 여유), d=1.15 면
  총 작업시간의 15%만 여유라 **계획보다 조금만 밀려도 신호가 산다.**

■ 왜 "무대가 망가지지 않는가" 를 함께 보나
  마감을 조이면 배가 더 자주 늦고, 늦으면 `c_vessel`(본선 지연 비용)이 오른다.
  그 자체는 의도한 것이다 — 지금은 본선 유휴가 Φ의 1.21%뿐이라 정책이 신경 쓸
  이유가 없다. 하지만 **너무 조이면** 어떤 정책으로도 못 지키는 마감이 되어
  `c_vessel` 이 Φ를 삼키고, 그러면 판정이 본선 지연 하나에 휘둘린다
  ([[YR-109]] 가 겪은 *구조적 최소 선석초과* 문제와 같은 함정).

  그래서 셋을 함께 본다:
    ① slack 이 살아났나        — 상한 포화율·분산
    ② 본선 지연이 얼마나 늘었나 — `c_vessel` 절대액과 Φ 비중
    ③ 나머지가 멀쩡한가        — 트럭 대기·재취급이 안 흔들렸나

■ 판정 기준 (실행 전 고정)
  통과   slack 상한 포화 < 60% **그리고** `c_vessel` 비중 < 15%
  재검토 포화는 풀렸는데 비중 ≥ 15% → d 를 1.15~2.0 사이에서 다시 고른다
  실패   포화가 안 풀림 → d 말고 다른 원인 (계획 식 자체를 다시 본다)

■ 정책은 안 붙인다 (`arm="NO_REALLOC"`)
  무대의 성질을 보는 것이지 성능을 보는 게 아니다. 재배치가 있든 없든 배의
  계획완료는 같고, 여기서 재는 것은 **신호의 분포와 비용 구성**이다.
"""
from __future__ import annotations

import statistics as st
import sys

sys.path.insert(0, "src")

from yard_rl.v4.stage.month import plan_days                      # noqa: E402
from yard_rl.v4.stage.month_run import run_month                  # noqa: E402

CLAMP_H = 2.0
CADENCE_S = 3600.0 / 27.5      # STREAM_MOVES_PER_H
#: 무대 성질을 보는 것이라 짧게 — 앞뒤 연결용 하루씩 + 가운데가 관측일
LOADS = (3_500, 7_500, 15_000, 3_500)
SEED = 9_100_777


def slack_saturation(d: float) -> float:
    """계획 궤적의 상한 포화율 — 0단계 유도(`slack = M·c·(d−1)`)를 그대로 쓴다."""
    from yard_rl.v4.stage.month import plan_month_vessels
    from yard_rl.v4.world.integrated.terminal_stream import OBS_24H
    from yard_rl.v4.world.integrated.yard_layout import terminal_layout
    days = plan_days(SEED, LOADS)
    rows = [r for rs in plan_month_vessels(days, terminal_layout(),
                                           obs=OBS_24H).values() for r in rs]
    sl = [float(r["moves"]) * CADENCE_S * (d - 1.0) / 3600.0 for r in rows
          if r.get("moves")]
    if not sl:
        return float("nan")
    return sum(1 for v in sl if abs(v) >= CLAMP_H) / len(sl), sl


def run(d: float) -> dict:
    res = run_month(seed=SEED, arm="NO_REALLOC", days=plan_days(SEED, LOADS),
                    n_days=len(LOADS), vessel_deadline_mult=d)
    obs = [r for r in res.live if r.train]
    tot = sum(r.phi_krw for r in obs) or 1.0
    return {
        "phi": tot,
        "c_wait": sum(r.c_wait for r in obs),
        "c_vessel": sum(r.c_vessel for r in obs),
        "c_rehandle": sum(r.c_rehandle for r in obs),
        "c_move": sum(r.c_move for r in obs),
        "p90": st.mean([r.p90_turn_time_s for r in obs]) / 3600.0,
    }


def main() -> int:
    print("=" * 74)
    print("YR-248 1단계 · 본선 여유 배수 재보정 (2.0 → 1.15)")
    print("=" * 74)

    print("\n■ ① 신호가 살아나는가 — slack 상한(±2h) 포화율\n")
    print(f"  {'d':>6}{'포화율':>10}{'중앙 slack':>13}{'분산 있나':>12}")
    sat = {}
    for d in (2.0, 1.15):
        s, sl = slack_saturation(d)
        sat[d] = s
        spread = "있음" if st.pstdev(sl) > 0.3 else "없음"
        print(f"  {d:>6.2f}{s:>9.1%}{st.median(sl):>12.1f}h{spread:>12}")

    print("\n■ ② 무대가 망가지지 않는가 — 비용 구성 (관측일 합계)\n")
    got = {d: run(d) for d in (2.0, 1.15)}
    print(f"  {'항':<12}{'d=2.0':>14}{'d=1.15':>14}{'변화':>12}")
    for k, lbl in (("phi", "Φ 합계"), ("c_wait", "트럭 대기"),
                   ("c_vessel", "본선 지연"), ("c_rehandle", "재취급"),
                   ("c_move", "크레인 이동")):
        a, b = got[2.0][k], got[1.15][k]
        chg = "—" if a == 0 else f"{(b - a) / a:+.1%}"
        print(f"  {lbl:<12}{a / 1e8:>13.2f}억{b / 1e8:>13.2f}억{chg:>12}")
    for d in (2.0, 1.15):
        g = got[d]
        print(f"  d={d:<5} 본선 비중 {g['c_vessel'] / g['phi']:>6.2%}"
              f" · 90분위 체류 {g['p90']:.1f}h")

    print("\n" + "=" * 74)
    s115 = sat[1.15]
    share = got[1.15]["c_vessel"] / got[1.15]["phi"]
    if s115 < 0.60 and share < 0.15:
        print(f"판정 — **통과.** 포화 {sat[2.0]:.0%} → {s115:.0%}, "
              f"본선 비중 {share:.1%} (<15%).")
        print("       신호가 살아났고 무대는 본선 지연에 삼켜지지 않았다. 2단계로 간다.")
    elif s115 >= 0.60:
        print(f"판정 — **실패.** 포화가 안 풀렸다({s115:.0%}). d 말고 다른 원인 —")
        print("       계획 식(`pc = start + M·c·d`) 자체를 다시 본다.")
    else:
        print(f"판정 — **재검토.** 포화는 풀렸으나({s115:.0%}) 본선 비중이 "
              f"{share:.1%} 로 15% 이상이다.")
        print("       마감이 과하게 조여 판정이 본선 지연 하나에 휘둘린다 —")
        print("       d 를 1.15~2.0 사이에서 다시 고른다 (YR-109 함정).")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
