r"""외부 트럭 작업 수요 그림 — 일일 수요 분포 + 시간대별 도착 과정.

    PYTHONPATH=src python scripts/v3/fig_demand.py

■ 왜 상수를 import 하는가
  수치를 손으로 옮겨 적으면 코드가 바뀔 때 그림만 조용히 옛날 값을 그린다. 여기서는
  `LOAD_WEIGHTS` 와 `DIURNAL_*` 를 **구현에서 직접 읽어** 그린다. 그림과 실험이
  어긋날 수 없다. 범례의 균등 기저 비율(38%)도 `DIURNAL_NIGHT_FRAC` 에서 만든다.

■ 양식은 `figstyle.py` 한 곳에서 온다 (사용자 지시 2026-08-30)
  결과 그림 셋(`fig_results.py`)과 **같은 문법**으로 그린다. 한 논문 안에서 그림마다
  색·범례 자리·패널 이름 규칙이 달라지면 독자가 그림마다 읽는 법을 새로 배워야 한다.

    · 위·오른쪽 테두리 없음 · 가로 격자만 아주 옅게 (`hgrid`)
    · 주 계열은 `P_BLUE`, 분해 성분은 `MUTED` — 색과 선모양이 늘 같이 간다
      (흑백 인쇄에서도 실선/파선/점선으로 갈린다)
    · 범례는 **그림 위 바깥에 하나** (`loc="outside upper center"`) — 앞판은 (b) 안
      오른쪽 위에 얹혀 곡선 머리와 겹쳤다
    · 패널 이름은 `panel()` 로 왼쪽 위에, **무슨 패널인지 이름으로** 말한다 —
      "(a)" 만 두면 독자가 캡션을 찾아 내려가야 한다
    · 축 밖으로 나간 값이 하나라도 있으면 `no_clip` 이 멈춘다
  · ★그림을 **LNCS 본문 폭 그대로** 그린다. 넓게 그려서 include 때 줄이면 글씨가
    같이 줄어 읽히지 않는다 (9인치를 0.86 textwidth 로 넣으면 46% 로 축소됐다).
  · ★그림은 **이미지만** 만든다 — 제목·설명은 넣지 않는다. 패널 이름만 두고
    나머지는 LaTeX 의 \caption 이 템플릿 규칙대로 그림 아래에 붙인다 (사용자 지시).

■ 무엇을 그리는가
  (a) 일일 외부 트럭 작업 수요 — 다섯 수준과 각각의 추첨 확률. 확률은 **퍼센트 축**
      하나로 읽는다 (앞판은 축이 0.0~0.2 소수, 막대 위 글씨는 30% 라 단위가 둘이었다).
  (b) 시간대별 도착률 λ(t) — 균등 기저와 가우시안 성분 셋의 합. 기저·성분을 따로
      그려서 **곡선의 형태가 문헌값이 아니라 본 연구의 합성 구성**임이 보이게 한다.

  ⚠️ 문헌이 뒷받침하는 것은 *수요 규모*와 *시간대별 도착이 균일하지 않다는 사실*이고,
     혼합의 구체적 형태(봉우리 시각·폭·가중)는 본 연구의 설계값이다. 캡션도 그렇게 쓴다.
  ⚠️ 모든 그림은 **영어**로 낸다 (사용자 지시) — 국문판도 같은 파일을 쓴다.
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import (BAND, INK, MUTED, P_BLUE, SMALL_PT, TEXTWIDTH_IN, apply,
                      hgrid, no_clip, panel, save)

from yard_rl.v3.world.integrated.terminal_stream import (
    DIURNAL_NIGHT_FRAC, DIURNAL_PEAKS)

from figdata import month

#: 부하 가중 추첨 — `month.py` 원본에서 든다 (torch 없이 그림이 그려지도록).
LOAD_WEIGHTS = month().LOAD_WEIGHTS

OUT = pathlib.Path("docs/paper/v3/figures")
REF_LOAD = 12_500          # (b) 를 그릴 대표 수요 수준


def rate_parts(total: int):
    """λ(t) 를 기저와 성분별로 나눠 돌려준다 — 구현식과 같은 분해다."""
    mean_h = total / 24.0
    base = DIURNAL_NIGHT_FRAC * mean_h
    w_sum = sum(w for _, _, w in DIURNAL_PEAKS)
    peak_mass = total - base * 24.0
    hs = [i * 0.05 for i in range(int(24 / 0.05) + 1)]
    comps = []
    for mu, sg, w in DIURNAL_PEAKS:
        a = peak_mass * (w / w_sum)
        comps.append([a * math.exp(-0.5 * ((h - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi)) for h in hs])
    tot = [base + sum(c[i] for c in comps) for i in range(len(hs))]
    return hs, base, comps, tot


def draw():
    #  (a) 는 눈금 글씨가 다섯 개("12,500" 처럼 긴 것 포함)라 조금 넓게 준다.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.35), layout="constrained",
        gridspec_kw={"width_ratios": [1.14, 1.0]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)

    # ── (a) 일일 수요 수준과 추첨 확률 ───────────────────────────
    loads = [ld for ld, _, _ in LOAD_WEIGHTS]
    pct = [w * 100.0 for _, w, _ in LOAD_WEIGHTS]
    xs = list(range(len(loads)))
    #  계열이 하나뿐이라 해칭을 쓰지 않는다 — 결과 그림에서 해칭은 "색으로 안 갈리는
    #  흰 막대" 하나에만 얹는 표시다. 여기서 사선을 깔면 뜻 없는 무늬가 된다.
    ax1.bar(xs, pct, width=0.62, facecolor=P_BLUE, edgecolor=INK,
            linewidth=0.7, zorder=3)
    for x, p in zip(xs, pct):
        ax1.text(x, p + 1.1, f"{p:.0f}%", ha="center", va="bottom",
                 fontsize=SMALL_PT, color=INK, zorder=4)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{ld:,}" for ld in loads])
    ax1.tick_params(axis="x", length=0, pad=3)
    ax1.set_xlabel("Daily demand (trucks/day)")
    ax1.set_ylabel("Draw probability (%)")
    ax1.set_yticks([0, 10, 20, 30])
    ax1.set_ylim(0, max(pct) * 1.32)
    ax1.set_xlim(-0.62, len(loads) - 0.38)
    no_clip(ax1, [p + 1.1 for p in pct], "수요 분포 (a)")
    hgrid(ax1)
    panel(ax1, "(a) Daily demand levels")

    # ── (b) 도착 과정 — 기저 + 가우시안 성분 ─────────────────────
    hs, base, comps, tot = rate_parts(REF_LOAD)
    #  기저 아래를 아주 옅게 채운다 — "이만큼은 하루 어느 시각에나 깔려 있다" 가
    #  선 하나보다 한눈에 읽힌다 (음영은 결과 그림의 `BAND` 와 같은 문법).
    ax2.fill_between([0, 24], 0, base, color=BAND, alpha=0.55, lw=0, zorder=0)
    for i, c in enumerate(comps):
        ax2.plot(hs, [base + v for v in c], color=MUTED, lw=0.85, ls=":",
                 zorder=2)
    ax2.plot([0, 24], [base, base], color=MUTED, lw=1.0, ls="--", zorder=3)
    ax2.plot(hs, tot, color=P_BLUE, lw=1.6, ls="-", zorder=4)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 6))
    ax2.set_ylim(0, max(tot) * 1.12)
    ax2.set_yticks([0, 500, 1000])
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Arrival rate (trucks/h)")
    no_clip(ax2, tot, "도착 과정 (b)")
    hgrid(ax2)
    panel(ax2, f"(b) Arrival process at {REF_LOAD:,}/day")

    # 범례는 그림 위 바깥에 한 줄 — 합(굵은 실선)을 먼저 읽고 그 아래 분해를 읽는
    # 순서가 되도록 Mixture 를 맨 앞에 둔다.
    fig.legend(
        [Line2D([], [], color=P_BLUE, lw=1.6, ls="-"),
         Line2D([], [], color=MUTED, lw=1.0, ls="--"),
         Line2D([], [], color=MUTED, lw=0.85, ls=":")],
        ["Mixture", f"Uniform base ({DIURNAL_NIGHT_FRAC:.0%})",
         "Gaussian components"],
        ncol=3, loc="outside upper center", handlelength=1.9,
        columnspacing=1.4, handletextpad=0.5)

    save(fig, OUT, "fig-demand")


if __name__ == "__main__":
    apply()
    print("외부 트럭 작업 수요 그림 (집 규칙 · figstyle)")
    draw()
    # 그림이 쓴 값을 그대로 보고한다 — 캡션·본문과 대조하라
    print("\n[그림이 사용한 구현 상수]")
    print("  일일 수요:", ", ".join(f"{ld:,}({w:.0%})" for ld, w, _ in LOAD_WEIGHTS))
    print("  야간 기저 비율:", DIURNAL_NIGHT_FRAC)
    print("  봉우리 (mu, sigma, weight):", DIURNAL_PEAKS)
