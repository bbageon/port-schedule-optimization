"""외부 트럭 작업 수요 그림 — 일일 수요 분포 + 시간대별 도착 과정.

    PYTHONPATH=src python scripts/v3/fig_demand.py

■ 왜 상수를 import 하는가
  수치를 손으로 옮겨 적으면 코드가 바뀔 때 그림만 조용히 옛날 값을 그린다. 여기서는
  `LOAD_WEIGHTS` 와 `DIURNAL_*` 를 **구현에서 직접 읽어** 그린다. 그림과 실험이
  어긋날 수 없다.

■ 양식 — LNCS (사용자 지시 2026-08-30 · 견본 이미지)
  · 그림틀을 **네 변 모두** 두른다
  · 계열은 **선 모양**으로 가른다 — 흑백으로 인쇄해도 읽힌다
  · 범례는 **테두리 있는 상자**로 그림 안에 둔다
  · 격자 없음
  · ★그림을 **LNCS 본문 폭 그대로** 그린다. 넓게 그려서 include 때 줄이면 글씨가
    같이 줄어 읽히지 않는다 (9인치를 0.86 textwidth 로 넣으면 46% 로 축소됐다).
  · ★그림은 **이미지만** 만든다 — 제목·설명은 넣지 않는다. 패널 표시 (a)(b) 만 두고
    나머지는 LaTeX 의 \caption 이 템플릿 규칙대로 그림 아래에 붙인다 (사용자 지시).

■ 무엇을 그리는가
  (a) 일일 외부 트럭 작업 수요 — 다섯 수준과 각각의 추첨 확률
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import TEXTWIDTH_IN, apply, hgrid, save

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.05))

    # ── (a) 일일 수요 수준과 추첨 확률 ───────────────────────────
    loads = [ld for ld, _, _ in LOAD_WEIGHTS]
    probs = [w for _, w, _ in LOAD_WEIGHTS]
    xs = list(range(len(loads)))
    ax1.bar(xs, probs, width=0.55, facecolor="white", edgecolor="black",
            linewidth=0.8, hatch="////")
    for x, p in zip(xs, probs):
        ax1.text(x, p + 0.010, f"{p:.0%}", ha="center", va="bottom")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{ld:,}" for ld in loads])
    ax1.set_xlabel("Trucks per day")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, max(probs) * 1.28)
    ax1.set_xlim(-0.6, len(loads) - 0.4)
    ax1.text(0.0, 1.03, "(a)", transform=ax1.transAxes, va="bottom", ha="left")

    # ── (b) 도착 과정 — 기저 + 가우시안 성분 ─────────────────────
    hs, base, comps, tot = rate_parts(REF_LOAD)
    ax2.plot(hs, tot, color="black", lw=1.5, ls="-", label="Mixture")
    ax2.plot([0, 24], [base, base], color="black", lw=0.9, ls="--",
             label="Uniform base (38%)")
    for i, c in enumerate(comps):
        ax2.plot(hs, [base + v for v in c], color="black", lw=0.7, ls=":",
                 label="Gaussian components" if i == 0 else None)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 6))
    ax2.set_ylim(0, max(tot) * 1.10)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Arrival rate (trucks/h)")
    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, va="bottom", ha="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.62, 1.36), ncol=1,
               labelspacing=0.3, handlelength=2.0, handletextpad=0.5,
               borderaxespad=0.0)

    # 위·오른쪽 테두리 제거 + 가로 격자만 (집 규칙 · figstyle)
    for ax in (ax1, ax2):
        hgrid(ax)

    fig.tight_layout()
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
