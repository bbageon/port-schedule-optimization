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
  · 격자 없음 · 작은 글씨 · 캡션은 LaTeX 쪽에서 그림 **아래**에 붙는다

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

from yard_rl.v3.stage.month import LOAD_WEIGHTS
from yard_rl.v3.world.integrated.terminal_stream import (
    DIURNAL_NIGHT_FRAC, DIURNAL_PEAKS)

OUT = pathlib.Path("docs/paper/v3/figures-demand")
REF_LOAD = 12_500          # (b) 를 그릴 대표 수요 수준


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.7,      # 얇은 그림틀
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fancybox": False,   # 각진 테두리 상자
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "axes.unicode_minus": False,
    })


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
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.1))

    # ── (a) 일일 수요 수준과 추첨 확률 ───────────────────────────
    loads = [ld for ld, _, _ in LOAD_WEIGHTS]
    probs = [w for _, w, _ in LOAD_WEIGHTS]
    xs = list(range(len(loads)))
    ax1.bar(xs, probs, width=0.55, facecolor="white", edgecolor="black",
            linewidth=0.8, hatch="////")
    for x, p in zip(xs, probs):
        ax1.text(x, p + 0.010, f"{p:.0%}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{ld:,}" for ld in loads])
    ax1.set_xlabel("Trucks per day")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, max(probs) * 1.28)
    ax1.set_xlim(-0.6, len(loads) - 0.4)
    ax1.set_title("(a) Daily external truck job demand", fontsize=8.5, loc="left")

    # ── (b) 도착 과정 — 기저 + 가우시안 성분 ─────────────────────
    hs, base, comps, tot = rate_parts(REF_LOAD)
    ax2.plot(hs, tot, color="black", lw=1.5, ls="-", label="Mixture")
    ax2.plot([0, 24], [base, base], color="black", lw=0.9, ls="--",
             label="Uniform base (38%)")
    for i, c in enumerate(comps):
        ax2.plot(hs, [base + v for v in c], color="black", lw=0.7, ls=":",
                 label="Gaussian components" if i == 0 else None)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.set_ylim(0, max(tot) * 1.40)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Arrival rate (trucks/h)")
    ax2.set_title(f"(b) Arrival process at {REF_LOAD:,} trucks/day",
                  fontsize=8.5, loc="left")
    ax2.legend(fontsize=7.2, loc="upper left", frameon=True, borderpad=0.5,
               handlelength=2.4)

    # 네 변 모두 두른다 (LNCS 견본)
    for ax in (ax1, ax2):
        for sp in ax.spines.values():
            sp.set_visible(True)
        ax.tick_params(labelsize=7.5, top=False, right=False)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = OUT / ("fig-demand." + ext)
        fig.savefig(f, dpi=220, bbox_inches="tight")
        print(f"  {f}")
    plt.close(fig)


if __name__ == "__main__":
    print("외부 트럭 작업 수요 그림 (LNCS 양식)")
    draw()
    # 그림이 쓴 값을 그대로 보고한다 — 캡션·본문과 대조하라
    print("\n[그림이 사용한 구현 상수]")
    print("  일일 수요:", ", ".join(f"{ld:,}({w:.0%})" for ld, w, _ in LOAD_WEIGHTS))
    print("  야간 기저 비율:", DIURNAL_NIGHT_FRAC)
    print("  봉우리 (mu, sigma, weight):", DIURNAL_PEAKS)
