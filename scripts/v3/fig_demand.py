"""외부 트럭 작업 수요 그림 — 일일 수요 분포 + 시간대별 도착 과정.

    PYTHONPATH=src python scripts/v3/fig_demand.py

■ 왜 상수를 import 하는가
  수치를 손으로 옮겨 적으면 코드가 바뀔 때 그림만 조용히 옛날 값을 그린다. 여기서는
  `LOAD_WEIGHTS` 와 `DIURNAL_*` 를 **구현에서 직접 읽어** 그린다. 그림과 실험이
  어긋날 수 없다.

■ 무엇을 그리는가
  (a) 일일 외부 트럭 작업 수요 — 다섯 수준과 각각의 추첨 확률
  (b) 시간대별 도착률 λ(t) — 균등 기저와 가우시안 성분 셋의 합. 기저·성분을 따로
      그려서 **곡선의 형태가 문헌값이 아니라 본 연구의 합성 구성**임이 보이게 한다.

  ⚠️ 문헌이 뒷받침하는 것은 *수요 규모*와 *시간대별 도착이 균일하지 않다는 사실*이고,
     혼합의 구체적 형태(봉우리 시각·폭·가중)는 본 연구의 설계값이다. 캡션도 그렇게 쓴다.
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
INK = "#1b1b1b"
BASE_C = "#8c8c8c"
PEAK_C = ("#c0504d", "#4f81bd", "#9bbb59")
TOTAL_C = "#1b1b1b"

TXT = {
    "en": dict(
        a="(a) Daily external truck job demand",
        b="(b) Time-of-day arrival process",
        xa="Trucks per day", ya="Probability",
        xb="Hour of day", yb="Arrival rate (trucks/h)",
        base="uniform base (38%)", total="mixture",
        peak="component {i} ($\\mu$={mu:.0f}h, $\\sigma$={sg:.1f}h)",
        note="levels drawn independently each day"),
    "ko": dict(
        a="(a) 일일 외부 트럭 작업 수요",
        b="(b) 시간대별 트럭 도착 과정",
        xa="하루 트럭 대수", ya="추첨 확률",
        xb="하루 중 시각 (시)", yb="도착률 (대/시간)",
        base="균등 기저 (38%)", total="합성 곡선",
        peak="성분 {i} ($\\mu$={mu:.0f}시, $\\sigma$={sg:.1f}시)",
        note="수준은 날마다 독립으로 추첨"),
}


def _fonts(lang):
    if lang == "ko":
        for fam in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
            try:
                matplotlib.font_manager.findfont(fam, fallback_to_default=False)
                plt.rcParams["font.family"] = fam
                break
            except Exception:
                continue
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def rate_parts(total: int):
    """λ(t) 를 기저와 성분별로 나눠 돌려준다 — 구현식과 같은 분해다."""
    mean_h = total / 24.0
    b = DIURNAL_NIGHT_FRAC * mean_h
    w_sum = sum(w for _, _, w in DIURNAL_PEAKS)
    peak_mass = total - b * 24.0
    hs = [i * 0.05 for i in range(int(24 / 0.05) + 1)]
    comps = []
    for mu, sg, w in DIURNAL_PEAKS:
        a = peak_mass * (w / w_sum)
        comps.append([a * math.exp(-0.5 * ((h - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi)) for h in hs])
    tot = [b + sum(c[i] for c in comps) for i in range(len(hs))]
    return hs, b, comps, tot


def draw(lang: str):
    _fonts(lang)
    t = TXT[lang]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # ── (a) 일일 수요 수준과 추첨 확률 ───────────────────────────
    loads = [ld for ld, _, _ in LOAD_WEIGHTS]
    probs = [w for _, w, _ in LOAD_WEIGHTS]
    xs = range(len(loads))
    ax1.bar(xs, probs, width=0.62, color=BASE_C, edgecolor=INK, linewidth=0.7)
    for x, p in zip(xs, probs):
        ax1.text(x, p + 0.012, f"{p:.0%}", ha="center", va="bottom",
                 fontsize=8.5, color=INK)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([f"{ld:,}" for ld in loads], fontsize=8.5)
    ax1.set_xlabel(t["xa"], fontsize=9)
    ax1.set_ylabel(t["ya"], fontsize=9)
    ax1.set_title(t["a"], fontsize=10, loc="left", color=INK)
    ax1.set_ylim(0, max(probs) * 1.25)
    ax1.tick_params(labelsize=8.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(0.98, 0.94, t["note"], transform=ax1.transAxes, ha="right",
             va="top", fontsize=7.5, color="#5a5a5a")

    # ── (b) 도착 과정 — 기저 + 성분 셋 ──────────────────────────
    ref = 12_500          # 혼잡 수준 하나를 대표로 그린다
    hs, b, comps, tot = rate_parts(ref)
    ax2.axhline(b, color=BASE_C, lw=1.4, ls="--", label=t["base"])
    for i, (c, (mu, sg, _)) in enumerate(zip(comps, DIURNAL_PEAKS)):
        ax2.plot(hs, [b + v for v in c], color=PEAK_C[i], lw=1.0, ls=":",
                 label=t["peak"].format(i=i + 1, mu=mu, sg=sg))
    ax2.plot(hs, tot, color=TOTAL_C, lw=1.8, label=t["total"])
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.set_xlabel(t["xb"], fontsize=9)
    ax2.set_ylabel(t["yb"], fontsize=9)
    ax2.set_title(t["b"] + f"  ({ref:,})", fontsize=10, loc="left", color=INK)
    ax2.tick_params(labelsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, max(tot) * 1.42)          # 범례가 곡선을 안 가리게
    ax2.legend(fontsize=7.2, frameon=False, loc="upper left", ncol=2,
               columnspacing=1.0, handlelength=1.6)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = OUT / ("fig-demand." + ext)
        fig.savefig(f, dpi=200, bbox_inches="tight")
        print(f"  {f}")
    plt.close(fig)


if __name__ == "__main__":
    print("외부 트럭 작업 수요 그림")
    draw("en")          # ★모든 그림은 영어로 낸다 (사용자 지시 2026-08-30)
    # 그림이 쓴 값을 그대로 보고한다 — 캡션·본문과 대조하라
    print("\n[그림이 사용한 구현 상수]")
    print("  일일 수요:", ", ".join(f"{ld:,}({w:.0%})" for ld, w, _ in LOAD_WEIGHTS))
    print("  야간 기저 비율:", DIURNAL_NIGHT_FRAC)
    print("  봉우리 (mu, sigma, weight):", DIURNAL_PEAKS)
