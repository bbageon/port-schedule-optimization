"""재배치 결정 흐름 그림 — 운영 경로와 학습 경로의 분리.

    PYTHONPATH=src python scripts/v3/fig_arch.py

■ 왜 다시 그리는가
  이전 그림(손으로 쓴 SVG)은 세 가지가 어긋나 있었다.
    · 그림 **안에** "Fig. 1. ..." 캡션이 박혀 LaTeX \caption 과 중복됐다
    · 760pt 폭으로 그려 0.52\textwidth 로 넣으니 글씨가 읽히지 않았다
    · 색(남색 상자)에 의존해 흑백 인쇄에서 구분이 사라졌다

■ 양식 — LNCS (사용자 지시 · 견본 이미지)
  · 본문 폭 그대로 그린다 → include 때 축소되지 않아 글씨 크기가 산다
  · 흑백 · 얇은 테두리 · 상자는 흰 채움 (해칭은 글씨를 덮어서 안 쓴다)
  · 두 경로를 테두리 모양으로 가른다 — 운영은 실선, 학습은 파선
  · 그림에는 **제목을 넣지 않는다** — 설명은 LaTeX \caption 이 그림 아래에 붙인다
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

OUT = pathlib.Path("docs/paper/v3/figures")
TEXTWIDTH_IN = 4.80          # LNCS(llncs) 본문 폭

#: 운영 경로 — (제목, 아래 첨언)
STEPS = [
    ("Observe", "block state,\npublic ETA"),
    ("Propose", "argmin of\npredicted cost"),
    ("Accept", "accept vs.\nreject"),
    ("Commit", "consent +\ncapacity"),
    ("Execute", "yard and crane\nstate update"),
]
#: 학습 경로 — 분기 세 세계
WORLDS = ("observed", "proposal\nreversed", "acceptance\nreversed")


def draw():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.linewidth": 0.7,
    })
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.28))
    ax.set_xlim(0, 100)
    ax.set_ylim(3, 52)
    ax.axis("off")

    # ── 운영 경로 ────────────────────────────────────────────
    ax.text(0, 49.5, "Operating path — trained networks only, no branch simulation",
            fontsize=6.2, va="bottom")
    bw, bh, gap = 16.4, 12.0, 4.4
    y = 34.0
    for i, (title, sub) in enumerate(STEPS):
        x = i * (bw + gap)
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor="white",
                               edgecolor="black", linewidth=0.8))
        ax.text(x + bw / 2, y + bh - 3.4, title, ha="center", va="center",
                fontsize=7.0, fontweight="bold")
        ax.text(x + bw / 2, y + 3.4, sub, ha="center", va="center", fontsize=5.6)
        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrow(x + bw + 0.5, y + bh / 2, gap - 1.6, 0,
                                    width=0.18, head_width=1.5, head_length=1.5,
                                    length_includes_head=True, color="black"))

    # ── 학습 경로 ────────────────────────────────────────────
    ly, lh = 6.0, 18.0
    ax.add_patch(Rectangle((0, ly), 100, lh, facecolor="white",
                           edgecolor="black", linewidth=0.8, linestyle="--"))
    ax.text(1.6, ly + lh - 2.6,
            "Learning path (offline) — clone state and random stream, "
            "run up to three worlds for $H$ = 3 h",
            fontsize=6.2, va="center")
    ww, wgap = 20.0, 6.0
    x0 = 6.0
    for i, w in enumerate(WORLDS):
        x = x0 + i * (ww + wgap)
        ax.add_patch(Rectangle((x, ly + 3.0), ww, 8.0, facecolor="white",
                               edgecolor="black", linewidth=0.7))
        ax.text(x + ww / 2, ly + 7.0, w, ha="center", va="center", fontsize=5.8)
    ax.text(x0 + 3 * ww + 2 * wgap + 2.0, ly + 7.0,
            "centred $\\Phi$\ndifference\n= target",
            ha="left", va="center", fontsize=5.8)

    # 학습 → Propose·Accept 로 올라가는 화살표 (점선 = 학습 때만 흐른다)
    for cx in (1, 2):
        x = cx * (bw + gap) + bw / 2
        ax.annotate("", xy=(x, y - 0.6), xytext=(x, ly + lh + 0.6),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black",
                                    linestyle="--", shrinkA=0, shrinkB=0))
    ax.text(2 * (bw + gap) + bw + 3.0, (y + ly + lh) / 2,
            "trained weights\n(training only)",
            ha="left", va="center", fontsize=5.8)

    fig.tight_layout(pad=0.1)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = OUT / ("fig-arch." + ext)
        fig.savefig(f, dpi=400, bbox_inches="tight", pad_inches=0.02)
        print(f"  {f}")
    plt.close(fig)


if __name__ == "__main__":
    print("재배치 결정 흐름 그림 (LNCS 양식 · 제목 없음)")
    draw()
