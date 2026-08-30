"""(현재 미사용 — 사용자가 만든 fig-mlp.png 를 쓴다) Publication figure: the cost network evaluated once per candidate.

Run from the repository root::

    python scripts/v3/fig_mlp.py

Drawn at the Springer LNCS text width in the same visual language as
``fig_arch.py``. PDF and SVG are the publication sources; PNG is a preview.

⚠️ The activation is **ReLU**, matching ``src/yard_rl/v3/actors/nets.py``.
   An earlier hand-made image said GELU, which the implementation never used.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TEXTWIDTH_IN = 4.80
ROOT = pathlib.Path(__file__).resolve().parents[2]
#: 그림은 한 곳에만 둔다 — 두 논문이 ../figures/ 로 같이 본다.
TARGETS = (ROOT / "docs/paper/v3/figures",)

INK = "#202124"
MUTED = "#5F6368"
RULE = "#B7BDC3"
P_BLUE = "#2F6690"
P_BLUE_BG = "#EDF4F8"
NEUTRAL_BG = "#F5F6F7"

#: (제목, 아래 첨언, 채움, 테두리) — 선형층과 활성층을 번갈아 놓는다.
LAYERS = (
    ("Linear", "$d_{in}\\rightarrow 64$", P_BLUE_BG, P_BLUE),
    ("ReLU", "", NEUTRAL_BG, RULE),
    ("Linear", "$64\\rightarrow 64$", P_BLUE_BG, P_BLUE),
    ("ReLU", "", NEUTRAL_BG, RULE),
    ("Linear", "$64\\rightarrow 1$", P_BLUE_BG, P_BLUE),
)


def _box(ax, x, y, w, h, title, detail="", *, fc="white", ec=RULE,
         title_color=INK, lw=0.85):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.18,rounding_size=0.7",
        facecolor=fc, edgecolor=ec, linewidth=lw))
    ty = y + h * (0.60 if detail else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=title_color)
    if detail:
        ax.text(x + w / 2, y + h * 0.27, detail, ha="center", va="center",
                fontsize=5.6, color=MUTED)


def _arrow(ax, x1, y1, x2, y2, *, color=INK, lw=0.85):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=8, shrinkA=0, shrinkB=0))


def draw():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 1.32))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    y, h = 30.0, 42.0

    # 폭은 xlim(0..100) 안에서 맞춘다 — 넘치면 출력 상자가 잘린다.
    in_w, lay_w, gap, out_w = 14.0, 11.4, 2.6, 12.4

    # 입력 — 후보 하나의 특징 벡터
    _box(ax, 0.5, y, in_w, h, "Candidate $x$",
         "$\\in\\mathbb{R}^{d_{in}}$", fc="white", ec=RULE)

    prev = 0.5 + in_w
    x = prev + gap
    for title, detail, fc, ec in LAYERS:
        _arrow(ax, prev + 0.6, y + h / 2, x - 0.6, y + h / 2)
        _box(ax, x, y, lay_w, h, title, detail, fc=fc, ec=ec,
             title_color=ec if fc is P_BLUE_BG else INK)
        prev = x + lay_w
        x = prev + gap

    # 출력 — 그 후보의 예상 비용 하나
    _arrow(ax, prev + 0.6, y + h / 2, x - 0.6, y + h / 2)
    _box(ax, x, y, out_w, h, "$\\hat c(x)$", "expected cost",
         fc="white", ec=INK)

    ax.text(50.0, 13.0,
            "One forward pass per candidate; the policy takes the "
            "$\\arg\\min$ over candidates.",
            ha="center", va="center", fontsize=5.4, color=MUTED)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            out = target / f"fig-mlp-drawn.{ext}"
            fig.savefig(out, dpi=450, bbox_inches="tight", pad_inches=0.025,
                        facecolor="white")
            print(out)
    plt.close(fig)


if __name__ == "__main__":
    draw()
