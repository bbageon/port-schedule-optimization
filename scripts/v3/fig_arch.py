"""(현재 미사용 — 사용자가 만든 fig-system.png 를 쓴다) Publication figure: online decision flow and offline counterfactual learning.

Run from the repository root::

    python scripts/v3/fig_arch.py

The figure is drawn at the Springer LNCS text width. PDF and SVG are the
publication sources; PNG is only a quick preview. Labels are deliberately
short because the explanation belongs in the LaTeX caption.
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

# Okabe-Ito-inspired, print-safe palette. Meaning is also carried by labels
# and line styles, so no distinction relies on colour alone.
INK = "#202124"
MUTED = "#5F6368"
RULE = "#B7BDC3"
P_BLUE = "#2F6690"
P_BLUE_BG = "#EDF4F8"
A_GREEN = "#3B7D6B"
A_GREEN_BG = "#EDF6F2"
NEUTRAL_BG = "#F5F6F7"
WARM_BG = "#FBF4E8"


def _box(ax, x, y, w, h, title, detail="", *, fc="white", ec=RULE,
         title_color=INK, lw=0.8, dashed=False, title_size=6.6,
         detail_size=5.7):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18,rounding_size=0.7",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=title_color)
    if detail:
        ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center",
                fontsize=detail_size, color=MUTED, linespacing=1.12)
    return patch


def _arrow(ax, x1, y1, x2, y2, *, color=INK, dashed=False, label=None,
           label_dy=1.7, lw=0.85):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=(0, (3, 2)) if dashed else "solid",
            mutation_scale=8, shrinkA=0, shrinkB=0,
        ),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label,
                ha="center", va="bottom", fontsize=5.4, color=MUTED)


def draw():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # (a) Online inference.
    ax.text(0.5, 96.7, "(a) Online inference — every 60 s", fontsize=7.0,
            fontweight="bold", color=INK, va="top")
    y, h = 71.0, 17.0
    xs = (0.8, 21.1, 41.4, 61.7, 82.0)
    widths = (15.5, 15.5, 15.5, 15.5, 17.2)
    boxes = (
        ("Observe", "block state\npublic ETA", "white", RULE, INK),
        ("Proposal", "score feasible\nactions", P_BLUE_BG, P_BLUE, P_BLUE),
        ("Acceptance", "accept vs. reject", A_GREEN_BG, A_GREEN, A_GREEN),
        ("Commit", "consent +\ncapacity", WARM_BG, INK, INK),
        ("Execute", "update assignment\nand terminal state", "white", RULE, INK),
    )
    for x, w, (title, detail, fc, ec, tc) in zip(xs, widths, boxes):
        _box(ax, x, y, w, h, title, detail, fc=fc, ec=ec,
             title_color=tc, lw=0.9, title_size=6.25)
    for i in range(4):
        _arrow(ax, xs[i] + widths[i] + 0.5, y + h / 2,
               xs[i + 1] - 0.5, y + h / 2)
    # KEEP / REJECT falls through to the environment with the assignment
    # unchanged. The arrows land on the environment box so they point at
    # something; the label sits between them, clear of both.
    keep_xs = (xs[1] + widths[1] / 2, xs[2] + widths[2] / 2)
    for x, color in zip(keep_xs, (P_BLUE, A_GREEN)):
        _arrow(ax, x, y - 0.4, x, 60.4, color=color, dashed=True, lw=0.65)
    # 짧게 둔다 — 두 화살표 사이에 딱 들어가야 선을 갉지 않는다.
    # "무엇이 안 바뀌는가" 는 캡션이 설명한다.
    ax.text(sum(keep_xs) / 2, 65.2, "KEEP / REJECT", ha="center", va="center",
            fontsize=5.3, color=MUTED)

    # Environment: the single source of state transition and cost.
    _box(ax, 9.0, 47.2, 82.0, 13.0,
         "Discrete-event terminal environment",
         "gate  →  21 blocks × 2 yard cranes  →  vessel streams\n"
         "cost Φ: dwell · travel · rehandling · vessel idle",
         fc=NEUTRAL_BG, ec=RULE, lw=0.8, title_size=6.25,
         detail_size=5.35)
    # 라벨을 화살표 옆으로 뺀다 — 선 위에 얹으면 서로 갉아먹는다.
    _arrow(ax, xs[4] + widths[4] / 2, y - 0.4, 88.5, 60.4, color=INK, lw=0.7)
    ax.text(91.5, 65.4, "state transition", ha="left", va="center",
            fontsize=5.4, color=MUTED)
    _arrow(ax, 11.5, 60.4, xs[0] + widths[0] / 2, y - 0.4,
           color=MUTED, dashed=True, lw=0.65)
    ax.text(0.5, 65.4, "next state", ha="left", va="center",
            fontsize=5.4, color=MUTED)

    # (b) Offline counterfactual label generation.
    ax.text(0.5, 41.7, "(b) Offline learning — counterfactual simulation only",
            fontsize=7.0, fontweight="bold", color=INK, va="top")
    _box(ax, 1.0, 10.0, 14.5, 18.0, "Snapshot", "state +\nrandom stream",
         fc="white", ec=RULE)

    world_x, world_w = 23.0, 21.0
    world_y = (28.0, 19.0, 10.0)
    world_titles = ("Observed action", "Proposal reversed", "Acceptance reversed")
    world_edges = (INK, P_BLUE, A_GREEN)
    for wy, title, edge in zip(world_y, world_titles, world_edges):
        _box(ax, world_x, wy, world_w, 6.4, title, "$H=3$ h → $\\Phi_H$",
             fc="white", ec=edge, title_color=edge, lw=0.8,
             title_size=5.55, detail_size=4.9)
        _arrow(ax, 15.8, 19.0, world_x - 0.6, wy + 3.2,
               color=edge, lw=0.7)

    _box(ax, 51.5, 13.0, 18.0, 16.0, "Centre paired costs",
         "$y=(\\Phi_H-\\bar\\Phi_H)/10^5$",
         fc=NEUTRAL_BG, ec=RULE, title_size=5.7, detail_size=5.3)
    # 세 화살표를 한 점에 몰지 않는다 — 겹쳐서 뭉치면 어디로 가는지 안 보인다.
    for wy, edge, ty in zip(world_y, world_edges, (25.0, 21.0, 17.0)):
        _arrow(ax, world_x + world_w + 0.5, wy + 3.2,
               51.0, ty, color=edge, lw=0.65)

    _box(ax, 77.0, 13.0, 21.5, 16.0, "Fit both cost nets",
         "Huber loss · Adam\nproduces $\\theta$ and $\\psi$",
         fc=WARM_BG, ec=RULE, title_size=5.9, detail_size=5.2)
    _arrow(ax, 70.0, 21.0, 76.4, 21.0, color=INK)
    ax.text(99.0, 5.0,
            "Counterfactual branches generate labels offline; online decisions use only the fitted networks.",
            ha="right", va="center", fontsize=5.3, color=MUTED)

    fig.subplots_adjust(left=0.012, right=0.992, bottom=0.018, top=0.992)
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            out = target / f"fig-arch.{ext}"
            fig.savefig(out, dpi=450, bbox_inches="tight", pad_inches=0.025,
                        facecolor="white")
            print(out)
    plt.close(fig)


if __name__ == "__main__":
    draw()
