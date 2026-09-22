"""발표 슬라이드 공통 양식 — 모든 슬라이드가 이 한 곳을 쓴다.

■ 규칙
  · 16:9 · 1920×1080 PNG (figsize 12×6.75 in, dpi 160)
  · 한글 본문 폰트 Malgun Gothic (없으면 Noto Sans KR → NanumGothic)
  · 배경 흰색 · 제목 아래 가는 강조선 · 오른쪽 아래 쪽번호
  · 색은 논문 그림과 같은 팔레트를 쓴다 (발표와 논문이 따로 놀지 않도록)
  · **한 장에 한 가지 메시지** — 제목이 곧 결론이고, 본문은 근거다
  · 슬라이드 안에 표를 길게 넣지 않는다. 숫자는 두세 개만 크게.

■ 쓰는 법
    from deckstyle import Slide, save, INK, BLUE, GREEN, RED, MUTED
    s = Slide('제목', '부제 한 줄')
    s.bullet(0.86, '첫 줄')
    save(s, 3)            # → 03-....png 는 호출자가 이름을 준다
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- 폰트
_KO = ["Malgun Gothic", "Noto Sans KR", "NanumGothic", "Gulim"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
FONT = next((name for name in _KO if name in _available), "DejaVu Sans")
matplotlib.rcParams["font.family"] = FONT
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 색
INK = "#1A1D21"        # 본문 글자
MUTED = "#6B7280"      # 부제·각주
FAINT = "#E5E7EB"      # 옅은 칸·구분선
BLUE = "#2F6690"       # 제안망 · 개선 · 주된 강조
GREEN = "#3B7D6B"      # 수락망 · 통과
RED = "#B4534F"        # 악화 · 결함
AMBER = "#B9822F"      # 주의 · 진행 중
PAPER = "#FFFFFF"

W_IN, H_IN = 12.0, 6.75
DPI = 160
OUT = pathlib.Path(__file__).resolve().parent / "slides"

FOOT = "부산항 야드크레인 강화학습 · 연구 소개와 진행 상황 · 2026-09-22"


class Slide:
    """빈 16:9 화면 하나. 좌표는 0~1 비율이며 왼쪽 아래가 (0,0) 이다."""

    MARGIN = 0.062

    def __init__(self, title: str = "", subtitle: str = "", *, cover: bool = False):
        self.fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=PAPER)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.cover = cover
        if title and not cover:
            self.title(title, subtitle)

    # ------------------------------------------------------------ 기본 요소
    def title(self, text: str, subtitle: str = "") -> None:
        x = self.MARGIN
        self.ax.text(x, 0.905, text, fontsize=27, fontweight="bold", color=INK,
                     va="baseline", ha="left")
        self.ax.add_line(plt.Line2D([x, x + 0.055], [0.877, 0.877], color=BLUE,
                                    lw=3.2, solid_capstyle="butt"))
        if subtitle:
            self.ax.text(x, 0.838, subtitle, fontsize=13.5, color=MUTED,
                         va="baseline", ha="left")

    def text(self, x, y, s, *, size=13, color=INK, weight="normal", ha="left",
             va="center", **kw):
        return self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                            ha=ha, va=va, **kw)

    def bullet(self, y, s, *, x=None, size=13.5, color=INK, weight="normal",
               marker=True, indent=0.0):
        x = self.MARGIN if x is None else x
        if marker:
            self.ax.add_patch(plt.Circle((x + indent + 0.006, y), 0.0055, color=BLUE,
                                         zorder=3))
            x = x + 0.022
        return self.text(x + indent, y, s, size=size, color=color, weight=weight)

    def box(self, x, y, w, h, *, face="#FFFFFF", edge=FAINT, lw=1.4, radius=0.012,
            zorder=1, alpha=1.0):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                               facecolor=face, edgecolor=edge, linewidth=lw,
                               zorder=zorder, alpha=alpha, mutation_aspect=W_IN / H_IN)
        self.ax.add_patch(patch)
        return patch

    def band(self, x, y, w, h, *, face=FAINT, edge="none", lw=1.0, zorder=2):
        """납작한 사각형. 기본 zorder 는 box(1) 보다 **위**다 — 칸 안에 칸을
        그리는 일이 흔한데 아래에 깔리면 통째로 사라진다 (실제로 한 번 겪었다)."""
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                    linewidth=lw, zorder=zorder))

    def wrap(self, x, y, s, width, *, size=11.5, color=MUTED, leading=0.036,
             weight="normal", ha="left"):
        """글자 수로 줄을 나눠 여러 줄로 쓴다. 각주가 화면 밖으로 나가는 사고를 막는다.

        `width` 는 **한 줄에 넣을 글자 수**다 (한글 기준). matplotlib 은 자동
        줄바꿈이 없어서 길이를 직접 재야 한다.
        """
        words, line, lines = s.split(), "", []
        for word in words:
            trial = f"{line} {word}".strip()
            if len(trial) > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        for i, text in enumerate(lines):
            self.text(x, y - i * leading, text, size=size, color=color,
                      weight=weight, ha=ha)
        return len(lines)

    def stat(self, x, y, value, label, *, color=BLUE, size=40, label_size=12.5):
        """숫자 하나를 크게, 그 아래 무슨 숫자인지 한 줄."""
        self.text(x, y, value, size=size, color=color, weight="bold", ha="center",
                  va="center")
        self.text(x, y - 0.088, label, size=label_size, color=MUTED, ha="center",
                  va="center")

    def arrow(self, x0, y0, x1, y1, *, color=MUTED, lw=1.8, style="-|>", ls="-",
              rad=0.0, zorder=2):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=zorder,
                         arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                         linestyle=ls, shrinkA=0, shrinkB=0,
                                         connectionstyle=f"arc3,rad={rad}"))

    def note(self, y, s, *, color=MUTED, size=11.5, width=62):
        """맨 아래 각주 — 한계·출처를 숨기지 않는다. 길면 알아서 줄을 나눈다."""
        return self.wrap(self.MARGIN, y, s, width, size=size, color=color)


def save(slide: Slide, number: int, name: str) -> pathlib.Path:
    """쪽번호와 꼬리말을 붙여 `slides/NN-name.png` 로 저장한다."""
    OUT.mkdir(parents=True, exist_ok=True)
    if not slide.cover:
        slide.ax.text(1 - Slide.MARGIN, 0.045, str(number), fontsize=11.5,
                      color=MUTED, ha="right", va="center")
        slide.ax.text(Slide.MARGIN, 0.045, FOOT, fontsize=9.5, color="#9CA3AF",
                      ha="left", va="center")
    path = OUT / f"{number:02d}-{name}.png"
    slide.fig.savefig(path, dpi=DPI, facecolor=PAPER)
    plt.close(slide.fig)
    return path
