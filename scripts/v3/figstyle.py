r"""논문 그림의 집 규칙 — 모든 그림이 이 한 곳을 쓴다 (사용자 지시 2026-08-30).

    from figstyle import apply, save, TEXTWIDTH_IN, P_BLUE, A_GREEN

■ 규칙
  · 배경 흰색 · 위·오른쪽 테두리 제거 · 가로 격자만 아주 연하게
  · 세리프(Times New Roman) · **최종 출력 기준 8~10 pt**
  · 축선 0.8 · 자료선 1.4~1.7 · 범례 테두리 없음
  · 그래프 안에 굵은 제목을 넣지 않는다 — 설명은 LaTeX \caption 이 한다
  · 색과 함께 solid/dashed/hatch 를 **동시에** 써서 흑백에서도 갈린다
  · PDF(벡터)가 출판본 · PNG 600 dpi 는 검토용

■ 왜 폭이 4.80 in 인가
  LNCS 본문폭이다. 이 폭으로 그려 `width=\textwidth` 로 넣으면 축소가 없으므로
  여기서 정한 8~10 pt 가 **인쇄물에서도 그대로 8~10 pt** 다. 넓게 그려서 줄이면
  글씨가 같이 줄어 규칙이 깨진다.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 4.80        # LNCS(llncs) 본문폭

# 색은 뜻을 나르되 혼자 나르지 않는다 — 선모양·해칭이 늘 함께 간다.
P_BLUE = "#2F6690"         # 제안망 / 개선
A_GREEN = "#3B7D6B"        # 수락망
LOSS_RED = "#B4534F"       # 악화
INK = "#202124"
MUTED = "#5F6368"

#: 막대 계열 — 셋 다 "재배치 없음 대비" 라 같은 파랑 계단을 쓴다.
#:  셋째(블록만)는 흰 바탕 + 얇은 사선이라 따로 색이 없다 (해칭은 하나에만 쓴다).
BAR_MID = "#7FA5C0"

#: ★기준선이 다른 값(학습 기여)을 나르는 **마커**색 — 막대가 아니라 ◆ 로 그린다.
#:  막대와 같은 문법을 쓰면 "학습" 이 넷째 행동으로 읽힌다 (사용자 지시 2026-08-30).
MARK_GREY = "#6F757A"

#: 뒤에 까는 음영 (강조 구간 · 무거운 회차) — 자료를 가리지 않을 만큼만 옅게.
BAND = "#DCE3E8"

BASE_PT = 8.0              # 본문 글씨 (하한)
PANEL_PT = 8.5             # 패널 이름 ((a) ... )
LABEL_PT = 9.0             # 축 이름
TICK_PT = 8.0              # 눈금
SMALL_PT = 8.0             # 범례·주석 (하한을 지킨다)
GRID_ALPHA = 0.13   # 0선을 살리려면 격자는 더 옅어야 한다


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BASE_PT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelsize": LABEL_PT,
        "axes.labelcolor": INK,
        "axes.spines.top": False,       # 위·오른쪽 테두리 제거
        "axes.spines.right": False,
        "axes.grid": False,             # 가로 격자는 그림마다 켠다
        "grid.linewidth": 0.6,
        "grid.alpha": GRID_ALPHA,
        "grid.color": INK,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "hatch.linewidth": 0.5,         # 해칭은 가늘게 — 굵으면 막대 색을 덮는다
        "legend.frameon": False,        # 범례 테두리 없음
        "legend.fontsize": SMALL_PT,
        "lines.linewidth": 1.5,         # 자료선 1.4~1.7
        "pdf.fonttype": 42,             # 글꼴을 벡터로 심는다
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def hgrid(ax, which: str = "major") -> None:
    """가로 격자만 — 세로 격자는 눈을 어지럽힌다."""
    ax.grid(axis="y", which=which, alpha=GRID_ALPHA, linewidth=0.6, color=INK)
    ax.set_axisbelow(True)


def panel(ax, text: str) -> None:
    """패널 이름은 왼쪽 위에 **가늘게** — 굵은 제목이 아니라 길잡이다.

    여러 패널로 쪼갠 그림은 어느 것이 (a) 인지 그림 안에서 말해야 캡션이 가리킬 수
    있다. 설명은 여전히 캡션의 몫이라 이름만 짧게 둔다.
    """
    ax.set_title(text, loc="left", fontsize=PANEL_PT, color=INK, pad=3.0)


def no_clip(ax, values, what: str, axis: str = "y") -> None:
    """그림은 자료를 **자르지 않는다** — 축 밖으로 나간 값이 하나라도 있으면 멈춘다.

    앞판 학습곡선이 선형축 상한을 넘긴 봉우리를 조용히 잘라 냈다. 눈으로는 "선이
    위로 사라졌다" 로만 보여서 알아채기 어렵다. 축을 손볼 때마다 사람이 다시
    확인하는 대신, 그릴 때 기계가 확인한다.
    """
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    lo, hi = min(lo, hi), max(lo, hi)
    out = [v for v in values if not (lo <= v <= hi)]
    if out:
        raise AssertionError(
            f"{what}: {len(out)} value(s) outside axis "
            f"({lo:.4g}, {hi:.4g}); e.g. {out[:3]}. Widen the axis.")


def save(fig, out_dir, stem: str, root=None) -> None:
    """PDF(출판본) + PNG 600dpi(검토용)."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        f = out / f"{stem}.{ext}"
        fig.savefig(f, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white", **kw)
        print(f"  {f.relative_to(root) if root else f}")
    plt.close(fig)
