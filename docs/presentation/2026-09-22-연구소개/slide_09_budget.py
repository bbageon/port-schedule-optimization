"""9쪽 — 원인: 후보 열두 칸을 "지금 못 하는 일"이 다 차지한다.

이 장이 발표의 핵심이다. 숫자는 전부 재생 탐침 실측값이다
(`outputs/reports/yr317_v3_stall_diagnosis/legacy-probe-21000000-Y17-b/`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, Slide, save

BAYS = 8
BLOCKED = (1, 2, 3)          # 밀린 일이 놓인 베이 (0부터 셈)
CRANE_AT = (0, 2)            # 두 크레인이 선 베이


def block_diagram(s, x0, y0, width):
    """블록 하나를 옆에서 본 그림 — 베이 여덟 칸과 그 위에 선 크레인 두 대."""
    cell = width / BAYS
    bay_h, crane_h = 0.075, 0.048

    for i in range(BAYS):
        x = x0 + i * cell
        hot = i in BLOCKED
        s.band(x + 0.0015, y0, cell - 0.003, bay_h,
               face="#F6E3E1" if hot else "#F2F4F6",
               edge="#E0BFBC" if hot else FAINT, lw=1.0)
        s.text(x + cell / 2, y0 + bay_h / 2, str(i + 1), size=11,
               color=RED if hot else MUTED, ha="center")

    for i, name in zip(CRANE_AT, ("크레인 ㄱ", "크레인 ㄴ")):
        x = x0 + i * cell
        s.band(x + 0.0015, y0 + bay_h + 0.020, cell - 0.003, crane_h,
               face=BLUE, edge="none")
        s.text(x + cell / 2, y0 + bay_h + 0.020 + crane_h / 2, "YC",
               size=10.5, color="#FFFFFF", weight="bold", ha="center")
        s.text(x + cell / 2, y0 + bay_h + 0.020 + crane_h + 0.030, name,
               size=11, color=INK, ha="center")

    # 둘 사이에 일이 몰려 있다는 표시
    left = x0 + BLOCKED[0] * cell
    right = x0 + (BLOCKED[-1] + 1) * cell
    s.ax.add_line(__import__("matplotlib.pyplot", fromlist=["Line2D"]).Line2D(
        [left, right], [y0 - 0.026, y0 - 0.026], color=RED, lw=2.0))
    s.text((left + right) / 2, y0 - 0.060, "밀린 일 174건이 전부 이 사이",
           size=11.5, color=RED, ha="center")


def choice_list(s, x, top, width, *, escape_kept, label, label_color):
    """크레인이 한 번 결정할 때 보는 열두 칸."""
    row_h, gap = 0.0335, 0.0065
    s.text(x + width / 2, top, label, size=13.5, weight="bold",
           color=label_color, ha="center")
    for row in range(12):
        y = top - 0.036 - (row + 1) * (row_h + gap)
        if row == 11:
            face, edge, txt, tc = "#F2F4F6", FAINT, "기다린다", MUTED
        elif escape_kept and row == 0:
            face, edge, txt, tc = "#DEEAE5", GREEN, "두 칸 물러나기", GREEN
        else:
            face, edge, txt, tc = "#F8ECEB", "#E7CBC8", "지금 못 하는 일", RED
        s.box(x, y, width, row_h, face=face, edge=edge, lw=1.1, radius=0.007)
        s.text(x + width / 2, y + row_h / 2, txt, size=10.5, color=tc, ha="center")
    return top - 0.036 - 12 * (row_h + gap)


def build():
    s = Slide("원인은 행동이 없어서가 아니라, 목록에서 잘려 나가서였다",
              "크레인이 한 번 결정할 때 놓고 고르는 선택지는 열두 칸이다")

    # ---------------------------------------------------------- 왼쪽
    s.text(0.062, 0.760, "① 두 크레인이 서로를 막았다", size=15, weight="bold")
    s.box(0.062, 0.400, 0.372, 0.300, face="#FCFCFD")
    block_diagram(s, 0.088, 0.505, 0.320)
    s.wrap(0.078, 0.408, "서로 안전거리 안에 있어 둘 다 한 발도 못 나간다. "
                         "174건 전부가 '크레인 간섭'으로 거절된다.", 32,
           size=11.5, color=MUTED, leading=0.030)

    # ---------------------------------------------------------- 오른쪽
    s.text(0.470, 0.760, "② 그런데 목록은 급한 순서로만 채운다", size=15, weight="bold")
    col_w = 0.185
    bottom = choice_list(s, 0.470, 0.718, col_w, escape_kept=False,
                         label="수정 전", label_color=RED)
    choice_list(s, 0.470 + col_w + 0.085, 0.718, col_w, escape_kept=True,
                label="수정 후", label_color=GREEN)
    s.arrow(0.470 + col_w + 0.020, 0.440, 0.470 + col_w + 0.066, 0.440,
            color=INK, lw=2.2)

    s.text(0.470 + col_w / 2, bottom - 0.030, "고를 수 있는 건 '기다린다' 뿐",
           size=11.5, color=RED, ha="center")
    s.text(0.470 + col_w + 0.085 + col_w / 2, bottom - 0.030,
           "한 대가 물러나면 174건이 풀린다", size=11.5, color=GREEN, ha="center")

    s.note(0.128, "비켜서는 행동은 엔진에 원래 있었고 예약도 승인됐다. 목록 칸을 급한 "
                  "순서로만 채우느라 잘려 나갔을 뿐이다. 고친 것은 '할 수 있는가'를 급한 "
                  "정도보다 앞에 놓은 한 가지다. 칸 수는 그대로 열둘이다.", width=88)
    return s


if __name__ == "__main__":
    print(save(build(), 9, "budget"))
