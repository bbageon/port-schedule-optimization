"""7쪽 — 결과: 시간 조정만 되풀이해서 나타났다. 공간 조정은 아니었다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

ROWS = [
    ("시간만 조정", 16, BLUE),
    ("둘 다 조정", 4, RED),
    ("블록만 조정", 3, RED),
]

BAR_X = 0.215          # 칸이 시작하는 자리
CELL_W = 0.026         # 한 달이 한 칸
GAP = 0.002
BAR_H = 0.058
TOP_Y = 0.632          # 맨 위 막대의 아래쪽


def bar_row(s, y, label, count, color):
    """스무 칸을 깔고 그 가운데 '더 싼 달' 만큼을 색으로 채운다."""
    s.text(BAR_X - 0.022, y + BAR_H / 2, label, size=14, color=INK, ha="right")
    for i in range(20):
        x = BAR_X + i * CELL_W
        filled = i < count
        s.band(x, y, CELL_W - GAP, BAR_H,
               face=color if filled else "#F2F4F6",
               edge="none" if filled else FAINT, lw=1.0)
    # 숫자는 스무 칸이 모두 끝난 뒤에 둔다 — 칸 위에 글자가 겹치지 않도록
    s.text(BAR_X + 20 * CELL_W + 0.016, y + BAR_H / 2, str(count), size=24,
           weight="bold", color=color, ha="left")


def build():
    s = Slide("시간 조정은 재현됐고, 공간 조정은 재현되지 않았다",
              "스무 달 가운데 더 싸진 달의 수")

    s.text(0.062, 0.752, "아무것도 바꾸지 않았을 때보다 비용이 낮아진 달 — 한 칸이 한 달이다",
           size=12.5, color=MUTED)

    for i, (label, count, color) in enumerate(ROWS):
        bar_row(s, TOP_Y - i * 0.098, label, count, color)

    s.bullet(0.370, "시간만 조정했을 때 절감 중앙값은 12.7 퍼센트였다.", size=14)

    # ---------------------------------------------------------- 짚고 넘어갈 점
    s.box(0.062, 0.196, 0.876, 0.112, face="#FBF5EC", edge="#E3CDA6", lw=1.4)
    s.band(0.062, 0.212, 0.006, 0.080, face=AMBER, zorder=3)
    s.text(0.092, 0.272, "짚고 넘어갈 점", size=12.5, weight="bold", color=AMBER)
    s.text(0.092, 0.232, "사전에 등록한 통계 구간은 세 비교 모두 0을 포함했다. "
                         "즉 이 표만으로는 확정할 수 없다.", size=13.5, color=INK)

    s.note(0.140, "뒤에 나오는 시뮬레이터 결함 때문에 평균이 크게 흔들렸다. "
                  "그 이야기가 다음 장이다.", width=80)
    return s


if __name__ == "__main__":
    print(save(build(), 7, "result"))
