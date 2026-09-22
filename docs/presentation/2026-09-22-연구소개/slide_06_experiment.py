"""6쪽 — 실험을 어떻게 짰는가: 같은 달을 네 가지 정책으로 굴려 짝지어 비교한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

POLICIES = [
    ("재배정 없음", "아무것도 바꾸지 않는다", MUTED),
    ("시간만 조정", "시각만 바꾼다", BLUE),
    ("블록만 조정", "블록만 바꾼다", BLUE),
    ("둘 다 조정", "시각과 블록을 함께 바꾼다", BLUE),
]

LEFT_X, LEFT_W = 0.062, 0.400
BOX_H, BOX_GAP = 0.105, 0.026
BOTTOM = 0.238

RIGHT_X, RIGHT_W = 0.520, 0.418


def policy_cards(s):
    """정책 넷을 위에서 아래로 — 맨 위가 비교의 기준이다."""
    for i, (name, note, color) in enumerate(POLICIES):
        y = BOTTOM + (len(POLICIES) - 1 - i) * (BOX_H + BOX_GAP)
        base = color is MUTED
        s.box(LEFT_X, y, LEFT_W, BOX_H,
              face="#F7F8FA" if base else "#FFFFFF",
              edge=FAINT, lw=1.4)
        s.band(LEFT_X + 0.014, y + 0.026, 0.006, BOX_H - 0.052, face=color)
        s.text(LEFT_X + 0.038, y + 0.066, name, size=14.5, weight="bold",
               color=MUTED if base else INK)
        s.text(LEFT_X + 0.038, y + 0.034, note, size=11.5, color=MUTED)
        if base:
            s.text(LEFT_X + LEFT_W - 0.022, y + 0.066, "비교의 기준", size=11.5,
                   color=MUTED, ha="right")


def month_grid(s, x0, y0):
    """겹치지 않는 달 스무 개 — 열 개씩 두 줄."""
    cell_w, cell_h, gap = 0.030, 0.036, 0.004
    for i in range(20):
        col, row = i % 10, i // 10
        x = x0 + col * (cell_w + gap)
        y = y0 - row * (cell_h + gap)
        s.band(x, y, cell_w, cell_h, face="#EAF0F5", edge=BLUE, lw=1.0)
    return 10 * (cell_w + gap) - gap


def build():
    s = Slide("서로 겹치지 않는 스무 달을 각각 서른 날씩 굴렸다",
              "같은 달을 네 가지 정책으로 굴려 짝을 지어 비교한다")

    # ---------------------------------------------------------- 왼쪽: 정책 넷
    s.text(LEFT_X, 0.775, "① 바꿔 본 것은 네 가지뿐이다", size=15, weight="bold")
    policy_cards(s)

    # ---------------------------------------------------------- 오른쪽: 굴린 횟수
    s.text(RIGHT_X, 0.775, "② 달마다 네 번씩 굴렸다", size=15, weight="bold")
    s.box(RIGHT_X, BOTTOM, RIGHT_W, 0.498, face="#FCFCFD")

    grid_w = 10 * 0.034 - 0.004
    grid_x = RIGHT_X + (RIGHT_W - grid_w) / 2
    month_grid(s, grid_x, 0.625)
    s.text(RIGHT_X + RIGHT_W / 2, 0.692, "겹치지 않는 달 스무 개 · 달마다 서른 날",
           size=12, color=MUTED, ha="center")

    s.text(RIGHT_X + RIGHT_W / 2, 0.512, "스무 달 × 정책 넷 = 여든 번의 실행",
           size=15, weight="bold", color=INK, ha="center")

    s.stat(RIGHT_X + RIGHT_W * 0.30, 0.402, "20", "독립된 달", color=BLUE, size=42)
    s.stat(RIGHT_X + RIGHT_W * 0.72, 0.402, "80", "정책별 실행", color=INK, size=42)

    s.note(0.150, "학습은 한 번만 했고 평가할 때는 가중치를 고정했다. "
                  "학습을 여러 번 반복한 것은 아니다.", width=80)
    return s


if __name__ == "__main__":
    print(save(build(), 6, "experiment"))
