"""3쪽 — 문제: 트럭이 언제 올지와 어디로 갈지는 미리 정해져 있다.

배정이 그대로면 한 시간대·한 블록으로 몰리고, 그 몰림이 그대로 비용이 된다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

# ---------------------------------------------------------------- 윗줄 흐름도
STEPS = [
    ("외부 트럭", "올 시각이 정해져 있다", True),
    ("게이트", "들어오고 나간다", False),
    ("야드 블록", "갈 블록이 정해져 있다", True),
    ("크레인 작업", "블록마다 나눠 맡는다", False),
]
BOX_W, GAP = 0.175, 0.058
BOX_Y, BOX_H = 0.615, 0.125


def flow(s):
    """외부 트럭 → 게이트 → 야드 블록 → 크레인 작업. 미리 정해지는 두 칸은 파랗게."""
    for i, (name, sub, fixed) in enumerate(STEPS):
        x = Slide.MARGIN + i * (BOX_W + GAP)
        s.box(x, BOX_Y, BOX_W, BOX_H,
              face="#EDF3F8" if fixed else "#F7F8FA",
              edge=BLUE if fixed else FAINT, lw=1.7 if fixed else 1.4)
        s.text(x + BOX_W / 2, BOX_Y + 0.082, name, size=14.5, weight="bold",
               color=BLUE if fixed else INK, ha="center")
        s.text(x + BOX_W / 2, BOX_Y + 0.038, sub, size=11, color=MUTED, ha="center")
        if i < len(STEPS) - 1:
            s.arrow(x + BOX_W + 0.012, BOX_Y + BOX_H / 2,
                    x + BOX_W + GAP - 0.012, BOX_Y + BOX_H / 2,
                    color=MUTED, lw=2.0)


# ---------------------------------------------------------------- 아랫줄 결과
PAN_Y, PAN_H, PAN_W = 0.175, 0.285, 0.418


def bars(s, cx):
    """시간대별 도착 — 한 칸만 유난히 높다."""
    w, gap = 0.030, 0.010
    heights = [0.026, 0.036, 0.030, 0.098, 0.046, 0.028, 0.024]
    x0 = cx - (len(heights) * w + (len(heights) - 1) * gap) / 2
    for i, h in enumerate(heights):
        hot = i == 3
        s.band(x0 + i * (w + gap), 0.290, w, h,
               face=AMBER if hot else "#E9ECEF", edge="none")


def blocks(s, cx):
    """야드 블록 다섯 칸 — 한 칸만 유난히 높이 쌓였다."""
    w, gap = 0.042, 0.012
    x0 = cx - (5 * w + 4 * gap) / 2
    for i in range(5):
        hot = i == 2
        s.band(x0 + i * (w + gap), 0.290, w, 0.098 if hot else 0.070,
               face="#E7CBC8" if hot else "#F2F4F6",
               edge=RED if hot else FAINT, lw=1.4)


def panel(s, x, heading, axis_label, chip, accent, chip_face, draw):
    s.box(x, PAN_Y, PAN_W, PAN_H, face="#FCFCFD")
    s.text(x + 0.023, 0.418, heading, size=14.5, weight="bold")
    draw(s, x + PAN_W / 2)
    s.text(x + PAN_W / 2, 0.262, axis_label, size=10.5, color=MUTED, ha="center")
    s.box(x + 0.114, 0.190, 0.190, 0.052, face=chip_face, edge=accent, lw=1.3,
          radius=0.009, zorder=2)
    s.text(x + PAN_W / 2, 0.216, chip, size=13.5, weight="bold", color=accent,
           ha="center")


def build():
    s = Slide("트럭이 언제 올지와 어디로 갈지는 미리 정해진다",
              "그 배정이 그대로면 한 시간대, 한 블록으로 몰린다")

    flow(s)
    s.text(0.5, 0.562, "파란 칸 두 개가 미리 정해지는 것 — 언제 올지, 어디로 갈지",
           size=12.5, color=BLUE, ha="center")
    s.arrow(0.5, 0.532, 0.5, 0.487, color=MUTED, lw=2.0)

    panel(s, 0.062, "같은 시간대에 몰리면 트럭이 줄을 선다", "시간대",
          "트럭 대기 비용", AMBER, "#FBF3E4", bars)
    panel(s, 0.520, "같은 블록에 몰리면 크레인이 못 따라간다", "야드 블록",
          "본선 작업 지연", RED, "#F8ECEB", blocks)

    s.note(0.128, "비용은 트럭이 기다린 시간과 본선이 늦어진 시간을 돈으로 환산해 "
                  "더한 값이다.", width=80)
    return s


if __name__ == "__main__":
    print(save(build(), 3, "problem"))
