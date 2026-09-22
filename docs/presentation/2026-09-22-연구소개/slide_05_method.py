"""5쪽 — 방법: 두 정책이 제안하고 받아들인다. 값어치는 시뮬레이터가 직접 잰다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

STEP_Y, STEP_H, STEP_W, STEP_GAP = 0.545, 0.200, 0.255, 0.055
STEPS = [
    ("① 제안하는 정책", "지금 블록이 '이 트럭을 넘기겠다'고 내놓는다", BLUE),
    ("② 받아들이는 정책", "받을 블록이 '받겠다 / 안 받겠다'를 답한다", GREEN),
    ("③ 가운데에서 확정", "서로 겹치는 요청을 풀고 한 번에 반영한다", INK),
]


def steps(s):
    for i, (head, desc, accent) in enumerate(STEPS):
        x = Slide.MARGIN + i * (STEP_W + STEP_GAP)
        s.box(x, STEP_Y, STEP_W, STEP_H, face="#FCFCFD")
        s.text(x + 0.024, STEP_Y + 0.145, head, size=14.5, weight="bold",
               color=accent)
        s.band(x + 0.024, STEP_Y + 0.118, 0.046, 0.006, face=accent)
        s.wrap(x + 0.024, STEP_Y + 0.082, desc, 18, size=11.5, color=INK,
               leading=0.034)
        if i < len(STEPS) - 1:
            s.arrow(x + STEP_W + 0.010, STEP_Y + STEP_H / 2,
                    x + STEP_W + STEP_GAP - 0.010, STEP_Y + STEP_H / 2,
                    color=MUTED, lw=2.0)


def callout(s):
    """가르치는 신호 — 같은 상황을 두 번 굴려 비용 차이를 값어치로 삼는다."""
    s.box(0.062, 0.185, 0.876, 0.255, face="#F7F9FB")
    s.text(0.088, 0.398, "같은 상황을 두 번 굴린다", size=15, weight="bold")

    s.box(0.088, 0.298, 0.380, 0.058, face="#EDF3F8", edge=BLUE, lw=1.4,
          radius=0.009, zorder=2)
    s.text(0.108, 0.327, "한 번은 결정대로", size=12.5, color=BLUE, weight="bold")
    s.box(0.088, 0.222, 0.380, 0.058, face="#FFFFFF", edge=FAINT, lw=1.4,
          radius=0.009, zorder=2)
    s.text(0.108, 0.251, "한 번은 그 결정을 안 했다고 치고", size=12.5, color=MUTED)

    s.arrow(0.478, 0.327, 0.552, 0.305, color=MUTED, lw=1.8)
    s.arrow(0.478, 0.251, 0.552, 0.273, color=MUTED, lw=1.8)

    s.box(0.566, 0.222, 0.348, 0.134, face="#FFFFFF", edge=FAINT, lw=1.4,
          zorder=2)
    s.text(0.740, 0.322, "두 결과의 비용 차이", size=12, color=MUTED, ha="center")
    s.text(0.740, 0.270, "= 그 결정의 값어치", size=17, weight="bold", color=BLUE,
           ha="center")


def build():
    s = Slide("두 개의 정책이 제안하고 받아들인다",
              "그리고 그 결정이 얼마를 바꿨는지는 시뮬레이터가 직접 재서 가르친다")

    steps(s)
    callout(s)

    s.note(0.135, "정답을 사람이 매기지 않는다. 시뮬레이터가 두 세계를 굴려 스스로 "
                  "값을 매긴다.", width=80)
    return s


if __name__ == "__main__":
    print(save(build(), 5, "method"))
