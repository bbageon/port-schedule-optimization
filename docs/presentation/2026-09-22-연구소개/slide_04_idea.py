"""4쪽 — 생각: 이미 정해진 배정을 다시 조정한다.

바꿀 수 있는 손잡이는 둘뿐이다 — 오는 시각, 갈 블록.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

PAN_Y, PAN_H, PAN_W = 0.360, 0.370, 0.418


def panel(s, x, header, accent, desc, *, move_block):
    """한 칸 = 조정 한 가지. 위는 오는 시각, 아래는 갈 블록."""
    s.text(x, 0.760, header, size=15.5, weight="bold", color=accent)
    s.box(x, PAN_Y, PAN_W, PAN_H, face="#FCFCFD")

    # ---- 위: 오는 시각을 뒤로 미룬다
    s.text(x + 0.025, 0.688, "오는 시각", size=11.5, color=MUTED)
    s.band(x + 0.030, 0.6270, 0.360, 0.003, face=FAINT)
    s.band(x + 0.085, 0.612, 0.020, 0.032, face="#D9DDE2")
    s.text(x + 0.095, 0.586, "원래 시각", size=10.5, color=MUTED, ha="center")
    s.band(x + 0.270, 0.612, 0.020, 0.032, face=accent)
    s.text(x + 0.280, 0.586, "미룬 시각", size=10.5, color=accent, ha="center")
    s.arrow(x + 0.107, 0.662, x + 0.276, 0.662, color=accent, lw=2.0, rad=-0.28)

    # ---- 아래: 갈 블록은 그대로 두거나, 함께 바꾼다
    s.text(x + 0.025, 0.540, "갈 블록", size=11.5, color=MUTED)
    first_on = not move_block
    s.box(x + 0.070, 0.458, 0.090, 0.060,
          face="#EDF3F8" if first_on else "#F7F8FA",
          edge=BLUE if first_on else FAINT, lw=1.6 if first_on else 1.3,
          radius=0.009, zorder=2)
    s.text(x + 0.115, 0.488, "블록 ㄱ", size=12,
           color=BLUE if first_on else MUTED,
           weight="bold" if first_on else "normal", ha="center")
    s.box(x + 0.250, 0.458, 0.090, 0.060,
          face="#DEEAE5" if move_block else "#F7F8FA",
          edge=GREEN if move_block else FAINT, lw=1.6 if move_block else 1.3,
          radius=0.009, zorder=2)
    s.text(x + 0.295, 0.488, "블록 ㄴ", size=12,
           color=GREEN if move_block else MUTED,
           weight="bold" if move_block else "normal", ha="center")
    if move_block:
        s.arrow(x + 0.168, 0.488, x + 0.242, 0.488, color=GREEN, lw=2.2)
    else:
        s.text(x + 0.205, 0.488, "그대로", size=11.5, color=MUTED, ha="center")

    s.wrap(x + 0.025, 0.400, desc, 32, size=11.5, color=INK, leading=0.032)


def build():
    s = Slide("이미 정해진 배정을 다시 조정한다",
              "바꿀 수 있는 것은 두 가지, 언제 올지와 어디로 갈지")

    panel(s, 0.062, "① 시간만 바꾸기", BLUE,
          "트럭이 오는 시각을 뒤로 미룬다. 블록은 그대로 둔다.", move_block=False)
    panel(s, 0.520, "② 블록까지 바꾸기", GREEN,
          "트럭이 오는 시각을 미루면서, 갈 블록도 함께 바꾼다.", move_block=True)

    s.box(0.062, 0.245, 0.876, 0.070, face="#F4F7F9")
    s.text(0.5, 0.280, "둘 다 할 수도, 시간만 할 수도 있다. 이 연구는 두 가지를 "
                       "따로 재본다.", size=13.5, ha="center")

    s.note(0.165, "예약 정원 상한과 운송사 재예약 비용은 이번 모형에 넣지 않았다. "
                  "한계로 공개한다.", width=80)
    return s


if __name__ == "__main__":
    print(save(build(), 4, "idea"))
