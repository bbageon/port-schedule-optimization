"""11쪽 — 지금 논문에 빠진 것과, 그 빈틈을 메우려고 짠 확증 실험."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

CARD_W = 0.425
CARD_H = 0.150


def card(s, x, y, head, body):
    s.box(x, y, CARD_W, CARD_H, face="#FCFCFD", edge=FAINT, lw=1.4)
    s.band(x + 0.012, y + 0.028, 0.005, CARD_H - 0.056, face=BLUE)
    s.text(x + 0.038, y + 0.094, head, size=14.5, color=INK, weight="bold")
    s.text(x + 0.038, y + 0.047, body, size=12.5, color=MUTED)


def build():
    s = Slide("지금 논문에는 \"규칙보다 낫다\"는 증거가 없다",
              "비교 상대가 계속 \"아무것도 안 하기\" 하나뿐이었다")

    # ------------------------------------------------ 빈틈
    s.box(0.062, 0.600, 0.876, 0.185, face="#FCFCFD", edge=FAINT, lw=1.4)
    s.bullet(0.730, "이 저장소의 규약은 판정 기준을 '사람이 만든 규칙과 견주기'로 못박아 두었다.",
             x=0.092, size=12.5)
    s.bullet(0.675, "그런데 여든 번의 실행에 규칙 상대가 하나도 없었다.",
             x=0.092, size=12.5)

    s.band(0.700, 0.632, 0.0015, 0.120, face=FAINT)
    s.text(0.820, 0.722, "0", size=46, color=RED, weight="bold", ha="center")
    s.text(0.820, 0.660, "여든 번 실행에 들어간", size=12, color=MUTED, ha="center")
    s.text(0.820, 0.630, "규칙 상대의 수", size=12, color=MUTED, ha="center")

    # ------------------------------------------------ 확증 실험
    s.text(0.062, 0.548, "그래서 확증 실험은 이렇게 짰다", size=15.5, color=INK,
           weight="bold")

    card(s, 0.062, 0.345, "새 시드 스무 달", "아무도 판정에 쓴 적 없는 대역")
    card(s, 0.513, 0.345, "비교 대상 여덟 가지", "사람이 만든 규칙 둘을 포함한다")
    card(s, 0.062, 0.175, "학습을 세 번 반복", "세 판의 평균 하나만 판정한다")
    card(s, 0.513, 0.175, "미리 적어 둔 네 경우", "결과가 어느 방향이든 어떻게 쓸지")

    s.note(0.128, "잘 나온 학습 판만 골라 쓰는 일이 구조적으로 불가능하도록 설계했다.",
           width=70)
    return s


if __name__ == "__main__":
    print(save(build(), 11, "strategy"))
