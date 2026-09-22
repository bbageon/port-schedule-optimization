"""1쪽 — 표지.

글자만으로 세운 표지다. 제목 위에 파란 짧은 선 하나, 아래에 옅은 가로선과 날짜.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

X = 0.078


def build():
    s = Slide(cover=True)

    # 제목 위 파란 짧은 선
    s.band(X, 0.660, 0.090, 0.0075, face=BLUE)

    s.text(X, 0.552, "부산항 야드크레인 배정 재조정 강화학습",
           size=35, weight="bold", color=INK, va="center")
    s.text(X, 0.448, "연구 소개와 지금까지의 진행 상황",
           size=17, color=MUTED, va="center")

    # 아래 옅은 가로선과 날짜
    s.band(X, 0.212, 1 - 2 * X, 0.0016, face=FAINT)
    s.text(X, 0.150, "2026-09-22", size=12.5, color=MUTED, va="center")
    return s


if __name__ == "__main__":
    print(save(build(), 1, "cover"))
