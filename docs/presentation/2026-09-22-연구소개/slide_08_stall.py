"""8쪽 — 여든 번 가운데 아홉 번이 도중에 멈췄다. 기준 실행에서도 멈췄다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

STATS = [
    ("9", "멈춘 실행 (여섯 개 달에서)"),
    ("23.5일", "가장 오래 안 풀린 멈춤"),
    ("5~10배", "멈춘 달의 비용"),
]

LINES = [
    "아무것도 바꾸지 않는 기준 실행에서도 멈췄다. 그러니 정책이 나빠서가 아니다.",
    "환경이 멈춘 것이다. 멈춘 달의 값은 정책 비교에 쓸 수 없다.",
]

X, W = 0.062, 0.876


def build():
    s = Slide("그런데 여든 번 가운데 아홉 번이 도중에 멈췄다",
              "그리고 아무것도 바꾸지 않는 기준 실행에서도 멈췄다")

    # ---------------------------------------------------------- 숫자 셋
    s.box(X, 0.498, W, 0.240, face="#FCFCFD")
    for i, (value, label) in enumerate(STATS):
        x = X + W * (2 * i + 1) / 6
        s.stat(x, 0.655, value, label, color=RED, size=42)
        if i:
            s.band(X + W * i / 3, 0.545, 0.0015, 0.148, face=FAINT, zorder=3)

    # ---------------------------------------------------------- 그래서 무슨 뜻인가
    for i, line in enumerate(LINES):
        y = 0.350 - i * 0.108
        last = i == len(LINES) - 1      # 마지막 줄이 이 장의 결론이다
        s.box(X, y, W, 0.090,
              face="#FBF1F0" if last else "#FFFFFF",
              edge="#E7CBC8" if last else FAINT)
        s.bullet(y + 0.045, line, x=X + 0.022, size=14,
                 weight="bold" if last else "normal")

    s.note(0.140, "정책은 트럭의 시각과 블록만 정한다. "
                  "크레인에게 다음 일을 주는 규칙은 고정되어 있다.", width=80)
    return s


if __name__ == "__main__":
    print(save(build(), 8, "stall"))
