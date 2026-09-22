"""2쪽 — 한 장 요약.

세 칸이 나란히 선다: 무엇을 하는가 / 무엇을 알아냈는가 / 지금 무엇을 하는가.
숫자는 일부러 넣지 않았다 — 이 장은 뒤 장들의 길잡이다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

PANEL_TOP = 0.745
PANEL_BOTTOM = 0.305
PANEL_W = 0.276
PANEL_GAP = 0.024
PAD = 0.020
BODY_TOP = 0.598
LEADING = 0.044

XS = [0.062 + i * (PANEL_W + PANEL_GAP) for i in range(3)]


def panel(s, x, accent, heading, blocks, gap=0.030):
    """칸 하나 — 머리글 한 줄, 그 아래 짧은 문장 두셋."""
    s.box(x, PANEL_BOTTOM, PANEL_W, PANEL_TOP - PANEL_BOTTOM, face="#FCFCFD")
    s.text(x + PAD, 0.688, heading, size=14, weight="bold", color=accent)
    s.band(x + PAD, 0.660, 0.052, 0.0035, face=accent)

    y = BODY_TOP
    for text, color, weight, width in blocks:
        n = s.wrap(x + PAD, y, text, width, size=12, color=color,
                   weight=weight, leading=LEADING)
        y -= n * LEADING + gap
    return y


def build():
    s = Slide("한 장 요약", "무엇을 하는 연구이고, 지금 어디까지 왔는가")

    panel(s, XS[0], BLUE, "① 무엇을 하는가", [
        ("터미널에 오는 트럭의 예약 시각과 갈 블록을 학습된 정책이 다시 조정한다.",
         INK, "normal", 16),
        ("목표는 터미널 비용을 줄이는 것이다.", MUTED, "normal", 16),
    ], gap=0.060)

    panel(s, XS[1], GREEN, "② 무엇을 알아냈는가", [
        ("시간만 조정하면 스무 달 중 열여섯 달에서 더 쌌다.", GREEN, "bold", 15),
        ("블록까지 바꾸면 오히려 비쌌다.", RED, "normal", 16),
        ("공간 조정은 재현되지 않았다.", MUTED, "normal", 16),
    ])

    panel(s, XS[2], AMBER, "③ 지금 무엇을 하는가", [
        ("심사 대응 중이다.", INK, "normal", 16),
        ("시뮬레이터 결함 하나를 찾아 고쳤다.", MUTED, "normal", 16),
        ("규칙과 견주는 확증 실험을 준비해 두었다.", MUTED, "normal", 16),
    ])

    s.note(0.185, "이 연구의 목표는 실제 터미널 적용이 아니라 강화학습이 효과가 "
                  "있는지 검증하는 것이다.", width=62)
    return s


if __name__ == "__main__":
    print(save(build(), 2, "summary"))
