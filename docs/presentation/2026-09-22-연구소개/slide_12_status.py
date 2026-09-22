"""12쪽 — 진행 상황 한 판: 끝난 것 · 지금 도는 것 · 남은 것.

칸 높이는 셋이 같고, 줄 수가 적은 칸은 항목 사이를 벌려 위아래 가운데에 놓는다.
그래야 '할 일이 적은 칸'이 빈 상자로 보이지 않는다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import BLUE, FAINT, GREEN, INK, MUTED, RED, AMBER, Slide, save

COL_W = 0.272
COL_X = (0.062, 0.364, 0.666)

HEAD_Y, HEAD_H = 0.735, 0.050
BODY_Y, BODY_H = 0.225, 0.500

WRAP = 16
SIZE = 12
LEADING = 0.033
PAD = 0.050
GAP_MIN, GAP_MAX = 0.028, 0.078


def line_count(text, width):
    """deckstyle.wrap 과 같은 방식으로 줄 수를 미리 센다 (칸 배치를 정하려고)."""
    words, line, out = text.split(), "", []
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) > width and line:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return len(out)


def column(s, x, head, color, tint, edge, items):
    s.box(x, BODY_Y, COL_W, BODY_H, face="#FCFCFD", edge=FAINT, lw=1.4)
    s.box(x, HEAD_Y, COL_W, HEAD_H, face=tint, edge=edge, lw=1.4, zorder=2)
    s.text(x + COL_W / 2, HEAD_Y + HEAD_H / 2, head, size=14.5, color=color,
           weight="bold", ha="center")

    counts = [line_count(item, WRAP) for item in items]
    content = sum(counts) * LEADING
    slack = (BODY_H - 2 * PAD - content) / max(len(items) - 1, 1)
    gap = min(GAP_MAX, max(GAP_MIN, slack)) if len(items) > 1 else 0.0
    block = content + gap * (len(items) - 1)

    y = BODY_Y + BODY_H - (BODY_H - block) / 2 - 0.018
    for item, count in zip(items, counts):
        s.ax.add_patch(plt.Circle((x + 0.028, y), 0.0045, color=color, zorder=3))
        s.wrap(x + 0.046, y, item, WRAP, size=SIZE, color=INK, leading=LEADING)
        y -= count * LEADING + gap


def build():
    s = Slide("지금 어디까지 왔는가", "2026년 9월 22일 기준")

    column(s, COL_X[0], "끝난 것", GREEN, "#EAF2EF", "#BFD9D0", [
        "원고 열두 쪽 정정과 제출 검사 통과",
        "시뮬레이터 결함 원인 규명과 수정, 시험 422건 통과",
        "새 시드 스무 달과 공급 보정 완료",
        "확증 실험 사전등록 작성",
    ])

    column(s, COL_X[1], "지금 도는 것", AMBER, "#FAF3E6", "#E3CDA2", [
        "학습 세 판 (하루쯤)",
        "수정 전 엔진 재생 (대조군)",
    ])

    column(s, COL_X[2], "남은 것", MUTED, "#F2F4F6", FAINT, [
        "확증 실험 160판",
        "결과대로 논문과 답변서 다시 쓰기",
    ])

    s.note(0.128, "사전등록은 결과를 보기 전에 고정했고, 실행이 끝난 뒤에는 원문을 고치지 않는다.",
           width=70)
    return s


if __name__ == "__main__":
    print(save(build(), 12, "status"))
