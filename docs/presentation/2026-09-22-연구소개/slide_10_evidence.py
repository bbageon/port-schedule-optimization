"""10쪽 — 갈리는 것은 멈춤이 아니라 풀림이다.

⚠️ 한때 이 장에 "수정 전 7일째 296억 vs 수정 후 25억" 을 실었다. **틀렸다** —
앞은 서른 날을 다 돈 실행의 값이고 뒤는 아흐레에서 끊은 재생의 값이다. 못 끝낸
트럭의 대기비용은 남은 날들에 걸쳐 쌓이므로 아흐레에서 끊으면 그 누적이 없다.
같은 아흐레끼리 견주면 7일째는 26.1억 대 24.8억으로 비슷하다.

그래서 이 장은 **비용이 아니라 풀렸느냐**를 말한다. 비용 판정은 서른 날을 다 도는
확증 실험에서 한다.

출처: `outputs/reports/yr317_v3_stall_diagnosis/fix-validation-9d/comparison.md`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import AMBER, FAINT, GREEN, INK, MUTED, RED, Slide, save

COLS = (
    dict(label="수정 전", color=RED, face="#FBF1F0", edge="#E7CBC8",
         big="3,120건", small="마지막 순간 밀린 일",
         verdict="재생이 끝나는 아흐레까지\n한 번도 못 빠져나왔다",
         time="6.9시간"),
    dict(label="수정 후", color=GREEN, face="#F1F7F5", edge="#C8DED7",
         big="42건", small="마지막 순간 밀린 일",
         verdict="4.13일에 스스로 빠져나온 뒤\n다시 멈추지 않았다",
         time="2.3시간"),
)


def column(s, x, w, spec):
    s.box(x, 0.672, w, 0.056, face=spec["color"], edge="none", radius=0.010)
    s.text(x + w / 2, 0.700, spec["label"], size=14, weight="bold",
           color="#FFFFFF", ha="center")

    s.box(x, 0.330, w, 0.322, face=spec["face"], edge=spec["edge"])
    for i, line in enumerate(spec["verdict"].split("\n")):
        s.text(x + w / 2, 0.606 - i * 0.036, line, size=13, weight="bold",
               color=spec["color"], ha="center")

    s.ax.add_line(__import__("matplotlib.pyplot", fromlist=["Line2D"]).Line2D(
        [x + 0.030, x + w - 0.030], [0.523, 0.523], color=spec["edge"], lw=1.2))

    s.text(x + w / 2, 0.446, spec["big"], size=40, weight="bold",
           color=spec["color"], ha="center")
    s.text(x + w / 2, 0.390, spec["small"], size=12, color=MUTED, ha="center")
    s.text(x + w / 2, 0.355, f"같은 아흐레를 도는 데 {spec['time']}", size=11.5,
           color=MUTED, ha="center")


def build():
    s = Slide("둘 다 같은 날 멈췄다. 갈린 것은 빠져나왔느냐다",
              "같은 달·같은 블록·같은 정책으로 아흐레까지 두 번 굴려 비교했다")

    w, gap = 0.395, 0.086
    left = 0.062
    for i, spec in enumerate(COLS):
        column(s, left + i * (w + gap), w, spec)
    s.text(left + w + gap / 2, 0.446, "→", size=26, color=INK, ha="center")

    s.box(0.062, 0.232, 0.876, 0.070, face="#F4F7FA", edge="#DCE5EC")
    s.text(0.084, 0.280, "수정은 멈춤을 막지 않는다. 나올 문을 열어 준다.",
           size=13.5, weight="bold", color=INK)
    s.text(0.084, 0.250, "둘 다 4.03일에 똑같이 멈췄다. 멈춘 블록은 노는 것이 아니라 "
                         "못 하는 일 수천 건을 매 분 다시 계산한다.", size=12, color=MUTED)

    s.box(0.062, 0.128, 0.876, 0.082, face="#FCF8EF", edge=AMBER, lw=1.3)
    s.band(0.062, 0.128, 0.0045, 0.082, face=AMBER)
    s.text(0.084, 0.184, "비용으로는 아직 판정하지 않는다.", size=12.5,
           weight="bold", color="#8A6220")
    s.text(0.084, 0.155, "못 끝낸 트럭의 대기비용은 남은 날들에 걸쳐 쌓인다. 아흐레에서 "
                         "끊으면 그 누적이 안 보인다.", size=11.5, color=INK)

    s.note(0.086, "게다가 이 수정은 공짜가 아니다. 멈추기 전 며칠은 수정 후가 오히려 조금 "
                  "비쌌다. 서른 날 판정은 확증 실험에서 한다.", width=84)
    return s


if __name__ == "__main__":
    print(save(build(), 10, "evidence"))
