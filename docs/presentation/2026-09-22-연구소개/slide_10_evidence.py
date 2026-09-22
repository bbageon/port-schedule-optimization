"""10쪽 — 비용이 터지던 날을 고친 쪽은 평범하게 넘겼다.

같은 실행 **안에서** 부하가 같은 두 날(3일째·7일째)을 견준다. 두 실행을 가로질러
비교하면 하루 비용의 정의가 미세하게 다르지만, 한 실행 안에서 견주면 그 문제가 없다.

숫자 출처
  · 수정 전: 등록된 여든 번 기록 `run-7e2fb14/months/21000000/NO_REALLOC/daily-final.jsonl`
  · 수정 후: `outputs/reports/yr317_v3_stall_diagnosis/fix-validation-9d/`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckstyle import AMBER, BLUE, FAINT, GREEN, INK, MUTED, RED, Slide, save

COLS = (
    dict(label="수정 전", color=RED, face="#FBF1F0", edge="#E7CBC8",
         day3="26억", day7="296억", verdict="같은 부하인데 11배로 뛰었다",
         stall="재생이 끝날 때까지 안 풀렸다", backlog="밀린 일 1,941건"),
    dict(label="수정 후", color=GREEN, face="#F1F7F5", edge="#C8DED7",
         day3="24억", day7="25억", verdict="3일째와 사실상 같다",
         stall="4.13일에 스스로 풀렸다", backlog="밀린 일 40건"),
)


def column(s, x, w, spec):
    s.box(x, 0.690, w, 0.058, face=spec["color"], edge="none", radius=0.010)
    s.text(x + w / 2, 0.719, spec["label"], size=14, weight="bold",
           color="#FFFFFF", ha="center")

    s.box(x, 0.330, w, 0.340, face=spec["face"], edge=spec["edge"])

    # 3일째 — 참조값
    s.text(x + w / 2, 0.615, "3일째", size=12, color=MUTED, ha="center")
    s.text(x + w / 2, 0.571, spec["day3"], size=20, weight="bold",
           color=MUTED, ha="center")

    s.ax.add_line(__import__("matplotlib.pyplot", fromlist=["Line2D"]).Line2D(
        [x + 0.030, x + w - 0.030], [0.531, 0.531], color=spec["edge"], lw=1.2))

    # 7일째 — 문제의 날
    s.text(x + w / 2, 0.492, "7일째", size=12, color=spec["color"], ha="center")
    s.text(x + w / 2, 0.425, spec["day7"], size=44, weight="bold",
           color=spec["color"], ha="center")
    s.text(x + w / 2, 0.369, spec["verdict"], size=12.5, weight="bold",
           color=spec["color"], ha="center")


def build():
    s = Slide("비용이 터지던 날을, 고친 쪽은 평범하게 넘겼다",
              "같은 달·같은 블록·같은 정책으로 두 번 굴렸다. 아래는 가장 붐비는 날 둘이다")

    w, gap = 0.395, 0.086
    left = 0.062
    for i, spec in enumerate(COLS):
        column(s, left + i * (w + gap), w, spec)
    s.text(left + w + gap / 2, 0.425, "→", size=26, color=INK, ha="center")

    # 멈춤 자체는 똑같이 생겼다
    s.box(0.062, 0.222, 0.876, 0.078, face="#F4F7FA", edge="#DCE5EC")
    s.text(0.084, 0.276, "멈춘 순간은 둘이 똑같다. 갈린 것은 빠져나왔느냐다.",
           size=13.5, weight="bold", color=INK)
    s.text(0.084, 0.243,
           f"둘 다 4.03일에 멈췄다. 수정 전은 {COLS[0]['stall']} ({COLS[0]['backlog']}). "
           f"수정 후는 {COLS[1]['stall']} ({COLS[1]['backlog']}).",
           size=12, color=MUTED)

    # 한계
    s.box(0.062, 0.128, 0.876, 0.072, face="#FCF8EF", edge=AMBER, lw=1.3)
    s.band(0.062, 0.128, 0.0045, 0.072, face=AMBER)
    s.text(0.084, 0.176, "이 수정은 교착에서만 작동하지 않는다. 후보 칸이 꽉 찬 모든 순간에 "
                         "고르는 일이 달라진다.", size=12, color=INK)
    s.text(0.084, 0.148, "그래서 고친 엔진의 결과는 기존 여든 번과 나란히 놓고 비교할 수 없다.",
           size=12, weight="bold", color="#8A6220")

    s.note(0.086, "각 칸의 두 숫자는 같은 실행 안에서 부하가 같은 두 날을 견준 것이다. "
                  "시드 하나·블록 하나 진단이며, 정책이 더 낫다는 증거가 아니다.", width=84)
    return s


if __name__ == "__main__":
    print(save(build(), 10, "evidence"))
