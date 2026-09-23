"""14쪽 — 날마다 물량이 다르다. 곡선 모양은 그대로 두고 높이만 바뀐다.

왼쪽은 부하 다섯 종의 같은 곡선, 오른쪽은 추첨 확률과 실제로 뽑힌 비율이다.
자료는 `curvedata.py` 가 코드와 시드 은행 기록에서 직접 읽는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curvedata as cd
from deckstyle import AMBER, BLUE, FAINT, INK, MUTED, RED, Slide, save

# 부하는 정체가 아니라 크기다 — 색은 한 색의 옅음→짙음 단계로 준다(무지개 금지).
# 검사: 인접 단계 ΔL 0.077 ≥ 0.06 · 가장 옅은 단계 대비 2.08 ≥ 2.0 (흰 바탕).
RAMP = ["#8FB9D6", "#659CC3", "#3F7EAD", "#26669A", "#12446B"]
PROB = "#26669A"     # 추첨 확률
REAL = "#B9822F"     # 실제 뽑힌 비율 — 둘의 색약 분리 ΔE 22.2 (목표 8 이상)
LEFT = (0.072, 0.348, 0.495, 0.395)
RIGHT = (0.665, 0.348, 0.273, 0.395)


def curves(s):
    """부하 다섯 종의 곡선 한 장 — 같은 모양, 다른 높이."""
    ax = s.fig.add_axes(LEFT)
    ceiling = cd.crane_ceiling()
    rows = list(cd.loads())
    tops = []
    for (load, _p, _name), color in zip(rows, RAMP):
        xs, ys = cd.curve(load)
        ax.plot(xs, ys, color=color, lw=2.0, solid_capstyle="round", zorder=4)
        tops.append(max(ys))
    top = max(tops)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, top * 1.14)

    # 이름표는 곡선 위가 아니라 왼쪽 위 빈 자리에 견본으로 둔다 — 새벽 구간은
    # 다섯 곡선이 바닥에 붙어 있어 글자를 놓을 데가 없다.
    for i, ((load, _p, name), color) in enumerate(zip(reversed(rows),
                                                      reversed(RAMP))):
        y = top * (1.06 - 0.072 * i)
        ax.plot([0.5, 1.7], [y, y], color=color, lw=2.4, solid_capstyle="round",
                zorder=6)
        ax.text(2.1, y, f"{name} {load:,}대", fontsize=10.5, color=INK,
                va="center", fontweight="bold", zorder=6)

    ax.axhline(ceiling, color=RED, lw=1.5, ls=(0, (5, 3)), zorder=3)
    ax.text(23.6, ceiling + top * 0.030, f"크레인 처리 상한 {ceiling:,.0f}대/h",
            fontsize=10.5, color=INK, fontweight="bold", ha="right")

    ax.set_xticks(range(0, 25, 6))
    ax.set_xticklabels([f"{h}시" for h in range(0, 25, 6)])
    ax.tick_params(labelsize=10.5, colors=MUTED, length=3)
    ax.grid(axis="y", color=FAINT, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    return ceiling


def lottery(s):
    """추첨 확률과 실제로 뽑힌 비율 — 가로 막대 두 벌."""
    ax = s.fig.add_axes(RIGHT)
    by_load, n_days = cd.realized()
    rows = list(cd.loads())
    h = 0.34
    for i, (load, prob, name) in enumerate(rows):
        y = len(rows) - 1 - i
        real = by_load[load][0] / n_days
        ax.barh(y - h / 2 - 0.02, prob, height=h, color=PROB, zorder=3)
        ax.barh(y + h / 2 + 0.02, real, height=h, color=REAL, zorder=3)
        ax.text(prob + 0.008, y - h / 2 - 0.02, f"{prob:.0%}", fontsize=10,
                color=INK, va="center", fontweight="bold")
        ax.text(real + 0.008, y + h / 2 + 0.02, f"{real:.1%}", fontsize=10,
                color=INK, va="center", fontweight="bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{load:,}" for load, _p, _n in reversed(rows)],
                       fontsize=10.5)
    ax.set_xlim(0, 0.40)
    ax.set_xticks([])
    ax.tick_params(colors=MUTED, length=0)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.40, -0.80)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(FAINT)

    # 범례 — 색은 네모 견본이 맡고 글자는 먹색으로 둔다
    for x0, color, label in ((0.0, PROB, "추첨 확률"), (0.175, REAL, "실제 뽑힌 비율")):
        ax.barh(-0.70, 0.024, left=x0, height=0.30, color=color, zorder=4,
                clip_on=False)
        ax.text(x0 + 0.032, -0.70, label, fontsize=11, color=INK,
                fontweight="bold", va="center")
    return n_days


def build():
    s = Slide("날마다 물량이 다르다 — 다섯 중 하나를 추첨한다",
              "곡선 모양은 그대로 두고 높이만 바뀐다")

    s.text(0.072, 0.788, "① 같은 곡선, 다섯 높이 (대/시간)", size=14.5, weight="bold")
    s.text(0.665, 0.788, "② 어느 높이가 걸릴지는 추첨", size=14.5, weight="bold")
    ceiling = curves(s)
    n_days = lottery(s)

    s.box(0.072, 0.168, 0.866, 0.098, face="#FBF5EC", edge="#E3CDA6", lw=1.4)
    s.band(0.072, 0.182, 0.006, 0.070, face=AMBER, zorder=3)
    s.text(0.100, 0.236, "이 그림이 말하는 것", size=12.5, weight="bold", color=AMBER)
    s.text(0.100, 0.198, "붐비는 날은 봉우리에서 크레인이 댈 수 있는 양을 넘는다 — "
                         "그래서 낮에 줄이 쌓이고 밤에 빠지는 무대가 된다.",
           size=13.5, color=INK)

    s.note(0.126, f"실제 비율은 {n_days}일({cd.contract()['n_days']}일 × 20달)을 추첨한 "
                  f"결과이며, 확률에 맞추려고 날을 바꾸지 않았다. {ceiling:,.0f}대/h 선은 "
                  f"크레인 42대(21블록 × 2대)가 쉬지 않고 3분에 한 대씩 트럭만 칠 때의 "
                  f"낙관적 상한이다 — 실제 작업시간은 190~410초이고 같은 크레인이 본선 "
                  f"작업도 한다.", width=90)
    return s


if __name__ == "__main__":
    print(save(build(), 14, "load"))
