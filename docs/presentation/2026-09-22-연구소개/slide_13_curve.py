"""13쪽 — 시드를 만들 때 쓰는 하루 도착 곡선.

선은 시뮬레이터의 `diurnal_rate`, 점은 확증 캠페인용으로 실제 생성된 600일 가운데
이 물량인 날들의 시간대별 평균이다. 둘 다 `curvedata.py` 가 읽어 온다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curvedata as cd
from deckstyle import AMBER, BLUE, FAINT, GREEN, INK, MUTED, Slide, save

LOAD = 7_500                      # 설계 기준값 — 다섯 부하 중 가운데
LINE = "#26669A"                  # 덱 파랑을 채도 하한(OKLCH C≥0.10)에 맞춰 스냅한 값
FILL = "#E4EDF4"
FLOOR_FILL = "#CBD9E4"
BOX_X, BOX_W = 0.700, 0.238
AX = (0.075, 0.300, 0.585, 0.450)   # 왼쪽 아래 x·y, 폭, 높이 (그림 비율)


def chart(s):
    """도착률 한 장 — 채운 곡선 하나에 바닥선과 실제 점을 얹는다."""
    ax = s.fig.add_axes(AX)
    xs, ys = cd.curve(LOAD)
    floor = cd.night_floor(LOAD)

    ax.fill_between(xs, 0, ys, color=FILL, zorder=1)
    ax.fill_between(xs, 0, floor, color=FLOOR_FILL, zorder=2)
    ax.plot(xs, ys, color=LINE, lw=2.0, zorder=4, solid_capstyle="round")
    ax.axhline(floor, color=AMBER, lw=1.6, ls=(0, (5, 3)), zorder=5)

    by_load, _ = cd.realized()
    days, profile = by_load[LOAD]
    ax.scatter([h + 0.5 for h in range(24)], profile, s=26, color=LINE,
               edgecolor="#FFFFFF", linewidth=1.0, zorder=6)

    # 봉우리 셋 — 중심 시각에 세로 눈금과 몫
    for mu, _sigma, weight in cd.contract()["peaks"]:
        y = max(y for x, y in zip(xs, ys) if abs(x - mu) < 0.05)
        ax.plot([mu, mu], [0, y], color=MUTED, lw=0.9, ls=":", zorder=3)
        ax.annotate(f"{int(mu)}시\n{weight:.1%}", (mu, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10.5, color=INK,
                    linespacing=1.35)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(ys) * 1.34)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h}시" for h in range(0, 25, 3)])
    ax.tick_params(labelsize=10.5, colors=MUTED, length=3)
    ax.grid(axis="y", color=FAINT, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    # 단위는 세로로 눕히지 않는다 — 한글은 눕히면 읽기 나쁘다
    ax.text(0, max(ys) * 1.38, "대 / 시간", fontsize=10.5, color=MUTED,
            ha="left", va="bottom")

    # 직접 이름표 — 범례 상자를 따로 두지 않는다
    # 이름표 — 끌개선 없이 왼쪽 위 빈 곳에 견본을 직접 그린다.
    # 끌개선을 쓰면 곡선을 가로질러 글자와 겹쳤다(한 번 겪고 이렇게 바꿨다).
    top = max(ys) * 1.34
    ax.plot([0.55, 1.75], [top * 0.895] * 2, color=LINE, lw=2.0,
            solid_capstyle="round", clip_on=False)
    ax.text(2.1, top * 0.895, "설계 곡선", fontsize=11.5, color=INK,
            fontweight="bold", va="center")
    ax.scatter([1.15], [top * 0.805], s=26, color=LINE, edgecolor="#FFFFFF",
               linewidth=1.0, clip_on=False)
    ax.text(2.1, top * 0.805, f"실제 생성된 {days}일 평균", fontsize=11,
            color=MUTED, va="center")
    ax.annotate(f"바닥 {floor:.0f}대/h", (0.4, floor), textcoords="offset points",
                xytext=(2, -17), fontsize=10.5, color=INK, fontweight="bold")
    return floor


def side(s, y, accent, head, body):
    """오른쪽 설명 칸 하나."""
    s.box(BOX_X, y, BOX_W, 0.137, face="#FCFCFD")
    s.text(BOX_X + 0.020, y + 0.100, head, size=13.5, weight="bold", color=accent)
    s.band(BOX_X + 0.020, y + 0.078, 0.042, 0.0035, face=accent)
    s.wrap(BOX_X + 0.020, y + 0.050, body, 21, size=11, color=INK, leading=0.031)


def build():
    s = Slide("하루 도착 곡선 — 24시간 깔린 바닥 위에 봉우리 셋",
              "시드가 바꾸는 것은 곡선이 아니라 그 위에 찍히는 점이다")

    floor = chart(s)
    peaks = cd.contract()["peaks"]
    night = cd.contract()["night_fraction"]

    side(s, 0.612, BLUE, "봉우리 셋",
         f"오전 {int(peaks[0][0])}시 · 오후 {int(peaks[1][0])}시 · 저녁 "
         f"{int(peaks[2][0])}시. 봉우리 몫을 {peaks[0][2]:.1%} / {peaks[1][2]:.1%} / "
         f"{peaks[2][2]:.1%} 로 나눈다.")
    side(s, 0.456, AMBER, f"바닥 {night:.0%}",
         f"하루 물량의 {night:.0%}는 24시간에 고르게 깔린다({floor:.0f}대/h). "
         "게이트가 밤에도 돌기 때문이다.")
    side(s, 0.300, GREEN, "같은 곡선, 다른 점",
         "시드는 모양을 바꾸지 않는다. 1분 칸 안에서 도착 시각만 다시 뽑는다.")

    s.note(0.205, f"그림은 하루 {LOAD:,}대인 날이다. 선은 시뮬레이터의 도착률 함수이고 "
                  f"점은 확증 캠페인용으로 생성된 600일 가운데 이 물량인 날들의 시간대별 "
                  f"평균이다. 둘은 시간당 0.1대 안에서 겹친다 — 생성기가 곡선을 그대로 "
                  f"따른다는 뜻이다.", width=84)
    return s


if __name__ == "__main__":
    print(save(build(), 13, "curve"))
