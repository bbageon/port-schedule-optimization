"""결과 그림 셋 — 학습곡선 · 수요수준별 분해 · 날짜별 짝비교.

    PYTHONPATH=src python scripts/v3/fig_results.py

출력: docs/paper/v3/figures/final_figure/*.pdf (+ .png 600dpi 검토용)

■ 수치를 손으로 적지 않는다
  셋 다 원자료에서 직접 읽는다 — 학습곡선은 `outputs/v3/month-02/history.json`,
  나머지 둘은 `outputs/v3/judge-30d/arms/*.json` 과 `plan_month(9900950)` 의 부하다.
  표 2 를 옮겨 적으면 표와 그림이 조용히 어긋날 수 있어서 그렇게 하지 않는다.

■ 양식은 `figstyle.py` 한 곳에서 온다 (사용자 지시 2026-08-30)
  받은 명세의 7.16 in(IEEE 2단 전폭)만 4.80 in(LNCS 본문폭)으로 바꿨다. LNCS 본문에
  넣으면 7.16 in 은 67% 로 줄어 8~10 pt 규칙이 깨진다. 인코딩 규칙은 그대로다.

■ ★한 축에 겹쳐 그리지 않는다 (재설계 2026-08-30)
  탑티어 그림의 공통 습관은 "한 축은 한 가지 비교만 잰다" 이다. 앞판은 셋 다 이를
  어겨 읽히지 않았다:

    학습곡선  네 손실 + 다른 단위의 탐험률 ε 을 쌍축으로 겹침 → **3단 분리**
    분해막대  0.5억과 7.5억을 한 축에 → 작은 쪽이 0선에 눌림 → **부하 구간 2패널**
    짝비교    확대 그림을 본 그림 위에 끼워넣음(inset) → **나란한 2패널**

  세로축 단위가 다르거나 크기가 10배 이상 벌어지면 축을 나눈다. 나눈 뒤에는 어느
  패널인지 (a)(b) 로 표시하고, 범례는 그림에 하나만 둔다.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from figdata import month
from figstyle import (A_GREEN, BAND, BAR_MID, INK, LOSS_RED,
                      MARK_GREY, MUTED, P_BLUE, SMALL_PT, TEXTWIDTH_IN, apply,
                      hgrid, no_clip, panel, save)

OUT = ROOT / "docs/paper/v3/figures/final_figure"
HISTORY = ROOT / "outputs/v3/month-02/history.json"
#: ★대역은 **본문과 같아야 한다** — 그림만 먼저 옮기면 논문이 자기모순에 빠진다.
#:
#:  판정 대역(`judge-locked` · 9,400,000)으로 옮기려던 것을 **일시적으로 되돌린다.**
#:  그림이 판정 대역으로 앞서 나가 있는 동안 그림 5 는 "28/28 일 개선", 본문과 캡션은
#:  "26/28 일" 이라 한 논문 안에서 두 값이 부딪혔다. 본문·표의 205개 수치를 한 번에
#:  옮기기 전까지는 **그림도 진단 대역에 머문다.**
#:
#:  ⚠️ 옮기는 조건: `RL_TIME`·`RL_SPACE` 판정 실행이 끝나 `judge-locked` 에 여덟 팔이
#:     모두 들어오면, 아래 두 줄을 judge-locked·9,400,000 으로 되돌리고 **같은 커밋에서**
#:     본문·표 수치도 함께 옮긴다. 그림 셋은 원자료를 직접 읽으므로 두 줄이면 된다.
ARMS = ROOT / "outputs/v3/judge-30d/arms"
JUDGE_SEED = 9_900_950
HEAVY = 12_500             # 이 부하부터 "혼잡" — 학습회차 음영과 분해 패널의 경계

#: ★단위는 **백만원 하나로 통일**한다 (사용자 지시 2026-08-30).
#:  십억원으로 두면 원활한 수요 쪽이 0.02·0.04·0.06 이라 소수점만 읽게 된다.
#:  백만원이면 20·40·60 과 200·400·600 이 되어 두 패널 다 눈에 바로 들어온다.
#:  표는 억원(국문)·십억원(영문)을 쓰므로 **캡션이 환산을 한 줄로 적는다.**
MN = 1e6                   # 백만원 (million KRW)

#: 행동 막대 셋 — 셋 다 `재배치 없음` 대비라 **같은 문법**(막대)에 같은 파랑 계단이다.
#:
#: ★해칭은 하나에만 쓴다 (사용자 지시 2026-08-30). 파랑+사선·흰색+X·회색+사선이
#:  섞이면 작은 PDF 에서 오히려 안 갈린다. 색만으로 갈리지 않는 흰 막대 하나에만
#:  얇은 사선을 얹어 흑백 인쇄까지 버티게 한다.
BARS = (
    ("Both actions", P_BLUE, None),
    ("Timing only", BAR_MID, None),
    ("Block only", "white", "//"),
)

#: ★학습 기여는 **막대가 아니다.**
#:  앞 셋은 "어떤 행동을 허용했나" 의 결과인데, 이것은 `RL_EARLY − RL`
#:  (`--ckpt-early` = 학습 전 체크포인트) 즉 **효과의 분해값**이다. 같은 막대로 그리면
#:  독자가 "학습이라는 넷째 행동이 있나?" 로 읽는다. 그래서 ◆ + 세로선으로 문법을
#:  아예 바꾼다. 값은 표의 마지막 열 `학습의 몫`(Training contribution)과 같다.
MARK_LABEL = "Training contribution"


# ══════════════════════════ 원자료 ══════════════════════════
def load_history():
    h = json.loads(HISTORY.read_text(encoding="utf-8"))
    return list(range(1, len(h) + 1)), {
        "p_train": [r["seller_loss"] for r in h],
        "p_val": [r["val_seller_loss"] for r in h],
        "a_train": [r["buyer_loss"] for r in h],
        "a_val": [r["val_buyer_loss"] for r in h],
        "eps": [r["explore"] for r in h],
        "load": [r["load"] for r in h],
    }


def load_arms():
    arms = {}
    for f in ARMS.glob("arm_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        arms[d["arm"]] = {int(k): v for k, v in d["phi_by_day"].items()}
    load = {d.index: d.load for d in month().plan_month(JUDGE_SEED)}
    days = [i for i in sorted(arms["RL"]) if 1 <= i <= 28]
    return arms, load, days


# ══════════════════════════ ① 학습곡선 ══════════════════════════
def _roll(v, k=5):
    """중앙 이동평균 — 회차마다 수요가 달라 원자료가 크게 튄다.

    추세를 보이려고 원자료를 **지우지는 않는다.** 옅게 깔고 그 위에 평균을 얹는다.
    """
    out = []
    for i in range(len(v)):
        lo, hi = max(0, i - k // 2), min(len(v), i + k // 2 + 1)
        out.append(sum(v[lo:hi]) / (hi - lo))
    return out


def _heavy_bands(ax, ep, loads):
    """혼잡 회차에 옅은 띠 — 손실이 튀는 자리가 어디인지 먼저 말해 준다.

    회차마다 부하를 새로 뽑기 때문에 손실의 오르내림 상당수가 학습이 아니라
    **그날 뽑힌 수요**에서 온다. 띠가 없으면 17~24회차의 봉우리가 과적합처럼 보인다.
    """
    for e, L in zip(ep, loads):
        if L >= HEAVY:
            ax.axvspan(e - 0.5, e + 0.5, color=BAND, alpha=0.55, lw=0, zorder=0)


def fig_learning():
    ep, h = load_history()
    fig, axes = plt.subplots(
        3, 1, figsize=(TEXTWIDTH_IN, 4.30), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.38]})
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.10,
                                rect=(0, 0, 1, 1))
    ax_p, ax_a, ax_e = axes

    for ax, color, keys, name in (
            (ax_p, P_BLUE, ("p_train", "p_val"), "(a) Proposal network"),
            (ax_a, A_GREEN, ("a_train", "a_val"), "(b) Acceptance network")):
        _heavy_bands(ax, ep, h["load"])
        for key, ls in zip(keys, ("-", "--")):
            ax.plot(ep, h[key], ls, lw=0.7, color=color, alpha=0.30, zorder=2)
            ax.plot(ep, _roll(h[key]), ls, lw=1.6, color=color, zorder=3)
        # ★로그 축 — 손실이 0.002 에서 1.06 까지 세 자릿수를 오간다. 선형으로 두면
        #   위쪽 뾰족한 값을 자르거나(앞판이 그랬다) 아래쪽 절반이 0선에 눌린다.
        ax.set_yscale("log")
        ax.set_ylim(1.2e-3, 1.8)
        ax.set_yticks([0.01, 0.1, 1])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(which="minor", length=1.5, color=MUTED)
        no_clip(ax, [v for k in keys for v in h[k]], f"학습곡선 {name}")
        ax.set_ylabel("Huber loss")
        hgrid(ax)
        panel(ax, name)

    _heavy_bands(ax_e, ep, h["load"])
    ax_e.plot(ep, h["eps"], "-", lw=1.4, color=MUTED, zorder=3)
    ax_e.set_ylim(0, 0.58)
    no_clip(ax_e, h["eps"], "학습곡선 (c)")
    ax_e.set_yticks([0.0, 0.25, 0.5])
    ax_e.set_ylabel(r"$\varepsilon$")
    hgrid(ax_e)

    #  패널 이름은 축 이름 **뒤에** 붙인다 — `figstyle.panel` 이 축 이름 아랫줄로
    #  넣기 때문에, 먼저 부르면 `set_xlabel` 이 이름을 덮어쓴다.
    ax_e.set_xlabel("Training epoch")
    panel(ax_e, "(c) Exploration schedule")
    ax_e.set_xlim(0.5, len(ep) + 0.5)
    ax_e.set_xticks([1, 5, 10, 15, 20, 25, 30])

    # 범례는 그림에 하나 — 색은 패널이 이미 말하므로 회색 견본으로 **선모양**만 알린다.
    handles = [
        Line2D([], [], color=MUTED, lw=0.8, alpha=0.55),
        Line2D([], [], color=MUTED, lw=1.6, ls="-"),
        Line2D([], [], color=MUTED, lw=1.6, ls="--"),
        Patch(facecolor=BAND, alpha=0.55, lw=0),
    ]
    labels = ["per epoch", "train (5-epoch mean)", "validation (5-epoch mean)",
              f"demand $\\geq$ {HEAVY:,}"]
    fig.legend(handles, labels, ncol=2, loc="outside upper center",
               handlelength=1.9, columnspacing=1.4)

    save(fig, OUT, "fig-learning-curve", ROOT)


# ══════════════════════ ② 수요수준별 분해 ══════════════════════
def fig_decomposition():
    arms, load, days = load_arms()
    levels = sorted({load[i] for i in days})
    n_days = {L: sum(1 for i in days if load[i] == L) for L in levels}

    def agg(a, b):
        """a − b 를 부하별로 합산 (백만원). 양수 = 비용이 줄었다."""
        return {L: sum(arms[a][i] - arms[b][i] for i in days if load[i] == L) / MN
                for L in levels}

    need = ("RL_TIME", "RL_SPACE")
    if any(k not in arms for k in need):
        print(f"  (skipped) decomposition needs {need}; not run yet")
        return None, None
    series = (agg("NO_REALLOC", "RL"), agg("NO_REALLOC", "RL_TIME"),
              agg("NO_REALLOC", "RL_SPACE"))
    train = agg("RL_EARLY", "RL")          # 막대가 아니다 — ◆ 로 그린다
    total = sum(series[0].values())
    time_share = sum(series[1].values()) / total

    # ★한 패널로 합친다 (사용자 지시 2026-08-30) — 다섯 수요 수준은 **같은 축 하나**의
    #   연속된 눈금이다. 둘로 쪼개면 원활한 쪽과 혼잡한 쪽이 서로 다른 실험처럼 읽힌다.
    #   대신 혼잡 구간에 음영을 깔아, 쪼개기가 나르던 "비혼잡 → 혼잡 전환"을 유지한다.
    #   ⚠️ 값의 폭이 15배(47 vs 747백만원)라 원활한 쪽 막대는 작게 찍힌다. 그래서
    #      짧은 막대에는 숫자를 함께 적는다 — 부호와 크기를 표 없이도 읽게.
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, 2.86),
                           layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02)

    w = 0.185
    offs = [-w, 0.0, w]                    # 막대 셋
    mark_off = 0.42                        # ◆ — 막대 무리에서 확실히 떨어뜨린다
    xs = list(range(len(levels)))

    # 혼잡 구간 음영 — 축을 나누는 대신 배경으로 구간을 말한다.
    heavy_x = [x for x, L in zip(xs, levels) if L >= HEAVY]
    if heavy_x:
        ax.axvspan(min(heavy_x) - 0.5 + offs[0], max(heavy_x) + mark_off + 0.20,
                   color=BAND, alpha=0.55, lw=0, zorder=0)

    bars = []
    for k, (vals, (label, fc, hatch)) in enumerate(zip(series, BARS)):
        bars.append(ax.bar([x + offs[k] for x in xs], [vals[L] for L in levels],
                           w, facecolor=fc, edgecolor=INK, linewidth=0.7,
                           hatch=hatch, label=label, zorder=3))

    # ◆ + 세로선 — 막대와 **다른 문법**이라 "넷째 행동" 으로 안 읽힌다.
    mx = [x + mark_off for x in xs]
    mv = [train[L] for L in levels]
    ax.vlines(mx, 0, mv, color=MARK_GREY, lw=1.0, zorder=3)
    mark = ax.scatter(mx, mv, s=26, marker="D", facecolor=MARK_GREY,
                      edgecolor="white", linewidth=0.5, zorder=4,
                      label=MARK_LABEL)

    vals_all = [v[L] for v in series for L in levels] + mv
    lo, hi = min(vals_all), max(vals_all)
    ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
    no_clip(ax, vals_all, "분해")

    # 짧은 막대에만 값을 적는다 — 긴 막대는 축만으로 읽히므로 숫자를 얹으면 어수선하다.
    span = hi - lo
    for k, vals in enumerate(series):
        for x, L in zip(xs, levels):
            v = vals[L]
            if abs(v) < 0.12 * span:
                ax.text(x + offs[k], v + (0.018 * span if v >= 0 else -0.018 * span),
                        f"{v:.0f}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=6.5, color=INK, zorder=5)

    # ★0 선이 이 그림의 핵심이다 — 7,500 의 학습 기여도, 15,000 의 블록만도 음수다.
    #   그래서 격자는 아주 옅게, 0 선만 굵게, 아래 테두리는 아예 없앤다.
    hgrid(ax)
    ax.axhline(0, lw=1.1, color=INK, zorder=5)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{L:,}\n$n$ = {n_days[L]}" for L in levels])
    ax.set_xlim(offs[0] - w / 2 - 0.20, len(levels) - 1 + mark_off + 0.20)
    ax.text((min(heavy_x) + max(heavy_x)) / 2 + 0.1, hi + 0.055 * span,
            "congested", ha="center", va="bottom", fontsize=SMALL_PT,
            color=MUTED, zorder=5)
    ax.set_ylabel("Cost reduction (million KRW)")
    ax.set_xlabel("Daily external-truck demand (jobs/day)")

    # 범례는 한 줄로 좁게 둔다. 빈 칸을 끼워 넣으면 범례가 그림 폭 전체로 벌어져
    # 맨 오른쪽 항목이 (b) 패널의 머리말처럼 읽힌다. ◆ 라는 **다른 기호**가 이미
    # 갈라 주므로 억지 간격은 필요 없다.
    fig.legend(list(bars) + [mark],
               [b.get_label() for b in bars] + [MARK_LABEL],
               ncol=4, loc="outside upper center", handlelength=1.3,
               columnspacing=1.1, handletextpad=0.45)

    save(fig, OUT, "fig-decomposition", ROOT)
    return total, time_share


# ═════════════════ ③ 날짜별 짝비교 (정렬 롤리팝) ═════════════════
def _lolli(ax, rank, delta, s=16, lw=0.9):
    for x, v in zip(rank, delta):
        ax.plot([x, x], [0, v], lw=lw, alpha=0.65, zorder=2,
                color=P_BLUE if v > 0 else LOSS_RED)
    for keep, color, mark in ((True, P_BLUE, "o"), (False, LOSS_RED, "s")):
        pts = [(x, v) for x, v in zip(rank, delta) if (v > 0) is keep]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=s, zorder=3,
                       color=color, marker=mark, edgecolor="none")
    ax.axhline(0, lw=0.9, color=INK, zorder=4)


def _span(vals, pad_lo=0.10, pad_hi=0.08):
    """0 선을 반드시 품는 y 범위.

    지는 날이 하나도 없으면 min(vals) 가 양수라, 그것을 하한으로 쓰면 0 선과
    가장 작은 막대가 잘린다 (판정 대역 9,400,000 이 실제로 그렇다).
    """
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    span = hi - lo or 1.0
    return lo - span * pad_lo, hi + span * pad_hi


def fig_paired():
    arms, _, days = load_arms()
    delta = sorted((arms["NO_REALLOC"][i] - arms["RL"][i]) / MN for i in days)
    rank = list(range(1, len(delta) + 1))
    win = sum(1 for d in delta if d > 0)
    cut = len(delta) - 4          # 상위 4일이 축을 지배한다 — 나머지는 (b) 에서 본다

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.62), layout="constrained",
        gridspec_kw={"width_ratios": [1, 1]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)

    # ── (a) 28일 전부
    _lolli(ax_a, rank, delta, s=12)
    ax_a.axvspan(0.4, cut + 0.6, color=BAND, alpha=0.45, lw=0, zorder=0)
    ax_a.set_xlim(0.2, len(delta) + 0.8)
    ax_a.set_xticks([1, 10, 20, 28])
    ax_a.set_ylim(*_span(delta))
    no_clip(ax_a, delta, "짝비교 (a)")
    hgrid(ax_a)
    panel(ax_a, "(a) All 28 days")
    ax_a.text(0.05, 0.97,
              f"{win} / {len(delta)} days improved\n"
              "two-sided sign test, $p<0.001$",
              transform=ax_a.transAxes, ha="left", va="top", color=INK,
              linespacing=1.35)
    ax_a.text((cut + 1) / 2, _span(delta)[0] * 0.55, "shown in (b)", ha="center", va="center",
              fontsize=7.5, color=MUTED)

    # ── (b) ★확대 — 네 날이 축을 지배해서 나머지 24일이 0선에 뭉갠다. 26/28 이라는
    #        주장은 저 작은 값들의 **부호**에 달려 있으므로 축을 따로 준다.
    _lolli(ax_b, rank[:cut], delta[:cut], s=13, lw=0.9)
    ax_b.set_xlim(0.2, cut + 0.8)
    ax_b.set_xticks([1, 10, 20, 24])
    ax_b.set_ylim(*_span(delta[:cut], pad_hi=0.17))
    no_clip(ax_b, delta[:cut], "짝비교 (b)")
    hgrid(ax_b)
    panel(ax_b, f"(b) Ranks 1–{cut}, enlarged")

    fig.supylabel("Cost reduction (KRW m)",
                  fontsize=plt.rcParams["axes.labelsize"], color=INK)
    fig.supxlabel("Operating days ranked by paired cost difference",
                  fontsize=plt.rcParams["axes.labelsize"], color=INK)

    fig.legend([Line2D([], [], color=P_BLUE, marker="o", ls="-", lw=1.0, ms=3.6),
                Line2D([], [], color=LOSS_RED, marker="s", ls="-", lw=1.0, ms=3.4)],
               ["RL cheaper", "RL costlier"], ncol=2,
               loc="outside upper center", handlelength=1.6, columnspacing=1.8)

    save(fig, OUT, "fig-paired-daily", ROOT)
    return win, len(delta)


if __name__ == "__main__":
    apply()
    print("result figures (LNCS text width 4.80 in, serif 8-10 pt)")
    fig_learning()
    total, share = fig_decomposition()
    win, n = fig_paired()
    print()
    print("[figure values read from raw data - cross-check the table]")
    if total is not None:
        print(f"  decomposition total {total:.0f} million KRW, timing share {share:.1%}")
    print(f"  paired: {win}/{n} days improved")
