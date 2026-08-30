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

import importlib.util
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

from figstyle import (A_GREEN, BAND, BAR_GREY, BAR_MID, BAR_PALE, INK,
                      LOSS_RED, MUTED, P_BLUE, TEXTWIDTH_IN, apply, hgrid,
                      panel, save)

OUT = ROOT / "docs/paper/v3/figures/final_figure"
HISTORY = ROOT / "outputs/v3/month-02/history.json"
ARMS = ROOT / "outputs/v3/judge-30d/arms"
JUDGE_SEED = 9_900_950
HEAVY = 12_500             # 이 부하부터 "혼잡" — 학습회차 음영과 분해 패널의 경계

#: 행동 분해 막대 — 색과 해칭을 **함께** 준다 (흑백에서도 갈리게).
#: 앞 셋은 `재배치 없음` 대비, 마지막 하나는 `이른순 배정` 대비다. 기준선이 다르므로
#: 파랑 계단에서 떼어내 회색으로 두고 이름에 대비 상대를 적는다.
BARS = (
    ("Both actions", P_BLUE, None),
    ("Timing only", BAR_MID, "//"),
    ("Block only", BAR_PALE, "xx"),
    ("Learned vs. earliest-first", BAR_GREY, "\\\\"),
)


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


def _plan_month():
    """부하 계획만 읽는다 — `torch` 없이도 그림이 그려지도록 파일에서 직접 든다.

    `yard_rl.v3.stage` 를 통하면 패키지 __init__ 이 actors → torch 까지 끌고 온다.
    그림은 학습을 하지 않으므로 그 무게를 질 이유가 없다. 계획 규칙 자체는 같은
    `month.py` 한 파일에서 오므로 값이 갈릴 여지는 없다.
    """
    path = ROOT / "src/yard_rl/v3/stage/month.py"
    spec = importlib.util.spec_from_file_location("_month_plan", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_month_plan"] = mod
    spec.loader.exec_module(mod)
    return mod.plan_month


def load_arms():
    arms = {}
    for f in ARMS.glob("arm_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        arms[d["arm"]] = {int(k): v for k, v in d["phi_by_day"].items()}
    load = {d.index: d.load for d in _plan_month()(JUDGE_SEED)}
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
        3, 1, figsize=(TEXTWIDTH_IN, 3.95), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.38]})
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.06,
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
        ax.set_ylim(1.7e-3, 1.5)
        ax.set_yticks([0.01, 0.1, 1])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(which="minor", length=1.5, color=MUTED)
        ax.set_ylabel("Huber loss")
        hgrid(ax)
        panel(ax, name)

    _heavy_bands(ax_e, ep, h["load"])
    ax_e.plot(ep, h["eps"], "-", lw=1.4, color=MUTED, zorder=3)
    ax_e.set_ylim(0, 0.58)
    ax_e.set_yticks([0.0, 0.25, 0.5])
    ax_e.set_ylabel(r"$\varepsilon$")
    hgrid(ax_e)
    panel(ax_e, "(c) Exploration schedule")

    ax_e.set_xlabel("Training epoch")
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
        """a − b 를 부하별로 합산 (억원). 양수 = 비용이 줄었다."""
        return {L: sum(arms[a][i] - arms[b][i] for i in days if load[i] == L) / 1e8
                for L in levels}

    series = (agg("NO_REALLOC", "RL"), agg("NO_REALLOC", "RL_TIME"),
              agg("NO_REALLOC", "RL_SPACE"), agg("RL_EARLY", "RL"))
    total = sum(series[0].values())
    time_share = sum(series[1].values()) / total

    # ★부하 구간으로 축을 나눈다 — 원활한 날의 절감(0.5억)과 초혼잡한 날의 절감
    #   (7.5억)은 15배 차이다. 한 축에 두면 앞의 셋이 0선에 눌려 부호조차 안 보인다.
    light = [L for L in levels if L < HEAVY]
    heavy = [L for L in levels if L >= HEAVY]

    fig, (ax_l, ax_h) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.65), layout="constrained",
        gridspec_kw={"width_ratios": [len(light), len(heavy)]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)

    # 넷째 막대는 **기준선이 다르다** — 앞 셋에서 반 칸 띄워 눈이 먼저 알아채게 한다.
    w = 0.18
    offs = [(k - 1.5) * w + (0.35 * w if k == 3 else 0.0) for k in range(len(BARS))]
    mid = sum(offs) / len(offs)
    bars = []
    for ax, group, name in ((ax_l, light, "(a) Light to moderate demand"),
                            (ax_h, heavy, "(b) Congested demand")):
        xs = list(range(len(group)))
        for k, (vals, (label, fc, hatch)) in enumerate(zip(series, BARS)):
            b = ax.bar([x + offs[k] for x in xs], [vals[L] for L in group],
                       w, facecolor=fc, edgecolor=INK, linewidth=0.7,
                       hatch=hatch, label=label, zorder=3)
            if ax is ax_l:
                bars.append(b)
        # ★0 선이 해석의 핵심이다 — 블록만 쓰면 혼잡에서 아래로 내려간다.
        ax.axhline(0, lw=0.9, color=INK, zorder=4)
        ax.set_xticks([x + mid for x in xs])
        ax.set_xticklabels([f"{L:,}\n({n_days[L]} d)" for L in group])
        ax.set_xlim(offs[0] - 0.22, len(group) - 1 + offs[-1] + 0.22)
        lo = min(v[L] for v in series for L in group)
        hi = max(v[L] for v in series for L in group)
        pad = 0.09 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        hgrid(ax)
        panel(ax, name)

    fig.supylabel("Cost reduction (KRW 100 M)",
                  fontsize=plt.rcParams["axes.labelsize"], color=INK)
    fig.supxlabel("Daily external truck job demand (days observed)",
                  fontsize=plt.rcParams["axes.labelsize"], color=INK)

    fig.legend(list(bars), [b.get_label() for b in bars], ncol=2,
               loc="outside upper center", handlelength=1.5,
               columnspacing=1.4, handletextpad=0.5)

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


def fig_paired():
    arms, _, days = load_arms()
    delta = sorted((arms["NO_REALLOC"][i] - arms["RL"][i]) / 1e8 for i in days)
    rank = list(range(1, len(delta) + 1))
    win = sum(1 for d in delta if d > 0)
    cut = len(delta) - 4          # 상위 4일이 축을 지배한다 — 나머지는 (b) 에서 본다

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.45), layout="constrained",
        gridspec_kw={"width_ratios": [1, 1]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)

    # ── (a) 28일 전부
    _lolli(ax_a, rank, delta, s=12)
    ax_a.axvspan(0.4, cut + 0.6, color=BAND, alpha=0.45, lw=0, zorder=0)
    ax_a.set_xlim(0.2, len(delta) + 0.8)
    ax_a.set_xticks([1, 10, 20, 28])
    ax_a.set_ylim(-0.40, 4.10)
    hgrid(ax_a)
    panel(ax_a, "(a) All 28 days")
    ax_a.text(0.05, 0.97,
              f"{win} / {len(delta)} days improved\n"
              "two-sided sign test, $p<0.001$",
              transform=ax_a.transAxes, ha="left", va="top", color=INK,
              linespacing=1.35)
    ax_a.text((cut + 1) / 2, -0.27, "shown in (b)", ha="center", va="center",
              fontsize=7.5, color=MUTED)

    # ── (b) ★확대 — 네 날이 축을 지배해서 나머지 24일이 0선에 뭉갠다. 26/28 이라는
    #        주장은 저 작은 값들의 **부호**에 달려 있으므로 축을 따로 준다.
    _lolli(ax_b, rank[:cut], delta[:cut], s=13, lw=0.9)
    ax_b.set_xlim(0.2, cut + 0.8)
    ax_b.set_xticks([1, 10, 20, 24])
    ax_b.set_ylim(-0.205, 0.375)
    hgrid(ax_b)
    panel(ax_b, f"(b) Ranks 1–{cut}, enlarged")

    fig.supylabel("Cost reduction (KRW 100 M)",
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
    print(f"  decomposition total {total:.2f} (x100M KRW), timing share {share:.1%}")
    print(f"  paired: {win}/{n} days improved")
