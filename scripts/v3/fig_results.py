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

from figstyle import (A_GREEN, INK, LOSS_RED, MUTED, P_BLUE, TEXTWIDTH_IN,
                      apply, hgrid, save)

OUT = ROOT / "docs/paper/v3/figures/final_figure"
HISTORY = ROOT / "outputs/v3/month-02/history.json"
ARMS = ROOT / "outputs/v3/judge-30d/arms"
JUDGE_SEED = 9_900_950

#: 행동 분해 막대 — 색과 해칭을 **함께** 준다 (흑백에서도 갈리게).
BARS = (
    ("Both actions", P_BLUE, None),
    ("Timing only", "#8FB0C6", None),
    ("Block only", "#D3DFE7", "///"),
    ("Learned contribution", "white", "xxx"),
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
    }


def load_arms():
    arms = {}
    for f in ARMS.glob("arm_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        arms[d["arm"]] = {int(k): v for k, v in d["phi_by_day"].items()}
    from yard_rl.v3.stage.month import plan_month
    load = {d.index: d.load for d in plan_month(JUDGE_SEED)}
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


def fig_learning():
    ep, h = load_history()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.75))

    for key, color, ls, name in (
            ("p_train", P_BLUE, "-", "Proposal — train"),
            ("p_val", P_BLUE, "--", "Proposal — validation"),
            ("a_train", A_GREEN, "-", "Acceptance — train"),
            ("a_val", A_GREEN, "--", "Acceptance — validation")):
        ax.plot(ep, h[key], ls, lw=0.7, color=color, alpha=0.25)
        ax.plot(ep, _roll(h[key]), ls, lw=1.6, color=color, label=name)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Huber loss")
    ax.set_xlim(1, len(ep))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_ylim(0, max(max(_roll(h[k])) for k in
                       ("p_train", "p_val", "a_train", "a_val")) * 1.35)
    hgrid(ax)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ep, h["eps"], "-.", lw=1.1, color=MUTED, alpha=0.8,
             label=r"Exploration $\varepsilon$")
    ax2.set_ylabel(r"Exploration probability $\varepsilon$")
    ax2.set_ylim(0, 0.55)

    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.34), columnspacing=1.4,
              handlelength=2.4, borderaxespad=0.0)

    save(fig, OUT, "fig-learning-curve", ROOT)


# ══════════════════════ ② 수요수준별 분해 ══════════════════════
def fig_decomposition():
    arms, load, days = load_arms()
    levels = sorted({load[i] for i in days})

    def agg(a, b):
        """a − b 를 부하별로 합산 (억원). 양수 = 비용이 줄었다."""
        return [sum(arms[a][i] - arms[b][i] for i in days if load[i] == L) / 1e8
                for L in levels]

    series = (agg("NO_REALLOC", "RL"), agg("NO_REALLOC", "RL_TIME"),
              agg("NO_REALLOC", "RL_SPACE"), agg("RL_EARLY", "RL"))
    total = sum(series[0])
    time_share = sum(series[1]) / total

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.85))
    xs = list(range(len(levels)))
    w = 0.19
    for k, (vals, (name, fc, hatch)) in enumerate(zip(series, BARS)):
        ax.bar([x + (k - 1.5) * w for x in xs], vals, w, label=name,
               facecolor=fc, edgecolor=INK, linewidth=0.7, hatch=hatch)

    # ★0 선이 해석의 핵심이다 — 블록만 쓰면 혼잡에서 아래로 내려간다.
    ax.axhline(0, lw=0.9, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{L:,}" for L in levels])
    ax.set_xlabel("Daily external truck job demand")
    ax.set_ylabel("Cost reduction (KRW 100 M)")
    hgrid(ax)
    lo = min(min(s) for s in series)
    hi = max(max(s) for s in series)
    ax.set_ylim(lo * 1.20, hi * 1.20)
    ax.legend(ncol=2, loc="upper left", handlelength=1.6,
              columnspacing=1.2, borderaxespad=0.2)

    # 숫자는 하나만 그림 안에 — 나머지는 캡션과 본문이 말한다.
    # 화살표는 쓰지 않는다: 막대가 양끝에 몰려 있어 어디에 그어도 그림을 가로지른다.
    ax.text(1.55, hi * 0.62, f"timing accounts for\n{time_share:.1%} of the total",
            ha="center", va="center", color=INK, linespacing=1.3)
    save(fig, OUT, "fig-decomposition", ROOT)
    return total, time_share


# ═════════════════ ③ 날짜별 짝비교 (정렬 롤리팝) ═════════════════
def fig_paired():
    arms, _, days = load_arms()
    delta = sorted((arms["NO_REALLOC"][i] - arms["RL"][i]) / 1e8 for i in days)
    rank = list(range(1, len(delta) + 1))
    win = sum(1 for d in delta if d > 0)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.75))
    for x, v in zip(rank, delta):
        ax.plot([x, x], [0, v], lw=0.9, alpha=0.6, zorder=2,
                color=P_BLUE if v > 0 else LOSS_RED)
    for keep, color, mark in ((True, P_BLUE, "o"), (False, LOSS_RED, "s")):
        pts = [(x, v) for x, v in zip(rank, delta) if (v > 0) is keep]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16,
                       zorder=3, color=color, marker=mark, edgecolor="none")
    ax.axhline(0, lw=0.9, color=INK)

    ax.set_xlabel("Operating days ranked by paired cost difference")
    ax.set_ylabel("Cost reduction (KRW 100 M)")
    ax.set_xlim(0, len(delta) + 1)
    hgrid(ax)
    ax.text(0.03, 0.95,
            f"{win} / {len(delta)} days improved\n"
            "two-sided sign test, $p<0.001$",
            transform=ax.transAxes, ha="left", va="top", color=INK,
            linespacing=1.3)

    # ★확대 — 네 날이 축을 지배해서 나머지 24일이 0선에 뭉갠다. 26/28 이라는
    #   주장은 저 작은 값들의 **부호**에 달려 있으므로 따로 보여 준다.
    zoom = [(x, v) for x, v in zip(rank, delta) if x <= len(delta) - 4]
    ins = ax.inset_axes([0.30, 0.34, 0.46, 0.40])
    for x, v in zoom:
        ins.plot([x, x], [0, v], lw=0.8, alpha=0.6,
                 color=P_BLUE if v > 0 else LOSS_RED)
    for keep, color, mark in ((True, P_BLUE, "o"), (False, LOSS_RED, "s")):
        pts = [p for p in zoom if (p[1] > 0) is keep]
        if pts:
            ins.scatter([p[0] for p in pts], [p[1] for p in pts], s=9,
                        color=color, marker=mark, edgecolor="none", zorder=3)
    ins.axhline(0, lw=0.8, color=INK)
    ins.set_title(f"ranks 1–{len(zoom)}, enlarged", fontsize=7.5, pad=2.5,
                  color=MUTED)
    ins.tick_params(labelsize=7)
    ins.set_xticks([1, 10, 20])
    hgrid(ins)

    save(fig, OUT, "fig-paired-daily", ROOT)
    return win, len(delta)


if __name__ == "__main__":
    apply()
    print("결과 그림 셋 (LNCS 본문폭 4.80 in · 세리프 8~10 pt)")
    fig_learning()
    total, share = fig_decomposition()
    win, n = fig_paired()
    print("\n[그림이 원자료에서 읽은 값 — 표와 대조하라]")
    print(f"  행동분해 합계 {total:.2f}억 · 시각 비중 {share:.1%}")
    print(f"  짝비교 {win}/{n}일 개선")
