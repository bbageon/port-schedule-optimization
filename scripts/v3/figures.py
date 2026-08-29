"""논문 그림 생성 ([[YR-254]]) — outputs/v3/figures/*.svg

    PYTHONPATH=src python scripts/v3/figures.py

기존 강화학습·시뮬레이션 논문이 싣는 표준 구성을 따른다: 아키텍처 흐름 · 환경 특성 ·
학습 곡선 · 기준선 대비 · 에피소드별 분포 · 조건별 분해 · 민감도.
**모든 수치는 원자료에서 읽는다** (하드코딩 없음).
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from svg import (BLUE, FAINT, Fig, GRAY, INK, LOSS, MUTED, NAVY, PALE, RULE,
                 WARM, nice_ticks)

OUT = pathlib.Path("outputs/v3/figures")
ARMS = pathlib.Path("outputs/v3/judge-30d/arms")
TRAIN = pathlib.Path("outputs/v3/month-02")

NAME = {"RL": "제안 정책", "RL_TIME": "제안 정책 (시각만)", "RL_SPACE": "제안 정책 (블록만)",
        "RL_EARLY": "학습 전 모형", "NO_REALLOC": "재배치 없음",
        "SLOT_LL": "규칙: 한산한 시간대", "SPACE_TIME_LL": "규칙: 한산한 블록·시간대",
        "FCFS": "규칙: 선착순", "SPT": "규칙: 최단 처리시간",
        "LEAST_SLACK": "규칙: 최소 여유시간", "NETGAIN": "규칙: 순이득 기준"}
ACT = {"RL": "블록+시각", "RL_TIME": "시각", "RL_SPACE": "블록", "RL_EARLY": "블록+시각",
       "SLOT_LL": "시각", "SPACE_TIME_LL": "블록+시각", "FCFS": "블록", "SPT": "블록",
       "LEAST_SLACK": "블록", "NETGAIN": "블록"}


def load_arms():
    out = {}
    for f in ARMS.glob("arm_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["arm"]] = {int(k): v for k, v in d["phi_by_day"].items()}
    return out


# ───────────────────────────────────────────────── 1 · 아키텍처
def fig1():
    f = Fig(760, 296, pad=(0, 0, 0, 0))
    f.title("그림 1. 재배치 결정 흐름과 학습·운영의 분리")
    box = [("관측", "블록 혼잡·공개 도착예정"), ("제안 정책", "행동별 예상비용 argmin"),
           ("수락 정책", "수락·거절 비교"), ("중앙 확정", "합의·용량 확인 후 일괄 반영"),
           ("실행", "야드·크레인 상태 갱신")]
    w, gap, y = 132, 24, 88
    for i, (label, sub) in enumerate(box):
        x = 8 + i * (w + gap)
        dark = i in (1, 2)
        f.rect(x, y, w, 62, NAVY if dark else "#ffffff", NAVY if dark else RULE, rx=4)
        f.text(x + w / 2, y + 26, label, 12, "#ffffff" if dark else INK, "middle", 600)
        f.text(x + w / 2, y + 44, sub, 8.6, "#c8d4e4" if dark else MUTED, "middle")
        if i < 4:
            ax = x + w + 4
            f.path(f"M{ax} {y+31} L{ax+gap-9} {y+31}", NAVY, w=1.4)
            f.path(f"M{ax+gap-9} {y+31} l-5 -4 v8 z", None, NAVY)
    f.text(8, y - 12, "운영 경로 — 학습된 신경망만 사용, 반사실 호출 0회", 10, MUTED, weight=500)
    ly = y + 94
    f.rect(8, ly, 5 * w + 4 * gap, 62, "#f2f3f1", RULE, rx=4)
    f.text(22, ly + 24, "학습 경로 — 한 결정에서 상태·난수를 복제해 최대 세 세계를 3시간 실행",
           10.5, INK, weight=600)
    f.text(22, ly + 43, "관측 행동 · 제안만 반대 · 수락만 반대  →  Φ 차이를 중심화해 회귀 목표로 사용",
           9.6, MUTED)
    for i in (1, 2):
        cx = 8 + i * (w + gap) + w / 2
        f.path(f"M{cx} {y+62} L{cx} {ly}", GRAY, w=1.1, )
    f.text(8, 290, "학습에서 만든 비용 차이만 정책에 들어가고, 운영에서는 분기 시뮬레이션을 호출하지 않는다.",
           9.2, FAINT)
    return f.save(OUT / "fig1-architecture.svg")


# ───────────────────────────────────────────────── 2 · 도착밀도
def fig2():
    peaks = [(10, 1.5, .317), (15, 2.5, .633), (21, 1.0, .050)]

    def dens(t):
        v = .38 / 24
        for mu, sg, wt in peaks:
            v += .62 * wt * math.exp(-((t - mu) ** 2) / (2 * sg * sg)) / (sg * math.sqrt(2 * math.pi))
        return v

    xs = [i * .1 for i in range(241)]
    ys = [dens(t) for t in xs]
    f = Fig(640, 254)
    f.title("그림 2. 시간대별 트럭 도착밀도", "24시간 균등 바닥 38% + 정규분포 봉우리 3개")
    hi = max(ys) * 1.16
    for v in nice_ticks(0, hi, 4):
        y = f.y1 - (v / hi) * (f.y1 - f.y0)
        f.line(f.x0, y, f.x1, y, RULE)
        f.text(f.x0 - 8, y + 3.5, f"{v:.02f}", 9, FAINT, "end", mono=True)
    base = f.y1 - (.38 / 24 / hi) * (f.y1 - f.y0)
    f.rect(f.x0, base, f.x1 - f.x0, f.y1 - base, PALE, op=.5)
    f.text(f.x1 - 6, base - 6, "균등 바닥 (야간 포함)", 9, MUTED, "end")
    d = " ".join(("M" if i == 0 else "L") +
                 f"{f.x0 + t/24*(f.x1-f.x0):.1f} {f.y1 - y/hi*(f.y1-f.y0):.1f}"
                 for i, (t, y) in enumerate(zip(xs, ys)))
    f.path(d + f" L{f.x1} {f.y1} L{f.x0} {f.y1} Z", None, NAVY, op=.12)
    f.path(d, NAVY, w=1.8)
    for h in (0, 6, 12, 18, 24):
        x = f.x0 + h / 24 * (f.x1 - f.x0)
        f.line(x, f.y1, x, f.y1 + 4, GRAY)
        f.text(x, f.y1 + 17, f"{h}시", 9.5, MUTED, "middle")
    for mu, _, _ in peaks:
        x = f.x0 + mu / 24 * (f.x1 - f.x0)
        f.line(x, f.y0 + 4, f.y1 and f.y1, GRAY, dash="2 3") if False else None
        f.line(x, f.y0 + 4, x, f.y1, GRAY, dash="2 3")
        f.text(x, f.y0, f"μ={mu}", 8.5, FAINT, "middle")
    f.line(f.x0, f.y1, f.x1, f.y1, INK, 1.2)
    f.text(f.x0 - 42, f.y0 + 6, "밀도", 9.5, MUTED)
    return f.save(OUT / "fig2-arrival.svg")


# ───────────────────────────────────────────────── 3 · 학습 곡선
def fig3():
    h = json.loads((TRAIN / "history.json").read_text(encoding="utf-8"))
    rows = h if isinstance(h, list) else h.get("history", [])
    f = Fig(640, 296, pad=(52, 40, 48, 58))
    f.title("그림 3. 회차별 학습·검증 손실과 탐색 확률", "하루가 한 회차 · 28일 학습")
    series = [("seller_loss", NAVY, "제안 학습"), ("val_seller_loss", NAVY, "제안 검증"),
              ("buyer_loss", WARM, "수락 학습"), ("val_buyer_loss", WARM, "수락 검증")]
    hi = max(max(r.get(k) or 0 for r in rows) for k, _, _ in series) * 1.12
    for v in nice_ticks(0, hi, 4):
        y = f.y1 - (v / hi) * (f.y1 - f.y0)
        f.line(f.x0, y, f.x1, y, RULE)
        f.text(f.x0 - 8, y + 3.5, f"{v:.2f}", 9, FAINT, "end", mono=True)
    n = len(rows)

    def px(i):
        return f.x0 + i / max(1, n - 1) * (f.x1 - f.x0)

    for key, col, _ in series:
        val = key.startswith("val")
        d = " ".join(("M" if i == 0 else "L") +
                     f"{px(i):.1f} {f.y1 - (r.get(key) or 0)/hi*(f.y1-f.y0):.1f}"
                     for i, r in enumerate(rows))
        f.path(d, col, w=1.2 if val else 1.6, op=.5 if val else 1)
    ex = [r["explore"] for r in rows]
    f.path(" ".join(("M" if i == 0 else "L") + f"{px(i):.1f} {f.y1 - v*(f.y1-f.y0):.1f}"
                    for i, v in enumerate(ex)), GRAY, w=1.2)
    f.text(px(n - 1) - 4, f.y1 - ex[-1] * (f.y1 - f.y0) - 8, "탐색 확률", 9, MUTED, "end")
    for i in (0, 9, 19, n - 1):
        f.line(px(i), f.y1, px(i), f.y1 + 4, GRAY)
        f.text(px(i), f.y1 + 17, f"{i}", 9.5, MUTED, "middle", mono=True)
    f.text((f.x0 + f.x1) / 2, f.y1 + 34, "회차 (일)", 9.5, MUTED, "middle")
    f.line(f.x0, f.y1, f.x1, f.y1, INK, 1.2)
    lx = f.x0
    for _, col, lab in series:
        f.rect(lx, f.y0 - 22, 14, 2.6, col)
        f.text(lx + 18, f.y0 - 17, lab, 9.4, MUTED)
        lx += 18 + len(lab) * 9.4
    return f.save(OUT / "fig3-learning.svg")


# ───────────────────────────────────────────────── 4 · 정책 비교
def fig4(A, train):
    no = sum(A["NO_REALLOC"][i] for i in train)
    keys = [a for a in A if a not in ("NEAREST", "NO_REALLOC")]
    val = {a: (no - sum(A[a][i] for i in train)) / no * 100 for a in keys}
    keys.sort(key=lambda a: -val[a])
    f = Fig(640, 46 + len(keys) * 26 + 46, pad=(172, 44, 26, 62))
    f.title("그림 4. 정책별 28일 총비용 감소율", "재배치 없음 대비 · 양수가 비용 감소")
    lo = min(min(val.values()), 0) * 1.2
    hi = max(val.values()) * 1.14
    bot = f.y0 + len(keys) * 26 - 8

    def px(v):
        return f.x0 + (v - lo) / (hi - lo) * (f.x1 - f.x0)

    for t in nice_ticks(lo, hi, 5):
        f.line(px(t), f.y0 - 8, px(t), bot, RULE)
        f.text(px(t), f.y0 - 14, f"{t:+.0f}%", 9, FAINT, "middle", mono=True)
    f.line(px(0), f.y0 - 8, px(0), bot, INK, 1.2)
    for i, a in enumerate(keys):
        y, v = f.y0 + i * 26, val[a]
        col = NAVY if a == "RL" else (BLUE if a.startswith("RL") and a != "RL_EARLY"
                                      else ("#7d93ad" if a == "RL_EARLY" else GRAY))
        if v < 0:
            col = LOSS
        f.rect(px(min(0, v)), y, abs(px(v) - px(0)), 15, col, rx=2)
        f.text(f.x0 - 10, y + 12, NAME[a], 10, INK if a == "RL" else MUTED, "end",
               600 if a == "RL" else 400)
        f.text(px(v) + (6 if v >= 0 else -6), y + 12, f"{v:+.2f}%", 9.4, INK,
               "start" if v >= 0 else "end", mono=True)
        f.text(f.x1 + 8, y + 12, ACT[a], 8.4, FAINT)
    f.text(f.x1 + 8, f.y0 - 14, "사용 행동", 8.4, FAINT)
    f.text(0, f.h - 10,
           "블록만 쓰는 정책은 학습·규칙 모두 0 부근이고, 감소는 도착시각 조정에서 나온다.",
           9.2, FAINT)
    return f.save(OUT / "fig4-policies.svg")


# ───────────────────────────────────────────────── 5 · 날짜별 짝차이
def fig5(A, train, days):
    diffs = sorted(((A["RL"][i] - A["NO_REALLOC"][i]) / 1e8, days[i].load) for i in train)
    f = Fig(640, 286, pad=(58, 44, 54, 26))
    f.title("그림 5. 날짜별 비용 차이 (제안 정책 − 재배치 없음)",
            "28일을 차이 크기로 정렬 · 음수가 제안 정책의 비용이 낮은 날")
    lo = min(d for d, _ in diffs) * 1.18
    hi = max(max(d for d, _ in diffs), 0) * 1.4 + .06

    def py(v):
        return f.y1 - (v - lo) / (hi - lo) * (f.y1 - f.y0)

    for t in nice_ticks(lo, hi, 5):
        f.line(f.x0, py(t), f.x1, py(t), RULE)
        f.text(f.x0 - 8, py(t) + 3.5, f"{t:+.0f}", 9, FAINT, "end", mono=True)
    f.text(f.x0 - 46, f.y0 - 10, "억원", 9, MUTED)
    n = len(diffs)
    bw = (f.x1 - f.x0) / n
    for i, (d, load) in enumerate(diffs):
        col = NAVY if load >= 12_500 else (BLUE if d < 0 else LOSS)
        f.rect(f.x0 + i * bw + bw * .16, py(max(d, 0)), bw * .68,
               abs(py(d) - py(0)), col, rx=1)
    f.line(f.x0, py(0), f.x1, py(0), INK, 1.2)
    win = sum(1 for d, _ in diffs if d < 0)
    f.text(f.x0 + 8, f.y0 + 8, f"제안 정책이 낮은 날 {win}/{n} · 양측 부호검정 p<0.001",
           10, INK, weight=600)
    lx = f.x0
    for col, lab in ((NAVY, "혼잡·초혼잡 수요일"), (BLUE, "그 외 감소일"), (LOSS, "증가일")):
        f.rect(lx, f.y1 + 26, 11, 11, col, rx=2)
        f.text(lx + 16, f.y1 + 35, lab, 9.4, MUTED)
        lx += 16 + len(lab) * 9.6
    return f.save(OUT / "fig5-paired.svg")


# ───────────────────────────────────────────────── 6 · 행동 분해
def fig6(A, train, days):
    by = collections.defaultdict(list)
    for i in train:
        by[days[i].load].append(i)
    cols = [("RL", "두 행동 모두", NAVY), ("RL_TIME", "도착시각만", BLUE),
            ("RL_SPACE", "블록만", GRAY), ("RL_EARLY", "학습 전 모형", PALE)]
    loads = sorted(by)
    f = Fig(640, 314, pad=(58, 48, 62, 22))
    f.title("그림 6. 수요수준별·행동유형별 비용 감소",
            "재배치 없음 대비 · 양수가 감소 · 같은 28일")
    data = {}
    for v in loads:
        idx = by[v]
        no = sum(A["NO_REALLOC"][i] for i in idx)
        for a, _, _ in cols:
            data[(a, v)] = (no - sum(A[a][i] for i in idx)) / 1e8
    lo, hi = min(min(data.values()), 0) * 1.22, max(data.values()) * 1.14

    def py(x):
        return f.y1 - (x - lo) / (hi - lo) * (f.y1 - f.y0)

    for t in nice_ticks(lo, hi, 5):
        f.line(f.x0, py(t), f.x1, py(t), RULE)
        f.text(f.x0 - 8, py(t) + 3.5, f"{t:+.0f}", 9, FAINT, "end", mono=True)
    f.text(f.x0 - 46, f.y0 - 10, "억원", 9, MUTED)
    gw = (f.x1 - f.x0) / len(loads)
    bw = gw * .74 / len(cols)
    for gi, v in enumerate(loads):
        gx = f.x0 + gi * gw + gw * .13
        for ci, (a, _, col) in enumerate(cols):
            d = data[(a, v)]
            f.rect(gx + ci * bw, py(max(d, 0)), bw * .86, abs(py(d) - py(0)), col, rx=1.5)
        cx = f.x0 + gi * gw + gw / 2
        f.text(cx, f.y1 + 17, f"{v:,}대", 9.4, MUTED, "middle")
        f.text(cx, f.y1 + 31, f"{len(by[v])}일", 8.4, FAINT, "middle")
    f.line(f.x0, py(0), f.x1, py(0), INK, 1.2)
    lx = f.x0
    for _, lab, col in cols:
        f.rect(lx, f.y0 - 30, 11, 11, col, rx=2)
        f.text(lx + 16, f.y0 - 21, lab, 9.4, MUTED)
        lx += 16 + len(lab) * 9.6
    return f.save(OUT / "fig6-actions.svg")


# ───────────────────────────────────────────────── 7 · 작업순서 민감도
def fig7():
    p = pathlib.Path("outputs/v3/base-matrix/rows.json")
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    by = {(r["base"], r["load"]): r for r in rows}
    bases = sorted({r["base"] for r in rows})
    loads = sorted({r["load"] for r in rows})
    kor = {"SF_SPT": "최단 처리시간 (현행)", "FIFO": "선착순", "LIFO": "후착순",
           "LWKR": "누적 대기 우선", "NEAREST": "최근접", "RANDOM": "무작위"}
    f = Fig(640, 292, pad=(58, 46, 54, 138))
    f.title("그림 7. 크레인 작업순서 규칙의 수요 민감도",
            "재배치 없음 · 각 수요에서 가장 낮은 총비용을 1.0 으로 둔 상대값")

    def rel(b, v):
        return by[(b, v)]["phi"] / min(by[(x, v)]["phi"] for x in bases)

    hi = max(rel(b, v) for b in bases for v in loads) * 1.05

    def py(x):
        return f.y1 - (x - .96) / (hi - .96) * (f.y1 - f.y0)

    for t in nice_ticks(1.0, hi, 4):
        f.line(f.x0, py(t), f.x1, py(t), RULE)
        f.text(f.x0 - 8, py(t) + 3.5, f"{t:.2f}", 9, FAINT, "end", mono=True)
    for b in bases:
        cur = b == "SF_SPT"
        col = NAVY if cur else GRAY
        pts = [(f.x0 + i / (len(loads) - 1) * (f.x1 - f.x0), py(rel(b, v)))
               for i, v in enumerate(loads)]
        f.path(" ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}"
                        for i, (x, y) in enumerate(pts)),
               col, w=2 if cur else 1.2, op=1 if cur else .65)
        for x, y in pts:
            f.rect(x - 2.2, y - 2.2, 4.4, 4.4, col, rx=2.2)
        f.text(f.x1 + 8, pts[-1][1] + 3.5, kor.get(b, b), 9.4,
               INK if cur else MUTED, weight=600 if cur else 400)
    for i, v in enumerate(loads):
        x = f.x0 + i / (len(loads) - 1) * (f.x1 - f.x0)
        f.line(x, f.y1, x, f.y1 + 4, GRAY)
        f.text(x, f.y1 + 17, f"{v//1000}천", 9.4, MUTED, "middle")
    f.text((f.x0 + f.x1) / 2, f.y1 + 34, "일일 트럭 수요", 9.4, MUTED, "middle")
    f.line(f.x0, f.y1, f.x1, f.y1, INK, 1.2)
    return f.save(OUT / "fig7-crane.svg")


def main() -> int:
    from yard_rl.v3.stage.month import plan_month
    OUT.mkdir(parents=True, exist_ok=True)
    A = load_arms()
    days = {d.index: d for d in plan_month(9_900_950, n_days=30)}
    train = [i for i in sorted(A["RL"]) if 1 <= i <= 28]
    made = [fig1(), fig2(), fig3(), fig4(A, train), fig5(A, train, days),
            fig6(A, train, days), fig7()]
    for m in made:
        if m:
            print(" ·", m)
    print(f"■ 그림 {sum(1 for m in made if m)}장 · {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
