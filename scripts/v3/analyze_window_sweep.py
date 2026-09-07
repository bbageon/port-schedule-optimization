"""YR-299 A — 재검토 창 W 민감도. *"도착 30분 전 재배정은 촉박하지 않나"* 에 답한다.

    PYTHONPATH=src python scripts/v3/analyze_window_sweep.py

■ 무엇을 재나
  결정 시점을 도착 **30분 / 1시간 / 2시간** 전으로 옮기며 이득이 얼마나 깎이는지.
  ⚠️ 망은 30분 조건에서 배웠으므로 **큰 창에 불리한 편향**이 있다. 그런데도 이득이
     유지되면 그 자체로 강한 결과다(1단계 = 평가만, 재학습 없음).
"""
from __future__ import annotations

import json
import pathlib
from math import comb

ROOT = pathlib.Path("outputs/v3/window-sweep")
WINDOWS = ((1800, "30분"), (3600, "1시간"), (7200, "2시간"))


def sign_p(d):
    n = sum(1 for x in d if abs(x) > 1e-9)
    k = sum(1 for x in d if x > 1e-9)
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0


def exact_sr(d):
    d = [x for x in d if x != 0.0]
    n = len(d)
    if n == 0:
        return 1.0
    o = sorted(range(n), key=lambda i: abs(d[i]))
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[o[j + 1]]) == abs(d[o[i]]):
            j += 1
        for k in range(i, j + 1):
            rk[o[k]] = (i + j) / 2 + 1
        i = j + 1
    W = sum(rk[i] for i in range(n) if d[i] > 0)
    tot = n * (n + 1) // 2
    dp = [0] * (tot + 1)
    dp[0] = 1
    for r in range(1, n + 1):
        for s in range(tot, r - 1, -1):
            dp[s] += dp[s - r]
    return min(1.0, 2 * sum(dp[:int(min(W, tot - W)) + 1]) / 2 ** n)


rows = []
for sec, lbl in WINDOWS:
    d = ROOT / f"w{sec}" / "arms"
    if not (d / "arm_RL.json").exists():
        print(f"  {lbl}: 자료 없음")
        continue
    A = {}
    for f in d.glob("arm_*.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        A[j["arm"]] = ({int(k): v for k, v in j["phi_by_day"].items()}, j)
    D = [i for i in sorted(A["RL"][0]) if 1 <= i <= 28]
    B = sum(A["NO_REALLOC"][0][i] for i in D)
    R = sum(A["RL"][0][i] for i in D)
    diff = [A["NO_REALLOC"][0][i] - A["RL"][0][i] for i in D]
    jr = A["RL"][1]
    rows.append(dict(lbl=lbl, base=B, rl=R, cut=(B - R) / B,
                     win=sum(1 for x in diff if x > 0), p=sign_p(diff),
                     sr=exact_sr(diff), traded=jr["traded"],
                     sp=jr["n_space"], tm=jr["n_time"]))

print("=" * 72)
print("YR-299 A · 재검토 창 W 민감도 (평가만 · 재학습 없음)")
print("=" * 72)
print(f"\n  {'창':>6}{'재배치없음':>12}{'학습정책':>11}{'감소':>8}"
      f"{'낮은날':>8}{'부호p':>9}{'순위p':>9}")
for r in rows:
    print(f"  {r['lbl']:>6}{r['base']/1e8:>11.2f}억{r['rl']/1e8:>10.2f}억"
          f"{r['cut']:>8.2%}{r['win']:>6}/28{r['p']:>9.4f}{r['sr']:>9.4f}")

print(f"\n  {'창':>6}{'거래':>10}{'공간':>10}{'시간':>10}{'공간비중':>10}")
for r in rows:
    print(f"  {r['lbl']:>6}{r['traded']:>10,}{r['sp']:>10,}{r['tm']:>10,}"
          f"{r['sp']/max(1,r['sp']+r['tm']):>9.1%}")

if len(rows) >= 2:
    a, z = rows[0], rows[-1]
    keep = z['cut'] / a['cut']
    print("\n" + "=" * 72)
    print(f"판정 — 창을 30분 → {z['lbl']}으로 넓히면 이득이 "
          f"{a['cut']:.2%} → {z['cut']:.2%} ({keep:.0%} 유지)")
    if keep >= 0.8:
        print("       **유지된다.** 1~2시간 여유를 주고도 실행 가능하다 —")
        print("       망이 30분에서 배워 큰 창에 불리한데도 버텼으므로 더 강한 결과다.")
    elif keep >= 0.5:
        print("       **부분 유지.** 여유를 주면 이득이 줄지만 사라지지 않는다.")
    else:
        print("       **급감한다.** 가치가 막판 정보에 있다 —")
        print("       실시간 정보 공유(스마트항만 인프라)의 가치를 정량화한 셈이다.")
    print("=" * 72)
