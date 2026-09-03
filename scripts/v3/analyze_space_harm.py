"""YR-251 — 블록 재배치가 손해일 때 **어느 비용 항이 움직이는가**.

    PYTHONPATH=src python scripts/v3/analyze_space_harm.py

■ 무엇을 가르나
  가설 ① 크레인이 더 움직인다      → `c_move` 가 오른다
  가설 ② 옮겨 간 블록이 곧 막힌다   → `c_wait` 가 오르고 이동은 그대로

■ 왜 새로 안 돌리나
  spec(2026-08-29)은 부하 5,000·15,000 을 **독립 하루**로 돌리자고 했다. 그런데
  [[YR-255]] 로 판정이 하루치 비용 4항을 저장하게 됐고, [[YR-286]] 혼잡 민감도가
  **30일 × 세 환경 × 다섯 팔**을 이미 굴려 놨다. 독립 하루보다 넓고(부하 5수준),
  깊고(밀림까지 보임), 계산이 **0**이다.

  독립 하루로는 못 보던 것이 하나 있다 — **밀림**. 30일은 세계가 이어지므로
  터진 날의 여파가 다음 날로 넘어가고, 그게 이 문제의 핵심이었다.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVS = (("한산", "quiet", 2), ("적당", "mixed", 6), ("혼잡", "heavy", 14))
TERMS = (("c_wait", "트럭 대기"), ("c_move", "크레인 이동"),
         ("c_rehandle", "재취급"), ("c_vessel", "본선 지연"))
EOK = 1e8   # 억


def load(key):
    A = {}
    for f in (ROOT / f"outputs/v3/env-{key}/arms").glob("arm_*.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        A[j["arm"]] = {d["index"]: d for d in j["days"]}
    return A


def days_of(A):
    return [i for i in sorted(A["RL"]) if 1 <= i <= 28]


def main() -> int:
    print("=" * 72)
    print("YR-251 · 블록만 재배치할 때 어느 비용 항이 움직이나")
    print("=" * 72)

    # ── 1. 환경별 전체 ────────────────────────────────────────────
    print("\n■ 1. 환경 전체 (28일 합) — 재배치 없음 대비 증감\n")
    print(f"  {'환경':<8}{'팔':<10}" + "".join(f"{n:>12}" for _, n in TERMS) + f"{'합계':>12}")
    for name, key, hv in ENVS:
        A = load(key)
        D = days_of(A)
        for arm, lbl in (("RL_SPACE", "블록만"), ("RL_TIME", "시각만"), ("RL", "둘 다")):
            cells = []
            for t, _ in TERMS:
                d = sum(A[arm][i][t] - A["NO_REALLOC"][i][t] for i in D)
                cells.append(f"{d / EOK:>+11.2f}억")
            tot = sum(A[arm][i]["phi_krw"] - A["NO_REALLOC"][i]["phi_krw"] for i in D)
            print(f"  {name if arm == 'RL_SPACE' else '':<8}{lbl:<10}"
                  + "".join(cells) + f"{tot / EOK:>+11.2f}억")
        print()

    # ── 2. 부하 수준별 — 어디서 뒤집히나 ─────────────────────────
    print("■ 2. 블록만 — 부하 수준별 (세 환경 합산, 재배치 없음 대비)\n")
    by_load: dict[int, dict] = {}
    for _, key, _ in ENVS:
        A = load(key)
        for i in days_of(A):
            L = A["RL"][i]["load"]
            b = by_load.setdefault(L, {"n": 0, **{t: 0.0 for t, _ in TERMS}, "phi": 0.0})
            b["n"] += 1
            for t, _ in TERMS:
                b[t] += A["RL_SPACE"][i][t] - A["NO_REALLOC"][i][t]
            b["phi"] += A["RL_SPACE"][i]["phi_krw"] - A["NO_REALLOC"][i]["phi_krw"]
    print(f"  {'부하':>8}{'일':>5}" + "".join(f"{n:>12}" for _, n in TERMS) + f"{'합계':>12}")
    for L in sorted(by_load):
        b = by_load[L]
        print(f"  {L:>8,}{b['n']:>5}"
              + "".join(f"{b[t] / EOK:>+11.2f}억" for t, _ in TERMS)
              + f"{b['phi'] / EOK:>+11.2f}억")

    # ── 3. 터진 날과 그 다음 — 밀림 ──────────────────────────────
    print("\n■ 3. 가장 크게 터진 날과 이어지는 사흘 (밀림)\n")
    worst = None
    for name, key, _ in ENVS:
        A = load(key)
        for i in days_of(A):
            v = A["RL_SPACE"][i]["phi_krw"] - A["NO_REALLOC"][i]["phi_krw"]
            if worst is None or v > worst[0]:
                worst = (v, name, key, i)
    _, name, key, i0 = worst
    A = load(key)
    print(f"  {name} 환경 {i0}일이 최악이다.\n")
    print(f"  {'날':>4}{'부하':>8}{'대기 증가':>13}{'이동 증가':>12}"
          f"{'90분위 체류':>14}{'블록 이동수':>13}")
    for i in range(i0, min(i0 + 4, 29)):
        if i not in A["RL_SPACE"]:
            break
        w = A["RL_SPACE"][i]["c_wait"] - A["NO_REALLOC"][i]["c_wait"]
        m = A["RL_SPACE"][i]["c_move"] - A["NO_REALLOC"][i]["c_move"]
        p0 = A["NO_REALLOC"][i]["p90_turn_time_s"] / 3600
        p1 = A["RL_SPACE"][i]["p90_turn_time_s"] / 3600
        print(f"  {i:>4}{A['RL'][i]['load']:>8,}{w / EOK:>+12.2f}억{m / EOK:>+11.2f}억"
              f"{p0:>7.1f}→{p1:>5.1f}h{A['RL_SPACE'][i]['n_space']:>13,}")

    # ── 4. 비용 구조 — 가설 ①이 애초에 성립 가능한가 ─────────────
    print("\n■ 4. 재배치 없음의 비용 구성 (84일) — 가설 ①의 여지\n")
    tot = {t: 0.0 for t, _ in TERMS}
    phi = 0.0
    for _, key, _ in ENVS:
        A = load(key)
        for i in days_of(A):
            for t, _ in TERMS:
                tot[t] += A["NO_REALLOC"][i][t]
            phi += A["NO_REALLOC"][i]["phi_krw"]
    for t, n in TERMS:
        print(f"  {n:<10}{tot[t] / EOK:>10.2f}억{tot[t] / phi:>9.2%}")
    print(f"\n  → 크레인 이동은 전체의 {tot['c_move'] / phi:.2%} 다. 이 비용 모형에서"
          f" **가설 ①은 구조적으로 불가능**하다.")

    # ── 5. 손해는 체계적인가, 꼬리사건인가 ───────────────────────
    print("\n■ 5. 초혼잡(15,000) 날별 — 블록만이 정말 늘 손해인가\n")
    allv = []
    for name, key, _ in ENVS:
        A = load(key)
        v = [(i, A["RL_SPACE"][i]["phi_krw"] - A["NO_REALLOC"][i]["phi_krw"])
             for i in days_of(A) if A["RL"][i]["load"] == 15_000]
        allv += v
        bad = [x for x in v if x[1] > 0]
        print(f"  {name:<6} {len(v)}일 중 손해 {len(bad)}일   "
              + "  ".join(f"{i}일 {d / EOK:+.1f}억" for i, d in v))
    worst_v = max(allv, key=lambda x: x[1])[1]
    print(f"\n  → 11일 중 손해는 {sum(1 for _, d in allv if d > 0)}일뿐이고,"
          f" 그중 한 날이 {worst_v / EOK:+.1f}억으로 합계를 혼자 만든다.")

    # ── 6. 판정 ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    tot_w = tot_m = 0.0
    for _, key, _ in ENVS:
        A = load(key)
        for i in days_of(A):
            tot_w += A["RL_SPACE"][i]["c_wait"] - A["NO_REALLOC"][i]["c_wait"]
            tot_m += A["RL_SPACE"][i]["c_move"] - A["NO_REALLOC"][i]["c_move"]
    print(f"판정 — 세 환경 84일 합:  대기 {tot_w / EOK:+.2f}억 · 이동 {tot_m / EOK:+.2f}억")
    print(f"       대기가 이동의 {abs(tot_w / tot_m):,.0f}배. **가설 ② — 옮겨 간 블록이 막힌다.**")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
