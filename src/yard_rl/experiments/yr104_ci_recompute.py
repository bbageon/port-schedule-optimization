"""YR-104 — 공용 _ci t계수 결함의 소급 재계산 (저장 원자료 → 정확 df t 로 재산출).

결함: 구 `_ci` 가 df=19 외 fallback 2.1 사용 → df≤18 판정의 CI 가 반보수(협소).
대상: n=8 판정(YR-099-G1·오라클·YR-101, df=7 → 2.365 — 12.6% 확대) 중심.
n≥20 판정(yr090·yr100 behave: df=19 정확 / yr041·candidate_eval: df=23, 2.1>2.069 보수)
은 방향 확인만. YR-098 은 n=40 자체 CI — 보수 방향 확인.
판정 규칙: 재계산 CI 로 각 박제 결론(유의/무의/기각)이 바뀌는지 표로 정직 박제.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, stdev

_T95 = {5: 2.571, 7: 2.365, 9: 2.262, 19: 2.093, 23: 2.069, 39: 2.023}
OUT = Path("outputs/reports/yr104_ci_correction")
R = Path("outputs/reports")


def ci(d, decimals=2):
    m, n = fmean(d), len(d)
    se = stdev(d) / n ** 0.5
    t = _T95.get(n - 1) or (2.042 if n - 1 >= 30 else 2.571)
    return [round(m, decimals), round(m - t * se, decimals), round(m + t * se, decimals), n, t]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(exp, metric, d, old, verdict_rule):
        new = ci(d)
        old_sig = old[2] < 0 if old else None
        new_sig = new[2] < 0
        rows.append({"exp": exp, "metric": metric, "n": new[3], "t_correct": new[4],
                     "old_ci": old, "new_ci": new[:3],
                     "verdict_rule": verdict_rule,
                     "significant_old": old_sig, "significant_new": new_sig,
                     "conclusion_changed": (old_sig is not None and old_sig != new_sig)})

    # --- YR-099-G1 (n=8, df=7)
    d = json.loads((R / "yr099_transfer_mvp/results.json").read_text(encoding="utf-8"))
    add("YR-099-G1 예측-quote", "d_total", [r["d_total"] for r in d["rows"]],
        d["g1_d_total_ci"], "상한<0=상금")
    d = json.loads((R / "yr099_transfer_mvp/results_oracle.json").read_text(encoding="utf-8"))
    add("YR-099-G1 오라클", "d_total", [r["d_total"] for r in d["rows"]],
        d["g1_d_total_ci"], "상한<0=상금(진단)")
    d = json.loads((R / "yr099_transfer_mvp/results_ens_K5.json").read_text(encoding="utf-8"))
    add("YR-101 ensemble K=5", "d_total", [r["d_total"] for r in d["rows"]],
        d["g1_d_total_ci"], "상한<0=회수")

    # --- YR-041-a (n=24, df=23 — 구 2.1 은 보수)
    d = json.loads((R / "yr041_locked/results.json").read_text(encoding="utf-8"))
    rc, rs = d["rows_candidate"], d["rows_sf"]
    add("YR-041-a JR1800", "d_berth", [a["berth"] - b["berth"] for a, b in zip(rc, rs)],
        d["d_berth_ci"], "상한<5=비열등")
    add("YR-041-a JR1800", "d_p95", [a["p95"] - b["p95"] for a, b in zip(rc, rs)],
        d["d_p95_ci"], "상한<3=비열등(구 계약)")
    add("YR-041-a JR1800", "d_total", [a["total"] - b["total"] for a, b in zip(rc, rs)],
        d["d_total_ci"], "보고")

    # --- YR-098 확증 (n=40 자체 CI — 방향 확인용 재기록)
    d = json.loads((R / "yr098_authority_null/results.json").read_text(encoding="utf-8"))
    vf = d["confirm"]["VF"]["total_cost"]
    rows.append({"exp": "YR-098 확증 VF", "metric": "total_cost", "n": vf["n"],
                 "t_correct": _T95.get(vf["n"] - 1, 2.023),
                 "old_ci": [vf["mean"], vf["lo"], vf["hi"]],
                 "new_ci": "자체 CI(n=40) — 정확 t=2.023 ≤ 사용값 → 보수 방향(안전), 재계산 불요",
                 "verdict_rule": "상한<0=GO", "significant_old": vf["hi"] < 0,
                 "significant_new": vf["hi"] < 0, "conclusion_changed": False})

    n_changed = sum(1 for r in rows if r["conclusion_changed"])
    res = {"rows": rows, "n_conclusion_changed": n_changed,
           "note": "df=19(n=20) 사용 판정(yr090·yr100 behave)은 원래 정확 — 재계산 불요. "
                   "df=23(n=24) 은 구 2.1 이 보수라 결론 안전."}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    for r in rows:
        print(f"{r['exp']:24s} {r['metric']:8s} n={r['n']:3d} t={r['t_correct']} "
              f"old={r['old_ci']} new={r['new_ci']} changed={r['conclusion_changed']}")
    print(f"\n결론 변경: {n_changed}건")
    return res


if __name__ == "__main__":
    main()
    print("DONE")
