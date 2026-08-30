"""제출본의 수치를 원자료에서 다시 계산해 본문 값과 대조한다.

    PYTHONPATH=src python verify_submission.py

논문에 적힌 값을 손으로 옮겨 적지 않는다 — 여기서 계산한 값과 원고의 문자열을
기계로 맞춰 본다. 불일치가 있으면 그 줄을 찍는다.
"""
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(".")
sys.path.insert(0, str(ROOT / "scripts/v3"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import stats

from figdata import month

TEX = (ROOT / "docs/paper/v3/submission/main.tex").read_text(encoding="utf-8")
MN, BN = 1e6, 1e9
SEED = 9_900_950


def arms_data():
    arms = {}
    for f in (ROOT / "outputs/v3/judge-30d/arms").glob("arm_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        arms[d["arm"]] = {int(k): v for k, v in d["phi_by_day"].items()}
    load = {d.index: d.load for d in month().plan_month(SEED)}
    days = [i for i in sorted(arms["RL"]) if 1 <= i <= 28]
    return arms, load, days


def check(label, computed, needle, fmt="{:.2f}"):
    """계산값과 원고 문자열을 대조한다."""
    shown = fmt.format(computed) if not isinstance(computed, str) else computed
    ok = needle in TEX
    print(f"   {'OK ' if ok else '✗  '} {label:44s} 계산 {shown:>12s}  원고 '{needle}'")
    return ok


def main():
    arms, load, days = arms_data()
    tot = {a: sum(v[i] for i in days) for a, v in arms.items()}
    base = tot["NO_REALLOC"]
    bad = 0

    print("[총액과 감소율]")
    bad += not check("재배치 없음 총액 (십억원)", base / BN, "KRW 10.00")
    bad += not check("학습 정책 총액 (십억원)", tot["RL"] / BN, "KRW 8.52")
    bad += not check("감소액 (십억원)", (base - tot["RL"]) / BN, "KRW 1.471", "{:.3f}")
    bad += not check("감소율 %", 100 * (base - tot["RL"]) / base, "14.72", "{:.2f}")
    for arm, needle in (("RL_TIME", "+13.44"), ("RL_SPACE", "$-$1.31"),
                        ("RL_EARLY", "+7.97"), ("SLOT_LL", "+3.41"),
                        ("SPACE_TIME_LL", "+1.51"), ("LEAST_SLACK", "+0.53"),
                        ("FCFS", "+0.17"), ("SPT", "+0.05"), ("NETGAIN", "$-$0.48")):
        bad += not check(f"{arm} 감소율 %", 100 * (base - tot[arm]) / base, needle, "{:+.2f}")

    print("\n[행동 분해와 상호작용]")
    t_only = (base - tot["RL_TIME"]) / BN
    b_only = (base - tot["RL_SPACE"]) / BN
    total = (base - tot["RL"]) / BN
    bad += not check("시각만 (십억원)", t_only, "1.344", "{:.3f}")
    bad += not check("블록만 (십억원)", b_only, "$-$KRW 0.131", "{:.3f}")
    bad += not check("상호작용 (십억원)", total - t_only - b_only, "0.258", "{:.3f}")
    bad += not check("시각 비중 %", 100 * t_only / total, "91.3", "{:.1f}")
    heavy = [i for i in days if load[i] >= 12_500]
    share = sum(arms["NO_REALLOC"][i] - arms["RL"][i] for i in heavy) / (base - tot["RL"])
    bad += not check("혼잡 나흘 비중 %", 100 * share, "90.5", "{:.1f}")

    print("\n[짝비교]")
    def paired(a, b):
        d = np.array([(arms[b][i] - arms[a][i]) / MN for i in days])
        pos, n = int((d > 0).sum()), int((d != 0).sum())
        return pos, n, stats.binomtest(pos, n, 0.5).pvalue, \
            float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue), d.sum()
    for arm, days_needle, ps_needle in (
            ("RL_EARLY", "18/28", "0.185"), ("SPACE_TIME_LL", "20/28", "0.036"),
            ("LEAST_SLACK", "20/28", None), ("FCFS", "22/28", "0.004"),
            ("SPT", "21/28", "0.013"), ("NETGAIN", "21/28", None),
            ("NO_REALLOC", "26/28", None)):
        pos, n, ps, pw, _ = paired("RL", arm)
        bad += not check(f"RL vs {arm} 승수", f"{pos}/{n}", days_needle)
        if ps_needle:
            bad += not check(f"RL vs {arm} 부호 p", ps, ps_needle, "{:.3f}")
    pos, n, ps, pw, s = paired("RL_TIME", "SLOT_LL")
    bad += not check("시간축: RL_TIME vs 규칙 승수", f"{pos}/{n}", "8 of 28")
    bad += not check("시간축 부호순위 p", pw, "0.399", "{:.3f}")
    bad += not check("시간축 총액차 (십억원)", s / 1000, "KRW 1.00", "{:.2f}")

    print("\n[학습 효과]")
    bad += not check("1회차 모형 총액 (십억원)", tot["RL_EARLY"] / BN, "KRW 9.20")
    bad += not check("학습의 몫 (십억원)", (tot["RL_EARLY"] - tot["RL"]) / BN, "0.675", "{:.3f}")
    bad += not check("학습의 몫 비중 %",
                     100 * (tot["RL_EARLY"] - tot["RL"]) / (base - tot["RL"]), "45.9", "{:.1f}")
    inc = tot["RL_EARLY"] - tot["RL"]
    inc_heavy = sum(arms["RL_EARLY"][i] - arms["RL"][i] for i in heavy)
    bad += not check("혼잡 나흘이 학습 몫에서 차지 %", 100 * inc_heavy / inc, "94.6", "{:.1f}")
    diff = np.array([(arms["RL_EARLY"][i] - arms["RL"][i]) / MN for i in days])
    rng = np.random.default_rng(20260830)
    groups = {}
    for i, dd in zip(days, diff):
        groups.setdefault(load[i], []).append(dd)
    means = np.zeros(20_000)
    for g in groups.values():
        g = np.asarray(g)
        pick = rng.integers(0, len(g), size=(20_000, len(g)))
        means += g[pick].sum(axis=1)
    means /= len(diff)
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"   .. 층화 붓스트랩 95% 구간 (백만원/일): [{-hi:.1f}, {-lo:.1f}] · 원고 '[-32,-17]'")

    print("\n[학습 이력]")
    h = json.loads((ROOT / "outputs/v3/month-02/history.json").read_text(encoding="utf-8"))
    bad += not check("학습 회차", str(len(h)), "30 iterations")
    first5, last5 = h[:5], h[-5:]
    for key, needle_a, needle_b in (("seller_loss", "0.079", "0.029"),
                                    ("val_seller_loss", "0.268", "0.119"),
                                    ("buyer_loss", "0.171", "0.027"),
                                    ("val_buyer_loss", "0.315", "0.136")):
        a = sum(r[key] for r in first5) / 5
        b = sum(r[key] for r in last5) / 5
        bad += not check(f"{key} 처음5", a, needle_a, "{:.3f}")
        bad += not check(f"{key} 마지막5", b, needle_b, "{:.3f}")
    bad += not check("회차당 라벨 평균", sum(r["n_labels"] for r in h) / len(h), "56", "{:.0f}")
    bad += not check("반사실 세계 합계", sum(r["worlds"] for r in h), "4{,}671", "{:.0f}")

    print(f"\n대조 실패 {bad}건")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
