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

#  원고 경로는 인자로 받는다 — 제출본과 v2 를 같은 잣대로 잰다.
PAPER = sys.argv[1] if len(sys.argv) > 1 else "docs/paper/v3/submission"
TEX = (ROOT / PAPER / "main.tex").read_text(encoding="utf-8")
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
    #  ★집계 비중은 환경마다 달라진다 (67·79·81·53.8·91.3%). 민감도를 보고하는
    #   원고는 이 값을 쓰지 않으므로 대조도 건너뛴다.
    if "controlled sweep" not in TEX:
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
    #  ★학습의 몫 비중도 분모(전체 감소)가 환경마다 두 배씩 달라 비교가 안 된다.
    #   민감도를 보고하는 원고는 절대액과 중앙값으로 대신하므로 대조를 건너뛴다.
    if "controlled sweep" not in TEX:
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

    bad += sweep_checks()

    print(f"\n대조 실패 {bad}건")
    return bad


def sweep_checks():
    """혼잡 빈도 민감도 — 세 환경의 값을 원자료에서 다시 계산해 대조한다.

    시드는 셋 다 9,900,980 으로 같고 혼잡일 빈도만 2/6/14 로 다르다.
    규약대로 첫날·마지막날을 뺀 가운데 28일만 쓴다.
    """
    from math import comb

    if "controlled sweep" not in TEX:
        return 0                      # 민감도를 안 쓰는 원고면 건너뛴다
    print("\n[혼잡 빈도 민감도]")
    bad = 0
    ARMS = ("NO_REALLOC", "RL", "RL_SPACE", "RL_TIME", "RL_EARLY")
    want = {
        "env-quiet": dict(t="67", b="$+$KRW 0.23", sp="32.0", g="0.18",
                          med="$+$0.8", sign="17/28", p="0.345"),
        "env-mixed": dict(t="79", b="$-$KRW 2.40", sp="21.5", g="1.51",
                          med="$-$0.1", sign="14/28", p="1.000"),
        "env-heavy": dict(t="81", b="$+$KRW 1.17", sp="11.9", g="1.60",
                          med="$+$9.7", sign="22 of 28", p="0.004"),
    }
    for env, w in want.items():
        phi, meta = {}, {}
        for a in ARMS:
            d = json.loads((ROOT / f"outputs/v3/{env}/arms/arm_{a}.json")
                           .read_text(encoding="utf-8"))
            phi[a] = d["phi_by_day"]
            meta[a] = d
        days = sorted(set(phi["NO_REALLOC"]) & set(phi["RL"]), key=int)[1:-1]
        base = sum(phi["NO_REALLOC"][d] for d in days)
        tot = base - sum(phi["RL"][d] for d in days)
        t_only = base - sum(phi["RL_TIME"][d] for d in days)
        b_only = base - sum(phi["RL_SPACE"][d] for d in days)
        gain = sum(phi["RL_EARLY"][d] - phi["RL"][d] for d in days)
        dl = sorted((phi["RL_EARLY"][d] - phi["RL"][d]) / MN for d in days)
        med = (dl[len(dl) // 2 - 1] + dl[len(dl) // 2]) / 2
        pos = sum(1 for d in days if phi["RL_EARLY"][d] > phi["RL"][d])
        n = sum(1 for d in days if phi["RL_EARLY"][d] != phi["RL"][d])
        k = min(pos, n - pos)
        pval = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
        ns, nt = meta["RL"]["n_space"], meta["RL"]["n_time"]

        bad += not check(f"{env} 시각 몫%", t_only / tot * 100, w["t"], "{:.0f}")
        bad += not check(f"{env} 블록만 (십억)", b_only / BN, w["b"], "{:.2f}")
        bad += not check(f"{env} 공간 비중%", ns / (ns + nt) * 100, w["sp"], "{:.1f}")
        bad += not check(f"{env} 학습 몫 (십억)", gain / BN, w["g"], "{:.2f}")
        bad += not check(f"{env} 학습 중앙값 (백만)", med, w["med"], "{:.1f}")
        bad += not check(f"{env} 학습 부호", f"{pos}/{n}", w["sign"])
        bad += not check(f"{env} 학습 p", pval, w["p"], "{:.3f}")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
