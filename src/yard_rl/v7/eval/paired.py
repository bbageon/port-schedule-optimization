"""날 단위 짝비교 통계 — 부호검정 · 정확 윌콕슨 · 층화 부트스트랩 ([[YR-314]]).

논문 스크립트 `scripts/v3/paired_tests.py` 와 **같은 검정·같은 설정**을 v4 안에 둔다.
그 스크립트는 scipy 를 쓰는데 실행 환경(WSL venv)에 scipy 가 없어, 정확검정을
**순수 파이썬으로** 구현했다 — 28일이면 정확 분포를 그대로 셀 수 있다.

    부호검정     양측 정확 이항 — 동점(차이 0)은 뺀다
    윌콕슨       양측 정확 부호순위 — 순위합의 정확 분포를 DP 로 센다 (근사 없음)
    부트스트랩   하루 평균 차이의 95% 백분위 구간 · 층(부하 구간) 안에서만 되뽑는다

■ 왜 정확검정인가
  논문 [[YR-233]] 이 정확검정으로 수를 냈다 — 근사식을 쓰면 값이 달라진다
  (예: 정확 0.0017 vs 정규근사 0.0026). 결론은 같아도 논문에 싣는 수는 하나여야
  하므로 재현 가능한 정확검정으로 고정한다.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb

#: 되뽑기 횟수 · 난수 시드 — 논문 스크립트와 같다 (재현용)
N_BOOT = 20_000
BOOT_SEED = 20260830


def sign_test(diff) -> dict:
    """양측 정확 이항 부호검정. `diff < 0` 이 *"학습 쪽이 쌌다"* 다."""
    d = [x for x in diff if abs(x) > 1e-9]
    ties = len(diff) - len(d)
    n = len(d)
    win = sum(1 for x in d if x < 0)
    if n == 0:
        return {"n": 0, "win": 0, "ties": ties, "p": 1.0}
    k = min(win, n - win)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"n": n, "win": win, "ties": ties, "p": float(min(1.0, 2 * tail))}


def _midranks(vals):
    """|차이| 의 평균순위(동점은 평균) — 정수 정확분포를 위해 **2배**한 정수로 낸다."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks2 = [0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r2 = (i + 1) + (j + 1)               # 평균순위 × 2 = (첫 순위 + 끝 순위)
        for k in range(i, j + 1):
            ranks2[order[k]] = r2
        i = j + 1
    return ranks2


def wilcoxon_exact(diff) -> dict:
    """양측 **정확** 윌콕슨 부호순위검정 — 차이의 크기까지 쓴다.

    H0 아래서 각 순위는 독립적으로 1/2 확률로 양쪽에 붙는다. 순위합 W+ 의 정확
    분포를 DP 로 센다(순위를 2배한 정수로 동점 평균순위까지 정확히 다룬다).
    동점(차이 0)은 뺀다.
    """
    d = [x for x in diff if abs(x) > 1e-9]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "p": 1.0}
    r2 = _midranks([abs(x) for x in d])
    w2 = sum(r for r, x in zip(r2, d) if x > 0)       # W+ × 2
    total = sum(r2)
    # dp[s] = 순위합(×2)이 s 인 부분집합 수
    dp = [0] * (total + 1)
    dp[0] = 1
    for r in r2:
        for s in range(total, r - 1, -1):
            dp[s] += dp[s - r]
    denom = 2 ** n
    lo = sum(dp[: w2 + 1]) / denom                 # P(W+ ≤ w)
    hi = sum(dp[w2:]) / denom                      # P(W+ ≥ w)
    return {"n": n, "w_plus": w2 / 2.0, "p": float(min(1.0, 2 * min(lo, hi)))}


def boot_ci(diff, strata=None, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """하루 평균 차이의 95% 백분위 부트스트랩 구간.

    `strata` 를 주면 **층화** — 부하 구간별로 그 층의 날짜 안에서만 되뽑아 설계가
    정한 수요 구성을 보존한다. 돌려주는 `p_below0` 는 평균이 음수인 되뽑기 비율.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    d = np.asarray(list(diff), dtype=float)
    if len(d) == 0:
        return {"lo": 0.0, "hi": 0.0, "p_below0": 0.0, "mean": 0.0}
    if strata is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        groups: dict = {}
        for i, s in enumerate(strata):
            groups.setdefault(s, []).append(i)
        means = np.zeros(n_boot)
        for g in groups.values():
            gi = np.asarray(g)
            pick = rng.integers(0, len(gi), size=(n_boot, len(gi)))
            means += d[gi[pick]].sum(axis=1)
        means /= len(d)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "p_below0": float((means < 0).mean()),
            "mean": float(d.mean())}


def paired_summary(phi_a: dict, phi_b: dict, strata: dict | None = None) -> dict:
    """A(학습) − B(규칙) 를 날마다 짝지어 세 검정을 한 번에.

    `phi_a`, `phi_b` : {날 → Φ(원)}. 두 쪽에 다 있는 날만 쓴다.
    `strata`         : {날 → 층 이름} (부하 구간). 없으면 비층화.
    """
    days = sorted(set(phi_a) & set(phi_b))
    diff = [phi_a[k] - phi_b[k] for k in days]
    ratio = [(phi_a[k] - phi_b[k]) / max(1e-9, phi_b[k]) for k in days]
    st = sorted(ratio)
    med = st[len(st) // 2] if st else 0.0
    return {
        "days": days, "n": len(days),
        "sign": sign_test(diff),
        "wilcoxon": wilcoxon_exact(diff),
        "boot": boot_ci(diff, [strata[k] for k in days] if strata else None),
        "mean_diff_krw": (sum(diff) / len(diff)) if diff else 0.0,
        "median_ratio": med,
        "mean_ratio": (sum(ratio) / len(ratio)) if ratio else 0.0,
    }
