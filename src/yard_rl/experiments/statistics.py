"""paired 통계 — 구현계획 03 §5.

같은 seed(공통난수)의 Baseline·대안 정책을 짝지어 평균차와 95% CI 를 계산.
scipy 미사용 — t 임계값은 내장 테이블 (df>30 은 정규근사).
"""
from __future__ import annotations

import math

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
        27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_crit_95(df: int) -> float:
    if df < 1:
        raise ValueError("df >= 1 필요")
    return _T95.get(df, 1.96)


def paired_diff(base: list[float], alt: list[float]) -> dict:
    """diff = alt - base (음수 = 대안이 더 낮음/좋음, 비용성 지표 기준)."""
    if len(base) != len(alt) or len(base) < 2:
        raise ValueError("paired 표본 길이 불일치 또는 n<2")
    n = len(base)
    diffs = [a - b for a, b in zip(alt, base)]
    mean_d = sum(diffs) / n
    var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = t_crit_95(n - 1)
    base_mean = sum(base) / n
    pct = (mean_d / base_mean * 100.0) if abs(base_mean) > 1e-12 else float("nan")
    same_dir = sum(1 for d in diffs if (d < 0) == (mean_d < 0)) if mean_d != 0 else 0
    return {"n": n, "mean_base": base_mean, "mean_alt": sum(alt) / n,
            "mean_diff": mean_d, "ci_lo": mean_d - t * se, "ci_hi": mean_d + t * se,
            "pct_change": pct, "seeds_same_direction": same_dir,
            "significant": (mean_d - t * se) * (mean_d + t * se) > 0}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s)) - 1)))
    return s[idx]
