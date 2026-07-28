"""정확한 t·χ² 분위수 (의존성 없음) — YR-106-b 게이트 C.

**왜 필요한가**: evalkit 이 t 값을 df 별 **하드코딩 표**로 들고 있었다. 표에 없는 df 는
조용히 대표값으로 대체돼 검정력 계산이 틀어졌다 (실측: `_T80` 에 df 21·22·24~28 이
없어 0.920(df=5 값)으로 떨어짐 — 참값 ~0.859 대비 7% 과대 → MDE 과대·필요 n 과대).
n=21·42 로 올리려면 임의 df 가 필요하므로 표를 **분포 함수 계산**으로 대체한다.

구현: 정칙화 불완전 베타(연분수)로 t 의 CDF, 정칙화 불완전 감마(급수+연분수)로 χ² 의 CDF,
분위수는 이분법. numpy·scipy 불필요, 결정론적, 배정도에서 ~1e-10 수렴.
"""
from __future__ import annotations

import math

_MAXIT = 300
_EPS = 3e-16
_FPMIN = 1e-300


# ----------------------------------------------------------------- 불완전 베타
def _betacf(a: float, b: float, x: float) -> float:
    """연분수 (Lentz) — I_x(a,b) 의 연분수 부분."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """정칙화 불완전 베타 I_x(a,b) ∈ [0,1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


# ----------------------------------------------------------------- 불완전 감마
def _gser(a: float, x: float) -> float:
    ap, s, delta = a, 1.0 / a, 1.0 / a
    for _ in range(_MAXIT):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * _EPS:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    b, c = x + 1.0 - a, 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc_lower(a: float, x: float) -> float:
    """정칙화 하부 불완전 감마 P(a,x) ∈ [0,1]."""
    if x <= 0.0:
        return 0.0
    return _gser(a, x) if x < a + 1.0 else 1.0 - _gcf(a, x)


# ----------------------------------------------------------------- 분포
def t_cdf(t: float, df: float) -> float:
    """스튜던트 t 누적분포."""
    if df <= 0:
        raise ValueError("df > 0")
    x = df / (df + t * t)
    tail = 0.5 * betainc(0.5 * df, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def chi2_cdf(x: float, k: float) -> float:
    return gammainc_lower(0.5 * k, 0.5 * x) if x > 0 else 0.0


def _bisect(fn, target: float, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def t_ppf(p: float, df: float) -> float:
    """t 분위수. p 는 (0,1) 의 **누적확률** (양측 95% 상한은 p=0.975)."""
    if not 0.0 < p < 1.0:
        raise ValueError("0 < p < 1")
    if p == 0.5:
        return 0.0
    hi = 2.0
    while t_cdf(hi, df) < max(p, 1.0 - p):
        hi *= 2.0
        if hi > 1e12:
            break
    v = _bisect(lambda t: t_cdf(t, df), max(p, 1.0 - p), 0.0, hi)
    return v if p > 0.5 else -v


def chi2_ppf(p: float, k: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("0 < p < 1")
    hi = max(2.0, k)
    while chi2_cdf(hi, k) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    return _bisect(lambda x: chi2_cdf(x, k), p, 0.0, hi)


def sd_upper_conf(sd: float, df: int, conf: float = 0.80) -> float:
    """표본 표준편차의 **상측 신뢰한계** — 파일럿 sd 의 불확실성 반영.

    (n−1)s²/σ² ~ χ²_{n−1} 이므로 σ ≤ s·√(df / χ²_{1−conf, df}) 가 conf 신뢰도로 성립한다.
    파일럿 n=8 이면 80% 상한이 약 1.3×s — 필요 표본수가 ~1.7배로 늘어난다.
    이 보정을 빼면 **필요 시드를 체계적으로 과소추정**한다.
    """
    if df < 1 or sd <= 0:
        return sd
    return sd * math.sqrt(df / chi2_ppf(1.0 - conf, df))
