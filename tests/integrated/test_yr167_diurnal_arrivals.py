"""5차 계약 — 24시간 이중 피크 도착 생성 계약 (사전등록 §A 동결본 검사).

사전등록: .claude/docs/strategy-history/2026-08-11-24시간-이중피크-상수-유도-사전등록.md
여기 값들은 **문서에서 온 것**이고, 코드가 문서를 지키는지 검사한다(반대 아님).
"""
import pytest

from yard_rl.integrated.terminal_stream import (DIURNAL_DAY_S, DIURNAL_DAY_TOTAL,
                                                diurnal_arrivals, diurnal_rate)


def _hourly(ts):
    h = [0] * 24
    for t in ts:
        h[int(t // 3600)] += 1
    return h


def test_total_and_window():
    ts = diurnal_arrivals(11)
    assert len(ts) == DIURNAL_DAY_TOTAL == 3600
    assert 0.0 <= ts[0] and ts[-1] < DIURNAL_DAY_S
    assert ts == sorted(ts)


def test_night_floor_matches_frozen_value():
    """야간 저점(하루 최소) = 일평균의 15% = 22.5대/h (동결값).

    특정 시각의 λ 는 저점 + 봉우리 꼬리라서 새벽(2~4시)에서만 저점과 일치한다
    (예: 22시는 오후 봉우리 3.5σ 꼬리가 남아 23.4대/h) — 계약은 **최소값**이다.
    """
    grid = [diurnal_rate(x * 60) * 3600 for x in range(24 * 60)]
    assert min(grid) == pytest.approx(22.5, abs=0.05)
    for h in (2, 3, 4):
        assert diurnal_rate(h * 3600) * 3600 == pytest.approx(22.5, abs=0.05)


def test_peak_windows_are_morning_and_afternoon():
    """봉우리 두 개가 오전 10~12시·오후 13~17시 창 안에 있어야 한다(국내 실측)."""
    ts = diurnal_arrivals(11)
    h = _hourly(ts)
    morning = max(range(9, 13), key=lambda i: h[i])
    afternoon = max(range(13, 18), key=lambda i: h[i])
    assert 10 <= morning <= 12 and 13 <= afternoon <= 17
    assert h[morning] > 3 * h[3] and h[afternoon] > 3 * h[3]   # 야간 대비 뚜렷


def test_instantaneous_peak_matches_corrected_derivation():
    """순간 최대 λ = 488.7대/h @ 11.15시 (A7 정정본)."""
    grid = [(x / 60.0, diurnal_rate(x * 60) * 3600) for x in range(24 * 60)]
    hh, lam = max(grid, key=lambda p: p[1])
    assert lam == pytest.approx(488.7, abs=1.0)
    assert hh == pytest.approx(11.15, abs=0.1)


def test_planned_hourly_peak_is_w11_reference():
    """계획 시간대별 최대(적분) = 461.2대 — W11 비교 기준."""
    hourly = [sum(diurnal_rate((h + i / 60) * 3600) * 60 for i in range(60))
              for h in range(24)]
    assert max(hourly) == pytest.approx(461.2, abs=1.0)
    assert sum(hourly) == pytest.approx(3600.0, abs=1.0)


def test_deterministic_and_seed_varies_sample_not_shape():
    a, b = diurnal_arrivals(11), diurnal_arrivals(12)
    assert diurnal_arrivals(11) == a          # 같은 시드 = 같은 명단
    assert a != b                              # 다른 시드 = 다른 표본
    ha, hb = _hourly(a), _hourly(b)
    # 형상은 시드와 무관 — 시간대별 분포가 서로 10% 안
    for i in range(24):
        assert abs(ha[i] - hb[i]) <= max(6, 0.10 * max(ha[i], hb[i]))
