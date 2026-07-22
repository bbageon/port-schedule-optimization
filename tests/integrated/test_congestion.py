"""혼잡도 컴포넌트 회귀 테스트 — 다이얼·레벨·기존 프리셋 호환·조합."""
from __future__ import annotations

import pytest

from yard_rl.integrated.congestion import (CongestionSpec, congestion,
                                           list_levels)
from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (calibrated_load_params,
                                             generate_terminal_scenario)

PROF = build_calibrated_profile()          # 2크레인
LEVELS = ["idle", "light", "normal", "busy", "rush", "saturation"]


def test_all_levels_compile_and_generate():
    prev = 0
    for lv in LEVELS:
        p = congestion(lv).to_gen_params(PROF)
        sc = generate_terminal_scenario(PROF, 770000, p)
        assert len(sc.jobs) > 0
        assert p.n_external >= prev           # 부하 단조 증가
        prev = p.n_external


def test_normal_matches_calibrated_mid():
    """congestion(normal) == calibrated_load_params(mid) — 기존 프리셋 상위 일반화."""
    a = congestion("normal").to_gen_params(PROF)
    b = calibrated_load_params("mid")
    for f in ("n_external", "arrival_peak_amp", "arrival_peak_width_frac", "fill_ratio",
              "gate_travel_mu_s", "n_vessels", "vessel_moves", "sts_move_interval_s",
              "vessel_deadline_mult", "gaussian"):
        assert getattr(a, f) == getattr(b, f), f


def test_busy_matches_calibrated_high():
    a = congestion("busy").to_gen_params(PROF)
    assert a.n_external == calibrated_load_params("high").n_external == 80


def test_n_external_from_crane_count_and_horizon():
    """전체 크레인수÷블록 아님 — trucks/h/crane × 크레인 × 도착창(h)."""
    c = congestion("normal")                  # 7/h/crane
    assert c.n_external_for(PROF) == 7 * len(PROF.cranes) * 4      # 56
    assert c.n_external_for(PROF, horizon_s=7200.0) == 7 * len(PROF.cranes) * 2


def test_vessel_pressure_dials():
    assert congestion("idle").to_gen_params(PROF).n_vessels == 0        # off
    tight = congestion("normal", vessel_pressure="tight").to_gen_params(PROF)
    assert tight.vessel_deadline_mult == 1.15 and tight.vessel_moves == 24
    assert tight.n_external == 56              # 트럭은 normal 그대로 (직교)


def test_dial_override():
    c = congestion("rush", trucks_per_hour_per_crane=12.0)
    assert c.to_gen_params(PROF).n_external == 12 * 2 * 4           # 96


def test_invalid_inputs():
    with pytest.raises(ValueError):
        congestion("typhoon")
    with pytest.raises(ValueError):
        CongestionSpec(7.0, 1.0, 0.25, 0.30, "explode")
    with pytest.raises(ValueError):
        CongestionSpec(-1.0, 1.0, 0.25, 0.30, "normal")


def test_list_levels_surface():
    lv = {d["level"] for d in list_levels()}
    assert lv == set(LEVELS)
