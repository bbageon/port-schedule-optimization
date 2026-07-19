"""YR-002 재기준화 (D5·D1) — 문헌 보정 프로파일 v2·부하 현실화 계약.

핵심 계약: ① 기본 TerminalGenParams 시나리오는 변경 전과 **바이트 동일**
(피크 warp 는 amp=0 항등·추가 난수 소비 없음) ② calibrated 프로파일은 문헌
보정값(ARMG 속도·10열 6단·gate 210s) ③ 피크는 창 안에 질량을 집중시킨다.
"""
import hashlib

import pytest

from yard_rl.integrated.fixtures import build_integrated_profile
from yard_rl.integrated.profiles import build_calibrated_profile, build_dgt_approx_profile
from yard_rl.integrated.scenario_gen import (
    TerminalGenParams, _peak_warp, calibrated_load_params, generate_terminal_scenario)


def _fingerprint(profile, seed, params=None):
    sc = generate_terminal_scenario(profile, seed, params or TerminalGenParams())
    parts = [f"{j.job_id}|{j.flow}|{j.actual_gate_in}|{j.actual_block_arrival}|{j.provided_eta}"
             for j in sc.jobs]
    for cid in sorted(sc.containers):
        c = sc.containers[cid]
        parts.append(f"{cid}|{c.bay}|{c.row}|{c.tier}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def test_default_scenario_frozen_golden():
    """변경 전(commit 8ddbef7 직전) 실측 지문 — 기본 파라미터 시나리오 불변 guard."""
    assert _fingerprint(build_integrated_profile(), 530000) == "edfd8dde05fc469d"
    assert _fingerprint(build_integrated_profile(), 700000) == "f93ae26660eaf96a"
    assert _fingerprint(build_dgt_approx_profile(), 700000) == "53f174ae3614c3d6"


def test_peak_warp_identity_and_bounds():
    for u in (0.0, 0.1, 0.37, 0.5, 0.99):
        assert _peak_warp(u, 0.0, 0.5, 0.25) == u          # amp=0 항등 (골든 보존 계약)
    prev = -1.0
    for i in range(101):                                   # 단조·경계 [0,1]
        x = _peak_warp(i / 100.0, 2.0, 0.5, 0.25)
        assert 0.0 <= x <= 1.0 + 1e-12 and x > prev
        prev = x
    assert _peak_warp(1.0, 2.0, 0.5, 0.25) == pytest.approx(1.0)


def test_peak_concentrates_arrivals():
    prof = build_calibrated_profile()
    params = calibrated_load_params("mid", gaussian=False)
    sc = generate_terminal_scenario(prof, 700500, params)
    arr = [j.actual_block_arrival for j in sc.jobs if j.actual_block_arrival is not None]
    lo, hi = 0.375 * params.horizon_s, 0.625 * params.horizon_s   # 피크 창 (중앙 25%)
    share = sum(1 for a in arr if lo <= a < hi) / len(arr)
    assert share > 0.34                                    # 균등 0.25 대비 유의 집중 (기대 0.4)
    sc2 = generate_terminal_scenario(prof, 700500, params)
    assert [j.actual_block_arrival for j in sc2.jobs] == [j.actual_block_arrival
                                                          for j in sc.jobs]  # 결정론


def test_calibrated_profile_literature_values():
    p = build_calibrated_profile()
    assert p.terminal_id == "SNP-ARMG-STD" and p.assumed is True
    assert p.block.row_count == 10 and p.block.tier_max == 6      # 공식 장비 페이지
    assert len(p.cranes) == 2
    for c in p.cranes:
        assert c.gantry_speed_mps == pytest.approx(4.0)           # Kalmar ASC 하한
        assert c.hoist_speed_loaded_mps == pytest.approx(0.58)
    assert p.gate_travel_estimate_s == pytest.approx(210.0)       # 문헌 2~5분 중앙값
    assert p.long_wait_sla_s == pytest.approx(1800.0)             # 안전운임제 warning 단계


def test_calibrated_load_params_levels():
    mid, high = calibrated_load_params("mid"), calibrated_load_params("high")
    cur = calibrated_load_params("current")
    assert (mid.n_external, high.n_external, cur.n_external) == (56, 80, 40)
    for p in (mid, high, cur):
        assert p.arrival_peak_amp > 0 and p.gate_travel_mu_s == pytest.approx(210.0)
    assert calibrated_load_params("mid", n_external=60).n_external == 60   # override
    with pytest.raises(ValueError):
        calibrated_load_params("extreme")
    with pytest.raises(ValueError):
        TerminalGenParams(arrival_peak_amp=-0.5)
    with pytest.raises(ValueError):
        TerminalGenParams(arrival_peak_width_frac=0.0)


def test_nondefault_load_recorded_in_meta():
    prof = build_calibrated_profile()
    sc = generate_terminal_scenario(prof, 700501, calibrated_load_params("mid"))
    assert sc.meta["arrival_peak"] == (1.0, 0.5, 0.25)
    assert sc.meta["gate_travel_mu_s"] == pytest.approx(210.0)
    sc0 = generate_terminal_scenario(prof, 700501, TerminalGenParams())
    assert "arrival_peak" not in sc0.meta and "gate_travel_mu_s" not in sc0.meta
