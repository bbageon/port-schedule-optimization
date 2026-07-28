"""YR-109 — 본선 마감 물리 정합 (opt-in) + 구조적 최소초과 진단."""
import dataclasses

import pytest

from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import calibrated_load_params, generate_terminal_scenario

SEED = 852_000


def _scen(dmult, achievable):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params("high", vessel_deadline_mult=dmult),
                            time_contract_v2=True, vessel_deadline_achievable=achievable)
    return generate_terminal_scenario(prof, SEED, p)


def test_legacy_tight_deadline_is_physically_unachievable():
    """문제 재현: dmult<1 이면 야드가 무한히 빨라도 못 지키는 상수 초과가 남는다."""
    s = _scen(0.5, False)
    floors = [v.structural_min_overrun_s() for v in s.vessels]
    assert all(f > 0 for f in floors), "tight 셀에 구조적 최소초과가 있어야 함(문제 상태)"
    for v, f in zip(s.vessels, floors):
        # 해석식 항등: 상수 초과 = moves·cadence·(1−dmult)  (시드마다 크기는 다름)
        expect = v.plan.total_moves * v.plan.sts_move_interval_s * (1.0 - 0.5)
        assert f == pytest.approx(expect)
    assert min(floors) > 600, "무시할 수 없는 크기여야 함 (감사 대역 실측 33~38분)"


def test_achievable_mode_removes_structural_floor():
    s = _scen(0.5, True)
    assert all(v.structural_min_overrun_s() == 0.0 for v in s.vessels)
    assert s.meta.get("vessel_deadline_achievable") is True


def test_achievable_mode_is_noop_when_already_achievable():
    """dmult>=1 이면 클램프가 아무것도 바꾸지 않는다 (loose 셀 불변)."""
    a, b = _scen(2.0, False), _scen(2.0, True)
    for va, vb in zip(a.vessels, b.vessels):
        assert va.plan.planned_completion_s == vb.plan.planned_completion_s
        assert va.plan.etd_s == vb.plan.etd_s


def test_off_is_byte_identical_for_tight_cell():
    """opt-in OFF = 기존 생성 그대로 (골든 계약)."""
    prof = build_calibrated_profile()
    base = dataclasses.replace(calibrated_load_params("high", vessel_deadline_mult=0.5),
                               time_contract_v2=True)
    a = generate_terminal_scenario(prof, SEED, base)
    b = _scen(0.5, False)
    for va, vb in zip(a.vessels, b.vessels):
        assert va.plan.planned_completion_s == vb.plan.planned_completion_s
    assert [j.job_id for j in a.jobs] == [j.job_id for j in b.jobs]
    assert "vessel_deadline_achievable" not in a.meta


def test_tight_stays_tighter_than_loose_after_clamp():
    """정합 모드에서도 tightness 축이 살아 있어야 한다 (여유 0 vs 여유 100%)."""
    tight = _scen(0.5, True)
    loose = _scen(2.0, True)
    for vt, vl in zip(tight.vessels, loose.vessels):
        assert vt.plan.planned_completion_s < vl.plan.planned_completion_s


def test_job_deadlines_follow_clamped_plan():
    """job.deadline 도 같은 dmult 를 쓰므로 함께 이동해야 한다(계약 일관)."""
    legacy, fixed = _scen(0.5, False), _scen(0.5, True)
    dl_legacy = [j.deadline for j in legacy.jobs if j.vessel_id is not None]
    dl_fixed = [j.deadline for j in fixed.jobs if j.vessel_id is not None]
    assert dl_legacy and len(dl_legacy) == len(dl_fixed)
    assert all(f > l for f, l in zip(dl_fixed, dl_legacy))
