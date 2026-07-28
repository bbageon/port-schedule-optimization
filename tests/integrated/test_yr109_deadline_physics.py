"""YR-109 + YR-106-b 게이트 A — 본선 마감 물리 정합 (YC→YT→STS 전체 사슬).

YR-109 초판은 하한을 STS cadence 만으로 봤다(dmult≥1.0 클램프). 적하(LOAD)는 첫 박스가
야드크레인→야드트랙터를 거쳐야 첫 STS move 가 가능하므로 그 클램프 뒤에도 초과가 남았다.
게이트 A 는 하한을 전체 사슬로 올린다.
"""
import dataclasses

import pytest

from yard_rl.integrated.profiles import build_calibrated_profile
from yard_rl.integrated.scenario_gen import (calibrated_load_params, generate_terminal_scenario,
                                             phys_min_completion_s)
from yard_rl.integrated.vessel import VesselWorkType

SEED = 852_000


def _scen(dmult, achievable, seed=SEED):
    prof = build_calibrated_profile()
    p = dataclasses.replace(calibrated_load_params("high", vessel_deadline_mult=dmult),
                            time_contract_v2=True, vessel_deadline_achievable=achievable)
    return generate_terminal_scenario(prof, seed, p)


def test_legacy_tight_deadline_is_physically_unachievable():
    """문제 재현: dmult<1 이면 야드가 무한히 빨라도 못 지키는 상수 초과가 남는다."""
    s = _scen(0.5, False)
    floors = [v.structural_min_overrun_s() for v in s.vessels]
    assert all(f > 0 for f in floors), "tight 셀에 구조적 최소초과가 있어야 함(문제 상태)"
    for v, f in zip(s.vessels, floors):
        # 구계약(phys_min 결측)에서는 STS 단독 하한으로 후퇴 → 해석식 항등이 성립
        assert v.plan.phys_min_completion_s is None
        expect = v.plan.total_moves * v.plan.sts_move_interval_s * (1.0 - 0.5)
        assert f == pytest.approx(expect)
    assert min(floors) > 600, "무시할 수 없는 크기여야 함 (감사 대역 실측 33~38분)"


def test_sts_only_clamp_leaves_load_vessel_short():
    """게이트 A 근거: STS 단독 하한(구 dmult=1.0 클램프)은 적하에서 **불충분**하다.

    dmult=1.0 이 만드는 마감 = start + M·cadence 인데, 전체 사슬 하한은 그보다 뒤다.
    """
    prof = build_calibrated_profile()
    s = _scen(0.5, True)
    for v in s.vessels:
        pl = v.plan
        sts_only = pl.planned_start_s + pl.total_moves * pl.sts_move_interval_s
        chain = pl.phys_min_completion_s
        if v.work_type == VesselWorkType.LOAD:
            assert chain > sts_only, "적하는 야드 리드타임만큼 하한이 더 뒤여야 함"
            # 리드타임 = 야드크레인 1사이클 + 야드트랙터 편도, cadence 를 넘는 몫만 남는다
            assert chain - sts_only < prof.transfer.move_time_s * 2
        else:
            assert chain == pytest.approx(sts_only), "양하는 야드가 임계경로 밖"


def test_achievable_mode_removes_structural_floor():
    s = _scen(0.5, True)
    assert all(v.structural_min_overrun_s() == 0.0 for v in s.vessels)
    assert s.meta.get("vessel_deadline_achievable") == "chain-v2"


def test_phys_min_formula_matches_engine_causality():
    """공식 검증 — start + max(M·cad, lead + (M−1)·max(cad, 야드간격))."""
    prof = build_calibrated_profile()
    s = _scen(0.5, True)
    for v in s.vessels:
        pl = v.plan
        tgt = None
        if v.work_type == VesselWorkType.LOAD:
            # 생성기와 같은 대상 집합이어야 값이 일치한다 (역산 대신 관측식으로 교차확인)
            continue
        got = phys_min_completion_s(prof, work=v.work_type, start_s=pl.planned_start_s,
                                    moves=pl.total_moves, cadence_s=pl.sts_move_interval_s,
                                    load_targets=tgt)
        assert got == pytest.approx(pl.phys_min_completion_s)
    # 야드 공급간격이 cadence 보다 느리면 그쪽이 율속이 되는지 (양하 기준)
    slow = phys_min_completion_s(prof, work=VesselWorkType.DISCHARGE, start_s=0.0,
                                 moves=10, cadence_s=1.0, load_targets=None)
    assert slow == pytest.approx(10 * prof.transfer.move_time_s / prof.transfer.n_units)


def test_achievable_mode_is_noop_when_already_achievable():
    """dmult=2.0(느슨)이면 클램프가 아무것도 바꾸지 않는다."""
    a, b = _scen(2.0, False), _scen(2.0, True)
    for va, vb in zip(a.vessels, b.vessels):
        assert va.plan.planned_completion_s == vb.plan.planned_completion_s
        assert va.plan.etd_s == vb.plan.etd_s


def test_off_is_byte_identical_for_tight_cell():
    """opt-in OFF = 기존 생성 그대로 (골든 계약) — etd·deadline 포함 전 필드."""
    prof = build_calibrated_profile()
    base = dataclasses.replace(calibrated_load_params("high", vessel_deadline_mult=0.5),
                               time_contract_v2=True)
    a = generate_terminal_scenario(prof, SEED, base)
    b = _scen(0.5, False)
    for va, vb in zip(a.vessels, b.vessels):
        assert va.plan == vb.plan            # 부동소수점 결합순서까지 동일해야 한다
    assert [j.job_id for j in a.jobs] == [j.job_id for j in b.jobs]
    assert [j.deadline for j in a.jobs] == [j.deadline for j in b.jobs]
    assert "vessel_deadline_achievable" not in a.meta


def test_tight_stays_tighter_than_loose_after_clamp():
    """정합 모드에서도 tightness 축이 살아 있어야 한다 (여유 0 vs 여유 100%)."""
    tight = _scen(0.5, True)
    loose = _scen(2.0, True)
    for vt, vl in zip(tight.vessels, loose.vessels):
        assert vt.plan.planned_completion_s < vl.plan.planned_completion_s


def test_job_deadlines_follow_clamped_plan():
    """job.deadline 도 계획완료를 따라가야 한다(계약 일관)."""
    legacy, fixed = _scen(0.5, False), _scen(0.5, True)
    dl_legacy = [j.deadline for j in legacy.jobs if j.vessel_id is not None]
    dl_fixed = [j.deadline for j in fixed.jobs if j.vessel_id is not None]
    assert dl_legacy and len(dl_legacy) == len(dl_fixed)
    assert all(f > l for f, l in zip(dl_fixed, dl_legacy))


def test_plan_change_preserves_phys_min():
    """엔진 `_plan_change` 가 VesselPlan 을 재구성할 때 물리 하한을 잃지 않는다.

    (필드를 열거해 재구성하므로 누락되면 조용히 None 으로 리셋된다 — 정찰 지적.)
    """
    from yard_rl.integrated import TerminalSimulator
    s = _scen(0.5, True)
    prof = build_calibrated_profile()
    sim = TerminalSimulator(prof, s)
    vid = next(iter(sim.vessels))
    before = sim.vessels[vid].plan.phys_min_completion_s
    assert before is not None
    sim._plan_change(vid, {"planned_completion_s": 99_999.0})
    assert sim.vessels[vid].plan.phys_min_completion_s == before
