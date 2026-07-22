"""YR-082-A — 터미널 선택기 회귀 테스트 (10개 전부 선택·실행 + 충실도 게이트)."""
from __future__ import annotations

import pytest

from yard_rl.integrated.scenario_gen import (calibrated_load_params,
                                             generate_terminal_scenario)
from yard_rl.integrated.terminal_registry import (StructureBlockedError,
                                                  build_stress_profile,
                                                  faithful_terminals, list_terminals)

FAITHFUL = {"PNIT", "PNC", "HJNC", "HPNT"}
ALL10 = FAITHFUL | {"DGT", "BNCT", "BCT", "BPT_SINSEONDAE", "BPT_GAMMAN", "HKT"}


def test_all_ten_selectable_and_runnable():
    """10개 전부 선택 → 실행 프로파일 + 시나리오 생성 (막지 않는다)."""
    ids = {t["id"] for t in list_terminals()}
    assert ids == ALL10
    for tid in ALL10:
        env = build_stress_profile(tid)
        sc = generate_terminal_scenario(env.profile, 770000, calibrated_load_params("mid"))
        assert len(sc.jobs) > 0
        assert env.profile.terminal_id == f"{tid}-STRESS"


def test_faithful_flag_and_warnings():
    for tid in FAITHFUL:
        env = build_stress_profile(tid)
        assert env.faithful is True
        assert env.warnings == ()
        assert "NAMEPLATE" not in env.data_grade
    for tid in ALL10 - FAITHFUL:
        env = build_stress_profile(tid)
        assert env.faithful is False
        assert env.warnings                       # 미충실 경고 존재
        assert "구조미충실" in env.data_grade
        assert any("주장 금지" in w for w in env.warnings)


def test_faithful_terminals_list():
    assert set(faithful_terminals()) == FAITHFUL


def test_require_faithful_blocks_nonfaithful():
    for tid in FAITHFUL:
        build_stress_profile(tid, require_faithful=True)         # 통과
    for tid in ALL10 - FAITHFUL:
        with pytest.raises(StructureBlockedError):
            build_stress_profile(tid, require_faithful=True)


def test_transfer_kind_nameplate():
    assert build_stress_profile("DGT").profile.transfer.kind == "AGV"
    assert build_stress_profile("BNCT").profile.transfer.kind == "SC"
    assert build_stress_profile("HJNC").profile.transfer.kind == "YT"


def test_pnit_rail_gap_overlay():
    """PNIT 확인 레일간격(28.4)→열폭 2.84 오버레이 (유일한 실물리 확인값 차이)."""
    pnit = build_stress_profile("PNIT").profile
    hjnc = build_stress_profile("HJNC").profile
    assert pnit.block.row_width_m == 2.84
    assert hjnc.block.row_width_m == 3.1
