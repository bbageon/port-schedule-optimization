"""YR-083 step1·5 — 구조계약 스키마·compiler·호환성 판정 회귀 테스트.

엔진 소비(mask/resolver) 미포함 — 계약 표현·판정·"조용한 유실 0" 불변만 검사.
"""
from __future__ import annotations

from yard_rl.contract.structure import (CraneSide, StructureContract, VehicleType)
from yard_rl.integrated.structure_compiler import (CompatVerdict, compile_all,
                                                   compile_terminal)

PROVISIONAL = {"PNIT", "PNC", "HJNC", "HPNT"}


def test_compile_all_ten_terminals():
    assert len(compile_all()) == 10


def test_provisional_zero_shot():
    for tid in PROVISIONAL:
        cs = compile_terminal(tid)
        assert cs.engine_verdict is CompatVerdict.ZERO_SHOT_COMPATIBLE
        assert cs.target_verdict is CompatVerdict.ZERO_SHOT_COMPATIBLE
        assert cs.unsupported == ()
        assert not cs.contract.is_role_split


def test_hjnc_orientation_recorded_not_dropped():
    """확인된 orientation(수평)은 엔진에 자리 없어도 engine_ignored 로 명시 — 조용한 유실 0.

    단 실행 불변이라 ZERO_SHOT 은 유지(engine_ignored 는 판정을 막지 않는다)."""
    cs = compile_terminal("HJNC")
    ignored = {a for a, _ in cs.engine_ignored}
    assert "layout.orientation" in ignored           # HORIZONTAL confirmed
    assert "layout.block_count" in ignored            # 21블록 confirmed (엔진 단일블록)
    assert cs.engine_verdict is CompatVerdict.ZERO_SHOT_COMPATIBLE
    assert cs.unsupported == ()                        # ignored 는 unsupported 아님


def test_dgt_role_split_unsupported_now_adapt_later():
    cs = compile_terminal("DGT")
    assert cs.contract.is_role_split
    assert cs.engine_verdict is CompatVerdict.STRUCTURE_UNSUPPORTED
    assert cs.target_verdict is CompatVerdict.SCHEMA_ADAPTATION_REQUIRED
    asp = {a for a, _ in cs.unsupported}
    assert "crane_role_split" in asp
    assert "transfer_fleet" in asp          # AGV 미모형
    assert cs.contract.sides == {CraneSide.LANDSIDE, CraneSide.WATERSIDE}
    assert VehicleType.AGV in cs.contract.vehicle_types


def test_sc_and_conventional_unsupported_both():
    for tid in ("BNCT", "BCT", "BPT_SINSEONDAE", "BPT_GAMMAN", "HKT"):
        cs = compile_terminal(tid)
        assert cs.engine_verdict is CompatVerdict.STRUCTURE_UNSUPPORTED
        assert cs.target_verdict is CompatVerdict.STRUCTURE_UNSUPPORTED


def test_no_silent_drop_invariant():
    """구조 사실이 있으면 반드시 supported 또는 unsupported 로 계상 — 조용한 유실 0."""
    for cs in compile_all():
        # 역할분리면 미지원 목록에 반드시 등장
        if cs.contract.is_role_split:
            assert any(a == "crane_role_split" for a, _ in cs.unsupported)
        # 비엔진 차종(AGV/SC)이면 반드시 등장
        nonengine = cs.contract.vehicle_types - {VehicleType.YT, VehicleType.EXTERNAL_TRUCK}
        if nonengine:
            assert any(a == "transfer_fleet" for a, _ in cs.unsupported)
        # 모든 unsupported 는 사유 문자열을 가진다
        assert all(why for _, why in cs.unsupported)
        # ZERO_SHOT 는 미지원 0 과 필요충분 (engine_ignored 는 무관)
        assert (cs.engine_verdict is CompatVerdict.ZERO_SHOT_COMPATIBLE) == (not cs.unsupported)
        # engine_ignored 도 전부 사유를 가진다 (조용한 유실 0)
        assert all(why for _, why in cs.engine_ignored)


def test_schema_properties():
    cs = compile_terminal("DGT")
    c = cs.contract
    assert isinstance(c, StructureContract)
    assert c.is_role_split is True
    # transfer_points 는 육/해측 분리 시에만
    assert {tp.side for tp in c.transfer_points} == {CraneSide.LANDSIDE, CraneSide.WATERSIDE}
    hjnc = compile_terminal("HJNC").contract
    assert hjnc.transfer_points == ()
    assert hjnc.sides == {CraneSide.SHARED}


def test_unsupported_reasons_in_report_reasons():
    """미지원 각 항목의 사유가 reasons 에도 박제되어 조용히 버려지지 않음."""
    cs = compile_terminal("DGT")
    joined = " ".join(cs.reasons)
    for asp, _ in cs.unsupported:
        assert any(asp in r or "미지원" in r for r in cs.reasons)
    assert "재증류" in joined or "재학습" in joined
