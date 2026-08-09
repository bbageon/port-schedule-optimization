"""YR-150 — 본선 배치 독립화·flow fallback 원장 계약 (외부 감사 반영 2026-08-09).

구판 build_fixed_wip 은 본선을 이름순 앞 12블록(Y01~Y12)에만 두고 시작 시각도 블록
순서와 단조 상관이라 "뒷번호 = 본선 없음"이 시드 구조에 새겨졌다(인공 편향). 정정 후
계약: 블록·슬롯·종류 세 축 독립 추첨, 슬롯 5%~95%, 양하/적하 6:6, seedbank 전 블록
노출, flow fallback 은 원장 기록 + 공식 자격시험 0 요구.
"""
import pytest

from yard_rl.integrated.profiles import build_h21_profile
from yard_rl.integrated.terminal_stream import (ObservationContract,
                                                TerminalStreamParams,
                                                build_fixed_wip,
                                                vessel_placement)
from yard_rl.integrated.yard_layout import terminal_layout

SEED = 6_300_000
OBS = ObservationContract()
LAYOUT = terminal_layout()


def _pl(seed: int) -> dict:
    return vessel_placement(LAYOUT, seed, TerminalStreamParams(load_4h=63), OBS)


# ------------------------------------------------------------------ 배치 추첨 계약 (경량)
def test_deterministic_same_seed():
    assert _pl(SEED) == _pl(SEED)


def test_twelve_distinct_blocks_within_layout():
    pl = _pl(SEED)
    assert len(pl) == 12 and set(pl) <= set(LAYOUT.ids)


def test_work_type_balance_6_6():
    works = [v["work"] for v in _pl(SEED).values()]
    assert works.count("DISCHARGE") == 6 and works.count("LOAD") == 6


def test_slots_are_permutation_spanning_5_to_95():
    pl = _pl(SEED)
    ks = sorted(v["slot_k"] for v in pl.values())
    assert ks == list(range(12))                      # 슬롯 12개 전부·중복 없음
    starts = sorted(v["start_s"] for v in pl.values())
    assert starts[0] == pytest.approx(0.05 * OBS.observe_s)
    assert starts[-1] == pytest.approx(0.95 * OBS.observe_s)   # 구판은 87.5% 까지만


def test_placement_varies_by_seed():
    base = _pl(SEED)
    assert any(_pl(SEED + d) != base for d in (1, 2, 3))


def test_block_order_start_time_correlation_removed():
    """블록 이름순으로 늘어놓았을 때 시작 시각이 단조증가하면 안 된다(구판 결함 재현 방지)."""
    monotone = 0
    for d in range(5):
        pl = _pl(SEED + d)
        starts = [pl[b]["start_s"] for b in sorted(pl)]
        if starts == sorted(starts):
            monotone += 1
    assert monotone == 0


def test_seedbank_exposure_covers_all_blocks():
    """seedbank 폭(50 시드)에서 21블록 전부가 최소 1회 본선을 받아야 한다(감사 요구)."""
    exposure = {b: 0 for b in LAYOUT.ids}
    for d in range(50):
        for b in _pl(SEED + d):
            exposure[b] += 1
    assert min(exposure.values()) >= 1


def test_more_processes_than_blocks_rejected():
    with pytest.raises(ValueError):
        vessel_placement(LAYOUT, SEED,
                         TerminalStreamParams(load_4h=63, vessels_total=22), OBS)


# ------------------------------------------------------------------ 생성기 통합 (중량 1회)
@pytest.fixture(scope="module")
def built():
    return build_fixed_wip(build_h21_profile(), SEED, wip_target=63)


def test_build_matches_placement_ledger(built):
    """시나리오 실물 = 배치 원장: 본선 있는 블록·작업종류·시작 시각이 원장과 일치."""
    pl = built["vessel_placement"]
    with_vessels = {b for b, s in built["scenarios"].items() if s.vessels}
    assert with_vessels == set(pl)
    for b in with_vessels:
        vs = built["scenarios"][b].vessels
        assert len(vs) == 1
        assert vs[0].work_type.value == pl[b]["work"]
        assert vs[0].plan.planned_start_s == pytest.approx(pl[b]["start_s"], abs=1e-3)


def test_planned_vessel_rate_still_within_anchor(built):
    """슬롯 5%~95% 정정 후에도 계획 작업률이 유도 앵커(145~170 moves/h) 안 — ≈150."""
    planned = 0.0
    for s in built["scenarios"].values():
        for v in s.vessels:
            planned += min(v.plan.total_moves,
                           max(0.0, (OBS.observe_s - v.plan.planned_start_s)
                               / v.plan.sts_move_interval_s))
    rate = planned / (OBS.observe_s / 3600.0)
    assert 145.0 <= rate <= 170.0


def test_flow_fallback_zero_and_ledgered(built):
    """기본 계약(장치율 30%·share 0.6)에서 fallback 0 + 원장 필드 전량 존재."""
    assert built["flow_fallbacks_total"] == 0
    for e in built["pool"]:
        assert e["requested_flow"] == e["flow"] and e["fallback_reason"] is None


def test_flow_fallback_detected_when_inventory_starved():
    """반출 재고를 인위로 말리면 fallback 이 원장에 잡혀야 한다(침묵 전환 금지 검증)."""
    params = TerminalStreamParams(load_4h=63, gate_out_share=1.0, fill_ratio=0.05)
    b = build_fixed_wip(build_h21_profile(), SEED, wip_target=63, params=params)
    assert b["flow_fallbacks_total"] > 0
    flipped = [e for e in b["pool"] if e["fallback_reason"] == "no_free_target"]
    assert flipped and all(e["requested_flow"] == "GATE_OUT"
                           and e["flow"] == "GATE_IN" for e in flipped)


def test_fill_ledger_carries_fallback_fields(built):
    """32차 부채 2: 채움 원장에도 requested/realized/reason 이 남아야 한다."""
    for e in built["fill"]:
        assert "requested_flow" in e and "fallback_reason" in e


def test_background_seed_shared_across_load_cells():
    """32차 부채 3: background_seed 고정 시 부하 셀들이 같은 배경(본선 배치·초기 적재·
    반출대상 순서)을 공유하고, 트럭 pool 만 셀 시드에 따라 달라져야 한다."""
    a = build_fixed_wip(build_h21_profile(), SEED + 50, wip_target=63,
                        background_seed=SEED)
    b = build_fixed_wip(build_h21_profile(), SEED + 75, wip_target=63,
                        background_seed=SEED)
    assert a["vessel_placement"] == b["vessel_placement"]
    for blk in a["scenarios"]:
        sa, sb = a["scenarios"][blk], b["scenarios"][blk]
        assert sa.containers == sb.containers                  # 초기 적재 동일
        assert [v.plan.planned_start_s for v in sa.vessels] == \
               [v.plan.planned_start_s for v in sb.vessels]    # 본선 동일
    assert a["pool"] != b["pool"]                              # 트럭 열은 셀별 상이
